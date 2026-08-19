"""The triton backward — one adjoint for every fused triton schedule.

An adjoint does not have to mirror its forward. All this kernel needs is
``(lambda, eta)`` at the chunk boundaries; how the forward got there — walking
time serially, tiling it, or scanning it — changes nothing here. So
``triton_recurrent``, ``triton_chunk`` and ``triton_pscan`` write the same
``[B, M, NCK, S]`` checkpoints and share this one backward, exactly as the Metal
kernels share ``kla_scan_bwd.metal``.

It is the *exact* adjoint, and cheaper than the composed one the CUDA kernels
carry. Differentiating a trace-normalized prefix product means a 4-component
adjoint through a chain of 4x4 Jacobians, which is ill-conditioned in float32
once the composed matrix degenerates toward rank-1. Differentiating the
*recurrence* instead gives a scalar per-step gain::

    d lambda_t / d lambda_{t-1} = a^2 / den_t^2,   den_t = a^2 + p.lambda_{t-1}

and the whole backward is two reverse affine recurrences carrying one scalar
each.

Two identities keep every quantity local to the replayed chunk, so nothing here
needs a tile shifted by one timestep (triton has no cheap shift along the scan
axis, and both of the obvious workarounds divide by a gain that is small exactly
when the filter is forgetting):

1. ``alpha_t . eta_{t-1} = eta_t - r_t`` — removes eta_{t-1}.
2. ``alpha_{t+1} . nu^eta_{t+1} = nu^eta_t - deta_t`` — removes the shifted
   adjoint from the lambda recurrence's source term.

The multipliers themselves need no shift either: ``den_{t+1} = a^2 + p.lambda_t``
reads lambda at t, which the replay already has.

Reverse scans are a flip, a forward :func:`tl.associative_scan`, and a flip
back. Positions past ``L`` in a partial chunk carry a zero source, and the
incoming carry is zero, so they contribute nothing whatever their multiplier.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_EPS = 1e-12


@triton.jit
def _tn_combine(la, lb, lc, ld, ra, rb, rc, rd):
    """Compose two 2x2 Moebius matrices (R after L), then divide by the trace."""
    a = ra * la + rb * lc
    b = ra * lb + rb * ld
    c = rc * la + rd * lc
    d = rc * lb + rd * ld
    inv = 1.0 / tl.maximum(a + d, _EPS)
    return a * inv, b * inv, c * inv, d * inv


@triton.jit
def _aff_combine(la, lb, ra, rb):
    """Compose two affine maps x -> m.x + b, ``left`` being the earlier one."""
    return ra * la, ra * lb + rb


@triton.jit
def _reverse_affine(mult_next, src, carry, t, BLOCK_L: tl.constexpr):
    """``nu_t = src_t + mult_next_t . nu_{t+1}``, with ``nu_L = carry``.

    ``mult_next_t`` is the multiplier relating ``nu_t`` to ``nu_{t+1}`` — the
    caller supplies it already in that alignment, which is why nothing is
    shifted here. Returns the tile and the value at ``t = 0``, which is the
    carry the *earlier* chunk resumes from.
    """
    mf = tl.flip(mult_next, 0)
    sf = tl.flip(src, 0)
    ga, gb = tl.associative_scan((mf, sf), axis=0, combine_fn=_aff_combine)
    nu = tl.flip(ga * carry[None, :] + gb, 0)
    out_carry = tl.sum(tl.where((t == 0)[:, None], nu, 0.0), axis=0)
    return nu, out_carry


@triton.jit
def _fused_bwd_kernel(
    dy_ptr,  # [B,M,L]
    dyvar_ptr,  # [B,M,L]
    dlam_fin_ptr,  # [B,M,S]
    deta_fin_ptr,  # [B,M,S]
    msi_ptr,  # v.Lambda^v [B,M,L]
    si_ptr,  # Lambda^v   [B,M,L]
    h_ptr,  # k, the key     [B,L,S]
    w_ptr,  # q, the query   [B,L,S]
    a_ptr,  # decay a        [M,S]
    p_ptr,  # process noise p [M,S] (the forward calls this `q_ptr`)
    lam_ck_ptr,  # [B,M,NCK,S]
    eta_ck_ptr,  # [B,M,NCK,S]
    dmsi_ptr,  # out [B,M,L]
    dsi_ptr,  # out [B,M,L]
    dh_ptr,  # out [B,L,S]  (atomic: contracts over m)
    dw_ptr,  # out [B,L,S]  (atomic: contracts over m)
    da_ptr,  # out [M,S]    (atomic: contracts over b and t)
    dp_ptr,  # out [M,S]    (atomic: contracts over b and t)
    dlam0_ptr,  # out [B,M,S]
    deta0_ptr,  # out [B,M,S]
    lam0_ptr,  # [B,M,S]
    eta0_ptr,  # [B,M,S]
    M,
    L,
    S,
    N_CHUNKS,
    PRIOR: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    t = tl.arange(0, BLOCK_L)

    a_st = tl.load(a_ptr + m * S + s, mask=s_mask, other=1.0)
    p_st = tl.load(p_ptr + m * S + s, mask=s_mask, other=0.0)
    a2_st = tl.maximum(a_st * a_st, _EPS)
    # 1/a, floored the same way a^2 is. It only ever multiplies (eta_t - r_t),
    # which is itself proportional to a, so the pair is finite as a -> 0 even
    # though the factor alone is not.
    inv_a_st = a_st / a2_st

    dlam_fin = tl.load(dlam_fin_ptr + (b * M + m) * S + s, mask=s_mask, other=0.0)
    deta_fin = tl.load(deta_fin_ptr + (b * M + m) * S + s, mask=s_mask, other=0.0)

    nu_lam = tl.zeros([BLOCK_S], tl.float32)
    nu_eta = tl.zeros([BLOCK_S], tl.float32)
    da_acc = tl.zeros([BLOCK_S], tl.float32)
    dp_acc = tl.zeros([BLOCK_S], tl.float32)

    base_ml = (b * M + m) * L
    base_hw = b * L * S
    base_ck = (b * M + m) * N_CHUNKS

    for c in range(N_CHUNKS):
        cc = N_CHUNKS - 1 - c  # chunks walk backwards
        tt = cc * BLOCK_L + t
        t_mask = tt < L
        hoff = base_hw + tt[:, None] * S + s[None, :]
        hw_mask = t_mask[:, None] & s_mask[None, :]

        # -- replay the chunk forward from its checkpoint ---------------------
        lam_c = tl.load(lam_ck_ptr + (base_ck + cc) * S + s, mask=s_mask, other=1.0)
        eta_c = tl.load(eta_ck_ptr + (base_ck + cc) * S + s, mask=s_mask, other=0.0)

        msi = tl.load(msi_ptr + base_ml + tt, mask=t_mask, other=0.0)[:, None]
        si = tl.load(si_ptr + base_ml + tt, mask=t_mask, other=0.0)[:, None]
        h = tl.load(h_ptr + hoff, mask=hw_mask, other=0.0)
        wv = tl.load(w_ptr + hoff, mask=hw_mask, other=0.0)

        raw_phi = si * h * h
        phi = tl.maximum(raw_phi, _EPS)
        rr = msi * h

        zero = tl.zeros([BLOCK_L, BLOCK_S], tl.float32)
        a_t = a_st[None, :] + zero
        p_t = p_st[None, :] + zero
        a2_t = a2_st[None, :] + zero

        A = (1.0 + p_t * phi) / a2_t
        C = p_t / a2_t
        D = zero + 1.0

        sA, sB, sC, sD = tl.associative_scan(
            (A, phi, C, D), axis=0, combine_fn=_tn_combine
        )
        lam = (sA * lam_c[None, :] + sB) / tl.maximum(sC * lam_c[None, :] + sD, _EPS)
        # lambda_{t-1} by inverting the leaf; A - C.lambda = 1/(a^2.den) > 0, so
        # this is the stable direction (unlike inverting the affine step).
        lam_prev = (lam - phi) / tl.maximum(A - C * lam, _EPS)
        den = tl.maximum(a2_t + p_t * lam_prev, _EPS)
        alpha = a_t / den

        ga, gb = tl.associative_scan((alpha, rr), axis=0, combine_fn=_aff_combine)
        eta = ga * eta_c[None, :] + gb

        var = 1.0 / tl.maximum(lam, _EPS)
        mean = eta * var

        # -- adjoints of the read-out -----------------------------------------
        dy = tl.load(dy_ptr + base_ml + tt, mask=t_mask, other=0.0)[:, None]
        dyvar = tl.load(dyvar_ptr + base_ml + tt, mask=t_mask, other=0.0)[:, None]

        if PRIOR:
            mean_out = a_t * mean
            var_out = a2_t * var + p_t
        else:
            mean_out = mean
            var_out = var

        d_mean_out = dy
        d_var_out = dyvar
        dw = dy * mean_out + dyvar * 2.0 * wv * var_out
        # y = sum_s q.mean_out and yvar = sum_s q^2.var_out, so the incoming
        # scalars reach this lane weighted by q and q^2.
        d_mean_out = d_mean_out * wv
        d_var_out = d_var_out * wv * wv

        if PRIOR:
            d_mean = d_mean_out * a_t
            d_var = d_var_out * a2_t
            da_acc += tl.sum(
                tl.where(hw_mask, d_mean_out * mean + d_var_out * 2.0 * a_t * var, 0.0),
                axis=0,
            )
            dp_acc += tl.sum(tl.where(hw_mask, d_var_out, 0.0), axis=0)
        else:
            d_mean = d_mean_out
            d_var = d_var_out

        d_eta_dir = d_mean * var
        d_var_tot = d_var + d_mean * eta
        d_lam_dir = -d_var_tot * var * var

        # d(final state) lands on the last timestep, in whichever chunk holds it.
        is_fin = (tt == L - 1)[:, None]
        d_lam_dir += tl.where(is_fin, dlam_fin[None, :], 0.0)
        d_eta_dir += tl.where(is_fin, deta_fin[None, :], 0.0)
        d_lam_dir = tl.where(hw_mask, d_lam_dir, 0.0)
        d_eta_dir = tl.where(hw_mask, d_eta_dir, 0.0)

        # -- the two reverse recurrences --------------------------------------
        # den_{t+1} reads lambda_t, which the replay already has, so both
        # multipliers arrive correctly aligned with no shift.
        den_next = tl.maximum(a2_t + p_t * lam, _EPS)
        alpha_next = a_t / den_next
        g_next = a2_t / (den_next * den_next)

        nu_eta_t, nu_eta = _reverse_affine(alpha_next, d_eta_dir, nu_eta, t, BLOCK_L)

        # lambda_t also moves the gain at t+1, which moves eta_{t+1}. That path
        # carries nu^eta_{t+1}; identity (2) writes it without a shift.
        alpha_nu_next = nu_eta_t - d_eta_dir
        src_lam = d_lam_dir - eta * (p_t / den_next) * alpha_nu_next
        nu_lam_t, nu_lam = _reverse_affine(g_next, src_lam, nu_lam, t, BLOCK_L)

        # -- input gradients ---------------------------------------------------
        # lambda_t = lambda_{t-1}/den_t + phi_t, so d lambda_t / d phi_t = 1.
        dphi = tl.where(raw_phi > _EPS, nu_lam_t, 0.0)
        dr = nu_eta_t

        dsi = tl.sum(tl.where(hw_mask, dphi * h * h, 0.0), axis=1)
        dmsi = tl.sum(tl.where(hw_mask, dr * h, 0.0), axis=1)
        tl.store(dsi_ptr + base_ml + tt, dsi, mask=t_mask)
        tl.store(dmsi_ptr + base_ml + tt, dmsi, mask=t_mask)

        # dk and dq contract over channels, which no single program owns.
        dh = dphi * 2.0 * si * h + dr * msi
        tl.atomic_add(dh_ptr + hoff, dh, mask=hw_mask)
        tl.atomic_add(dw_ptr + hoff, dw, mask=hw_mask)

        # -- static dynamics, contracted over batch and time -------------------
        # alpha_t.eta_{t-1} = eta_t - r_t (identity 1), so d(alpha) rides on
        # d ln(alpha)/d. : 1/a - 2a/den for a, and -lambda_{t-1}/den for p.
        a_eta_prev = eta - rr
        d_lam_da = -lam_prev * 2.0 * a_t / (den * den)
        d_lam_dp = -lam_prev * lam_prev / (den * den)
        da_acc += tl.sum(
            tl.where(
                hw_mask,
                nu_lam_t * d_lam_da
                + nu_eta_t * a_eta_prev * (inv_a_st[None, :] - 2.0 * a_t / den),
                0.0,
            ),
            axis=0,
        )
        dp_acc += tl.sum(
            tl.where(
                hw_mask,
                nu_lam_t * d_lam_dp + nu_eta_t * a_eta_prev * (-lam_prev / den),
                0.0,
            ),
            axis=0,
        )

    tl.atomic_add(da_ptr + m * S + s, da_acc, mask=s_mask)
    tl.atomic_add(dp_ptr + m * S + s, dp_acc, mask=s_mask)

    # -- the boundary state, one step before t = 0 ----------------------------
    lam0 = tl.load(lam0_ptr + (b * M + m) * S + s, mask=s_mask, other=1.0)
    eta0 = tl.load(eta0_ptr + (b * M + m) * S + s, mask=s_mask, other=0.0)
    den0 = tl.maximum(a2_st + p_st * lam0, _EPS)
    alpha0 = a_st / den0
    dlam0 = (a2_st / (den0 * den0)) * nu_lam - eta0 * (p_st / den0) * (alpha0 * nu_eta)
    deta0 = alpha0 * nu_eta
    tl.store(dlam0_ptr + (b * M + m) * S + s, dlam0, mask=s_mask)
    tl.store(deta0_ptr + (b * M + m) * S + s, deta0, mask=s_mask)


def scan_backward(
    dy,
    dyvar,
    dlam_fin,
    deta_fin,
    msi_t,
    si_t,
    h,
    w,
    a,
    p,
    lam0,
    eta0,
    lam_ck,
    eta_ck,
    prior: bool = False,
    block_l: int = 64,
    num_warps: int = 4,
):
    """The shared triton backward.

    ``msi_t``/``si_t``/``dy``/``dyvar`` are ``[B,M,L]`` (the permuted layout the
    fused kernels work in); ``h``/``w`` are ``[B,L,S]``; the checkpoints are
    ``[B,M,NCK,S]`` written at a stride of ``block_l``. Returns gradients in
    the caller's own layout: ``(dmsi, dsi, dh, dw, da, dp, dlam0, deta0)`` with
    ``dmsi``/``dsi`` as ``[B,M,L]``.
    """
    B, M, L = msi_t.shape
    S = h.shape[2]
    n_chunks = lam_ck.shape[2]
    dev = msi_t.device
    f32 = torch.float32

    dmsi = torch.empty(B, M, L, device=dev, dtype=f32)
    dsi = torch.empty(B, M, L, device=dev, dtype=f32)
    # dk/dq contract over channels and da/dp over batch and time; neither axis
    # is owned by a single program, so these four accumulate atomically and must
    # start at zero.
    dh = torch.zeros(B, L, S, device=dev, dtype=f32)
    dw = torch.zeros(B, L, S, device=dev, dtype=f32)
    da = torch.zeros_like(a)
    dp = torch.zeros_like(p)
    dlam0 = torch.empty(B, M, S, device=dev, dtype=f32)
    deta0 = torch.empty(B, M, S, device=dev, dtype=f32)

    _fused_bwd_kernel[(B * M,)](
        dy,
        dyvar,
        dlam_fin,
        deta_fin,
        msi_t,
        si_t,
        h,
        w,
        a,
        p,
        lam_ck,
        eta_ck,
        dmsi,
        dsi,
        dh,
        dw,
        da,
        dp,
        dlam0,
        deta0,
        lam0,
        eta0,
        M,
        L,
        S,
        n_chunks,
        PRIOR=bool(prior),
        BLOCK_L=block_l,
        BLOCK_S=triton.next_power_of_2(S),
        num_warps=num_warps,
    )
    return dmsi, dsi, dh, dw, da, dp, dlam0, deta0
