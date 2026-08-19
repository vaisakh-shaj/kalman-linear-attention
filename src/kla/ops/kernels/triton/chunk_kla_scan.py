"""``triton_chunk`` — the whole scan in one kernel, time tiled.

One triton program owns one ``(batch, channel)`` pair and streams the sequence
in ``BLOCK_L`` chunks, doing *everything* in registers with no intermediate
``[B,L,M,S]`` tensor in HBM (the per-(b,l,m,s) torch "glue" of the unfused cells
is what left them ~6-11x the CUDA kernel):

    load k,q,v·Λ^v,Λ^v  →  φ,r  →  leaves  →  trace-norm Möbius scan → λ
    →  λ_{t-1} by inverting the leaf  →  α  →  affine scan → η
    →  readout  y=Σ_s q·η/λ,  yvar=Σ_s q²/λ

Kernel-internal names differ from the paper's notation; the mapping is
``msi``→v·Λ^v, ``si``→Λ^v, ``h``→k (key), ``w``→q (query), and the kernel's
``q``/``q_ptr`` argument is the *process noise* p — not the query.

Key trick: λ_{t-1} = (D·λ_t − B)/(A − C·λ_t) recovers the previous precision
locally (``A − C·λ = 1/(a²·den) > 0`` so it is stable, and it returns λ0 exactly
at t=0), so the α-gain needs no cross-chunk λ shift.

``a``/``p`` are static ``[M, S]`` (the paper's time-invariant dynamics), loaded
once per program outside the sequence loop.

The backward is :mod:`kla.ops.kernels.triton.kla_scan_bwd`, shared with the
other two fused cells: this forward writes the same ``[B, M, NCK, S]``
checkpoints at the same stride, which is all that backward needs.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_EPS = 1e-12


@triton.jit
def _tn_combine(la, lb, lc, ld, ra, rb, rc, rd):
    a = ra * la + rb * lc
    b = ra * lb + rb * ld
    c = rc * la + rd * lc
    d = rc * lb + rd * ld
    inv = 1.0 / tl.maximum(a + d, 1e-12)
    return a * inv, b * inv, c * inv, d * inv


@triton.jit
def _aff_combine(la, lb, ra, rb):
    return ra * la, ra * lb + rb


@triton.jit
def _fused_fwd_kernel(
    msi_ptr,  # v·Λ^v [B,M,L]
    si_ptr,  # Λ^v [B,M,L]
    h_ptr,  # k, the key [B,L,S]
    w_ptr,  # q, the query [B,L,S]
    a_ptr,  # decay a [M,S]
    q_ptr,  # process noise p [M,S]
    lam0_ptr,  # λ boundary [B,M,S]
    eta0_ptr,  # η boundary [B,M,S]
    y_ptr,  # out: y [B,M,L]
    yvar_ptr,  # out: y_var [B,M,L]
    lam_fin_ptr,  # out: final λ [B,M,S]
    eta_fin_ptr,  # out: final η [B,M,S]
    lam_ck_ptr,  # out: λ entering each chunk [B,M,NCK,S] (see STORE_CK)
    eta_ck_ptr,  # out: η entering each chunk [B,M,NCK,S]
    M,
    L,
    S,
    N_CHUNKS,
    STORE_CK: tl.constexpr,
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

    lam0 = tl.load(lam0_ptr + (b * M + m) * S + s, mask=s_mask, other=1.0)  # [S]
    c_eta = tl.load(
        eta0_ptr + (b * M + m) * S + s, mask=s_mask, other=0.0
    )  # η boundary

    # Static dynamics are loop-invariant, so hoist them out of the chunk loop.
    a_st = tl.load(a_ptr + m * S + s, mask=s_mask, other=1.0)
    q_st = tl.load(q_ptr + m * S + s, mask=s_mask, other=0.0)
    a2_st = a_st * a_st

    cA = tl.zeros([BLOCK_S], tl.float32) + 1.0
    cB = tl.zeros([BLOCK_S], tl.float32)
    cC = tl.zeros([BLOCK_S], tl.float32)
    cD = tl.zeros([BLOCK_S], tl.float32) + 1.0

    base_ml = (b * M + m) * L  # μσ⁻¹[b,m,:], output[b,m,:]
    base_hw = b * L * S  # h[b,:,:], w[b,:,:]
    base_ck = (b * M + m) * N_CHUNKS

    for c in range(N_CHUNKS):
        tt = c * BLOCK_L + t
        t_mask = tt < L

        # The state *entering* this chunk is what kla_scan_bwd resumes from, so
        # the store precedes the scan. It is free: the carry already exists.
        if STORE_CK:
            lam_in = (cA * lam0 + cB) / tl.maximum(cC * lam0 + cD, 1e-12)
            tl.store(lam_ck_ptr + (base_ck + c) * S + s, lam_in, mask=s_mask)
            tl.store(eta_ck_ptr + (base_ck + c) * S + s, c_eta, mask=s_mask)
        hoff = base_hw + tt[:, None] * S + s[None, :]
        hw_mask = t_mask[:, None] & s_mask[None, :]

        msi = tl.load(msi_ptr + base_ml + tt, mask=t_mask, other=0.0)[:, None]
        si = tl.load(si_ptr + base_ml + tt, mask=t_mask, other=0.0)[:, None]
        h = tl.load(h_ptr + hoff, mask=hw_mask, other=0.0)
        wv = tl.load(w_ptr + hoff, mask=hw_mask, other=0.0)

        phi = tl.maximum(si * h * h, 1e-12)
        rr = msi * h

        # Broadcast a/q to the full [BLOCK_L, S] tile (static a/q load as [1, S]).
        zero = tl.zeros([BLOCK_L, BLOCK_S], tl.float32)
        a_t = a_st[None, :] + zero
        q_t = q_st[None, :] + zero
        a2_t = a2_st[None, :] + zero

        A = (1.0 + q_t * phi) / a2_t
        C = q_t / a2_t
        D = zero + 1.0

        sA, sB, sC, sD = tl.associative_scan(
            (A, phi, C, D), axis=0, combine_fn=_tn_combine
        )
        fA = sA * cA[None, :] + sB * cC[None, :]
        fB = sA * cB[None, :] + sB * cD[None, :]
        fC = sC * cA[None, :] + sD * cC[None, :]
        fD = sC * cB[None, :] + sD * cD[None, :]
        inv = 1.0 / tl.maximum(fA + fD, 1e-12)
        fA, fB, fC, fD = fA * inv, fB * inv, fC * inv, fD * inv

        lam = (fA * lam0[None, :] + fB) / tl.maximum(fC * lam0[None, :] + fD, 1e-12)
        # λ_{t-1} by inverting the leaf (A − C·λ = 1/(a²·den) > 0).
        lam_prev = (lam - phi) / tl.maximum(A - C * lam, 1e-12)
        alpha = a_t / tl.maximum(a2_t + q_t * lam_prev, 1e-12)

        ga, gb = tl.associative_scan((alpha, rr), axis=0, combine_fn=_aff_combine)
        eta = ga * c_eta[None, :] + gb

        var = 1.0 / tl.maximum(lam, 1e-12)
        mean = eta * var
        if PRIOR:
            # decode_from_prior: read out one predict step ahead.
            mean = a_t * mean
            var = a2_t * var + q_t
        y = tl.sum(tl.where(hw_mask, mean * wv, 0.0), axis=1)
        yvar = tl.sum(tl.where(hw_mask, var * wv * wv, 0.0), axis=1)
        tl.store(y_ptr + base_ml + tt, y, mask=t_mask)
        tl.store(yvar_ptr + base_ml + tt, yvar, mask=t_mask)

        last = tl.minimum(BLOCK_L, L - c * BLOCK_L) - 1
        sel = (t == last)[:, None]
        cA = tl.sum(tl.where(sel, fA, 0.0), axis=0)
        cB = tl.sum(tl.where(sel, fB, 0.0), axis=0)
        cC = tl.sum(tl.where(sel, fC, 0.0), axis=0)
        cD = tl.sum(tl.where(sel, fD, 0.0), axis=0)
        c_eta = tl.sum(tl.where(sel, eta, 0.0), axis=0)

    # Final filter state: λ_{L-1} from the accumulated Möbius matrix, η_{L-1}.
    lam_fin = (cA * lam0 + cB) / tl.maximum(cC * lam0 + cD, 1e-12)
    tl.store(lam_fin_ptr + (b * M + m) * S + s, lam_fin, mask=s_mask)
    tl.store(eta_fin_ptr + (b * M + m) * S + s, c_eta, mask=s_mask)


def chunk_forward(
    msi,
    si,
    h,
    w,
    a,
    q,
    lam0,
    eta0,
    checkpoints: bool = False,
    prior: bool = False,
    block_l: int = 64,
    num_warps: int = 4,
):
    """Fused forward. msi/si [B,L,M], h/w [B,L,S], lam0/eta0 [B,M,S].

    ``a``/``q`` (decay and process noise) are static ``[M, S]``. Returns
    ``(y, yvar)`` ``[B,L,M]``, the final ``(lam, eta)``, the permuted
    ``[B,M,L]`` inputs the backward wants, and the ``[B,M,NCK,S]`` checkpoints.
    With ``checkpoints=False`` the last two are one-element placeholders — the
    kernel takes the flag and never writes them.
    """
    B, L, M = msi.shape
    S = h.shape[2]

    msi_t = msi.permute(0, 2, 1).contiguous()  # [B,M,L]
    si_t = si.permute(0, 2, 1).contiguous()
    h_c = h.contiguous()
    w_c = w.contiguous()
    a_c = a.contiguous()
    q_c = q.contiguous()
    lam0_c = lam0.contiguous()
    eta0_c = eta0.contiguous()

    y = torch.empty(B, M, L, device=msi.device, dtype=torch.float32)
    yvar = torch.empty(B, M, L, device=msi.device, dtype=torch.float32)
    lam_fin = torch.empty(B, M, S, device=msi.device, dtype=torch.float32)
    eta_fin = torch.empty(B, M, S, device=msi.device, dtype=torch.float32)

    n_chunks = triton.cdiv(L, block_l)
    block_s = triton.next_power_of_2(S)
    n_ck = n_chunks if checkpoints else 1
    ck_shape = (B, M, n_ck, S) if checkpoints else (1,)
    lam_ck = torch.empty(*ck_shape, device=msi.device, dtype=torch.float32)
    eta_ck = torch.empty(*ck_shape, device=msi.device, dtype=torch.float32)
    _fused_fwd_kernel[(B * M,)](
        msi_t,
        si_t,
        h_c,
        w_c,
        a_c,
        q_c,
        lam0_c,
        eta0_c,
        y,
        yvar,
        lam_fin,
        eta_fin,
        lam_ck,
        eta_ck,
        M,
        L,
        S,
        n_chunks,
        STORE_CK=bool(checkpoints),
        PRIOR=bool(prior),
        BLOCK_L=block_l,
        BLOCK_S=block_s,
        num_warps=num_warps,
    )
    return (
        y.permute(0, 2, 1).contiguous(),
        yvar.permute(0, 2, 1).contiguous(),
        lam_fin,
        eta_fin,
        msi_t,
        si_t,
        h_c,
        w_c,
        lam0_c,
        eta0_c,
        lam_ck,
        eta_ck,
    )


class _FusedKLAScan(torch.autograd.Function):
    """The fused triton forward, with the shared triton backward behind it."""

    @staticmethod
    def forward(ctx, msi, si, h, w, a, q, lam0, eta0, prior, block_l):
        needs_grad = any(ctx.needs_input_grad)
        (
            y,
            yvar,
            lam_fin,
            eta_fin,
            msi_t,
            si_t,
            h_c,
            w_c,
            lam0_c,
            eta0_c,
            lam_ck,
            eta_ck,
        ) = chunk_forward(
            msi,
            si,
            h,
            w,
            a,
            q,
            lam0,
            eta0,
            checkpoints=needs_grad,
            prior=prior,
            block_l=block_l,
        )
        ctx.prior = prior
        ctx.block_l = block_l
        ctx.save_for_backward(
            msi_t, si_t, h_c, w_c, a, q, lam0_c, eta0_c, lam_ck, eta_ck
        )
        return y, yvar, lam_fin, eta_fin

    @staticmethod
    def backward(ctx, dy, dyvar, dlam_fin, deta_fin):
        from kla.ops.kernels.triton.kla_scan_bwd import scan_backward

        msi_t, si_t, h_c, w_c, a, q, lam0_c, eta0_c, lam_ck, eta_ck = ctx.saved_tensors
        dmsi, dsi, dh, dw, da, dp, dlam0, deta0 = scan_backward(
            dy.permute(0, 2, 1).contiguous(),
            dyvar.permute(0, 2, 1).contiguous(),
            dlam_fin.contiguous(),
            deta_fin.contiguous(),
            msi_t,
            si_t,
            h_c,
            w_c,
            a,
            q,
            lam0_c,
            eta0_c,
            lam_ck,
            eta_ck,
            prior=ctx.prior,
            block_l=ctx.block_l,
        )
        return (
            dmsi.permute(0, 2, 1),
            dsi.permute(0, 2, 1),
            dh,
            dw,
            da,
            dp,
            dlam0,
            deta0,
            None,
            None,
        )


def chunk_kla_scan(msi, si, h, w, a, q, lam0, eta0, prior=False, block_l: int = 64):
    """Differentiable fused KLA scan → ``(y, y_var, lam_fin, eta_fin)``."""
    return _FusedKLAScan.apply(msi, si, h, w, a, q, lam0, eta0, prior, block_l)
