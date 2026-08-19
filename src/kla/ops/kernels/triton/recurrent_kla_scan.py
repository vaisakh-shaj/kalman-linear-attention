"""``triton_recurrent`` — the whole scan in one kernel, time serial.

One program owns one ``(batch, channel)`` pair and walks the sequence a
timestep at a time, holding all ``BLOCK_S`` states as a vector. The Möbius map
is *applied* to a running λ rather than composed with its neighbours, so there
is no 2×2 matrix, no trace normalization and no overflow path — about a quarter
of ``triton_chunk``'s arithmetic, traded against the instruction-level
parallelism a ``[BLOCK_L, S]`` tile gives it.

The lane count is the same either way (``B*M*S``); what differs is how many
timesteps are in flight. So this is the one to reach for when there are already
sequences and channels to spend — decode, and training at any real batch size —
and ``triton_chunk`` when there are not.

The backward is :mod:`kla.ops.kernels.triton.kla_scan_bwd`, shared with
``triton_chunk``. This forward writes the same ``[B, M, NCK, S]`` checkpoints at
the same stride, which is all that backward needs; it is chunk-shaped itself,
and does not care that the forward was not.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_EPS = 1e-12


@triton.jit
def _recurrent_fwd_kernel(
    msi_ptr,  # v·Λ^v [B,M,L]
    si_ptr,  # Λ^v [B,M,L]
    h_ptr,  # k, the key [B,L,S]
    w_ptr,  # q, the query [B,L,S]
    a_ptr,  # decay a [M,S]
    p_ptr,  # process noise p [M,S]
    lam0_ptr,  # [B,M,S]
    eta0_ptr,  # [B,M,S]
    y_ptr,  # out [B,M,L]
    yvar_ptr,  # out [B,M,L]
    lam_fin_ptr,  # out [B,M,S]
    eta_fin_ptr,  # out [B,M,S]
    lam_ck_ptr,  # out [B,M,NCK,S] (see STORE_CK)
    eta_ck_ptr,  # out [B,M,NCK,S]
    M,
    L,
    S,
    N_CHUNKS,
    STORE_CK: tl.constexpr,
    PRIOR: tl.constexpr,
    CK_STRIDE: tl.constexpr,  # must match the backward's BLOCK_L
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S

    lam = tl.load(lam0_ptr + (b * M + m) * S + s, mask=s_mask, other=1.0)
    eta = tl.load(eta0_ptr + (b * M + m) * S + s, mask=s_mask, other=0.0)

    # Static dynamics are loop-invariant, so hoist them out of the time loop.
    a_st = tl.load(a_ptr + m * S + s, mask=s_mask, other=1.0)
    p_st = tl.load(p_ptr + m * S + s, mask=s_mask, other=0.0)
    a2 = tl.maximum(a_st * a_st, _EPS)

    base_ml = (b * M + m) * L
    base_ck = (b * M + m) * N_CHUNKS

    for t in range(L):
        if STORE_CK:
            # The state *entering* step t is what the backward resumes from, so
            # the store precedes the update. The stride is the backward's chunk,
            # not anything this kernel has of its own.
            is_ck = (t % CK_STRIDE) == 0
            ck = base_ck + t // CK_STRIDE
            tl.store(lam_ck_ptr + ck * S + s, lam, mask=s_mask & is_ck)
            tl.store(eta_ck_ptr + ck * S + s, eta, mask=s_mask & is_ck)

        msi_t = tl.load(msi_ptr + base_ml + t)
        si_t = tl.load(si_ptr + base_ml + t)
        hoff = (b * L + t) * S + s
        h = tl.load(h_ptr + hoff, mask=s_mask, other=0.0)
        wv = tl.load(w_ptr + hoff, mask=s_mask, other=0.0)

        phi = tl.maximum(si_t * h * h, _EPS)
        den = tl.maximum(a2 + p_st * lam, _EPS)
        alpha = a_st / den
        lam = lam / den + phi
        eta = alpha * eta + msi_t * h

        var = 1.0 / tl.maximum(lam, _EPS)
        mean = eta * var
        if PRIOR:  # decode_from_prior: read out one predict step ahead
            mean = a_st * mean
            var = a2 * var + p_st
        y = tl.sum(tl.where(s_mask, mean * wv, 0.0), axis=0)
        yvar = tl.sum(tl.where(s_mask, var * wv * wv, 0.0), axis=0)
        tl.store(y_ptr + base_ml + t, y)
        tl.store(yvar_ptr + base_ml + t, yvar)

    tl.store(lam_fin_ptr + (b * M + m) * S + s, lam, mask=s_mask)
    tl.store(eta_fin_ptr + (b * M + m) * S + s, eta, mask=s_mask)


def recurrent_forward(
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
    """Recurrent forward. Same returns as the chunk one, including the permuted
    ``[B,M,L]`` inputs and the ``[B,M,NCK,S]`` checkpoints the backward wants."""
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
    n_ck = n_chunks if checkpoints else 1
    ck_shape = (B, M, n_ck, S) if checkpoints else (1,)
    lam_ck = torch.empty(*ck_shape, device=msi.device, dtype=torch.float32)
    eta_ck = torch.empty(*ck_shape, device=msi.device, dtype=torch.float32)

    _recurrent_fwd_kernel[(B * M,)](
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
        CK_STRIDE=block_l,
        BLOCK_S=triton.next_power_of_2(S),
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


class _RecurrentKLAScan(torch.autograd.Function):
    """Serial-time triton forward, with the shared triton backward behind it."""

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
        ) = recurrent_forward(
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


def recurrent_kla_scan(msi, si, h, w, a, q, lam0, eta0, prior=False, block_l: int = 64):
    """Differentiable recurrent KLA scan → ``(y, y_var, lam_fin, eta_fin)``."""
    return _RecurrentKLAScan.apply(msi, si, h, w, a, q, lam0, eta0, prior, block_l)
