"""``triton_fused_recurrent``, the whole scan in one kernel, time serial.

One program owns one ``(batch, channel)`` pair and walks the sequence a
timestep at a time, holding all ``BLOCK_S`` states as a vector. The Möbius map
is *applied* to a running λ rather than composed with its neighbours, so there
is no 2×2 matrix, no trace normalization and no overflow path, about a quarter
of ``triton_fused_chunk``'s arithmetic, traded against the instruction-level
parallelism a ``[BLOCK_L, S]`` tile gives it.

The lane count is the same either way (``B*M*S``); what differs is how many
timesteps are in flight. So this is the one to reach for when there are already
sequences and channels to spend, decode, and training at any real batch size,
and ``triton_fused_chunk`` when there are not.

The backward is :mod:`kla.ops.kernels.triton.kla_scan_bwd`, shared with
``triton_fused_chunk``. This forward writes the same ``[B, M, NCK, S]`` checkpoints at
the same stride, which is all that backward needs; it is chunk-shaped itself,
and does not care that the forward was not.
"""

from __future__ import annotations

import triton
import triton.language as tl

from kla.ops.kernels.triton._host import Cell, make_scan
from kla.ops.kernels.triton._tuning import RECURRENT_BLOCK_L

_EPS = tl.constexpr(1e-12)


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


CELL = Cell(
    kernel=_recurrent_fwd_kernel,
    default_block_l=RECURRENT_BLOCK_L,
    tile_key="CK_STRIDE",
    warp_rows=1,
)

recurrent_kla_scan = make_scan(
    CELL,
    "recurrent_kla_scan",
    """Differentiable recurrent KLA scan → ``(y, y_var, lam_fin, eta_fin)``.""",
)
