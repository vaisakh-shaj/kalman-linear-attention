"""``triton_pscan`` — the scan as a reduce-then-scan, with no serial carry.

``triton_recurrent`` walks time serially inside one program. ``triton_chunk``
tiles time and walks the *tiles* serially, one program carrying the state from
each to the next. This drops that last serial axis: the sequence is cut into
``NCK`` chunks of ``BLOCK_L`` steps, every chunk is reduced by its own program,
and the cross-chunk dependency is settled by a parallel scan over the ``NCK``
aggregates rather than by anyone waiting for their neighbour.

The grid is ``B*M*NCK`` rather than ``B*M``, which is the point: at batch-1
prefill the other two implementations leave the GPU with ``B*M`` programs and nothing
to overlap. The trade is ``[B, M, NCK, S]`` of aggregates in HBM and roughly
three touches of each element instead of one.

It stays fused in the sense this repo uses the word — the intermediates are per
*chunk*, never the per-timestep ``[B, L, M, S]`` the unfused path builds — and
two of those aggregate arrays are the backward's checkpoints, which this
implementation computes rather than stores: "the state entering chunk c" is exactly
what the scan over aggregates produces. So
:mod:`kla.ops.kernels.triton.kla_scan_bwd` runs behind it unchanged, at the
same stride, for the same exact gradients.

Five kernels, two reduce-scan-apply rounds. Round two cannot be folded into
round one: the affine leaf ``α_t`` reads ``λ_{t-1}``, so those leaves do not
exist until the Möbius scan has produced λ. The apply kernel recomputes λ rather
than reading it back, which is what keeps the intermediates per-chunk.

Transcribed from ``kernels/mps/pscan_kla_scan.metal``, which runs and is checked
against the torch reference — forward and gradients, over chunk counts of 1, 2,
3, 5 and 17.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

_EPS = 1e-12


@triton.jit
def _mob_compose(a1, b1, c1, d1, a2, b2, c2, d2):
    """The map "2 after 1": the matrix product, divided by its own trace.

    λ is invariant under rescaling the four entries, so the normalization only
    buys exponent range — the same scheme the torch reference
    (``_mobius_combine_tracenorm``) and both other backends use.
    """
    a = a2 * a1 + b2 * c1
    b = a2 * b1 + b2 * d1
    c = c2 * a1 + d2 * c1
    d = c2 * b1 + d2 * d1
    inv = 1.0 / tl.maximum(a + d, _EPS)
    return a * inv, b * inv, c * inv, d * inv


@triton.jit
def _pscan_decode(pid, M, N_CHUNKS):
    """``pid`` → ``(b, m, c)``, the flat index over ``[B, M, NCK]``."""
    c = pid % N_CHUNKS
    bm = pid // N_CHUNKS
    return bm // M, bm % M, c


# --------------------------------------------------------- 1: reduce the chunks


@triton.jit
def _mob_reduce_kernel(
    si_ptr,  # Λ^v [B,M,L]
    h_ptr,  # k, the key [B,L,S]
    a_ptr,  # decay a [M,S]
    p_ptr,  # process noise p [M,S]
    mob_a_ptr,  # out [B,M,NCK,S], the four entries of the chunk's map
    mob_b_ptr,
    mob_c_ptr,
    mob_d_ptr,
    M,
    L,
    S,
    N_CHUNKS,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b, m, c = _pscan_decode(pid, M, N_CHUNKS)

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    a_st = tl.load(a_ptr + m * S + s, mask=s_mask, other=1.0)
    p_st = tl.load(p_ptr + m * S + s, mask=s_mask, other=0.0)
    inv_a2 = 1.0 / tl.maximum(a_st * a_st, _EPS)

    base_ml = (b * M + m) * L
    # Steps past the end of the sequence contribute the identity, so a partial
    # final chunk composes to exactly what its live prefix does.
    acc_a = tl.full((BLOCK_S,), 1.0, tl.float32)
    acc_b = tl.zeros((BLOCK_S,), tl.float32)
    acc_c = tl.zeros((BLOCK_S,), tl.float32)
    acc_d = tl.full((BLOCK_S,), 1.0, tl.float32)

    for i in range(BLOCK_L):
        t = c * BLOCK_L + i
        live = t < L
        # Clamped rather than masked: the loads stay unconditional and the dead
        # lanes are folded away by the `where` below.
        tc = tl.minimum(t, L - 1)
        si_t = tl.load(si_ptr + base_ml + tc)
        h = tl.load(h_ptr + (b * L + tc) * S + s, mask=s_mask, other=0.0)

        phi = tl.maximum(si_t * h * h, _EPS)
        leaf_a = tl.where(live, (1.0 + p_st * phi) * inv_a2, 1.0)
        leaf_b = tl.where(live, phi, 0.0)
        leaf_c = tl.where(live, p_st * inv_a2, 0.0)
        acc_a, acc_b, acc_c, acc_d = _mob_compose(
            acc_a, acc_b, acc_c, acc_d, leaf_a, leaf_b, leaf_c, 1.0
        )

    off = pid * S + s
    tl.store(mob_a_ptr + off, acc_a, mask=s_mask)
    tl.store(mob_b_ptr + off, acc_b, mask=s_mask)
    tl.store(mob_c_ptr + off, acc_c, mask=s_mask)
    tl.store(mob_d_ptr + off, acc_d, mask=s_mask)


# ----------------------------------------------- 2/4: one doubling round each
#
# dst[c] = compose(src[c-off], src[c]) for c >= off, src[c] otherwise. Run with
# off = 1, 2, 4, ... < NCK and the buffers swapped, this leaves the inclusive
# prefix in the last destination. Doing it across launches rather than inside
# one program means nothing bounds NCK.


@triton.jit
def _mob_step_kernel(
    src_a_ptr,
    src_b_ptr,
    src_c_ptr,
    src_d_ptr,
    dst_a_ptr,
    dst_b_ptr,
    dst_c_ptr,
    dst_d_ptr,
    M,
    S,
    N_CHUNKS,
    OFF,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    _, _, c = _pscan_decode(pid, M, N_CHUNKS)

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    here = pid * S + s
    mine_a = tl.load(src_a_ptr + here, mask=s_mask, other=1.0)
    mine_b = tl.load(src_b_ptr + here, mask=s_mask, other=0.0)
    mine_c = tl.load(src_c_ptr + here, mask=s_mask, other=0.0)
    mine_d = tl.load(src_d_ptr + here, mask=s_mask, other=1.0)

    # Out of range reads the identity, which composes to a pass-through — so the
    # `c < OFF` case needs no branch of its own.
    prev = tl.maximum(pid - OFF, 0) * S + s
    take = s_mask & (c >= OFF)
    prev_a = tl.load(src_a_ptr + prev, mask=take, other=1.0)
    prev_b = tl.load(src_b_ptr + prev, mask=take, other=0.0)
    prev_c = tl.load(src_c_ptr + prev, mask=take, other=0.0)
    prev_d = tl.load(src_d_ptr + prev, mask=take, other=1.0)

    out_a, out_b, out_c, out_d = _mob_compose(
        prev_a, prev_b, prev_c, prev_d, mine_a, mine_b, mine_c, mine_d
    )
    tl.store(dst_a_ptr + here, out_a, mask=s_mask)
    tl.store(dst_b_ptr + here, out_b, mask=s_mask)
    tl.store(dst_c_ptr + here, out_c, mask=s_mask)
    tl.store(dst_d_ptr + here, out_d, mask=s_mask)


@triton.jit
def _aff_step_kernel(
    src_al_ptr,
    src_r_ptr,
    dst_al_ptr,
    dst_r_ptr,
    M,
    S,
    N_CHUNKS,
    OFF,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    _, _, c = _pscan_decode(pid, M, N_CHUNKS)

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    here = pid * S + s
    mine_al = tl.load(src_al_ptr + here, mask=s_mask, other=1.0)
    mine_r = tl.load(src_r_ptr + here, mask=s_mask, other=0.0)

    prev = tl.maximum(pid - OFF, 0) * S + s
    take = s_mask & (c >= OFF)
    prev_al = tl.load(src_al_ptr + prev, mask=take, other=1.0)
    prev_r = tl.load(src_r_ptr + prev, mask=take, other=0.0)

    tl.store(dst_al_ptr + here, mine_al * prev_al, mask=s_mask)
    tl.store(dst_r_ptr + here, mine_al * prev_r + mine_r, mask=s_mask)


# ------------------------------------ 3: seed lambda, then reduce the affine


@triton.jit
def _aff_reduce_kernel(
    msi_ptr,  # v·Λ^v [B,M,L]
    si_ptr,  # Λ^v [B,M,L]
    h_ptr,  # k [B,L,S]
    a_ptr,
    p_ptr,
    lam0_ptr,  # [B,M,S]
    mob_a_ptr,  # the inclusive prefix from round one
    mob_b_ptr,
    mob_c_ptr,
    mob_d_ptr,
    lam_ck_ptr,  # out [B,M,NCK,S] — a checkpoint and this round's seed
    aff_al_ptr,  # out [B,M,NCK,S]
    aff_r_ptr,
    M,
    L,
    S,
    N_CHUNKS,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b, m, c = _pscan_decode(pid, M, N_CHUNKS)

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    a_st = tl.load(a_ptr + m * S + s, mask=s_mask, other=1.0)
    p_st = tl.load(p_ptr + m * S + s, mask=s_mask, other=0.0)
    a2 = tl.maximum(a_st * a_st, _EPS)

    # The exclusive prefix maps the initial state to the state entering this
    # chunk. Chunk 0 reads the identity out of range, so it needs no branch.
    prev = tl.maximum(pid - 1, 0) * S + s
    take = s_mask & (c > 0)
    pa = tl.load(mob_a_ptr + prev, mask=take, other=1.0)
    pb = tl.load(mob_b_ptr + prev, mask=take, other=0.0)
    pc = tl.load(mob_c_ptr + prev, mask=take, other=0.0)
    pd = tl.load(mob_d_ptr + prev, mask=take, other=1.0)

    lam_start = tl.load(lam0_ptr + (b * M + m) * S + s, mask=s_mask, other=1.0)
    lam = (pa * lam_start + pb) / tl.maximum(pc * lam_start + pd, _EPS)
    tl.store(lam_ck_ptr + pid * S + s, lam, mask=s_mask)

    acc_al = tl.full((BLOCK_S,), 1.0, tl.float32)
    acc_r = tl.zeros((BLOCK_S,), tl.float32)

    base_ml = (b * M + m) * L
    for i in range(BLOCK_L):
        t = c * BLOCK_L + i
        live = t < L
        tc = tl.minimum(t, L - 1)
        si_t = tl.load(si_ptr + base_ml + tc)
        msi_t = tl.load(msi_ptr + base_ml + tc)
        h = tl.load(h_ptr + (b * L + tc) * S + s, mask=s_mask, other=0.0)

        phi = tl.maximum(si_t * h * h, _EPS)
        den = tl.maximum(a2 + p_st * lam, _EPS)
        # The gain reads λ_{t-1}, so it is formed before the update.
        leaf_al = tl.where(live, a_st / den, 1.0)
        leaf_r = tl.where(live, msi_t * h, 0.0)
        acc_al, acc_r = leaf_al * acc_al, leaf_al * acc_r + leaf_r
        lam = tl.where(live, lam / den + phi, lam)

    tl.store(aff_al_ptr + pid * S + s, acc_al, mask=s_mask)
    tl.store(aff_r_ptr + pid * S + s, acc_r, mask=s_mask)


# ------------------------- 5: seed eta, replay the chunk, and read out


@triton.jit
def _apply_kernel(
    msi_ptr,
    si_ptr,
    h_ptr,  # k [B,L,S]
    w_ptr,  # q, the query [B,L,S]
    a_ptr,
    p_ptr,
    eta0_ptr,  # [B,M,S]
    lam_ck_ptr,  # [B,M,NCK,S], from round one
    aff_al_ptr,  # the inclusive prefix from round two
    aff_r_ptr,
    y_ptr,  # out [B,M,L]
    yvar_ptr,  # out [B,M,L]
    lam_fin_ptr,  # out [B,M,S]
    eta_fin_ptr,  # out [B,M,S]
    eta_ck_ptr,  # out [B,M,NCK,S]
    M,
    L,
    S,
    N_CHUNKS,
    PRIOR: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b, m, c = _pscan_decode(pid, M, N_CHUNKS)

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    a_st = tl.load(a_ptr + m * S + s, mask=s_mask, other=1.0)
    p_st = tl.load(p_ptr + m * S + s, mask=s_mask, other=0.0)
    a2 = tl.maximum(a_st * a_st, _EPS)

    lam = tl.load(lam_ck_ptr + pid * S + s, mask=s_mask, other=1.0)

    prev = tl.maximum(pid - 1, 0) * S + s
    take = s_mask & (c > 0)
    p_al = tl.load(aff_al_ptr + prev, mask=take, other=1.0)
    p_r = tl.load(aff_r_ptr + prev, mask=take, other=0.0)
    eta_start = tl.load(eta0_ptr + (b * M + m) * S + s, mask=s_mask, other=0.0)
    eta = p_al * eta_start + p_r
    tl.store(eta_ck_ptr + pid * S + s, eta, mask=s_mask)

    base_ml = (b * M + m) * L
    for i in range(BLOCK_L):
        t = c * BLOCK_L + i
        live = t < L
        tc = tl.minimum(t, L - 1)
        si_t = tl.load(si_ptr + base_ml + tc)
        msi_t = tl.load(msi_ptr + base_ml + tc)
        hoff = (b * L + tc) * S + s
        h = tl.load(h_ptr + hoff, mask=s_mask, other=0.0)
        wv = tl.load(w_ptr + hoff, mask=s_mask, other=0.0)

        phi = tl.maximum(si_t * h * h, _EPS)
        den = tl.maximum(a2 + p_st * lam, _EPS)
        alpha = a_st / den
        lam = tl.where(live, lam / den + phi, lam)
        eta = tl.where(live, alpha * eta + msi_t * h, eta)

        var = 1.0 / tl.maximum(lam, _EPS)
        mean = eta * var
        if PRIOR:  # decode_from_prior: read out one predict step ahead
            mean = a_st * mean
            var = a2 * var + p_st
        y = tl.sum(tl.where(s_mask, mean * wv, 0.0), axis=0)
        yvar = tl.sum(tl.where(s_mask, var * wv * wv, 0.0), axis=0)
        # Clamped address plus the mask: a dead lane neither writes nor points
        # past the end of the row.
        tl.store(y_ptr + base_ml + tc, y, mask=live)
        tl.store(yvar_ptr + base_ml + tc, yvar, mask=live)

    # The last chunk holds the state at L, whatever the tail alignment.
    last = s_mask & (c == N_CHUNKS - 1)
    tl.store(lam_fin_ptr + (b * M + m) * S + s, lam, mask=last)
    tl.store(eta_fin_ptr + (b * M + m) * S + s, eta, mask=last)


def pscan_forward(
    msi,
    si,
    h,
    w,
    a,
    q,
    lam0,
    eta0,
    prior: bool = False,
    block_l: int = 64,
    num_warps: int = 4,
):
    """Parallel-scan forward. Same returns as the other two triton forwards.

    It takes no ``checkpoints`` flag: the checkpoints are the scan's own
    intermediates, so there is nothing to skip.
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

    dev, f32 = msi.device, torch.float32
    n_ck = triton.cdiv(L, block_l)
    block_s = triton.next_power_of_2(S)
    grid = (B * M * n_ck,)

    def agg():
        return torch.empty(B, M, n_ck, S, device=dev, dtype=f32)

    mob = [agg() for _ in range(4)]
    mob_alt = [agg() for _ in range(4)]
    aff = [agg() for _ in range(2)]
    aff_alt = [agg() for _ in range(2)]
    lam_ck, eta_ck = agg(), agg()

    y = torch.empty(B, M, L, device=dev, dtype=f32)
    yvar = torch.empty(B, M, L, device=dev, dtype=f32)
    lam_fin = torch.empty(B, M, S, device=dev, dtype=f32)
    eta_fin = torch.empty(B, M, S, device=dev, dtype=f32)

    _mob_reduce_kernel[grid](
        si_t,
        h_c,
        a_c,
        q_c,
        *mob,
        M,
        L,
        S,
        n_ck,
        BLOCK_L=block_l,
        BLOCK_S=block_s,
        num_warps=num_warps,
    )
    off = 1
    while off < n_ck:
        _mob_step_kernel[grid](
            *mob,
            *mob_alt,
            M,
            S,
            n_ck,
            off,
            BLOCK_S=block_s,
            num_warps=num_warps,
        )
        mob, mob_alt = mob_alt, mob
        off <<= 1

    _aff_reduce_kernel[grid](
        msi_t,
        si_t,
        h_c,
        a_c,
        q_c,
        lam0_c,
        *mob,
        lam_ck,
        *aff,
        M,
        L,
        S,
        n_ck,
        BLOCK_L=block_l,
        BLOCK_S=block_s,
        num_warps=num_warps,
    )
    off = 1
    while off < n_ck:
        _aff_step_kernel[grid](
            *aff,
            *aff_alt,
            M,
            S,
            n_ck,
            off,
            BLOCK_S=block_s,
            num_warps=num_warps,
        )
        aff, aff_alt = aff_alt, aff
        off <<= 1

    _apply_kernel[grid](
        msi_t,
        si_t,
        h_c,
        w_c,
        a_c,
        q_c,
        eta0_c,
        lam_ck,
        *aff,
        y,
        yvar,
        lam_fin,
        eta_fin,
        eta_ck,
        M,
        L,
        S,
        n_ck,
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


class _PScanKLAScan(torch.autograd.Function):
    """Reduce-then-scan triton forward, with the shared triton backward behind it."""

    @staticmethod
    def forward(ctx, msi, si, h, w, a, q, lam0, eta0, prior, block_l):
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
        ) = pscan_forward(msi, si, h, w, a, q, lam0, eta0, prior=prior, block_l=block_l)
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


def pscan_kla_scan(msi, si, h, w, a, q, lam0, eta0, prior=False, block_l: int = 64):
    """Differentiable parallel-scan KLA scan → ``(y, y_var, lam_fin, eta_fin)``."""
    return _PScanKLAScan.apply(msi, si, h, w, a, q, lam0, eta0, prior, block_l)
