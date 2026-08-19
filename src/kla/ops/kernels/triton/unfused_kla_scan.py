"""The scans behind the unfused triton cells — λ and η, three implementations each.

"Unfused" here means the kernels consume already-built per-``(b,l,m,s)`` leaf
coefficients and write per-``(b,l,m,s)`` results, with the sufficient statistics
and the read-out done in torch around them. That costs several HBM passes the
fused cells avoid, and buys a scan that is coefficient-agnostic: nothing here
depends on how ``a``/``p`` were discretized.

Two ideas are common to every implementation, and shared with the other backends:

1. **Linear-space, trace-normalized** 2×2 matmul instead of a log-space
   ``logaddexp`` combine. The Möbius map ``λ ↦ (Aλ+B)/(Cλ+D)`` is stored as the
   matrix ``[[A,B],[C,D]]``; composing two maps is a plain matmul, and dividing
   every entry by the trace ``A+D`` after each step keeps the entries O(1)
   without ever leaving linear space (``λ`` is scale-invariant, so trace-norm
   does not change the result — it only buys numerical stability).

2. **All ``S`` states in parallel**, as a tile axis, on a grid over
   ``(batch, channel)`` — rather than one tiny program per ``(b, m, s)``.

The three implementations differ only in how time is walked (see
``docs/implementations.md``):

``recurrent``
    One program per ``(b, m)``, a timestep at a time, the map *applied* to a
    running λ. No composition, no trace normalization, nothing carried but the
    state itself.
``chunk`` (the default)
    One program per ``(b, m)``, streaming ``BLOCK_L`` timesteps at a time:
    ``tl.associative_scan`` within a tile, the running 2×2 matrix carried
    across tiles.
``pscan``
    A program per ``(b, m, chunk)``, so no program waits on its neighbour: each
    chunk is reduced on its own and the chunks are resolved by a parallel scan
    over their aggregates, ping-ponged across launches.

There is **one backward, not one per implementation** — see :class:`MobiusScan`. Both
recurrences have a scalar adjoint that depends on the *values* λ and η, not on
the order they were produced in, so the reverse scan is the same work whichever
forward ran. It uses the chunk implementation throughout.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _tracenorm_combine(la, lb, lc, ld, ra, rb, rc, rd):
    """Compose two 2×2 Möbius matrices (R∘L = R·L), then divide by the trace."""
    a = ra * la + rb * lc
    b = ra * lb + rb * ld
    c = rc * la + rd * lc
    d = rc * lb + rd * ld
    inv = 1.0 / tl.maximum(a + d, 1e-12)
    return a * inv, b * inv, c * inv, d * inv


@triton.jit
def _tiled_mobius_fwd_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    lam0_ptr,
    out_ptr,
    M,
    L,
    S,
    N_CHUNKS,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M
    base = (b * M + m) * L * S  # offset into [B, M, L, S]

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    t = tl.arange(0, BLOCK_L)

    lam0 = tl.load(lam0_ptr + (b * M + m) * S + s, mask=s_mask, other=1.0)  # [S]

    # Running cumulative Möbius matrix per state, init = identity.
    cA = tl.zeros([BLOCK_S], tl.float32) + 1.0
    cB = tl.zeros([BLOCK_S], tl.float32)
    cC = tl.zeros([BLOCK_S], tl.float32)
    cD = tl.zeros([BLOCK_S], tl.float32) + 1.0

    for c in range(N_CHUNKS):
        tt = c * BLOCK_L + t
        t_mask = tt < L
        off = base + tt[:, None] * S + s[None, :]
        mask = t_mask[:, None] & s_mask[None, :]

        lA = tl.load(A_ptr + off, mask=mask, other=1.0)
        lB = tl.load(B_ptr + off, mask=mask, other=0.0)
        lC = tl.load(C_ptr + off, mask=mask, other=0.0)
        lD = tl.load(D_ptr + off, mask=mask, other=1.0)

        # Inclusive scan along time (axis 0); each state column is independent.
        sA, sB, sC, sD = tl.associative_scan(
            (lA, lB, lC, lD), axis=0, combine_fn=_tracenorm_combine
        )

        # Fold in the carry from earlier chunks: full = scan ∘ carry.
        fA = sA * cA[None, :] + sB * cC[None, :]
        fB = sA * cB[None, :] + sB * cD[None, :]
        fC = sC * cA[None, :] + sD * cC[None, :]
        fD = sC * cB[None, :] + sD * cD[None, :]
        inv = 1.0 / tl.maximum(fA + fD, 1e-12)
        fA, fB, fC, fD = fA * inv, fB * inv, fC * inv, fD * inv

        # λ_t = (A·λ0 + B) / (C·λ0 + D)  (scale-invariant: trace-norm cancels).
        num = fA * lam0[None, :] + fB
        den = fC * lam0[None, :] + fD
        tl.store(out_ptr + off, num / tl.maximum(den, 1e-12), mask=mask)

        # Carry forward = cumulative matrix at the last valid timestep of the chunk.
        last = tl.minimum(BLOCK_L, L - c * BLOCK_L) - 1
        sel = (t == last)[:, None]
        cA = tl.sum(tl.where(sel, fA, 0.0), axis=0)
        cB = tl.sum(tl.where(sel, fB, 0.0), axis=0)
        cC = tl.sum(tl.where(sel, fC, 0.0), axis=0)
        cD = tl.sum(tl.where(sel, fD, 0.0), axis=0)


def _mobius_lambda_chunk(
    A: torch.Tensor,  # [B, L, M, S] leaf coeff (1+qφ)/a²
    B: torch.Tensor,  # [B, L, M, S] leaf coeff φ
    C: torch.Tensor,  # [B, L, M, S] leaf coeff q/a²
    D: torch.Tensor,  # [B, L, M, S] leaf coeff 1
    lam0: torch.Tensor | None = None,  # [B, M, S] initial precision (default 1)
    block_l: int = 128,
) -> torch.Tensor:
    """λ_t [B, L, M, S], chunk-implementationd: scan a tile, carry the matrix."""
    Bd, L, Mc, S = A.shape

    def to_bmls(x):
        return x.permute(0, 2, 1, 3).contiguous()  # [B, M, L, S]

    Ai, Bi, Ci, Di = to_bmls(A), to_bmls(B), to_bmls(C), to_bmls(D)
    if lam0 is None:
        lam0 = torch.ones(Bd, Mc, S, device=A.device, dtype=torch.float32)
    lam0 = lam0.contiguous()
    out = torch.empty_like(Ai)

    n_chunks = triton.cdiv(L, block_l)
    block_s = triton.next_power_of_2(S)
    num_warps = 2 if block_l <= 128 else 4
    _tiled_mobius_fwd_kernel[(Bd * Mc,)](
        Ai,
        Bi,
        Ci,
        Di,
        lam0,
        out,
        Mc,
        L,
        S,
        n_chunks,
        BLOCK_L=block_l,
        BLOCK_S=block_s,
        num_warps=num_warps,
    )
    return out.permute(0, 2, 1, 3).contiguous()  # [B, L, M, S]


@triton.jit
def _affine_combine(la, lb, ra, rb):
    """Compose affine maps x ↦ a·x + b  (R∘L): (a_R·a_L, a_R·b_L + b_R)."""
    return ra * la, ra * lb + rb


@triton.jit
def _tiled_linear_fwd_kernel(
    alpha_ptr,
    r_ptr,
    out_ptr,
    M,
    L,
    S,
    N_CHUNKS,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M
    base = (b * M + m) * L * S

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    t = tl.arange(0, BLOCK_L)

    ca = tl.zeros([BLOCK_S], tl.float32) + 1.0  # running gain
    cb = tl.zeros([BLOCK_S], tl.float32)  # running value (η)

    for c in range(N_CHUNKS):
        tt = c * BLOCK_L + t
        t_mask = tt < L
        off = base + tt[:, None] * S + s[None, :]
        mask = t_mask[:, None] & s_mask[None, :]

        al = tl.load(alpha_ptr + off, mask=mask, other=1.0)
        rr = tl.load(r_ptr + off, mask=mask, other=0.0)
        sa, sb = tl.associative_scan((al, rr), axis=0, combine_fn=_affine_combine)

        # full = scan ∘ carry: η_t = sa·cb + sb,  gain = sa·ca
        fb = sa * cb[None, :] + sb
        fa = sa * ca[None, :]
        tl.store(out_ptr + off, fb, mask=mask)

        last = tl.minimum(BLOCK_L, L - c * BLOCK_L) - 1
        sel = (t == last)[:, None]
        ca = tl.sum(tl.where(sel, fa, 0.0), axis=0)
        cb = tl.sum(tl.where(sel, fb, 0.0), axis=0)


def _affine_eta_chunk(
    alpha: torch.Tensor,  # [B, L, M, S] gain α_t
    r: torch.Tensor,  # [B, L, M, S] input r_t (η0 already folded into t=0)
    block_l: int = 128,
) -> torch.Tensor:
    """η_t [B, L, M, S], chunk-implementationd: scan a tile, carry the map."""
    Bd, L, Mc, S = alpha.shape

    def to_bmls(x):
        return x.permute(0, 2, 1, 3).contiguous()

    ai, ri = to_bmls(alpha), to_bmls(r)
    out = torch.empty_like(ai)
    n_chunks = triton.cdiv(L, block_l)
    block_s = triton.next_power_of_2(S)
    num_warps = 2 if block_l <= 128 else 4
    _tiled_linear_fwd_kernel[(Bd * Mc,)](
        ai,
        ri,
        out,
        Mc,
        L,
        S,
        n_chunks,
        BLOCK_L=block_l,
        BLOCK_S=block_s,
        num_warps=num_warps,
    )
    return out.permute(0, 2, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# recurrent — one program per (b, m), a timestep at a time, the map applied.
# No composition, so no 2x2 matrix and no trace normalization: the cheapest
# arithmetic of the three, and the least parallelism.
# ---------------------------------------------------------------------------


@triton.jit
def _recurrent_mobius_fwd_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    lam0_ptr,
    out_ptr,
    M,
    L,
    S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M
    base = (b * M + m) * L * S

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    lam = tl.load(lam0_ptr + (b * M + m) * S + s, mask=s_mask, other=1.0)

    for t in range(L):
        off = base + t * S + s
        lA = tl.load(A_ptr + off, mask=s_mask, other=1.0)
        lB = tl.load(B_ptr + off, mask=s_mask, other=0.0)
        lC = tl.load(C_ptr + off, mask=s_mask, other=0.0)
        lD = tl.load(D_ptr + off, mask=s_mask, other=1.0)
        lam = (lA * lam + lB) / tl.maximum(lC * lam + lD, 1e-12)
        tl.store(out_ptr + off, lam, mask=s_mask)


@triton.jit
def _recurrent_affine_fwd_kernel(
    alpha_ptr,
    r_ptr,
    out_ptr,
    M,
    L,
    S,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // M
    m = pid % M
    base = (b * M + m) * L * S

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    eta = tl.zeros([BLOCK_S], tl.float32)  # eta_{-1} = 0; eta0 is folded into r_0

    for t in range(L):
        off = base + t * S + s
        al = tl.load(alpha_ptr + off, mask=s_mask, other=1.0)
        rr = tl.load(r_ptr + off, mask=s_mask, other=0.0)
        eta = al * eta + rr
        tl.store(out_ptr + off, eta, mask=s_mask)


# ---------------------------------------------------------------------------
# pscan — a program per (b, m, chunk), so nothing waits on its neighbour.
#
# Reduce each chunk to one aggregate, resolve the aggregates with a parallel
# scan, then apply. The doubling rounds are the fused cell's, imported rather
# than repeated: same [B, M, NCK, S] aggregate layout, same semantics.
#
# Rows past the end of the sequence load the identity, so a chunk's aggregate is
# simply the last row of its inclusive tile scan, whatever the tail alignment.
# ---------------------------------------------------------------------------


@triton.jit
def _pscan_mob_reduce_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    ma_ptr,
    mb_ptr,
    mc_ptr,
    md_ptr,
    M,
    L,
    S,
    N_CHUNKS,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % N_CHUNKS
    bm = pid // N_CHUNKS
    base = bm * L * S

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    t = tl.arange(0, BLOCK_L)
    tt = c * BLOCK_L + t
    off = base + tt[:, None] * S + s[None, :]
    mask = (tt < L)[:, None] & s_mask[None, :]

    lA = tl.load(A_ptr + off, mask=mask, other=1.0)
    lB = tl.load(B_ptr + off, mask=mask, other=0.0)
    lC = tl.load(C_ptr + off, mask=mask, other=0.0)
    lD = tl.load(D_ptr + off, mask=mask, other=1.0)
    sA, sB, sC, sD = tl.associative_scan(
        (lA, lB, lC, lD), axis=0, combine_fn=_tracenorm_combine
    )

    sel = (t == BLOCK_L - 1)[:, None]
    agg = pid * S + s
    tl.store(ma_ptr + agg, tl.sum(tl.where(sel, sA, 0.0), axis=0), mask=s_mask)
    tl.store(mb_ptr + agg, tl.sum(tl.where(sel, sB, 0.0), axis=0), mask=s_mask)
    tl.store(mc_ptr + agg, tl.sum(tl.where(sel, sC, 0.0), axis=0), mask=s_mask)
    tl.store(md_ptr + agg, tl.sum(tl.where(sel, sD, 0.0), axis=0), mask=s_mask)


@triton.jit
def _pscan_mob_apply_kernel(
    A_ptr,
    B_ptr,
    C_ptr,
    D_ptr,
    lam0_ptr,
    ma_ptr,
    mb_ptr,
    mc_ptr,
    md_ptr,
    out_ptr,
    M,
    L,
    S,
    N_CHUNKS,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % N_CHUNKS
    bm = pid // N_CHUNKS
    base = bm * L * S

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    t = tl.arange(0, BLOCK_L)
    tt = c * BLOCK_L + t
    off = base + tt[:, None] * S + s[None, :]
    mask = (tt < L)[:, None] & s_mask[None, :]

    # The exclusive prefix maps lam0 to the state entering this chunk. Chunk 0
    # reads the identity out of range, so it needs no branch of its own.
    prev = tl.maximum(pid - 1, 0) * S + s
    take = s_mask & (c > 0)
    pa = tl.load(ma_ptr + prev, mask=take, other=1.0)
    pb = tl.load(mb_ptr + prev, mask=take, other=0.0)
    pc = tl.load(mc_ptr + prev, mask=take, other=0.0)
    pd = tl.load(md_ptr + prev, mask=take, other=1.0)
    lam0 = tl.load(lam0_ptr + bm * S + s, mask=s_mask, other=1.0)
    lam_in = (pa * lam0 + pb) / tl.maximum(pc * lam0 + pd, 1e-12)

    lA = tl.load(A_ptr + off, mask=mask, other=1.0)
    lB = tl.load(B_ptr + off, mask=mask, other=0.0)
    lC = tl.load(C_ptr + off, mask=mask, other=0.0)
    lD = tl.load(D_ptr + off, mask=mask, other=1.0)
    sA, sB, sC, sD = tl.associative_scan(
        (lA, lB, lC, lD), axis=0, combine_fn=_tracenorm_combine
    )

    num = sA * lam_in[None, :] + sB
    den = sC * lam_in[None, :] + sD
    tl.store(out_ptr + off, num / tl.maximum(den, 1e-12), mask=mask)


@triton.jit
def _pscan_aff_reduce_kernel(
    alpha_ptr,
    r_ptr,
    aa_ptr,
    ar_ptr,
    M,
    L,
    S,
    N_CHUNKS,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % N_CHUNKS
    bm = pid // N_CHUNKS
    base = bm * L * S

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    t = tl.arange(0, BLOCK_L)
    tt = c * BLOCK_L + t
    off = base + tt[:, None] * S + s[None, :]
    mask = (tt < L)[:, None] & s_mask[None, :]

    al = tl.load(alpha_ptr + off, mask=mask, other=1.0)
    rr = tl.load(r_ptr + off, mask=mask, other=0.0)
    sa, sb = tl.associative_scan((al, rr), axis=0, combine_fn=_affine_combine)

    sel = (t == BLOCK_L - 1)[:, None]
    agg = pid * S + s
    tl.store(aa_ptr + agg, tl.sum(tl.where(sel, sa, 0.0), axis=0), mask=s_mask)
    tl.store(ar_ptr + agg, tl.sum(tl.where(sel, sb, 0.0), axis=0), mask=s_mask)


@triton.jit
def _pscan_aff_apply_kernel(
    alpha_ptr,
    r_ptr,
    aa_ptr,
    ar_ptr,
    out_ptr,
    M,
    L,
    S,
    N_CHUNKS,
    BLOCK_L: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % N_CHUNKS
    bm = pid // N_CHUNKS
    base = bm * L * S

    s = tl.arange(0, BLOCK_S)
    s_mask = s < S
    t = tl.arange(0, BLOCK_L)
    tt = c * BLOCK_L + t
    off = base + tt[:, None] * S + s[None, :]
    mask = (tt < L)[:, None] & s_mask[None, :]

    # eta_{-1} = 0 (eta0 is folded into r_0 by the caller), so the exclusive
    # prefix evaluated at zero is just its offset -- and the identity chunk 0
    # reads out of range gives zero, as it must.
    prev = tl.maximum(pid - 1, 0) * S + s
    eta_in = tl.load(ar_ptr + prev, mask=s_mask & (c > 0), other=0.0)

    al = tl.load(alpha_ptr + off, mask=mask, other=1.0)
    rr = tl.load(r_ptr + off, mask=mask, other=0.0)
    sa, sb = tl.associative_scan((al, rr), axis=0, combine_fn=_affine_combine)
    tl.store(out_ptr + off, sa * eta_in[None, :] + sb, mask=mask)


def _doubling(kernel, bufs, alts, grid, M, S, n_ck, block_s, num_warps):
    """Hillis-Steele over the chunk axis, ping-ponging two sets of buffers."""
    off = 1
    while off < n_ck:
        kernel[grid](
            *bufs, *alts, M, S, n_ck, off, BLOCK_S=block_s, num_warps=num_warps
        )
        bufs, alts = alts, bufs
        off <<= 1
    return bufs


# ---------------------------------------------------------------------------
# The two public forwards: pick a implementation, get the same lambda / eta.
# ---------------------------------------------------------------------------

IMPLEMENTATIONS = ("recurrent", "chunk", "pscan")


def _geometry(shape, block_l):
    Bd, L, Mc, S = shape
    return Bd, L, Mc, S, triton.cdiv(L, block_l), triton.next_power_of_2(S)


def mobius_lambda(A, B, C, D, lam0=None, implementation="chunk", block_l: int = 128):
    """λ_t [B, L, M, S] from the leaf coefficients, on the named implementation."""
    if implementation == "chunk":
        return _mobius_lambda_chunk(A, B, C, D, lam0, block_l)
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation {implementation!r}; "
            f"expected one of {IMPLEMENTATIONS}"
        )

    Bd, L, Mc, S, n_ck, block_s = _geometry(A.shape, block_l)
    num_warps = 2 if block_l <= 128 else 4

    def to_bmls(x):
        return x.permute(0, 2, 1, 3).contiguous()  # [B, M, L, S]

    Ai, Bi, Ci, Di = to_bmls(A), to_bmls(B), to_bmls(C), to_bmls(D)
    if lam0 is None:
        lam0 = torch.ones(Bd, Mc, S, device=A.device, dtype=torch.float32)
    lam0 = lam0.contiguous()
    out = torch.empty_like(Ai)

    if implementation == "recurrent":
        _recurrent_mobius_fwd_kernel[(Bd * Mc,)](
            Ai,
            Bi,
            Ci,
            Di,
            lam0,
            out,
            Mc,
            L,
            S,
            BLOCK_S=block_s,
            num_warps=num_warps,
        )
        return out.permute(0, 2, 1, 3).contiguous()

    from kla.ops.kernels.triton.pscan_kla_scan import _mob_step_kernel

    grid = (Bd * Mc * n_ck,)
    agg = [
        torch.empty(Bd, Mc, n_ck, S, device=A.device, dtype=torch.float32)
        for _ in range(8)
    ]
    _pscan_mob_reduce_kernel[grid](
        Ai,
        Bi,
        Ci,
        Di,
        *agg[:4],
        Mc,
        L,
        S,
        n_ck,
        BLOCK_L=block_l,
        BLOCK_S=block_s,
        num_warps=num_warps,
    )
    incl = _doubling(
        _mob_step_kernel, agg[:4], agg[4:], grid, Mc, S, n_ck, block_s, num_warps
    )
    _pscan_mob_apply_kernel[grid](
        Ai,
        Bi,
        Ci,
        Di,
        lam0,
        *incl,
        out,
        Mc,
        L,
        S,
        n_ck,
        BLOCK_L=block_l,
        BLOCK_S=block_s,
        num_warps=num_warps,
    )
    return out.permute(0, 2, 1, 3).contiguous()


def affine_eta(alpha, r, implementation="chunk", block_l: int = 128):
    """η_t [B, L, M, S] = α_t·η_{t-1} + r_t, on the named implementation."""
    if implementation == "chunk":
        return _affine_eta_chunk(alpha, r, block_l)
    if implementation not in IMPLEMENTATIONS:
        raise ValueError(
            f"Unknown implementation {implementation!r}; "
            f"expected one of {IMPLEMENTATIONS}"
        )

    Bd, L, Mc, S, n_ck, block_s = _geometry(alpha.shape, block_l)
    num_warps = 2 if block_l <= 128 else 4

    def to_bmls(x):
        return x.permute(0, 2, 1, 3).contiguous()

    ai, ri = to_bmls(alpha), to_bmls(r)
    out = torch.empty_like(ai)

    if implementation == "recurrent":
        _recurrent_affine_fwd_kernel[(Bd * Mc,)](
            ai, ri, out, Mc, L, S, BLOCK_S=block_s, num_warps=num_warps
        )
        return out.permute(0, 2, 1, 3).contiguous()

    from kla.ops.kernels.triton.pscan_kla_scan import _aff_step_kernel

    grid = (Bd * Mc * n_ck,)
    agg = [
        torch.empty(Bd, Mc, n_ck, S, device=alpha.device, dtype=torch.float32)
        for _ in range(4)
    ]
    _pscan_aff_reduce_kernel[grid](
        ai,
        ri,
        *agg[:2],
        Mc,
        L,
        S,
        n_ck,
        BLOCK_L=block_l,
        BLOCK_S=block_s,
        num_warps=num_warps,
    )
    incl = _doubling(
        _aff_step_kernel, agg[:2], agg[2:], grid, Mc, S, n_ck, block_s, num_warps
    )
    _pscan_aff_apply_kernel[grid](
        ai,
        ri,
        *incl,
        out,
        Mc,
        L,
        S,
        n_ck,
        BLOCK_L=block_l,
        BLOCK_S=block_s,
        num_warps=num_warps,
    )
    return out.permute(0, 2, 1, 3).contiguous()


# ---------------------------------------------------------------------------
# Differentiable wrappers (forward = tiled kernel, backward = reverse scan).
#
# Both recurrences have a *scalar* adjoint, so each backward is a reverse affine
# scan ν_t = ḡ_t + gain_{t+1}·ν_{t+1}, which equals a forward affine scan on the
# time-flipped sequence — reusing :func:`affine_eta` (no new kernel).
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _reverse_affine(mult: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """ν_t = src_t + mult_{t+1}·ν_{t+1}, via a forward scan on the flipped axis."""
    src_f = src.flip(1)
    mult_f = mult.flip(1)
    # m_τ = mult_{L-τ}: shift the flipped multipliers right by one (m_0 unused).
    m = torch.cat([torch.zeros_like(mult_f[:, :1]), mult_f[:, :-1]], dim=1)
    u = affine_eta(m.contiguous(), src_f.contiguous())
    return u.flip(1)


class MobiusScan(torch.autograd.Function):
    """λ_t = (A_t·λ_{t-1} + B_t)/(C_t·λ_{t-1} + D_t): any forward, one backward.

    The adjoint reads the *values* λ and λ_{t-1}, not the order they were
    produced in, so ``implementation`` reaches the forward and stops there.
    """

    @staticmethod
    def forward(ctx, A, B, C, D, lam0, implementation):
        lam = mobius_lambda(A, B, C, D, lam0, implementation)
        ctx.save_for_backward(A, B, C, D, lam0, lam)
        return lam

    @staticmethod
    def backward(ctx, dlam):
        A, B, C, D, lam0, lam = ctx.saved_tensors
        lam_prev = torch.cat([lam0.unsqueeze(1), lam[:, :-1]], dim=1)
        den = (C * lam_prev + D).clamp_min(_EPS)
        gain = (A * D - B * C) / (den * den)  # ∂λ_t/∂λ_{t-1}
        nu = _reverse_affine(gain, dlam)
        nu_over_den = nu / den
        dA = nu_over_den * lam_prev
        dB = nu_over_den
        dC = -nu_over_den * lam * lam_prev
        dD = -nu_over_den * lam
        dlam0 = (nu[:, 0] * gain[:, 0]) if ctx.needs_input_grad[4] else None
        return dA, dB, dC, dD, dlam0, None


class AffineScan(torch.autograd.Function):
    """η_t = α_t·η_{t-1} + r_t (η_{-1}=0): any forward, one backward."""

    @staticmethod
    def forward(ctx, alpha, r, implementation):
        eta = affine_eta(alpha, r, implementation)
        ctx.save_for_backward(alpha, eta)
        return eta

    @staticmethod
    def backward(ctx, deta):
        alpha, eta = ctx.saved_tensors
        eta_prev = torch.cat([torch.zeros_like(eta[:, :1]), eta[:, :-1]], dim=1)
        mu = _reverse_affine(alpha, deta)
        d_alpha = mu * eta_prev if ctx.needs_input_grad[0] else None
        d_r = mu if ctx.needs_input_grad[1] else None
        return d_alpha, d_r, None


def mobius_scan(A, B, C, D, lam0=None, implementation="chunk"):
    """Differentiable Möbius scan → λ_t [B, L, M, S]."""
    if lam0 is None:
        Bd, _, Mc, S = A.shape
        lam0 = torch.ones(Bd, Mc, S, device=A.device, dtype=torch.float32)
    return MobiusScan.apply(A, B, C, D, lam0, implementation)


def affine_scan(alpha, r, implementation="chunk"):
    """Differentiable affine scan → η_t [B, L, M, S]."""
    return AffineScan.apply(alpha, r, implementation)
