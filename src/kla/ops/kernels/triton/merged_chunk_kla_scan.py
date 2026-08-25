"""``triton_merged_chunk``, ``triton_fused_chunk``'s two scans, in one.

Same shape as :mod:`kla.ops.kernels.triton.chunk_kla_scan`: one program owns
one ``(batch, channel)`` pair and streams the sequence in ``BLOCK_L`` tiles,
doing everything in registers with no ``[B,L,M,S]`` tensor in HBM. Same grid,
same carry, same checkpoints, same backward. What changes is that the leaf
scanned along the tile is the 3x3 map of :func:`_mrg_combine`, which carries η
alongside λ, so there is one ``tl.associative_scan`` where that file runs two.

    load k,q,v·Λ^v,Λ^v  →  φ,r  →  3x3 leaves  →  one trace-normed scan
    →  apply to the carry → λ, η  →  readout  y=Σ_s q·η/λ,  yvar=Σ_s q²/λ

Three things follow, and they are the reason this file exists:

- **One scan instead of two.** ``tl.associative_scan`` over ``BLOCK_L`` is
  ``log2(BLOCK_L)`` rounds of shuffles and predication; that cost is halved.
- **The λ-inversion trick goes away.** ``chunk_kla_scan`` recovers
  ``λ_{t-1} = (λ_t − φ)/(A − C·λ_t)`` by inverting the leaf, specifically so the
  α-gain needs no cross-chunk λ shift. The merged scan never forms α at all,
  its readout has λ and η directly.
- **The carry is two scalars, not five.** ``chunk_kla_scan`` carries the 2x2
  Möbius map accumulated from ``t=0`` plus ``η``; here the tile prefix is
  applied to ``(λ, η)`` at the tile boundary and only that pair crosses it, as
  in the Metal cell this is transcribed from.

Kernel-internal names follow the other triton files: ``msi``→v·Λ^v, ``si``→Λ^v,
``h``→k (key), ``w``→q (query), and the ``q_ptr`` argument is the *process
noise* p, not the query.

The backward is :mod:`kla.ops.kernels.triton.kla_scan_bwd`, unchanged and
unaware, exactly as for the two-scan cells: this forward writes the same
``[B, M, NCK, S]`` checkpoints at the same stride and with the same convention
(the value *entering* step t), and that backward replays a scalar recurrence
from them, never seeing a composed map of any size.

Transcribed from ``kernels/mps/merged_chunk_kla_scan.metal`` and the algebra in
``kernels/mps/kla_merged.metal``; ``kla.ops.kla_ops._merged_combine`` is the
float64 reference both are checked against.
"""

from __future__ import annotations

import triton
import triton.language as tl

from kla.ops.kernels.triton._host import Cell, make_scan
from kla.ops.kernels.triton._tuning import CHUNK_BLOCK_L

_EPS = tl.constexpr(1e-12)


@triton.jit
def _mrg_combine(la, lb, lc, ld, lqa, lqb, ls, ra, rb, rc, rd, rqa, rqb, rs):
    """The map "R after L", trace-normalized.

    Lower block-triangular with a scalar (3,3), so this is never a full 3x3
    product: ``P = P₂·P₁`` is the same 2x2 the other cells compose,
    ``q = q₂·P₁ + s₂·q₁`` is a 1x2 row through it, and ``s = s₂·s₁`` is one
    multiply. Dividing all seven by the 2x2 block's trace is free, λ = u/v and
    η = w/v are both invariant under a common rescale of (u,v,w), and it is
    load-bearing: ``s`` accumulates a⁻ⁿ, which overflows fp32 outright for a
    decaying filter.
    """
    a = ra * la + rb * lc
    b = ra * lb + rb * ld
    c = rc * la + rd * lc
    d = rc * lb + rd * ld
    qa = rqa * la + rqb * lc + rs * lqa
    qb = rqa * lb + rqb * ld + rs * lqb
    s = rs * ls
    inv = 1.0 / tl.maximum(a + d, _EPS)
    return a * inv, b * inv, c * inv, d * inv, qa * inv, qb * inv, s * inv


@triton.jit
def _merged_chunk_fwd_kernel(
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

    # The carry is (λ, η) themselves, not a map accumulated from t=0.
    c_lam = tl.load(lam0_ptr + (b * M + m) * S + s, mask=s_mask, other=1.0)
    c_eta = tl.load(eta0_ptr + (b * M + m) * S + s, mask=s_mask, other=0.0)

    # Static dynamics are loop-invariant, so hoist them out of the chunk loop.
    a_st = tl.load(a_ptr + m * S + s, mask=s_mask, other=1.0)
    q_st = tl.load(q_ptr + m * S + s, mask=s_mask, other=0.0)
    a2_st = tl.maximum(a_st * a_st, _EPS)
    inv_a2_st = 1.0 / a2_st
    inv_a_st = 1.0 / (a_st + tl.where(a_st < 0.0, -_EPS, _EPS))

    base_ml = (b * M + m) * L  # msi/si[b,m,:], output[b,m,:]
    base_hw = b * L * S  # h[b,:,:], w[b,:,:]
    base_ck = (b * M + m) * N_CHUNKS

    for c in range(N_CHUNKS):
        tt = c * BLOCK_L + t
        t_mask = tt < L

        # The state *entering* this chunk is what kla_scan_bwd resumes from, so
        # the store precedes the scan. Both checkpoints are written here; in the
        # two-scan cell η did not exist yet at this point.
        if STORE_CK:
            tl.store(lam_ck_ptr + (base_ck + c) * S + s, c_lam, mask=s_mask)
            tl.store(eta_ck_ptr + (base_ck + c) * S + s, c_eta, mask=s_mask)

        hoff = base_hw + tt[:, None] * S + s[None, :]
        hw_mask = t_mask[:, None] & s_mask[None, :]

        msi = tl.load(msi_ptr + base_ml + tt, mask=t_mask, other=0.0)[:, None]
        si = tl.load(si_ptr + base_ml + tt, mask=t_mask, other=0.0)[:, None]
        h = tl.load(h_ptr + hoff, mask=hw_mask, other=0.0)
        wv = tl.load(w_ptr + hoff, mask=hw_mask, other=0.0)

        phi = tl.maximum(si * h * h, _EPS)
        rr = msi * h

        # Broadcast a/p to the full [BLOCK_L, S] tile (they load as [1, S]).
        zero = tl.zeros([BLOCK_L, BLOCK_S], tl.float32)
        a_t = a_st[None, :] + zero
        q_t = q_st[None, :] + zero
        a2_t = a2_st[None, :] + zero
        inv_a2_t = inv_a2_st[None, :] + zero

        # The 3x3 leaf, built from (φ, r, a, p) alone, nothing in it reads λ,
        # which is the entire point.
        C = q_t * inv_a2_t
        A = (1.0 + q_t * phi) * inv_a2_t
        D = zero + 1.0
        Qa = rr * C
        Qb = rr  # r·D with D = 1
        Sg = inv_a_st[None, :] + zero

        # Rows past the end of the sequence must compose as the identity, or a
        # partial final chunk would not reduce to what its live prefix does.
        A = tl.where(t_mask[:, None], A, 1.0)
        B = tl.where(t_mask[:, None], phi, 0.0)
        C = tl.where(t_mask[:, None], C, 0.0)
        Qa = tl.where(t_mask[:, None], Qa, 0.0)
        Qb = tl.where(t_mask[:, None], Qb, 0.0)
        Sg = tl.where(t_mask[:, None], Sg, 1.0)

        sA, sB, sC, sD, sQa, sQb, sS = tl.associative_scan(
            (A, B, C, D, Qa, Qb, Sg), axis=0, combine_fn=_mrg_combine
        )

        # Apply the inclusive prefix to the homogeneous vector (λ, 1, η). Both
        # quotients share the denominator the 2x2 read-out already forms.
        den = tl.maximum(sC * c_lam[None, :] + sD, _EPS)
        lam = (sA * c_lam[None, :] + sB) / den
        eta = (sQa * c_lam[None, :] + sQb + sS * c_eta[None, :]) / den

        var = 1.0 / tl.maximum(lam, _EPS)
        mean = eta * var
        if PRIOR:
            # decode_from_prior: read out one predict step ahead.
            mean = a_t * mean
            var = a2_t * var + q_t
        y = tl.sum(tl.where(hw_mask, mean * wv, 0.0), axis=1)
        yvar = tl.sum(tl.where(hw_mask, var * wv * wv, 0.0), axis=1)
        tl.store(y_ptr + base_ml + tt, y, mask=t_mask)
        tl.store(yvar_ptr + base_ml + tt, yvar, mask=t_mask)

        # One carry for both recurrences, taken from the last live row.
        last = tl.minimum(BLOCK_L, L - c * BLOCK_L) - 1
        sel = (t == last)[:, None]
        c_lam = tl.sum(tl.where(sel, lam, 0.0), axis=0)
        c_eta = tl.sum(tl.where(sel, eta, 0.0), axis=0)

    tl.store(lam_fin_ptr + (b * M + m) * S + s, c_lam, mask=s_mask)
    tl.store(eta_fin_ptr + (b * M + m) * S + s, c_eta, mask=s_mask)


CELL = Cell(
    kernel=_merged_chunk_fwd_kernel,
    default_block_l=CHUNK_BLOCK_L,
)

merged_chunk_kla_scan = make_scan(
    CELL,
    "merged_chunk_kla_scan",
    """Differentiable merged fused KLA scan → ``(y, y_var, lam_fin, eta_fin)``.""",
)
