"""Tiled, trace-normalized Möbius scan (forward) — the fast KLA precision scan.

The triton precision scan. Two ideas, both shared with the CUDA kernel:

1. **Linear-space, trace-normalized** 2×2 matmul instead of a log-space
   ``logaddexp`` combine. The Möbius map ``λ ↦ (Aλ+B)/(Cλ+D)`` is stored as the
   matrix ``[[A,B],[C,D]]``; composing two maps is a plain matmul, and dividing
   every entry by the trace ``A+D`` after each step keeps the entries O(1)
   without ever leaving linear space (``λ`` is scale-invariant, so trace-norm
   does not change the result — it only buys numerical stability).

2. **``[BLOCK_L, S]`` tiles on a ``(B, M)`` grid.** One program owns one
   ``(batch, channel)`` pair and streams the sequence in chunks of ``BLOCK_L``
   timesteps, scanning all ``S`` states in parallel (the tile's second axis) and
   carrying the running 2×2 matrix per state across chunks — the same mapping
   the CUDA kernel uses, rather than one tiny program per ``(b, m, s)`` each
   scanning the whole sequence serially.

Forward + exact backward (a scalar-adjoint reverse scan, see
:class:`TiledMobiusScan`). The scan itself is coefficient-agnostic: it consumes
already-built per-(b,l,m,s) leaf coefficients, so nothing here depends on how
``a``/``p`` were discretized.
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


def tiled_mobius_lambda(
    A: torch.Tensor,  # [B, L, M, S] leaf coeff (1+qφ)/a²
    B: torch.Tensor,  # [B, L, M, S] leaf coeff φ
    C: torch.Tensor,  # [B, L, M, S] leaf coeff q/a²
    D: torch.Tensor,  # [B, L, M, S] leaf coeff 1
    lam0: torch.Tensor | None = None,  # [B, M, S] initial precision (default 1)
    block_l: int = 128,
) -> torch.Tensor:
    """Return λ_t [B, L, M, S] via the tiled trace-normalized Möbius scan."""
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


def tiled_linear_eta(
    alpha: torch.Tensor,  # [B, L, M, S] gain α_t
    r: torch.Tensor,  # [B, L, M, S] input r_t (η0 already folded into t=0)
    block_l: int = 128,
) -> torch.Tensor:
    """Return η_t [B, L, M, S] via the tiled affine scan η_t = α_t·η_{t-1} + r_t."""
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
# Differentiable wrappers (forward = tiled kernel, backward = reverse scan).
#
# Both recurrences have a *scalar* adjoint, so each backward is a reverse affine
# scan ν_t = ḡ_t + gain_{t+1}·ν_{t+1}, which equals a forward affine scan on the
# time-flipped sequence — reusing :func:`tiled_linear_eta` (no new kernel).
# ---------------------------------------------------------------------------

_EPS = 1e-12


def _reverse_affine(mult: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """ν_t = src_t + mult_{t+1}·ν_{t+1}, via a forward scan on the flipped axis."""
    src_f = src.flip(1)
    mult_f = mult.flip(1)
    # m_τ = mult_{L-τ}: shift the flipped multipliers right by one (m_0 unused).
    m = torch.cat([torch.zeros_like(mult_f[:, :1]), mult_f[:, :-1]], dim=1)
    u = tiled_linear_eta(m.contiguous(), src_f.contiguous())
    return u.flip(1)


class TiledMobiusScan(torch.autograd.Function):
    """λ_t = (A_t·λ_{t-1} + B_t)/(C_t·λ_{t-1} + D_t), tiled forward + exact backward."""

    @staticmethod
    def forward(ctx, A, B, C, D, lam0):
        lam = tiled_mobius_lambda(A, B, C, D, lam0)
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
        return dA, dB, dC, dD, dlam0


class TiledAffineScan(torch.autograd.Function):
    """η_t = α_t·η_{t-1} + r_t (η_{-1}=0), tiled forward + exact backward."""

    @staticmethod
    def forward(ctx, alpha, r):
        eta = tiled_linear_eta(alpha, r)
        ctx.save_for_backward(alpha, eta)
        return eta

    @staticmethod
    def backward(ctx, deta):
        alpha, eta = ctx.saved_tensors
        eta_prev = torch.cat([torch.zeros_like(eta[:, :1]), eta[:, :-1]], dim=1)
        mu = _reverse_affine(alpha, deta)
        d_alpha = mu * eta_prev if ctx.needs_input_grad[0] else None
        d_r = mu if ctx.needs_input_grad[1] else None
        return d_alpha, d_r


def tiled_mobius_scan(A, B, C, D, lam0=None):
    """Differentiable tiled Möbius scan → λ_t [B, L, M, S]."""
    if lam0 is None:
        Bd, _, Mc, S = A.shape
        lam0 = torch.ones(Bd, Mc, S, device=A.device, dtype=torch.float32)
    return TiledMobiusScan.apply(A, B, C, D, lam0)


def tiled_affine_scan(alpha, r):
    """Differentiable tiled affine scan → η_t [B, L, M, S]."""
    return TiledAffineScan.apply(alpha, r)
