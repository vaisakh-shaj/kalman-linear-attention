"""Lane-per-state Möbius and affine scans on Metal — the composed MPS path.

The MPS counterpart of :mod:`kla.ops.kernels.triton.tiled_mobius_scan`: the two
KLA recurrences as standalone differentiable ops over ``[B, L, M, S]``
coefficient tensors, which the backend surrounds with ordinary torch
elementwise work. One thread owns one ``(b, m, s)`` triple and time is the
serial axis, so each recurrence is applied directly rather than scanned — which
is why these kernels need neither matrix composition nor trace normalization.

Backward is the exact adjoint: each recurrence has a *scalar* per-step gain, so
the adjoint is the reverse affine recurrence ``ν_t = ḡ_t + gain_{t+1}·ν_{t+1}``.
:func:`reverse_affine` walks time downwards in a kernel of its own, which is
free when time is already the serial axis and saves the three ``[B, L, M, S]``
materializations triton's flip-and-rescan costs.
"""

from __future__ import annotations

import torch

from kla.ops.kernels.mps._shaders import check_inputs, scan_library

_EPS = 1e-12
_GROUP = 256


def _dims(x: torch.Tensor) -> tuple[int, int, int, int]:
    B, L, M, S = x.shape
    return B, L, M, S


def _launch(kernel, args, n_threads: int) -> None:
    kernel(*args, threads=n_threads, group_size=min(_GROUP, n_threads))


def mobius_lambda(
    A: torch.Tensor,  # [B, L, M, S] leaf coeff (1+pφ)/a²
    B: torch.Tensor,  # [B, L, M, S] leaf coeff φ
    C: torch.Tensor,  # [B, L, M, S] leaf coeff p/a²
    D: torch.Tensor,  # [B, L, M, S] leaf coeff 1
    lam0: torch.Tensor | None = None,  # [B, M, S] initial precision (default 1)
) -> torch.Tensor:
    """λ_t = (A_t·λ_{t-1} + B_t)/(C_t·λ_{t-1} + D_t) → ``[B, L, M, S]``."""
    check_inputs(A, B, C, D)
    Bd, L, M, S = _dims(A)
    if lam0 is None:
        lam0 = torch.ones(Bd, M, S, device=A.device, dtype=torch.float32)
    lam0 = lam0.contiguous()
    check_inputs(lam0)

    out = torch.empty_like(A)
    n = Bd * M * S
    _launch(
        scan_library().kla_mobius_scan_fwd,
        (out, A, B, C, D, lam0, n, L, M, S),
        n,
    )
    return out


def affine_eta(alpha: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """η_t = α_t·η_{t-1} + r_t (η_{-1}=0) → ``[B, L, M, S]``."""
    check_inputs(alpha, r)
    Bd, L, M, S = _dims(alpha)
    out = torch.empty_like(alpha)
    n = Bd * M * S
    _launch(scan_library().kla_affine_scan_fwd, (out, alpha, r, n, L, M, S), n)
    return out


def reverse_affine(mult: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """ν_t = src_t + mult_{t+1}·ν_{t+1} (ν_L=0) → ``[B, L, M, S]``."""
    check_inputs(mult, src)
    Bd, L, M, S = _dims(mult)
    out = torch.empty_like(mult)
    n = Bd * M * S
    _launch(scan_library().kla_affine_scan_rev, (out, mult, src, n, L, M, S), n)
    return out


class _MobiusScan(torch.autograd.Function):
    """λ_t = (A_t·λ_{t-1} + B_t)/(C_t·λ_{t-1} + D_t), kernel fwd + exact backward."""

    @staticmethod
    def forward(ctx, A, B, C, D, lam0):
        lam = mobius_lambda(A, B, C, D, lam0)
        ctx.save_for_backward(A, B, C, D, lam0, lam)
        return lam

    @staticmethod
    def backward(ctx, dlam):
        A, B, C, D, lam0, lam = ctx.saved_tensors
        lam_prev = torch.cat([lam0.unsqueeze(1), lam[:, :-1]], dim=1)
        den = (C * lam_prev + D).clamp_min(_EPS)
        gain = (A * D - B * C) / (den * den)  # ∂λ_t/∂λ_{t-1}
        nu = reverse_affine(gain.contiguous(), dlam.contiguous())
        nu_over_den = nu / den
        dA = nu_over_den * lam_prev
        dB = nu_over_den
        dC = -nu_over_den * lam * lam_prev
        dD = -nu_over_den * lam
        dlam0 = (nu[:, 0] * gain[:, 0]) if ctx.needs_input_grad[4] else None
        return dA, dB, dC, dD, dlam0


class _AffineScan(torch.autograd.Function):
    """η_t = α_t·η_{t-1} + r_t (η_{-1}=0), kernel fwd + exact backward."""

    @staticmethod
    def forward(ctx, alpha, r):
        eta = affine_eta(alpha, r)
        ctx.save_for_backward(alpha, eta)
        return eta

    @staticmethod
    def backward(ctx, deta):
        alpha, eta = ctx.saved_tensors
        eta_prev = torch.cat([torch.zeros_like(eta[:, :1]), eta[:, :-1]], dim=1)
        mu = reverse_affine(alpha, deta.contiguous())
        d_alpha = mu * eta_prev if ctx.needs_input_grad[0] else None
        d_r = mu if ctx.needs_input_grad[1] else None
        return d_alpha, d_r


def mobius_scan(A, B, C, D, lam0=None):
    """Differentiable Möbius scan → λ_t ``[B, L, M, S]``."""
    if lam0 is None:
        Bd, _, M, S = A.shape
        lam0 = torch.ones(Bd, M, S, device=A.device, dtype=torch.float32)
    return _MobiusScan.apply(A, B, C, D, lam0)


def affine_scan(alpha, r):
    """Differentiable affine scan → η_t ``[B, L, M, S]``."""
    return _AffineScan.apply(alpha, r)
