"""Triton backend for the KLA scan.

Composes two ``[BLOCK_L, S]``-tiled, ``(B, M)``-grid scans (forward + exact
backward), both from :mod:`kla.ops.kernels.triton.tiled_mobius_scan`:

1. ``tiled_mobius_scan`` — the precision recurrence λ_t = (Aλ' + B)/(Cλ' + D) as
   a *linear-space, trace-normalized* Möbius scan, the same formulation the CUDA
   kernel uses: composing the 2×2 maps is a plain matmul normalized by the trace,
   which keeps entries O(1) without log-space since λ is scale-invariant.
   Backward is a scalar-adjoint reverse scan.
2. ``tiled_affine_scan`` — the information-vector recurrence η_t = α_t·η' + r_t.

One program owns one ``(batch, channel)`` pair and streams the sequence in
chunks of ``BLOCK_L`` timesteps, carrying the per-state running matrix across
chunks, rather than launching ``B·M·S`` tiny programs. λ/η match the sequential
reference to ~1e-6 and gradients to ~3e-7; ~3× faster than torch in training and
~7–8× in inference.

The elementwise pre/post-processing (sufficient statistics, discretized
coefficients, readout) is shared with the torch backend. The initial precision
λ0 is passed to the tiled scan; η0 is folded into r_1 += α_1·η0.

The no-grad forward instead takes the single fused kernel
(:func:`~kla.ops.kernels.triton.fused_kla_scan.fused_kla_forward`), which avoids
those round trips entirely.

That choice is grad-mode driven under ``backend="triton"``; ``triton_fused`` and
``triton_composed`` pin one path each (see ``kernel`` below).

Status: the *composed* (training) path is still glue-heavy — the per-(b,l,m,s)
torch elementwise around the kernels does several HBM passes, leaving it ~6–11×
the fully-fused CUDA kernel; the next step is a fused **backward**. Note
``backend="auto"`` already prefers triton on CUDA (see
:func:`kla.ops.kla_ops._resolve_auto`).
"""

from __future__ import annotations

from typing import Optional

import torch

from kla.ops.kla_ops import (
    EPS,
    P_MIN,
    KLAState,
    _broadcast_ap,
    _sufficient_stats,
    init_state,
)


def _require_kernels():
    try:
        from kla.ops.kernels.triton.fused_kla_scan import fused_kla_forward
        from kla.ops.kernels.triton.tiled_mobius_scan import (
            tiled_affine_scan,
            tiled_mobius_scan,
        )
    except ImportError as e:
        raise NotImplementedError(
            f"The triton KLA backend needs the triton package and a CUDA device ({e}); "
            "use backend='torch' (or 'auto')."
        ) from e
    return tiled_mobius_scan, tiled_affine_scan, fused_kla_forward


def kla_scan_triton(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
    kernel: str = "auto",
):
    """Triton-kernel KLA scan. Same contract as :func:`kla.ops.kla_scan_torch`.

    ``kernel`` pins which of the two triton paths runs: ``"auto"``
    (``backend="triton"``) takes the fused forward when nothing needs an adjoint
    and the tiled scans otherwise, while ``"fused"`` and ``"composed"`` are one
    each. The fused kernel is forward-only, so pinning it raises rather than
    silently falling back when a backward would be needed.
    """
    tiled_mobius_scan, tiled_affine_scan, fused_kla_forward = _require_kernels()
    if not v.is_cuda:
        raise NotImplementedError("The triton KLA backend requires CUDA tensors.")
    if kernel not in ("auto", "fused", "composed"):
        raise ValueError(
            f"Unknown triton kernel {kernel!r}; expected 'auto', 'fused' or 'composed'"
        )

    v = v.float()
    lambda_v = lambda_v.float()
    k = k.float()
    q = q.float()

    B, L, M = v.shape
    S = k.shape[2]
    if initial_state is None:
        initial_state = init_state(B, M, S, device=v.device)
    lam0 = initial_state.lam.float()
    eta0 = initial_state.eta.float()

    if kernel == "fused":
        if torch.is_grad_enabled() and any(
            t.requires_grad for t in (v, lambda_v, k, q, a, p)
        ):
            raise NotImplementedError(
                "The fused triton kernel is forward-only and has no backward; use "
                "backend='triton' (which picks it whenever no adjoint is needed) "
                "or 'triton_composed'."
            )
        if decode_from_prior:
            raise NotImplementedError(
                "The fused triton kernel does not support decode_from_prior; use "
                "backend='triton' or 'triton_composed'."
            )

    # Single fused kernel for the common inference/prefill case (no grad,
    # no prior decode) — no [B,L,M,S] HBM round-trips. The kernel consumes the
    # folded information mean v·Λ^v (pre-folded here; the kernel is unchanged).
    use_fused = kernel == "fused" or (
        kernel == "auto" and not torch.is_grad_enabled() and not decode_from_prior
    )
    if use_fused:
        # p is floored here to match _broadcast_ap on the composed path below —
        # the fused kernel has no internal guard against a non-positive p.
        y, y_var, lam_fin, eta_fin = fused_kla_forward(
            v * lambda_v,
            lambda_v,
            k,
            q,
            a.float(),
            p.float().clamp_min(P_MIN),
            lam0,
            eta0,
        )
        return y, y_var, KLAState(lam=lam_fin, eta=eta_fin)

    phi, r = _sufficient_stats(v, lambda_v, k)
    a_, p_ = _broadcast_ap(a.float(), p.float(), phi)
    a2 = (a_ * a_).clamp_min(EPS)

    # Precision scan λ_t via the tiled, trace-normalized linear-space Möbius
    # recurrence λ_t = (A·λ' + B)/(C·λ' + D), with leaf A=(1+pφ)/a², B=φ, C=p/a²,
    # D=1. Differentiable (forward + exact reverse-scan backward).
    A_lin = ((1.0 + p_ * phi) / a2).expand_as(phi).contiguous()
    C_lin = (p_ / a2).expand_as(phi).contiguous()
    lam = tiled_mobius_scan(A_lin, phi.contiguous(), C_lin, torch.ones_like(phi), lam0)
    var = 1.0 / lam.clamp_min(EPS)

    # Gain α_t needs λ_{t-1}
    lam_prev = torch.cat((lam0.unsqueeze(1), lam[:, :-1]), dim=1)
    denom = (a2 + p_ * lam_prev).clamp_min(EPS)
    alpha = a_ / denom

    # Fold η0 into the first step's input, then scan the affine recurrence (η_{-1}=0).
    r = torch.cat((r[:, :1] + alpha[:, :1] * eta0.unsqueeze(1), r[:, 1:]), dim=1)
    eta = tiled_affine_scan(alpha.contiguous(), r.contiguous())

    mean = eta * var
    if decode_from_prior:
        mean = a_ * mean
        var = a2 * var + p_

    y = torch.einsum("blms,bls->blm", mean, q)
    y_var = torch.einsum("blms,bls->blm", var, q * q)
    return y, y_var, KLAState(lam=lam[:, -1], eta=eta[:, -1])
