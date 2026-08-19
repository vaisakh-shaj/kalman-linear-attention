"""MPS (Apple silicon) backend for the KLA scan — two strategies, one device.

Metal shaders compiled at first use through :func:`torch.mps.compile_shader`,
which ships inside torch, so there is no toolchain and no extra dependency.

``backend="mps"`` / ``"mps_fused"``
    One kernel for the forward and one for the backward, with no ``[B,L,M,S]``
    intermediate in device memory — only two ``[B, M, ceil(L/CHUNK), S]``
    checkpoints the backward replays from. Capped at ``MAX_DSTATE`` states.
``backend="mps_tiled"``
    Forward only, with time as a parallel axis. The other two put one thread on
    each ``(batch, channel, state)`` triple, which stops filling the GPU below
    roughly 6k of them — batch-1 prefill on a narrow model. This one splits each
    tile of timesteps across a threadgroup instead, ~2x faster there and ~2.5x
    slower everywhere else, since composing the Möbius maps costs more than
    applying them.
``backend="mps_composed"``
    The two recurrences as standalone scan kernels with torch elementwise glue
    around them, the shape the triton backend has. The glue costs a stack of
    ``[B,L,M,S]`` round trips, so training runs ~20× the fused kernel; it has no
    ``d_state`` cap.

Both are exact in the backward, and both carry the filter state in and out
differentiably. That the *fused* one manages this is the interesting part, since
the equally-fused CUDA kernel does neither: these kernels put one thread on each
``(batch, channel, state)`` triple and walk time serially, so the Möbius map is
applied to a running λ rather than composed with its neighbours, which leaves
the adjoint elementary (``∂λ_t/∂λ_{t-1} = 1/(a²·den²)``).

Everything runs in float32: Metal has no float64, so ``gradcheck`` still needs
``backend="torch"``.
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
        from kla.ops.kernels.mps._shaders import require_mps
        from kla.ops.kernels.mps.fused_kla_scan import fused_kla_scan
        from kla.ops.kernels.mps.lane_mobius_scan import affine_scan, mobius_scan
    except ImportError as e:  # pragma: no cover - torch without MPS support
        raise NotImplementedError(
            f"The MPS KLA backend could not be imported ({e}); "
            "use backend='torch' (or 'auto')."
        ) from e
    require_mps()
    return mobius_scan, affine_scan, fused_kla_scan


def _unsupported(msg: str, which: str = "fused") -> "NotImplementedError":
    return NotImplementedError(
        f"The {which} MPS KLA backend {msg}. "
        "Use backend='mps_composed', which has no such limit, or 'torch'."
    )


def _prepare(v, lambda_v, k, q, initial_state):
    """Cast to the kernels' float32 and materialize the initial state."""
    v = v.float().contiguous()
    lambda_v = lambda_v.float().contiguous()
    k = k.float().contiguous()
    q = q.float().contiguous()
    B, _, M = v.shape
    S = k.shape[2]
    if initial_state is None:
        initial_state = init_state(B, M, S, device=v.device)
    return v, lambda_v, k, q, initial_state.lam.float(), initial_state.eta.float()


def kla_scan_mps_composed(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
):
    """Composed MPS scan. Same contract as :func:`kla.ops.kla_scan_torch`."""
    mobius_scan, affine_scan, _ = _require_kernels()
    if not v.is_mps:
        raise NotImplementedError("The MPS KLA backend requires 'mps' tensors.")

    v, lambda_v, k, q, lam0, eta0 = _prepare(v, lambda_v, k, q, initial_state)

    phi, r = _sufficient_stats(v, lambda_v, k)
    a_, p_ = _broadcast_ap(a.float(), p.float(), phi)
    a2 = (a_ * a_).clamp_min(EPS)

    # Precision scan λ_t = (A·λ' + B)/(C·λ' + D), leaf A=(1+pφ)/a², B=φ, C=p/a²,
    # D=1 — the same leaves the triton and CUDA kernels build, applied per step
    # rather than composed. Differentiable (kernel forward + exact reverse scan).
    A_lin = ((1.0 + p_ * phi) / a2).expand_as(phi).contiguous()
    C_lin = (p_ / a2).expand_as(phi).contiguous()
    lam = mobius_scan(A_lin, phi.contiguous(), C_lin, torch.ones_like(phi), lam0)
    var = 1.0 / lam.clamp_min(EPS)

    # Gain α_t needs λ_{t-1}.
    lam_prev = torch.cat((lam0.unsqueeze(1), lam[:, :-1]), dim=1)
    denom = (a2 + p_ * lam_prev).clamp_min(EPS)
    alpha = a_ / denom

    # Fold η0 into the first step's input, then scan the affine recurrence.
    r = torch.cat((r[:, :1] + alpha[:, :1] * eta0.unsqueeze(1), r[:, 1:]), dim=1)
    eta = affine_scan(alpha.contiguous(), r.contiguous())

    mean = eta * var
    if decode_from_prior:
        mean = a_ * mean
        var = a2 * var + p_

    y = torch.einsum("blms,bls->blm", mean, q)
    y_var = torch.einsum("blms,bls->blm", var, q * q)
    return y, y_var, KLAState(lam=lam[:, -1], eta=eta[:, -1])


def kla_scan_mps_tiled(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
):
    """Tiled MPS forward. Same contract as :func:`kla.ops.kla_scan_torch`.

    Forward-only, like ``triton_fused``: it raises rather than silently falling
    back when an adjoint would be needed.
    """
    from kla.ops.kernels.mps._shaders import MAX_DSTATE, require_mps
    from kla.ops.kernels.mps.tiled_kla_scan import tiled_forward

    require_mps()
    if not v.is_mps:
        raise _unsupported("requires 'mps' tensors", "tiled")
    if a.dim() != 2:
        raise _unsupported("expects a/p of shape [M, S]", "tiled")
    S = k.shape[2]
    if S > MAX_DSTATE:
        raise _unsupported(f"supports d_state <= {MAX_DSTATE} (got {S})", "tiled")
    if torch.is_grad_enabled() and any(
        t.requires_grad for t in (v, lambda_v, k, q, a, p)
    ):
        raise NotImplementedError(
            "The tiled MPS kernel is forward-only and has no backward; use "
            "backend='mps' for training."
        )

    v, lambda_v, k, q, lam0, eta0 = _prepare(v, lambda_v, k, q, initial_state)
    y, y_var, lam_fin, eta_fin = tiled_forward(
        (v * lambda_v).contiguous(),
        lambda_v,
        k,
        q,
        a.float().contiguous(),
        p.float().clamp_min(P_MIN).contiguous(),
        lam0.contiguous(),
        eta0.contiguous(),
        decode_from_prior,
    )
    return y, y_var, KLAState(lam=lam_fin, eta=eta_fin)


def kla_scan_mps_fused(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
):
    """Fully fused MPS scan. Same contract as :func:`kla.ops.kla_scan_torch`."""
    from kla.ops.kernels.mps._shaders import MAX_DSTATE

    _, _, fused_kla_scan = _require_kernels()
    if not v.is_mps:
        raise _unsupported("requires 'mps' tensors")
    if a.dim() != 2:
        raise _unsupported("expects a/p of shape [M, S]")
    S = k.shape[2]
    if S > MAX_DSTATE:
        raise _unsupported(f"supports d_state <= {MAX_DSTATE} (got {S})")

    v, lambda_v, k, q, lam0, eta0 = _prepare(v, lambda_v, k, q, initial_state)

    # The kernel consumes the folded information mean v·Λ^v, so fold it in torch
    # and let autograd split d(v·Λ^v) back into dv and d(Λ^v). Flooring p here
    # rather than in the kernel does the same for the floor's subgradient.
    y, y_var, lam_fin, eta_fin = fused_kla_scan(
        (v * lambda_v).contiguous(),
        lambda_v,
        k,
        q,
        a.float().contiguous(),
        p.float().clamp_min(P_MIN).contiguous(),
        lam0.contiguous(),
        eta0.contiguous(),
        decode_from_prior,
    )
    return y, y_var, KLAState(lam=lam_fin, eta=eta_fin)
