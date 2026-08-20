"""MPS (Apple silicon) backend for the KLA scan — two strategies, one device.

Metal shaders compiled at first use through :func:`torch.mps.compile_shader`,
which ships inside torch, so there is no toolchain and no extra dependency.

``backend="mps_recurrent"`` (the default)
    One kernel for the forward and one for the backward, with no ``[B,L,M,S]``
    intermediate in device memory — only two ``[B, M, ceil(L/CHUNK), S]``
    checkpoints the backward replays from. Capped at ``MAX_DSTATE`` states.
``backend="mps_chunk"``
    Time as a parallel axis. The recurrent one puts a thread on each
    ``(batch, channel, state)`` triple, which stops filling the GPU below
    roughly 6k of them — batch-1 prefill on a narrow model. This one splits each
    tile of timesteps across a threadgroup instead, ~2x faster there and ~2.5x
    slower everywhere else, since composing the Möbius maps costs more than
    applying them. Its backward is the recurrent kernel's, run from checkpoints
    the chunk forward writes.

Both are exact in the backward, and both carry the filter state in and out
differentiably, where the equally-fused CUDA ``v2_*`` kernels do neither. That
falls out of the implementation: ``mps_recurrent`` puts one thread on each
``(batch, channel, state)`` triple and walks time serially, so the Möbius map is
applied to a running λ rather than composed with its neighbours, which leaves
the adjoint elementary (``∂λ_t/∂λ_{t-1} = 1/(a²·den²)``). ``mps_chunk`` then
reuses that same adjoint.

Everything runs in float32: Metal has no float64, so ``gradcheck`` still needs
``backend="torch"``.
"""

from __future__ import annotations

from typing import Optional

import torch

from kla.ops.kla_ops import (
    P_MIN,
    KLAState,
    init_state,
)


def _require_kernels():
    try:
        from kla.ops.kernels.mps._shaders import require_mps
        from kla.ops.kernels.mps.recurrent_kla_scan import recurrent_kla_scan
    except ImportError as e:  # pragma: no cover - torch without MPS support
        raise NotImplementedError(
            f"The MPS KLA backend could not be imported ({e}); "
            "use backend='torch' (or 'auto')."
        ) from e
    require_mps()
    return recurrent_kla_scan


def _unsupported(msg: str, which: str = "recurrent") -> "NotImplementedError":
    return NotImplementedError(
        f"The {which} MPS KLA backend {msg}. "
        "Use backend='torch', which has no such limit."
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


def kla_scan_mps_chunk(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
):
    """Time-parallel MPS scan. Same contract as :func:`kla.ops.kla_scan_torch`."""
    from kla.ops.kernels.mps._shaders import MAX_DSTATE, require_mps
    from kla.ops.kernels.mps.chunk_kla_scan import chunk_kla_scan

    require_mps()
    if not v.is_mps:
        raise _unsupported("requires 'mps' tensors", "chunk")
    if a.dim() != 2:
        raise _unsupported("expects a/p of shape [M, S]", "chunk")
    S = k.shape[2]
    if S > MAX_DSTATE:
        raise _unsupported(f"supports d_state <= {MAX_DSTATE} (got {S})", "chunk")

    v, lambda_v, k, q, lam0, eta0 = _prepare(v, lambda_v, k, q, initial_state)

    # As in kla_scan_mps_recurrent: fold v·Λ^v in torch and let autograd split
    # d(v·Λ^v) back into dv and d(Λ^v), and floor p here so the floor's
    # subgradient is torch's too.
    y, y_var, lam_fin, eta_fin = chunk_kla_scan(
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


def kla_scan_mps_merged_chunk(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
):
    """Time-parallel MPS scan, one scan for both recurrences.

    :func:`kla_scan_mps_chunk` with the precision map and the information vector
    folded into a single 3x3 composition, so the tile runs three phases instead
    of six. Same contract as :func:`kla.ops.kla_scan_torch`, same backward.
    """
    from kla.ops.kernels.mps._shaders import MAX_DSTATE, require_mps
    from kla.ops.kernels.mps.merged_chunk_kla_scan import merged_chunk_kla_scan

    require_mps()
    if not v.is_mps:
        raise _unsupported("requires 'mps' tensors", "merged chunk")
    if a.dim() != 2:
        raise _unsupported("expects a/p of shape [M, S]", "merged chunk")
    S = k.shape[2]
    if S > MAX_DSTATE:
        raise _unsupported(
            f"supports d_state <= {MAX_DSTATE} (got {S})", "merged chunk"
        )

    v, lambda_v, k, q, lam0, eta0 = _prepare(v, lambda_v, k, q, initial_state)

    # As in every other cell: fold v·Λ^v in torch so autograd splits d(v·Λ^v)
    # back into dv and d(Λ^v), and floor p here so the floor's subgradient is
    # torch's too.
    y, y_var, lam_fin, eta_fin = merged_chunk_kla_scan(
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


def kla_scan_mps_pscan(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
):
    """Reduce-then-scan MPS scan. Same contract as :func:`kla.ops.kla_scan_torch`."""
    from kla.ops.kernels.mps._shaders import MAX_DSTATE, require_mps
    from kla.ops.kernels.mps.pscan_kla_scan import pscan_kla_scan

    require_mps()
    if not v.is_mps:
        raise _unsupported("requires 'mps' tensors", "pscan")
    if a.dim() != 2:
        raise _unsupported("expects a/p of shape [M, S]", "pscan")
    S = k.shape[2]
    if S > MAX_DSTATE:
        raise _unsupported(f"supports d_state <= {MAX_DSTATE} (got {S})", "pscan")

    v, lambda_v, k, q, lam0, eta0 = _prepare(v, lambda_v, k, q, initial_state)

    # As in the other two cells: fold v·Λ^v in torch so autograd splits
    # d(v·Λ^v) back into dv and d(Λ^v), and floor p here so the floor's
    # subgradient is torch's too.
    y, y_var, lam_fin, eta_fin = pscan_kla_scan(
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


def kla_scan_mps_merged_pscan(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
):
    """Reduce-then-scan MPS scan, one round for both recurrences.

    :func:`kla_scan_mps_pscan` with the second reduce-scan-apply round removed:
    the merged 3x3 leaf carries η, so there is no set of affine leaves waiting
    on λ. Same contract as :func:`kla.ops.kla_scan_torch`, same backward.
    """
    from kla.ops.kernels.mps._shaders import MAX_DSTATE, require_mps
    from kla.ops.kernels.mps.merged_pscan_kla_scan import merged_pscan_kla_scan

    require_mps()
    if not v.is_mps:
        raise _unsupported("requires 'mps' tensors", "merged pscan")
    if a.dim() != 2:
        raise _unsupported("expects a/p of shape [M, S]", "merged pscan")
    S = k.shape[2]
    if S > MAX_DSTATE:
        raise _unsupported(
            f"supports d_state <= {MAX_DSTATE} (got {S})", "merged pscan"
        )

    v, lambda_v, k, q, lam0, eta0 = _prepare(v, lambda_v, k, q, initial_state)

    # As in every other cell: fold v·Λ^v in torch so autograd splits d(v·Λ^v)
    # back into dv and d(Λ^v), and floor p here so the floor's subgradient is
    # torch's too.
    y, y_var, lam_fin, eta_fin = merged_pscan_kla_scan(
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


def kla_scan_mps_recurrent(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
):
    """Serial-time MPS scan. Same contract as :func:`kla.ops.kla_scan_torch`."""
    from kla.ops.kernels.mps._shaders import MAX_DSTATE

    recurrent_kla_scan = _require_kernels()
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
    y, y_var, lam_fin, eta_fin = recurrent_kla_scan(
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
