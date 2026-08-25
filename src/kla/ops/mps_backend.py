"""MPS (Apple silicon) backend for the KLA scan, three cells, one backward.

Metal shaders compiled at first use through :func:`torch.mps.compile_shader`,
which ships inside torch, so there is no toolchain and no extra dependency.

``kla_scan_mps_recurrent`` (``mps_fused_recurrent``, the default)
    One thread per ``(batch, channel, state)`` triple, time as the serial axis,
    the Möbius map *applied* to a running λ rather than composed with its
    neighbours. Nothing but the state itself is carried, and no ``[B,L,M,S]``
    intermediate reaches device memory.
``kla_scan_mps_chunk`` (``mps_fused_chunk``)
    Time as a parallel axis instead: each tile of timesteps is split across a
    threadgroup, which fills the GPU at the low lane counts the recurrent grid
    leaves it short at, batch-1 prefill on a narrow model. Composing the maps
    costs more than applying them, so it loses everywhere else.
``kla_scan_mps_merged_chunk`` (``mps_merged_chunk``)
    ``mps_fused_chunk`` with the precision and information recurrences folded into a
    single 3x3 map, so the threadgroup scans once instead of twice and the
    ``var_h``/``alpha_h``/``r_h`` per-thread arrays disappear. 15-30% faster than
    ``mps_fused_chunk`` at every shape measured; see ``docs/benchmarks/mps.md``.

The crossover is around 8k lanes (``B x d_inner x d_state``) on an M5 Pro, and
every realistic config sits above it, which is why ``backend="mps"`` resolves to
``mps_fused_recurrent``. **Do not carry that number to another device**, the same
serial chain is latency-bound on an L40S and the crossover moves by two orders
of magnitude.

All three are exact in the backward and carry the filter state in and out
differentiably. That falls out of the recurrent kernel: applying the map leaves
the adjoint elementary (``∂λ_t/∂λ_{t-1} = 1/(a²·den²)``), and the two chunk
cells reuse it, replaying from ``[B, M, ceil(L/CHUNK), S]`` checkpoints their
forwards write.

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
