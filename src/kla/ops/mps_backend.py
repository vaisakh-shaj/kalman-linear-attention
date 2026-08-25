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
    ``mps_fused_chunk`` with the precision and information recurrences folded into
    a single 3x3 map, so the threadgroup scans once instead of twice and the
    per-thread arrays that carried one phase's output to the next disappear. The
    faster of the two chunk cells here.

The two strategies cross over at a lane count (``B x d_inner x d_state``) below
which the recurrent grid cannot fill the GPU. Every realistic config sits above
it, which is why ``backend="mps"`` resolves to ``mps_fused_recurrent``. **That
crossover is a property of the device**: the same serial chain is latency-bound
on a discrete GPU, where it lands in a different place entirely. See
``docs/benchmarks/mps.md``.

All three are exact in the backward and carry the filter state in and out
differentiably, which falls out of the recurrent kernel: applying the map leaves
the adjoint elementary (``∂λ_t/∂λ_{t-1} = 1/(a²·den²)``), and the chunk cells
reuse it, replaying from the ``[B, M, ceil(L/CHUNK), S]`` checkpoints their
forwards write.

Everything runs in float32: Metal has no float64, so ``gradcheck`` still needs
``backend="torch"``.
"""

from __future__ import annotations

import torch

from kla.ops.kla_ops import (
    P_MIN,
    KLAState,
    init_state,
)


def _require_kernels():
    """Import the Metal cells, or explain why this machine cannot run them."""
    try:
        from kla.ops.kernels.mps._shaders import require_mps
        from kla.ops.kernels.mps.chunk_kla_scan import chunk_kla_scan
        from kla.ops.kernels.mps.merged_chunk_kla_scan import merged_chunk_kla_scan
        from kla.ops.kernels.mps.recurrent_kla_scan import recurrent_kla_scan
    except ImportError as e:  # pragma: no cover - torch without MPS support
        raise NotImplementedError(
            f"The MPS KLA backend could not be imported ({e}); "
            "use backend='torch' (or 'auto')."
        ) from e
    require_mps()
    return {
        "recurrent": recurrent_kla_scan,
        "chunk": chunk_kla_scan,
        "merged chunk": merged_chunk_kla_scan,
    }


def _unsupported(msg: str, which: str) -> NotImplementedError:
    return NotImplementedError(
        f"The {which} MPS KLA backend {msg}. "
        "Use backend='torch', which has no such limit."
    )


def _mps_scan(which, v, lambda_v, k, q, a, p, initial_state, decode_from_prior):
    """Check, cast and run one Metal cell. Only ``which`` differs between them."""
    from kla.ops.kernels.mps._shaders import MAX_DSTATE

    kernel = _require_kernels()[which]
    if not v.is_mps:
        raise _unsupported("requires 'mps' tensors", which)
    if a.dim() != 2:
        raise _unsupported("expects a/p of shape [M, S]", which)
    S = k.shape[2]
    if S > MAX_DSTATE:
        raise _unsupported(f"supports d_state <= {MAX_DSTATE} (got {S})", which)

    v = v.float().contiguous()
    lambda_v = lambda_v.float().contiguous()
    B, _, M = v.shape
    if initial_state is None:
        initial_state = init_state(B, M, S, device=v.device)

    # The kernels consume the folded information mean v·Λ^v, so fold it in torch
    # and let autograd split d(v·Λ^v) back into dv and d(Λ^v). Flooring p here
    # rather than in the kernel does the same for the floor's subgradient.
    y, y_var, lam_fin, eta_fin = kernel(
        (v * lambda_v).contiguous(),
        lambda_v,
        k.float().contiguous(),
        q.float().contiguous(),
        a.float().contiguous(),
        p.float().clamp_min(P_MIN).contiguous(),
        initial_state.lam.float().contiguous(),
        initial_state.eta.float().contiguous(),
        decode_from_prior,
    )
    return y, y_var, KLAState(lam=lam_fin, eta=eta_fin)


def _cell(which: str, doc: str):
    """One backend entry point. Same contract as :func:`kla.ops.kla_scan_torch`."""

    def run(
        v: torch.Tensor,
        lambda_v: torch.Tensor,
        k: torch.Tensor,
        q: torch.Tensor,
        a: torch.Tensor,
        p: torch.Tensor,
        initial_state: KLAState | None = None,
        decode_from_prior: bool = False,
    ):
        return _mps_scan(
            which, v, lambda_v, k, q, a, p, initial_state, decode_from_prior
        )

    run.__name__ = f"kla_scan_mps_{which.replace(' ', '_')}"
    run.__qualname__ = run.__name__
    run.__doc__ = doc
    return run


kla_scan_mps_recurrent = _cell(
    "recurrent",
    "Serial-time MPS scan. Same contract as :func:`kla.ops.kla_scan_torch`.",
)
kla_scan_mps_chunk = _cell(
    "chunk",
    "Time-parallel MPS scan. Same contract as :func:`kla.ops.kla_scan_torch`.",
)
kla_scan_mps_merged_chunk = _cell(
    "merged chunk",
    "Time-parallel MPS scan, one scan for both recurrences: "
    ":func:`kla_scan_mps_chunk` with the precision map and the information "
    "vector folded into a single 3x3 composition, so the tile runs three phases "
    "instead of six. Same contract, same backward.",
)
