"""Triton backend for the KLA scan, three cells, one backward.

``triton_fused_recurrent``, ``triton_fused_chunk`` and ``triton_merged_chunk`` put the
sufficient statistics, both recurrences and the read-out in one kernel, so no
``[B, L, M, S]`` intermediate is ever written. All three write the same
``[B, M, NCK, S]`` checkpoints at the same stride, and
:mod:`kla.ops.kernels.triton.kla_scan_bwd` replays from them, one backward,
whichever forward ran.

They differ only in how time is walked and how many scans it takes.
``recurrent`` applies the map along a serial axis; ``chunk`` tiles time,
composes within a tile and carries across tiles; ``merged_chunk`` is ``chunk``
with the precision and information recurrences folded into a single 3x3 map in
homogeneous coordinates, so η is never scanned separately. See
``docs/implementations.md``.

The precision recurrence λ_t = (Aλ' + B)/(Cλ' + D) is a *linear-space,
trace-normalized* Möbius scan throughout, the same formulation the CUDA and
Metal kernels use: composing the 2×2 maps is a plain matmul normalized by the
trace, which keeps entries O(1) without log-space since λ is scale-invariant.
The information vector η_t = α_t·η' + r_t is a plain affine scan. Both adjoints
are scalar, which is what makes every cell here exact.

``backend="triton"`` resolves to ``triton_merged_chunk``, which beat the
two-scan cell at every shape measured on an L40S and never lost, and
``backend="auto"`` already prefers triton on CUDA. See
``docs/benchmarks/cuda.md``.
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
        from kla.ops.kernels.triton.chunk_kla_scan import chunk_kla_scan
        from kla.ops.kernels.triton.merged_chunk_kla_scan import (
            merged_chunk_kla_scan,
        )
        from kla.ops.kernels.triton.recurrent_kla_scan import recurrent_kla_scan
    except ImportError as e:
        raise NotImplementedError(
            f"The triton KLA backend needs the triton package and a CUDA device ({e}); "
            "use backend='torch' (or 'auto')."
        ) from e
    # merged: fused *and* one scan instead of two. `recurrent` has no merged
    # cell and never will -- it applies the map rather than composing it, so it
    # already does both recurrences in one pass.
    return {
        "recurrent": recurrent_kla_scan,
        "chunk": chunk_kla_scan,
        "merged_chunk": merged_chunk_kla_scan,
    }


def kla_scan_triton(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
    kernel: str = "merged_chunk",
):
    """Triton-kernel KLA scan. Same contract as :func:`kla.ops.kla_scan_torch`.

    ``kernel`` picks the implementation: ``"recurrent"`` walks time serially,
    ``"chunk"`` tiles it, and ``"merged_chunk"`` is ``"chunk"`` with the
    precision and information scans folded into one 3x3 map. All three share one
    backward. Each runs exactly what it names, nothing is selected by grad
    mode, so the kernels a run used are a function of the config and the machine
    and nothing else.
    """
    kernels = _require_kernels()
    if not v.is_cuda:
        raise NotImplementedError("The triton KLA backend requires CUDA tensors.")
    if kernel not in kernels:
        raise ValueError(
            f"Unknown triton kernel {kernel!r}; expected one of {', '.join(kernels)}"
        )

    B, _, M = v.shape
    S = k.shape[2]
    if initial_state is None:
        initial_state = init_state(B, M, S, device=v.device)

    # No [B,L,M,S] HBM round-trips. The kernels consume the folded information
    # mean v·Λ^v (pre-folded here, so autograd splits d(v·Λ^v) back into dv and
    # d(Λ^v)), and p is floored here because they have no internal guard against
    # a non-positive p, flooring in torch also hands torch the floor's
    # subgradient.
    v = v.float()
    lambda_v = lambda_v.float()
    y, y_var, lam_fin, eta_fin = kernels[kernel](
        v * lambda_v,
        lambda_v,
        k.float(),
        q.float(),
        a.float(),
        p.float().clamp_min(P_MIN),
        initial_state.lam.float(),
        initial_state.eta.float(),
        decode_from_prior,
    )
    return y, y_var, KLAState(lam=lam_fin, eta=eta_fin)
