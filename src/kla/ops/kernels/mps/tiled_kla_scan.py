"""Tiled KLA forward on Metal — time as a parallel axis.

The counterpart of :mod:`kla.ops.kernels.mps.fused_kla_scan` for the shapes that
one cannot fill. A threadgroup owns a ``(batch, channel)`` pair and splits each
tile of timesteps across its threads, so the grid is ``ROWS`` times wider than
the lane-per-state kernels' ``B*M*S``. That matters at batch-1 prefill and
nowhere else; see the header of ``tiled_kla_scan.metal`` for the phases and for
why the extra parallelism costs about 4x the arithmetic.

Forward only — there is no adjoint here, and none is wanted: the fused
kernels' per-step backward is exact and cheaper.
"""

from __future__ import annotations

import torch

from kla.ops.kernels.mps._shaders import (
    DEFAULT_ITEMS,
    check_inputs,
    tile_geometry,
    tiled_library,
)


def _grid(B: int, M: int, S: int):
    """``(threads, group_size)`` — one threadgroup per ``(batch, channel)``."""
    block_s, rows = tile_geometry(S)
    return (block_s, rows * M, B), (block_s, rows, 1)


def tiled_forward(
    msi: torch.Tensor,  # v·Λ^v  [B, L, M]
    si: torch.Tensor,  # Λ^v    [B, L, M]
    k: torch.Tensor,  # key    [B, L, S]
    q: torch.Tensor,  # query  [B, L, S]
    a: torch.Tensor,  # decay         [M, S]
    p: torch.Tensor,  # process noise [M, S]
    lam0: torch.Tensor,  # [B, M, S]
    eta0: torch.Tensor,  # [B, M, S]
    prior: bool = False,
    items: int = DEFAULT_ITEMS,
):
    """Tiled forward. Returns ``(y, y_var, lam_fin, eta_fin)``."""
    check_inputs(msi, si, k, q, a, p, lam0, eta0)
    B, L, M = msi.shape
    S = k.shape[2]
    lib = tiled_library(S, items)

    dev = msi.device
    y = torch.empty(B, L, M, device=dev, dtype=torch.float32)
    yvar = torch.empty(B, L, M, device=dev, dtype=torch.float32)
    lam_fin = torch.empty(B, M, S, device=dev, dtype=torch.float32)
    eta_fin = torch.empty(B, M, S, device=dev, dtype=torch.float32)

    threads, group = _grid(B, M, S)
    lib.kla_tiled_fwd(
        y,
        yvar,
        lam_fin,
        eta_fin,
        msi,
        si,
        k,
        q,
        a,
        p,
        lam0,
        eta0,
        L,
        M,
        S,
        int(prior),
        threads=threads,
        group_size=group,
    )
    return y, yvar, lam_fin, eta_fin
