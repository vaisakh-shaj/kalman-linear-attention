"""``mps_merged_chunk``, ``mps_fused_chunk``'s six phases in three, one scan not two.

Identical in shape to :mod:`kla.ops.kernels.mps.chunk_kla_scan`: same
threadgroup per ``(batch, channel)``, same tiles of ``ROWS * ITEMS`` timesteps,
same grid, same checkpoints, same backward. The difference is entirely inside
the tile. That kernel composes a 2x2 Möbius map for λ, walks it to produce λ,
then builds the affine leaves ``(α, r)`` that walk unlocked and scans *those*,
because α_t reads λ_{t-1}, so the second set of leaves cannot exist any earlier.
This one composes the 3x3 map of ``kla_merged.metal``, which carries η in the
same homogeneous coordinates, so its leaf depends on ``(φ, r, a, p)`` alone and
one scan does both.

What that buys: one threadgroup scan instead of two (each is ``log2(ROWS)``
Hillis-Steele rounds with two barriers apiece), one broadcast instead of two,
and the disappearance of the per-thread arrays that existed only to carry one
phase's output to another -- registers back, on the kernel whose entire reason
to exist is occupancy.

The backward is :func:`~kla.ops.kernels.mps.kla_scan_bwd.scan_backward`,
unchanged and shared with every other MPS cell: it replays a scalar recurrence
from ``[B, M, NCK, S]`` checkpoints and never sees a composed map, so merging
the forward is invisible to it and the exact-gradient contract is unchanged.
"""

from __future__ import annotations

from kla.ops.kernels.mps._host import Cell, make_scan
from kla.ops.kernels.mps._shaders import merged_chunk_library, tile_geometry

CELL = Cell(
    library=merged_chunk_library,
    entry="kla_merged_chunk_fwd",
    geometry=tile_geometry,
    tiled=True,
)

merged_chunk_kla_scan = make_scan(
    CELL,
    "merged_chunk_kla_scan",
    """Differentiable merged chunk KLA scan → ``(y, y_var, lam_fin, eta_fin)``.""",
)
