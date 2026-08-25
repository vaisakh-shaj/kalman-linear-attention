"""``mps_fused_chunk``, the scan on Metal with time as a parallel axis.

The counterpart of :mod:`kla.ops.kernels.mps.recurrent_kla_scan` for the shapes
that one cannot fill. A threadgroup owns a ``(batch, channel)`` pair and splits
each tile of timesteps across its threads, so the grid is ``ROWS`` times wider
than the lane-per-state kernel's ``B*M*S``. That matters at batch-1 prefill and
nowhere else; see the header of ``chunk_kla_scan.metal`` for the phases and for
why the extra parallelism costs about 4x the arithmetic.

The backward is :func:`~kla.ops.kernels.mps.kla_scan_bwd.scan_backward`, shared
with ``mps_fused_recurrent``. An adjoint does not have to mirror its forward: it
recovers λ from the checkpoints this forward writes, same layout, same stride,
then walks a *scalar* reverse recurrence down the serial state lanes, which is
the same work whichever forward got there. So the gradients are exact, at the
tight tolerance, with no composed-map Jacobian anywhere.
"""

from __future__ import annotations

from kla.ops.kernels.mps._host import Cell, make_scan
from kla.ops.kernels.mps._shaders import chunk_library, tile_geometry

CELL = Cell(
    library=chunk_library,
    entry="kla_chunk_fwd",
    geometry=tile_geometry,
    tiled=True,
)

chunk_kla_scan = make_scan(
    CELL,
    "chunk_kla_scan",
    """Differentiable chunk KLA scan → ``(y, y_var, lam_fin, eta_fin)``.""",
)
