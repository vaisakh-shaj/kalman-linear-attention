"""``mps_fused_recurrent``, the whole scan in one kernel, time serial.

Sufficient statistics, both recurrences and the read-out in one kernel, so no
``[B, L, M, S]`` intermediate is ever written. The Möbius map is *applied* to a
running λ rather than composed, which is what leaves the adjoint elementary; the
backward is :mod:`kla.ops.kernels.mps.kla_scan_bwd`, shared with ``mps_fused_chunk``.

``d_state`` is capped at :data:`~kla.ops.kernels.mps._shaders.MAX_DSTATE`. The
replay scheme, reduction layout and atomics are described in the headers of
``recurrent_kla_scan.metal`` and ``kla_scan_bwd.metal``.
"""

from __future__ import annotations

from kla.ops.kernels.mps._host import Cell, make_scan
from kla.ops.kernels.mps._shaders import launch_geometry, recurrent_library

CELL = Cell(
    library=recurrent_library,
    entry="kla_recurrent_fwd",
    geometry=launch_geometry,
    tiled=False,
)

recurrent_kla_scan = make_scan(
    CELL,
    "recurrent_kla_scan",
    """Differentiable recurrent KLA scan → ``(y, y_var, lam_fin, eta_fin)``.""",
)
