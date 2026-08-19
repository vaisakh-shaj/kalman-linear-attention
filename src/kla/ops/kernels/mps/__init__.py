"""Metal (MPS) kernels for the KLA scan.

Three implementations of the same scan:

``fused_kla_scan``
    The default. Everything in one kernel with no ``[B, L, M, S]`` intermediate,
    plus a hand-written fused backward that is the exact adjoint.

``tiled_kla_scan``
    Forward only, with time as a parallel axis, for the batch-1 prefill shapes
    the other two leave the GPU short of threads on.

``lane_mobius_scan``
    Standalone Möbius and affine scan primitives over ``[B, L, M, S]``
    coefficients, with torch elementwise work around them — the shape the triton
    backend has. No ``d_state`` ceiling.

The ``.metal`` sources sit alongside; :mod:`._shaders` compiles them through
:func:`torch.mps.compile_shader`, so there is no build step and no toolchain to
install.
"""

from kla.ops.kernels.mps._shaders import MAX_DSTATE, is_available, launch_geometry

__all__ = ["MAX_DSTATE", "is_available", "launch_geometry"]
