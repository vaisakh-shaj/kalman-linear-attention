"""Metal (MPS) kernels for the KLA scan.

Three forwards, one per cell, and one backward they share (see
``docs/implementations.md`` for what the names mean):

``recurrent_kla_scan``
    ``mps_fused_recurrent``. One thread per ``(b, m, s)``, time serial, the Möbius map
    applied rather than composed. No ``[B, L, M, S]`` intermediate.

``chunk_kla_scan``
    ``mps_fused_chunk``. Time as a parallel axis, for the prefill shapes the
    other one leaves the GPU short of threads on.

``merged_chunk_kla_scan``
    ``mps_merged_chunk``. ``chunk`` with both recurrences folded into one 3x3
    map in homogeneous coordinates: one threadgroup scan instead of two, one
    broadcast instead of two, and fewer registers per thread.

``kla_scan_bwd``
    The exact adjoint, shared. All three forwards write the same
    ``[B, M, NCK, S]`` checkpoints, and the reverse walk is over the serial
    state lanes whichever forward produced them.

The ``.metal`` sources sit alongside; :mod:`._shaders` compiles them through
:func:`torch.mps.compile_shader`, so there is no build step and no toolchain to
install.
"""

from kla.ops.kernels.mps._shaders import MAX_DSTATE, is_available, launch_geometry

__all__ = ["MAX_DSTATE", "is_available", "launch_geometry"]
