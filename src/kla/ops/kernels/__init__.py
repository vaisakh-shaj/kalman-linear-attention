"""Low-level kernels for the KLA scan.

- ``triton/`` — triton kernels (require a CUDA device + the triton package;
  imported lazily by the backends so the rest of the library stays portable).
- ``cuda/`` — (future) C++/CUDA sources, JIT-compiled via
  ``torch.utils.cpp_extension`` by :mod:`kla.ops.cuda_backend`.
"""
