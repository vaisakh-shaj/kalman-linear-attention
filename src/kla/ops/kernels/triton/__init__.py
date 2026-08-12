"""Triton kernels for the KLA scan.

- ``tiled_mobius_scan`` — the production triton scans used by
  :mod:`kla.ops.triton_backend`: a ``[BLOCK_L, S]``-tiled, ``(B, M)``-grid,
  linear-space *trace-normalized* Möbius scan (precision λ) and affine scan
  (information vector η), each with an exact handwritten backward.

Do not import this package eagerly from portable code paths — the modules
import ``triton`` at module level.
"""
