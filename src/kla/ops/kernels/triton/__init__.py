"""Triton kernels for the KLA scan.

Two families, matching the two halves of the naming scheme (see
``docs/implementations.md``):

- ``recurrent_kla_scan``, ``chunk_kla_scan``, ``pscan_kla_scan`` — the fused
  cells, one kernel each with no ``[B, L, M, S]`` intermediate, plus
  ``kla_scan_bwd``, the one exact backward all three share.
- ``unfused_kla_scan`` — the standalone λ and η scans behind the unfused cells,
  three schedules each over already-built leaf coefficients, with torch glue
  around them and one exact backward again.

Do not import this package eagerly from portable code paths — the modules
import ``triton`` at module level.
"""
