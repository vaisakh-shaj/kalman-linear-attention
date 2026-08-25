"""Triton kernels for the KLA scan.

Three fused forwards, ``recurrent_kla_scan``, ``chunk_kla_scan`` and
``merged_chunk_kla_scan``, one kernel each with no ``[B, L, M, S]``
intermediate, plus ``kla_scan_bwd``, the one exact backward all three share.
See ``docs/implementations.md`` for what the names mean and ``_tuning`` for the
launch constants they take.

Do not import this package eagerly from portable code paths, the modules
import ``triton`` at module level.
"""
