"""Launch constants for the triton cells, and the one rule that ties them.

``block_l`` and ``num_warps`` are per-call arguments on every triton kernel here,
but they are not independent: a program's tile is ``[BLOCK_L, BLOCK_S]`` with
``BLOCK_S = next_pow2(d_state)``, and the warps have to cover it. The values
below come from a sweep over d_state and block_l; see docs/benchmarks/cuda.md
for the tables, and re-run it before changing them.
"""

from __future__ import annotations

WARP_TILE = 512
"""Tile elements per warp. Below this the extra warps have nothing to do; above
it the reverse scans and ``tl.associative_scan`` spill, and the cliff is steep."""

CHUNK_BLOCK_L = 16
"""``chunk`` / ``merged_chunk``, and the checkpoint stride they hand the shared
backward.

One number does two jobs, the forward's tile width *and* the stride the backward
replays from, because ``tl.associative_scan`` runs over the tile axis itself.
That forces a compromise: the forward is insensitive over a range of widths, the
backward is not, since its replay cost is linear in the stride, and a shorter
stride trades peak memory (more, smaller checkpoints) for backward time. The
CUDA kernels split the two (``KLA_ITEMS`` and ``KLA_CHUNK``) and get an optimum
for each."""

RECURRENT_BLOCK_L = 16
"""``recurrent``. Its forward holds one ``[BLOCK_S]`` vector and walks time
serially, so this is *purely* the shared backward's checkpoint stride -- the
forward uses it for nothing else."""


def warps_for(block_l: int, block_s: int) -> int:
    """Warps for a ``[block_l, block_s]`` tile, clamped to [1, 8].

    Deriving this rather than fixing it matters most in the backward, which does
    not choose its own ``block_l``: it must use the stride its forward wrote
    checkpoints at. A fixed count is a trap in both directions -- too few warps
    for a tall tile spills the reverse scans, too many over a short one is
    oversubscribed -- and both cost far more than the ratio does to compute. The
    forwards derive it for the same reason: nothing stops a caller passing a tile
    width the shipped default never sees.
    """
    warps = 1
    while warps < 8 and block_l * block_s > WARP_TILE * warps:
        warps *= 2
    return warps
