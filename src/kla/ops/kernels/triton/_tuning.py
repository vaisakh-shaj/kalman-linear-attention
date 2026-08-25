"""Launch constants for the triton cells, and the one rule that ties them.

``block_l`` and ``num_warps`` are per-call arguments on every triton kernel
here, but they are not independent: a program's tile is ``[BLOCK_L, BLOCK_S]``
with ``BLOCK_S = next_pow2(d_state)``, and the warps have to cover it. Tuned on
an L40S over d_state 8/16/32/64 and block_l 16..256; see docs/benchmarks/cuda.md
for the tables.

The shipped defaults before this were ``block_l=64, num_warps=4`` everywhere,
which is 1.7-2.7x off, four warps over a [64, 16] tile is oversubscribed, and
the ratio below is what the sweep landed on.
"""

from __future__ import annotations

WARP_TILE = 512
"""Tile elements per warp. Below this the extra warps have nothing to do; above
it the reverse scans and ``tl.associative_scan`` spill, and the cliff is steep,
a [256, 16] tile at one warp measured 15x its four-warp time."""

CHUNK_BLOCK_L = 16
"""``chunk`` / ``merged_chunk``, and the checkpoint stride they hand the shared
backward. The forward is flat between 16 and 32; the backward is not, because
its replay cost is linear in the stride. 16 won training by 20% for +14% peak
memory (more, smaller checkpoints).

One number does two jobs here, the forward's tile width *and* the stride the
backward replays from, because ``tl.associative_scan`` runs over the tile axis
itself. The CUDA kernels split the two (``KLA_ITEMS`` and ``KLA_CHUNK``) and get
an optimum for each; triton takes the compromise."""

RECURRENT_BLOCK_L = 16
"""``recurrent``. Its forward holds one ``[BLOCK_S]`` vector and walks time
serially, so this is *purely* the shared backward's checkpoint stride, the
forward uses it for nothing else, and 16 is that backward's optimum."""


def warps_for(block_l: int, block_s: int) -> int:
    """Warps for a ``[block_l, block_s]`` tile, clamped to [1, 8].

    Deriving this matters most in the backward, which does not choose its own
    ``block_l``: it must use the stride its forward wrote checkpoints at, and a
    fixed count is a trap in both directions. One warp is right for ``chunk``'s
    [16, 16] tile and ruinous for a [128, 16] one, the reverse scans spill and
    the backward runs ~12x slower than the same work at four warps, while four
    warps over [16, 16] is oversubscribed. The forwards take it for the same
    reason: nothing stops a caller passing ``block_l=256``, and at one warp that
    measured 15x its own optimum.
    """
    warps = 1
    while warps < 8 and block_l * block_s > WARP_TILE * warps:
        warps *= 2
    return warps
