"""Generic associative-scan utilities for the pure-torch backend.

Three interchangeable implementations of an inclusive scan over tuples of
tensors (see ``docs/implementations.md`` for the implementations they name):

- ``associative``: ``torch._higher_order_ops.associative_scan`` (PyTorch 2.8+).
  A pscan — the whole sequence at once, no serial carry.
- ``doubling``: vectorized Hillis–Steele doubling. Also a pscan. O(L log L) work
  but only elementwise ops and ~log2(L) kernel launches; works everywhere, fully
  autograd-compatible.
- ``chunk``: doubling inside a chunk, a serial carry across chunks. O(L) work
  and O(L/C) serial steps, so it is the middle ground between the two above and
  ``sequential``.
There is no ``sequential`` entry here, and that is the point: a serial walk does
not need an associative combine at all — it can *apply* the update to a running
value instead of composing two of them. That is
:func:`kla.ops.kla_ops._recurrent_lambda_eta`, built on
``torch._higher_order_ops.scan``, and it lives next to the recurrence it applies
because it is specific to it.

All combine functions take ``(left, right)`` tuples where ``left`` is the
earlier prefix, and return the composed element.
"""

from __future__ import annotations

import functools
from typing import Callable, Sequence

import torch

CombineFn = Callable[
    [tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]
]


def doubling_scan(
    combine_fn: CombineFn, xs: Sequence[torch.Tensor], dim: int
) -> tuple[torch.Tensor, ...]:
    """Inclusive scan via Hillis–Steele doubling."""
    ys = tuple(xs)
    length = ys[0].size(dim)
    offset = 1
    while offset < length:
        left = tuple(t.narrow(dim, 0, length - offset) for t in ys)
        right = tuple(t.narrow(dim, offset, length - offset) for t in ys)
        combined = combine_fn(left, right)
        ys = tuple(
            torch.cat((t.narrow(dim, 0, offset), c), dim=dim)
            for t, c in zip(ys, combined)
        )
        offset *= 2
    return ys


DEFAULT_CHUNK = 64
"""Timesteps per chunk in :func:`chunk_scan`. Trades the serial carry (once per
chunk) against the doubling scan's O(C log C) work inside one."""


def chunk_scan(
    combine_fn: CombineFn,
    xs: Sequence[torch.Tensor],
    dim: int,
    chunk: int = DEFAULT_CHUNK,
) -> tuple[torch.Tensor, ...]:
    """Inclusive scan, chunked: parallel inside a chunk, serial across chunks.

    The torch counterpart of the GPU ``chunk`` kernels. Each chunk is scanned
    with :func:`doubling_scan`, then the previous chunk's last element is
    composed into every element of this one — which is the serial carry, paid
    once per chunk rather than once per timestep.
    """
    length = xs[0].size(dim)
    blocks: list[tuple[torch.Tensor, ...]] = []
    carry: tuple[torch.Tensor, ...] | None = None
    for start in range(0, length, chunk):
        width = min(chunk, length - start)
        block = doubling_scan(
            combine_fn, tuple(t.narrow(dim, start, width) for t in xs), dim
        )
        if carry is not None:
            # The carry is the earlier prefix, so it goes on the left.
            prefix = tuple(c.unsqueeze(dim).expand_as(b) for c, b in zip(carry, block))
            block = combine_fn(prefix, block)
        carry = tuple(t.select(dim, width - 1) for t in block)
        blocks.append(block)
    return tuple(torch.cat([b[i] for b in blocks], dim=dim) for i in range(len(xs)))


@functools.cache
def _associative_scan_available() -> bool:
    try:
        from torch._higher_order_ops import associative_scan  # noqa: F401
    except ImportError:
        return False
    return True


def associative_scan(
    combine_fn: CombineFn, xs: Sequence[torch.Tensor], dim: int
) -> tuple[torch.Tensor, ...]:
    """Inclusive scan via torch's higher-order associative_scan op."""
    from torch._higher_order_ops import associative_scan as _scan

    def combine(left, right):
        return combine_fn(tuple(left), tuple(right))

    out = _scan(
        combine, tuple(x.contiguous() for x in xs), dim=dim, combine_mode="generic"
    )
    return tuple(out)


def resolve_scan(
    scan_impl: str,
) -> Callable[[CombineFn, Sequence[torch.Tensor], int], tuple[torch.Tensor, ...]]:
    if scan_impl == "auto":
        scan_impl = "associative" if _associative_scan_available() else "doubling"
    if scan_impl == "associative":
        return associative_scan
    if scan_impl == "doubling":
        return doubling_scan
    if scan_impl == "chunk":
        return chunk_scan
    raise ValueError(f"Unknown scan_impl: {scan_impl!r}")
