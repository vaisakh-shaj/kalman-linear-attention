"""Generic associative-scan utilities for the pure-torch backend.

Two interchangeable parallel implementations of an inclusive scan over tuples
of tensors, plus a sequential reference:

- ``associative``: ``torch._higher_order_ops.associative_scan`` (PyTorch 2.8+).
- ``doubling``: vectorized Hillis–Steele doubling. O(L log L) work but only
  elementwise ops and ~log2(L) kernel launches; works everywhere, fully
  autograd-compatible.
- ``sequential``: python loop, used as a correctness reference and for short
  sequences / decode.

All combine functions take ``(left, right)`` tuples where ``left`` is the
earlier prefix, and return the composed element.
"""

from __future__ import annotations

import functools
from typing import Callable, Sequence

import torch

CombineFn = Callable[[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]], tuple[torch.Tensor, ...]]


def doubling_scan(combine_fn: CombineFn, xs: Sequence[torch.Tensor], dim: int) -> tuple[torch.Tensor, ...]:
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


def sequential_scan(combine_fn: CombineFn, xs: Sequence[torch.Tensor], dim: int) -> tuple[torch.Tensor, ...]:
    """Inclusive scan via a python loop (reference implementation)."""
    length = xs[0].size(dim)
    state = tuple(t.select(dim, 0) for t in xs)
    outs = [state]
    for t in range(1, length):
        step = tuple(x.select(dim, t) for x in xs)
        state = combine_fn(state, step)
        outs.append(state)
    return tuple(torch.stack([o[i] for o in outs], dim=dim) for i in range(len(xs)))


@functools.cache
def _associative_scan_available() -> bool:
    try:
        from torch._higher_order_ops import associative_scan  # noqa: F401
    except ImportError:
        return False
    return True


def associative_scan(combine_fn: CombineFn, xs: Sequence[torch.Tensor], dim: int) -> tuple[torch.Tensor, ...]:
    """Inclusive scan via torch's higher-order associative_scan op."""
    from torch._higher_order_ops import associative_scan as _scan

    def combine(left, right):
        return combine_fn(tuple(left), tuple(right))

    out = _scan(combine, tuple(x.contiguous() for x in xs), dim=dim, combine_mode="generic")
    return tuple(out)


def resolve_scan(scan_impl: str) -> Callable[[CombineFn, Sequence[torch.Tensor], int], tuple[torch.Tensor, ...]]:
    if scan_impl == "auto":
        scan_impl = "associative" if _associative_scan_available() else "doubling"
    if scan_impl == "associative":
        return associative_scan
    if scan_impl == "doubling":
        return doubling_scan
    if scan_impl == "sequential":
        return sequential_scan
    raise ValueError(f"Unknown scan_impl: {scan_impl!r}")
