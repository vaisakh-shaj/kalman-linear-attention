"""Compilation and launch geometry for the KLA Metal shaders.

The ``.metal`` sources next to this file are compiled at first use with
:func:`torch.mps.compile_shader`, which is part of torch itself.

Sources are *specialized*, the way triton specializes on a ``tl.constexpr``:
the state width and the backward's chunk length are ``#define``\\ d into the
source before compiling, so the reductions unroll to a fixed shape and the
backward's replay buffers are fixed-size thread-private arrays. One library is
compiled per distinct geometry and cached for the process, so a model with a
single ``d_state`` compiles once.

Geometry. A threadgroup is ``[BLOCK_S, ROWS]``: ``BLOCK_S = next_pow2(d_state)``
lanes covering every state of one channel, and ``ROWS`` channels stacked on top.
``BLOCK_S <= 32`` keeps a row inside one SIMD-group, which is what lets the
read-out reduction be a shuffle butterfly rather than a trip through threadgroup
memory; past that the generator pins ``ROWS = 1`` and the wider reduction goes
through threadgroup memory instead.
"""

from __future__ import annotations

import functools
from pathlib import Path

import torch

_SRC_DIR = Path(__file__).resolve().parent

MAX_DSTATE = 128
"""Largest ``d_state`` the fused kernels take: one threadgroup must hold every
state of a channel, so ``next_pow2(d_state)`` has to fit in a threadgroup."""

DEFAULT_CHUNK = 16
"""Timesteps the fused backward replays per checkpoint. Trades the forward's
checkpoint traffic (``[B, M, ceil(L/CHUNK), S]``, written once) against the
backward's two thread-private replay buffers (``CHUNK`` floats each)."""

DEFAULT_ITEMS = 8
"""Timesteps each thread of the tiled forward walks serially. Trades the tile
scan's overhead (paid once per ``ITEMS`` steps) against parallelism, since the
tile covers ``ROWS * ITEMS`` timesteps at once."""

_TG_THREADS = 256  # threadgroup size to aim for


def _next_pow2(n: int) -> int:
    return 1 << max(0, (n - 1).bit_length())


def is_available() -> bool:
    """True when this machine can build and run the shaders."""
    return (
        torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
        and hasattr(torch.mps, "compile_shader")
    )


def require_mps(what: str = "The MPS KLA backend") -> None:
    if not is_available():
        raise NotImplementedError(
            f"{what} needs an Apple-silicon GPU and a torch built with MPS "
            "support (torch.mps.compile_shader); use backend='torch' (or 'auto')."
        )


def launch_geometry(d_state: int) -> tuple[int, int]:
    """``(BLOCK_S, ROWS)`` for a given state width — see the module docstring."""
    block_s = _next_pow2(d_state)
    rows = max(1, _TG_THREADS // block_s) if block_s <= 32 else 1
    return block_s, rows


@functools.lru_cache(maxsize=None)
def _library(stem: str, block_s: int, rows: int, chunk: int):
    """Compile ``<stem>.metal`` for one geometry (cached for the process)."""
    require_mps()
    prelude = (
        f"#define KLA_BLOCK_S {block_s}\n"
        f"#define KLA_ROWS {rows}\n"
        f"#define KLA_CHUNK {chunk}\n"
    )
    source = "\n".join(
        [
            prelude,
            (_SRC_DIR / "kla_common.metal").read_text(),
            (_SRC_DIR / f"{stem}.metal").read_text(),
        ]
    )
    return torch.mps.compile_shader(source)


def scan_library():
    """The composed-strategy primitives. Shape-independent, so compiled once."""
    return _library("lane_mobius_scan", 1, 1, 1)


def tile_geometry(d_state: int) -> tuple[int, int]:
    """``(BLOCK_S, ROWS)`` for the tiled forward, where ROWS spans *time*.

    Unlike :func:`launch_geometry` the second axis is never pinned to 1 — it is
    the whole point of that kernel — so the read-out reduction takes its
    threadgroup-memory path once past a SIMD-group's width.
    """
    block_s = _next_pow2(d_state)
    return block_s, max(1, _TG_THREADS // block_s)


def tiled_library(d_state: int, items: int = DEFAULT_ITEMS):
    """The tiled forward, specialized on the state width and the tile shape."""
    if d_state > MAX_DSTATE:
        raise NotImplementedError(
            f"The tiled MPS kernel supports d_state <= {MAX_DSTATE} (got {d_state}); "
            "use backend='mps_composed' or 'torch' for this case."
        )
    block_s, rows = tile_geometry(d_state)
    return _library("tiled_kla_scan", block_s, rows, items)


def fused_library(d_state: int, chunk: int = DEFAULT_CHUNK):
    """The fused forward/backward, specialized on the state width."""
    if d_state > MAX_DSTATE:
        raise NotImplementedError(
            f"The fused MPS kernels support d_state <= {MAX_DSTATE} (got {d_state}); "
            "use backend='mps_composed' or 'torch' for this case."
        )
    block_s, rows = launch_geometry(d_state)
    return _library("fused_kla_scan", block_s, rows, chunk)


def check_inputs(*tensors: torch.Tensor) -> None:
    """Every kernel here reads raw float32 pointers with computed offsets."""
    for t in tensors:
        if not t.is_mps:
            raise NotImplementedError(
                "The MPS KLA backend requires tensors on an 'mps' device."
            )
        if t.dtype != torch.float32:
            raise NotImplementedError(
                f"The MPS KLA kernels are float32-only (got {t.dtype}); Metal has "
                "no float64. Use backend='torch' for a float64 (gradcheck) run."
            )
