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
"""Largest ``d_state`` the Metal kernels take: one threadgroup must hold every
state of a channel, so ``next_pow2(d_state)`` has to fit in a threadgroup."""

DEFAULT_CHUNK = 16
"""Checkpoint stride: timesteps the backward replays per checkpoint. Trades the
forward's checkpoint traffic (``[B, M, ceil(L/CHUNK), S]``, written once)
against the backward's two thread-private replay buffers (``CHUNK`` floats
each). Both forwards write at this stride, so either can hand its checkpoints to
:func:`~kla.ops.kernels.mps.kla_scan_bwd.scan_backward`."""

DEFAULT_ITEMS = 8
"""Timesteps each thread of ``mps_chunk``'s forward walks serially. Trades the tile
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
def _library(stems: tuple, block_s: int, rows: int, chunk: int, items: int = 0):
    """Compile ``<stem>.metal`` sources for one geometry (cached per process)."""
    require_mps()
    prelude = (
        f"#define KLA_BLOCK_S {block_s}\n"
        f"#define KLA_ROWS {rows}\n"
        f"#define KLA_CHUNK {chunk}\n"
        f"#define KLA_ITEMS {items}\n"
    )
    source = "\n".join(
        [prelude, (_SRC_DIR / "kla_common.metal").read_text()]
        + [(_SRC_DIR / f"{s}.metal").read_text() for s in stems]
    )
    return torch.mps.compile_shader(source)


def tile_geometry(d_state: int) -> tuple[int, int]:
    """``(BLOCK_S, ROWS)`` for ``mps_chunk``, where ROWS spans *time*.

    Unlike :func:`launch_geometry` the second axis is never pinned to 1 — it is
    the whole point of that kernel — so the read-out reduction takes its
    threadgroup-memory path once past a SIMD-group's width.
    """
    block_s = _next_pow2(d_state)
    return block_s, max(1, _TG_THREADS // block_s)


def chunk_library(d_state: int, items: int = DEFAULT_ITEMS, chunk: int = DEFAULT_CHUNK):
    """``mps_chunk``'s forward, specialized on the state width and tile shape.

    ``items`` is this kernel's own tile depth; ``chunk`` is the *backward's*
    checkpoint stride, which it only writes at. The two are independent, and
    conflating them under one name is a bug waiting to happen -- hence the
    separate ``KLA_ITEMS`` and ``KLA_CHUNK`` defines.
    """
    _require_dstate(d_state, "mps_chunk")
    block_s, rows = tile_geometry(d_state)
    return _library(("chunk_kla_scan",), block_s, rows, chunk, items)


def recurrent_library(d_state: int, chunk: int = DEFAULT_CHUNK):
    """``mps_recurrent``'s forward, specialized on the state width."""
    _require_dstate(d_state, "mps_recurrent")
    block_s, rows = launch_geometry(d_state)
    return _library(("recurrent_kla_scan",), block_s, rows, chunk)


def pscan_library(d_state: int, chunk: int = DEFAULT_CHUNK):
    """``mps_pscan``'s five kernels, specialized on the state width.

    ``chunk`` is both the reduce-then-scan's chunk length and the checkpoint
    stride here — the same number, because the states entering each chunk *are*
    the checkpoints this schedule already computes.
    """
    _require_dstate(d_state, "mps_pscan")
    block_s, rows = tile_geometry(d_state)
    return _library(("pscan_kla_scan",), block_s, rows, chunk)


def bwd_library(d_state: int, chunk: int = DEFAULT_CHUNK):
    """The backward both schedules share. Lane-per-state, like the recurrent
    forward, whatever geometry the forward that wrote the checkpoints used."""
    _require_dstate(d_state, "the MPS backward")
    block_s, rows = launch_geometry(d_state)
    return _library(("kla_scan_bwd",), block_s, rows, chunk)


def _require_dstate(d_state: int, who: str) -> None:
    if d_state > MAX_DSTATE:
        raise NotImplementedError(
            f"{who} supports d_state <= {MAX_DSTATE} (got {d_state}); "
            "use backend='torch' for this case."
        )


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
