"""The host half every Metal cell shares: launch, autograd, checkpoints.

The three forwards differ only in the compiled library, the kernel entry point
and the launch geometry, so the tensor bookkeeping around them -- output and
checkpoint allocation, the grid, and the :class:`torch.autograd.Function` that
hands it all to the one shared backward -- lives here rather than three times
over. A cell is a :class:`Cell` naming those three things; see :func:`make_scan`.

Checkpoints are allocated only when a backward will run: with
``checkpoints=False`` both tensors are one-element placeholders and the kernel,
which takes the flag as an argument, never writes them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import torch

from kla.ops.kernels.mps._shaders import DEFAULT_CHUNK, DEFAULT_ITEMS, check_inputs
from kla.ops.kernels.mps.kla_scan_bwd import scan_backward


class Cell(NamedTuple):
    """How to launch one Metal forward.

    ``library(d_state, ...)`` compiles for the shape (see
    :mod:`kla.ops.kernels.mps._shaders`), ``entry`` names the kernel inside it,
    and ``geometry(d_state)`` gives the ``(BLOCK_S, ROWS)`` its threadgroup
    wants. ``tiled`` says whether ROWS spans time, which is both what makes the
    grid ``ROWS`` times wider and what makes ``items`` meaningful.
    """

    library: Callable
    entry: str
    geometry: Callable
    tiled: bool


def _grid(cell: Cell, B: int, M: int, S: int):
    """``(threads, group_size)`` for one problem shape.

    A tiled cell puts one threadgroup on each ``(batch, channel)`` pair. A
    lane-per-state cell instead rounds the channel axis up to whole threadgroups:
    it reduces across a group, so the padding threads have to exist and take
    part rather than be dispatched away.
    """
    block_s, rows = cell.geometry(S)
    m_axis = rows * M if cell.tiled else -(-M // rows) * rows
    return (block_s, m_axis, B), (block_s, rows, 1)


def launch_forward(
    cell: Cell,
    msi: torch.Tensor,  # v·Λ^v  [B, L, M]
    si: torch.Tensor,  # Λ^v    [B, L, M]
    k: torch.Tensor,  # key    [B, L, S]
    q: torch.Tensor,  # query  [B, L, S]
    a: torch.Tensor,  # decay         [M, S]
    p: torch.Tensor,  # process noise [M, S]
    lam0: torch.Tensor,  # [B, M, S]
    eta0: torch.Tensor,  # [B, M, S]
    checkpoints: bool = False,
    prior: bool = False,
    items: int = DEFAULT_ITEMS,
    chunk: int = DEFAULT_CHUNK,
):
    """Run one forward → ``(y, y_var, lam_fin, eta_fin, lam_ck, eta_ck)``.

    ``y`` / ``y_var`` are ``[B, L, M]``; the state tensors are ``[B, M, S]``.
    ``prior`` is ``decode_from_prior``: it moves the read-out one predict step
    ahead.
    """
    check_inputs(msi, si, k, q, a, p, lam0, eta0)
    B, L, M = msi.shape
    S = k.shape[2]
    lib = cell.library(S, items, chunk) if cell.tiled else cell.library(S, chunk)

    def empty(*shape):
        return torch.empty(*shape, device=msi.device, dtype=torch.float32)

    y, yvar = empty(B, L, M), empty(B, L, M)
    lam_fin, eta_fin = empty(B, M, S), empty(B, M, S)

    n_ck = -(-L // chunk) if checkpoints else 1
    ck_shape = (B, M, n_ck, S) if checkpoints else (1,)
    lam_ck, eta_ck = empty(*ck_shape), empty(*ck_shape)

    threads, group = _grid(cell, B, M, S)
    getattr(lib, cell.entry)(
        y,
        yvar,
        lam_fin,
        eta_fin,
        lam_ck,
        eta_ck,
        msi,
        si,
        k,
        q,
        a,
        p,
        lam0,
        eta0,
        L,
        M,
        S,
        n_ck,
        int(checkpoints),
        int(prior),
        threads=threads,
        group_size=group,
    )
    return y, yvar, lam_fin, eta_fin, lam_ck, eta_ck


class _KLAScan(torch.autograd.Function):
    """Any Metal forward, with the one shared exact backward behind it.

    The cell is an ordinary non-tensor argument, so all three implementations
    share this class instead of subclassing it: the backward replays a scalar
    recurrence from the checkpoints and never learns which forward wrote them.
    """

    @staticmethod
    def forward(ctx, msi, si, k, q, a, p, lam0, eta0, prior, items, chunk, cell):
        y, yvar, lam_fin, eta_fin, lam_ck, eta_ck = launch_forward(
            cell,
            msi,
            si,
            k,
            q,
            a,
            p,
            lam0,
            eta0,
            checkpoints=any(ctx.needs_input_grad),
            prior=prior,
            items=items,
            chunk=chunk,
        )
        ctx.chunk = chunk
        ctx.prior = prior
        ctx.save_for_backward(msi, si, k, q, a, p, lam_ck, eta_ck)
        return y, yvar, lam_fin, eta_fin

    @staticmethod
    def backward(ctx, dy, dyvar, dlam_fin, deta_fin):
        msi, si, k, q, a, p, lam_ck, eta_ck = ctx.saved_tensors
        grads = scan_backward(
            dy.contiguous(),
            dyvar.contiguous(),
            dlam_fin.contiguous(),
            deta_fin.contiguous(),
            msi,
            si,
            k,
            q,
            a,
            p,
            lam_ck,
            eta_ck,
            prior=ctx.prior,
            chunk=ctx.chunk,
        )
        return (*grads, None, None, None, None)


def make_scan(cell: Cell, name: str, doc: str):
    """The differentiable entry point for one cell → ``(y, y_var, lam, eta)``."""

    def scan(
        msi,
        si,
        k,
        q,
        a,
        p,
        lam0,
        eta0,
        prior: bool = False,
        items: int = DEFAULT_ITEMS,
        chunk: int = DEFAULT_CHUNK,
    ):
        return _KLAScan.apply(
            msi, si, k, q, a, p, lam0, eta0, prior, items, chunk, cell
        )

    scan.__name__ = name
    scan.__qualname__ = name
    scan.__doc__ = doc
    return scan
