"""``mps_chunk`` — the scan on Metal with time as a parallel axis.

The counterpart of :mod:`kla.ops.kernels.mps.recurrent_kla_scan` for the shapes
that one cannot fill. A threadgroup owns a ``(batch, channel)`` pair and splits
each tile of timesteps across its threads, so the grid is ``ROWS`` times wider
than the lane-per-state kernel's ``B*M*S``. That matters at batch-1 prefill and
nowhere else; see the header of ``chunk_kla_scan.metal`` for the phases and for
why the extra parallelism costs about 4x the arithmetic.

The backward is :func:`~kla.ops.kernels.mps.kla_scan_bwd.scan_backward`, shared
with ``mps_recurrent``. An adjoint does not have to mirror its forward: it
recovers λ from the checkpoints this forward writes — same layout, same stride —
then walks a *scalar* reverse recurrence down the serial state lanes, which is
the same work whichever forward got there. So the gradients are exact, at the
tight tolerance, with no composed-map Jacobian anywhere.
"""

from __future__ import annotations

import torch

from kla.ops.kernels.mps._shaders import (
    DEFAULT_CHUNK,
    DEFAULT_ITEMS,
    check_inputs,
    chunk_library,
    tile_geometry,
)
from kla.ops.kernels.mps.kla_scan_bwd import scan_backward


def _grid(B: int, M: int, S: int):
    """``(threads, group_size)`` — one threadgroup per ``(batch, channel)``."""
    block_s, rows = tile_geometry(S)
    return (block_s, rows * M, B), (block_s, rows, 1)


def chunk_forward(
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
    """Chunk forward. Returns ``(y, y_var, lam_fin, eta_fin, lam_ck, eta_ck)``.

    With ``checkpoints=False`` the two checkpoint tensors are one-element
    placeholders — the kernel takes the flag and never writes them.
    """
    check_inputs(msi, si, k, q, a, p, lam0, eta0)
    B, L, M = msi.shape
    S = k.shape[2]
    lib = chunk_library(S, items, chunk)

    dev = msi.device
    y = torch.empty(B, L, M, device=dev, dtype=torch.float32)
    yvar = torch.empty(B, L, M, device=dev, dtype=torch.float32)
    lam_fin = torch.empty(B, M, S, device=dev, dtype=torch.float32)
    eta_fin = torch.empty(B, M, S, device=dev, dtype=torch.float32)

    n_ck = -(-L // chunk) if checkpoints else 1
    ck_shape = (B, M, n_ck, S) if checkpoints else (1,)
    lam_ck = torch.empty(*ck_shape, device=dev, dtype=torch.float32)
    eta_ck = torch.empty(*ck_shape, device=dev, dtype=torch.float32)

    threads, group = _grid(B, M, S)
    lib.kla_chunk_fwd(
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


class _ChunkKLAScan(torch.autograd.Function):
    """Time-parallel Metal forward, with the shared exact backward behind it."""

    @staticmethod
    def forward(ctx, msi, si, k, q, a, p, lam0, eta0, prior, items, chunk):
        needs_grad = any(ctx.needs_input_grad)
        y, yvar, lam_fin, eta_fin, lam_ck, eta_ck = chunk_forward(
            msi,
            si,
            k,
            q,
            a,
            p,
            lam0,
            eta0,
            checkpoints=needs_grad,
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
        return (*grads, None, None, None)


def chunk_kla_scan(
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
    """Differentiable chunk KLA scan → ``(y, y_var, lam_fin, eta_fin)``."""
    return _ChunkKLAScan.apply(msi, si, k, q, a, p, lam0, eta0, prior, items, chunk)
