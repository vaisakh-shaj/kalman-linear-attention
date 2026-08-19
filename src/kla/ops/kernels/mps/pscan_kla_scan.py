"""``mps_pscan`` — the scan as a reduce-then-scan, with no serial carry.

``mps_chunk`` still walks its tiles in order, one threadgroup carrying the state
from each to the next. This drops that: the sequence is cut into chunks, each is
reduced on its own, and the cross-chunk dependency is settled by a parallel scan
over the chunk aggregates. Depth is ``log(NCK)`` rather than ``NCK``, paid for
with ``[B, M, NCK, S]`` of aggregates in device memory and roughly three touches
of each element instead of one.

Two of those aggregate arrays are the backward's checkpoints, which this
schedule computes rather than stores: "the state entering chunk c" is precisely
what the scan produces. So :func:`~kla.ops.kernels.mps.kla_scan_bwd.scan_backward`
runs behind this forward with no changes at all — same layout, same stride, same
convention — and the gradients are the same exact ones.

See the header of ``pscan_kla_scan.metal`` for the five kernels and why the
Möbius and affine rounds cannot be merged.
"""

from __future__ import annotations

import torch

from kla.ops.kernels.mps._shaders import (
    DEFAULT_CHUNK,
    check_inputs,
    pscan_library,
    tile_geometry,
)
from kla.ops.kernels.mps.kla_scan_bwd import scan_backward

_FLAT_GROUP = 256


def _apply_grid(B: int, M: int, S: int, n_ck: int):
    """``(threads, group_size)`` for the read-out kernel.

    ``[BLOCK_S, ROWS]`` with x over states and y over chunks. The chunk axis is
    rounded up to whole threadgroups: the kernel reduces across a row, so the
    padding threads have to exist and take part rather than be dispatched away.
    """
    block_s, rows = tile_geometry(S)
    ck_padded = -(-n_ck // rows) * rows
    return (block_s, ck_padded, B * M), (block_s, rows, 1)


def _scan_aggregates(lib, step, src, dst, S: int, n_ck: int, total: int):
    """Hillis-Steele over the chunk axis, ping-ponging two buffers.

    Work-inefficient by a log factor, but the axis is ``KLA_CHUNK`` times
    shorter than the sequence, and doing it across launches rather than inside
    one threadgroup means ``n_ck`` is bounded by nothing.
    """
    off = 1
    while off < n_ck:
        step(dst, src, S, n_ck, off, total, threads=total, group_size=_FLAT_GROUP)
        src, dst = dst, src
        off <<= 1
    return src


def pscan_forward(
    msi: torch.Tensor,  # v·Λ^v  [B, L, M]
    si: torch.Tensor,  # Λ^v    [B, L, M]
    k: torch.Tensor,  # key    [B, L, S]
    q: torch.Tensor,  # query  [B, L, S]
    a: torch.Tensor,  # decay         [M, S]
    p: torch.Tensor,  # process noise [M, S]
    lam0: torch.Tensor,  # [B, M, S]
    eta0: torch.Tensor,  # [B, M, S]
    prior: bool = False,
    chunk: int = DEFAULT_CHUNK,
):
    """Parallel-scan forward → ``(y, y_var, lam_fin, eta_fin, lam_ck, eta_ck)``.

    Unlike the other two forwards this one takes no ``checkpoints`` flag: the
    checkpoints are the scan's own intermediates, so there is nothing to skip.
    """
    check_inputs(msi, si, k, q, a, p, lam0, eta0)
    B, L, M = msi.shape
    S = k.shape[2]
    lib = pscan_library(S, chunk)

    dev = msi.device
    f32 = torch.float32
    n_ck = -(-L // chunk)
    total = B * M * n_ck * S

    mob = torch.empty(B, M, n_ck, S, 4, device=dev, dtype=f32)
    mob_alt = torch.empty_like(mob)
    aff = torch.empty(B, M, n_ck, S, 2, device=dev, dtype=f32)
    aff_alt = torch.empty_like(aff)
    lam_ck = torch.empty(B, M, n_ck, S, device=dev, dtype=f32)
    eta_ck = torch.empty(B, M, n_ck, S, device=dev, dtype=f32)

    y = torch.empty(B, L, M, device=dev, dtype=f32)
    yvar = torch.empty(B, L, M, device=dev, dtype=f32)
    lam_fin = torch.empty(B, M, S, device=dev, dtype=f32)
    eta_fin = torch.empty(B, M, S, device=dev, dtype=f32)

    lib.kla_pscan_mob_reduce(
        mob,
        si,
        k,
        p,
        a,
        L,
        M,
        S,
        n_ck,
        total,
        threads=total,
        group_size=_FLAT_GROUP,
    )
    mob_in = _scan_aggregates(lib, lib.kla_pscan_mob_step, mob, mob_alt, S, n_ck, total)

    lib.kla_pscan_aff_reduce(
        aff,
        lam_ck,
        mob_in,
        msi,
        si,
        k,
        a,
        p,
        lam0,
        L,
        M,
        S,
        n_ck,
        total,
        threads=total,
        group_size=_FLAT_GROUP,
    )
    aff_in = _scan_aggregates(lib, lib.kla_pscan_aff_step, aff, aff_alt, S, n_ck, total)

    threads, group = _apply_grid(B, M, S, n_ck)
    lib.kla_pscan_apply(
        y,
        yvar,
        lam_fin,
        eta_fin,
        eta_ck,
        lam_ck,
        aff_in,
        msi,
        si,
        k,
        q,
        a,
        p,
        eta0,
        L,
        M,
        S,
        n_ck,
        int(prior),
        threads=threads,
        group_size=group,
    )
    return y, yvar, lam_fin, eta_fin, lam_ck, eta_ck


class _PScanKLAScan(torch.autograd.Function):
    """Reduce-then-scan Metal forward, with the shared exact backward behind it."""

    @staticmethod
    def forward(ctx, msi, si, k, q, a, p, lam0, eta0, prior, chunk):
        y, yvar, lam_fin, eta_fin, lam_ck, eta_ck = pscan_forward(
            msi, si, k, q, a, p, lam0, eta0, prior=prior, chunk=chunk
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
        return (*grads, None, None)


def pscan_kla_scan(
    msi, si, k, q, a, p, lam0, eta0, prior: bool = False, chunk: int = DEFAULT_CHUNK
):
    """Differentiable parallel-scan KLA scan → ``(y, y_var, lam_fin, eta_fin)``."""
    return _PScanKLAScan.apply(msi, si, k, q, a, p, lam0, eta0, prior, chunk)
