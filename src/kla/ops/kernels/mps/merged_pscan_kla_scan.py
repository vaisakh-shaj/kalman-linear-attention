"""``mps_merged_pscan`` — ``mps_pscan``'s five kernels in three, one round not two.

Same implementation as :mod:`kla.ops.kernels.mps.pscan_kla_scan`: cut the
sequence into chunks, reduce each independently, settle the cross-chunk
dependency with a Hillis-Steele scan over the aggregates rather than a carry.
Same ``[B, M, NCK, S]`` shape of intermediate, same checkpoints, same backward.

What goes away is the second reduce-scan-apply round. That one exists only
because the affine leaf ``α_t`` reads ``λ_{t-1}``, so the information vector's
leaves cannot be built until the Möbius round has produced λ. The 3x3 leaf of
``kla_merged.metal`` depends on ``(φ, r, a, p)`` alone, so:

    reduce → step (×log2 NCK) → apply

replaces reduce → step → reduce → step → apply. Each element is touched about
twice instead of three times, and the doubling scan — the part whose depth is
``log(NCK)`` rather than constant — runs once.

The aggregate is the same pair of ping-ponged buffers, one element wider: 8
floats (7 live, 1 padding) against a float4 plus a float2, so 16 floats per
``(b, m, c, s)`` against 12, and two buffers rather than four. Against that,
``lam_ck`` no longer needs a kernel of its own to produce it — the apply kernel
writes both checkpoints from the one prefix it applies.

The backward is :func:`~kla.ops.kernels.mps.kla_scan_bwd.scan_backward`,
unchanged, shared with every other MPS cell.
"""

from __future__ import annotations

import torch

from kla.ops.kernels.mps._shaders import (
    DEFAULT_CHUNK,
    check_inputs,
    merged_pscan_library,
    tile_geometry,
)
from kla.ops.kernels.mps.kla_scan_bwd import scan_backward

_FLAT_GROUP = 256

_AGG_FLOATS = 8
"""Floats per aggregate: ``KlaMerged`` is two float4s. Seven are live — the
2x2 precision block, the two-entry η row and the (3,3) scalar — and the eighth
is padding, so the aggregate is two vector loads rather than seven scalar ones.
See the header of ``kla_merged.metal`` for why D is carried rather than
reconstructed as ``1 - A``."""


def _apply_grid(B: int, M: int, S: int, n_ck: int):
    """``(threads, group_size)`` for the read-out kernel.

    ``[BLOCK_S, ROWS]`` with x over states and y over chunks. The chunk axis is
    rounded up to whole threadgroups: the kernel reduces across a row, so the
    padding threads have to exist and take part rather than be dispatched away.
    """
    block_s, rows = tile_geometry(S)
    ck_padded = -(-n_ck // rows) * rows
    return (block_s, ck_padded, B * M), (block_s, rows, 1)


def _scan_aggregates(lib, src, dst, S: int, n_ck: int, total: int):
    """Hillis-Steele over the chunk axis, ping-ponging two buffers.

    One of these, where :mod:`~kla.ops.kernels.mps.pscan_kla_scan` runs two.
    """
    off = 1
    while off < n_ck:
        lib.kla_merged_pscan_step(
            dst, src, S, n_ck, off, total, threads=total, group_size=_FLAT_GROUP
        )
        src, dst = dst, src
        off <<= 1
    return src


def merged_pscan_forward(
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
    """Merged parallel-scan forward → ``(y, y_var, lam_fin, eta_fin, lam_ck, eta_ck)``.

    Like the two-scan version this takes no ``checkpoints`` flag: the
    checkpoints are the scan's own intermediates, so there is nothing to skip.
    """
    check_inputs(msi, si, k, q, a, p, lam0, eta0)
    B, L, M = msi.shape
    S = k.shape[2]
    lib = merged_pscan_library(S, chunk)

    dev = msi.device
    f32 = torch.float32
    n_ck = -(-L // chunk)
    total = B * M * n_ck * S

    mrg = torch.empty(B, M, n_ck, S, _AGG_FLOATS, device=dev, dtype=f32)
    mrg_alt = torch.empty_like(mrg)
    lam_ck = torch.empty(B, M, n_ck, S, device=dev, dtype=f32)
    eta_ck = torch.empty(B, M, n_ck, S, device=dev, dtype=f32)

    y = torch.empty(B, L, M, device=dev, dtype=f32)
    yvar = torch.empty(B, L, M, device=dev, dtype=f32)
    lam_fin = torch.empty(B, M, S, device=dev, dtype=f32)
    eta_fin = torch.empty(B, M, S, device=dev, dtype=f32)

    lib.kla_merged_pscan_reduce(
        mrg,
        msi,
        si,
        k,
        a,
        p,
        L,
        M,
        S,
        n_ck,
        total,
        threads=total,
        group_size=_FLAT_GROUP,
    )
    mrg_in = _scan_aggregates(lib, mrg, mrg_alt, S, n_ck, total)

    threads, group = _apply_grid(B, M, S, n_ck)
    lib.kla_merged_pscan_apply(
        y,
        yvar,
        lam_fin,
        eta_fin,
        lam_ck,
        eta_ck,
        mrg_in,
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
        int(prior),
        threads=threads,
        group_size=group,
    )
    return y, yvar, lam_fin, eta_fin, lam_ck, eta_ck


class _MergedPScanKLAScan(torch.autograd.Function):
    """One-round reduce-then-scan Metal forward, with the shared exact backward."""

    @staticmethod
    def forward(ctx, msi, si, k, q, a, p, lam0, eta0, prior, chunk):
        y, yvar, lam_fin, eta_fin, lam_ck, eta_ck = merged_pscan_forward(
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


def merged_pscan_kla_scan(
    msi, si, k, q, a, p, lam0, eta0, prior: bool = False, chunk: int = DEFAULT_CHUNK
):
    """Differentiable merged parallel-scan KLA scan.

    Returns ``(y, y_var, lam_fin, eta_fin)``.
    """
    return _MergedPScanKLAScan.apply(msi, si, k, q, a, p, lam0, eta0, prior, chunk)
