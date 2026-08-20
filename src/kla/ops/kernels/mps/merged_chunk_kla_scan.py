"""``mps_merged_chunk`` — ``mps_chunk``'s six phases in three, one scan not two.

Identical in shape to :mod:`kla.ops.kernels.mps.chunk_kla_scan`: same
threadgroup per ``(batch, channel)``, same tiles of ``ROWS * ITEMS`` timesteps,
same grid, same checkpoints, same backward. The difference is entirely inside
the tile. That kernel composes a 2x2 Möbius map for λ, walks it to produce λ,
then builds the affine leaves ``(α, r)`` that walk unlocked and scans *those* —
because α_t reads λ_{t-1}, so the second set of leaves cannot exist any earlier.
This one composes the 3x3 map of ``kla_merged.metal``, which carries η in the
same homogeneous coordinates, so its leaf depends on ``(φ, r, a, p)`` alone and
one scan does both.

What that buys, concretely: one threadgroup scan instead of two (each is
``log2(ROWS)`` Hillis-Steele rounds with two barriers apiece), one broadcast
instead of two, and the disappearance of the ``var_h``/``alpha_h``/``r_h``
per-thread arrays that existed only to carry one phase's output to another —
24 registers at ``ITEMS=8``, on the kernel whose entire reason to exist is
occupancy.

The backward is :func:`~kla.ops.kernels.mps.kla_scan_bwd.scan_backward`,
unchanged, shared with every other MPS cell. It replays a scalar recurrence from
``[B, M, NCK, S]`` checkpoints and never sees a composed map, so merging the
forward is invisible to it — see ``tests/test_backends.py``, which holds it to
the same exact-gradient contract as the rest.
"""

from __future__ import annotations

import torch

from kla.ops.kernels.mps._shaders import (
    DEFAULT_CHUNK,
    DEFAULT_ITEMS,
    check_inputs,
    merged_chunk_library,
    tile_geometry,
)
from kla.ops.kernels.mps.kla_scan_bwd import scan_backward


def _grid(B: int, M: int, S: int):
    """``(threads, group_size)`` — one threadgroup per ``(batch, channel)``."""
    block_s, rows = tile_geometry(S)
    return (block_s, rows * M, B), (block_s, rows, 1)


def merged_chunk_forward(
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
    """Merged chunk forward. Returns ``(y, y_var, lam_fin, eta_fin, lam_ck, eta_ck)``.

    With ``checkpoints=False`` the two checkpoint tensors are one-element
    placeholders — the kernel takes the flag and never writes them.
    """
    check_inputs(msi, si, k, q, a, p, lam0, eta0)
    B, L, M = msi.shape
    S = k.shape[2]
    lib = merged_chunk_library(S, items, chunk)

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
    lib.kla_merged_chunk_fwd(
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


class _MergedChunkKLAScan(torch.autograd.Function):
    """One-scan time-parallel Metal forward, with the shared exact backward."""

    @staticmethod
    def forward(ctx, msi, si, k, q, a, p, lam0, eta0, prior, items, chunk):
        needs_grad = any(ctx.needs_input_grad)
        y, yvar, lam_fin, eta_fin, lam_ck, eta_ck = merged_chunk_forward(
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


def merged_chunk_kla_scan(
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
    """Differentiable merged chunk KLA scan → ``(y, y_var, lam_fin, eta_fin)``."""
    return _MergedChunkKLAScan.apply(
        msi, si, k, q, a, p, lam0, eta0, prior, items, chunk
    )
