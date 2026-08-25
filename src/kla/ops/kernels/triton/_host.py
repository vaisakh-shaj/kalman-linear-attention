"""The host half every triton cell shares: launch, autograd, checkpoints.

The three forwards differ only in the kernel object and two launch details, so
the tensor bookkeeping around them -- the ``[B,M,L]`` permutes the kernels index
in, the output and checkpoint allocations, the warp count, and the
:class:`torch.autograd.Function` that hands it all to the one shared backward --
lives here rather than three times over. A cell is then a kernel plus a
:class:`Cell` describing how to launch it; see :func:`make_scan`.

Checkpoints are allocated only when a backward will run: with
``checkpoints=False`` both tensors are one-element placeholders and the kernel,
which takes the flag as a ``constexpr``, never writes them.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton

from kla.ops.kernels.triton._tuning import warps_for


class Cell(NamedTuple):
    """How to launch one triton forward.

    ``tile_key`` is the kernel's ``constexpr`` for the tile / checkpoint stride:
    the chunk cells scan over it (``BLOCK_L``), the recurrent one only writes
    checkpoints at it (``CK_STRIDE``). ``warp_rows`` is the tile height the warp
    count is derived from -- ``None`` means "the stride itself", which is right
    for a cell that scans the tile and wrong for one holding a single
    ``[BLOCK_S]`` vector, hence ``1`` there.
    """

    kernel: object
    default_block_l: int
    tile_key: str = "BLOCK_L"
    warp_rows: int | None = None


def launch_forward(
    cell: Cell,
    msi,
    si,
    h,
    w,
    a,
    q,
    lam0,
    eta0,
    checkpoints: bool = False,
    prior: bool = False,
    block_l: int | None = None,
    num_warps: int | None = None,
):
    """Run one forward. msi/si [B,L,M], h/w [B,L,S], lam0/eta0 [B,M,S].

    ``a``/``q`` (decay and process noise) are static ``[M, S]``. Returns
    ``(y, yvar)`` ``[B,L,M]``, the final ``(lam, eta)``, the permuted ``[B,M,L]``
    inputs the backward wants, and the ``[B,M,NCK,S]`` checkpoints.
    """
    block_l = cell.default_block_l if block_l is None else block_l
    B, L, M = msi.shape
    S = h.shape[2]

    msi_t = msi.permute(0, 2, 1).contiguous()  # [B,M,L]
    si_t = si.permute(0, 2, 1).contiguous()
    h_c, w_c = h.contiguous(), w.contiguous()
    lam0_c, eta0_c = lam0.contiguous(), eta0.contiguous()

    def empty(*shape):
        return torch.empty(*shape, device=msi.device, dtype=torch.float32)

    y, yvar = empty(B, M, L), empty(B, M, L)
    lam_fin, eta_fin = empty(B, M, S), empty(B, M, S)

    n_chunks = triton.cdiv(L, block_l)
    block_s = triton.next_power_of_2(S)
    if num_warps is None:
        rows = block_l if cell.warp_rows is None else cell.warp_rows
        num_warps = warps_for(rows, block_s)
    ck_shape = (B, M, n_chunks, S) if checkpoints else (1,)
    lam_ck, eta_ck = empty(*ck_shape), empty(*ck_shape)

    cell.kernel[(B * M,)](
        msi_t,
        si_t,
        h_c,
        w_c,
        a.contiguous(),
        q.contiguous(),
        lam0_c,
        eta0_c,
        y,
        yvar,
        lam_fin,
        eta_fin,
        lam_ck,
        eta_ck,
        M,
        L,
        S,
        n_chunks,
        STORE_CK=bool(checkpoints),
        PRIOR=bool(prior),
        BLOCK_S=block_s,
        num_warps=num_warps,
        **{cell.tile_key: block_l},
    )
    return (
        y.permute(0, 2, 1).contiguous(),
        yvar.permute(0, 2, 1).contiguous(),
        lam_fin,
        eta_fin,
        msi_t,
        si_t,
        h_c,
        w_c,
        lam0_c,
        eta0_c,
        lam_ck,
        eta_ck,
    )


class _KLAScan(torch.autograd.Function):
    """Any triton forward, with the one shared triton backward behind it.

    The cell is an ordinary non-tensor argument, so all three implementations
    share this class instead of subclassing it: the backward replays from
    checkpoints and never learns which forward wrote them.
    """

    @staticmethod
    def forward(ctx, msi, si, h, w, a, q, lam0, eta0, prior, block_l, cell):
        (
            y,
            yvar,
            lam_fin,
            eta_fin,
            msi_t,
            si_t,
            h_c,
            w_c,
            lam0_c,
            eta0_c,
            lam_ck,
            eta_ck,
        ) = launch_forward(
            cell,
            msi,
            si,
            h,
            w,
            a,
            q,
            lam0,
            eta0,
            checkpoints=any(ctx.needs_input_grad),
            prior=prior,
            block_l=block_l,
        )
        ctx.prior = prior
        ctx.block_l = block_l
        ctx.save_for_backward(
            msi_t, si_t, h_c, w_c, a, q, lam0_c, eta0_c, lam_ck, eta_ck
        )
        return y, yvar, lam_fin, eta_fin

    @staticmethod
    def backward(ctx, dy, dyvar, dlam_fin, deta_fin):
        from kla.ops.kernels.triton.kla_scan_bwd import scan_backward

        msi_t, si_t, h_c, w_c, a, q, lam0_c, eta0_c, lam_ck, eta_ck = ctx.saved_tensors
        dmsi, dsi, dh, dw, da, dp, dlam0, deta0 = scan_backward(
            dy.permute(0, 2, 1).contiguous(),
            dyvar.permute(0, 2, 1).contiguous(),
            dlam_fin.contiguous(),
            deta_fin.contiguous(),
            msi_t,
            si_t,
            h_c,
            w_c,
            a,
            q,
            lam0_c,
            eta0_c,
            lam_ck,
            eta_ck,
            prior=ctx.prior,
            block_l=ctx.block_l,
        )
        return (
            dmsi.permute(0, 2, 1),
            dsi.permute(0, 2, 1),
            dh,
            dw,
            da,
            dp,
            dlam0,
            deta0,
            None,
            None,
            None,
        )


def make_scan(cell: Cell, name: str, doc: str):
    """The differentiable entry point for one cell → ``(y, y_var, lam, eta)``."""

    def scan(msi, si, h, w, a, q, lam0, eta0, prior=False, block_l=None):
        block_l = cell.default_block_l if block_l is None else block_l
        return _KLAScan.apply(msi, si, h, w, a, q, lam0, eta0, prior, block_l, cell)

    scan.__name__ = name
    scan.__qualname__ = name
    scan.__doc__ = doc
    return scan
