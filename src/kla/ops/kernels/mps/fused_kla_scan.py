"""Single fused KLA scan on Metal: sufficient statistics, both recurrences and
the read-out in one kernel each way, so no ``[B, L, M, S]`` intermediate is ever
written. The backward is the exact adjoint and the filter state is
differentiable in both directions.

``d_state`` is capped at :data:`~kla.ops.kernels.mps._shaders.MAX_DSTATE`. The
replay scheme, reduction layout and atomics are described in the header of
``fused_kla_scan.metal``.
"""

from __future__ import annotations

import torch

from kla.ops.kernels.mps._shaders import (
    DEFAULT_CHUNK,
    check_inputs,
    fused_library,
    launch_geometry,
)


def _grid(B: int, M: int, S: int):
    """``(threads, group_size)`` for one problem shape.

    The grid is ``(BLOCK_S, M, B)`` with the channel axis rounded up to whole
    threadgroups: the kernels reduce across a group, so the padding threads have
    to exist and take part rather than be dispatched away.
    """
    block_s, rows = launch_geometry(S)
    m_padded = -(-M // rows) * rows
    return (block_s, m_padded, B), (block_s, rows, 1)


def fused_forward(
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
    chunk: int = DEFAULT_CHUNK,
):
    """Fused forward. Returns ``(y, y_var, lam_fin, eta_fin, lam_ck, eta_ck)``.

    ``y`` / ``y_var`` are ``[B, L, M]``; the state tensors are ``[B, M, S]``.
    With ``checkpoints=False`` the two checkpoint tensors are one-element
    placeholders — the kernel takes the flag and never writes them. ``prior``
    is ``decode_from_prior``: it moves the read-out one predict step ahead.
    """
    check_inputs(msi, si, k, q, a, p, lam0, eta0)
    B, L, M = msi.shape
    S = k.shape[2]
    lib = fused_library(S, chunk)

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
    lib.kla_fused_fwd(
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


def fused_backward(
    dy,
    dyvar,
    dlam_fin,
    deta_fin,
    msi,
    si,
    k,
    q,
    a,
    p,
    lam_ck,
    eta_ck,
    prior: bool = False,
    chunk: int = DEFAULT_CHUNK,
):
    """Fused backward. Returns ``(dmsi, dsi, dk, dq, da, dp, dlam0, deta0)``."""
    B, L, M = msi.shape
    S = k.shape[2]
    lib = fused_library(S, chunk)
    n_ck = lam_ck.shape[2]

    dev = msi.device
    f32 = torch.float32
    dmsi = torch.empty(B, L, M, device=dev, dtype=f32)
    dsi = torch.empty(B, L, M, device=dev, dtype=f32)
    dlam0 = torch.empty(B, M, S, device=dev, dtype=f32)
    deta0 = torch.empty(B, M, S, device=dev, dtype=f32)
    # dk/dq contract over channels and da/dp over batch and time; neither axis
    # is owned by a single threadgroup, so these four are accumulated atomically
    # and must start at zero.
    dk = torch.zeros(B, L, S, device=dev, dtype=f32)
    dq = torch.zeros(B, L, S, device=dev, dtype=f32)
    da = torch.zeros(M, S, device=dev, dtype=f32)
    dp = torch.zeros(M, S, device=dev, dtype=f32)

    threads, group = _grid(B, M, S)
    lib.kla_fused_bwd(
        dk,
        dq,
        da,
        dp,
        dmsi,
        dsi,
        dlam0,
        deta0,
        dy,
        dyvar,
        dlam_fin,
        deta_fin,
        msi,
        si,
        k,
        q,
        a,
        p,
        lam_ck,
        eta_ck,
        L,
        M,
        S,
        n_ck,
        int(prior),
        threads=threads,
        group_size=group,
    )
    return dmsi, dsi, dk, dq, da, dp, dlam0, deta0


class _FusedKLAScan(torch.autograd.Function):
    """Autograd wrapper over the fused forward/backward Metal kernels."""

    @staticmethod
    def forward(ctx, msi, si, k, q, a, p, lam0, eta0, prior, chunk):
        needs_grad = any(ctx.needs_input_grad)
        y, yvar, lam_fin, eta_fin, lam_ck, eta_ck = fused_forward(
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
            chunk=chunk,
        )
        ctx.chunk = chunk
        ctx.prior = prior
        ctx.save_for_backward(msi, si, k, q, a, p, lam_ck, eta_ck)
        return y, yvar, lam_fin, eta_fin

    @staticmethod
    def backward(ctx, dy, dyvar, dlam_fin, deta_fin):
        msi, si, k, q, a, p, lam_ck, eta_ck = ctx.saved_tensors
        dmsi, dsi, dk, dq, da, dp, dlam0, deta0 = fused_backward(
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
        return dmsi, dsi, dk, dq, da, dp, dlam0, deta0, None, None


def fused_kla_scan(
    msi, si, k, q, a, p, lam0, eta0, prior: bool = False, chunk: int = DEFAULT_CHUNK
):
    """Differentiable fused KLA scan → ``(y, y_var, lam_fin, eta_fin)``."""
    return _FusedKLAScan.apply(msi, si, k, q, a, p, lam0, eta0, prior, chunk)
