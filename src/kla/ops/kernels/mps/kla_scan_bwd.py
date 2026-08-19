"""The Metal backward — one exact adjoint for both MPS implementations.

``mps_recurrent`` and ``mps_chunk`` write the same ``[B, M, NCK, S]``
checkpoints at the same stride, so they share this. The kernel is lane-per-state
whatever geometry the forward used: the reverse walk is over the serial state
lanes either way, which is why composing the map in a forward buys the backward
nothing and costs it nothing. See the header of ``kla_scan_bwd.metal``.
"""

from __future__ import annotations

import torch

from kla.ops.kernels.mps._shaders import DEFAULT_CHUNK, bwd_library, launch_geometry


def _grid(B: int, M: int, S: int):
    """``(threads, group_size)`` for one problem shape.

    The grid is ``(BLOCK_S, M, B)`` with the channel axis rounded up to whole
    threadgroups: the kernel reduces across a group, so the padding threads have
    to exist and take part rather than be dispatched away.
    """
    block_s, rows = launch_geometry(S)
    m_padded = -(-M // rows) * rows
    return (block_s, m_padded, B), (block_s, rows, 1)


def scan_backward(
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
    """The shared backward. Returns ``(dmsi, dsi, dk, dq, da, dp, dlam0, deta0)``."""
    B, L, M = msi.shape
    S = k.shape[2]
    lib = bwd_library(S, chunk)
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
    lib.kla_scan_bwd(
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
