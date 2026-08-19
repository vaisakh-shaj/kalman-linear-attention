"""Triton backend for the KLA scan — six cells, two families, one backward each.

Fused (``triton_recurrent``, ``triton_chunk``, ``triton_pscan``)
    The sufficient statistics, both recurrences and the read-out in one kernel,
    so no ``[B, L, M, S]`` intermediate is written. Each writes the same
    ``[B, M, NCK, S]`` checkpoints at the same stride, and
    :mod:`kla.ops.kernels.triton.kla_scan_bwd` replays from them — one
    backward, whichever forward ran.

Unfused (``triton_unfused_*``)
    The same recurrences as standalone scans over already-built leaf
    coefficients, with the statistics and read-out done in torch around them
    (:mod:`kla.ops.kernels.triton.unfused_kla_scan`). The per-``(b,l,m,s)``
    elementwise costs several HBM passes — which is exactly the cost the fused
    family exists to avoid — so these are the reference the fused path is
    checked against, reaching the same numbers by a different route.

Within each family the three cells differ only in how time is walked:
``recurrent`` applies the map along a serial axis, ``chunk`` tiles time and
carries across tiles, ``pscan`` resolves the tiles with a parallel scan and
carries nothing. See ``docs/implementations.md``.

The precision recurrence λ_t = (Aλ' + B)/(Cλ' + D) is a *linear-space,
trace-normalized* Möbius scan throughout, the same formulation the CUDA and
Metal kernels use: composing the 2×2 maps is a plain matmul normalized by the
trace, which keeps entries O(1) without log-space since λ is scale-invariant.
The information vector η_t = α_t·η' + r_t is a plain affine scan. Both adjoints
are scalar, which is what makes every cell here exact.

On the unfused path the initial precision λ0 is passed into the scan and η0 is
folded into r_0 += α_0·η0. λ/η match the sequential reference to ~1e-6 and
gradients to ~3e-7. Note ``backend="auto"`` already prefers triton on CUDA.
"""

from __future__ import annotations

from typing import Optional

import torch

from kla.ops.kla_ops import (
    EPS,
    P_MIN,
    KLAState,
    _broadcast_ap,
    _sufficient_stats,
    init_state,
)


def _require_kernels():
    try:
        from kla.ops.kernels.triton.chunk_kla_scan import chunk_kla_scan
        from kla.ops.kernels.triton.pscan_kla_scan import pscan_kla_scan
        from kla.ops.kernels.triton.recurrent_kla_scan import recurrent_kla_scan
        from kla.ops.kernels.triton.unfused_kla_scan import (
            affine_scan,
            mobius_scan,
        )
    except ImportError as e:
        raise NotImplementedError(
            f"The triton KLA backend needs the triton package and a CUDA device ({e}); "
            "use backend='torch' (or 'auto')."
        ) from e
    fused = {
        "recurrent": recurrent_kla_scan,
        "chunk": chunk_kla_scan,
        "pscan": pscan_kla_scan,
    }
    return mobius_scan, affine_scan, fused


def kla_scan_triton(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
    kernel: str = "unfused_chunk",
):
    """Triton-kernel KLA scan. Same contract as :func:`kla.ops.kla_scan_torch`.

    ``kernel`` picks the implementation: ``"recurrent"`` walks time serially,
    ``"chunk"`` tiles it, ``"pscan"`` resolves the tiles with a parallel scan
    instead of a serial carry, and ``"unfused_chunk"`` is the tiled scans with
    torch glue. The first three share one backward. Each runs exactly what it
    names — nothing is selected by grad mode, so the kernels a run used are a
    function of the config and the machine and nothing else.
    """
    mobius_scan, affine_scan, fused = _require_kernels()
    if not v.is_cuda:
        raise NotImplementedError("The triton KLA backend requires CUDA tensors.")
    unfused = {f"unfused_{s}": s for s in fused}
    if kernel not in (*fused, *unfused):
        raise ValueError(
            f"Unknown triton kernel {kernel!r}; expected one of "
            f"{', '.join((*fused, *unfused))}"
        )

    v = v.float()
    lambda_v = lambda_v.float()
    k = k.float()
    q = q.float()

    B, L, M = v.shape
    S = k.shape[2]
    if initial_state is None:
        initial_state = init_state(B, M, S, device=v.device)
    lam0 = initial_state.lam.float()
    eta0 = initial_state.eta.float()

    if kernel in fused:
        # No [B,L,M,S] HBM round-trips. These consume the folded information
        # mean v·Λ^v (pre-folded here, so autograd splits d(v·Λ^v) back into dv
        # and d(Λ^v)), and p is floored here to match _broadcast_ap on the
        # unfused path below — the kernels have no internal guard against a
        # non-positive p, and flooring in torch gives the floor's subgradient to
        # torch as well.
        y, y_var, lam_fin, eta_fin = fused[kernel](
            v * lambda_v,
            lambda_v,
            k,
            q,
            a.float(),
            p.float().clamp_min(P_MIN),
            lam0,
            eta0,
            decode_from_prior,
        )
        return y, y_var, KLAState(lam=lam_fin, eta=eta_fin)

    phi, r = _sufficient_stats(v, lambda_v, k)
    a_, p_ = _broadcast_ap(a.float(), p.float(), phi)
    a2 = (a_ * a_).clamp_min(EPS)

    # Precision scan λ_t via the trace-normalized linear-space Möbius recurrence
    # λ_t = (A·λ' + B)/(C·λ' + D), with leaf A=(1+pφ)/a², B=φ, C=p/a², D=1.
    # Differentiable, and the schedule reaches the forward only: the adjoint
    # reads the values λ and λ_{t-1}, not the order they were produced in.
    A_lin = ((1.0 + p_ * phi) / a2).expand_as(phi).contiguous()
    C_lin = (p_ / a2).expand_as(phi).contiguous()
    schedule = unfused[kernel]
    lam = mobius_scan(
        A_lin, phi.contiguous(), C_lin, torch.ones_like(phi), lam0, schedule
    )
    var = 1.0 / lam.clamp_min(EPS)

    # Gain α_t needs λ_{t-1}
    lam_prev = torch.cat((lam0.unsqueeze(1), lam[:, :-1]), dim=1)
    denom = (a2 + p_ * lam_prev).clamp_min(EPS)
    alpha = a_ / denom

    # Fold η0 into the first step's input, then scan the affine recurrence (η_{-1}=0).
    r = torch.cat((r[:, :1] + alpha[:, :1] * eta0.unsqueeze(1), r[:, 1:]), dim=1)
    eta = affine_scan(alpha.contiguous(), r.contiguous(), schedule)

    mean = eta * var
    if decode_from_prior:
        mean = a_ * mean
        var = a2 * var + p_

    y = torch.einsum("blms,bls->blm", mean, q)
    y_var = torch.einsum("blms,bls->blm", var, q * q)
    return y, y_var, KLAState(lam=lam[:, -1], eta=eta[:, -1])
