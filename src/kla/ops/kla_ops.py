"""Functional KLA scan ops and the backend dispatcher.

The core operation is a parallel Kalman filter in information form, factorized
per (channel m, state s) pair. Inputs are in paper notation: value ``v``, value
precision ``Λ^v`` (= 1/σ²_v), observation map / key ``k``, readout / query ``q``,
discrete decay ``a`` and process noise ``p``. The value enters the filter only
through the natural-parameter product ``v·Λ^v`` (formed here, in the ops):

predict:  λ⁻_t = λ_{t-1} / (a² + p·λ_{t-1})            (precision)
update:   λ_t  = λ⁻_t + φ_t,        φ_t = Λ^v_t · k²_t        (outer product m⊗s)
gain:     α_t  = a / (a² + p·λ_{t-1})
info vec: η_t  = α_t·η_{t-1} + r_t,  r_t = (v_t·Λ^v_t) · k_t
output:   mean_t = η_t/λ_t, var_t = 1/λ_t
readout:  y_t[m] = Σ_s q_t[s]·mean_t[m,s],  yvar_t[m] = Σ_s q²_t[s]·var_t[m,s]

The precision recurrence is a Möbius (linear-fractional) map, composed with a
parallel associative scan; the information vector is a first-order affine
recurrence, also scanned in parallel. The Möbius maps are composed in linear
space with trace normalization by default (``mobius_impl="linear"``, matching
both GPU backends); ``mobius_impl="log"`` selects the older log-space combine,
which is slower but has more exponent headroom. See :func:`kla_scan_torch`.

Shapes (B batch, L seq, M channels, S states):
    v, lambda_v : [B, L, M]   value and value precision Λ^v
    k           : [B, L, S]   observation map (key)
    q           : [B, L, S]   readout (query)
    a, p        : [M, S]      discrete decay and process noise (time-invariant)
    state (λ, η): [B, M, S] each

Everything here runs in float32 regardless of input dtype; the layer casts
back to the working dtype.
"""

from __future__ import annotations

import functools
from typing import Callable, NamedTuple, Optional

import torch

from kla.ops.scan import resolve_scan

EPS = 1e-12
P_MIN = 1e-12  # floor on the process noise p


class KLAState(NamedTuple):
    """Recurrent state of the filter: posterior precision and information vector."""

    lam: torch.Tensor  # [B, M, S] posterior precision Λ_t
    eta: torch.Tensor  # [B, M, S] information vector η_t = Λ_t·mean_t


def init_state(
    batch: int, d_model: int, d_state: int, device=None, dtype=torch.float32
) -> KLAState:
    """Standard-normal prior: unit covariance (precision 1), zero mean."""
    return KLAState(
        lam=torch.ones(batch, d_model, d_state, device=device, dtype=dtype),
        eta=torch.zeros(batch, d_model, d_state, device=device, dtype=dtype),
    )


def _mobius_combine_log(left, right):
    """Compose two 2x2 Möbius matrices stored as (logA, logB, logC, logD)."""
    a1, b1, c1, d1 = left
    a2, b2, c2, d2 = right
    return (
        torch.logaddexp(a2 + a1, b2 + c1),
        torch.logaddexp(a2 + b1, b2 + d1),
        torch.logaddexp(c2 + a1, d2 + c1),
        torch.logaddexp(c2 + b1, d2 + d1),
    )


def _mobius_combine_tracenorm(left, right):
    """Compose two 2x2 Möbius matrices in LINEAR space, normalized by the trace.

    The plain-matmul counterpart of :func:`_mobius_combine_log`, and the same
    scheme the triton kernels use (``unfused_kla_scan._tracenorm_combine``).
    Composing the maps is an ordinary 2x2 product; dividing all four entries by
    the trace afterwards keeps them O(1) without ever leaving linear space.

    Rescaling is free because λ is invariant under it: (A,B,C,D) and
    (kA,kB,kC,kD) define the same map λ ↦ (Aλ+B)/(Cλ+D). So the normalizer
    cancels in the read-out and only buys numerical range.

    The trace can never vanish *for this matrix family*, which is what makes the
    scheme safe here rather than merely convenient. Every leaf is entrywise
    positive -- A=(1+pφ)/a² > 0, B=φ > 0, C=p/a² > 0, D=1 > 0, guaranteed by the
    clamps on p, φ and a² -- and entrywise-positive 2x2 matrices are closed under
    multiplication, so every node of the scan tree is positive too. Hence
    A+D > 0 always, whatever order the associative scan happens to combine in.
    The ``clamp_min`` below is belt-and-braces, not load-bearing.

    What normalization bounds, precisely: the *diagonal*, since A+D=1 with both
    positive forces A,D ∈ (0,1). B and C are off-diagonal and are NOT bounded by
    1 -- they routinely exceed it. What holds for them is a product bound: the
    leaf determinant is AD-BC = (1+pφ)/a² - φp/a² = 1/a² > 0, determinants are
    multiplicative, and normalizing divides the determinant by (A+D)² > 0, so
    det > 0 survives every composition. With A+D=1 we have AD ≤ 1/4, hence
        B·C = AD - det < 1/4,
    so B and C cannot both be large -- one grows only as the other shrinks. In
    practice the composed map converges to a rank-1 projector (det → 0, which is
    just the filter forgetting its initial condition), and B·C approaches 1/4
    from below while every entry stays O(1). Empirically the entries reach a
    fixed point rather than drifting; see tests/test_mobius_impl.py.
    """
    a1, b1, c1, d1 = left
    a2, b2, c2, d2 = right
    a = a2 * a1 + b2 * c1
    b = a2 * b1 + b2 * d1
    c = c2 * a1 + d2 * c1
    d = c2 * b1 + d2 * d1
    inv = 1.0 / (a + d).clamp_min(EPS)
    return a * inv, b * inv, c * inv, d * inv


def _affine_combine(left, right):
    """Compose two affine maps x ↦ a·x + b stored as (a, b)."""
    a1, b1 = left
    a2, b2 = right
    return (a2 * a1, a2 * b1 + b2)


def _compute_dtype(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Widen low-precision inputs to float32; pass float32/float64 through.

    The scan is numerically delicate, so bf16/fp16 activations are always
    widened. float64 is *preserved* rather than downcast, which is what lets
    :func:`torch.autograd.gradcheck` run against these ops (see
    ``tests/test_gradcheck.py``) — the GPU backends are float32-only and cannot
    be gradchecked.
    """
    return tuple(
        t if t.dtype in (torch.float32, torch.float64) else t.float() for t in tensors
    )


def _state_dtype(t: torch.Tensor) -> torch.dtype:
    """dtype the filter state should carry alongside an input tensor ``t``."""
    return t.dtype if t.dtype in (torch.float32, torch.float64) else torch.float32


def _sufficient_stats(v, lambda_v, k, floor: bool = False):
    """Value/precision/key → per-(m, s) statistics φ and r (outer product m⊗s).

    Information form: φ = Λ^v ⊗ k², r = (v·Λ^v) ⊗ k. The value ``v`` is folded
    into its natural parameter ``v·Λ^v`` here, so raw ``v`` goes no further.

    ``floor`` clamps φ up to EPS. It defaults to OFF; the CUDA kernels apply it
    unconditionally instead::

        kla_matmul_scan_ops.cuh    phi = fmaxf(raw_phi, KLA_EPS);
        kla_matmul_bwd_kernel.cuh  float phi_mask = (raw_phi[i] > KLA_EPS) ...

    Note it is two things there, a forward floor *and* a backward gradient mask
    -- and ``clamp_min`` reproduces both, since its subgradient is zero wherever
    it binds. So the floor silently kills gradients into Λ^v and k at every
    (b,l,m,s) where the key is near zero.

    Dropping it is safe for the linear/trace-normalized Möbius scan (the
    default), where φ appears only as the leaf B and inside A=(1+pφ)/a², never
    as log φ. At φ=0 the leaf is [[A,0],[C,1]]: still non-negative, trace still
    positive, and λ_t = Aλ'/(Cλ'+1) is exactly the right pure-prediction step
    for an observation carrying no information. The clamp instead injects 1e-12
    of spurious information *and* drops the gradient.

    It is NOT safe for ``mobius_impl="log"``, which takes ``phi.log()``, so
    :func:`kla_scan_torch` turns it back on there.

    The fused triton forward keeps its own internal ``tl.maximum(.., 1e-12)``.
    That path is no-grad, so the masking cannot bite; only the 1e-12 forward
    difference remains, far below every parity tolerance.
    """
    phi = lambda_v.unsqueeze(-1) * (k * k).unsqueeze(-2)  # [B,L,M,1]·[B,L,1,S]
    r = (v * lambda_v).unsqueeze(-1) * k.unsqueeze(-2)
    return (phi.clamp_min(EPS) if floor else phi), r


def _broadcast_ap(a, p, like):
    """View the [M, S] coefficients as [1, 1, M, S] for broadcasting over B, L."""
    a = a.view(1, 1, *a.shape)
    p = p.view(1, 1, *p.shape)
    return a.to(like), p.clamp_min(P_MIN).to(like)


def _recurrent_lambda_eta(phi, r, a, p, lam0, eta0):
    """Both recurrences in one carry-based pass — *applied*, never composed.

    The recurrent implementation proper, and the only torch path that matches what
    ``recurrent`` means in the kernels::

        den_t = a² + p·λ_{t-1};  λ_t = λ_{t-1}/den_t + φ_t;
        η_t   = (a/den_t)·η_{t-1} + r_t

    ``torch._higher_order_ops.scan`` carries ``(λ, η)`` along the sequence, so
    there is no 2×2 matrix, no trace normalization and no prefix-product tensor
    — about a quarter of the arithmetic of composing, and ``mobius_impl`` has
    nothing to represent because nothing is composed. It also fuses what the
    composing path has to do in two passes: the gain α_t reads λ_{t-1}, which a
    carry already has in hand and an associative scan has to recover afterwards.

    ``a`` and ``p`` are read from the closure. The HOP lifts them as additional
    inputs and routes gradients back into them, which is what lets the whole
    recurrence stay one graph node.
    """
    from torch._higher_order_ops.scan import scan as _scan

    a2 = (a * a).clamp_min(EPS)

    def step(carry, x):
        lam_prev, eta_prev = carry
        phi_t, r_t = x
        den = (a2 + p * lam_prev).clamp_min(EPS)
        lam = lam_prev / den + phi_t
        eta = (a / den) * eta_prev + r_t
        # The HOP forbids the carry and the stacked output aliasing.
        return (lam, eta), (lam.clone(), eta.clone())

    def dense(x):
        # Canonical strides, not just `.contiguous()`: a size-1 dimension makes
        # any stride contiguous as far as torch is concerned, and the HOP
        # compares the carry's metadata literally.
        return torch.empty_like(x, memory_format=torch.contiguous_format).copy_(x)

    (lam_fin, eta_fin), (lam, eta) = _scan(
        step, (dense(lam0), dense(eta0)), (phi.contiguous(), r.contiguous()), dim=1
    )
    return lam, eta, lam_fin, eta_fin


def kla_scan_torch(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    initial_state: Optional[KLAState] = None,
    scan_impl: str = "auto",
    decode_from_prior: bool = False,
    mobius_impl: str = "linear",
):
    """Parallel-scan torch implementation. Returns (y, y_var, final_state).

    ``scan_impl`` picks *how* the scan is parallelized (see
    :func:`kla.ops.scan.resolve_scan`); ``mobius_impl`` picks how the precision
    map is *represented* while it is composed. The two are orthogonal, except
    that ``scan_impl="sequential"`` composes nothing — it takes the recurrent
    path in :func:`_recurrent_lambda_eta` and ignores ``mobius_impl``.

    ``mobius_impl="linear"`` (default) composes the 2x2 maps as plain matmuls
    normalized by the trace -- no transcendentals in the combine, and the same
    scheme both GPU backends use, so torch is now a faithful reference for them
    rather than a second opinion computed a different way.

    ``mobius_impl="log"`` keeps the entries as logs and combines with
    ``logaddexp``. It is kept as a reference implementation of the same map,
    showing how else the composition can be done -- not as a fallback, since
    trace normalization is stable on its own. The one place the two differ at
    all is extreme decay: after trace-normalization D ≈ a²/(1+pφ), which
    underflows float32 once ā drops below ~1e-19 (Δ·|a| ≳ 44), and the layer
    inits |a| = 1 so reaching that would take |a| ≳ 440 at Δ=0.1. float64 has
    ~10x the exponent range, so it is a float32-only consideration either way
    and gradcheck is unaffected.
    """
    v, lambda_v, k, q = _compute_dtype(v, lambda_v, k, q)
    dtype = v.dtype

    B, L, M = v.shape
    S = k.shape[2]
    if initial_state is None:
        initial_state = init_state(B, M, S, device=v.device, dtype=dtype)
    lam0 = initial_state.lam.to(dtype)
    eta0 = initial_state.eta.to(dtype)

    if scan_impl == "auto" and not torch.is_grad_enabled():
        # associative_scan torch.compiles per sequence length, which thrashes the
        # dynamo cache under varying-length inference; doubling is compile-free.
        scan_impl = "doubling"
    if scan_impl == "sequential":
        # The recurrent implementation applies the map instead of composing it, so
        # there is no combine to hand a generic scan and mobius_impl has
        # nothing to represent. It also needs no phi floor: nothing takes a log.
        phi, r = _sufficient_stats(v, lambda_v, k)
        a_s = a.to(dtype)
        p_s = p.clamp_min(P_MIN).to(dtype)
        lam, eta, lam_fin, eta_fin = _recurrent_lambda_eta(phi, r, a_s, p_s, lam0, eta0)
        var = 1.0 / lam.clamp_min(EPS)
        mean = eta * var
        if decode_from_prior:
            a2_s = (a_s * a_s).clamp_min(EPS)
            mean = a_s * mean
            var = a2_s * var + p_s
        y = torch.einsum("blms,bls->blm", mean, q)
        y_var = torch.einsum("blms,bls->blm", var, q * q)
        return y, y_var, KLAState(lam=lam_fin, eta=eta_fin)

    scan = resolve_scan(scan_impl)
    # The log-space combine takes phi.log(), so it needs the floor; the linear
    # one does not (see _sufficient_stats).
    phi, r = _sufficient_stats(v, lambda_v, k, floor=(mobius_impl == "log"))
    a_, p_ = _broadcast_ap(a, p, phi)
    a2 = (a_ * a_).clamp_min(EPS)

    # Precision Möbius scan: λ_t = (Aλ' + B) / (Cλ' + D), with leaf
    # A=(1+pφ)/a², B=φ, C=p/a², D=1. Two representations, same map.
    if mobius_impl == "linear":
        A = ((1.0 + p_ * phi) / a2).expand_as(phi)
        C = (p_ / a2).expand_as(phi)
        D = torch.ones_like(phi)
        pA, pB, pC, pD = scan(_mobius_combine_tracenorm, (A, phi, C, D), dim=1)
        lam0_ = lam0.unsqueeze(1)
        lam = (pA * lam0_ + pB) / (pC * lam0_ + pD).clamp_min(EPS)
        var = 1.0 / lam.clamp_min(EPS)
    elif mobius_impl == "log":
        log_a2 = a2.log()
        logA = (torch.log1p(p_ * phi) - log_a2).expand_as(phi)
        logB = phi.log()
        logC = (p_.log() - log_a2).expand_as(phi)
        logD = torch.zeros_like(phi)

        pA, pB, pC, pD = scan(_mobius_combine_log, (logA, logB, logC, logD), dim=1)

        log_lam0 = lam0.clamp_min(EPS).log().unsqueeze(1)
        log_lam = torch.logaddexp(pA + log_lam0, pB) - torch.logaddexp(
            pC + log_lam0, pD
        )
        lam = log_lam.exp()  # posterior precision [B, L, M, S]
        # exp(-log λ) rather than 1/λ: one fewer roundtrip through the exponent,
        # and it stays finite where λ itself would overflow.
        var = (-log_lam).exp()
    else:
        raise ValueError(
            f"Unknown mobius_impl {mobius_impl!r}; expected 'linear' or 'log'"
        )

    # Information-vector affine scan; the gain α_t depends on λ_{t-1}.
    lam_prev = torch.cat((lam0.unsqueeze(1), lam[:, :-1]), dim=1)
    denom = (a2 + p_ * lam_prev).clamp_min(EPS)
    alpha = a_ / denom

    pAlpha, pR = scan(_affine_combine, (alpha, r), dim=1)
    eta = pAlpha * eta0.unsqueeze(1) + pR

    mean = eta * var

    if decode_from_prior:
        mean = a_ * mean
        var = a2 * var + p_

    y = torch.einsum("blms,bls->blm", mean, q)
    y_var = torch.einsum("blms,bls->blm", var, q * q)
    return y, y_var, KLAState(lam=lam[:, -1], eta=eta[:, -1])


def kla_step(
    v: torch.Tensor,  # [B, M]
    lambda_v: torch.Tensor,  # [B, M]
    k: torch.Tensor,  # [B, S]
    q: torch.Tensor,  # [B, S] (readout / query)
    a: torch.Tensor,  # [M, S] discrete decay
    p: torch.Tensor,  # [M, S] process noise
    state: KLAState,
    decode_from_prior: bool = False,
):
    """Single recurrent filter step for autoregressive decode.

    Returns (y [B, M], y_var [B, M], new_state).
    """
    v, lambda_v, k, q = _compute_dtype(v, lambda_v, k, q)
    dtype = v.dtype

    phi = (lambda_v.unsqueeze(-1) * (k * k).unsqueeze(-2)).clamp_min(
        EPS
    )  # [B,M,1]·[B,1,S]
    r = (v * lambda_v).unsqueeze(-1) * k.unsqueeze(-2)

    a_ = a.to(dtype).unsqueeze(0)  # [1, M, S], broadcasts over the batch
    p_ = p.to(dtype).unsqueeze(0).clamp_min(P_MIN)
    a2 = (a_ * a_).clamp_min(EPS)

    denom = (a2 + p_ * state.lam).clamp_min(EPS)
    lam = state.lam / denom + phi  # λ⁻ + φ
    eta = (a_ / denom) * state.eta + r

    var = 1.0 / lam.clamp_min(EPS)
    mean = eta * var
    if decode_from_prior:
        mean = a_ * mean
        var = a2 * var + p_

    y = torch.einsum("bms,bs->bm", mean, q)
    y_var = torch.einsum("bms,bs->bm", var, q * q)
    return y, y_var, KLAState(lam=lam, eta=eta)


def kla_scan_reference(
    v,
    lambda_v,
    k,
    q,
    a,
    p,
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
):
    """Sequential reference: applies :func:`kla_step` along the sequence."""
    B, L, M = v.shape
    S = k.shape[2]
    state = initial_state or init_state(B, M, S, device=v.device, dtype=_state_dtype(v))
    ys, yvars = [], []
    for t in range(L):
        y_t, yvar_t, state = kla_step(
            v[:, t],
            lambda_v[:, t],
            k[:, t],
            q[:, t],
            a,
            p,
            state,
            decode_from_prior=decode_from_prior,
        )
        ys.append(y_t)
        yvars.append(yvar_t)
    return torch.stack(ys, dim=1), torch.stack(yvars, dim=1), state


# --------------------------------------------------------------- the registry
#
# One entry per implementation, named "<backend>[_unfused]_<implementation>" (see
# docs/implementations.md). "fused" is the default and carries no token; a bare
# backend name aliases that backend's default implementation, which is "chunk".
#
# The record carries only what the dispatcher and `python -m kla` actually read.
# Everything in the contract -- forward, exact backward, state carry, prior
# decode, fp32 -- is required of every implementation, so it is not a per-cell
# flag. `exact_bwd` is the one exception: the two prior CUDA kernels are kept
# precisely because they violate it, as the comparison for the exact ones.


class Impl(NamedTuple):
    """One scan implementation: where it compiles and how it walks time."""

    backend: str  # torch | triton | cuda | mps
    implementation: str  # recurrent | chunk | pscan
    fused: bool
    max_d_state: Optional[int]  # None = no ceiling
    exact_bwd: bool
    fn: Callable


def _torch_scan(scan_impl: str) -> Callable:
    """The torch backend at one implementation. `scan_impl` is now internal to it."""

    def run(*args, mobius_impl="linear", **kwargs):
        return kla_scan_torch(
            *args, scan_impl=scan_impl, mobius_impl=mobius_impl, **kwargs
        )

    return run


def _triton_scan(kernel: str) -> Callable:
    def run(*args, **kwargs):
        from kla.ops.triton_backend import kla_scan_triton

        return kla_scan_triton(*args, kernel=kernel, **kwargs)

    return run


def _cuda_scan(kernel_version: str) -> Callable:
    """One of the prior v2_* kernels, kept as the approximate-backward baseline."""

    def run(*args, **kwargs):
        from kla.ops.cuda_backend import kla_scan_cuda

        return kla_scan_cuda(*args, kernel_version=kernel_version, **kwargs)

    return run


def _cuda_exact(name: str) -> Callable:
    def run(*args, **kwargs):
        import kla.ops.cuda_backend as cuda

        return getattr(cuda, name)(*args, **kwargs)

    return run


def _mps_scan(name: str) -> Callable:
    def run(*args, **kwargs):
        import kla.ops.mps_backend as mps

        return getattr(mps, name)(*args, **kwargs)

    return run


_BACKENDS: dict[str, Impl] = {
    # torch -- portable reference, the only one that runs float64
    "torch_unfused_recurrent": Impl(
        "torch", "recurrent", False, None, True, _torch_scan("sequential")
    ),
    "torch_unfused_chunk": Impl(
        "torch", "chunk", False, None, True, _torch_scan("chunk")
    ),
    "torch_unfused_pscan": Impl(
        "torch", "pscan", False, None, True, _torch_scan("auto")
    ),
    # triton
    "triton_recurrent": Impl(
        "triton", "recurrent", True, None, True, _triton_scan("recurrent")
    ),
    "triton_chunk": Impl("triton", "chunk", True, None, True, _triton_scan("chunk")),
    "triton_pscan": Impl("triton", "pscan", True, None, True, _triton_scan("pscan")),
    "triton_unfused_recurrent": Impl(
        "triton", "recurrent", False, None, True, _triton_scan("unfused_recurrent")
    ),
    "triton_unfused_chunk": Impl(
        "triton", "chunk", False, None, True, _triton_scan("unfused_chunk")
    ),
    "triton_unfused_pscan": Impl(
        "triton", "pscan", False, None, True, _triton_scan("unfused_pscan")
    ),
    # cuda
    "cuda_recurrent": Impl(
        "cuda", "recurrent", True, 64, True, _cuda_exact("kla_scan_cuda_recurrent")
    ),
    "cuda_chunk": Impl(
        "cuda", "chunk", True, 64, True, _cuda_exact("kla_scan_cuda_chunk")
    ),
    "cuda_pscan": Impl(
        "cuda", "pscan", True, 64, True, _cuda_exact("kla_scan_cuda_pscan")
    ),
    # prior kernels, kept as the approximate-backward comparison
    "cuda_v2_2": Impl("cuda", "chunk", True, 64, False, _cuda_scan("v2_2")),
    "cuda_v2_1": Impl("cuda", "chunk", True, 64, False, _cuda_scan("v2_1")),
    # mps
    "mps_recurrent": Impl(
        "mps", "recurrent", True, 128, True, _mps_scan("kla_scan_mps_recurrent")
    ),
    "mps_chunk": Impl("mps", "chunk", True, 128, True, _mps_scan("kla_scan_mps_chunk")),
    "mps_pscan": Impl("mps", "pscan", True, 128, True, _mps_scan("kla_scan_mps_pscan")),
}

# A bare backend name is that backend's default implementation. `chunk` was the
# placeholder everywhere; torch and mps have been measured since (see
# docs/benchmarks/mps.md) and both moved to `recurrent`, which won every shape
# either is realistically used at. triton and cuda are still unmeasured.
_ALIASES = {
    "torch": "torch_unfused_recurrent",
    "triton": "triton_chunk",
    "cuda": "cuda_chunk",
    "mps": "mps_recurrent",
}


@functools.cache
def _triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


@functools.cache
def _mps_available() -> bool:
    try:
        from kla.ops.kernels.mps import is_available
    except Exception:
        return False
    return is_available()


# Device -> the backend "auto" picks there, if its kernels are importable. The
# `cuda` kernels are deliberately absent: their backward is an approximate
# adjoint, so they stay opt-in until the exact cells land.
_AUTO = (
    ("is_cuda", "triton", _triton_available),
    ("is_mps", "mps", _mps_available),
)


def resolve_impl(name: str, x: Optional[torch.Tensor] = None) -> str:
    """Resolve ``"auto"`` and bare backend names to one implementation name."""
    if name == "auto":
        if x is None:
            return _ALIASES["torch"]
        for attr, backend, available in _AUTO:
            if getattr(x, attr) and available():
                return _ALIASES[backend]
        return _ALIASES["torch"]
    return _ALIASES.get(name, name)


def kla_scan(
    v: torch.Tensor,
    lambda_v: torch.Tensor,
    k: torch.Tensor,
    q: torch.Tensor,
    a: torch.Tensor,
    p: torch.Tensor,
    *,
    initial_state: Optional[KLAState] = None,
    backend: str = "auto",
    decode_from_prior: bool = False,
    mobius_impl: str = "linear",
):
    """Dispatch the KLA scan to the requested implementation.

    Inputs in paper notation: value ``v``, value precision ``lambda_v`` (Λ^v),
    key ``k``, query ``q``, discrete decay ``a``, process noise ``p``.

    ``backend`` takes an implementation name ("mps_recurrent"), a bare backend
    name for that backend's default ("mps"), or "auto". "auto" is the only value
    that is not a fixed implementation: it reads the device and nothing else, so
    the kernels a run used are a function of the config and the machine and
    nothing else. See docs/implementations.md for the naming scheme.
    """
    name = resolve_impl(backend, v)
    try:
        impl = _BACKENDS[name]
    except KeyError:
        raise ValueError(
            f"Unknown backend {backend!r}; expected 'auto', a backend name "
            f"({', '.join(_ALIASES)}), or one of {sorted(_BACKENDS)}"
        ) from None

    S = k.shape[2]
    if impl.max_d_state is not None and S > impl.max_d_state:
        raise NotImplementedError(
            f"{name} supports d_state <= {impl.max_d_state} (got {S}); "
            "use backend='torch', which has no ceiling."
        )

    kwargs = {}
    if impl.backend == "torch":
        # mobius_impl is a torch-only knob: the GPU kernels are hard-wired to
        # the linear, trace-normalized composition.
        kwargs["mobius_impl"] = mobius_impl
    return impl.fn(
        v,
        lambda_v,
        k,
        q,
        a,
        p,
        initial_state=initial_state,
        decode_from_prior=decode_from_prior,
        **kwargs,
    )


def backend_names() -> tuple:
    """Every implementation name :func:`kla_scan` accepts, in dispatch order.

    Read off the registry itself, so it cannot drift from what ``backend=``
    will take. Bare backend names (see :data:`_ALIASES`) are accepted too.
    """
    return tuple(_BACKENDS)


def implementations() -> dict:
    """The registry: implementation name -> :class:`Impl`."""
    return dict(_BACKENDS)


def backend_aliases() -> dict:
    """Bare backend name -> the implementation it currently resolves to."""
    return dict(_ALIASES)


def default_device() -> str:
    """The accelerator ``python -m kla`` and ``resolve_backend`` assume."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_backend(device=None) -> str:
    """The implementation ``backend="auto"`` resolves to on ``device``."""
    if device is None:
        device = default_device()
    return resolve_impl("auto", torch.empty(0, device=device))
