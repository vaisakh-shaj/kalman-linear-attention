"""Cross-backend parity: every available backend against the sequential reference.

The backends implement the same math with very different numerics, so each gets
its own tolerance profile rather than one shared threshold:

Implementations are named ``<backend>[_unfused]_<schedule>`` — see
``docs/implementations.md``.

* ``torch_unfused_*`` — the reference. Forward and backward both tight.
* ``triton_*`` — hand-written kernels with an exact reverse-scan adjoint. Tight.
* ``cuda_v2_*`` — the prior CUDA kernels. Their forward is bit-exact, but the
  trace-normalized Möbius **backward is not the exact adjoint** (see the parity
  notes in :mod:`kla.ops.cuda_backend`): gradients that flow through the
  precision scan are only accurate to ~5-15 % relative, while the ones that flow
  through the information-vector / read-out path are exact.
* ``mps_*`` — the Metal kernels (:mod:`kla.ops.mps_backend`). Tight *despite*
  being fully fused: applying the Möbius map per step instead of composing it
  makes the adjoint elementary, so they have none of the CUDA kernels' gradient
  looseness even though they are the same fully-fused shape.

The ``cuda_v2_*`` point is why :data:`PROFILES` splits the gradient inputs into
``exact_grads`` and ``loose_grads``. Asserting a tight threshold on the loose
group would fail on a *correct* build — the looseness is a documented property
of that backward, not a bug to be caught here. Those two are the only
implementations allowed it; everything else must be exact.

Backends that cannot run here raise :class:`NotImplementedError` from the
dispatcher and are skipped, so this file is meaningful on a CPU-only box, on a
GPU node with or without a CUDA toolkit, and on a Mac.
"""

import dataclasses
import math
import textwrap

import pytest
import torch

from kla.ops import KLAState, init_state, kla_scan, kla_scan_reference

# Input order accepted by every backend, used to label gradient comparisons.
INPUT_NAMES = ("v", "lambda_v", "k", "q", "a", "p")


@dataclasses.dataclass(frozen=True)
class Profile:
    """Per-backend accuracy contract."""

    name: str
    fwd_atol: float
    fwd_rtol: float
    exact_grads: tuple[str, ...]
    """Inputs whose gradient the backend computes exactly (tight threshold)."""
    loose_grads: tuple[str, ...] = ()
    """Inputs with a documented approximate gradient (relaxed threshold)."""
    exact_grad_tol: float = 2e-3
    loose_grad_tol: float = 0.25
    returns_state: bool = True
    supports_initial_state: bool = True
    clips_phi: bool = False
    """True for v2_1, which caps ``phi`` at 1000. Inverts the assertion in
    :func:`test_high_information_tokens_are_not_clipped` — that backend is
    *required* to deviate, so the test fails loudly if its clamp ever stops
    engaging."""
    scan_form: str = (
        "composes the same maps with a parallel associative scan over "
        "trace-normalized 2x2 matrices"
    )
    """How this backend parallelizes the recurrence, for the printed reading of
    the forward comparison. The point of that comparison is that the reference
    and the backend reach the same numbers by *different* routes, so the report
    has to say which route."""


_LANE_FORM = (
    "applies the same map down B*M*S independent lanes, one thread per "
    "(batch, channel, state), with time as the serial axis"
)

PROFILES = {
    p.name: p
    for p in [
        Profile("torch_unfused_pscan", 2e-4, 1e-4, exact_grads=INPUT_NAMES),
        Profile("torch_unfused_recurrent", 2e-4, 1e-4, exact_grads=INPUT_NAMES),
        Profile("torch_unfused_chunk", 2e-4, 1e-4, exact_grads=INPUT_NAMES),
        # Both triton chunk cells get the exact-gradient contract: the fused one
        # shares kla.ops.kernels.triton.kla_scan_bwd, which differentiates the
        # recurrence rather than the composed map.
        Profile(
            "triton_recurrent",
            5e-4,
            5e-4,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
            scan_form=_LANE_FORM,
        ),
        Profile(
            "triton_chunk",
            5e-4,
            5e-4,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
        ),
        Profile(
            "triton_pscan",
            5e-4,
            5e-4,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
            scan_form=(
                "composes the same maps per chunk, then resolves the chunks "
                "with a parallel scan carrying nothing between them"
            ),
        ),
        # The three unfused cells share one backward too: the adjoint reads the
        # values lambda and eta, not the order a forward produced them in.
        Profile(
            "triton_unfused_recurrent",
            5e-4,
            5e-4,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
            scan_form=_LANE_FORM,
        ),
        Profile(
            "triton_unfused_chunk",
            5e-4,
            5e-4,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
        ),
        Profile(
            "triton_unfused_pscan",
            5e-4,
            5e-4,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
            scan_form=(
                "composes the same maps per chunk, then resolves the chunks "
                "with a parallel scan carrying nothing between them"
            ),
        ),
        # The exact CUDA cells: same algebra as v2_* below, but they
        # differentiate the recurrence rather than the composition, so every
        # input is tight -- see kernels/cuda/scan/kla_scan_bwd.cuh.
        Profile(
            "cuda_recurrent",
            5e-4,
            5e-4,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
            scan_form=_LANE_FORM,
        ),
        # Composed forward, but the same lane-per-state backward, so it gets the
        # same exact-gradient contract.
        Profile(
            "cuda_chunk",
            1e-3,
            1e-3,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
        ),
        # Same composition, resolved across chunks by a parallel scan instead of
        # a serial carry; the backward is the same one again.
        Profile(
            "cuda_pscan",
            1e-3,
            1e-3,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
            scan_form=(
                "composes the same maps per chunk, then resolves the chunks "
                "with a parallel scan carrying nothing between them"
            ),
        ),
        # Exact forward; the precision-scan adjoint is approximate. dv and
        # dq ride the information-vector / read-out path and stay exact, whereas
        # d(lambda_v) additionally feeds the precision scan, so it is loose.
        Profile(
            "cuda_v2_2",
            1e-4,
            1e-4,
            exact_grads=("v", "q"),
            loose_grads=("lambda_v", "k", "a", "p"),
            returns_state=False,
            supports_initial_state=False,
        ),
        # The harder-clamped variant. Same numerics as v2_2 on well-conditioned
        # inputs -- it diverges only where its phi ceiling engages, which is what
        # `clips_phi` pins down.
        Profile(
            "cuda_v2_1",
            1e-4,
            1e-4,
            exact_grads=("v", "q"),
            loose_grads=("lambda_v", "k", "a", "p"),
            returns_state=False,
            supports_initial_state=False,
            clips_phi=True,
        ),
        # Both Metal strategies get the exact-gradient contract. For the fused
        # one that is the interesting claim: it is the same "one kernel, hand-
        # written backward" shape as cuda_v2_* above, yet every input is tight,
        # because a per-step Moebius map has an elementary adjoint where a
        # trace-normalized prefix product does not.
        Profile(
            "mps_recurrent",
            5e-4,
            5e-4,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
            scan_form=_LANE_FORM,
        ),
        # Composed forward, but the same lane-per-state backward, so it gets the
        # same exact-gradient contract.
        Profile(
            "mps_chunk",
            1e-3,
            1e-3,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
        ),
        # Same composition, resolved across chunks by a parallel scan instead of
        # a serial carry -- and the same lane-per-state backward again, reading
        # checkpoints this schedule produces rather than stores.
        Profile(
            "mps_pscan",
            1e-3,
            1e-3,
            exact_grads=INPUT_NAMES,
            exact_grad_tol=1e-2,
            scan_form=(
                "composes the same maps per chunk, then resolves the chunks "
                "with a parallel scan carrying nothing between them"
            ),
        ),
    ]
}

ALL_BACKENDS = list(PROFILES)
MPS_BACKENDS = [name for name in PROFILES if name.startswith("mps")]


def make_inputs(device, B=2, L=64, M=16, S=8, requires_grad=False):
    """Well-conditioned inputs: lambda_v and p strictly positive, a inside (0, 1)."""
    g = torch.Generator(device="cpu").manual_seed(0)

    def to(t):
        return t.to(device).requires_grad_(requires_grad)

    return (
        to(torch.randn(B, L, M, generator=g)),  # v
        to(torch.rand(B, L, M, generator=g) + 0.5),  # lambda_v (Λ^v > 0)
        to(torch.randn(B, L, S, generator=g) * 0.3),  # k
        to(torch.randn(B, L, S, generator=g) * 0.3),  # q
        to(torch.rand(M, S, generator=g) * 0.5 + 0.4),  # a
        to(torch.rand(M, S, generator=g) * 0.05 + 0.01),  # p
    )


def run_backend(backend, inputs, **kwargs):
    """Dispatch, converting an unsupported-configuration error into a skip."""
    try:
        return kla_scan(*inputs, backend=backend, **kwargs)
    except NotImplementedError as e:
        pytest.skip(f"backend {backend!r} unavailable/unsupported here: {e}")


def rel_err(got, ref):
    """Max abs deviation normalized by the reference's own scale.

    Gradients of the static ``a``/``p`` accumulate over batch x time, so their
    absolute magnitude says nothing on its own — normalizing by ``ref`` keeps one
    threshold meaningful across all six inputs.
    """
    return ((got - ref).abs().max() / (ref.abs().max() + 1.0)).item()


# Backend family -> the device it runs on, and how to ask whether we have one.
_DEVICE = {"torch": "cpu", "triton": "cuda", "cuda": "cuda", "mps": "mps"}
_HAVE = {
    "cpu": lambda: True,
    "cuda": torch.cuda.is_available,
    "mps": torch.backends.mps.is_available,
}


def device_for(backend):
    """torch is the CPU baseline; each kernel family names its own device."""
    device = _DEVICE[backend.split("_", 1)[0]]
    if not _HAVE[device]():
        pytest.skip(f"backend {backend!r} needs a {device} device")
    return device


def report(title, rows, why=None):
    """Print a measurement table and the reading of it, then let the caller assert.

    A pass/fail threshold tells you a backend is inside its documented budget; it
    does not tell you *where* inside. For the CUDA kernels that distinction
    matters -- their adjoint is approximate by construction, so the useful
    question is whether today's deviation is 5 % or 24 % of a 25 % budget, and a
    green test hides both. Printing also means a failing run shows every input's
    number rather than stopping at the first one over budget.

    ``why`` explains what the numbers mean for *this* backend, derived from the
    measured values rather than canned. Several of these tests pass for opposite
    reasons depending on the profile -- v2_1 passes the high-phi test by
    *deviating* -- so a bare green dot is genuinely ambiguous.

    Captured by default; run with ``-s`` (or ``-rA``) to see it:

        pytest tests/test_backends.py -k cuda -s

    ``rows`` is ``(label, value, annotation)``.
    """
    width = max((len(r[0]) for r in rows), default=0)
    print(f"\n  {title}")
    for label, value, note in rows:
        print(f"    {label:<{width}}  {value:>10.3e}   {note}")
    if why:
        print(textwrap.indent(textwrap.fill(" ".join(why.split()), 88), "    | "))


# --------------------------------------------------------------------- forward


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_forward_matches_reference(backend):
    """y and y_var from each backend must match the sequential reference."""
    prof = PROFILES[backend]
    device = device_for(backend)
    inputs = make_inputs(device)

    y, y_var, _ = run_backend(backend, inputs)
    y_ref, y_var_ref, _ = kla_scan_reference(*inputs)

    d_y = (y - y_ref).abs().max().item()
    d_var = (y_var - y_var_ref).abs().max().item()
    report(
        f"{backend}: forward deviation vs the sequential reference",
        [
            ("max|dy|", d_y, f"atol {prof.fwd_atol:g}"),
            ("max|dy_var|", d_var, f"rtol {prof.fwd_rtol:g}"),
        ],
        why=f"""
        PASS means agreement to {max(d_y, d_var):.1e} between two *different*
        algorithms, not two runs of one: the reference is a sequential python loop over
        kla_step, while {backend} {prof.scan_form}. Only the algebra is shared, so
        agreement at float32 noise level says the parallel form is genuinely
        equivalent rather than merely self-consistent.
        """,
    )

    torch.testing.assert_close(y, y_ref, atol=prof.fwd_atol, rtol=prof.fwd_rtol)
    torch.testing.assert_close(y_var, y_var_ref, atol=prof.fwd_atol, rtol=prof.fwd_rtol)
    assert (y_var >= 0).all(), "read-out variance must be non-negative"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_final_state_matches_reference(backend):
    """The carried (lambda, eta) state must match, for backends that return it."""
    prof = PROFILES[backend]
    device = device_for(backend)
    inputs = make_inputs(device)

    _, _, state = run_backend(backend, inputs)
    if not prof.returns_state:
        assert state is None, f"{backend} documents state=None (forward-only kernel)"
        return

    _, _, ref = kla_scan_reference(*inputs)
    torch.testing.assert_close(state.lam, ref.lam, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(state.eta, ref.eta, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_chunked_state_carry(backend):
    """Splitting the sequence and threading state must equal one full scan."""
    prof = PROFILES[backend]
    device = device_for(backend)
    v, lambda_v, k, q, a, p = make_inputs(device)
    if not prof.supports_initial_state:
        pytest.skip(f"{backend} does not accept a carried initial state")

    y_full, _, _ = run_backend(backend, (v, lambda_v, k, q, a, p))
    mid = v.shape[1] // 2
    head = (v[:, :mid], lambda_v[:, :mid], k[:, :mid], q[:, :mid], a, p)
    tail = (v[:, mid:], lambda_v[:, mid:], k[:, mid:], q[:, mid:], a, p)

    y1, _, state = run_backend(backend, head)
    y2, _, _ = run_backend(backend, tail, initial_state=state)

    torch.testing.assert_close(
        torch.cat([y1, y2], 1), y_full, atol=prof.fwd_atol, rtol=prof.fwd_rtol
    )


# -------------------------------------------------------------------- backward


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_backward_matches_reference(backend):
    """Per-input gradient parity, against each backend's documented contract."""
    prof = PROFILES[backend]
    device = device_for(backend)
    inputs = make_inputs(device, requires_grad=True)
    refs = tuple(t.detach().clone().requires_grad_(True) for t in inputs)

    y, y_var, _ = run_backend(backend, inputs)
    y_ref, y_var_ref, _ = kla_scan_reference(*refs)

    # Squaring y keeps the read-out path in the loss; summing y_var keeps the
    # precision path in it. Both are needed to give every input a gradient.
    (y.square().sum() + y_var.sum()).backward()
    (y_ref.square().sum() + y_var_ref.sum()).backward()

    # Measure every input first, print the table, and only then assert. Asserting
    # inside the loop would stop at the first input over budget and hide the rest
    # -- exactly the inputs you need to see when a kernel change moves the
    # approximate adjoint.
    rows, over = [], []
    for name, got, ref in zip(INPUT_NAMES, inputs, refs):
        assert got.grad is not None, f"{backend}: no gradient reached {name}"
        assert torch.isfinite(got.grad).all(), f"{backend}: non-finite d{name}"
        err = rel_err(got.grad, ref.grad)
        loose = name in prof.loose_grads
        budget = prof.loose_grad_tol if loose else prof.exact_grad_tol
        rows.append(
            (
                f"d{name}",
                err,
                f"{'loose' if loose else 'exact'}  budget {budget:<7g} "
                f"using {100 * err / budget:6.2f}%",
            )
        )
        if err >= budget:
            over.append(
                f"d{name} {err:.3e} >= {budget:g} ({'loose' if loose else 'exact'})"
            )

    worst_name, worst_err, _ = max(
        rows,
        key=lambda r: (
            r[1]
            / (
                prof.loose_grad_tol
                if r[0][1:] in prof.loose_grads
                else prof.exact_grad_tol
            )
        ),
    )
    worst_budget = (
        prof.loose_grad_tol
        if worst_name[1:] in prof.loose_grads
        else prof.exact_grad_tol
    )
    report(
        f"{backend}: gradient deviation vs the sequential reference",
        rows,
        why=f"""
        Worst input is {worst_name} at {100 * worst_err / worst_budget:.2f}% of its
        budget. {
            "All six gradients are exact adjoints, so anything above ~1e-6 would "
            "mean a real bug."
            if not prof.loose_grads
            else f"The loose group ({', '.join(prof.loose_grads)}) flows through the "
            "trace-normalized Moebius backward, which is NOT the exact adjoint of the "
            "forward compose -- a few percent there is the documented contract, not a "
            f"regression. The exact group ({', '.join(prof.exact_grads)}) rides the "
            "information-vector and read-out paths, which are exact, so those "
            "must stay tight."
        } PASS means every input is inside the budget its own path earns; it
        does not mean the gradients are correct to float32 -- read the percentages.
        """,
    )
    assert not over, (
        f"{backend}: gradient(s) outside the documented budget: " + "; ".join(over)
    )


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_gradient_descent_reduces_loss(backend):
    """End-to-end usefulness check, ported from the original CUDA tests.

    Gradient *parity* thresholds can pass on a gradient that is subtly wrong in a
    way that still points roughly downhill -- and the CUDA backward is knowingly
    approximate, so parity alone is weak evidence there. Ten Adam steps that
    actually reduce the loss is the check that the gradients are usable.

    The step is *projected*: three of the six inputs are constrained -- lambda_v is
    a precision and p a process-noise variance (both > 0), and a is a discrete
    decay in (0, 1). Adam's normalized step is ~lr regardless of gradient scale, so
    with p starting near 0.01 an unprojected run walks it negative within two
    steps, (a^2 + p*lambda) crosses zero, and the recursion diverges. That says
    nothing about gradient quality: :class:`~kla.layers.KLALayer` never optimizes
    these directly -- it learns ``lambda_log`` and ``process_noise`` and maps them
    through ``a = -exp(.)`` and a floor, so the constraints always hold in
    practice. The projection reproduces that guarantee.
    """
    device = device_for(backend)
    inputs = make_inputs(device, requires_grad=True)
    _, lambda_v, _, _, a, p = inputs
    target = torch.randn(2, 64, 16, device=device)

    # Probe once so an unsupported backend skips before the optimizer is built.
    run_backend(backend, inputs)

    opt = torch.optim.Adam(inputs, lr=1e-2)
    losses = []
    for _ in range(10):
        opt.zero_grad()
        y, _, _ = kla_scan(*inputs, backend=backend)
        loss = (y - target).pow(2).mean()
        loss.backward()
        opt.step()
        with torch.no_grad():  # project back onto the valid domain
            lambda_v.clamp_(min=1e-3)
            p.clamp_(min=1e-6)
            a.clamp_(min=1e-3, max=0.999)
        losses.append(loss.item())

    assert all(math.isfinite(x) for x in losses), (
        f"{backend}: non-finite loss: {losses}"
    )
    assert losses[-1] < losses[0], f"{backend}: loss did not decrease: {losses}"


@pytest.mark.parametrize("bad_p", [0.0, -0.05])
@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_non_positive_process_noise_stays_finite(backend, bad_p):
    """A non-positive process noise must be floored, not propagated into a NaN.

    ``p = 0`` is a real configuration (``KLAConfig.zero_process_noise``), and a
    negative p is reachable whenever ``p`` is optimized directly. Every backend
    floors it at ``P_MIN``; without that floor ``(a² + p·λ)`` crosses zero and the
    recursion diverges on the GPU paths while torch stays finite -- same input,
    different answer per backend. This guards it.
    """
    device = device_for(backend)
    v, lambda_v, k, q, a, p = make_inputs(device)
    p = torch.full_like(p, bad_p)

    y, y_var, _ = run_backend(backend, (v, lambda_v, k, q, a, p))
    assert torch.isfinite(y).all(), f"{backend}: non-finite y at p={bad_p}"
    assert torch.isfinite(y_var).all(), f"{backend}: non-finite y_var at p={bad_p}"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_near_zero_decay_stays_finite(backend):
    """A vanishing discrete decay must be floored, not divided by.

    ``a_bar = exp(-Δ·exp(lambda_log))`` decays *doubly* exponentially, so a
    lambda_log that drifts up during training drives it toward zero and the leaf
    matrix ``A = (1 + p·φ)/a²`` toward infinity. The scan composes two *raw*
    leaves before the first trace-normalization, so ``A²`` overflows float32
    around ``a² ≈ 5e-20``; the normalizer then computes ``inf/inf`` and NaN
    poisons the rest of the scan. torch floors ``a²`` at ``EPS``
    (:mod:`kla.ops.kla_ops`) and v2_2 matches it -- v2_1 divides unguarded.

    The floor is semantically inert: ``a_bar = 1e-6`` per step is already total
    forgetting over any horizon, so flooring changes only whether the arithmetic
    stays finite, never what the model computes.
    """
    prof = PROFILES[backend]
    device = device_for(backend)
    v, lambda_v, k, q, a, p = make_inputs(device)
    a = torch.full_like(a, 1e-10)  # a² = 1e-20, past the float32 overflow point

    y, y_var, _ = run_backend(backend, (v, lambda_v, k, q, a, p))
    report(
        f"{backend}: near-zero decay (a = 1e-10, a² = 1e-20)",
        [
            (
                "nonfinite y",
                float((~torch.isfinite(y)).sum()),
                "entries; 0 = floored correctly",
            ),
            ("nonfinite y_var", float((~torch.isfinite(y_var)).sum()), "entries"),
        ],
        why="""
        The leaf A = (1+p.phi)/a^2 reaches ~1e20 here, and the scan composes two RAW
        leaves before the first trace-normalization, so A^2 ~ 1e40 overflows float32
        (max 3.4e38); the normalizer then computes inf/inf = NaN. Flooring a^2 at 1e-12
        caps A at ~1e12, so A^2 ~ 1e24 stays finite. """
        + (
            "SKIPPED for v2_1, which divides unguarded by design."
            if prof.clips_phi
            else "PASS = 0 nonfinite entries, i.e. the floor is active. It costs "
            "nothing: a decay of 1e-6 already annihilates the state in one step, "
            "so flooring changes only whether the arithmetic survives, never what "
            "the model computes."
        ),
    )

    # clips_phi marks v2_1, which carries *both* unguarded paths -- the phi
    # ceiling and this unfloored 1/a². Rename the flag if a third one shows up.
    if prof.clips_phi:
        pytest.skip("v2_1 divides by an unfloored a² by design")
    assert torch.isfinite(y).all(), f"{backend}: non-finite y at a=1e-10"
    assert torch.isfinite(y_var).all(), f"{backend}: non-finite y_var at a=1e-10"


@pytest.mark.parametrize("backend", ALL_BACKENDS)
def test_high_information_tokens_are_not_clipped(backend):
    """A large per-token information gain must not be capped by any backend.

    ``phi = Λ^v·k²`` is how much one observation sharpens the precision. ``v2_1``
    caps it at 1000 (``compute_phi_r``) while leaving ``r = v·Λ^v·k`` uncapped --
    but ``mean = r/phi`` only equals ``v/k`` because ``Λ^v`` cancels between the
    two, so a token saturating by ``ρ = phi/1000`` comes out with its mean *and*
    its variance scaled by ``ρ``. Under the layer defaults phi reaches
    ``1/obs_var_min = 1e4``, so that cap sits inside the operating range, not
    above it -- yet :func:`make_inputs` keeps phi ~0.1 and never trips it. These
    inputs do.
    """
    prof = PROFILES[backend]
    device = device_for(backend)
    v, lambda_v, k, q, a, p = make_inputs(device)
    lambda_v = lambda_v * 1e4  # Λ^v up to ~1e4, matching obs_var_min = 1e-4
    phi_max = (lambda_v.unsqueeze(-1) * (k * k).unsqueeze(-2)).max().item()
    assert phi_max > 1e3, f"inputs never reach the v2_1 ceiling (phi_max={phi_max:g})"

    y, y_var, _ = run_backend(backend, (v, lambda_v, k, q, a, p))
    y_ref, y_var_ref, _ = kla_scan_reference(v, lambda_v, k, q, a, p)
    err_y, err_var = rel_err(y, y_ref), rel_err(y_var, y_var_ref)

    rho = phi_max / 1e3
    verdict = "clipping kernel — deviation REQUIRED" if prof.clips_phi else "must match"
    report(
        f"{backend}: high-phi deviation ({verdict})",
        [
            ("max phi", phi_max, f"ceiling 1e3, so up to {rho:.1f}x saturation"),
            ("rel|dy|", err_y, "posterior mean"),
            ("rel|dy_var|", err_var, "posterior variance"),
        ],
        why=f"""
        phi = Lambda^v.k^2 is one token's information gain. v2_1 caps it at 1000 in the
        lambda recursion but leaves r = (v.Lambda^v).k uncapped -- and mean = r/phi only
        equals v/k because Lambda^v cancels between them, so a saturated token has BOTH
        its mean and its variance inflated by rho = phi/1000, here up to {rho:.1f}x.
        """
        + (
            f"""
        This backend PASSES BY DEVIATING ({max(err_y, err_var):.2e} > 5e-3): that
        confirms its clamp is still engaging. A pass in the other direction would mean
        v2_1 had silently stopped clipping -- which is why the assertion is inverted
        here rather than skipped.
        """
            if prof.clips_phi
            else f"""
        This backend matches the reference to {max(err_y, err_var):.1e} at
        {rho:.1f}x past where v2_1 clips, so PASS means the ceiling is genuinely gone
        and the Lambda^v cancellation is intact.
        """
        ),
    )

    # Clipping shows up as an O(ρ) deviation, far above float32 scan noise.
    if prof.clips_phi:
        assert err_y > 5e-3 or err_var > 5e-3, (
            f"{backend} is the clipping kernel and must still deviate here "
            f"(y {err_y:.2e}, y_var {err_var:.2e} at phi_max={phi_max:g}) -- if this "
            "passes, v2_1's phi ceiling is no longer engaging"
        )
    else:
        assert err_y < 5e-3, (
            f"{backend}: y deviates at phi_max={phi_max:g} ({err_y:.2e})"
        )
        assert err_var < 5e-3, (
            f"{backend}: y_var deviates at phi_max={phi_max:g} ({err_var:.2e})"
        )


# ------------------------------------------------------- fused triton kernel
#
# The fused kernel is only reached when grad is *disabled* (see the gate in
# kla.ops.triton_backend), so every grad-enabled triton test above exercises the
# composed tiled path instead. These tests target the fused kernel directly.

needs_triton = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the fused triton kernel needs CUDA"
)

BLOCK_L = 64  # chunk_kla_scan.chunk_forward default chunk length


@needs_triton
@pytest.mark.parametrize("L", [16, BLOCK_L, 100, 2 * BLOCK_L])
def test_fused_matches_reference(L):
    """Fused forward vs the sequential reference, across chunk-boundary cases.

    ``L=16`` stays inside one chunk; ``L=64`` is exactly one full chunk; ``L=100``
    forces a *masked partial tail* on the second chunk; ``L=128`` is two full
    chunks. The cross-chunk Möbius/eta carry only runs for L > BLOCK_L, so the
    multi-chunk cases are the ones that exercise it.
    """
    inputs = make_inputs("cuda", L=L)
    with torch.no_grad():
        y, y_var, state = kla_scan(*inputs, backend="triton")
    y_ref, y_var_ref, ref = kla_scan_reference(*inputs)

    torch.testing.assert_close(y, y_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(y_var, y_var_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(state.lam, ref.lam, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(state.eta, ref.eta, atol=1e-3, rtol=1e-4)


@needs_triton
@pytest.mark.parametrize("L", [BLOCK_L, 100])
def test_fused_matches_composed(L):
    """The two triton paths must agree; only the grad-mode gate selects between them."""
    inputs = make_inputs("cuda", L=L)
    with torch.no_grad():
        y_fused, var_fused, s_fused = kla_scan(*inputs, backend="triton")
    y_comp, var_comp, s_comp = kla_scan(
        *inputs, backend="triton"
    )  # grad on -> composed

    torch.testing.assert_close(y_fused, y_comp, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(var_fused, var_comp, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(s_fused.lam, s_comp.lam, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(s_fused.eta, s_comp.eta, atol=1e-3, rtol=1e-4)


@needs_triton
@pytest.mark.parametrize("bad_p", [0.0, -0.05])
def test_fused_non_positive_process_noise(bad_p):
    """Same floor guard, for the fused kernel specifically.

    The grad-enabled test above only reaches the composed path, and the composed
    and fused paths floor ``p`` at different call sites.
    """
    v, lambda_v, k, q, a, p = make_inputs("cuda")
    p = torch.full_like(p, bad_p)
    with torch.no_grad():
        y, y_var, state = kla_scan(v, lambda_v, k, q, a, p, backend="triton")
    assert torch.isfinite(y).all() and torch.isfinite(y_var).all()
    assert torch.isfinite(state.lam).all() and torch.isfinite(state.eta).all()


@needs_triton
def test_fused_carried_initial_state():
    """lam0/eta0 in and lam_fin/eta_fin out, across a chunk boundary."""
    L = 100
    v, lambda_v, k, q, a, p = make_inputs("cuda", L=L)
    with torch.no_grad():
        y_full, _, _ = kla_scan(v, lambda_v, k, q, a, p, backend="triton")
        mid = 40  # inside the first chunk, so the split is not chunk-aligned
        y1, _, state = kla_scan(
            v[:, :mid],
            lambda_v[:, :mid],
            k[:, :mid],
            q[:, :mid],
            a,
            p,
            backend="triton",
        )
        y2, _, _ = kla_scan(
            v[:, mid:],
            lambda_v[:, mid:],
            k[:, mid:],
            q[:, mid:],
            a,
            p,
            backend="triton",
            initial_state=state,
        )
    torch.testing.assert_close(torch.cat([y1, y2], 1), y_full, atol=5e-4, rtol=5e-4)


@needs_triton
def test_fused_non_unit_initial_state():
    """A non-trivial prior must actually be honoured (not silently reset to unit)."""
    inputs = make_inputs("cuda", L=100)
    B, _, M = inputs[0].shape
    S = inputs[2].shape[2]
    base = init_state(B, M, S, device="cuda")
    state = KLAState(lam=base.lam * 3.0, eta=base.eta + 0.7)

    with torch.no_grad():
        y, _, _ = kla_scan(*inputs, backend="triton", initial_state=state)
        y_unit, _, _ = kla_scan(*inputs, backend="triton")
    y_ref, _, _ = kla_scan_reference(*inputs, initial_state=state)

    torch.testing.assert_close(y, y_ref, atol=5e-4, rtol=5e-4)
    assert not torch.allclose(y, y_unit, atol=1e-3), "initial state was ignored"


# ---------------------------------------------------------- the MPS backends
#
# The shared matrix above already runs both Metal backends against the
# reference. What it cannot see is the *kernel geometry*: the fused kernels put
# one thread on each (batch, channel, state) triple and one threadgroup on
# [next_pow2(d_state), ROWS] channels, so correctness depends on shapes that
# never come up in a single well-chosen test case -- padding lanes at
# s >= d_state and m >= d_inner, a read-out reduction that switches from SIMD
# shuffles to threadgroup memory past 32 states, and a backward that replays the
# sequence in KLA_CHUNK-sized pieces from checkpoints. These target that.

needs_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="the Metal kernels need an Apple GPU"
)

CHUNK = 16  # kla.ops.kernels.mps._shaders.DEFAULT_CHUNK
SIMD = 32  # a row wider than this drops to the threadgroup-memory reduction


@needs_mps
@pytest.mark.parametrize("backend", MPS_BACKENDS)
@pytest.mark.parametrize("L", [1, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK, 100])
def test_mps_chunk_boundaries(backend, L):
    """Forward and gradients across every position relative to the replay chunk.

    The fused backward does not store λ_t or η_t. It checkpoints them every
    ``CHUNK`` steps and replays the chunk forward before walking the adjoint
    back down it, so the arithmetic differs between a sequence that fills its
    chunks exactly and one that leaves a partial tail -- and the tail is
    predicated inside a fully unrolled loop, which is exactly the kind of thing
    that is either right or off by one. ``L=1`` also pins the degenerate case
    where every carry is still at its seed value.
    """
    inputs = make_inputs("mps", L=L, requires_grad=True)
    refs = tuple(t.detach().cpu().clone().requires_grad_(True) for t in inputs)

    y, y_var, state = run_backend(backend, inputs)
    y_ref, y_var_ref, ref_state = kla_scan_reference(*refs)
    (y.square().sum() + y_var.sum()).backward()
    (y_ref.square().sum() + y_var_ref.sum()).backward()

    torch.testing.assert_close(y.cpu(), y_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(y_var.cpu(), y_var_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(state.lam.cpu(), ref_state.lam, atol=1e-3, rtol=1e-4)
    for name, got, ref in zip(INPUT_NAMES, inputs, refs):
        err = rel_err(got.grad.cpu(), ref.grad)
        assert err < 1e-2, f"{backend} L={L}: d{name} off by {err:.2e}"


@needs_mps
@pytest.mark.parametrize("backend", MPS_BACKENDS)
@pytest.mark.parametrize("S", [1, 3, SIMD // 2, SIMD, SIMD + 1, 2 * SIMD])
@pytest.mark.parametrize("M", [1, 33])
def test_mps_threadgroup_geometry(backend, S, M):
    """Padding lanes must contribute exactly zero to every reduction.

    A threadgroup is ``[next_pow2(S), ROWS]`` channels wide, so unless both
    ``S`` and ``M`` divide those, some threads own no real (channel, state) at
    all. They cannot simply exit -- the read-out reduction is a SIMD shuffle
    (or, past ``SIMD`` states, a barrier), and both need every lane present. So
    they run the whole kernel on neutral inputs instead, and this checks that
    "neutral" really is neutral: an S of 3 leaves 5 padding lanes per row and an
    M of 33 leaves 15 padding rows in the last group.

    ``S`` also crosses the point where the read-out reduction switches
    implementation, which is a different code path rather than a different
    constant.
    """
    inputs = make_inputs("mps", B=2, L=CHUNK + 3, M=M, S=S, requires_grad=True)
    refs = tuple(t.detach().cpu().clone().requires_grad_(True) for t in inputs)

    y, y_var, _ = run_backend(backend, inputs)
    y_ref, y_var_ref, _ = kla_scan_reference(*refs)
    (y.square().sum() + y_var.sum()).backward()
    (y_ref.square().sum() + y_var_ref.sum()).backward()

    torch.testing.assert_close(y.cpu(), y_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(y_var.cpu(), y_var_ref, atol=5e-4, rtol=5e-4)
    for name, got, ref in zip(INPUT_NAMES, inputs, refs):
        err = rel_err(got.grad.cpu(), ref.grad)
        assert err < 1e-2, f"{backend} S={S} M={M}: d{name} off by {err:.2e}"


@needs_mps
@pytest.mark.parametrize("backend", ["mps_chunk", "mps_pscan"])
@pytest.mark.parametrize("L", [CHUNK, 100])
def test_mps_strategies_agree(backend, L):
    """The three Metal schedules must agree.

    They share no forward kernel -- ``mps_recurrent`` applies the Moebius map
    down B*M*S serial lanes, ``mps_chunk`` composes it with time as a parallel
    axis and carries across tiles, ``mps_pscan`` composes it per chunk and
    carries nothing -- so this is an independent cross-check of all three,
    reaching the same numbers by different routes. The composed routes carry
    more rounding, hence the looser budget than any has against the reference.
    """
    inputs = make_inputs("mps", L=L)
    with torch.no_grad():
        yr, vr, sr = kla_scan(*inputs, backend="mps_recurrent")
        yc, vc, sc = kla_scan(*inputs, backend=backend)
    torch.testing.assert_close(yc, yr, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(vc, vr, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(sc.lam, sr.lam, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(sc.eta, sr.eta, atol=1e-3, rtol=1e-3)


@needs_mps
@pytest.mark.parametrize("backend", MPS_BACKENDS)
def test_mps_state_gradient_flows(backend):
    """Gradients must flow *through* the returned filter state, not stop at it.

    This is the one the CUDA kernel cannot do at all -- it returns ``None`` for
    the state and is training-only for that reason. Both Metal backends carry
    ``lam0``/``eta0`` in and hand the final state back inside the graph, which
    is what makes truncated BPTT over chunked sequences work. A backward that
    merely ignored the incoming state gradient would still pass every other test
    in this file, so it needs its own: the loss here reads *only* the state.
    """
    inputs = make_inputs("mps", L=CHUNK + 5, requires_grad=True)
    refs = tuple(t.detach().cpu().clone().requires_grad_(True) for t in inputs)

    _, _, state = run_backend(backend, inputs)
    _, _, ref_state = kla_scan_reference(*refs)
    (state.lam.square().sum() + state.eta.sum()).backward()
    (ref_state.lam.square().sum() + ref_state.eta.sum()).backward()

    rows = []
    for name, got, ref in zip(INPUT_NAMES, inputs, refs):
        if ref.grad is None:  # q enters only through the read-out
            assert got.grad is None or got.grad.abs().max() == 0
            continue
        rows.append((f"d{name}", rel_err(got.grad.cpu(), ref.grad), "budget 1e-2"))
    report(
        f"{backend}: gradient of the carried state vs the reference",
        rows,
        why="""
        The loss reads the returned (lambda, eta) and nothing else, so every number
        here arrived by seeding the reverse walk with d(lam_fin)/d(eta_fin). PASS
        means chunked training carries gradient across the chunk boundary; the CUDA
        backend has no equivalent, since it returns no state.
        """,
    )
    over = [f"{n} {e:.2e}" for n, e, _ in rows if e >= 1e-2]
    assert not over, f"{backend}: state gradients outside budget: {'; '.join(over)}"


TILE = 128  # ROWS * ITEMS at d_state=16: the tiled kernel's timesteps per pass


@needs_mps
@pytest.mark.parametrize("L", [1, 7, TILE - 1, TILE, TILE + 1, 2 * TILE, 300])
@pytest.mark.parametrize("S", [1, 3, 16, 32, 64])
def test_mps_chunk_forward(L, S):
    """The time-parallel forward, across every position relative to a tile.

    This kernel is the one that splits a tile of timesteps across a threadgroup
    and composes 2x2 Moebius matrices to stitch them, so its failure modes are
    the boundary ones: a sequence that ends mid-tile leaves later threads with
    no work, and their aggregates have to be the identity rather than garbage.
    ``S`` also crosses the width where the read-out reduction stops fitting in a
    SIMD-group and starts using barriers -- which every thread must now reach,
    including ones whose timesteps are all past the end.
    """
    inputs = make_inputs("mps", B=2, L=L, M=3, S=S)
    refs = tuple(t.cpu() for t in inputs)

    with torch.no_grad():
        y, y_var, state = kla_scan(*inputs, backend="mps_chunk")
    y_ref, y_var_ref, ref_state = kla_scan_reference(*refs)

    torch.testing.assert_close(y.cpu(), y_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(y_var.cpu(), y_var_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(state.lam.cpu(), ref_state.lam, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(state.eta.cpu(), ref_state.eta, atol=1e-3, rtol=1e-3)


@needs_mps
def test_mps_chunk_backward_is_the_recurrent_kernel():
    """``mps_chunk``'s adjoint does not mirror its forward, and must not.

    The forward *composes* the Moebius maps to make time parallel; the backward
    replays from the checkpoints it wrote and walks a scalar reverse recurrence
    down the serial state lanes -- the same kernel ``mps_recurrent`` uses. So
    the two must agree to far better than either's budget against the
    reference: the checkpoints are the only thing passing between them, and a
    stride or index mismatch would show up here first.
    """
    grads = []
    for backend in ("mps_recurrent", "mps_chunk"):
        inputs = make_inputs("mps", L=100, requires_grad=True)
        y, y_var, _ = kla_scan(*inputs, backend=backend)
        (y.square().sum() + y_var.sum()).backward()
        grads.append([t.grad for t in inputs])
    for name, gr, gc in zip(INPUT_NAMES, *grads):
        torch.testing.assert_close(gc, gr, atol=1e-4, rtol=1e-3, msg=f"d{name}")


@needs_mps
@pytest.mark.parametrize("L", [1, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK, 100])
def test_mps_chunk_checkpoints_span_every_length(L):
    """The checkpoint stride is independent of the tile depth, so both bound.

    ``KLA_CHUNK`` (where the backward resumes) and ``KLA_ITEMS`` (how deep one
    thread walks) used to share a name. They do not divide each other, so a
    sequence that ends mid-tile, mid-chunk, or exactly on either boundary is the
    thing that catches a mismatch.
    """
    inputs = make_inputs("mps", L=L, requires_grad=True)
    refs = tuple(t.detach().cpu().clone().requires_grad_(True) for t in inputs)
    y, y_var, _ = kla_scan(*inputs, backend="mps_chunk")
    (y.square().sum() + y_var.sum()).backward()
    y_ref, y_var_ref, _ = kla_scan_reference(*refs)
    (y_ref.square().sum() + y_var_ref.sum()).backward()
    for name, got, ref in zip(INPUT_NAMES, inputs, refs):
        err = rel_err(got.grad.cpu(), ref.grad)
        assert err < 1e-2, f"L={L}: d{name} off by {err:.2e}"


@needs_mps
@pytest.mark.parametrize(
    "L", [1, CHUNK, CHUNK + 1, 3 * CHUNK - 7, 5 * CHUNK, 17 * CHUNK]
)
def test_mps_pscan_doubling_depth(L):
    """``mps_pscan`` resolves its chunks across launches, so the depth is data.

    The other two forwards carry state from one tile to the next inside a single
    dispatch. This one runs ``ceil(log2(NCK))`` doubling rounds over the chunk
    aggregates, ping-ponging two buffers -- so the round count, which buffer
    holds the answer at the end, and the ``c < off`` pass-through are all
    functions of the sequence length. The lengths here give NCK of 1, 1, 2, 3, 5
    and 17: a scan with no rounds at all, powers of two, and the odd counts
    where the last round only touches part of the axis.
    """
    inputs = make_inputs("mps", B=2, L=L, M=3, S=8, requires_grad=True)
    refs = tuple(t.detach().cpu().clone().requires_grad_(True) for t in inputs)

    y, y_var, state = kla_scan(*inputs, backend="mps_pscan")
    (y.square().sum() + y_var.sum()).backward()
    y_ref, y_var_ref, ref_state = kla_scan_reference(*refs)
    (y_ref.square().sum() + y_var_ref.sum()).backward()

    torch.testing.assert_close(y.cpu(), y_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(y_var.cpu(), y_var_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(state.lam.cpu(), ref_state.lam, atol=1e-3, rtol=1e-3)
    for name, got, ref in zip(INPUT_NAMES, inputs, refs):
        err = rel_err(got.grad.cpu(), ref.grad)
        assert err < 1e-2, f"L={L}: d{name} off by {err:.2e}"


@needs_mps
@pytest.mark.parametrize("backend", ["auto", *MPS_BACKENDS])
def test_mps_decode_from_prior(backend):
    """The prior read-out has to be inside the fused kernel, not around it.

    The composed path can apply it afterwards in torch, on the ``[B,L,M,S]``
    mean and variance it already materialized. The fused kernel materializes
    neither, so ``mean = a.mean`` and ``var = a^2.var + p`` happen between the
    per-lane update and the reduction, and their adjoint has to feed back into
    ``a`` and ``p`` as well as into the scan.
    """
    inputs = make_inputs("mps", L=CHUNK + 5, requires_grad=True)
    refs = tuple(t.detach().cpu().clone().requires_grad_(True) for t in inputs)

    y, y_var, _ = run_backend(backend, inputs, decode_from_prior=True)
    y_ref, y_var_ref, _ = kla_scan_reference(*refs, decode_from_prior=True)
    (y.square().sum() + y_var.sum()).backward()
    (y_ref.square().sum() + y_var_ref.sum()).backward()

    torch.testing.assert_close(y.cpu(), y_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(y_var.cpu(), y_var_ref, atol=5e-4, rtol=5e-4)
    for name, got, ref in zip(INPUT_NAMES, inputs, refs):
        err = rel_err(got.grad.cpu(), ref.grad)
        assert err < 1e-2, f"{backend}: d{name} off by {err:.2e}"


@needs_mps
def test_mps_auto_does_not_route_around_the_d_state_ceiling():
    """ "auto" resolves on the device alone, so a limit surfaces as an error.

    ``d_state`` past the Metal kernels' cap is the one request they do not
    cover. Silently swapping in a backend that does would make the kernels a run
    used a function of its inputs, so the refusal names the one with no ceiling
    and leaves the choice to the caller. The message is the whole user interface
    for it.
    """
    from kla.ops.kernels.mps import MAX_DSTATE

    wide = make_inputs("mps", B=1, L=8, M=2, S=MAX_DSTATE + 1)
    for backend in ("auto", "mps", "mps_recurrent", "mps_chunk"):
        with pytest.raises(NotImplementedError, match="d_state") as excinfo:
            kla_scan(*wide, backend=backend)
        assert "torch" in str(excinfo.value)

    cpu = tuple(t.cpu() for t in wide)
    y, y_var, _ = kla_scan(*cpu, backend="torch")
    y_ref, y_var_ref, _ = kla_scan_reference(*cpu)
    torch.testing.assert_close(y, y_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(y_var, y_var_ref, atol=5e-4, rtol=5e-4)


@needs_mps
@pytest.mark.parametrize("backend", MPS_BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_mps_widens_low_precision_inputs(backend, dtype):
    """Half-precision activations must be widened to float32, as torch's are.

    The scan is numerically delicate and every backend runs it in float32
    regardless of what came in (:func:`kla.ops.kla_ops._compute_dtype`); the
    layer casts back afterwards. Worth pinning on Metal specifically because the
    kernels read raw float32 pointers with computed offsets, so a half tensor
    reaching one would not be a precision question -- it would be a
    misinterpreted buffer.

    float64 has no test here because it cannot arise: MPS refuses to allocate
    one at all, which is also why ``tests/test_gradcheck.py`` skips this device.
    """
    inputs = tuple(t.to(dtype) for t in make_inputs("mps"))
    refs = tuple(t.cpu() for t in inputs)

    y, y_var, _ = run_backend(backend, inputs)
    y_ref, y_var_ref, _ = kla_scan_reference(*refs)

    assert y.dtype == torch.float32, "the scan runs and returns float32"
    torch.testing.assert_close(y.cpu(), y_ref, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(y_var.cpu(), y_var_ref, atol=5e-4, rtol=5e-4)


# ------------------------------------------------------ dispatch-table cover
#
# PROFILES is written by hand, so a backend added to the dispatcher would
# silently get no parity coverage at all. This is the guard.

# Every implementation in the registry is profiled above. Anything added
# without a profile fails test_every_backend_is_covered, which is the point.
COVERED_ELSEWHERE: dict[str, str] = {}


def test_every_backend_is_covered():
    """Each dispatcher entry is either profiled here or explicitly excused."""
    from kla.ops import backend_names

    unaccounted = [
        name
        for name in backend_names()
        if name not in PROFILES and name not in COVERED_ELSEWHERE
    ]
    assert not unaccounted, (
        f"backend(s) {unaccounted} have no parity coverage: add a Profile, or a "
        "COVERED_ELSEWHERE entry saying which test covers them"
    )
