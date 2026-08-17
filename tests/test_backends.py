"""Cross-backend parity: every available backend against the sequential reference.

The three backends implement the same math with very different numerics, so each
gets its own tolerance profile rather than one shared threshold:

* ``torch``  — the reference implementation. Forward and backward both tight.
* ``triton`` — hand-written kernels with an exact reverse-scan adjoint. Tight.
* ``cuda``   — the fused CUDA kernel. Its forward is bit-exact, but its
  trace-normalized Möbius **backward is not the exact adjoint** (see the parity
  notes in :mod:`kla.ops.cuda_backend`): gradients that flow through the
  precision scan are only accurate to ~5-15 % relative, while the ones that flow
  through the information-vector / read-out path are exact.

That last point is why :data:`PROFILES` splits the gradient inputs into
``exact_grads`` and ``loose_grads``. Asserting a tight threshold on the loose
group would fail on a *correct* build — the looseness is a documented property
of that backward, not a bug to be caught here. If you want exact gradients, use
``backend="torch"`` or ``"triton"``.

Backends that cannot run (no CUDA, no triton, no nvcc) raise
:class:`NotImplementedError` from the dispatcher and are skipped, so this file is
meaningful on a CPU-only box (where it degrades to torch-only) and on a GPU node
without a CUDA toolkit (torch + triton).
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


PROFILES = {
    p.name: p
    for p in [
        Profile("torch", 2e-4, 1e-4, exact_grads=INPUT_NAMES),
        Profile("triton", 5e-4, 5e-4, exact_grads=INPUT_NAMES, exact_grad_tol=1e-2),
        # v2.1: exact forward; the precision-scan adjoint is approximate. dv and
        # dq ride the information-vector / read-out path and stay exact, whereas
        # d(lambda_v) additionally feeds the precision scan, so it is loose.
        Profile(
            "cuda",
            1e-4,
            1e-4,
            exact_grads=("v", "q"),
            loose_grads=("lambda_v", "k", "a", "p"),
            returns_state=False,
            supports_initial_state=False,
        ),
        # The harder-clamped variant. Same numerics as `cuda` on well-conditioned
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
    ]
}

ALL_BACKENDS = list(PROFILES)


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


def device_for(backend):
    """torch is the CPU baseline; the kernel backends require a GPU."""
    if backend == "torch":
        return "cpu"
    if not torch.cuda.is_available():
        pytest.skip(f"backend {backend!r} needs CUDA")
    return "cuda"


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
        kla_step, while {backend} is a parallel associative scan
        ({"in log space" if backend == "torch" else "in linear space, normalized"}).
        Only the algebra is shared, so agreement at float32 noise level says the
        parallel form is genuinely equivalent rather than merely self-consistent.
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

BLOCK_L = 64  # fused_kla_scan.fused_kla_forward default chunk length


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
