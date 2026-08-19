"""Finite-difference gradient checks (float64) for the KLA scan.

These validate the *math* — the Möbius precision recurrence, the affine
information recurrence, and the read-out — against central differences, which is
the one check that does not assume any hand-written adjoint is correct.

Why this only covers the torch backend
--------------------------------------
gradcheck needs float64 to get a usable finite-difference signal, and the triton
and CUDA kernels are float32-only. So the coverage split is:

* the *algebra* is gradchecked here, in float64, on the torch backend;
* the kernels' hand-written backwards are validated by gradient parity against
  that (now gradchecked) torch path, in ``tests/test_backends.py``.

This is what makes the parity thresholds there meaningful: without a
finite-difference anchor, comparing autograd against autograd would only prove
the two implementations differentiate the same graph, not that the graph is
right.

These are marked ``slow`` (gradcheck runs 2 forward passes per input element);
run with ``-m "not slow"`` to skip them.
"""

import pytest
import torch

from kla.ops import init_state, kla_scan_torch, kla_step

pytestmark = pytest.mark.slow

# Small enough that gradcheck's O(numel) perturbation loop stays quick, and the
# ops' internal clamps (EPS on phi, P_MIN on p) stay far from active -- a clamp
# that engages would zero the analytic gradient and fail the check spuriously.
B, L, M, S = 1, 8, 3, 2


def _inputs(device, seq=True, seed=0):
    if device == "mps":
        # Kernel backends are anchored by gradient parity in test_backends.py.
        pytest.skip("gradcheck needs float64; MPS is float32-only")
    g = torch.Generator(device="cpu").manual_seed(seed)
    kw = dict(dtype=torch.float64, device=device)

    def leaf(t):
        return t.to(**kw).requires_grad_(True)

    shape_v = (B, L, M) if seq else (B, M)
    shape_k = (B, L, S) if seq else (B, S)
    return (
        leaf(torch.randn(*shape_v, generator=g)),  # v
        leaf(torch.rand(*shape_v, generator=g) + 0.5),  # lambda_v, well clear of 0
        leaf(torch.randn(*shape_k, generator=g) * 0.5),  # k
        leaf(torch.randn(*shape_k, generator=g) * 0.5),  # q
        leaf(torch.rand(M, S, generator=g) * 0.4 + 0.5),  # a in (0.5, 0.9)
        leaf(torch.rand(M, S, generator=g) * 0.04 + 0.02),  # p well above P_MIN
    )


@pytest.mark.parametrize("scan_impl", ["doubling", "sequential"])
def test_gradcheck_scan_float64(device, scan_impl):
    """Central differences vs autograd through the full parallel scan."""
    inputs = _inputs(device)

    def f(*args):
        y, y_var, _ = kla_scan_torch(*args, scan_impl=scan_impl)
        return y, y_var

    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_scan_with_initial_state(device):
    """The lam0/eta0 boundary terms must also be differentiated correctly."""
    inputs = _inputs(device)
    base = init_state(B, M, S, device=device, dtype=torch.float64)
    state = type(base)(lam=base.lam * 2.0, eta=base.eta + 0.3)

    def f(*args):
        y, y_var, _ = kla_scan_torch(*args, initial_state=state, scan_impl="doubling")
        return y, y_var

    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize("decode_from_prior", [False, True])
def test_gradcheck_step_float64(device, decode_from_prior):
    """The O(1) decode recurrence, including the predict-ahead branch."""
    inputs = _inputs(device, seq=False)
    state = init_state(B, M, S, device=device, dtype=torch.float64)

    def f(*args):
        y, y_var, _ = kla_step(*args, state, decode_from_prior=decode_from_prior)
        return y, y_var

    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_gradcheck_decode_from_prior_scan(device):
    """``decode_from_prior`` adds a predict step after the scan; check it too."""
    inputs = _inputs(device)

    def f(*args):
        y, y_var, _ = kla_scan_torch(
            *args, scan_impl="doubling", decode_from_prior=True
        )
        return y, y_var

    assert torch.autograd.gradcheck(f, inputs, eps=1e-6, atol=1e-5, rtol=1e-3)


def test_float64_is_preserved(device):
    """Guards the dtype policy the checks above depend on.

    The ops widen bf16/fp16 to float32 but must pass float64 through untouched --
    an unconditional ``.float()`` anywhere in the chain would silently downcast
    and make every gradcheck here meaningless (it would still *pass*, just at
    float32 precision with a float64 finite-difference step).
    """
    inputs = _inputs(device)
    y, y_var, state = kla_scan_torch(*inputs)
    assert y.dtype == torch.float64
    assert y_var.dtype == torch.float64
    assert state.lam.dtype == torch.float64 and state.eta.dtype == torch.float64
