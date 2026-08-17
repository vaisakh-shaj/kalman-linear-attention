"""The plain and mamba blocks, and the phi-floor correction.

Two named presets over ``value_rank`` / ``var_rank``::

    plain block   value_rank="full", var_rank="full"   <- the defaults, the paper
    mamba block   value_rank="conv", var_rank="dt"

The load-bearing property is that **neither is visible to the scan**. Both emit
v [B,L,M], Lambda^v [B,L,M], k/q [B,L,S], so no backend, kernel or backward
changes. :func:`test_scan_sees_identical_shapes` pins that down directly.
"""

import math

import pytest
import torch

from kla.configs import KLAConfig
from kla.layers.kla_layer import KLALayer
from kla.ops.kla_ops import _sufficient_stats

D, S, B, L = 64, 8, 2, 12
PLAIN = dict(value_rank="full", var_rank="full")
MAMBA = dict(value_rank="conv", var_rank="dt")


def _cfg(**kw):
    return KLAConfig(d_state=S, expand=2, backend="torch", **kw)


def _params(layer):
    return sum(p.numel() for p in layer.parameters())


# ------------------------------------------------------------------ shapes


@pytest.mark.parametrize(
    "preset",
    [
        PLAIN,
        MAMBA,
        dict(value_rank="dt", var_rank="dt"),
        dict(value_rank=16, var_rank=4),
    ],
)
def test_forward_shape(preset):
    lay = KLALayer(D, _cfg(**preset))
    out = lay(torch.randn(B, L, D))
    assert out.shape == (B, L, D) and torch.isfinite(out).all()


def test_scan_sees_identical_shapes():
    """Every block must hand the scan the same four tensors, shape and dtype.

    This is the whole reason no kernel work was needed. If a future block breaks
    it, the failure should surface here rather than as a confusing kernel error.
    """
    ref = None
    for preset in (PLAIN, MAMBA, dict(value_rank="dt", var_rank="dt")):
        lay = KLALayer(D, _cfg(**preset))
        z = torch.randn(B, L, lay.d_inner)
        sig = [(t.shape, t.dtype) for t in lay._project_sensors(z)]
        if ref is None:
            ref = sig
        assert sig == ref, f"{preset} changed the scan's inputs: {sig} != {ref}"


def test_conv_value_is_exactly_z():
    """value_rank='conv' must pass z through untouched, not just a same-shaped one."""
    lay = KLALayer(D, _cfg(**MAMBA))
    z = torch.randn(B, L, lay.d_inner)
    v, _, _, _ = lay._project_sensors(z)
    assert torch.equal(v, z.float())


# ------------------------------------------------------------- parameters


def test_plain_is_the_published_default():
    """The defaults must still build the paper's layer, or every stored result moves."""
    assert KLAConfig().value_rank == "full" and KLAConfig().var_rank == "full"
    M = 2 * D
    plain = KLALayer(D, _cfg(**PLAIN))
    # sensor_proj emits v(M) + logvar(M) + k(S) + q(S), and nothing expands.
    assert plain.sensor_proj.out_features == 2 * M + 2 * S
    assert plain.value_expand is None and plain.var_expand is None


def test_mamba_block_is_cheaper_and_wired_right():
    M, r = 2 * D, math.ceil(D / 8)
    m = KLALayer(D, _cfg(**MAMBA))
    assert m.sensor_proj.out_features == r + 2 * S  # no value slice at all
    assert m.value_expand is None  # v = z, nothing to lift
    assert (m.var_expand.in_features, m.var_expand.out_features) == (r, M)
    assert _params(m) < _params(KLALayer(D, _cfg(**PLAIN)))


def test_explicit_rank_is_honoured():
    m = KLALayer(D, _cfg(value_rank=16, var_rank=4))
    assert m.sensor_proj.out_features == 16 + 4 + 2 * S
    assert m.value_expand.in_features == 16 and m.var_expand.in_features == 4


def test_dt_follows_dt_rank_override():
    """'dt' resolves through cfg.dt_rank, so one knob governs every bottleneck."""
    m = KLALayer(D, _cfg(value_rank="dt", var_rank="dt", dt_rank=5))
    assert m.value_expand.in_features == 5 and m.var_expand.in_features == 5


# --------------------------------------------------------------- gradients


@pytest.mark.parametrize("preset", [PLAIN, MAMBA])
def test_all_parameters_get_gradients(preset):
    lay = KLALayer(D, _cfg(**preset))
    lay(torch.randn(B, L, D)).square().sum().backward()
    dead = [
        n
        for n, p in lay.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    assert not dead, f"{preset}: no/non-finite gradient for {dead}"


# ------------------------------------------------------------- phi correction


def test_phi_floor_is_off_by_default():
    """The old kernel floored phi AND masked its gradient; clamp_min does both.

    With a key of exactly zero the floor would (a) inject 1e-12 of information
    that is not there and (b) zero the gradient into lambda_v and k. Neither
    should happen on the default path.
    """
    lam = torch.rand(B, L, 4).add(0.5).requires_grad_(True)
    k = torch.zeros(B, L, 3, requires_grad=True)  # exactly zero -> phi == 0
    v = torch.randn(B, L, 4)

    phi, _ = _sufficient_stats(v, lam, k)  # floor=False (default)
    assert torch.equal(phi, torch.zeros_like(phi)), (
        "phi was floored on the default path"
    )
    phi.sum().backward()
    assert lam.grad is not None and k.grad is not None
    # The gradient of phi = lam (x) k^2 wrt k is 2*lam*k, which IS zero at k=0.
    # The point is that the *floor* is not what zeroed it -- lam's gradient is
    # sum over s of k^2 = 0 here too, so use a non-zero key to see the masking.
    lam2 = torch.rand(B, L, 4).add(0.5).requires_grad_(True)
    k2 = torch.full((B, L, 3), 1e-9, requires_grad=True)  # phi ~ 1e-18 < EPS
    phi2, _ = _sufficient_stats(v, lam2, k2)
    phi2.sum().backward()
    assert lam2.grad.abs().sum() > 0, "gradient into lambda_v was masked"
    assert k2.grad.abs().sum() > 0, "gradient into k was masked"


def test_phi_floor_still_available_for_log_space():
    lam = torch.rand(B, L, 4).add(0.5)
    k = torch.zeros(B, L, 3)
    phi, _ = _sufficient_stats(torch.randn(B, L, 4), lam, k, floor=True)
    assert (phi > 0).all(), "floor=True must keep phi strictly positive for log()"


@pytest.mark.parametrize("mobius_impl", ["linear", "log"])
def test_scan_finite_with_zero_keys(mobius_impl):
    """A zero key is a no-information observation; both scans must survive it."""
    from kla.ops.kla_ops import kla_scan_torch

    g = torch.Generator().manual_seed(0)
    v = torch.randn(B, L, 4, generator=g)
    lam = torch.randn(B, L, 4, generator=g).abs() + 0.5
    k = torch.randn(B, L, 3, generator=g)
    k[:, ::2] = 0.0  # every other token blind
    q = torch.randn(B, L, 3, generator=g)
    a = -(torch.rand(4, 3, generator=g) + 0.1)
    p = torch.rand(4, 3, generator=g) + 1e-3
    y, y_var, st = kla_scan_torch(v, lam, k, q, a, p, mobius_impl=mobius_impl)
    for name, t in (("y", y), ("y_var", y_var), ("lam", st.lam), ("eta", st.eta)):
        assert torch.isfinite(t).all(), f"{mobius_impl}: non-finite {name}"
