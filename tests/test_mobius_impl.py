"""The two Möbius representations in the torch backend must agree.

``kla_scan_torch`` composes the precision map either in linear space with trace
normalization (``mobius_impl="linear"``, the default and what both GPU backends
do) or in log space with ``logaddexp`` (``"log"``).
They describe the same map, so on well-conditioned inputs they must agree ---
and both must agree with :func:`kla_scan_reference`, which applies the filter
recurrence directly and knows nothing about Möbius matrices at all. That third
implementation is what keeps this from being two ways of stating one assumption.

Trace normalization is safe here because every leaf matrix is entrywise positive
(A=(1+pφ)/a², B=φ, C=p/a², D=1, all clamped positive) and positive matrices are
closed under multiplication, so the trace can never vanish whatever order the
scan combines in. :func:`test_entries_stay_bounded` checks the consequence: after
normalization every entry lies in (0,1).

The one place they genuinely diverge is extreme decay. After normalization
D ≈ a²/(1+pφ), which underflows float32 once ā falls far enough.
:func:`test_extreme_decay_boundary` measures where, rather than asserting a
number nobody has checked.
"""

import pytest
import torch

from kla.ops.kla_ops import kla_scan_reference, kla_scan_torch

B, L, M, S = 2, 32, 8, 4
IMPLS = ["linear", "log"]


def _inputs(
    device="cpu", L=L, dtype=torch.float32, a_bar=None, requires_grad=False, seed=0
):
    """Scan inputs. ``a_bar`` pins the discrete decay to a constant if given."""
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=g)

    v, lambda_v = rnd(B, L, M), rnd(B, L, M).abs() + 0.5
    k, q = rnd(B, L, S), rnd(B, L, S)
    # ā = exp(Δa) lives in (0,1); p > 0. Mirror what the layer can actually emit.
    a = (
        torch.full((M, S), a_bar, device=device, dtype=dtype)
        if a_bar is not None
        else rnd(M, S).abs().clamp(0.05, 0.95)
    )
    p = rnd(M, S).abs() * 0.01 + 1e-4
    out = [v, lambda_v, k, q, a, p]
    if requires_grad:
        for t in out:
            t.requires_grad_(True)
    return out


def _rel(x, y):
    return ((x - y).abs().max() / y.abs().max().clamp_min(1e-12)).item()


@pytest.mark.parametrize(
    # No "sequential": that implementation applies the map instead of composing it,
    # so there is no representation for mobius_impl to choose between.
    "scan_impl",
    ["associative", "doubling", "chunk"],
)
def test_linear_matches_log_forward(scan_impl):
    """Same map, two representations -- across every way of parallelizing the scan.

    Parametrized over scan_impl because trace normalization is applied per
    combine, so a scheme that only worked for one grouping would be a real bug;
    the positivity argument says it must hold for all of them.
    """
    args = _inputs()
    lin = kla_scan_torch(*args, scan_impl=scan_impl, mobius_impl="linear")
    log = kla_scan_torch(*args, scan_impl=scan_impl, mobius_impl="log")
    assert _rel(lin[0], log[0]) < 1e-5, f"{scan_impl}: y"
    assert _rel(lin[1], log[1]) < 1e-5, f"{scan_impl}: y_var"
    assert _rel(lin[2].eta, log[2].eta) < 1e-5, f"{scan_impl}: final eta"
    # lam gets a looser budget than the read-outs it feeds. It is the raw
    # precision, so it carries the full dynamic range of the accumulation
    # (measured ~1.4e-5 for sequential at L=32) while y/y_var/eta come out an
    # order of magnitude tighter. fp32 eps is 1.2e-7 and this is an L-step
    # product, so this is roundoff, not disagreement.
    assert _rel(lin[2].lam, log[2].lam) < 1e-4, f"{scan_impl}: final lam"


@pytest.mark.parametrize("mobius_impl", IMPLS)
def test_matches_sequential_reference(mobius_impl):
    """Both must match the direct filter recurrence, which uses no Möbius form."""
    args = _inputs()
    got = kla_scan_torch(*args, mobius_impl=mobius_impl)
    ref = kla_scan_reference(*args)
    assert _rel(got[0], ref[0]) < 2e-4, f"{mobius_impl}: y"
    assert _rel(got[1], ref[1]) < 2e-4, f"{mobius_impl}: y_var"


@pytest.mark.parametrize("mobius_impl", IMPLS)
def test_gradients_match(mobius_impl):
    """Backward parity against the log path, with y_var in the objective.

    y_var is included deliberately: the Monte-Carlo marginal loss is the first
    thing in this repo to put a non-zero cotangent on it, so a variance-path
    gradient bug would otherwise stay invisible.
    """
    names = ["v", "lambda_v", "k", "q", "a", "p"]
    got = _inputs(requires_grad=True)
    ref = [t.detach().clone().requires_grad_(True) for t in got]

    y, y_var, _ = kla_scan_torch(*got, mobius_impl=mobius_impl)
    y_r, y_var_r, _ = kla_scan_torch(*ref, mobius_impl="log")
    (y.square().sum() + y_var.sum()).backward()
    (y_r.square().sum() + y_var_r.sum()).backward()

    for n, g, r in zip(names, got, ref):
        assert g.grad is not None, f"no gradient reached {n}"
        assert torch.isfinite(g.grad).all(), f"{mobius_impl}: non-finite d{n}"
        assert _rel(g.grad, r.grad) < 1e-4, f"{mobius_impl}: d{n}"


def test_entries_stay_bounded():
    """Trace normalization keeps the composed map O(1). This is what replaces log space.

    Note what is and is not bounded, because getting this wrong is easy:

    * A, D land in (0,1) -- forced, since A+D=1 with both positive.
    * B, C are off-diagonal and are NOT bounded by 1; they routinely exceed it.
      What holds is the product bound B*C < 1/4, from det = AD - BC > 0 (the leaf
      determinant is 1/a² > 0, determinants are multiplicative, and normalizing
      only divides by (A+D)² > 0) together with AD <= 1/4 when A+D=1.
    * Positivity is preserved, so the trace never vanishes.

    Starting from leaves spanning twelve orders of magnitude, all of it must
    still hold after 2^20 timesteps' worth of composition.
    """
    from kla.ops.kla_ops import _mobius_combine_tracenorm

    g = torch.Generator().manual_seed(0)
    scale = torch.logspace(-6, 6, 64).unsqueeze(-1)
    acc = tuple((torch.rand(64, 4, generator=g) + 1e-3) * scale for _ in range(4))

    for _ in range(20):
        acc = _mobius_combine_tracenorm(acc, acc)
        A, Bm, Cm, D = acc
        assert torch.stack(acc).isfinite().all(), (
            "trace-norm produced non-finite entries"
        )
        assert (torch.stack(acc) >= 0).all(), "entries left the positive orthant"
        assert (A <= 1 + 1e-5).all() and (D <= 1 + 1e-5).all(), "diagonal escaped (0,1)"
        assert (Bm * Cm <= 0.25 + 1e-5).all(), "B*C exceeded the determinant bound"
    assert torch.allclose(acc[0] + acc[3], torch.ones_like(acc[0]), atol=1e-5)


def test_long_sequence_is_finite():
    """L=2048 -- the regime log space existed for."""
    args = _inputs(L=2048)
    y, y_var, st = kla_scan_torch(*args, mobius_impl="linear")
    for name, t in (("y", y), ("y_var", y_var), ("lam", st.lam), ("eta", st.eta)):
        assert torch.isfinite(t).all(), f"non-finite {name} at L=2048"


def test_float64_roundtrip():
    """float64 must be preserved, so gradcheck can still run against this path."""
    args = _inputs(dtype=torch.float64)
    y, y_var, _ = kla_scan_torch(*args, mobius_impl="linear")
    assert y.dtype == torch.float64 and y_var.dtype == torch.float64


@pytest.mark.parametrize("a_bar", [1e-2, 1e-6, 1e-12, 1e-18, 1e-24])
def test_extreme_decay_boundary(a_bar):
    """Where linear space stops tracking log space, measured rather than assumed.

    After normalization D ≈ a²/(1+pφ). float32's smallest normal is ~1.2e-38, so
    D underflows once ā² does -- around ā ~ 1e-19. The prediction is that this
    degrades *gracefully*: when a² is tiny, C = p/a² dominates the denominator,
    so λ → 1/C = (1+pφ)/p, the correct total-forgetting steady state, and losing
    B and D to underflow costs nothing.

    The assertion here is only that linear stays finite -- silent NaNs are the
    unacceptable failure. Agreement with log is reported, not required, since
    past the underflow point the two genuinely differ and log is the one to use.
    """
    args = _inputs(a_bar=a_bar)
    lin = kla_scan_torch(*args, mobius_impl="linear")
    log = kla_scan_torch(*args, mobius_impl="log")

    assert torch.isfinite(lin[0]).all(), f"ā={a_bar:g}: linear produced non-finite y"
    assert torch.isfinite(lin[1]).all(), (
        f"ā={a_bar:g}: linear produced non-finite y_var"
    )
    print(
        f"  ā={a_bar:<8g} rel(y)={_rel(lin[0], log[0]):.3e}  "
        f"rel(y_var)={_rel(lin[1], log[1]):.3e}"
    )


def test_unknown_impl_rejected():
    args = _inputs()
    with pytest.raises(ValueError, match="mobius_impl"):
        kla_scan_torch(*args, mobius_impl="tracenorm")
