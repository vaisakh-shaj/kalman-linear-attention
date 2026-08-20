"""The merged 3x3 formulation, pinned before any kernel is written.

``kla.ops.kla_ops._merged_combine`` folds the precision scan and the
information-vector scan into one associative combine, by writing the whole step
as a 3x3 linear map in homogeneous coordinates. The design rests on two claims,
and this file is where both are checked:

1. **It is the same map.** In float64 the merged scan must reproduce
   :func:`kla_scan_reference` -- which applies the filter recurrence directly
   and knows nothing about matrices -- to roundoff, for λ, η and the read-outs.
2. **Normalization is load-bearing.** The (3,3) entry accumulates a⁻ⁿ, which
   overflows float32 outright for a decaying filter.
   :func:`test_unnormalized_overflows` runs the unnormalized combine and shows
   it failing, so the argument for dividing by the 2x2's trace is a measurement
   rather than a claim. :func:`test_normalized_s_decays` shows the other half:
   normalized, that entry decays to zero, which is the filter forgetting its
   initial η.

The third thing worth knowing is what merging *costs* in float32. λ comes out
bit-identical, because the 2x2 block is the same block composed in the same
order. η is the coordinate that changes representation, and it comes out
slightly worse -- 1.5-3x, on a quantity already at the fp32 floor, and below the
error in λ that both paths share. PLAN.md predicted the opposite from a
prototype at much larger error scales; :func:`test_fp32_no_worse_than_two_scans`
records what actually happens here instead.
"""

import pytest
import torch

from kla.ops.kla_ops import (
    EPS,
    KLAState,
    _affine_combine,
    _broadcast_ap,
    _merged_combine,
    _merged_leaves,
    _merged_readout,
    _mobius_combine_tracenorm,
    _sufficient_stats,
    init_state,
    kla_scan_reference,
    kla_scan_torch,
)
from kla.ops.scan import doubling_scan

B, L, M, S = 2, 64, 4, 8


def _inputs(device="cpu", L=L, dtype=torch.float64, a_bar=None, seed=0):
    """Scan inputs, mirroring what the layer can actually emit."""
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=g)

    v, lambda_v = rnd(B, L, M), rnd(B, L, M).abs() + 0.5
    k, q = rnd(B, L, S), rnd(B, L, S)
    a = (
        torch.full((M, S), a_bar, device=device, dtype=dtype)
        if a_bar is not None
        else rnd(M, S).abs().clamp(0.05, 0.95)
    )
    p = rnd(M, S).abs() * 0.01 + 1e-4
    return [v, lambda_v, k, q, a, p]


def _state(args):
    v, _, k = args[0], args[1], args[2]
    return init_state(v.shape[0], v.shape[2], k.shape[2], dtype=v.dtype)


def _rel(x, y):
    return ((x - y).abs().max() / y.abs().max().clamp_min(1e-12)).item()


# ------------------------------------------------------- λ and η, not just y
#
# The read-outs contract over the state axis, which hides per-(m,s) error. These
# three helpers expose λ and η themselves, which is what the algebra is about.


def _reference_lambda_eta(v, lambda_v, k, a, p, lam0, eta0):
    """Sequential ground truth: apply the recurrence, compose nothing."""
    phi, r = _sufficient_stats(v, lambda_v, k)
    a_, p_ = _broadcast_ap(a, p, phi)
    a2 = (a_ * a_).clamp_min(EPS)
    lam, eta = lam0, eta0
    lams, etas = [], []
    for t in range(v.shape[1]):
        den = (a2[:, 0] + p_[:, 0] * lam).clamp_min(EPS)
        eta = (a_[:, 0] / den) * eta + r[:, t]
        lam = lam / den + phi[:, t]
        lams.append(lam)
        etas.append(eta)
    return torch.stack(lams, 1), torch.stack(etas, 1)


def _merged_lambda_eta(v, lambda_v, k, a, p, lam0, eta0):
    """One scan: compose the 3x3 leaves, then read λ = u/v and η = w/v off it."""
    phi, r = _sufficient_stats(v, lambda_v, k)
    a_, p_ = _broadcast_ap(a, p, phi)
    a2 = (a_ * a_).clamp_min(EPS)
    prefix = doubling_scan(_merged_combine, _merged_leaves(phi, r, a_, p_, a2), dim=1)
    return _merged_readout(prefix, lam0, eta0)


def _two_scan_lambda_eta(v, lambda_v, k, a, p, lam0, eta0):
    """Today's path: compose the 2x2, materialize λ, derive α, compose the affine."""
    phi, r = _sufficient_stats(v, lambda_v, k)
    a_, p_ = _broadcast_ap(a, p, phi)
    a2 = (a_ * a_).clamp_min(EPS)
    A = ((1.0 + p_ * phi) / a2).expand_as(phi)
    C = (p_ / a2).expand_as(phi)
    D = torch.ones_like(phi)
    pA, pB, pC, pD = doubling_scan(_mobius_combine_tracenorm, (A, phi, C, D), dim=1)
    lam0_ = lam0.unsqueeze(1)
    lam = (pA * lam0_ + pB) / (pC * lam0_ + pD).clamp_min(EPS)
    lam_prev = torch.cat((lam0.unsqueeze(1), lam[:, :-1]), dim=1)
    alpha = a_ / (a2 + p_ * lam_prev).clamp_min(EPS)
    pAlpha, pR = doubling_scan(_affine_combine, (alpha, r), dim=1)
    return lam, pAlpha * eta0.unsqueeze(1) + pR


# ------------------------------------------------------------ 1: the identity


def test_merged_matches_reference_float64():
    """The load-bearing correctness claim: same map, to float64 roundoff."""
    args = _inputs(dtype=torch.float64)
    st = _state(args)
    v, lambda_v, k, _, a, p = args
    lam_ref, eta_ref = _reference_lambda_eta(v, lambda_v, k, a, p, st.lam, st.eta)
    lam, eta = _merged_lambda_eta(v, lambda_v, k, a, p, st.lam, st.eta)
    assert _rel(lam, lam_ref) < 1e-12, "lambda"
    assert _rel(eta, eta_ref) < 1e-10, "eta"


@pytest.mark.parametrize("scan_impl", ["chunk", "doubling", "associative"])
def test_merged_scan_matches_reference_float64(scan_impl):
    """Through the op itself, at every way of grouping the combine.

    Normalization is applied per combine, so a scheme that only held for one
    grouping would be a real bug -- the invariance argument says λ = u/v and
    η = w/v are untouched by *any* common rescale, so all groupings must agree.
    """
    args = _inputs(dtype=torch.float64)
    y, yv, st = kla_scan_torch(*args, scan_impl=scan_impl, merged=True)
    ry, ryv, rst = kla_scan_reference(*args)
    assert _rel(y, ry) < 1e-10, "y"
    assert _rel(yv, ryv) < 1e-10, "y_var"
    assert _rel(st.lam, rst.lam) < 1e-10, "final lam"
    assert _rel(st.eta, rst.eta) < 1e-10, "final eta"


def test_merged_carries_state_and_prior():
    """The contract every implementation owes: state in and out, prior decode."""
    args = _inputs(dtype=torch.float64)
    g = torch.Generator().manual_seed(7)
    st0 = KLAState(
        lam=torch.rand(B, M, S, generator=g, dtype=torch.float64) + 0.5,
        eta=torch.randn(B, M, S, generator=g, dtype=torch.float64),
    )
    for prior in (False, True):
        y, yv, st = kla_scan_torch(
            *args, initial_state=st0, decode_from_prior=prior, merged=True
        )
        ry, ryv, rst = kla_scan_reference(
            *args, initial_state=st0, decode_from_prior=prior
        )
        assert _rel(y, ry) < 1e-10, f"y (prior={prior})"
        assert _rel(yv, ryv) < 1e-10, f"y_var (prior={prior})"
        assert _rel(st.eta, rst.eta) < 1e-10, f"eta (prior={prior})"


# ---------------------------------------------------- 2: normalization is real


def _unnormalized_combine(left, right):
    """:func:`_merged_combine` with the trace division removed. Overflows."""
    a1, b1, c1, d1, qa1, qb1, s1 = left
    a2, b2, c2, d2, qa2, qb2, s2 = right
    return (
        a2 * a1 + b2 * c1,
        a2 * b1 + b2 * d1,
        c2 * a1 + d2 * c1,
        c2 * b1 + d2 * d1,
        qa2 * a1 + qb2 * c1 + s2 * qa1,
        qa2 * b1 + qb2 * d1 + s2 * qb1,
        s2 * s1,
    )


def test_unnormalized_overflows():
    """Why the trace division is not cosmetic: without it, fp32 dies.

    s accumulates a⁻ⁿ. At a=0.5 over 200 steps that is 2²⁰⁰ ≈ 1.6e60 -- and the
    other entries grow too, so the composed map leaves float32's range entirely.
    Run in float32 precisely because that is the dtype every kernel uses.
    """
    args = _inputs(dtype=torch.float32, L=200, a_bar=0.5)
    v, lambda_v, k, _, a, p = args
    phi, r = _sufficient_stats(v, lambda_v, k)
    a_, p_ = _broadcast_ap(a, p, phi)
    a2 = (a_ * a_).clamp_min(EPS)
    leaves = _merged_leaves(phi, r, a_, p_, a2)

    raw = doubling_scan(_unnormalized_combine, leaves, dim=1)
    assert not torch.isfinite(raw[6]).all(), (
        "the unnormalized (3,3) entry was expected to overflow float32 here; "
        "if it no longer does, the normalization argument needs revisiting"
    )

    normalized = doubling_scan(_merged_combine, leaves, dim=1)
    for i, t in enumerate(normalized):
        assert torch.isfinite(t).all(), f"normalized entry {i} is not finite"


def test_normalized_s_decays():
    """Normalized, s decays to zero -- the filter forgetting its initial η.

    ∏τ grows faster than a⁻ⁿ, so the ratio goes the other way. That is not just
    numerically convenient, it is the right physics: after enough steps η_t is
    determined by the observations and not by η₀.
    """
    args = _inputs(dtype=torch.float64, L=200, a_bar=0.5)
    v, lambda_v, k, _, a, p = args
    phi, r = _sufficient_stats(v, lambda_v, k)
    a_, p_ = _broadcast_ap(a, p, phi)
    a2 = (a_ * a_).clamp_min(EPS)
    s = doubling_scan(_merged_combine, _merged_leaves(phi, r, a_, p_, a2), dim=1)[6]
    head, tail = s[:, 4].abs().max().item(), s[:, -1].abs().max().item()
    assert tail < head, f"s did not decay: {head} -> {tail}"
    assert tail < 1e-6, f"s should be negligible by L=200, got {tail}"


def test_2x2_block_stays_bounded():
    """The precision block is the block it always was: A, D ∈ (0,1), A+D = 1."""
    args = _inputs(dtype=torch.float32, L=200, a_bar=0.5)
    v, lambda_v, k, _, a, p = args
    phi, r = _sufficient_stats(v, lambda_v, k)
    a_, p_ = _broadcast_ap(a, p, phi)
    a2 = (a_ * a_).clamp_min(EPS)
    prefix = doubling_scan(_merged_combine, _merged_leaves(phi, r, a_, p_, a2), dim=1)
    # From t=1 on. The prefix at t=0 is the bare leaf, which no combine has
    # touched and so no trace has divided -- there A=(1+pφ)/a² and D=1.
    A, Bm, C, D = (t[:, 1:] for t in prefix[:4])
    assert torch.all(A > 0) and torch.all(A < 1)
    assert torch.all(D > 0) and torch.all(D < 1)
    assert _rel(A + D, torch.ones_like(A)) < 1e-5
    # B·C < 1/4 -- the product bound the determinant argument gives.
    assert (Bm * C).max().item() < 0.25 + 1e-4


def _combine_6(left, right):
    """:func:`_merged_combine` carrying six values, with D recovered as 1 - A."""
    a1, b1, c1, d1, qa1, qb1, s1 = left
    a2, b2, c2, d2, qa2, qb2, s2 = right
    a = a2 * a1 + b2 * c1
    b = a2 * b1 + b2 * d1
    c = c2 * a1 + d2 * c1
    d = c2 * b1 + d2 * d1
    qa = qa2 * a1 + qb2 * c1 + s2 * qa1
    qb = qa2 * b1 + qb2 * d1 + s2 * qb1
    inv = 1.0 / (a + d).clamp_min(EPS)
    a = a * inv
    return a, b * inv, c * inv, 1.0 - a, qa * inv, qb * inv, s2 * s1 * inv


def test_reconstructing_D_is_not_free():
    """Why the carried map is seven values and not six.

    After a combine A + D = 1, so D looks redundant, and dropping it would put
    the merged aggregate at 24 bytes -- exactly what the two-scan pair costs, and
    the difference between a wash and a regression for ``mps_merged_pscan``,
    whose doubling rounds are bandwidth-bound on precisely this array.

    It does not work, and not marginally. D is the *small* entry (roughly
    a²/(1+pφ)), so recovering it as 1 - A gives it an absolute error of one ulp
    of A -- a relative error of eps/D, in the entry λ is most sensitive to.
    Measured in float32 against a float64 reference, relative error in λ::

                       carried      reconstructed
        a=0.30  L=128   5.0e-7           2.7e-4
        a=0.70  L=128   4.2e-7           1.0e-4
        a=0.95  L=128   7.7e-7           3.8e-6

    Three orders of magnitude at strong decay, and it lands on λ, which every
    other implementation computes to 5e-7. One float per lane is much cheaper
    than that, so the kernels carry D and pad to eight.
    """
    for a_bar, floor in ((0.3, 20.0), (0.7, 20.0)):
        args = _inputs(dtype=torch.float64, L=128, a_bar=a_bar)
        st = _state(args)
        v, lambda_v, k, _, a, p = args
        lam_ref, _ = _reference_lambda_eta(v, lambda_v, k, a, p, st.lam, st.eta)

        args32 = [t.float() for t in args]
        st32 = _state(args32)
        v, lambda_v, k, _, a, p = args32
        phi, r = _sufficient_stats(v, lambda_v, k)
        a_, p_ = _broadcast_ap(a, p, phi)
        a2 = (a_ * a_).clamp_min(EPS)
        leaves = _merged_leaves(phi, r, a_, p_, a2)

        lam7, _ = _merged_readout(
            doubling_scan(_merged_combine, leaves, dim=1), st32.lam, st32.eta
        )
        lam6, _ = _merged_readout(
            doubling_scan(_combine_6, leaves, dim=1), st32.lam, st32.eta
        )
        err7 = _rel(lam7.double(), lam_ref)
        err6 = _rel(lam6.double(), lam_ref)
        assert err6 > floor * err7, (
            f"a={a_bar}: reconstructing D cost only {err6 / err7:.0f}x "
            f"({err6:.1e} vs {err7:.1e}). If it is genuinely this cheap now, the "
            "kernels could drop to a 24-byte aggregate -- see the docstring."
        )


# --------------------------------------------------- 3: no worse than two scans


@pytest.mark.parametrize("a_bar", [0.3, 0.7, 0.95])
@pytest.mark.parametrize("length", [8, 16, 64, 128])
def test_fp32_no_worse_than_two_scans(a_bar, length):
    """λ bit-identical, η at the same order of magnitude, neither drifting with L.

    The 2x2 block is composed identically, leaf for leaf and combine for
    combine, so λ must match to the bit -- and it does, which is the sharpest
    statement in this file. η is the part that actually changes representation,
    and it is where merging has to be shown not to cost anything.

    PLAN.md claimed η would come out *better* at every point, from a prototype
    whose two-scan errors ran to 3e-4. It does not reproduce on this repo's
    input distribution, where both paths sit at float32 roundoff and merged is
    consistently the slightly worse of the two. Measured against a float64
    reference, relative error in η, over a=0.3/0.7/0.95 and L=8..512::

        merged     1.4e-7 .. 4.9e-7
        two-scan   0.9e-7 .. 3.4e-7

    So merged costs a factor of roughly 1.5-3x on η, on a quantity that is
    already at the fp32 floor -- both stay below the error in λ itself, which is
    2e-7..8e-7 over the same grid and which *both* paths share exactly. Neither
    grows with sequence length, which is the property that would have mattered.
    The bound below is set from that measurement rather than from the claim.
    """
    args64 = _inputs(dtype=torch.float64, L=length, a_bar=a_bar)
    st64 = _state(args64)
    v, lambda_v, k, _, a, p = args64
    lam_ref, eta_ref = _reference_lambda_eta(v, lambda_v, k, a, p, st64.lam, st64.eta)

    args32 = [t.float() for t in args64]
    st32 = _state(args32)
    v, lambda_v, k, _, a, p = args32
    lam_m, eta_m = _merged_lambda_eta(v, lambda_v, k, a, p, st32.lam, st32.eta)
    lam_2, eta_2 = _two_scan_lambda_eta(v, lambda_v, k, a, p, st32.lam, st32.eta)

    assert torch.equal(lam_m, lam_2), "the 2x2 block is not composed identically"

    err_m = _rel(eta_m.double(), eta_ref)
    err_2 = _rel(eta_2.double(), eta_ref)
    lam_err = _rel(lam_m.double(), lam_ref)
    assert err_m < 2e-6, f"merged eta left the fp32 floor: {err_m:.2e}"
    assert err_m <= max(5 * err_2, 2e-7), (
        f"merged eta {err_m:.2e} vs two-scan {err_2:.2e}: the gap widened beyond "
        "the measured 1.5-3x, which means something other than roundoff"
    )
    assert err_m <= 4 * lam_err, (
        f"merged eta {err_m:.2e} now dominates lambda {lam_err:.2e}, which both "
        "paths compute identically"
    )
