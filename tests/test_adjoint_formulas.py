"""The shared adjoint formulas, checked against autograd in double precision.

Every fused kernel in this repo differentiates the *recurrence* rather than the
composed Moebius map -- see docs/implementations.md. That makes the whole
backward two reverse affine recurrences carrying one scalar each, and it is why
these gradients are exact where the cuda_v2_* ones are not.

The kernels that implement it (``chunk_kla_scan.metal``,
``kla.ops.kernels.triton.kla_scan_bwd``, and the CUDA cells) need a GPU to run.
The *formulas* do not, so they are pinned here, on CPU, in float64. Two of them
carry the whole design and are the ones worth stating outright:

1. ``alpha_t . eta_{t-1} = eta_t - r_t``
2. ``alpha_{t+1} . nu^eta_{t+1} = nu^eta_t - deta_t``

Both let a kernel avoid a tile shifted by one timestep, which triton has no
cheap primitive for and which the obvious alternatives buy by dividing through a
gain that is small exactly when the filter is forgetting. If a kernel's
gradients go wrong, run this first: it separates a mistake in the mathematics
from a mistake in the indexing.
"""

import pytest
import torch

EPS = 1e-12


def _fwd(msi, si, k, q, a, p, lam0, eta0, prior):
    L_N = msi.shape[1]
    a2 = torch.clamp(a * a, min=EPS)
    lam_prev = lam0
    eta_prev = eta0
    ys, yvs = [], []
    for t in range(L_N):
        phi = torch.clamp(si[:, t, :, None] * k[:, t, None, :] ** 2, min=EPS)
        r = msi[:, t, :, None] * k[:, t, None, :]
        den = torch.clamp(a2 + p * lam_prev, min=EPS)
        lam = lam_prev / den + phi
        alpha = a / den
        eta = alpha * eta_prev + r
        var = 1.0 / torch.clamp(lam, min=EPS)
        mean = eta * var
        if prior:
            mean_o, var_o = a * mean, a2 * var + p
        else:
            mean_o, var_o = mean, var
        ys.append((mean_o * q[:, t, None, :]).sum(-1))
        yvs.append((var_o * q[:, t, None, :] ** 2).sum(-1))
        lam_prev, eta_prev = lam, eta
    return torch.stack(ys, 1), torch.stack(yvs, 1), lam_prev, eta_prev


def _manual_bwd(msi, si, k, q, a, p, lam0, eta0, dy, dyv, dlam_fin, deta_fin, prior):
    L_N = msi.shape[1]
    a2 = torch.clamp(a * a, min=EPS)
    inv_a = a / a2
    # replay, collecting per-step quantities
    lam_prev, eta_prev = lam0, eta0
    L_, E_, LP_, DEN_, AL_, PHI_, R_, RAW_ = [], [], [], [], [], [], [], []
    for t in range(L_N):
        raw = si[:, t, :, None] * k[:, t, None, :] ** 2
        phi = torch.clamp(raw, min=EPS)
        r = msi[:, t, :, None] * k[:, t, None, :]
        den = torch.clamp(a2 + p * lam_prev, min=EPS)
        lam = lam_prev / den + phi
        alpha = a / den
        eta = alpha * eta_prev + r
        L_.append(lam)
        E_.append(eta)
        LP_.append(lam_prev)
        DEN_.append(den)
        AL_.append(alpha)
        PHI_.append(phi)
        R_.append(r)
        RAW_.append(raw)
        lam_prev, eta_prev = lam, eta

    dmsi = torch.zeros_like(msi)
    dsi = torch.zeros_like(si)
    dk = torch.zeros_like(k)
    dq = torch.zeros_like(q)
    da = torch.zeros_like(a)
    dp = torch.zeros_like(p)
    nu_lam = torch.zeros_like(lam0)
    nu_eta = torch.zeros_like(eta0)

    for t in range(L_N - 1, -1, -1):
        lam, eta, lp, den, r, phi, raw = (
            L_[t],
            E_[t],
            LP_[t],
            DEN_[t],
            R_[t],
            PHI_[t],
            RAW_[t],
        )
        var = 1.0 / torch.clamp(lam, min=EPS)
        mean = eta * var
        mean_o, var_o = (a * mean, a2 * var + p) if prior else (mean, var)
        qt = q[:, t, None, :]
        dyt, dyvt = dy[:, t, :, None], dyv[:, t, :, None]

        dq[:, t] += (dyt * mean_o + dyvt * 2.0 * qt * var_o).sum(1)
        d_mean_o = dyt * qt
        d_var_o = dyvt * qt * qt
        if prior:
            d_mean, d_var = d_mean_o * a, d_var_o * a2
            da += (d_mean_o * mean + d_var_o * 2.0 * a * var).sum(0)
            dp += d_var_o.sum(0)
        else:
            d_mean, d_var = d_mean_o, d_var_o

        d_eta_dir = d_mean * var
        d_var_tot = d_var + d_mean * eta
        d_lam_dir = -d_var_tot * var * var
        if t == L_N - 1:
            d_lam_dir = d_lam_dir + dlam_fin
            d_eta_dir = d_eta_dir + deta_fin

        # the two reverse recurrences, using den_{t+1} = a2 + p*lam_t
        den_next = torch.clamp(a2 + p * lam, min=EPS)
        alpha_next = a / den_next
        g_next = a2 / (den_next * den_next)
        nu_eta = d_eta_dir + alpha_next * nu_eta
        alpha_nu_next = nu_eta - d_eta_dir  # == alpha_{t+1}*nu_eta_{t+1}
        src_lam = d_lam_dir - eta * (p / den_next) * alpha_nu_next
        nu_lam = src_lam + g_next * nu_lam

        dphi = torch.where(raw > EPS, nu_lam, torch.zeros_like(nu_lam))
        dr = nu_eta
        dsi[:, t] += (dphi * k[:, t, None, :] ** 2).sum(-1)
        dmsi[:, t] += (dr * k[:, t, None, :]).sum(-1)
        dk[:, t] += (
            dphi * 2.0 * si[:, t, :, None] * k[:, t, None, :] + dr * msi[:, t, :, None]
        ).sum(1)

        a_eta_prev = eta - r  # == alpha_t * eta_{t-1}
        da += (
            nu_lam * (-lp * 2.0 * a / (den * den))
            + nu_eta * a_eta_prev * (inv_a - 2.0 * a / den)
        ).sum(0)
        dp += (
            nu_lam * (-lp * lp / (den * den)) + nu_eta * a_eta_prev * (-lp / den)
        ).sum(0)

    den0 = torch.clamp(a2 + p * lam0, min=EPS)
    alpha0 = a / den0
    dlam0 = (a2 / (den0 * den0)) * nu_lam - eta0 * (p / den0) * (alpha0 * nu_eta)
    deta0 = alpha0 * nu_eta
    return dmsi, dsi, dk, dq, da, dp, dlam0, deta0


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("L", [1, 5, 23])
def test_adjoint_matches_autograd(prior, L):
    """The hand-derived backward is the exact adjoint, to machine precision."""
    torch.manual_seed(0)
    B, M, S = 2, 3, 4
    args = [
        torch.randn(B, L, M),
        torch.rand(B, L, M) + 0.5,
        torch.randn(B, L, S) * 0.5,
        torch.randn(B, L, S) * 0.5,
        torch.rand(M, S) * 0.5 + 0.4,
        torch.rand(M, S) * 0.05 + 0.01,
        torch.rand(B, M, S) + 0.5,
        torch.randn(B, M, S) * 0.3,
    ]
    args = [t.double().requires_grad_(True) for t in args]

    y, yv, lam_fin, eta_fin = _fwd(*args, prior)
    seeds = [torch.randn_like(t) for t in (y, yv, lam_fin, eta_fin)]
    torch.autograd.backward([y, yv, lam_fin, eta_fin], seeds)

    got = _manual_bwd(*[t.detach() for t in args], *seeds, prior)
    for name, g, t in zip("msi si k q a p lam0 eta0".split(), got, args):
        scale = t.grad.abs().max().item() + 1e-9
        err = (g - t.grad).abs().max().item() / scale
        assert err < 1e-10, f"d{name} off by {err:.2e}"


@pytest.mark.parametrize("prior", [False])
def test_the_two_shift_free_identities_hold(prior):
    """The identities the kernels lean on, asserted directly.

    A kernel that gets one of these wrong still produces plausible gradients --
    they are only wrong by a factor that varies with the decay -- so they are
    worth pinning separately from the end-to-end check above.
    """
    torch.manual_seed(0)
    B, L, M, S = 2, 12, 3, 4
    EPS_ = 1e-12
    msi = torch.randn(B, L, M).double()
    si = (torch.rand(B, L, M) + 0.5).double()
    k = (torch.randn(B, L, S) * 0.5).double()
    a = (torch.rand(M, S) * 0.5 + 0.4).double()
    p = (torch.rand(M, S) * 0.05 + 0.01).double()
    lam, eta = (
        (torch.rand(B, M, S) + 0.5).double(),
        (torch.randn(B, M, S) * 0.3).double(),
    )
    a2 = torch.clamp(a * a, min=EPS_)

    for t in range(L):
        phi = torch.clamp(si[:, t, :, None] * k[:, t, None, :] ** 2, min=EPS_)
        r = msi[:, t, :, None] * k[:, t, None, :]
        den = torch.clamp(a2 + p * lam, min=EPS_)
        alpha = a / den
        eta_next = alpha * eta + r
        # (1) alpha_t . eta_{t-1} = eta_t - r_t
        torch.testing.assert_close(alpha * eta, eta_next - r, atol=1e-14, rtol=1e-12)
        lam, eta = lam / den + phi, eta_next

    # (2) nu_eta_t - deta_t = alpha_{t+1} . nu_eta_{t+1}, by construction of the
    # reverse recurrence nu_t = deta_t + alpha_{t+1}.nu_{t+1}.
    deta = torch.randn(B, L, M, S).double()
    alphas = torch.rand(B, L, M, S).double() * 0.9 + 0.05
    nu = torch.zeros(B, M, S).double()
    for t in range(L - 1, -1, -1):
        nu_next = nu
        nu = deta[:, t] + alphas[:, t] * nu_next
        torch.testing.assert_close(nu - deta[:, t], alphas[:, t] * nu_next)


CHUNK = 16  # the checkpoint stride every fused kernel writes at


def _checkpointed_bwd(msi, si, k, q, a, p, lam0, eta0, dy, dyv, dlf, def_, prior):
    """The adjoint again, structured the way the kernels actually run it.

    Forward writes (lambda, eta) every CHUNK steps; the backward walks chunks in
    reverse, replays CHUNK steps from each checkpoint into fixed-size buffers,
    then runs the adjoint back down over them. Tail iterations past ``L`` are
    masked rather than trimmed, because a constant trip count is what keeps the
    replay buffers in registers.

    This is the structural half of the design, and the half that fails by one
    rather than by a factor: a checkpoint stored after the step instead of
    before, a replay that starts one late, a tail that advances the recurrence.
    """
    L_N = msi.shape[1]
    a2 = torch.clamp(a * a, min=EPS)
    n_ck = -(-L_N // CHUNK)

    # -- forward, checkpointing the state entering every CHUNK-th step --------
    lam, eta = lam0, eta0
    lam_ck, eta_ck = [], []
    for t in range(L_N):
        if t % CHUNK == 0:
            lam_ck.append(lam)
            eta_ck.append(eta)
        phi = torch.clamp(si[:, t, :, None] * k[:, t, None, :] ** 2, min=EPS)
        den = torch.clamp(a2 + p * lam, min=EPS)
        lam = lam / den + phi
        eta = (a / den) * eta + msi[:, t, :, None] * k[:, t, None, :]
    assert len(lam_ck) == n_ck

    dmsi = torch.zeros_like(msi)
    dsi = torch.zeros_like(si)
    dk = torch.zeros_like(k)
    dq = torch.zeros_like(q)
    da = torch.zeros_like(a)
    dp = torch.zeros_like(p)
    nu_lam = torch.zeros_like(lam0)
    nu_eta = torch.zeros_like(eta0)

    for c in range(n_ck - 1, -1, -1):
        t0 = c * CHUNK
        n = min(CHUNK, L_N - t0)

        # -- replay, constant trip count, data masked -------------------------
        lam, eta = lam_ck[c], eta_ck[c]
        lam_h, eta_h = [], []
        for i in range(CHUNK):
            lam_h.append(lam)
            eta_h.append(eta)
            if i >= n:
                continue
            t = t0 + i
            phi = torch.clamp(si[:, t, :, None] * k[:, t, None, :] ** 2, min=EPS)
            den = torch.clamp(a2 + p * lam, min=EPS)
            lam = lam / den + phi
            eta = (a / den) * eta + msi[:, t, :, None] * k[:, t, None, :]

        # -- adjoint back down the chunk --------------------------------------
        for i in range(CHUNK - 1, -1, -1):
            if i >= n:
                continue
            t = t0 + i
            lam_prev, eta_prev = lam_h[i], eta_h[i]
            raw = si[:, t, :, None] * k[:, t, None, :] ** 2
            phi = torch.clamp(raw, min=EPS)
            r = msi[:, t, :, None] * k[:, t, None, :]
            den = torch.clamp(a2 + p * lam_prev, min=EPS)
            lam_t = lam_prev / den + phi
            eta_t = (a / den) * eta_prev + r

            var = 1.0 / torch.clamp(lam_t, min=EPS)
            mean = eta_t * var
            mean_o, var_o = (a * mean, a2 * var + p) if prior else (mean, var)
            qt = q[:, t, None, :]
            dyt, dyvt = dy[:, t, :, None], dyv[:, t, :, None]

            dq[:, t] += (dyt * mean_o + dyvt * 2.0 * qt * var_o).sum(1)
            d_mean_o, d_var_o = dyt * qt, dyvt * qt * qt
            if prior:
                d_mean, d_var = d_mean_o * a, d_var_o * a2
                da += (d_mean_o * mean + d_var_o * 2.0 * a * var).sum(0)
                dp += d_var_o.sum(0)
            else:
                d_mean, d_var = d_mean_o, d_var_o

            d_eta_dir = d_mean * var
            d_lam_dir = -(d_var + d_mean * eta_t) * var * var
            if t == L_N - 1:
                d_lam_dir = d_lam_dir + dlf
                d_eta_dir = d_eta_dir + def_

            den_next = torch.clamp(a2 + p * lam_t, min=EPS)
            src_lam = d_lam_dir - eta_t * a * p * nu_eta / (den_next * den_next)
            nu_eta = d_eta_dir + (a / den_next) * nu_eta
            nu_lam = src_lam + (a2 / (den_next * den_next)) * nu_lam

            dphi = torch.where(raw > EPS, nu_lam, torch.zeros_like(nu_lam))
            dsi[:, t] += (dphi * k[:, t, None, :] ** 2).sum(-1)
            dmsi[:, t] += (nu_eta * k[:, t, None, :]).sum(-1)
            dk[:, t] += (
                dphi * 2.0 * si[:, t, :, None] * k[:, t, None, :]
                + nu_eta * msi[:, t, :, None]
            ).sum(1)

            da += (
                nu_lam * (-lam_prev * 2.0 * a / (den * den))
                + nu_eta * eta_prev * ((den - 2.0 * a2) / (den * den))
            ).sum(0)
            dp += (
                nu_lam * (-lam_prev * lam_prev / (den * den))
                + nu_eta * eta_prev * (-a * lam_prev / (den * den))
            ).sum(0)

    den0 = torch.clamp(a2 + p * lam0, min=EPS)
    dlam0 = (a2 * nu_lam - eta0 * a * p * nu_eta) / (den0 * den0)
    deta0 = (a / den0) * nu_eta
    return dmsi, dsi, dk, dq, da, dp, dlam0, deta0


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("L", [1, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK, 40])
def test_checkpointed_replay_matches_autograd(prior, L):
    """The chunked replay is the exact adjoint at every alignment.

    The lengths are the ones that catch an off-by-one: shorter than a chunk,
    one under, exactly on the boundary, one over, exactly two, and a ragged
    tail.
    """
    torch.manual_seed(1)
    B, M, S = 2, 3, 4
    args = [
        torch.randn(B, L, M),
        torch.rand(B, L, M) + 0.5,
        torch.randn(B, L, S) * 0.5,
        torch.randn(B, L, S) * 0.5,
        torch.rand(M, S) * 0.5 + 0.4,
        torch.rand(M, S) * 0.05 + 0.01,
        torch.rand(B, M, S) + 0.5,
        torch.randn(B, M, S) * 0.3,
    ]
    args = [t.double().requires_grad_(True) for t in args]

    y, yv, lam_fin, eta_fin = _fwd(*args, prior)
    seeds = [torch.randn_like(t) for t in (y, yv, lam_fin, eta_fin)]
    torch.autograd.backward([y, yv, lam_fin, eta_fin], seeds)

    got = _checkpointed_bwd(*[t.detach() for t in args], *seeds, prior)
    for name, g, t in zip("msi si k q a p lam0 eta0".split(), got, args):
        err = (g - t.grad).abs().max().item() / (t.grad.abs().max().item() + 1e-9)
        assert err < 1e-10, f"L={L} prior={prior}: d{name} off by {err:.2e}"
