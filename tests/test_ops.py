"""Numerical parity tests for the core KLA scan ops."""

import pytest
import torch

from kla.ops import KLAState, init_state, kla_scan, kla_scan_reference, kla_scan_torch, kla_step


def make_inputs(device, B=2, L=64, M=16, S=8):
    v = torch.randn(B, L, M, device=device)
    lambda_v = torch.rand(B, L, M, device=device) + 0.5  # value precision Λ^v (positive)
    k = torch.randn(B, L, S, device=device)
    q = torch.randn(B, L, S, device=device)
    a = torch.rand(M, S, device=device) * 0.5 + 0.4  # discrete decay, time-invariant
    p = torch.rand(M, S, device=device) * 0.05  # process noise, time-invariant
    return v, lambda_v, k, q, a, p


@pytest.mark.parametrize("scan_impl", ["doubling", "associative", "sequential"])
def test_parallel_matches_reference(device, scan_impl):
    inputs = make_inputs(device)
    y_ref, v_ref, s_ref = kla_scan_reference(*inputs)
    y, v, s = kla_scan_torch(*inputs, scan_impl=scan_impl)
    torch.testing.assert_close(y, y_ref, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(v, v_ref, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(s.lam, s_ref.lam, atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(s.eta, s_ref.eta, atol=1e-3, rtol=1e-4)


@pytest.mark.parametrize("decode_from_prior", [False, True])
def test_step_matches_scan(device, decode_from_prior):
    v, lambda_v, k, q, a, p = make_inputs(device, L=16)
    y_ref, v_ref, s_ref = kla_scan_torch(
        v, lambda_v, k, q, a, p, decode_from_prior=decode_from_prior
    )
    state = init_state(v.shape[0], v.shape[2], k.shape[2], device=device)
    ys, vs = [], []
    for t in range(v.shape[1]):
        y_t, v_t, state = kla_step(
            v[:, t], lambda_v[:, t], k[:, t], q[:, t], a, p, state,
            decode_from_prior=decode_from_prior,
        )
        ys.append(y_t)
        vs.append(v_t)
    torch.testing.assert_close(torch.stack(ys, 1), y_ref, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(torch.stack(vs, 1), v_ref, atol=2e-4, rtol=1e-4)


def test_chunked_scan_with_state(device):
    """Splitting the sequence and carrying state must match one full scan."""
    v, lambda_v, k, q, a, p = make_inputs(device)
    y_full, v_full, _ = kla_scan_torch(v, lambda_v, k, q, a, p)
    mid = v.shape[1] // 2
    y1, v1, s = kla_scan_torch(v[:, :mid], lambda_v[:, :mid], k[:, :mid], q[:, :mid], a, p)
    y2, v2, _ = kla_scan_torch(
        v[:, mid:], lambda_v[:, mid:], k[:, mid:], q[:, mid:], a, p, initial_state=s
    )
    torch.testing.assert_close(torch.cat([y1, y2], 1), y_full, atol=2e-4, rtol=1e-4)
    torch.testing.assert_close(torch.cat([v1, v2], 1), v_full, atol=2e-4, rtol=1e-4)


def test_gradients_match_reference(device):
    inputs_p = tuple(t.clone().requires_grad_(True) for t in make_inputs(device, L=32))
    inputs_r = tuple(t.detach().clone().requires_grad_(True) for t in inputs_p)
    y_p, v_p, _ = kla_scan_torch(*inputs_p, scan_impl="doubling")
    y_r, v_r, _ = kla_scan_reference(*inputs_r)
    (y_p.square().sum() + v_p.sum()).backward()
    (y_r.square().sum() + v_r.sum()).backward()
    for g_p, g_r in zip(inputs_p, inputs_r):
        torch.testing.assert_close(g_p.grad, g_r.grad, atol=2e-3, rtol=1e-3)


def test_dispatcher_backends(device):
    inputs = make_inputs(device, L=8)
    y, _, _ = kla_scan(*inputs, backend="auto")
    assert y.shape == (2, 8, 16)
    with pytest.raises(ValueError):
        kla_scan(*inputs, backend="nope")


needs_triton = pytest.mark.skipif(not torch.cuda.is_available(), reason="triton backend needs CUDA")


@needs_triton
def test_triton_backend_matches_reference():
    inputs = tuple(t.requires_grad_(True) for t in make_inputs("cuda"))
    inputs_r = tuple(t.detach().clone().requires_grad_(True) for t in inputs)
    y, v, s = kla_scan(*inputs, backend="triton")
    y_r, v_r, s_r = kla_scan_reference(*inputs_r)
    torch.testing.assert_close(y, y_r, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(v, v_r, atol=5e-4, rtol=5e-4)
    torch.testing.assert_close(s.lam, s_r.lam, atol=1e-3, rtol=1e-4)
    (y.square().sum() + v.sum()).backward()
    (y_r.square().sum() + v_r.sum()).backward()
    for g, g_r in zip(inputs, inputs_r):
        torch.testing.assert_close(g.grad, g_r.grad, atol=1e-2, rtol=1e-2)


@needs_triton
def test_triton_backend_chunked_state():
    v, lambda_v, k, q, a, p = make_inputs("cuda")
    y_full, _, _ = kla_scan(v, lambda_v, k, q, a, p, backend="triton")
    mid = v.shape[1] // 2
    y1, _, s = kla_scan(v[:, :mid], lambda_v[:, :mid], k[:, :mid], q[:, :mid], a, p, backend="triton")
    y2, _, _ = kla_scan(
        v[:, mid:], lambda_v[:, mid:], k[:, mid:], q[:, mid:], a, p,
        backend="triton", initial_state=s,
    )
    torch.testing.assert_close(torch.cat([y1, y2], 1), y_full, atol=5e-4, rtol=5e-4)


@needs_triton
def test_kla_layer_triton_backend():
    from kla import KLAConfig, KLALayer

    torch.manual_seed(0)
    x = torch.randn(2, 32, 64, device="cuda")
    layer_to = KLALayer(64, KLAConfig(d_state=8, backend="torch")).cuda()
    layer_tr = KLALayer(64, KLAConfig(d_state=8, backend="triton")).cuda()
    layer_tr.load_state_dict(layer_to.state_dict())
    y_to = layer_to(x)
    y_tr = layer_tr(x)
    torch.testing.assert_close(y_tr, y_to, atol=1e-4, rtol=1e-4)
    y_tr.sum().backward()
