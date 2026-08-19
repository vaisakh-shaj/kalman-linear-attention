"""Flag-space and statefulness tests for the unified KLALayer."""

import pytest
import torch

from kla import KLAConfig, KLALayer


def _decode_parity(layer: KLALayer, x: torch.Tensor) -> float:
    layer.eval()
    with torch.no_grad():
        full = layer(x)
        full = full[0] if isinstance(full, tuple) else full
        state = layer.init_state(x.shape[0], device=x.device)
        outs = []
        for t in range(x.shape[1]):
            out, state = layer(x[:, t : t + 1], state=state)
            out = out[0] if isinstance(out, tuple) else out
            outs.append(out)
        inc = torch.cat(outs, dim=1)
    return (full - inc).abs().max().item()


@pytest.mark.parametrize("discretization", ["ou", "zoh"])
def test_core_ablations_train_and_decode(device, discretization):
    cfg = KLAConfig(
        d_state=4,
        discretization=discretization,
    )
    layer = KLALayer(32, cfg).to(device)
    x = torch.randn(2, 16, 32, device=device)
    y = layer(x)
    assert y.shape == x.shape and torch.isfinite(y).all()
    y.sum().backward()
    assert all(
        p.grad is None or torch.isfinite(p.grad).all() for p in layer.parameters()
    )
    assert _decode_parity(layer, x) < 1e-4


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(use_conv=False),
        dict(use_gating=False),
        dict(use_lambda_skip=False),
        dict(lambda_skip_mode="vector"),
        dict(qk_norm=False),
        dict(zero_process_noise=True),
        dict(learnable_process_noise=False),
        dict(clip_value=5.0),
        dict(obs_var_max=10.0),
        dict(decode_from_prior=True),
        dict(checkpoint_ssm=True),
        dict(backend="torch_unfused_pscan"),
        dict(backend="torch_unfused_recurrent"),
        dict(expand=1.5),
        dict(bias=True),
    ],
)
def test_flag_space(device, kwargs):
    layer = KLALayer(32, KLAConfig(d_state=4, **kwargs)).to(device)
    x = torch.randn(2, 12, 32, device=device)
    y = layer(x)
    y = y[0] if isinstance(y, tuple) else y
    y.sum().backward()
    assert torch.isfinite(y).all()
    assert _decode_parity(layer, x) < 1e-4


def test_return_variance(device):
    layer = KLALayer(32, KLAConfig(d_state=4, return_variance=True)).to(device)
    x = torch.randn(2, 12, 32, device=device)
    y, y_var = layer(x)
    assert y.shape == y_var.shape == x.shape
    assert (y_var >= 0).all()


def test_chunked_prefill_matches_full(device):
    layer = KLALayer(32, KLAConfig(d_state=4)).to(device).eval()
    x = torch.randn(2, 24, 32, device=device)
    with torch.no_grad():
        full = layer(x)
        state = layer.init_state(2, device=device)
        y1, state = layer(x[:, :10], state=state)
        y2, state = layer(x[:, 10:], state=state)
    torch.testing.assert_close(torch.cat([y1, y2], 1), full, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_bf16_autocast():
    layer = KLALayer(64, KLAConfig(d_state=8)).cuda()
    x = torch.randn(2, 32, 64, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = layer(x)
    assert torch.isfinite(y).all()
    y.float().sum().backward()
