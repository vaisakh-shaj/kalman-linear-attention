"""Initialization, gradient, and checkpoint contracts for noise parameters."""
import pytest
import torch
from kla import KLAConfig, KLALayer, ModelConfig, SequenceModel


def test_process_noise_heuristic_and_gradient():
    layer = KLALayer(8, KLAConfig(d_state=4))
    a, p = layer._continuous_params()
    delta = torch.nn.functional.softplus(layer.delta.float()) + 1e-7
    torch.testing.assert_close(p, 3 * a.abs() * delta)
    p.sum().backward()
    torch.testing.assert_close(layer.p_log.grad, p)
    assert layer.delta.grad is None  # Coupled only at initialization.


@pytest.mark.parametrize('rank', ['full', 'auto', 2])
@pytest.mark.parametrize('bias', [False, True])
def test_constant_observation_head(rank, bias):
    cfg = KLAConfig(d_state=4, var_rank=rank, bias=bias)
    layer = KLALayer(8, cfg)
    for _ in range(2):
        _, precision, _, _ = layer._project_sensors(torch.randn(2, 7, layer.d_inner))
        torch.testing.assert_close(precision, torch.full_like(precision, 4))
        precision.sum().backward()
        assert torch.isfinite(layer.obs_noise_bias.grad).all()
        assert layer.obs_noise_bias.grad.abs().sum() > 0
        layer.reset_parameters()


@pytest.mark.parametrize('rank', ['full', 'auto', 2])
def test_model_initialization_preserves_cold_start(rank):
    model = SequenceModel(ModelConfig(vocab_size=16, d_model=8, n_layers=1),
                          KLAConfig(d_state=4, var_rank=rank, bias=True))
    layer = next(m for m in model.modules() if isinstance(m, KLALayer))
    _, precision, _, _ = layer._project_sensors(torch.randn(1, 5, layer.d_inner))
    torch.testing.assert_close(precision, torch.full_like(precision, 4))


def test_legacy_and_log_checkpoint_contracts():
    cfg = KLAConfig(d_state=4, process_noise_param='raw',
                    process_noise_init=0.01, obs_noise_init=None,
                    value_rank='full', var_rank='full')
    old = KLALayer(8, cfg)
    restored = KLALayer(8, cfg)
    restored.load_state_dict(old.state_dict())
    assert 'process_noise' in old.state_dict() and 'p_log' not in old.state_dict()
    assert 'obs_noise_bias' not in old.state_dict()
    torch.testing.assert_close(old._continuous_params()[1],
                               torch.full_like(old.process_noise, 0.01))
    new = KLALayer(8, KLAConfig(d_state=4))
    with pytest.raises(RuntimeError):
        new.load_state_dict(old.state_dict())
    new.load_state_dict(new.state_dict())


@pytest.mark.parametrize('noise_init', [0.02, 1])
def test_constant_log_noise(noise_init):
    layer = KLALayer(8, KLAConfig(d_state=4, process_noise_init=noise_init))
    torch.testing.assert_close(layer._continuous_params()[1],
                               torch.full_like(layer.p_log, noise_init))


@pytest.mark.parametrize('mode', ['learned', 'fixed', 'zero'])
@pytest.mark.parametrize('param', ['raw', 'log'])
def test_process_noise_modes(mode, param):
    layer = KLALayer(8, KLAConfig(d_state=4, process_noise_mode=mode,
                                process_noise_param=param, process_noise_init=0.02))
    name = 'p_log' if param == 'log' else 'process_noise'
    assert (name in dict(layer.named_parameters())) == (mode == 'learned')
    assert (name in dict(layer.named_buffers())) == (mode != 'learned')
    p = layer._continuous_params()[1]
    assert p.requires_grad == (mode == 'learned')
    if mode == 'zero':
        assert (p >= 0).all() and (p <= 1e-12).all()
    else:
        torch.testing.assert_close(p, torch.full_like(p, 0.02))
    if mode == 'learned':
        p.sum().backward()
        grad = getattr(layer, name).grad
        assert grad is not None and torch.isfinite(grad).all()
        assert (grad > 0).all()


@pytest.mark.parametrize('kwargs', [dict(obs_noise_init=0.0),
    dict(process_noise_mode='invalid'),
    dict(obs_noise_init=0.5, obs_var_max=0.2),
    *[dict(process_noise_init=value) for value in
      (-1, 0, float('nan'), float('inf'), 'constant', None, True)]])
def test_invalid_initialization(kwargs):
    with pytest.raises(ValueError):
        KLALayer(8, KLAConfig(d_state=4, **kwargs))
