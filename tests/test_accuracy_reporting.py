"""Check strict acceptance and legacy reporting without a GPU."""

import pytest
import torch

from kla import __main__ as cli
from kla.ops.accuracy import metrics


def test_accuracy_rule():
    ref = torch.tensor([1.0, 0.0])
    assert metrics(ref + 1e-5, ref)['passed']
    assert not metrics(ref + 0.01, ref)['passed']
    assert not metrics(torch.tensor([float('nan'), 0.0]), ref)['passed']


@pytest.mark.parametrize('backend,status', [
    ('cuda_v2_2', 'KNOWN FAIL'), ('cuda_v2_1', 'KNOWN FAIL'),
    ('cuda_v3', 'FAIL'), ('cuda_v3_fast', 'FAIL'), ('cuda', 'FAIL'),
])
def test_gradient_error_is_never_a_pass(monkeypatch, backend, status):
    import kla.ops

    def reference(*args):
        y = sum(t.sum() for t in args)
        return y, y, None

    def incorrect(*args, **kwargs):
        y, var, _ = reference(*args)
        p = args[-1]
        return y + 0.1 * (p.sum() - p.detach().sum()), var, None

    monkeypatch.setattr(kla.ops, 'kla_scan_reference', reference)
    monkeypatch.setattr(kla.ops, 'kla_scan', incorrect)
    result, detail = cli._gradients(backend, 'cpu')
    assert result == status
    assert f'{status} p:' in detail
    assert 'relL2=' in detail


def test_legacy_nan_is_failure(monkeypatch):
    import kla.ops

    def scan(*args, **kwargs):
        y = sum(t.sum() for t in args) * float('nan')
        return y, y, None

    monkeypatch.setattr(kla.ops, 'kla_scan', scan)
    monkeypatch.setattr(kla.ops, 'kla_scan_reference', scan)
    assert cli._gradients('cuda_v2_2', 'cpu')[0] == 'FAIL'
