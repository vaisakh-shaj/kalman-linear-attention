"""CUDA setup diagnostics must work before a GPU or compiler is available."""

import pytest
from torch.utils import cpp_extension

from kla.ops import cuda_backend


@pytest.fixture(autouse=True)
def clear_extension_cache():
    cuda_backend._load_extension.cache_clear()
    yield
    cuda_backend._load_extension.cache_clear()


@pytest.mark.parametrize("toolkit_exists", [False, True])
def test_missing_toolkit_explains_recovery(monkeypatch, tmp_path, toolkit_exists):
    monkeypatch.setattr(cpp_extension, "CUDA_HOME",
                        str(tmp_path) if toolkit_exists else None)

    def unexpected_build(**kwargs):
        pytest.fail("Missing toolkit should be diagnosed before compilation")

    monkeypatch.setattr(cpp_extension, "load", unexpected_build)
    with pytest.raises(RuntimeError) as error:
        cuda_backend._load_extension()
    message = str(error.value)
    assert "was not found" in message
    assert "CUDA build:" in message
    assert "CUDA_HOME" in message
    assert "Restart Python" in message
    assert "--test-backends cuda_v3_fast" in message
    assert "docs/backends.md#cuda-setup" in message


def fake_toolkit(monkeypatch, tmp_path):
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "nvcc").touch()
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(tmp_path))


def test_build_error_keeps_original_diagnostics(monkeypatch, tmp_path):
    fake_toolkit(monkeypatch, tmp_path)
    original = RuntimeError("host compiler rejected the build")

    def failed_build(**kwargs):
        raise original

    monkeypatch.setattr(cpp_extension, "load", failed_build)
    with pytest.raises(RuntimeError, match="KLA_JIT_VERBOSE=1") as error:
        cuda_backend._load_extension()
    assert error.value.__cause__ is original


def test_available_toolkit_builds_and_caches(monkeypatch, tmp_path):
    fake_toolkit(monkeypatch, tmp_path)
    calls = []
    extension = object()

    def successful_build(**kwargs):
        calls.append(kwargs)
        return extension

    monkeypatch.setattr(cpp_extension, "load", successful_build)
    assert cuda_backend._load_extension() is extension
    assert cuda_backend._load_extension() is extension
    assert len(calls) == 1
    assert calls[0]["name"] == "kla_matmul_scan_cuda_v3_fast"
