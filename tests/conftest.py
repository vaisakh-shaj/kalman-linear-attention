import pytest
import torch


def pytest_generate_tests(metafunc):
    if "device" in metafunc.fixturenames:
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda")
        metafunc.parametrize("device", devices)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
