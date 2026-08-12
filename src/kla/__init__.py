"""Kalman Linear Attention (KLA): attention as an exact parallel Kalman filter.

Quick start::

    import torch
    from kla import KLAConfig, KLALayer

    layer = KLALayer(d_model=512, config=KLAConfig(d_state=16))
    y = layer(torch.randn(2, 128, 512))          # [2, 128, 512]

Every KLA variant is one config away - see :class:`KLAConfig` and the readme.
Run ``python -m kla`` to print which scan backends are usable on this machine.
"""

from kla.configs import KLAConfig, ModelConfig
from kla.layers import KLALayer, KLALayerState
from kla.models import SequenceModel, build_mixer, register_mixer
from kla.ops import KLAState, kla_scan, kla_scan_reference, kla_step

__version__ = "0.1.0"


def backend_info() -> dict:
    """Which scan backends are usable here, and why not when they are not.

    Never raises and never imports triton eagerly, so it is safe to call in a
    notebook cell just to see what you have.
    """
    import torch

    from kla.ops.kla_ops import _triton_available

    cuda = torch.cuda.is_available()
    triton_ok = bool(cuda and _triton_available())
    return {
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "resolved_auto": "triton" if triton_ok else "torch",
        "backends": {
            "torch": (True, "always available; the reference implementation"),
            "triton": (
                triton_ok,
                "ready" if triton_ok
                else ("needs a CUDA device" if not cuda else "triton not importable"),
            ),
            "cuda": (
                cuda,
                "compiled on first use; needs a CUDA toolchain (nvcc)" if cuda
                else "needs a CUDA device",
            ),
        },
        "log_space_on": ["torch"],
    }


__all__ = [
    "KLAConfig",
    "KLALayer",
    "KLALayerState",
    "KLAState",
    "ModelConfig",
    "SequenceModel",
    "backend_info",
    "build_mixer",
    "register_mixer",
    "kla_scan",
    "kla_scan_reference",
    "kla_step",
    "__version__",
]
