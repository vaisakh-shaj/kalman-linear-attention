"""Kalman Linear Attention (KLA): attention as an exact parallel Kalman filter.

Quick start::

    import torch
    from kla import KLAConfig, KLALayer

    layer = KLALayer(d_model=512, config=KLAConfig(d_state=16))
    y = layer(torch.randn(2, 128, 512))          # [2, 128, 512]

Every KLA variant is one config away - see :class:`KLAConfig` and the readme.
Run ``python -m kla --check`` to test the scan backends on this machine.
"""

from kla.configs import KLAConfig, ModelConfig
from kla.layers import KLALayer, KLALayerState
from kla.models import SequenceModel, build_mixer, register_mixer
from kla.ops import (
    KLAState,
    backend_names,
    kla_scan,
    kla_scan_reference,
    kla_step,
    resolve_backend,
)

__version__ = "0.1.0"

__all__ = [
    "KLAConfig",
    "KLALayer",
    "KLALayerState",
    "KLAState",
    "ModelConfig",
    "SequenceModel",
    "backend_names",
    "build_mixer",
    "kla_scan",
    "kla_scan_reference",
    "kla_step",
    "register_mixer",
    "resolve_backend",
    "__version__",
]
