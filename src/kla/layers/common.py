"""Shared building blocks for KLA and baseline layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACTIVATIONS = {
    "silu": F.silu,
    "gelu": F.gelu,
    "relu": F.relu,
}


def get_activation(name: str):
    try:
        return _ACTIVATIONS[name]
    except KeyError:
        raise ValueError(
            f"Unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}"
        ) from None


def l2_norm(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize over the last dim (pure torch; portable across platforms)."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def inv_softplus_dt_init(
    shape: tuple[int, ...],
    dt_min: float,
    dt_max: float,
    dt_init_floor: float,
) -> torch.Tensor:
    """Mamba-style Δ init: log-uniform in [dt_min, dt_max], stored pre-softplus."""
    import math

    dt = torch.exp(
        torch.rand(*shape) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    ).clamp_min(dt_init_floor)
    return dt + torch.log(-torch.expm1(-dt))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class CausalConv1d(nn.Module):
    """Depthwise causal conv over the time axis with explicit state handling.

    Input/output are [B, L, D]. The recurrent state is the last
    ``kernel_size - 1`` pre-conv inputs, shaped [B, D, kernel_size - 1].
    """

    def __init__(
        self, dim: int, kernel_size: int, activation: str = "silu", bias: bool = True
    ):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.act = get_activation(activation)
        self.weight = nn.Parameter(torch.empty(dim, 1, kernel_size))
        self.bias = nn.Parameter(torch.empty(dim)) if bias else None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def init_state(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(
            batch, self.dim, self.kernel_size - 1, device=device, dtype=dtype
        )

    def forward(self, x: torch.Tensor, conv_state: torch.Tensor | None = None):
        """Returns (y [B, L, D], new_conv_state)."""
        B, L, D = x.shape
        x_t = x.transpose(1, 2)  # [B, D, L]
        if conv_state is None:
            conv_state = self.init_state(B, device=x.device, dtype=x_t.dtype)
        x_pad = torch.cat((conv_state.to(x_t.dtype), x_t), dim=-1)
        y = F.conv1d(x_pad, self.weight, self.bias, groups=self.dim)
        new_state = x_pad[..., x_pad.shape[-1] - (self.kernel_size - 1) :]
        return self.act(y).transpose(1, 2), new_state


class MLP(nn.Module):
    """Channel mixer: SwiGLU or plain GELU MLP."""

    def __init__(
        self, dim: int, ratio: float = 4.0, kind: str = "swiglu", bias: bool = False
    ):
        super().__init__()
        hidden = int(dim * ratio)
        self.kind = kind
        if kind == "swiglu":
            hidden = int(2 * hidden / 3)
            self.w12 = nn.Linear(dim, 2 * hidden, bias=bias)
            self.w3 = nn.Linear(hidden, dim, bias=bias)
        elif kind == "gelu":
            self.w1 = nn.Linear(dim, hidden, bias=bias)
            self.w2 = nn.Linear(hidden, dim, bias=bias)
        else:
            raise ValueError(f"Unknown MLP kind {kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kind == "swiglu":
            gate, up = self.w12(x).chunk(2, dim=-1)
            return self.w3(F.silu(gate) * up)
        return self.w2(F.gelu(self.w1(x)))
