"""Unified sequence model: embedding + N pre-norm blocks + LM head.

Each block is ``x += mixer(norm(x))`` followed by an optional channel mixer
``x += mlp(norm(x))``. The sequence mixer is built from the config object
passed in via a small registry: the package registers
:class:`~kla.configs.KLAConfig`; the repo-level ``experiments`` package
registers baseline mixers (Mamba, attention). All mixers share the interface
``forward(x, state=None, return_state=False)`` and ``init_state(batch)``, so
training frameworks see one standard ``forward(input_ids) -> logits`` model
regardless of architecture.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

import torch
import torch.nn as nn

from kla.configs import KLAConfig, ModelConfig
from kla.layers.common import MLP, RMSNorm
from kla.layers.kla_layer import KLALayer
from kla.models.readout import MarginalReadout

MixerBuilder = Callable[[int, Any], nn.Module]
_MIXER_REGISTRY: dict[type, MixerBuilder] = {}


def register_mixer(config_type: type, builder: MixerBuilder) -> None:
    """Register a builder ``(d_model, cfg) -> nn.Module`` for a config dataclass."""
    _MIXER_REGISTRY[config_type] = builder


def build_mixer(d_model: int, config: Any) -> nn.Module:
    builder = _MIXER_REGISTRY.get(type(config))
    if builder is None:
        raise TypeError(
            f"No mixer registered for config type {type(config).__name__}; "
            "call kla.models.register_mixer first."
        )
    return builder(d_model, config)


register_mixer(KLAConfig, KLALayer)


class Block(nn.Module):
    def __init__(self, d_model: int, mixer_config: Any, model_config: ModelConfig):
        super().__init__()
        self.mixer_norm = RMSNorm(d_model, model_config.norm_eps)
        self.mixer = build_mixer(d_model, mixer_config)
        if model_config.mlp != "none":
            self.mlp_norm = RMSNorm(d_model, model_config.norm_eps)
            self.mlp = MLP(d_model, model_config.mlp_ratio, model_config.mlp)
        else:
            self.mlp_norm = None
            self.mlp = None

    def mix(self, x: torch.Tensor):
        """Sequence-mixing half: ``(x + mixer(norm(x)), var)``.

        ``var`` is the mixer's output variance when it emits one (KLA with
        ``return_variance``), else ``None``. The residual branch is
        deterministic, so it contributes no variance.

        Split out of :meth:`forward` so a caller can sample *between* the two
        halves: :meth:`SequenceModel.logprobs` needs the post-mixer mean and
        variance before the channel mixer runs. Stateless -- the recurrent-state
        path stays in :meth:`forward`.
        """
        mixed = self.mixer(self.mixer_norm(x))
        mixed, var = mixed if isinstance(mixed, tuple) else (mixed, None)
        return x + mixed, var

    def channel(self, x: torch.Tensor) -> torch.Tensor:
        """Channel-mixing half: ``x + mlp(norm(x))``, identity when ``mlp="none"``."""
        return x if self.mlp is None else x + self.mlp(self.mlp_norm(x))

    def forward(self, x: torch.Tensor, state=None, return_state: bool = False):
        emit_state = return_state or state is not None
        if emit_state:
            mixed, new_state = self.mixer(
                self.mixer_norm(x), state=state, return_state=True
            )
            if isinstance(mixed, tuple):  # KLA with return_variance: use the mean path
                mixed = mixed[0]
            x = x + mixed
        else:
            x, _ = self.mix(
                x
            )  # variance discarded: cross-entropy has nowhere to put it
            new_state = None
        x = self.channel(x)
        return (x, new_state) if emit_state else x


class SequenceModel(MarginalReadout, nn.Module):
    """Decoder-only model over token IDs with a unified mixer interface."""

    def __init__(
        self,
        config: ModelConfig,
        mixer: Any | Sequence[Any] | None = None,
    ):
        super().__init__()
        self.config = config
        if mixer is None:
            mixer = KLAConfig()
        mixers = (
            list(mixer)
            if isinstance(mixer, (list, tuple))
            else [mixer] * config.n_layers
        )
        if len(mixers) != config.n_layers:
            raise ValueError(
                f"Got {len(mixers)} mixer configs for {config.n_layers} layers"
            )

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(Block(config.d_model, m, config) for m in mixers)
        self.norm_f = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            if not getattr(module.weight, "_no_reinit", False):
                nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None and not getattr(
                module.bias, "_no_reinit", False
            ):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def init_state(self, batch: int, device=None, dtype=None) -> list:
        device = device or self.device
        return [
            blk.mixer.init_state(batch, device=device, dtype=dtype)
            for blk in self.blocks
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        state: Optional[list] = None,
        return_state: bool = False,
    ):
        """input_ids [B, L] → logits [B, L, vocab]. Pass/receive per-layer state
        for chunked prefill and autoregressive decode."""
        emit_state = return_state or state is not None
        x = self.embedding(input_ids)
        new_states = []
        for i, blk in enumerate(self.blocks):
            if emit_state:
                x, s = blk(
                    x, state=state[i] if state is not None else None, return_state=True
                )
                new_states.append(s)
            else:
                x = blk(x)
        logits = self.lm_head(self.norm_f(x))
        if self.config.logit_softcap is not None:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        return (logits, new_states) if emit_state else logits

    def _trunk(self, input_ids: torch.Tensor):
        """``ids -> (mu, var)`` just after the last block's sequence mixer.

        Sampling happens before the channel mixer, so the SwiGLU is part of the
        head and is re-evaluated per sample. Only the last block's variance is
        used: propagating one *through* a later scan is undefined, and the
        residual branch is deterministic so it adds none.
        """
        x = self.embedding(input_ids)
        for blk in self.blocks[:-1]:
            x = blk(x)
        last = self.blocks[-1]
        mu, var = last.mix(x)
        self._last_block = last
        return mu, var

    def _head(self, x: torch.Tensor) -> torch.Tensor:
        """Channel mixer + final norm + LM head + softcap, as log-probabilities."""
        x = self._last_block.channel(x)
        logits = self.lm_head(self.norm_f(x)).float()
        if self.config.logit_softcap is not None:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy/temperature sampling with recurrent state (O(1) per token for
        KLA/Mamba mixers, KV cache for attention)."""
        self.eval()
        state = self.init_state(input_ids.shape[0])
        logits, state = self.forward(input_ids, state=state)
        out = [input_ids]
        token = self._sample(logits[:, -1], temperature, top_k)
        for _ in range(max_new_tokens - 1):
            out.append(token)
            logits, state = self.forward(token, state=state)
            token = self._sample(logits[:, -1], temperature, top_k)
        out.append(token)
        return torch.cat(out, dim=1)

    @staticmethod
    def _sample(
        logits: torch.Tensor, temperature: float, top_k: Optional[int]
    ) -> torch.Tensor:
        if temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            kth = torch.topk(logits, top_k, dim=-1).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = torch.softmax(logits.float(), dim=-1)
        return torch.multinomial(probs, num_samples=1)
