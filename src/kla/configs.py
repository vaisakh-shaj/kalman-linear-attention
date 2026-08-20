"""Configuration dataclasses for the KLA layer and sequence model.

These dataclasses are plain (no CLI-framework dependency) so the package stays
lightweight, but every field is annotated, so a generator such as tyro can
expose the full ablation space on the command line without any changes here.
"""

from __future__ import annotations

import dataclasses
from typing import Literal, Optional, Union

# Implementations are named "<backend>[_unfused|_merged]_<implementation>",
# where the implementation is how the kernel gets through the sequence:
# recurrent, chunk or pscan, and the middle token says how much is fused --
# "merged" being one scan for both recurrences rather than two. A bare backend
# name is that backend's default. "auto" is the only value whose meaning depends
# on the machine. See docs/implementations.md.
Backend = Literal[
    "auto",
    # bare backend names -- that backend's default implementation
    "torch",
    "triton",
    "cuda",
    "mps",
    # torch
    "torch_unfused_recurrent",
    "torch_unfused_chunk",
    "torch_unfused_pscan",
    # torch, merged -- one scan for both recurrences
    "torch_merged_chunk",
    "torch_merged_pscan",
    # triton
    "triton_recurrent",
    "triton_chunk",
    "triton_pscan",
    "triton_unfused_recurrent",
    "triton_unfused_chunk",
    "triton_unfused_pscan",
    # cuda
    "cuda_recurrent",
    "cuda_chunk",
    "cuda_pscan",
    # prior kernels, the only ones with an approximate backward
    "cuda_v2_2",
    "cuda_v2_1",
    # mps
    "mps_recurrent",
    "mps_chunk",
    "mps_pscan",
    "mps_merged_chunk",
    "mps_merged_pscan",
]
MobiusImpl = Literal["linear", "log"]

# How a d_inner-wide sensor signal is produced from the post-conv stream z.
#   "full"  one Linear(M, M): no bottleneck. The published architecture.
#   "dt"    low rank M -> r -> M, r = dt_rank or ceil(d_model/8). Named after
#           Mamba's dt_proj, which occupies the same slot (Delta is the only
#           d_inner-wide control signal Mamba has); the rank is twice Mamba's
#           own ceil(d_model/16), since sigma^2_v carries more than a timescale.
#   int     that rank explicitly. Only *saves* when 2*rank < d_inner.
#   "conv"  (value only) v = z, no projection at all. Mamba's move.
Rank = Union[int, Literal["full", "dt"]]
ValueRank = Union[int, Literal["full", "dt", "conv"]]


@dataclasses.dataclass
class KLAConfig:
    """Unified configuration for the Kalman Linear Attention (KLA) layer.

    A single :class:`~kla.layers.KLALayer` driven by this config covers the whole
    ablation space, rather than one layer class per variant. Algorithmic
    ablations and hardware/speed tricks are all toggled from here.
    """

    # --- dimensions -------------------------------------------------------
    d_state: int = 16
    """State dimension S per channel (the qk dim of the outer-product expansion)."""

    expand: float = 2.0
    """Channel expansion factor (Mamba-style). d_inner = expand * d_model."""

    # --- discretization ---------------------------------------------------
    # Following the KLA paper, the continuous transition ``a`` and process noise
    # ``p`` are learnable but time-invariant; the discretization step Δ is a
    # static learned parameter of shape [d_inner, d_state] (not input-dependent).
    dt_min: float = 0.001
    """Lower bound of the Δ initialization range."""

    dt_max: float = 0.1
    """Upper bound of the Δ initialization range."""

    dt_init_floor: float = 1e-4
    """Floor applied to the sampled Δ initialization."""

    discretization: Literal["ou", "zoh"] = "ou"
    """Continuous→discrete conversion. "ou" (Ornstein-Uhlenbeck) keeps the discrete
    decay positive and is critical for stacking layers; "zoh" uses q_d = Δ·q_c."""

    # --- process / observation noise --------------------------------------
    process_noise_scale: float = 0.01
    """Initial scale of the continuous process noise; 0.01 works best empirically."""

    learnable_process_noise: bool = True
    """If True the process noise is a trained parameter, otherwise a fixed buffer."""

    zero_process_noise: bool = False
    """Ablation: force (near-)zero process noise, recovering deterministic dynamics."""

    obs_var_min: float = 1e-4
    """Floor added to the predicted observation variance (softplus(log_var) + floor)."""

    obs_var_max: Optional[float] = None
    """Optional clamp on the predicted observation variance."""

    # --- sensor path shape: the "plain" and "mamba" blocks -----------------
    # Two knobs, both defaulting to the published architecture. The two named
    # blocks are just presets over them:
    #
    #   plain block   value_rank="full", var_rank="full"   <- these defaults
    #   mamba block   value_rank="conv", var_rank="dt"
    #
    # Neither touches the scan: both emit v [B,L,M], Lambda^v [B,L,M],
    # k/q [B,L,S], so kla_scan cannot tell them apart and no backend, kernel or
    # backward changes.
    value_rank: ValueRank = "full"
    """How the value ``v`` is produced from the post-conv stream ``z``.

    ``v`` is d_inner wide, so this is the single most expensive projection in
    the layer: "full" costs M^2 per block. "conv" (v = z) is Mamba's move and
    costs nothing at all.

    CAUTION on an integer / "dt" here: unlike the variance, ``v`` is the signal
    being *filtered*, and a low-rank map confines it to a fixed r-dimensional
    subspace of the channel space for every token. The read-out is not hard
    capped at rank r (the ``eta * 1/lambda`` gain is elementwise and nonlinear,
    as is softplus on the variance), but it is a real restriction on what the
    filter can observe. Untested; "full" and "conv" are the two published
    choices."""

    var_rank: Rank = "full"
    """How the observation noise ``log sigma^2_v`` is produced from ``z``.

    Low-ranking this is the safe one: it is a per-channel noise *level*, smooth
    and genuinely low-dimensional, and it is exactly what Mamba does to Delta.

    NOTE a rank only *saves* when 2*rank < d_inner, because the bottleneck costs
    two projections where the full map costs one. At d_inner=256 a rank of 256
    therefore costs *twice* the full projection rather than less, which is how
    the two static variants come out the same size at the MAD setting. Solve for
    the rank against a parameter budget rather than picking one by eye."""

    # --- projections / normalization --------------------------------------
    qk_norm: bool = True
    """L2-normalize the observation map k (key) and readout q (query) per token.
    Used in the paper architecture (Figure 7); helps when stacking many layers."""

    bias: bool = False
    """Use bias terms in the linear projections."""

    # --- short conv -------------------------------------------------------
    use_conv: bool = True
    """Apply a depthwise causal conv1d after the input projection (Mamba-style)."""

    conv_kernel_size: int = 4
    """Kernel size of the causal conv."""

    conv_activation: Literal["silu", "relu", "gelu"] = "silu"
    """Activation applied after the causal conv."""

    # --- gating / skip -----------------------------------------------------
    use_gating: bool = True
    """Project a parallel gate branch in in_proj and multiply it into the SSM output
    (element-wise multiplicative gating, as in the paper architecture / Figure 7)."""

    gating_activation: Literal["silu", "gelu", "relu"] = "silu"
    """Gate activation."""

    use_lambda_skip: bool = True
    """Add a learned λ-weighted skip connection from the post-conv activations."""

    lambda_skip_mode: Literal["scalar", "vector"] = "scalar"
    """Whether the λ skip weight is a scalar or per-channel vector."""

    lambda_skip_init: float = -1.0
    """Initial value of the λ skip weight."""

    # --- probabilistic outputs --------------------------------------------
    return_variance: bool = False
    """Also return the propagated output variance (needed for the reparametrisation
    trick / multi-sample losses)."""

    decode_from_prior: bool = False
    """Output the one-step-ahead prior prediction instead of the filtered posterior."""

    dt_rank: Optional[int] = None
    """Rank used by the ``"dt"`` setting of ``value_rank`` / ``var_rank``
    (named after Mamba's dt_rank). None = ceil(d_model / 8)."""

    # --- numerics ----------------------------------------------------------
    clip_value: Optional[float] = None
    """Optional symmetric clamp on h/w projections (and max clamp on process noise)."""

    # --- hardware / speed tricks -------------------------------------------
    backend: Backend = "auto"
    """Implementation of the core scan.

    "auto" reads the device and nothing else: triton on CUDA, Metal on Apple
    silicon, torch otherwise. It never selects a "cuda_v2_*" kernel, whose
    backward is an approximate adjoint.

    Every other value pins a code path. "torch", "triton", "cuda" and "mps" are
    their backend's default; a full
    "<backend>[_unfused|_merged]_<implementation>" name pins an exact one --
    see docs/implementations.md and :func:`kla.ops.kla_scan`."""

    mobius_impl: MobiusImpl = "linear"
    """How the precision Möbius map is represented while it is composed. Torch
    backend only -- the triton and CUDA kernels are hard-wired to "linear".
    Orthogonal to the implementation, which picks how the scan is parallelized.

    "linear"  (default) compose the 2x2 maps as plain matmuls normalized by the
              trace. No transcendentals in the combine, and the same scheme both
              GPU backends use -- so torch is a faithful reference for them.
    "log"     keep the entries as logs and combine with logaddexp. Slower, but
              strictly more exponent headroom: after trace-normalization
              D ~ a²/(1+pφ), which underflows float32 once ā drops below ~1e-19.

    Trace normalization is stable on its own -- the defaults init |a| = 1, and
    reaching that underflow would take |a| ≳ 440 at Δ=0.1 -- so "linear" is the
    one to use. "log" is kept as a reference implementation of the same map, to
    show how else the Möbius scan can be composed.
    See :func:`kla.ops.kla_scan_torch`."""

    checkpoint_ssm: bool = False
    """Recompute the SSM scan in backward (activation checkpointing) to save memory."""

    def d_inner(self, d_model: int) -> int:
        return int(self.expand * d_model)


@dataclasses.dataclass
class ModelConfig:
    """High-level sequence model: embedding + N blocks (mixer [+ MLP]) + LM head."""

    vocab_size: int = 50304
    d_model: int = 512
    n_layers: int = 6

    mlp: Literal["none", "swiglu", "gelu"] = "swiglu"
    """Channel mixer interleaved after each sequence mixer. "none" = mixer-only stack
    (the original MAD setups alternate mixer/swiglu)."""

    mlp_ratio: float = 4.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = False
    logit_softcap: Optional[float] = None
    """Optional tanh soft-capping of the output logits (as used in nanochat)."""
