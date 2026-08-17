"""Unified Kalman Linear Attention layer.

One class, driven entirely by :class:`kla.configs.KLAConfig`, covers the whole
ablation space rather than one variant per class. The architecture:

    x [B,L,D] → in_proj → (z, gate) [B,L,d_inner]
    z → causal conv → sensor_proj → (v, log σ²_v, k, q)   (paper notation)
    a, p, Δ → static (time-invariant) discrete dynamics (a_bar, p_bar)
    → KLA scan (information-form Kalman filter)  → y [B,L,d_inner]
    y → λ-skip + gating → out_proj → [B,L,D]

Projections follow the KLA paper: ``v`` is the observed value, ``σ²_v`` its
per-token observation-noise variance (via softplus of ``log σ²_v``), ``k`` the
observation map (key), and ``q`` the readout (query). ``qk_norm`` l2-normalizes
``k`` and ``q``. The scan op (:func:`kla.ops.kla_scan`) takes the same paper
notation ``(v, Λ^v, k, q, a, p)`` and folds the value into its natural
parameter ``v·Λ^v`` internally.

The heavy lifting (the parallel filter scan) is delegated to
:func:`kla.ops.kla_scan`, which dispatches on ``config.backend``.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from kla.configs import KLAConfig
from kla.layers.common import (
    CausalConv1d,
    get_activation,
    inv_softplus_dt_init,
    l2_norm,
)
from kla.ops import KLAState, init_state, kla_scan, kla_step


class KLALayerState(NamedTuple):
    """Full recurrent state of one layer: conv window + filter state."""

    conv: Optional[torch.Tensor]  # [B, d_inner, K-1]
    ssm: KLAState  # (lam, eta), each [B, d_inner, d_state]


class KLALayer(nn.Module):
    def __init__(self, d_model: int, config: KLAConfig | None = None):
        super().__init__()
        config = config if config is not None else KLAConfig()
        self.config = config
        self.d_model = d_model
        self.d_inner = config.d_inner(d_model)
        self.d_state = config.d_state

        M, S = self.d_inner, self.d_state

        proj_out = self.d_inner * 2 if config.use_gating else self.d_inner
        self.in_proj = nn.Linear(d_model, proj_out, bias=config.bias)

        self.conv = (
            CausalConv1d(self.d_inner, config.conv_kernel_size, config.conv_activation)
            if config.use_conv
            else None
        )

        # Sensor projection → v (M), log σ²_v (M), k (S), q (S).
        #
        # The two d_inner-wide signals (v and log σ²_v) can each be emitted at
        # full width, through a low-rank bottleneck, or -- for v only -- not at
        # all (v = z, Mamba's move). Only the WIDTHS of this one Linear change;
        # the four outputs keep their shapes, so nothing downstream of here
        # (scan, backward, any backend) is affected.
        auto_rank = config.dt_rank or math.ceil(d_model / 8)
        self._value_rank = auto_rank if config.value_rank == "dt" else config.value_rank
        self._var_rank = auto_rank if config.var_rank == "dt" else config.var_rank

        # Width this projection emits for each of the two wide signals.
        self._w_value = (
            0
            if self._value_rank == "conv"
            else (M if self._value_rank == "full" else int(self._value_rank))
        )
        self._w_var = M if self._var_rank == "full" else int(self._var_rank)

        self.sensor_proj = nn.Linear(
            self.d_inner, self._w_value + self._w_var + 2 * S, bias=config.bias
        )
        # Up-projections, present only for the low-rank settings.
        self.value_expand = (
            nn.Linear(int(self._value_rank), M, bias=config.bias)
            if self._value_rank not in ("full", "conv")
            else None
        )
        self.var_expand = (
            nn.Linear(int(self._var_rank), M, bias=config.bias)
            if self._var_rank != "full"
            else None
        )

        # Discretization step Δ: static learned parameter (paper: a, p time-invariant).
        self.delta = nn.Parameter(torch.empty(M, S))
        self.delta._no_weight_decay = True

        # Continuous-time transition: a = -exp(lambda_log) (always negative)
        self.lambda_log = nn.Parameter(torch.empty(M, S))
        self.lambda_log._no_weight_decay = True

        if config.learnable_process_noise and not config.zero_process_noise:
            self.process_noise = nn.Parameter(torch.empty(M, S))
        else:
            self.register_buffer("process_noise", torch.empty(M, S))

        if config.use_lambda_skip:
            if config.lambda_skip_mode == "vector":
                self.lambda_skip = nn.Parameter(torch.empty(M))
            else:
                self.lambda_skip = nn.Parameter(torch.empty(()))
        else:
            self.lambda_skip = None

        self.gate_act = (
            get_activation(config.gating_activation) if config.use_gating else None
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=config.bias)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        """(Re-)initialize all parameters. Safe to call after meta-device
        construction + ``to_empty`` (e.g. inside nanochat's init_weights)."""
        cfg = self.config
        M, S = self.d_inner, self.d_state
        for mod in self.modules():
            if isinstance(mod, (nn.Linear, CausalConv1d)):
                mod.reset_parameters()
        self.delta.copy_(
            inv_softplus_dt_init((M, S), cfg.dt_min, cfg.dt_max, cfg.dt_init_floor)
        )
        self.lambda_log.zero_()
        self.process_noise.fill_(
            0.0 if cfg.zero_process_noise else cfg.process_noise_scale
        )
        if self.lambda_skip is not None:
            self.lambda_skip.fill_(cfg.lambda_skip_init)

    # ------------------------------------------------------------------ state

    def init_state(self, batch: int, device=None, dtype=None) -> KLALayerState:
        conv = (
            self.conv.init_state(batch, device=device, dtype=dtype)
            if self.conv
            else None
        )
        return KLALayerState(
            conv=conv, ssm=init_state(batch, self.d_inner, self.d_state, device=device)
        )

    # ------------------------------------------------------------- parameters

    def _continuous_params(self):
        """Continuous-time (raw) dynamics: transition a and process noise p."""
        a = -torch.exp(self.lambda_log.float())  # [M, S], strictly negative
        p = self.process_noise.float()
        if self.config.zero_process_noise:
            p = p + 1e-12
        else:
            p = p.clamp_min(1e-7)
            if self.config.clip_value is not None:
                p = p.clamp_max(self.config.clip_value)
        return a, p

    def _discretize(self, delta: torch.Tensor, a: torch.Tensor, p: torch.Tensor):
        """Δ [..., M, S] (broadcastable) → discrete a_bar and process noise p_bar."""
        a_bar = torch.exp(delta * a)
        if self.config.discretization == "ou":
            lam_pos = a.abs().clamp_min(1e-8)
            p_bar = (p / (2.0 * lam_pos)) * (-torch.expm1(-2.0 * lam_pos * delta))
        else:  # zoh
            p_bar = p * delta
        return a_bar, p_bar

    def _compute_aq(self):
        """Discrete (a_bar, p_bar), each [M, S] (static, time-invariant Δ)."""
        a, p = self._continuous_params()
        delta = F.softplus(self.delta.float()) + 1e-7  # [M, S]
        return self._discretize(delta, a, p)

    # ------------------------------------------------------------ projections

    def _project_sensors(self, z: torch.Tensor):
        """z [B, L, d_inner] → (v, lambda_v [B,L,M], k [B,L,S], q [B,L,S]).

        Paper notation: v = observed value, σ²_v = observation-noise variance,
        Λ^v = 1/σ²_v = value precision, k = observation map (key), q = readout
        (query). The value enters the filter only via v·Λ^v, but that fold is
        done inside the scan ops, so raw ``v`` is passed through here.
        """
        cfg = self.config
        S = self.d_state

        v, log_sigma_v, k, q = torch.split(
            self.sensor_proj(z), [self._w_value, self._w_var, S, S], dim=-1
        )
        # v: taken straight from z ("conv"), lifted from a low-rank code, or
        # already full width. log σ²_v: lifted or already full width.
        if self._value_rank == "conv":
            v = z  # [B, L, M] -- the width _w_value is 0
        elif self.value_expand is not None:
            v = self.value_expand(v)
        if self.var_expand is not None:
            log_sigma_v = self.var_expand(log_sigma_v)

        if cfg.qk_norm:
            k = l2_norm(k)
            q = l2_norm(q)
        if cfg.clip_value is not None:
            k = k.clamp(-cfg.clip_value, cfg.clip_value)
            q = q.clamp(-cfg.clip_value, cfg.clip_value)

        sigma_v = (
            F.softplus(log_sigma_v.float()) + cfg.obs_var_min
        )  # observation-noise variance
        if cfg.obs_var_max is not None:
            sigma_v = sigma_v.clamp_max(cfg.obs_var_max)

        lambda_v = 1.0 / sigma_v  # value precision Λ^v, [B, L, M]
        return v.float(), lambda_v, k.float(), q.float()

    # ----------------------------------------------------------------- compute

    def _run_ssm(self, z: torch.Tensor, ssm_state: KLAState | None):
        cfg = self.config
        # v = value, lambda_v = precision Λ^v, k = key, q = readout (paper notation).
        v, lambda_v, k, q = self._project_sensors(z)
        a_bar, p_bar = self._compute_aq()  # discretized decay + process noise, [M, S]

        def scan(v, lambda_v, k, q, lam0, eta0):
            # lam0 is None for a fresh sequence: let the backend apply its own
            # unit prior (the only initial state the CUDA backend supports).
            state = None if lam0 is None else KLAState(lam=lam0, eta=eta0)
            return kla_scan(
                v,
                lambda_v,
                k,
                q,
                a_bar,
                p_bar,
                initial_state=state,
                backend=cfg.backend,
                scan_impl=cfg.scan_impl,
                mobius_impl=cfg.mobius_impl,
                decode_from_prior=cfg.decode_from_prior,
            )

        lam0 = ssm_state.lam if ssm_state is not None else None
        eta0 = ssm_state.eta if ssm_state is not None else None
        with torch.autocast(device_type=z.device.type, enabled=False):
            if cfg.checkpoint_ssm and torch.is_grad_enabled() and self.training:
                y, y_var, new_state = torch.utils.checkpoint.checkpoint(
                    scan,
                    v,
                    lambda_v,
                    k,
                    q,
                    lam0,
                    eta0,
                    use_reentrant=False,
                )
            else:
                y, y_var, new_state = scan(v, lambda_v, k, q, lam0, eta0)
        return y, y_var, new_state

    def _run_ssm_step(self, z: torch.Tensor, ssm_state: KLAState):
        """Single-token decode path. z: [B, 1, d_inner]."""
        cfg = self.config
        v, lambda_v, k, q = self._project_sensors(z)
        a_bar, p_bar = self._compute_aq()  # discretized decay + process noise, [M, S]
        with torch.autocast(device_type=z.device.type, enabled=False):
            y, y_var, new_state = kla_step(
                v[:, 0],
                lambda_v[:, 0],
                k[:, 0],
                q[:, 0],
                a_bar,
                p_bar,
                ssm_state,
                decode_from_prior=cfg.decode_from_prior,
            )
        return y.unsqueeze(1), y_var.unsqueeze(1), new_state

    def forward(
        self,
        x: torch.Tensor,
        state: KLALayerState | None = None,
        return_state: bool = False,
    ):
        """x: [B, L, d_model].

        Returns ``out`` — or ``(out, new_state)`` when ``return_state`` or a
        ``state`` is passed — where ``out`` is ``y`` or ``(y, y_var)`` when
        ``config.return_variance``.
        """
        cfg = self.config
        B, L, _ = x.shape
        emit_state = return_state or state is not None

        projected = self.in_proj(x)
        if cfg.use_gating:
            z, gate = projected.chunk(2, dim=-1)
        else:
            z, gate = projected, None

        conv_state = state.conv if state is not None else None
        if self.conv is not None:
            z, conv_state = self.conv(z, conv_state)

        residual = z

        ssm_state = state.ssm if state is not None else None
        if state is not None and L == 1:
            y, y_var, ssm_state = self._run_ssm_step(z, ssm_state)
        else:
            y, y_var, ssm_state = self._run_ssm(z, ssm_state)

        y = y.to(x.dtype)
        y_var = y_var.to(x.dtype)

        if self.lambda_skip is not None:
            y = y + self.lambda_skip * residual

        if gate is not None:
            g = self.gate_act(gate)
            y = y * g
            y_var = y_var * g.pow(2)

        out = self.out_proj(y)
        if cfg.return_variance:
            out_var = F.linear(y_var, self.out_proj.weight.pow(2))
            result = (out, out_var)
        else:
            result = out

        if emit_state:
            return result, KLALayerState(conv=conv_state, ssm=ssm_state)
        return result
