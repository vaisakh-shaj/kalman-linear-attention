"""Deterministic and Monte-Carlo read-outs for models carrying a posterior.

A KLA layer emits a mean *and* a variance. Cross-entropy has nowhere to put the
variance, so the standard forward path drops it. This module is what uses it:
instead of evaluating the head at the posterior mean, marginalise over the
posterior.

    deterministic   log softmax(head(mu))
    Monte Carlo     log (1/S) sum_s softmax(head(mu + eps_s * std))

softmax is nonlinear, so the two genuinely differ: averaging flattens the
distribution for tokens the filter is unsure about.

A model opts in by inheriting :class:`MarginalReadout` and splitting itself at
the sampling point::

    _trunk(input_ids) -> (mu, var)   everything up to the split
    _head(x)          -> log-probs   everything after it

*Where* a model splits is a modelling decision, not a detail -- everything
downstream of the split is re-evaluated per sample, and its nonlinearity is
what makes the marginal analytically intractable in the first place. Models
whose head needs more than one argument call :func:`marginal_logprobs` directly
with a closure.
"""

from __future__ import annotations

import math

import torch


def marginal_logprobs(
    head, mu, var, mc_samples: int, generator=None, owner: str = "model"
):
    """``log (1/S) sum_s softmax(head(mu + eps_s * std))``, all S samples at once.

    ``head`` must return log-probabilities and broadcast over a leading sample
    axis. The estimator is the identity

        log (1/S) sum_s p_s  ==  logsumexp_s(log p_s) - log S

    computed in log space so the sum never leaves it. The result is already a
    normalised distribution -- a mixture of softmaxes is one -- so applying
    softmax to it again would be a bug.

    Peak memory is ``S`` times the head's output, so an ``S x B x L x vocab``
    tensor at its widest. That is a few MB at MAD sizes and a few hundred at
    long-context ones; if it ever binds, evaluate in sequence chunks rather than
    reintroducing a sample loop.
    """
    if var is None:
        raise ValueError(
            f"{owner}: mc_samples>0 needs a mixer that emits a variance -- set "
            "return_variance=True on a KLAConfig. Baseline mixers (Mamba, GLA, "
            "GDN, attention) have no posterior to sample."
        )
    std = var.clamp_min(1e-12).sqrt()
    eps = torch.randn(
        (mc_samples, *mu.shape), device=mu.device, dtype=mu.dtype, generator=generator
    )
    log_p = head(mu + eps * std)  # [S, ..., vocab]
    return torch.logsumexp(log_p, dim=0) - math.log(mc_samples)


class MarginalReadout:
    """Mixin providing :meth:`logprobs` in terms of ``_trunk`` and ``_head``."""

    def logprobs(self, input_ids: torch.Tensor, mc_samples: int = 0, generator=None):
        """``(deterministic, mc)`` log-probabilities; ``mc`` is None when mc_samples=0.

        Both read-outs come from ONE trunk evaluation, so they are scored on
        identical weights and an identical scan. Running them as separate jobs
        would let kernel nondeterminism (measured at up to 0.047 accuracy on
        selective copying) swamp the decoding effect being measured.

        Deliberately not wrapped in ``no_grad``: evaluation already runs inside
        one, while a multi-sample *training* loss needs the reparametrised
        samples to carry gradients into the variance. ``y_var`` is
        differentiable on every backend.
        """
        mu, var = self._trunk(input_ids)
        det = self._head(mu)
        if not mc_samples:
            return det, None
        return det, marginal_logprobs(
            self._head, mu, var, mc_samples, generator, owner=type(self).__name__
        )
