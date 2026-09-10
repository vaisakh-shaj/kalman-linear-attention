"""Shared numerical acceptance criteria for backend parity checks."""

import torch

ATOL = 2e-5
RTOL = 2e-4
INPUT_NAMES = ("v", "lambda_v", "k", "q", "a", "p")
LEGACY = ("cuda_v2_1", "cuda_v2_2")


class AccuracyMismatch(AssertionError):
    """Finite results outside the shared tolerance."""


def metrics(got, ref):
    """Use max error <= atol + rtol * max reference; relL2 is diagnostic."""
    got, ref = got.detach().double(), ref.detach().double()
    finite = bool(torch.isfinite(got).all() and torch.isfinite(ref).all())
    error = (got - ref).abs().max().item()
    limit = ATOL + RTOL * ref.abs().max().item()
    rel_l2 = ((got - ref).norm() / ref.norm().clamp_min(1e-30)).item()
    return dict(finite=finite, passed=finite and error <= limit,
                max_abs=error, limit=limit, rel_l2=rel_l2)


def detail(name, m):
    return (f"{name}: max_abs={m['max_abs']:.3e} "
            f"limit={m['limit']:.3e} relL2={m['rel_l2']:.3e}")
