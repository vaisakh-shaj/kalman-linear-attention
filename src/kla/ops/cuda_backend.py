"""CUDA backend for the KLA scan (torch cpp_extension JIT).

Three cells, ``cuda_fused_recurrent``, ``cuda_fused_chunk`` and ``cuda_merged_chunk``,
and the one exact backward all three share, under ``kernels/cuda/scan/``. Each
puts the sufficient statistics, both recurrences and the read-out in a single
kernel, so no ``[B, L, M, S]`` intermediate is ever written; only the forward's
walk through time differs. Sources are JIT-compiled on first use via
:func:`torch.utils.cpp_extension.load`, so the published wheel ships no
prebuilt binaries.

``cuda_fused_recurrent``
    One thread per ``(batch, channel, state)`` lane, time serial, the Möbius map
    *applied* to a running λ. Its whole grid is the lane count, so it wants a lot
    of lanes, but on a 142-SM L40S its serial chain of divisions is
    latency-bound rather than throughput-bound, and its time is flat from 256
    lanes to 65536. That makes it the fastest cell only past ~64k lanes with a
    short sequence, and in training at a realistic shape.
``cuda_fused_chunk`` (``backend="cuda"``)
    Time tiled: ``KLA_ITEMS`` timesteps per thread, composed within a CUB block
    scan and carried across chunks. Fastest or within 3% at every shape measured
    on an L40S, which is why the bare alias points here.
``cuda_merged_chunk``
    ``cuda_fused_chunk`` with both recurrences folded into one 3x3 map in homogeneous
    coordinates (``kla_merged.cuh``), so the block scans once instead of twice.
    **It is the merged cell that costs rather than saves**, 3-29% slower than
    ``cuda_fused_chunk`` at every shape, because CUDA's second scan is only
    ``log2(ROWS)`` rounds of ``float2`` shuffles and merging widens the shared
    memory aggregate to pay for them. The identical transcription *wins* on
    triton and mps, where the second scan was expensive. Kept as the measured
    counterexample; see ``docs/benchmarks/cuda.md``.

Every cell is exact in the backward, carries the filter state in and out
differentiably, and supports ``decode_from_prior``. The backward differentiates
the *recurrence* rather than the composed Möbius map, which makes the per-step
gain a scalar (``dλ_t/dλ_{t-1} = a²/den_t²``), see ``kla_scan_bwd.cuh``.

Supported subset (anything else raises :class:`NotImplementedError`, so the
dispatcher's other backends stay usable):

* static ``a``/``p`` of shape ``[M, S]`` (the time-invariant discretized dynamics)
* ``d_state <= 64`` (``MAX_DSTATE``), float32 CUDA tensors

All layer-level features (projections, conv, qk-norm, discretization to
``a``/``p``, gating, λ-skip, variance read-out) are applied in PyTorch around
this scan, so these are drop-ins for :func:`kla.ops.kla_scan_torch`.

``backend="auto"`` prefers triton on CUDA rather than this backend, even though
these kernels are 1.2-2x faster on the same algebra: they need nvcc and a
matching C++ toolchain at first use, which ``auto`` cannot assume. Pin
``backend="cuda"`` to get them.

Build toolchain
---------------
The kernels must be compiled with a CUDA toolkit matching the installed torch
(CUDA 13 for the ``cu13x`` wheels; nvcc 12.9 is the wrong major version). That
toolchain is *not* a project dependency, the ``nvidia-cuda-nvcc-cu13`` /
``nvidia-cuda-cccl-cu13`` pip packages have no py3.14 wheels and don't belong in
the runtime lockfile. Provision it out-of-band (a py<=3.13 sidecar venv, or an
existing toolkit) and point ``CUDA_HOME`` (or ``KLA_CUDA_HOME``) at a tree with
``bin/nvcc`` + the cub/cccl headers + ``lib64/libcudart.so`` (``ninja`` must
also be importable, it drives the cpp_extension build). Set
``KLA_JIT_VERBOSE=1`` to see the build command line. torch already ships the
cu13 cudart + cusparse/cublas redist headers under ``site-packages/nvidia/*``,
and :func:`_load_scan_extension` adds every one of them to the include path
automatically, so a minimal nvcc+cccl toolkit is enough.

A killed build leaves a lock behind that hangs every later process with no
message: ``find ~/.cache/torch_extensions -name lock -delete``.
"""

from __future__ import annotations

import functools
import glob
import os
from typing import Optional

import torch

from kla.ops.kla_ops import P_MIN, KLAState, init_state

_KERNELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kernels", "cuda"
)

_NVCC_FLAGS = [
    "-O3",
    "-std=c++17",
    "--use_fast_math",
    # cu13 toolchains commonly mix nvcc and cccl/runtime header minor versions
    # (e.g. nvcc 13.2 against torch's bundled cu13.0 redist), skip CCCL's
    # CTK-version compat assertion, which only guards header/compiler skew.
    "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

MAX_DSTATE = 64
"""Largest ``d_state`` these kernels take: one block must hold every state of a
channel, because the read-out sums over the state axis."""


def _unsupported(msg: str) -> "NotImplementedError":
    return NotImplementedError(
        f"The CUDA KLA backend {msg}. Use backend='torch' or 'triton' for this case."
    )


_SCAN_DIR = os.path.join(_KERNELS_DIR, "scan")


def _tuning_flags() -> tuple[list[str], str]:
    """``-D`` overrides for the two build-time constants, and a name suffix.

    ``KLA_CHUNK`` (checkpoint stride) and ``KLA_ITEMS`` (timesteps one thread of
    the chunk forward walks) are ``#define``s, so tuning them means rebuilding.
    Both are guarded with ``#ifndef`` in ``kla_scan_common.cuh``; setting
    ``KLA_CUDA_CHUNK`` / ``KLA_CUDA_ITEMS`` in the environment passes them
    through as ``-D`` and gives the build its own extension name, so several
    settings can be measured without clobbering each other's cache. Unset means
    the defaults compiled into the header, and the plain extension name.
    """
    flags, parts = [], []
    pairs = (("KLA_CUDA_CHUNK", "KLA_CHUNK"), ("KLA_CUDA_ITEMS", "KLA_ITEMS"))
    for env, macro in pairs:
        raw = os.environ.get(env)
        if raw is None:
            continue
        val = int(raw)  # a bad value should fail here, not inside nvcc
        flags.append(f"-D{macro}={val}")
        parts.append(f"{macro.split('_')[1].lower()}{val}")
    return flags, ("_" + "_".join(parts) if parts else "")


@functools.cache
def _load_scan_extension():
    """JIT-compile and cache the exact-scan extension (first use only)."""
    if os.environ.get("KLA_CUDA_HOME") and not os.environ.get("CUDA_HOME"):
        os.environ["CUDA_HOME"] = os.environ["KLA_CUDA_HOME"]

    sources = [
        os.path.join(_SCAN_DIR, f)
        for f in sorted(os.listdir(_SCAN_DIR))
        if f.endswith((".cu", ".cpp"))
    ]
    if not sources:
        raise NotImplementedError(
            f"No CUDA scan sources found under {_SCAN_DIR}, "
            "use backend='torch' or 'triton'."
        )

    from torch.utils.cpp_extension import load

    site = os.path.dirname(os.path.dirname(torch.__file__))
    includes = [_SCAN_DIR] + sorted(
        glob.glob(os.path.join(site, "nvidia", "*", "include"))
    )
    tune_flags, suffix = _tuning_flags()
    return load(
        name=f"kla_scan_cuda{suffix}",
        sources=sources,
        extra_cuda_cflags=_NVCC_FLAGS + tune_flags,
        extra_cflags=["-O3", "-std=c++17"],
        extra_include_paths=includes,
        verbose=os.environ.get("KLA_JIT_VERBOSE", "0") == "1",
    )


class _CudaKLAScan(torch.autograd.Function):
    """One forward implementation, and the backward every implementation shares."""

    @staticmethod
    def forward(ctx, msi, si, k, q, a, p, lam0, eta0, prior, implementation):
        ext = _load_scan_extension()
        fwd = getattr(ext, f"{implementation}_fwd")
        y, yvar, lam_fin, eta_fin, lam_ck, eta_ck = fwd(
            msi, si, k, q, a, p, lam0, eta0, any(ctx.needs_input_grad), prior
        )
        ctx.prior = prior
        ctx.save_for_backward(msi, si, k, q, a, p, lam0, eta0, lam_ck, eta_ck)
        return y, yvar, lam_fin, eta_fin

    @staticmethod
    def backward(ctx, dy, dyvar, dlam_fin, deta_fin):
        ext = _load_scan_extension()
        msi, si, k, q, a, p, lam0, eta0, lam_ck, eta_ck = ctx.saved_tensors
        grads = ext.bwd(
            dy.contiguous(),
            dyvar.contiguous(),
            dlam_fin.contiguous(),
            deta_fin.contiguous(),
            msi,
            si,
            k,
            q,
            a,
            p,
            lam0,
            eta0,
            lam_ck,
            eta_ck,
            ctx.prior,
        )
        return (*grads, None, None)


def _cuda_scan(implementation: str):
    """Build one exact CUDA cell. Only the forward implementation differs."""

    def run(
        v: torch.Tensor,
        lambda_v: torch.Tensor,
        k: torch.Tensor,
        q: torch.Tensor,
        a: torch.Tensor,
        p: torch.Tensor,
        initial_state: Optional[KLAState] = None,
        decode_from_prior: bool = False,
    ):
        if not v.is_cuda:
            raise _unsupported("requires CUDA tensors")
        if a.dim() != 2:
            raise _unsupported("expects a/p of shape [M, S]")
        S = k.shape[2]
        if S > MAX_DSTATE:
            raise _unsupported(f"supports d_state <= {MAX_DSTATE} (got {S})")

        B, _, M = v.shape
        state = initial_state
        if state is None:
            state = init_state(B, M, S, device=v.device)

        # The kernel consumes the folded information mean v·Λ^v, so fold it in
        # torch and let autograd split d(v·Λ^v) back into dv and d(Λ^v).
        # Flooring p here rather than in the kernel does the same for the
        # floor's subgradient.
        y, y_var, lam_fin, eta_fin = _CudaKLAScan.apply(
            (v.float() * lambda_v.float()).contiguous(),
            lambda_v.float().contiguous(),
            k.float().contiguous(),
            q.float().contiguous(),
            a.float().contiguous(),
            p.float().clamp_min(P_MIN).contiguous(),
            state.lam.float().contiguous(),
            state.eta.float().contiguous(),
            decode_from_prior,
            implementation,
        )
        return y, y_var, KLAState(lam=lam_fin, eta=eta_fin)

    run.__name__ = f"kla_scan_cuda_{implementation}"
    run.__doc__ = (
        f"``cuda_{implementation}``. Same contract as :func:`kla.ops.kla_scan_torch`."
    )
    return run


kla_scan_cuda_recurrent = _cuda_scan("recurrent")
kla_scan_cuda_chunk = _cuda_scan("chunk")
# merged -- fused *and* one scan instead of two, via the 3x3 map in
# kernels/cuda/scan/kla_merged.cuh. `recurrent` has no merged cell and never
# will: it *applies* the map rather than composing it, so it already does λ and
# η in one pass.
kla_scan_cuda_merged_chunk = _cuda_scan("merged_chunk")
