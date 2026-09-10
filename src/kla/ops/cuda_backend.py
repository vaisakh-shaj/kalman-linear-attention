"""JIT-compiled CUDA scan with chunk checkpoints and scalar reverse scans.

``cuda`` / ``cuda_v3_fast`` use v3 with fast math; ``cuda_v3`` uses the
same sources without fast math. Both implement the corrected backward.
Legacy ``cuda_v2_2`` and ``cuda_v2_1`` retain their original gradient errors
for reproducibility; v2_1 also caps observation information.

Supports static dynamics [M,S], d_state <= 64, and float32 computation.
Only zero/unit initial state is supported; the final state is not returned.
``auto`` continues to select Triton on CUDA devices.

Sources compile on first use. Install a toolkit matching torch and set
CUDA_HOME (or KLA_CUDA_HOME); KLA_JIT_VERBOSE=1 shows build diagnostics.
"""

from __future__ import annotations

import functools
import glob
import os
from typing import Optional

import torch

from kla.ops.kla_ops import P_MIN, KLAState

_KERNELS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "kernels", "cuda"
)

KERNEL_VERSIONS = ("v3_fast", "v3", "v2_2", "v2_1")
DEFAULT_KERNEL_VERSION = "v3_fast"


def _csrc_dir(version: str) -> str:
    """Source directory for one kernel version (see the module docstring)."""
    if version not in KERNEL_VERSIONS:
        raise ValueError(
            f"Unknown CUDA kernel version {version!r}; "
            f"expected one of {list(KERNEL_VERSIONS)}"
        )
    return os.path.join(_KERNELS_DIR, "v3" if version == "v3_fast" else version)


_NVCC_FLAGS = [
    "-O3",
    # torch >= 2.13's headers use C++20 default member initializers on
    # bit-fields (c10/core/AutogradState.h); nvcc's frontend rejects those
    # under -std=c++17 even where the host compiler would accept them.
    "-std=c++17",
    "--use_fast_math",
    # cu13 toolchains commonly mix nvcc and cccl/runtime header minor versions
    # (e.g. nvcc 13.2 against torch's bundled cu13.0 redist) — skip CCCL's
    # CTK-version compat assertion, which only guards header/compiler skew.
    "-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
]

def _nvcc_flags(version: str) -> list[str]:
    """Keep separate fast and standard-math builds of the same v3 sources."""
    _csrc_dir(version)
    flags = list(_NVCC_FLAGS)
    if version == "v3":
        flags.remove("--use_fast_math")
    return flags


MAX_DSTATE = 64


def _extra_include_paths(version: str) -> list[str]:
    """Include dirs the build needs beyond CUDA_HOME.

    torch's extension headers pull in cusparse/cublas/… ; the matching redist
    headers ship with the torch wheel (the system CUDA toolkit — e.g. nix's —
    often omits them) under ``site-packages/nvidia/*/include``. Whatever is there
    belongs to the torch that is installed, so take all of it and do not inspect
    versions. CUDA 13 also moved the cub/cccl headers under
    ``$CUDA_HOME/include/cccl``; add it when present.
    """
    paths = [_csrc_dir(version)]
    site = os.path.dirname(os.path.dirname(torch.__file__))  # site-packages
    paths += sorted(glob.glob(os.path.join(site, "nvidia", "*", "include")))
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("KLA_CUDA_HOME")
    if cuda_home:
        cccl = os.path.join(cuda_home, "include", "cccl")
        if os.path.isdir(cccl):
            paths.append(cccl)
    return paths


@functools.cache
def _load_extension(version: str = DEFAULT_KERNEL_VERSION):
    """JIT-compile and cache one kernel version's extension (first use only).

    Each version gets its own extension name, so torch caches their builds
    separately and the two can be loaded into the same process.
    """
    if os.environ.get("KLA_CUDA_HOME") and not os.environ.get("CUDA_HOME"):
        os.environ["CUDA_HOME"] = os.environ["KLA_CUDA_HOME"]

    csrc = _csrc_dir(version)
    sources = (
        [
            os.path.join(csrc, f)
            for f in sorted(os.listdir(csrc))
            if f.endswith((".cu", ".cpp"))
        ]
        if os.path.isdir(csrc)
        else []
    )
    if not sources:
        raise NotImplementedError(
            f"No CUDA kernel sources found under {csrc} — "
            "use backend='torch' or 'triton'."
        )

    from torch.utils.cpp_extension import load

    return load(
        name=f"kla_matmul_scan_cuda_{version}",
        sources=sources,
        extra_cuda_cflags=_nvcc_flags(version),
        extra_cflags=["-O3", "-std=c++17"],
        extra_include_paths=_extra_include_paths(version),
        verbose=os.environ.get("KLA_JIT_VERBOSE", "0") == "1",
    )


def _to_cuda_scan_layout(x: torch.Tensor) -> torch.Tensor:
    """Convert [B,L,C] to [B,C,L] with unit time stride, including L=1."""
    x = x.transpose(1, 2).contiguous()
    # contiguous() may preserve a nonunit stride on singleton dimensions.
    if x.stride(-1) != 1:
        x = torch.empty(x.shape, dtype=x.dtype, device=x.device).copy_(x)
    return x


class _KLAMatmulScanFn(torch.autograd.Function):
    """Autograd wrapper over the fwd/bwd CUDA kernels.

    Operates on the kernel's native layout: model-axis tensors ``[B, M, L]`` and
    state-axis tensors ``[B, S, L]`` (the caller transposes from ``[B, L, ...]``).
    """

    @staticmethod
    def forward(ctx, mu_sigma_inv, sigma_inv, h, w, a, q, version):
        ext = _load_extension(version)
        ctx.version = version  # backward must compile against the same kernel
        msi = _to_cuda_scan_layout(mu_sigma_inv)  # [B, M, L]
        si = _to_cuda_scan_layout(sigma_inv)
        h_t = _to_cuda_scan_layout(h)  # [B, S, L]
        w_t = _to_cuda_scan_layout(w)
        a = a.contiguous()
        q = q.contiguous()

        y_t, yvar_t, mob_b, lin_b, lam_b = ext.fwd(msi, si, h_t, w_t, a, q)
        ctx.save_for_backward(msi, si, h_t, w_t, a, q, mob_b, lin_b, lam_b)
        return y_t.transpose(1, 2).contiguous(), yvar_t.transpose(1, 2).contiguous()

    @staticmethod
    def backward(ctx, dy, dyvar):
        ext = _load_extension(ctx.version)
        msi, si, h_t, w_t, a, q, mob_b, lin_b, lam_b = ctx.saved_tensors
        dy_t = _to_cuda_scan_layout(dy)
        dyvar_t = _to_cuda_scan_layout(dyvar)

        dmsi, dsi, dh, dw, da, dq = ext.bwd(
            dy_t, dyvar_t, msi, si, h_t, w_t, a, q, mob_b, lin_b, lam_b
        )
        return (
            dmsi.transpose(1, 2).contiguous(),
            dsi.transpose(1, 2).contiguous(),
            dh.transpose(1, 2).contiguous(),
            dw.transpose(1, 2).contiguous(),
            da,
            dq,
            None,  # version (non-tensor)
        )


def _unsupported(msg: str) -> "NotImplementedError":
    return NotImplementedError(
        f"The CUDA KLA backend {msg}. Use backend='torch' or 'triton' for this case."
    )


def kla_scan_cuda(
    v: torch.Tensor,  # [B, L, M]
    lambda_v: torch.Tensor,  # [B, L, M]  value precision Λ^v
    k: torch.Tensor,  # [B, L, S]
    q: torch.Tensor,  # [B, L, S]  readout (query)
    a: torch.Tensor,  # [M, S] (static only)
    p: torch.Tensor,  # process noise
    initial_state: Optional[KLAState] = None,
    decode_from_prior: bool = False,
    kernel_version: str = DEFAULT_KERNEL_VERSION,
):
    """CUDA KLA scan. Same return contract as :func:`kla.ops.kla_scan_torch`,
    except the final state is ``None`` (the kernel is forward/training only).

    ``kernel_version`` selects between the shipped kernels — see the module
    docstring. Defaults to ``v3_fast``; legacy versions remain selectable.
    """
    if not v.is_cuda:
        raise _unsupported("requires CUDA tensors")
    if a.dim() != 2:
        raise _unsupported("expects a/p of shape [M, S]")
    if decode_from_prior:
        raise _unsupported("does not support decode_from_prior")
    if initial_state is not None:
        raise _unsupported(
            "supports only the zero/unit initial state (no carried prefill state)"
        )
    S = k.shape[2]
    if S > MAX_DSTATE:
        raise _unsupported(f"supports d_state <= {MAX_DSTATE} (got {S})")

    # Run in float32 (the kernel's native dtype). The kernel consumes the folded
    # information mean v·Λ^v, so pre-fold it here rather than inside the kernel.
    msi = (v.float() * lambda_v.float()).contiguous()  # v·Λ^v, [B, L, M]
    si = lambda_v.float().contiguous()  # Λ^v
    k3 = k.float().contiguous()  # [B, L, S]
    q = q.float().contiguous()
    a = a.float()
    # Floor the process noise exactly as _broadcast_ap does for the torch path.
    # The kernel has no internal guard, so a non-positive p makes (a² + p·λ) cross
    # zero and the Möbius recursion diverge to NaN -- where torch/triton would
    # stay finite on the same input.
    p = p.float().clamp_min(P_MIN)

    # kernel_version is the last forward arg (backward returns None for it), so
    # both passes compile against the same kernel.
    y, y_var = _KLAMatmulScanFn.apply(msi, si, k3, q, a, p, kernel_version)
    return y, y_var, None
