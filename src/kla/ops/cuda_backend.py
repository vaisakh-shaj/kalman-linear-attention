"""CUDA backend for the KLA scan (torch cpp_extension JIT).

A fused CUDA kernel for the KLA core scan — a linear-space, *trace-normalized*
2×2 Möbius scan plus an affine information scan, both via CUB block scans
(register-resident, chunked with cross-chunk prefix carries). Sources are
JIT-compiled on first use via :func:`torch.utils.cpp_extension.load`, so the
published wheel ships no prebuilt binaries.

Kernel versions
---------------
Two kernels ship side by side, one directory each under ``kernels/cuda/``. The
algebra and the scan structure are identical; they differ only in their numerical
guards, and that difference is a genuine trade-off rather than a fix.

``v2_2`` — ``backend="cuda"``, the default
    The mathematically faithful one: it computes what torch and triton compute,
    with no ad-hoc bounds. (1) No ceiling on the per-token information gain
    ``phi = Λ^v·k²`` (the ``KLA_EPS`` floor stays). (2) The backward's
    ``phi_mask`` carries no upper condition, so ``d(Λ^v)`` and the phi-path of
    ``d(k)`` are never hard-zeroed. (3) ``a²`` is floored at ``KLA_EPS`` in both
    fwd and bwd, so ``1/a²`` cannot reach ``inf``.

    Because nothing caps ``phi``, bounding the inputs is the caller's job:
    ``obs_var_min`` / ``obs_var_max`` bound ``Λ^v`` and ``qk_norm`` /
    ``clip_value`` bound ``k``, and those apply on every backend identically.

``v2_1`` — ``backend="cuda_v2_1"``
    More heavily bounded against variance excursions, but not rigorous. It
    clamps ``phi`` to ``[1e-12, 1000]`` in ``compute_phi_r``, which hard-limits
    how much information a single token can inject and so keeps ``λ`` — and with
    it ``var = 1/λ`` — from swinging as far. That extra stability is real, and on
    badly-scaled inputs it can be the difference between a usable run and a
    blown-up one.

    It is not principled, though: ``r = (v·Λ^v)·k`` is left unclamped, and
    ``Λ^v`` only cancels out of ``mean = r/phi = v/k`` because *both* carry it.
    So a token saturating the cap by ``ρ = phi/1000`` emerges with its mean *and*
    its variance scaled by ``ρ``. Under the layer defaults ``phi`` reaches
    ``1/obs_var_min = 1e4``, so the cap sits *inside* the operating range rather
    than safely above it. It also divides by an unfloored ``a²``.

    Prefer ``v2_2`` and bound the inputs at the layer. Reach for ``v2_1`` when
    you need the harder clamp, or to reproduce a run made under it: a checkpoint
    *trained* on v2_1 has that clip baked into its learned ``σ²_v`` and will not
    reproduce its variances under v2_2 (or under torch/triton, which never had
    the clip). Pin the version to match the run.

Supported subset (anything else raises :class:`NotImplementedError`, so the
dispatcher's other backends stay usable):

* static ``a``/``q`` of shape ``[M, S]`` (the time-invariant discretized dynamics)
* ``d_state <= 64`` (``MAX_DSTATE``), float32 CUDA tensors
* zero/unit initial state (no carried prefill state) and
  ``decode_from_prior=False``

All layer-level features (projections, conv, qk-norm, discretization to
``a``/``q``, gating, λ-skip, variance read-out) are applied in PyTorch around
this scan, so the kernel is a drop-in for :func:`kla.ops.kla_scan_torch` on the
subset above. The kernel does not return the final filter state, so this path is
for training / full-sequence forward only (``backend="cuda"`` is never chosen by
``"auto"``).

Parity status (validated against the sequential reference on sm_86):

* **forward is bit-exact** (max abs error ~5e-7 in y and the variance);
* **backward is approximate on the precision-scan gradients** — d(sigma_inv),
  d(h), d(a), d(q) differ by ~5–15 % relative, while d(mu_sigma_inv) and d(w)
  (the information-vector / read-out path) are exact. This is a property of the
  kernel's trace-normalized Möbius backward, which is not the exact adjoint.
  Treat this backend as an exact *forward / inference* path and a training
  *speed* baseline; for exact gradients use ``backend="torch"`` or ``"triton"``.

Build toolchain
---------------
The kernel must be compiled with a CUDA toolkit matching the installed torch
(CUDA 13 for the ``cu130`` wheels; nix's nvcc 12.9 is the wrong major version).
That toolchain is *not* a project dependency — the ``nvidia-cuda-nvcc-cu13`` /
``nvidia-cuda-cccl-cu13`` pip packages have no py3.14 wheels and don't belong in
the runtime lockfile. Provision it out-of-band (a py≤3.13 sidecar venv, or an
existing toolkit) and point ``CUDA_HOME`` (or ``KLA_CUDA_HOME``) at a tree with
``bin/nvcc`` + the cub/cccl headers + ``lib64/libcudart.so`` (``ninja`` must
also be importable — it drives the cpp_extension build). Set
``KLA_JIT_VERBOSE=1`` to see the build command line. torch already ships the
cu13 cudart + cusparse/cublas redist headers under ``site-packages/nvidia/cu13``
— :func:`_extra_include_paths` adds them automatically, so a minimal nvcc+cccl
toolkit (e.g. nix's ``cuda-toolkit`` 13.x) is enough.
"""

from __future__ import annotations

import functools
import os
from typing import Optional

import torch

from kla.ops.kla_ops import P_MIN, KLAState

_KERNELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels", "cuda")

KERNEL_VERSIONS = ("v2_2", "v2_1")
DEFAULT_KERNEL_VERSION = "v2_2"


def _csrc_dir(version: str) -> str:
    """Source directory for one kernel version (see the module docstring)."""
    if version not in KERNEL_VERSIONS:
        raise ValueError(
            f"Unknown CUDA kernel version {version!r}; expected one of {list(KERNEL_VERSIONS)}"
        )
    return os.path.join(_KERNELS_DIR, version)

_NVCC_FLAGS = [
    "-O3",
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

MAX_DSTATE = 64


def _extra_include_paths(version: str) -> list[str]:
    """Include dirs the build needs beyond CUDA_HOME.

    torch's extension headers pull in cusparse/cublas/… ; the matching redist
    headers ship with the torch wheel under ``site-packages/nvidia/cu13/include``
    (the system CUDA toolkit — e.g. nix's — often omits them). CUDA 13 also moved
    the cub/cccl headers under ``$CUDA_HOME/include/cccl``; add it when present.
    """
    paths = [_csrc_dir(version)]
    site = os.path.dirname(os.path.dirname(torch.__file__))  # site-packages
    nvidia_inc = os.path.join(site, "nvidia", "cu13", "include")
    if os.path.isdir(nvidia_inc):
        paths.append(nvidia_inc)
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
    sources = [
        os.path.join(csrc, f)
        for f in sorted(os.listdir(csrc))
        if f.endswith((".cu", ".cpp"))
    ] if os.path.isdir(csrc) else []
    if not sources:
        raise NotImplementedError(
            f"No CUDA kernel sources found under {csrc} — "
            "use backend='torch' or 'triton'."
        )

    from torch.utils.cpp_extension import load

    return load(
        name=f"kla_matmul_scan_cuda_{version}",
        sources=sources,
        extra_cuda_cflags=_NVCC_FLAGS,
        extra_cflags=["-O3", "-std=c++17"],
        extra_include_paths=_extra_include_paths(version),
        verbose=os.environ.get("KLA_JIT_VERBOSE", "0") == "1",
    )


class _KLAMatmulScanFn(torch.autograd.Function):
    """Autograd wrapper over the fwd/bwd CUDA kernels.

    Operates on the kernel's native layout: model-axis tensors ``[B, M, L]`` and
    state-axis tensors ``[B, S, L]`` (the caller transposes from ``[B, L, ...]``).
    """

    @staticmethod
    def forward(ctx, mu_sigma_inv, sigma_inv, h, w, a, q, version):
        ext = _load_extension(version)
        ctx.version = version  # backward must compile against the same kernel
        msi = mu_sigma_inv.transpose(1, 2).contiguous()  # [B, M, L]
        si = sigma_inv.transpose(1, 2).contiguous()
        h_t = h.transpose(1, 2).contiguous()  # [B, S, L]
        w_t = w.transpose(1, 2).contiguous()
        a = a.contiguous()
        q = q.contiguous()

        y_t, yvar_t, mob_b, lin_b, lam_b = ext.fwd(msi, si, h_t, w_t, a, q)
        ctx.save_for_backward(msi, si, h_t, w_t, a, q, mob_b, lin_b, lam_b)
        return y_t.transpose(1, 2).contiguous(), yvar_t.transpose(1, 2).contiguous()

    @staticmethod
    def backward(ctx, dy, dyvar):
        ext = _load_extension(ctx.version)
        msi, si, h_t, w_t, a, q, mob_b, lin_b, lam_b = ctx.saved_tensors
        dy_t = dy.transpose(1, 2).contiguous()
        dyvar_t = dyvar.transpose(1, 2).contiguous()

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
    docstring. Defaults to ``v2_2``; pass ``"v2_1"`` for its harder ``phi`` clamp
    or to reproduce a run made under it.
    """
    if not v.is_cuda:
        raise _unsupported("requires CUDA tensors")
    if a.dim() != 2:
        raise _unsupported("expects a/p of shape [M, S]")
    if decode_from_prior:
        raise _unsupported("does not support decode_from_prior")
    if initial_state is not None:
        raise _unsupported("supports only the zero/unit initial state (no carried prefill state)")
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
