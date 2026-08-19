# Backends

The KLA layer is one algorithm with several implementations of its core scan.
By default `auto` is selected for the backend which tries to pick a performant option.

## Overview

Every backend carries all three schedules — `recurrent`, `chunk`, `pscan` — and
a bare backend name means that backend's default, which is `chunk`. See
[implementations.md](implementations.md) for what the schedules mean and for the
full matrix. This page is about the backends themselves: what each needs, and
when to pick it.

### Torch

Requires: nothing.

The portable reference. It runs everywhere — CPU, CUDA, Apple silicon — its
gradients are exact, and it is the only backend that runs in float64, which is
what lets `gradcheck` test it. It is also the only one with no `d_state`
ceiling.

All three torch cells are `unfused`: they build the `[B, L, M, S]` intermediates
in memory, using `torch._higher_order_ops.associative_scan` and friends. That is
the point — it is the readable version every kernel is checked against — but the
performance is correspondingly suboptimal.

### Triton

Requires: CUDA device and the `triton` package.

Almost as fast as the CUDA backend, and far easier to set up: on an NVIDIA GPU
`auto` selects triton and nothing else is needed. torch's own Linux wheels
already ship triton.

This is the only backend with both families. `triton_<schedule>` is fused — one
kernel, no `[B, L, M, S]` intermediates, an exact backward that replays from
checkpoints. `triton_unfused_<schedule>` runs the same recurrences as standalone
scan kernels with torch glue between them; it is slower by the HBM passes that
glue costs, and it is the reference the fused cells are checked against.

### CUDA

Requires: CUDA device and a CUDA compiler (`nvcc`), which JIT-compiles the
kernels on first use.

The fastest backend on CUDA devices, at the cost of a toolchain that can be
fiddly to get right. The kernels target `sm_80` through `sm_90` and stay inside
the 48 KB of shared memory a consumer Ampere/Ada part gives without an opt-in
carveout, so one build covers a 4090 and an H200.

`cuda_v2_1` and `cuda_v2_2` are earlier kernels kept beside the three scheme
cells. They are the only implementations with an approximate backward, which is
exactly why they are still here — they are the comparison for it.

### MPS

Requires: Apple silicon GPU. The Metal shaders are compiled on first use by
`torch.mps.compile_shader`, so there is no toolchain and no extra to install.

All three cells are fused and share one backward. Metal has no float64, so
`gradcheck` still needs `backend="torch"`.

## Installing

```bash
uv pip install "kla[triton]"
uv pip install "kla[cuda]"
```

torch's own Linux wheels already ship triton, so that extra is mostly a version
floor. The MPS backend has no extra at all.

The CUDA backend compiles with whatever `nvcc` is on your PATH.
A specific CUDA version can be chosen with UV like this:

```bash
uv pip install "kla[cuda]" --torch-backend cu126
```

## Checking and testing

```bash
python -m kla                        # version, device, and what "auto" resolves to
python -m kla --check-backends       # which backends are usable here, and the "auto" pick
python -m kla --test-backends all    # run each one's forward and gradients
python -m kla --test-backends cuda_v2_1   # ...or pin one exact implementation
```

`--check-backends` is a cheap capability probe — it looks for the device, the
package and `nvcc`, but compiles nothing, so `[x]` does not mean "the kernel
builds". Each row says what the backend is either way, and an unusable one says
what is missing or what failed. `[X]` marks the one `auto` resolves to here.

`--test-backends` is the authoritative answer. It takes a family, one exact
backend, or `all`, and runs the forward and the backward of every implementation
named, each against the sequential reference:

```
torch
  forward    ok       max|dy| 4.5e-08   max|dvar| 6.0e-08   (atol 0.0005)
  gradients  ok       worst dp 1.6e-07   (budget 0.01)
```

A family with more than one implementation reports each of them separately,
under its own heading. Set `KLA_JIT_VERBOSE=1` to see the CUDA backend's full
build command and log.

### Accuracy

Every implementation is checked against `kla.ops.kla_scan_reference`, a simple
sequential implementation, in both directions.

All of them are exact except `cuda_v2_1` and `cuda_v2_2`, whose gradients
through the precision scan land ~5–15 % off. Those kernels differentiate their
trace-normalized Möbius composition by carrying a 4-component adjoint through a
chain of 4×4 Jacobians; the composed matrix degenerates toward rank-1, so that
carry is ill-conditioned in float32 even though the forward — a scale-invariant
ratio of the same matrix — is not.

That is a property of those two kernels, not of the algorithm or of the
schedule. Every other cell differentiates the *recurrence* instead, where
`∂λ_t/∂λ_{t-1} = a²/den²` is a scalar, recovering `λ` from checkpoints rather
than from the composed map. It is both exact and cheaper: those kernels already
pay the forward recompute, and the Jacobian chain is added on top of it.

This is why each backend has exactly one backward rather than one per schedule.
The adjoint reads the state at the checkpoints, or the values `λ` and `η` — not
the order a forward produced them in — so it is the same work whichever forward
ran.
