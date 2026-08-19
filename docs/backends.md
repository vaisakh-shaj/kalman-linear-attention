# Backends

The KLA layer is one algorithm with several implementations of its core scan.
By default `auto` is selected for the backend which tries to pick a performant option.

## Overview

### Torch

This is a portable reference that uses the inbuilt `torch._higher_order_ops.associative_scan`.
The performance of this approach is currently suboptimal.

It runs everywhere — CPU, CUDA, Apple silicon — its gradients are exact, and it
is the only backend that runs in float64, which is what lets `gradcheck` test it.

### Triton

Requires: CUDA device and `triton` package

Almost as fast as the CUDA backend but more flexible and easier to setup.
On an NVIDIA GPU, `auto` should select this backend and it should not require additional setup.

`triton` takes a single fused kernel when nothing needs an adjoint and tiled
forward+backward scans otherwise. `triton_fused` and `triton_composed` pin one
each, so a config can name the kernels rather than depend on whether autograd
happened to be recording.

### CUDA

Requires: CUDA device and CUDA compiler

Fastest backend on CUDA devices but setting up CUDA dependencies can be tricky.

### MPS

Requires: Apple silicon GPU. The Metal shaders are compiled on first use by
`torch.mps.compile_shader`.

`mps` is `mps_fused`: one kernel each way, exact gradients, state carried in and
out. `mps_tiled` pins a forward-only kernel that parallelises over time instead
— ~2x faster for batch-1 prefill on a narrow model, ~2.5x slower once the
default has enough threads. `mps_composed` pins the triton-shaped scan kernels;
it is far slower, and exists for `d_state` past the fused ceiling.


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

Every backend is checked against `kla.ops.kla_scan_reference`,
a simple sequential implementation.

Every one of them is exact except the CUDA kernels, whose gradients through the
precision scan land ~5–15 % off. That kernel differentiates its trace-normalized
Möbius composition by carrying a 4-component adjoint through a chain of 4×4
Jacobians; the composed matrix degenerates toward rank-1, so that carry is
ill-conditioned in float32 even though the forward — a scale-invariant ratio of
the same matrix — is not. It is a property of that kernel rather than of the
algorithm: triton reaches ~3e-7 on the same hardware by carrying a *scalar*
adjoint over `λ` instead.
