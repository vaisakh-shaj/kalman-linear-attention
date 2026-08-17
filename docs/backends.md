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

### CUDA

Requires: CUDA device and CUDA compiler

Fastest backend on CUDA devices but setting up CUDA dependencies can be tricky.
Its gradients may be less accurate.

### MPS (Coming Soon)

Requires: Apple silicon GPU

## Installing

```bash
uv pip install "kla[triton]"
uv pip install "kla[cuda]"
```

The CUDA backend compiles with whatever `nvcc` is on your PATH.
A specific CUDA version can be chosen with UV like this:

```bash
uv pip install "kla[cuda]" --torch-backend cu126
```

## Checking and testing

```bash
python -m kla                       # version, device, and what "auto" resolves to
python -m kla --check-backends      # which backends are usable here, and the "auto" pick
python -m kla --test-backends all   # run each one's forward and gradients
```

`--check-backends` is a cheap capability probe — it looks for the device, the
package and `nvcc`, but compiles nothing, so `[x]` on a CUDA backend does not
mean "the kernel builds". Each row says what the backend is either way, and an
unusable one says what is missing or what failed. `[X]` marks the one `auto`
resolves to here.

`--test-backends` is the authoritative answer. Per backend it runs the forward
and the backward and checks both against the sequential reference:

```
torch
  forward    ok       max|dy| 4.5e-08   max|dvar| 6.0e-08   (atol 0.0005)
  gradients  ok       worst dp 1.6e-07   (budget 0.01)
```

Set `KLA_JIT_VERBOSE=1` to see the CUDA backend's full build command and log.

### Accuracy

Every backend is checked against `kla.ops.kla_scan_reference`,
a simple sequential implementation.
