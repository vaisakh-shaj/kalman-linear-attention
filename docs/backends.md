# Backends

The KLA layer is one algorithm with several implementations of its core scan.
By default `auto` is selected for the backend which tries to pick a performant option.

## Overview

Every backend carries all three implementation strategies — `recurrent`, `chunk`, `pscan`.
See
[implementations.md](implementations.md) for what the implementations mean and for the
full support matrix.

### Torch

A portable reference implementation that runs on CPU, CUDA, and Apple silicon but at lower performance.

### Triton

Requires: CUDA device

Almost as fast as the CUDA backend but easier to setup.
Does not require a CUDA compiler or extra toolchains.

### CUDA

Requires: CUDA device, CUDA compiler (`nvcc`)
and compatible C++ toolchain.

Compiles the CUDA kernels on first use.
CUDA 12.6 and 13.0 should be supported.
Make sure you have the same CUDA version as your PyTorch install, or at least a compatible one.
Make sure you have a C++ toolchain that is compatible with pytorch and supports C++17.
GCC 13.3 is known to work, older versions may cause `python -m kla --test-backends cuda` to fail.

### MPS

Requires: Apple silicon GPU with Metal toolchain installed.

The Metal shaders are compiled on first use by `torch.mps.compile_shader`.


## Installing

```bash
uv pip install kla
```

The CUDA backend compiles with whatever `nvcc` is on your PATH.
A specific CUDA version of torch can be chosen with UV like this or installed beforehand:

```bash
uv pip install "kla[cuda]" --torch-backend cu126
```

## Checking and testing

```bash
python -m kla                        # version, device, and what "auto" resolves to
python -m kla --check-backends       # which backends are usable here, and the "auto" pick
python -m kla --test-backends all    # run each one's forward and gradients
python -m kla --test-backends cuda_chunk   # ...or pin one exact implementation
```

To help debug compiler issues, set `KLA_JIT_VERBOSE=1` to see the CUDA backend's full
build command and log.
