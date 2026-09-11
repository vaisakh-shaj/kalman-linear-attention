# Backends

The KLA layer is one algorithm with several implementations of its core scan.
By default `auto` is selected for the backend which tries to pick a performant option.

## Overview

### Torch

The portable reference uses `torch._higher_order_ops.associative_scan`.
Standard PyTorch code makes it easy to understand and modify, but training is
slow and memory-intensive: it materializes expanded scan intermediates with a
state dimension for each token and channel.

It runs everywhere - CPU, CUDA, Apple silicon - its gradients are exact, and it
is the only backend that runs in float64, which is what lets `gradcheck` test it.

### Triton

Requires: CUDA device and `triton` package

Easier to set up and more flexible than the CUDA backend. During training, the
current Triton implementation materializes intermediate tensors of shape
`[batch, length, channels, d_state]` in GPU memory. CUDA instead saves
chunk-boundary checkpoints and recomputes intermediate states during backward,
substantially reducing training memory use.
On an NVIDIA GPU, `auto` should select this backend and it should not require additional setup.

`triton` takes a single fused kernel when nothing needs an adjoint and tiled
forward+backward scans otherwise. `triton_fused` and `triton_composed` pin one
each, so a config can name the kernels rather than depend on whether autograd
happened to be recording.

### CUDA

Requires: CUDA device and CUDA compiler

The fastest implementation in the paper's NVIDIA GPU training benchmarks,
substantially faster than Triton. See [Figure 4 and §5.2](https://arxiv.org/html/2602.10743v2).
It requires a compatible CUDA toolkit and C++ compiler; follow [CUDA setup](#cuda-setup).

The fused scan keeps the expanded filter state in registers and shared memory.
For backward, it saves chunk-boundary checkpoints and recomputes within each
chunk, avoiding a full `[batch, length, channels, d_state]` state history in GPU
memory. This gives a much smaller scan-memory footprint than the current Torch
and Triton training paths, following the same fusion and recomputation principle
as Mamba. Inputs, outputs, and checkpoints still occupy GPU memory.

This comparison concerns training: Triton's fused inference path also avoids
materializing the full state history.

### MPS

Requires: Apple silicon GPU. The Metal shaders are compiled on first use by
`torch.mps.compile_shader`.

`mps` is `mps_fused`: one kernel each way, exact gradients, state carried in and
out. `mps_tiled` pins a forward-only kernel that parallelises over time instead
- ~2x faster for batch-1 prefill on a narrow model, ~2.5x slower once the
default has enough threads. `mps_composed` pins the triton-shaped scan kernels;
it is far slower, and exists for `d_state` past the fused ceiling.


## Installing

First create and activate a virtual environment using the
[README installation steps](../readme.md#install), then install KLA:

```bash
uv pip install kla
```

In a new shell, activate the same environment with `source .venv/bin/activate`
before running the `python` commands below.

Choose a CUDA-enabled PyTorch build supported by your NVIDIA driver and GPU.
For example, to select the CUDA 12.6 build with UV:

```bash
uv pip install kla --torch-backend cu126
```

### CUDA setup

**PyTorch running on your GPU does not mean the CUDA compiler is installed.**
Its CUDA runtime runs existing kernels; KLA's CUDA backend also needs `nvcc`
to compile its kernels. A missing `CUDA_HOME` error usually means the toolkit
is not installed or its environment has not been loaded.

KLA discovers the CUDA toolkit automatically and compiles on first use of
`backend="cuda"`. Use a toolkit matching PyTorch's CUDA version where possible,
and a C++ compiler supported by both. Installing KLA does not install or select
compatible system tools. On a cluster, load your site's CUDA and compiler modules
first; set `CUDA_HOME` below if automatic discovery selects the wrong toolkit.

Inspect your environment:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda)"
nvcc --version
nvidia-smi
```

`torch.version.cuda` identifies PyTorch's CUDA build; `nvcc --version` identifies
the installed compiler toolkit. The CUDA version shown by `nvidia-smi` indicates
driver support, not the installed toolkit.

On a cluster, discover your site's modules, then load a toolkit matching
`torch.version.cuda` and a compatible compiler **before starting Python**:

```bash
module avail cuda
module avail gcc
# Example from our cluster for a CUDA 12.6 PyTorch build; names vary by site:
module load gcc-native/13.2 cuda/12.6
```

On a workstation, install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
and a compatible C++ compiler. Loading a cluster module or installing the
toolkit may configure discovery automatically; otherwise set `CUDA_HOME` below.
Exit and restart an existing Python session or notebook kernel after changing
the environment, because PyTorch caches toolkit discovery.

Tested configuration: NVIDIA GH200, CUDA toolkit 12.6, a CUDA 12.6 PyTorch build,
and GCC 13.3. This is a tested example, not a requirement for every GPU.

After installing KLA, check what is available:

```bash
python -m kla --check-backends
```

If the toolkit or compiler is not selected correctly, set these paths to your
installation (omit settings that are already correct):

```bash
export CUDA_HOME=/path/to/cuda
export CC=/path/to/gcc
export CXX=/path/to/g++
```

Verify compilation and forward/gradient accuracy:

```bash
python -m kla --test-backends cuda_v3_fast
```

The first run takes longer to compile; subsequent runs reuse the build cache.
If the error says **GCC is too old**, install/load a compatible compiler and
select it with `CC`/`CXX`. If the **CUDA toolkit is not found**, install/load it
and point `CUDA_HOME` at the directory containing `bin/nvcc`.

## Checking and testing

```bash
python -m kla                        # version, device, and what "auto" resolves to
python -m kla --check-backends       # which backends are usable here, and the "auto" pick
python -m kla --test-backends all    # run each one's forward and gradients
python -m kla --test-backends cuda_v2_1   # ...or pin one exact implementation
```

`--check-backends` is a cheap capability probe - it looks for the device, the
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

The default CUDA backend uses v3 with corrected scalar backward gradients and
chunk-boundary propagation. `cuda_v3_fast` pins this build; `cuda_v3` uses the
same source without fast math. Float32 results agree within numerical tolerances,
not bit-for-bit. Legacy `cuda_v2_2` and `cuda_v2_1` retain gradient errors and
remain available for reproducing earlier runs.

Accuracy checks share `max_abs_error <= 2e-5 + 2e-4 * max_abs_reference`
for every backend and input gradient. `PASS` means this strict check passed;
yellow `KNOWN FAIL` identifies a finite legacy precision-gradient mismatch.
NaNs and build errors remain failures. Explicit CLI checks exit nonzero for
known failures too; `all` remains a survey. Pytest tracks the fixed legacy
gradient cases as strict `xfail`: an unexpected pass requires review.
