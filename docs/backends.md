# Backends

Four backends, each carrying the same implementations. See
[implementations.md](implementations.md) for the matrix and the defaults.

| backend | requires | notes |
|---|---|---|
| `torch` | nothing | runs on CPU, CUDA and Apple silicon. The only float64 path |
| `triton` | CUDA device | no compiler needed, so `auto` picks it on NVIDIA |
| `cuda` | CUDA device, `nvcc`, C++17 toolchain | fastest, compiled on first use |
| `mps` | Apple silicon | shaders compiled on first use, no toolchain |

## Installing

```bash
uv pip install kla
uv pip install "kla[cuda]" --torch-backend cu126   # to pin a CUDA version
```

The `cuda` backend builds with whatever `nvcc` is on your PATH, which must match
your torch CUDA version. GCC 13.3 is known to work.

## Checking

```bash
python -m kla                       # device, and what "auto" resolves to
python -m kla --check-backends      # which backends are usable here
python -m kla --test-backends all   # run each one's forward and gradients
```

Set `KLA_JIT_VERBOSE=1` to see the `cuda` build log. If a build is interrupted it
leaves a lock that hangs later runs: `find ~/.cache/torch_extensions -name lock -delete`.
