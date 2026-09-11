# KLA - Kalman Linear Attention

**[Project page](https://kalman-linear-attention.github.io/)** · **[Paper (ICML 2026)](https://arxiv.org/abs/2602.10743)** · **[Poster](https://kalman-linear-attention.github.io/static/kla_poster_icml2026.pdf)**

A linear attention layer that is an **exact parallel Kalman filter**.
Unlike other linear attention layers which model the current state as a single point in the state space,
KLA models the current state as a **belief** over the state space, and updates it in closed form as the sequence arrives.
KLA retains the parallel training and prefill advantages of linear state space models such as GLA and Mamba,
while also providing a more expressive update (fractional linear / Möbius) that allows it to model uncertainty in the current state.
KLA can be used as a drop in replacement for other linear attention layers, and can be used in any architecture that uses attention.

|                      | Softmax attention | SSMs / GLA | **KLA**                        |
| -------------------- | ----------------- | ---------- | ------------------------------ |
| Expressivity         | nonlinear         | linear     | **fractional linear (Möbius)** |
| Training             | `O(T²)`           | `O(T)`     | `O(T)`                         |
| Inference            | `O(T)`            | `O(1)`     | `O(1)`                         |
| Sequence uncertainty | ❌                 | ❌          | ✅                              |
| Parallel training    | ✅                 | ✅          | ✅                              |

![The KLA block: two coupled streams](docs/figures/kla_block.png)

The blue stream is not a side-channel: the Kalman gain `alpha_t` is a function of
the model's own uncertainty, so it feeds back into the mean.

## Install

Use either installation method below with [uv](https://docs.astral.sh/uv/).
The commands create a virtual environment and activate it so `python` uses
that environment.

**PyPI:**
```bash
uv venv
source .venv/bin/activate
uv pip install kla
```

**From source:**
```bash
git clone https://github.com/vaisakh-shaj/kalman-linear-attention.git
cd kalman-linear-attention
uv venv
source .venv/bin/activate
uv pip install .
```

If you already cloned the repository, enter its directory and skip the clone
command. In each new shell, run `source .venv/bin/activate` from the directory
containing `.venv` before using `python`; you only need to create the environment
once.

The CUDA backend compiles on first use and requires a compatible CUDA toolkit
and C++ compiler, even if PyTorch already runs on your GPU. If `CUDA_HOME` or
`nvcc` is missing, install/load the toolkit before starting Python; see
[CUDA setup](docs/backends.md#cuda-setup) for commands and troubleshooting.

Runs on CPU, NVIDIA GPUs and Apple silicon out of the box. To see what this
machine will use:

```bash
python -m kla --check-backends
```

### Supported backends

KLA supports CPU, NVIDIA GPUs and Apple silicon. Speed comparisons below
refer to NVIDIA GPU training.

| Backend | Hardware and setup | Training speed | Training memory |
| --- | --- | --- | --- |
| `torch` | CPU, NVIDIA and Apple; standard PyTorch, easy to read and modify | Slow reference implementation | High: materializes expanded scan intermediates |
| `triton` | NVIDIA; easy setup with CUDA-enabled PyTorch and Triton | Faster than Torch | High: materializes expanded scan intermediates |
| `cuda` | NVIDIA; compatible CUDA toolkit and C++ compiler | Fastest in the paper's benchmarks; substantially faster than Triton | Much smaller scan-memory footprint |
| `mps` | Apple silicon; MPS-enabled PyTorch with Metal shader support | Fast, fused Metal kernels | Small scan-memory footprint: chunk checkpoints and recomputation |

We used the CUDA backend for our large-scale pretraining experiments on billions
of tokens with long sequence lengths.
It follows the same memory-saving principle as [Mamba's fused scan](https://arxiv.org/abs/2312.00752).

Use `backend="cuda"` for NVIDIA GPU training performance, or `backend="auto"`
for easier setup. For tensors on the corresponding device, `auto` selects Triton
on NVIDIA GPUs or MPS on Apple silicon when those kernels are available,
and Torch otherwise. See [backend details and setup](docs/backends.md).

## Structure

This repository is split in two parts:
- `src/kla`: The package containing the KLA layer and kernels.
- (Coming Soon) `experiments/` + `main.py` (with `nanochat/` and `mad/` submodules): Non-package code to reproduce the papers experiments.

The ancillary parts are:
- `docs/`: General documentation - [usage](docs/usage.md), [backends](docs/backends.md).
- `tests/`: Unit tests for the package.

### Package

```python
import torch
from kla import KLAConfig, KLALayer, ModelConfig, SequenceModel

layer = KLALayer(d_model=512, config=KLAConfig(d_state=16))
y = layer(torch.randn(2, 1024, 512))

# Inference after training: stateful prefill + O(1) per-token decode
layer.eval()
with torch.inference_mode():
    state = layer.init_state(batch=2)
    y, state = layer(torch.randn(2, 1024, 512), state=state)  # prefill
    y, state = layer(torch.randn(2, 1, 512), state=state)  # decode one token

# a full language model
model = SequenceModel(
    ModelConfig(vocab_size=50304, d_model=512, n_layers=6), KLAConfig()
)
logits = model(torch.randint(0, 50304, (2, 256)))
```

For NVIDIA GPU training, select `backend="cuda"` after completing
[CUDA setup](docs/backends.md#cuda-setup):

```python
layer = KLALayer(
    d_model=512,
    config=KLAConfig(d_state=16, backend="cuda"),
).cuda()
y = layer(torch.randn(2, 1024, 512, device="cuda"))
```

CUDA currently supports static dynamics with `d_state <= 64` and does not return
the filter state. Use `auto` for stateful prefill and decoding, as above.

Full API, config reference and the two published blocks: [docs/usage.md](docs/usage.md).

<details>
<summary>Block configurations: pretraining and MAD</summary>

The default is the **Mamba-style block**, used for pretraining for parameter
efficiency: values come directly from the conv stream, and observation variance
uses a low-rank projection. Our MAD experiments use the **plain block**, with
full value and variance projections. Both configurations use the same Kalman scan.

```python
from kla import KLAConfig

KLAConfig()  # default: value_rank="conv", var_rank="auto"
KLAConfig(value_rank="full", var_rank="full")  # plain block for MAD
```

</details>

### Experiments

*Coming Soon*

## Updates

- **2026-09-09:** Updated `backend="cuda"` to use v3 with faster backward computation
  and corrected gradients, including chunk-boundary fixes. Improved initialization
  of dynamics parameters and observation noise. Previous kernels remain selectable. See [backends](docs/backends.md).

## Citation

```bibtex
@article{shaj2026kla,
  title  = {Kalman Linear Attention: Parallel Bayesian Filtering For Efficient
            Language Modelling and State Tracking},
  author = {Shaj, Vaisakh and Barker, Cameron and Scannell, Aidan and
            Szecsenyi, Andras and Crowley, Elliot J. and Storkey, Amos},
  year   = {2026},
  eprint = {2602.10743},
  url    = {https://arxiv.org/abs/2602.10743},
}
```

MIT licensed.
