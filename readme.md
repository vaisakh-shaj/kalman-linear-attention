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

**PyPI:**
```bash
uv pip install kla    # or: uv add kla
```

**From source:**
```bash
git clone https://github.com/vaisakh-shaj/kalman-linear-attention.git kla
uv pip install ./kla
```

The CUDA backend compiles on first use and requires a compatible CUDA toolkit
and C++ compiler. See [CUDA setup](docs/backends.md#cuda-setup).

Runs on CPU, NVIDIA GPUs and Apple silicon out of the box. To see what this
machine will use:

```bash
python -m kla --check-backends
```

For backend selection and CUDA setup, see [the backend guide](docs/backends.md).

## Structure

This repository is split in two parts:
- `src/kla`: The package containing the KLA layer and kernels.
- (Coming Soon) `experiments/` + `main.py` (with `nanochat/` and `mad/` submodules): Non-package code to reproduce the papers experiments.

The ancillary parts are:
- `docs/`: General documentation — [usage](docs/usage.md), [backends](docs/backends.md).
- `tests/`: Unit tests for the package.

### Package

The default is the **Mamba-style block**, used for pretraining for parameter
efficiency: values come directly from the conv stream, and observation variance
uses a low-rank projection. Our MAD experiments use the **plain block**, with
full value and variance projections. Both configurations use the same Kalman scan.

```python
from kla import KLAConfig

KLAConfig()  # default: value_rank="conv", var_rank="auto"
KLAConfig(value_rank="full", var_rank="full")  # plain block for MAD
```

For existing checkpoints, use the projection settings and ranks they were trained
with; older default models used full projections.

```python
import torch
from kla import KLAConfig, KLALayer, ModelConfig, SequenceModel

layer = KLALayer(d_model=512, config=KLAConfig(d_state=16))
y = layer(torch.randn(2, 1024, 512))

# stateful prefill + O(1) decode
state = layer.init_state(batch=2)
y, state = layer(torch.randn(2, 1024, 512), state=state)  # prefill
y, state = layer(torch.randn(2, 1, 512), state=state)  # decode one token

# a full language model
model = SequenceModel(
    ModelConfig(vocab_size=50304, d_model=512, n_layers=6), KLAConfig()
)
logits = model(torch.randint(0, 50304, (2, 256)))
```

**For training on NVIDIA GPUs, install KLA using the instructions above and
follow [CUDA setup](docs/backends.md#cuda-setup), then set `backend="cuda"`.**
This selects the latest supported CUDA kernel included in your installed KLA
release; you do not need to choose a kernel version. For easier setup, keep
`backend="auto"`.

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
