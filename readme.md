# KLA - Kalman Linear Attention

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
# or
uv sync
```

Runs on CPU, NVIDIA GPUs and Apple silicon out of the box. To see what this
machine will use:

```bash
python -m kla --check-backends
```

The scan has four backends (torch, triton, cuda, mps) and three implementations each
— see [docs/implementations.md](docs/implementations.md) for the naming and
[docs/backends.md](docs/backends.md) for what each needs. For the fastest option
on NVIDIA hardware:

```bash
uv pip install -U "kla[cuda]" # --torch-backend cu126
```

## Structure

This repository is split in two parts:
- `src/kla`: The package containing the KLA layer and kernels.
- (Coming Soon) `experiments/` + `main.py` (with `nanochat/` and `mad/` submodules): Non-package code to reproduce the papers experiments.

The ancillary parts are:
- `docs/`: General documentation — [usage](docs/usage.md), [backends](docs/backends.md), [implementations](docs/implementations.md), [benchmarks](docs/benchmarks/).
- `tests/`: Unit tests for the package.

### Package

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

Full API, config reference and the two published blocks: [docs/usage.md](docs/usage.md).

### Experiments

*Coming Soon*

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
