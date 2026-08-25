# Usage

`KLALayer` is a sequence mixer. It takes `[B, L, d_model]` and returns
`[B, L, d_model]`, so it drops into any place an attention layer goes.

<img src="figures/kla_block_scaffold.png" alt="KLA inside a gated linear attention block" width="200">

It is the whole block: the projections, causal conv and gate around the filter
are already inside it.

```python
import torch
from kla import KLAConfig, KLALayer

layer = KLALayer(d_model=512, config=KLAConfig(d_state=16))
y = layer(torch.randn(2, 1024, 512))  # [2, 1024, 512]
```

`d_model` is a constructor argument, everything else lives on `KLAConfig`.
Omitting the config gives the published defaults.

## The two knobs that matter

| knob | what it does | typical |
| --- | --- | --- |
| `d_model` | model width | 128 to 4096 |
| `d_state` | filter state size per channel | 8 to 64 |

`d_state` is how much the filter remembers. Memory and compute scale linearly in
it, and 16 is a good default. Some backends cap it; see
[implementations.md](implementations.md).

## Stateful decode

Pass a state to get `O(1)` per-token decoding. The layer returns
`(out, new_state)` whenever you pass a state or set `return_state=True`:

```python
state = layer.init_state(batch=2)
y, state = layer(torch.randn(2, 1024, 512), state=state)  # prefill
y, state = layer(torch.randn(2, 1, 512), state=state)  # decode one token
```

## Uncertainty

KLA carries a belief, not a point estimate, so it can return the propagated
per-token, per-channel variance alongside the output:

```python
layer = KLALayer(d_model=512, config=KLAConfig(return_variance=True))
y, y_var = layer(torch.randn(2, 1024, 512))  # both [2, 1024, 512]
```

Set `decode_from_prior=True` to emit the one-step-ahead prior prediction instead
of the filtered posterior.

## A whole language model

```python
import torch
from kla import KLAConfig, ModelConfig, SequenceModel

model = SequenceModel(
    ModelConfig(vocab_size=50304, d_model=512, n_layers=6), KLAConfig()
)
logits = model(torch.randint(0, 50304, (2, 256)))  # [2, 256, 50304]
```

`SequenceModel` is embedding + N blocks (mixer, optionally followed by an MLP) +
LM head.

The block stack is generic over its sequence mixer, so a baseline comparison is a
config swap:

```python
from kla import register_mixer

register_mixer(MyMixerConfig, lambda d_model, cfg: MyMixer(d_model, cfg))
model = SequenceModel(ModelConfig(...), MyMixerConfig(...))
```

## The two published blocks

Two presets, differing only in how the sensor path is shaped. Neither touches the
scan.

```python
KLAConfig(value_rank="full", var_rank="full")  # plain block (the default)
KLAConfig(value_rank="conv", var_rank="dt")  # mamba block
```

Quality is comparable, so pick on parameter budget: at `d_model=512` the mamba
block is 1.79M parameters against plain's 3.76M. The paper uses plain for the MAD
synthetics and mamba for the FineWeb-Edu pretraining runs.

## Functional API

The scan is usable directly, without the layer:

```python
from kla.ops import kla_scan, kla_step, kla_scan_reference, init_state
```

In paper notation the inputs are value `v` and value precision `Λ^v`
(`[B, L, M]`), observation map `k` and readout `q` (`[B, L, S]`), and the
time-invariant discrete decay `a` and process noise `p` (`[M, S]`). All three
return `(y, y_var, final_state)`.

`kla_scan` is the dispatcher. `backend=` takes an implementation name
(`"mps_merged_chunk"`), a bare backend name (`"mps"`), or `"auto"`.
`kla_step` is one recurrent step for decode. `kla_scan_reference` is the
sequential loop every implementation is validated against.
