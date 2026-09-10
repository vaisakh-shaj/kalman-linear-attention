# Usage

`KLALayer` is a sequence mixer.
It takes `[B, L, d_model]` and returns `[B, L, d_model]`,
so it drops into any place an attention layer goes.

<img src="figures/kla_block_scaffold.png" alt="KLA inside a gated linear attention block" width="200">

`KLALayer` is the whole block — the projections, causal conv and gate around the
filter are already inside it.

```python
import torch
from kla import KLAConfig, KLALayer

layer = KLALayer(d_model=512, config=KLAConfig(d_state=16))
y = layer(torch.randn(2, 1024, 512))  # [2, 1024, 512]
```

`d_model` is a constructor argument; everything else lives on `KLAConfig`.
Omitting the config gives you the parameter-efficient Mamba-style block used for
pretraining. Select full projections explicitly for the MAD architecture.

## The two knobs that matter

```python
KLAConfig(d_state=16)  # and d_model, which you pass to the layer
```

| knob      | what it does                  | typical    |
| --------- | ----------------------------- | ---------- |
| `d_model` | model width                   | 128 - 4096 |
| `d_state` | filter state size per channel | 8 - 64     |

`d_state` is the one specific to KLA: how much the filter remembers. Memory and
compute scale linearly in it, and 16 is a good default. Below 8 the filter starts
to degenerate; above 64 you rarely gain. (64 is also the ceiling for the CUDA
backend, and 128 for the fused MPS one.)

Everything else has a sensible default.

## Stateful decode

Pass a state to get `O(1)` per-token decoding. The layer returns
`(out, new_state)` whenever you pass a state or set `return_state=True`:

```python
state = layer.init_state(batch=2)
y, state = layer(torch.randn(2, 1024, 512), state=state)  # prefill
y, state = layer(torch.randn(2, 1, 512), state=state)  # decode one token
```

Without a state it returns the tensor alone, which is why the plain call above
does not unpack.

## Uncertainty

KLA carries a belief, not a point estimate, so it can hand you the propagated
per-token, per-channel variance alongside the output:

```python
layer = KLALayer(d_model=512, config=KLAConfig(return_variance=True))
y, y_var = layer(torch.randn(2, 1024, 512))  # both [2, 1024, 512]
```

`return_variance` changes `out` from `y` to `(y, y_var)`. Combined with a state
that becomes `((y, y_var), new_state)`.

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
LM head. `ModelConfig` carries `vocab_size`, `d_model`, `n_layers`, `mlp`
(`"swiglu"` / `"gelu"` / `"none"`), `mlp_ratio`, `norm_eps`, `tie_embeddings`,
and an optional `logit_softcap`.

### Other mixers

The block stack is generic over its sequence mixer. Register a config type
against a builder and `SequenceModel` will use it:

```python
from kla import register_mixer

register_mixer(MyMixerConfig, lambda d_model, cfg: MyMixer(d_model, cfg))
model = SequenceModel(ModelConfig(...), MyMixerConfig(...))
```

That is what makes baseline comparisons a config swap rather than a fork.

## The two published blocks

Two config presets, differing only in how the sensor path is shaped. Neither
touches the scan — both emit the same tensors, so no backend or kernel changes.

```python
KLAConfig()  # mamba block (default): value_rank="conv", var_rank="auto"
KLAConfig(value_rank="full", var_rank="full")  # plain block used for MAD
```

`value_rank` controls how the value `v` is produced from the post-conv stream.
It is the most expensive projection in the layer: `"full"` costs `M²` per block,
`"conv"` (v = z, Mamba's move) costs nothing. `var_rank` does the same for the
observation noise, and low-ranking *that* is the safe one — it is a smooth
per-channel noise level, exactly what Mamba does to `Δ`.

A rank only *saves* when `2*rank < d_inner`: the bottleneck costs two projections
where the full map costs one. Solve for it against a parameter budget rather than
picking one by eye. `"auto"` uses a bottleneck of width `ceil(d_model / 8)`;
pass an integer directly to choose another width for either projection.
The defaults are `value_rank="conv"` and `var_rank="auto"`: no value projection
and an automatic bottleneck for variance. `"auto"` replaces the old `"dt"`
setting; the separate `dt_rank` override has been removed.

The paper uses plain for MAD and Mamba-style for FineWeb-Edu pretraining for
parameter efficiency. At `d_model=512`, the current Mamba-style configuration
has about 1.79M parameters against plain's 3.76M. Both use the same Kalman scan.
When loading an existing checkpoint, specify its original projection settings
and bottleneck widths explicitly; changing defaults does not convert weights.

## Functional API

The scan is usable directly, without the layer:

```python
from kla.ops import kla_scan, kla_step, kla_scan_reference, init_state
```

In paper notation the inputs are value `v` and value precision `Λ^v`
(`[B, L, M]`), observation map `k` and readout `q` (`[B, L, S]`), and the
time-invariant discrete decay `a` and process noise `p` (`[M, S]`). All three
return `(y, y_var, final_state)`.

`kla_scan` is the dispatcher — it takes `backend=`, plus `scan_impl` and
`mobius_impl` for the torch path. `kla_step` is one recurrent step for decode.
`kla_scan_reference` is the sequential loop every backend is validated against.

### Noise initialization

New layers learn continuous process noise as `exp(p_log)`, initialized to
`3 * abs(a) * delta` using each cell's sampled delta. Noise and delta then learn
independently. The observation-variance head starts at the constant `1/d_state`
with zero output weights and a learnable bias; `obs_noise_init` sets a different
baseline. The heuristic assumes normalized keys and is an initialization choice,
not a training-stability guarantee.

Use `process_noise_init=0.02` to initialize continuous noise to a constant.
This replaces `process_noise_init="constant", process_noise_scale=0.02`.
For legacy checkpoints, construct the layer with `process_noise_param="raw"`,
`process_noise_init=0.01`, and `obs_noise_init=None` before loading. Also match
the checkpoint's projection settings: old default models need
`value_rank="full", var_rank="full"`.
Raw noise and log noise have different state-dict keys; no automatic conversion
is performed.

`process_noise_mode` controls whether continuous noise is `"learned"` (default),
`"fixed"` at its initialized value, or `"zero"` regardless of initialization.
Fixed and zero modes use non-trainable buffers; the scan retains its numerical
noise floor. This replaces `learnable_process_noise` and `zero_process_noise`.
For example, `KLAConfig(process_noise_mode="fixed", process_noise_init=0.02)`
keeps continuous noise at 0.02 while the dynamics and delta can still learn.
