# Implementations

One algorithm, several implementations of its core scan. Each one is named by
two things: which **backend** compiles it, and which **implementation** it uses to get
through the sequence.

```
<backend>[_unfused]_<implementation>
```

## The three implementations

The scan is a recurrence: step `t` needs step `t-1`. The three implementations are
three answers to "how much of the sequence can run at once?"

### recurrent

Walk the sequence one step at a time. Nothing about time runs in parallel.

```
t:  0 → 1 → 2 → 3 → 4 → 5 → 6 → 7
```

Parallelism comes from everything *else* — one thread per (batch, channel,
state). Good when there are lots of sequences or channels to fill the GPU with.
Bad when there are not.

It is also the cheapest per step: a serial walk can *apply* the update to a
running value rather than compose two maps, which is about a quarter of the
arithmetic and materializes nothing. Every `recurrent` cell does this, torch
included — that one carries `(λ, η)` through
`torch._higher_order_ops.scan`, so `mobius_impl` has nothing to choose between
there because nothing is composed.

### chunk

Cut the sequence into chunks. Work inside a chunk in parallel, then hand the
result to the next chunk.

```
t:  [0 1 2 3] → [4 5 6 7]
     parallel     parallel
```

The chunks are still serial, but there are far fewer of them than there are
timesteps. This is the general-purpose choice and the default everywhere.

### pscan

Parallel scan. Every chunk starts at once, without waiting for the one before
it. A second pass stitches them together afterwards.

```
t:  [0 1 2 3]  [4 5 6 7]      all at once
         ↓          ↓
       stitch the chunks       second pass
```

Nothing is serial, so this fills the GPU even when there is only one short
sequence. It costs more total work than `chunk` to get there. Worth it for
batch-1 prefill of a long sequence, and rarely otherwise.

**Rule of thumb:** the implementations split on how many lanes `B x d_inner x d_state`
gives you, since that is `recurrent`'s entire grid. Measured on an M5 Pro:
`recurrent` above ~4k lanes, `chunk` from ~512 to ~4k, `pscan` below that.
Every realistic config is in `recurrent`'s range — 4k lanes is `d_model=128` at
batch 1 — which is why it is the default on the two backends that have been
measured. See [benchmarks/mps.md](benchmarks/mps.md).

## fused and unfused

**fused** (the default, no token in the name) does the whole scan inside the
kernel. The big intermediate tensors never reach memory.

**unfused** uses standalone scan kernels with ordinary torch code between them.
Slower, because those intermediates do reach memory, but it is easier to read
and to change. It is the reference the fused versions are checked against.

## The matrix

| | recurrent | chunk | pscan |
|---|---|---|---|
| **torch** | `torch_unfused_recurrent` | `torch_unfused_chunk` | `torch_unfused_pscan` |
| **triton** | `triton_recurrent` | `triton_chunk` | `triton_pscan` |
| **triton, unfused** | `triton_unfused_recurrent` | `triton_unfused_chunk` | `triton_unfused_pscan` |
| **cuda** | `cuda_recurrent` | `cuda_chunk` | `cuda_pscan` |
| **mps** | `mps_recurrent` | `mps_chunk` | `mps_pscan` |

A bare backend name means that backend's default:

| backend | default | why |
|---|---|---|
| `torch` | `torch_unfused_recurrent` | measured — fastest in both regimes, and 4–9x the smallest footprint |
| `mps` | `mps_recurrent` | measured — fastest at every shape past `d_model=128` |
| `triton` | `triton_chunk` | not yet measured; `chunk` is the safe middle |
| `cuda` | `cuda_chunk` | not yet measured; `chunk` is the safe middle |

`auto` picks the default for whichever device the tensors are on. See
[benchmarks/mps.md](benchmarks/mps.md) for the numbers behind the two that have
been measured, including where `chunk` and `pscan` do win.

Every cell in the table exists. Run `python -m kla --check-backends` for what
*this* machine can actually run — a cell needs its toolchain and its device
present — and see [rework-plan.md](rework-plan.md) for what has been measured
on real hardware.

`cuda_v2_1` and `cuda_v2_2` are earlier CUDA kernels, kept for comparison. They
are the only implementations with an approximate backward.

## What every implementation guarantees

- forward and backward
- an exact backward
- the filter state carried in and out, differentiable both ways
- `decode_from_prior`
- a float32 scan

Anything that misses one of these is a bug, not a variant. The two `cuda_v2_*`
kernels are the documented exceptions.

`d_state` has a per-implementation ceiling, because the read-out sums over the
state axis and all of one channel's states have to sit in one block. The
practical range is `d_state <= 32`, which is well under every ceiling and is
also the fastest path — 32 is one warp on CUDA and one SIMD-group on Metal, so
that sum is a register shuffle.
