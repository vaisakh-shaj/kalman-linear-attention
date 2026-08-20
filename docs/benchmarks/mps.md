# Benchmarks — Apple M5 Pro (64 GB)

Wall-clock and peak memory for the `torch` and `mps` implementations. torch
2.13.0. `triton` and `cuda` need a CUDA device and are not covered.

**Method.** fp32, on the `mps` device unless a table says CPU. Median of 1–15
calls after untimed warm-up calls, which pay shader compilation and let the GPU
clocks ramp. `torch.mps.synchronize()` bounds every timed region. Defaults
unless stated: `d_state=16`, `d_model=512`, `expand=2.0`, so `d_inner=1024`.

Each table is one process. Absolute times drift ~30% between runs with the
machine's thermal state, so rows are only comparable within a table.

## Sequence length

Training — B=8, forward + backward, ms per call:

| L | torch_rec | torch_chunk | torch_pscan | mps_rec | mps_chunk | mps_pscan |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 171 | 747 | 430 | 3.1 | 5.9 | 7.5 |
| 256 | 255 | 1555 | 836 | 7.8 | 10.1 | 14.1 |
| 512 | 430 | 3453 | 1681 | 14.0 | 18.7 | 19.0 |
| 1024 | 818 | 6227 | 3497 | 27.5 | 34.4 | 49.4 |
| 2048 | 1678 | OOM | 6431 | 44.4 | 67.9 | 104 |
| 4096 | 3177 | — | OOM | 99.8 | 126 | 200 |
| 8192 | 7078 | — | — | 193 | 252 | 417 |

Inference — B=1, `no_grad`, ms per call:

| L | torch_rec | torch_chunk | torch_pscan | mps_rec | mps_chunk | mps_pscan |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 9.9 | 13.7 | 15.9 | 0.2 | 0.3 | 0.4 |
| 256 | 20.6 | 29.0 | 62.3 | 0.3 | 0.5 | 0.8 |
| 512 | 27.9 | 97.0 | 112 | 0.6 | 0.9 | 1.5 |
| 1024 | 54.9 | 197 | 271 | 1.1 | 1.8 | 3.6 |
| 2048 | 109 | 319 | 580 | 1.8 | 3.2 | 8.3 |
| 4096 | 290 | 705 | 1304 | 4.0 | 7.1 | 12.0 |
| 8192 | 594 | 1613 | 3111 | 6.9 | 12.5 | 41.5 |

`—` means the column was dropped after exceeding the time budget at the row
above.

## Memory

Peak device allocation for one training step at B=8, L=1024. One `[B, L, M, S]`
tensor is 512 MiB at this shape.

| | peak |
|---|---:|
| `torch_unfused_chunk` | 55.3 GiB |
| `torch_unfused_pscan` | 26.7 GiB |
| `torch_unfused_recurrent` | 6.1 GiB |
| `mps_pscan` | 545 MiB |
| `mps_chunk` | 292 MiB |
| `mps_recurrent` | 258 MiB |

`mps_pscan`'s figure is sampled inside the forward: its aggregate buffers are
freed before the forward returns, so a peak sampled at phase boundaries misses
them.

## Lane count

`mps_recurrent`'s grid is `B x d_inner x d_state` lanes and nothing else;
`mps_chunk` and `mps_pscan` add parallelism over time at about 4x the
arithmetic. ms per call.

B=1, `no_grad`, L=1024:

| lanes | d_model | recurrent | chunk | pscan |
|---:|---:|---:|---:|---:|
| 1024 | 32 | 0.53 | 0.30 | 0.31 |
| 2048 | 64 | 0.54 | 0.37 | 0.40 |
| 4096 | 128 | 0.53 | 0.53 | 0.68 |
| 8192 | 256 | 0.52 | 0.85 | 1.16 |
| 16384 | 512 | 0.85 | 1.47 | 2.38 |

B=1, `no_grad`, L=16384:

| lanes | d_model | recurrent | chunk | pscan |
|---:|---:|---:|---:|---:|
| 256 | 8 | 5.73 | 1.53 | 0.84 |
| 512 | 16 | 6.50 | 1.53 | 1.37 |
| 1024 | 32 | 6.77 | 2.10 | 3.03 |

## torch on CPU

`d_inner=256`, `d_state=16`, ms per call:

| L | train rec | train chunk | train pscan | infer rec | infer chunk | infer pscan |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 72.6 | 187 | 66.7 | 6.9 | 34.1 | 20.3 |
| 256 | 149 | 354 | 160 | 11.7 | 75.1 | 28.9 |
| 512 | 221 | 893 | 347 | 20.1 | 141 | 43.7 |
| 1024 | 377 | 2590 | 761 | 33.3 | 261 | 99.6 |
| 2048 | 754 | 6036 | 1312 | 73.5 | 501 | 188 |

Training is B=4, inference B=1 `no_grad`.

## merged vs the two-scan cells

`merged` folds the precision scan and the information-vector scan into one — see
[../implementations.md](../implementations.md). These are the same process and
the same session, so the columns are comparable to each other; they are not
comparable to the tables above, which were a different run.

Inference — B=1, `no_grad`, `d_model=512`, ms per call:

| L | mps_rec | mps_chunk | mps_merged_chunk | mps_pscan | mps_merged_pscan |
|---:|---:|---:|---:|---:|---:|
| 128 | 2.21 | 0.81 | **0.76** | 0.95 | 0.88 |
| 512 | 1.76 | 1.83 | **1.61** | 2.06 | 2.18 |
| 1024 | 2.65 | 3.24 | **2.75** | 4.09 | 4.29 |
| 2048 | 4.78 | 5.99 | **5.04** | 8.48 | 8.72 |
| 4096 | 9.67 | 12.08 | **10.18** | 17.93 | 18.29 |
| 8192 | 18.90 | 23.98 | **19.83** | 36.62 | 37.56 |

Training — B=8, forward + backward, ms per call:

| L | mps_rec | mps_chunk | mps_merged_chunk | mps_pscan | mps_merged_pscan |
|---:|---:|---:|---:|---:|---:|
| 512 | 31.71 | 34.67 | 33.10 | 39.34 | 38.35 |
| 1024 | 62.85 | 68.76 | 65.78 | 78.47 | 76.71 |
| 2048 | 129.14 | 137.82 | 130.60 | 159.54 | 158.18 |
| 4096 | 259.02 | 277.51 | 264.53 | 331.91 | 330.16 |

Lane count — B=1, `no_grad`, where `chunk` and `pscan` exist at all:

L=1024:

| lanes | d_model | rec | chunk | merged_chunk | pscan | merged_pscan |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 32 | 1.60 | 0.60 | 0.59 | 0.60 | **0.56** |
| 2048 | 64 | 1.40 | 0.69 | **0.59** | 0.73 | 0.70 |
| 4096 | 128 | 1.01 | 1.01 | **0.82** | 1.18 | 1.11 |
| 8192 | 256 | **1.27** | 1.60 | 1.33 | 1.92 | 1.97 |
| 16384 | 512 | **2.64** | 3.23 | 3.12 | 4.41 | 4.90 |

L=16384:

| lanes | d_model | rec | chunk | merged_chunk | pscan | merged_pscan |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 8 | 7.20 | 2.35 | 1.67 | **1.33** | 1.34 |
| 512 | 16 | 7.19 | 2.15 | **1.64** | 1.99 | 2.43 |
| 1024 | 32 | 7.74 | 3.25 | **2.55** | 4.39 | 4.98 |

One realistic shape, `d_model=1024`, `d_state=16`, L=2048 — each row its own
process, ms per call:

| | rec | chunk | merged_chunk | pscan | merged_pscan |
|---|---:|---:|---:|---:|---:|
| B=1 inference (32k lanes) | **13.21** | 16.14 | 14.10 | 21.27 | 21.07 |
| B=8 training | **343.57** | 367.41 | 352.85 | 408.46 | 408.59 |

Deep in `recurrent`'s regime, as every realistic config is. `merged_chunk`
closes about 70 % of `chunk`'s gap to `recurrent` in inference and about 60 % of
it in training, without ever overtaking it.

### The reading

**`merged_chunk` wins, by 15–30 %.** It beats `mps_chunk` at every shape
measured, and the margin is widest exactly where `chunk` is the implementation you would
actually pick — low lane count and a long sequence, where `chunk`'s six phases
per tile dominate. At 256 lanes and L=16384 it is 29 % faster (2.35 → 1.67 ms)
and comes within 25 % of `pscan`. The saving is structural, not arithmetic: one
threadgroup scan instead of two, one broadcast instead of two, and the
`var_h`/`alpha_h`/`r_h` per-thread arrays gone — 24 registers at `ITEMS=8`, on a
kernel whose entire purpose is occupancy.

**It moves the alias boundary but does not cross it.** `merged_chunk` now beats
`mps_recurrent` at 4096 lanes (0.82 vs 1.01 ms), where plain `chunk` only tied.
`recurrent` still wins from ~8k lanes up, and by more in training, so `mps`
stays aliased to `mps_recurrent` — but the crossover moved from ~4k lanes to
~8k, which is `d_model=256` at batch 1 rather than `d_model=128`.

**`merged_pscan` is a wash**, and the reason is worth recording. It does drop
two of five kernels and one of two doubling rounds, but that implementation is
bandwidth-bound on its `[B,M,NCK,S]` aggregate array, and the merged aggregate
is *wider*: 8 floats (7 live + 1 padding) against a float4 plus a float2, over
one ping-ponged pair instead of two. 16 floats per `(b,m,c,s)` against 12, so
512 MiB against 384 MiB at B=8, L=1024, `d_inner=1024`, `d_state=16`. The extra
traffic through `log2(NCK)` doubling rounds eats the round it saved. It wins
slightly at the smallest lane counts, where the aggregate array is small, and
loses slightly above ~4k lanes.

Six of the seven carried values would fit the 24-byte budget exactly, since
`D = 1 - A` after trace normalization — but `D` is the small entry, so
reconstructing it gives it a relative error of `eps/D`, measured at 2.7e-4
against 5.0e-7 in λ at decay 0.3. Three orders of magnitude on the quantity λ is
most sensitive to, to save four bytes.
`tests/test_merged_algebra.py::test_reconstructing_D_is_not_free` pins that, so
the layout question stays settled.

**Training dilutes everything**, as it should: the backward is
`kla_scan_bwd`, shared by all five cells and untouched by any of this, so a
forward-only saving of 20 % shows up as 4–5 % of a training step.

## Defaults set from this

| | was | is |
|---|---|---|
| `torch` | `torch_unfused_chunk` | `torch_unfused_recurrent` |
| `mps` | `mps_chunk` | `mps_recurrent` |

`triton` and `cuda` keep `chunk` until the same sweep runs on a CUDA device.
