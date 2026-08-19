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

## Defaults set from this

| | was | is |
|---|---|---|
| `torch` | `torch_unfused_chunk` | `torch_unfused_recurrent` |
| `mps` | `mps_chunk` | `mps_recurrent` |

`triton` and `cuda` keep `chunk` until the same sweep runs on a CUDA device.
