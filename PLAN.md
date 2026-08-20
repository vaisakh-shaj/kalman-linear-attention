# The merged 3x3 formulation

Fold the precision scan and the information-vector scan into **one** associative
scan, by writing the whole step as a single 3x3 linear map in homogeneous
coordinates. Add it as a third value on the fusion axis, called `merged`.

Status: **implemented on torch and mps**, tested and measured. `triton` and
`cuda` are not done — no device here to run them on. Two of the plan's
predictions did not survive contact; both are marked ✗ below.

---

## The algebra

`λ` is already projective: with `λ = u/v` the precision step is the 2x2 Möbius
matrix the kernels compose today. The claim is that `η` rides in the same
coordinates.

From `C = p/a²`, `D = 1`:

```
v_t = C·u_{t-1} + D·v_{t-1} = (p·u_{t-1} + a²·v_{t-1}) / a² = den_t·v_{t-1} / a²
```

so the gain is a ratio of the coordinate the other scan already carries:

```
α_t = a / den_t = v_{t-1} / (a · v_t)
```

Put `w = v·η`. The `v_t` cancels and the whole step is linear:

```
w_t = v_t·η_t = v_t·α_t·η_{t-1} + v_t·r_t = (1/a)·w_{t-1} + r_t·v_t

[u]   [ A     B     0  ] [u]        A = (1+pφ)/a²    B = φ
[v] = [ C     D     0  ] [v]        C = p/a²         D = 1
[w]   [r·C   r·D   1/a ] [w]        λ = u/v,  η = w/v
```

Lower block-triangular, two structural zeros, and a constant `(3,3)`.
Composition is `[[P,0],[q,s]]`:

```
P = P₂·P₁          q = q₂·P₁ + s₂·q₁          s = s₂·s₁
```

**Normalization is load-bearing.** `s` accumulates `a⁻ⁿ`, which overflows fp32
outright for a decaying filter — measured 7e142 at `a=0.5`, L=200, unnormalized.
Dividing the whole 3x3 by the 2x2's trace fixes it, because `λ = u/v` and
`η = w/v` are both invariant under a common rescale of `(u,v,w)`, and `∏τ` grows
faster than `a⁻ⁿ`. `s` then *decays* to zero, which is the right physics — the
initial `η` stops mattering. The 2x2 block stays bounded by 1, exactly as now.

`D = 1 - A` after trace normalization, so the carried state *could* be six
values (`A, B, C, qa, qb, s`) against today's `4 + 2`. **It cannot.** `D` is the
small entry (≈ `a²/(1+pφ)`), so recovering it as `1 - A` gives it an absolute
error of one ulp of `A` — a relative error of `eps/D`, in the entry `λ` is most
sensitive to. Measured in fp32 against a float64 reference: `λ` error 2.7e-4
reconstructed against 5.0e-7 carried, at decay 0.3. Three orders of magnitude to
save four bytes. The kernels carry seven values padded to eight, so the
aggregate is two float4 loads;
`test_reconstructing_D_is_not_free` keeps the question settled.

### Validated — now in `tests/test_merged_algebra.py`

Re-run as a repo test before the kernels were written, as this section asked.
21 tests. What held and what did not:

- ✓ Correct: float64 vs the sequential reference — `λ` < 1e-12, `η` < 1e-10, at
  every grouping of the combine (chunk / doubling / associative).
- ✓ Normalization is load-bearing: `test_unnormalized_overflows` runs the
  unnormalized combine in fp32 and asserts it *fails*; `test_normalized_s_decays`
  shows the normalized `s` going to zero.
- ✗ **"η equal or better at every point" does not reproduce.** On this repo's
  input distribution both paths sit at the fp32 floor and merged is
  consistently the slightly worse of the two: `η` 1.4e-7..4.9e-7 merged against
  0.9e-7..3.4e-7 two-scan, over decays 0.3/0.7/0.95 and L=8..512. The prototype
  compared at error scales of 3e-4, which this input distribution never reaches.
  `λ` *is* bit-identical, as predicted, and both stay below `λ`'s own error
  (2e-7..8e-7) which the two paths share exactly. Neither drifts with L. So
  merging costs a factor of ~2 on a quantity at the fp32 floor — recorded rather
  than asserted away.

## Why it is worth doing

Not the arithmetic — that is a wash, ~15 multiplies per compose against ~14 for
the 2x2-plus-affine pair. It is the **dependency**.

Today `α_t` needs `λ_{t-1}`, so the affine leaves do not exist until the Möbius
scan has produced `λ`. Every composing implementation therefore runs two scans,
and everything about their structure follows from that. The merged leaf is built
from `(φ, r, a, p)` alone, so the dependency disappears.

`recurrent` never had this problem — it *applies* the map, which hands it
`λ_{t-1}` for free. So `merged` is what gives the composing implementations back
the one-pass property the recurrent one always had.

## Naming: `merged` on the fusion axis

`<backend>[_unfused|_merged]_<implementation>`. `fused` stays the default and
carries no token. The axis reads as an ordering of how much is combined:

| | intermediates | scans |
|---|---|---|
| `unfused` | `[B,L,M,S]` in memory | two |
| `fused` | per-chunk only | two |
| `merged` | per-chunk only | **one** |

**`merged` does not apply to `recurrent`.** That implementation applies rather
than composes, so it already does `λ` and `η` in one pass; a merged variant
would be the same kernel under a second name, which is the defect this whole
naming scheme exists to prevent. Four new cells per backend at most:

| | recurrent | chunk | pscan |
|---|---|---|---|
| **fused** | `<b>_recurrent` | `<b>_chunk` | `<b>_pscan` |
| **merged** | — *(identical to fused)* | `<b>_merged_chunk` | `<b>_merged_pscan` |

**Resolved: torch gets both cells**, and `torch_merged_*` means "unfused, but
one scan". The single axis cannot spell both tokens, and unfused is what torch
always is, so there is nothing to disambiguate. They earn their place by being
the only merged cells that run float64 — which is what lets the merged algebra
be *gradchecked* (`test_gradcheck_merged_scan_float64`) instead of only compared
against another float32 implementation. `triton` and `cuda` have no merged cells
yet: no device here.

## What changes, per implementation

### chunk — the bigger win

Six phases collapse to three. D and E exist *only* to build the affine leaves
that C's `λ` unlocked.

| today | merged |
|---|---|
| A compose Möbius leaves | A compose 3x3 leaves |
| B tile scan (Möbius) | B tile scan, once |
| C walk applying → `λ, α, r` | C walk applying → `λ, η`, read out |
| D compose affine leaves | — |
| E tile scan (affine) | — |
| F walk applying → `η`, read out | — |

Three concrete effects:

- **One threadgroup scan instead of two.** `kla_tile_scan_mobius` and
  `kla_tile_scan_affine` are each `log2(ROWS)` Hillis-Steele rounds with two
  barriers per round. Halved, along with the two `kla_tile_broadcast` calls.
- **The per-thread arrays disappear.** `var_h[ITEMS]`, `alpha_h[ITEMS]`,
  `r_h[ITEMS]` in `chunk_kla_scan.metal` exist only to carry phase C's output to
  D and F. That is 24 registers per thread at `ITEMS=8`, freed — and occupancy
  is the entire reason the chunk implementation exists.
- **A numerical trick becomes unnecessary.** `chunk_kla_scan.py` recovers
  `λ_{t-1} = (D·λ_t − B)/(A − C·λ_t)` by inverting the leaf, specifically so the
  α-gain needs no cross-chunk `λ` shift. Merged never forms `α`, so the
  inversion and its stability argument both go.

### pscan

Five kernels to three: one reduce, one set of doubling rounds, one apply. The
second reduce-scan-apply round disappears with the dependency that forced it.

Memory is roughly a wash — the aggregate goes from `4 + 2` floats to `6`, and
both are ping-ponged.

### torch

`chunk_scan` in `scan.py` runs the Möbius scan, materializes `λ`, derives `α`,
then scans again. Same halving, and it is the cheapest place to prototype the
combine.

## Checklist

### Phase 0 — pin the algebra ✓

- [x] `tests/test_merged_algebra.py`: the float64 identity, the fp32 comparison
      against the two-scan composition, and an explicit overflow test showing
      the unnormalized form failing. 21 tests. It caught the η claim (above).
- [x] Carried layout: **seven values**, padded to eight. Six with `D = 1 - A`
      costs three orders of magnitude on λ — measured, see the algebra section.

### Phase 1 — torch first, where it is cheapest to be wrong ✓

- [x] `_merged_combine` in `kla_ops.py`, next to `_mobius_combine_tracenorm`,
      with `_merged_leaves` and `_merged_readout` beside it
- [x] `torch_merged_chunk`, `torch_merged_pscan` (naming resolved above)
- [x] Parity against `kla_scan_reference` at the existing tolerances, plus
      float64 gradcheck of the merged composition

### Phase 2 — kernels

- [x] `mps_merged_chunk` — `kla_merged.metal` (the shared 3x3 algebra) plus
      `merged_chunk_kla_scan.metal`. Six phases to three.
- [x] `mps_merged_pscan` — `merged_pscan_kla_scan.metal`. Five kernels to three.
- [ ] `triton_merged_chunk`, `triton_merged_pscan` — no CUDA device here
- [ ] `cuda_merged_chunk`, `cuda_merged_pscan` — no CUDA device here
- [x] Backward: **unchanged, and confirmed unchanged.** `kla_scan_bwd.metal`,
      `kla_scan_bwd.py` and the shared `scan_backward` are untouched — both
      merged cells write the same `[B,M,NCK,S]` checkpoints at the same stride
      with the same convention, and both are held to the same exact-gradient
      contract as every other cell in `tests/test_backends.py`. Nothing leaked.

### Phase 3 — measure ✓

- [x] `merged_chunk` vs `fused_chunk`: **a clear win, 15–30 %**, widest exactly
      where `chunk` is the implementation you would pick — 29 % at 256 lanes,
      L=16384. Numbers in `docs/benchmarks/mps.md`.
- [x] `merged_chunk` vs `mps_recurrent`: it crosses at 4096 lanes (0.82 vs
      1.01 ms) where plain `chunk` only tied, but `recurrent` still wins from
      ~8k lanes up and by more in training.
- [x] Aliases revisited and **left alone**: the crossover moved from ~4k lanes
      to ~8k, which is `d_model=256` at batch 1 rather than `d_model=128`. Every
      realistic config is still on `recurrent`'s side of it.
- [x] Folded into `docs/benchmarks/mps.md` and `docs/implementations.md`.

`merged_pscan` is the one that did not pay: ✗ **a wash, and sometimes a loss.**
It does drop two of five kernels and one of two doubling rounds, but that
implementation is bandwidth-bound on its `[B,M,NCK,S]` aggregate, and the merged
aggregate is *wider* — 16 floats per `(b,m,c,s)` against 12, over one
ping-ponged pair instead of two. The extra traffic through `log2(NCK)` rounds
eats the round it saved. The plan called this "roughly a wash" on the assumption
of six carried values; at seven-padded-to-eight it is a wash with the sign
occasionally wrong.

## The depth question — answered, and the answer is no

Both ways out were measured, on the real kernels, with no new code: each variant
is a parameter the merged cells already take.

- **A** — replace the serial within-chunk walk with a Hillis-Steele across
  threads. This is `mps_merged_chunk` at `ITEMS=1`, where the tile becomes a
  pure `log2(ROWS)` scan. Depth 5 instead of 20.
- **B** — no chunks at all, doubling in device memory over the whole sequence.
  This is `mps_merged_pscan` at `CHUNK=1`. Depth 16 instead of 42.

B=1, L=16384, `d_inner=64`, `d_state=16` (1024 lanes — the batch-1 prefill
regime these implementations exist for), forward only, ms:

| A: ITEMS | tile depth | ms |   | B: CHUNK | depth | aggregate | ms |
|---:|---:|---:|---|---:|---:|---:|---:|
| **1** | **5** | **3.02** |   | **1** | **16** | **1024 MiB** | **68.12** |
| 2 | 6 | 2.05 |   | 4 | 20 | 256 MiB | 15.48 |
| 4 | 8 | 1.56 |   | 16 | 42 | 64 MiB | 3.63 |
| 8 | 12 | 1.41 |   | 64 | 136 | 16 MiB | 1.12 |
| 16 | 20 | 1.21 |   | 128 | 263 | 8 MiB | 0.98 |
| 64 | 68 | **1.15** |   | 256 | 518 | 4 MiB | **0.91** |
|  |  |  |   | 512 | 1029 | 2 MiB | 1.00 |

Both curves run monotonically the *wrong* way for depth, and they do it at the
lowest lane count in the sweep — the case most favourable to the depth argument.
A at depth 5 is 2.6x slower than at depth 68. B at depth 16 is **75x** slower
than at depth 136. The same shape holds at 2048 and 8192 lanes.

So the premise is wrong for this hardware. 41 stages at L=8192 is not a latency
problem: a serial walk costs no parallelism (every thread walks its own slice
concurrently) while it removes memory traffic, threadgroup barriers and
instructions. Trading depth for width buys nothing here because nothing was
waiting on depth — the kernels are bandwidth- and occupancy-bound, and both A
and B spend bandwidth to buy latency they do not need. B is the worse of the two
by far, exactly as its 8x memory would predict.

### B built properly, not as a limit of the chunked kernel

The sweep above reaches B by driving the chunked kernel to `CHUNK=1`, which
leaves it paying for a reduce pass and checkpoint writes it would not need. So B
was also written as a real kernel: one leaf per timestep, Hillis-Steele doubling
over the whole sequence in device memory, no chunk and no serial walk anywhere,
depth `log2(L)` — both merged and two-scan. Forward only, since the backward is
the same shared `kla_scan_bwd` for every cell and would add the same constant to
all of them. Scan only, so these are not comparable to the whole-layer tables in
`docs/benchmarks/mps.md`.

`d_state=16`, L=2048, `d_inner=2048`, ms per forward:

| | B=1 | B=8 |
|---|---:|---:|
| `recurrent` | **2.13** | **13.31** |
| `chunk` | 5.11 | 38.01 |
| `merged_chunk` | 3.18 | 23.39 |
| `pscan` (C=16) | 10.11 | 80.15 |
| `merged_pscan` (C=16) | 9.94 | 78.87 |
| **non-chunked two-scan** (depth log L) | **150.42** | **1214.19** |
| **non-chunked merged** (depth log L) | **196.30** | **1584.97** |

Both are numerically correct — `max|Δy|` 3.9e-7 and 4.5e-7 against the
sequential reference, the same as every other cell. They are 47x and 62x slower
than `merged_chunk`, and the aggregate arrays are 24 GiB and 32 GiB at B=8
against 256 MiB for the chunked pscan.

Through the whole layer, same shape, one process — `d_model=1024`,
`d_state=16`, L=2048, ms per call. The non-chunked cells were given the shared
backward (their apply kernels write the same checkpoints at the same stride), so
the training column is measured rather than derived, and their gradients match
the sequential reference to 3e-7 like every other cell:

| cell | B=1 inference | B=8 training |
|---|---:|---:|
| `mps_recurrent` | **13.14** | **340.57** |
| `mps_chunk` | 16.18 | 361.41 |
| `mps_merged_chunk` | 14.09 | 345.61 |
| `mps_pscan` | 20.92 | 401.12 |
| `mps_merged_pscan` | 21.68 | 418.78 |
| non-chunked two-scan | 163.27 | 1552.35 |
| non-chunked merged | 211.86 | 1914.71 |

12x and 16x the whole-layer inference cost of `recurrent`, and 4.6x its training
cost — the layer's projections and the shared backward dilute it, and it is
still this bad. B=8 training does fit: the aggregates are freed when the forward
returns, so only the checkpoints are held.

Note the sign flips here: merged is *slower* than two-scan for the non-chunked
form, uniquely. Everywhere else merged wins or ties. Once the aggregate is
`[B,M,L,S]` rather than `[B,M,NCK,S]`, the 16-vs-12-floats-per-element cost is
being paid on every timestep through `log2(L)` doubling rounds, and it swamps
the round that merging saves. The same mechanism that makes `merged_pscan` a
wash at `CHUNK=16` makes merged actively worse at `CHUNK=1`.

**Neither A nor B should be built.** The depth question is closed on mps until
someone re-runs this on a machine where it might come out differently. The
experimental kernels are in the session scratchpad
(`nonchunked.metal`, `nonchunked.py`), not the repo — they are a measurement,
not a cell.

What the sweep *does* say: `KLA_CHUNK = 16` leaves 2–3x on the table for long
batch-1 prefill, where `CHUNK = 64..256` is the flat optimum. That is not free
to change — `KLA_CHUNK` is also the backward's checkpoint stride, and the replay
buffers are thread-private arrays of that length, so raising it trades register
pressure in the backward. A separate inference-only stride would collect it. New
work, and unrelated to `merged`.
