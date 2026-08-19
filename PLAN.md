# The merged 3x3 formulation

Fold the precision scan and the information-vector scan into **one** associative
scan, by writing the whole step as a single 3x3 linear map in homogeneous
coordinates. Add it as a third value on the fusion axis, called `merged`.

Status: designed and validated numerically, **not implemented**.

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

`D = 1 - A` after trace normalization, so the carried state is six values
(`A, B, C, qa, qb, s`) against today's `4 + 2`.

### Already validated

Prototype: `scratch/fused3x3.py`, `scratch/compare2v3.py` (not in the repo).

- Correct: float64 vs the sequential reference, L=200 — `λ` 6e-16, `η` 1e-12.
- fp32 stability: normalized `s` decays; unnormalized it overflows.
- No worse than today: fp32 against a float64 reference, decays 0.3/0.7/0.95,
  lengths 8/16/64/128. `λ` bit-identical (same 2x2 block), `η` equal or better
  at every point — e.g. decay 0.3 at L=128, `3.1e-4` today vs `4.6e-5` merged.

**Re-run these as a repo test before writing a kernel.** The whole design rests
on the normalization argument, and it is one line to get wrong.

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

Open: whether torch gets `torch_merged_chunk` / `torch_merged_pscan`. Torch has
no fused cells, so `merged` there would mean "unfused, but one scan" — which the
single axis cannot express. Either accept that torch's `merged` implies unfused,
or leave torch out.

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

### Phase 0 — pin the algebra

- [ ] `tests/test_merged_algebra.py`: the float64 identity, the fp32 comparison
      against the two-scan composition, and an explicit overflow test showing
      the unnormalized form failing. This is the load-bearing test.
- [ ] Decide the carried layout: 6 values with `D = 1 - A`, or 7 with `D` kept.
      Measure — the reconstruct costs an add, the extra word costs a register.

### Phase 1 — torch first, where it is cheapest to be wrong

- [ ] `_merged_combine` in `kla_ops.py`, next to `_mobius_combine_tracenorm`
- [ ] `torch_merged_chunk`, `torch_merged_pscan` (pending the naming question)
- [ ] Parity against `kla_scan_reference` at the existing tolerances

### Phase 2 — kernels

- [ ] `mps_merged_chunk` — do this one first. It is the only backend that can be
      run here, and the chunk kernel is where the structural win is largest.
- [ ] `mps_merged_pscan`
- [ ] `triton_merged_chunk`, `triton_merged_pscan`
- [ ] `cuda_merged_chunk`, `cuda_merged_pscan`
- [ ] Backward: **unchanged, and confirm it stays unchanged.** It replays from
      `[B,M,NCK,S]` checkpoints and never sees a composed map, so `merged` must
      not touch it. If it does, something has leaked.

### Phase 3 — measure

- [ ] `merged_chunk` vs `fused_chunk` at the shapes in
      `docs/benchmarks/mps.md`. Expect a clear win.
- [ ] `merged_chunk` vs `mps_recurrent` — the interesting one, see below.
- [ ] Revisit the aliases if `merged_chunk` crosses `recurrent`.
- [ ] Fold the result into `docs/benchmarks/mps.md` and
      `docs/implementations.md`.

## Open: the depth question, still parked

Separate from this work, and unaffected by it. The three fused `pscan` cells
walk each chunk *serially* before any doubling starts — `range(16)` on mps and
cuda, `range(64)` on triton — so depth is `C + log2(L/C) + C`, which is 41
stages at L=8192 rather than 13. Two ways out:

- **A** — replace the serial within-chunk walk with a Hillis-Steele across
  threads, as `triton_unfused_pscan` already does via `tl.associative_scan`.
  Depth `log2 L`, memory unchanged, still fused.
- **B** — no chunks at all, doubling in device memory over the whole sequence.
  Depth `log2 L`, but it must materialize `[B,L,M,S]` — roughly 8x the memory,
  and the cell stops being fused by this repo's definition.

`merged` helps either way, so it lands first.
