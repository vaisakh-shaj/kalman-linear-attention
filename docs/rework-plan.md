# KLA backend/implementation rework — *complete, kept for the record*

Superseded as the active plan by `PLAN.md` in the root, which covers the merged
3x3 formulation. This file records the naming rework and the coverage push, and
still holds the known-risk lists for the kernels that have never been run.

Rename the scan implementations onto a two-axis scheme, then fill the matrix so
every cell has a forward, an exact backward, state carry and prior decode.

**Priority is coverage.** Get every cell existing and correct. Tune each one
later, on the hardware that matters for it.

**No back-compat.** Old backend names, `scan_impl`, and deprecation shims are
deleted outright, not aliased. Nothing in the repo is released, so a clean break
costs nothing and the alternative is boilerplate that outlives its purpose.

---

## The scheme

Name: `<backend>[_unfused]_<implementation>`. Fused is the default and carries no
token. A bare `<backend>` is a fixed alias for `<backend>_chunk`.

| | recurrent | chunk *(default)* | pscan |
|---|---|---|---|
| **torch** | `torch_unfused_recurrent` | `torch_unfused_chunk` | `torch_unfused_pscan` |
| **triton** | `triton_recurrent` | `triton_chunk` | `triton_pscan` |
| **triton, unfused** | `triton_unfused_recurrent` | `triton_unfused_chunk` | `triton_unfused_pscan` |
| **cuda** | `cuda_recurrent` | `cuda_chunk` | `cuda_pscan` |
| **mps** | `mps_recurrent` | `mps_chunk` | `mps_pscan` |
| **cuda, prior work** | — | `cuda_v2_1`, `cuda_v2_2` | — |

See `docs/implementations.md` for what the three implementations mean.

`cuda_v2_1` and `cuda_v2_2` stay as they are. They are the only cells with an
approximate backward, which is exactly what makes them worth keeping: they are
the comparison for the exact-versus-approximate measurement.

## Contract

Every cell in the scheme must have:

- forward and backward
- exact backward — `exact_grads = INPUT_NAMES`, `exact_grad_tol <= 1e-2`
- state carry in and out, differentiable both ways
- `decode_from_prior`
- fp32 scan
- a declared `max_d_state`

A cell that misses one has a bug, not a variant. `cuda_v2_1` / `cuda_v2_2` are
the two exceptions, and they declare `exact_bwd=False`.

## Design rules

**Exactness comes from the backward, not the implementation.** Do not differentiate
the Möbius composition. Recover `λ_t` from a checkpoint replay, then run one
reverse scan carrying a *scalar* adjoint:

```
ν_t = ḡ_t + gain_{t+1}·ν_{t+1},    gain_t = (A·D − B·C)/den² = 1/(a²·den²)
```

This is cheaper than the 4×4 Jacobian chain in `cuda/v2_2`, not more expensive:
that kernel already pays the forward recompute, and adds ~64 MACs/step plus
`kNThreads × 16` floats of shared memory on top.

**`d_state <= 32` is the fast path.** 32 is one warp on CUDA and one SIMD-group
on Metal, so the read-out reduction is a register shuffle with no shared memory.
Design for it first; 33–64 takes the shared-memory fallback.

**Portability: consumer Ampere through H200, one kernel.** Budget ≤48 KB shared
memory per block and take no opt-in carveout — sizing for H200's 228 KB/SM
silently fails to launch on a 4090. Target `sm_80, 86, 89, 90`. Avoid sm_90-only
features: wgmma, TMA / `cp.async.bulk`, thread-block clusters, distributed
shared memory. The scan is fp32 scalar FMA work, so no tensor-core path applies
anyway.

**Precision.** The scan is fp32. No GEMM is fused into any scan kernel — every
`nn.Linear` sits outside the `autocast(enabled=False)` region in
`kla_layer.py`, so autocast already gives them fp16/bf16. Nothing to do; an fp16
scan later becomes a per-cell `scan_dtype`.

**`max_d_state` is declared, not worked around.** No state-axis tiling, no
auto-reroute. Above the cap, raise and name a cell that fits. Real configs use
`d_state <= 32`, which is 2–4× below every cap.

---

## Phase 0 — Registry, naming, config ✅

- [x] `Impl` record in `kla_ops.py`: `backend`, `implementation`, `fused`,
      `max_d_state`, `exact_bwd`, `fn`. Six fields, all load-bearing.
- [x] `_BACKENDS` maps cell name → `Impl`. `backend_names()` reads it.
- [x] `Backend` literal in `configs.py` rewritten to the scheme names.
- [x] Bare `<backend>` aliases resolve through `_ALIASES`.
- [x] `ScanImpl` and `KLAConfig.scan_impl` deleted.
- [x] `mobius_impl` kept — different axis (algebra), torch-only.
- [x] `resolve_impl` / `resolve_backend` pick the device's default cell.
- [x] `__main__.py` groups by `Impl.backend`, not `name.split("_", 1)`.
- [x] `--check-backends` rewritten: capability rows for available backends,
      names only for unavailable ones, uniform contract as a footer.
- [x] `PROFILES` re-keyed in `tests/test_backends.py`.
- [x] Other tests updated. 214 pass, ruff clean.

### Kernel source names

Renamed onto the scheme, since "tiled" and "fused" said nothing about which
cell a file served — and the shared backward living in a file called *fused*
was the same confusion:

| was | is | serves |
|---|---|---|
| `chunk_kla_scan.metal` (fwd) | `recurrent_kla_scan.metal` | `mps_recurrent` |
| `chunk_kla_scan.metal` (bwd) | `kla_scan_bwd.metal` | **both cells** |
| `tiled_kla_scan.metal` | `chunk_kla_scan.metal` | `mps_chunk` |

Kernel entry points followed: `kla_recurrent_fwd`, `kla_chunk_fwd`,
`kla_scan_bwd`. The `.py` wrappers and `_shaders.py` library builders match
(`recurrent_library`, `chunk_library`, `bwd_library`). The CUDA sources were
named to the same pattern from the start, and the triton ones were brought onto
it in Phase 2:

| was | is | serves |
|---|---|---|
| `fused_kla_scan.py` | `chunk_kla_scan.py` | `triton_chunk` |
| `fused_kla_bwd.py` | `kla_scan_bwd.py` | **all three fused cells** |
| `tiled_mobius_scan.py` | `unfused_kla_scan.py` | all three unfused cells |

Same defect in each case: "fused" and "tiled" named a *property* several files
share rather than the cell the file serves, and the shared backward living in a
file called *fused* was the worst of it.

### Decisions taken during Phase 0

**Aliases are the end-goal values.** Every bare backend name resolves to
`<backend>_chunk` (`torch` to `torch_unfused_chunk`). No transitional pointers.
A backend whose chunk cell does not exist yet simply raises — the gap is the
work, not a case to engineer around.

**`mps_unfused_recurrent` dropped**, along with `lane_mobius_scan.{py,metal}`
and `scan_library`. MPS has no uncapped cell now; above `d_state = 128` the
refusal names `torch`.

**Triton's grad-mode dispatch removed.** `backend="triton"` used to pick the
fused kernel under no-grad and the composed one otherwise — the "different
kernels under one name" defect this rework exists to remove. `kernel` is now
`chunk` or `unfused_chunk`, each running what it names.

**`scan_impl` survives inside `kla_scan_torch`.** Gone from `KLAConfig` and from
`kla_scan`; the three torch cells are built by binding it. An implementation
detail, not a user-facing knob.

## Phase 1 — Shared machinery

Build once per backend, instantiate three ways. Phase 2 rows depend on it.

- [x] **MPS**: scalar-adjoint reverse scan + checkpoint replay — already in
      `kla_fused_bwd`, and now shared. `mps_chunk`'s forward writes checkpoints
      at the same stride and hands them straight to `scan_backward`, so the
      chunk cell has an exact backward without a second adjoint kernel. This is
      the pattern for triton and CUDA: **one backward per backend, not one per
      implementation.**
- [x] **triton**: `kernels/triton/kla_scan_bwd.py` — chunk-shaped replay from
      `[B,M,NCK,S]` checkpoints, two reverse affine recurrences via
      `tl.flip` + `tl.associative_scan`. **Written blind; not yet run.**
- [x] **cuda**: `kernels/cuda/scan/kla_scan_bwd.cuh` — lane-per-state replay
      from `[B,M,NCK,S]` checkpoints into fixed-size register buffers, two
      reverse recurrences carrying one scalar each. **Written blind; not yet
      compiled.**
- [x] Reduce-then-scan skeleton for `pscan`: chunk reduce → scan the
      `[B,M,NCK,S]` aggregates → apply and read out. Landed on MPS first, where
      it can be run: `pscan_kla_scan.metal`. The scan over aggregates is
      Hillis-Steele *across launches*, ping-ponging two buffers, so no
      threadgroup ceiling bounds the sequence length. Two rounds are needed,
      not one — the affine leaf `alpha_t` reads `lambda_{t-1}`, so it does not
      exist until the Möbius scan has produced lambda.
- [x] `d_state <= 32` warp / SIMD-group read-out, shared-memory fallback above.

## Phase 2 — Coverage

Each backend's `recurrent` cell lands first — cheapest exact backward, and it
validates the shared machinery before the harder implementations use it.

**torch** — unblocks the `chunk` default

- [x] `torch_unfused_recurrent` — rename
- [x] `torch_unfused_pscan` — rename
- [x] `torch_unfused_chunk` — new `chunk_scan` in `scan.py`; `torch` now
      aliases it, so the `chunk` default holds on torch

**mps** — most complete today, so it proves the machinery

- [x] `mps_recurrent` — rename, declare cap
- [x] `mps_chunk` — backward added; gradients ~5e-7 vs the reference. `mps` now
      aliases it, so the `chunk` default holds on MPS
- [x] `mps_pscan` — new; five-kernel reduce-then-scan, checkpoints fall out of
      the implementation so the shared backward runs behind it unchanged

**triton**

- [x] `triton_unfused_chunk` — rename
- [x] `triton_recurrent` — new forward; reuses `kla_scan_bwd`. **Untested.**
- [x] `triton_chunk` — backward + prior decode added. **Untested on hardware.**
- [x] `triton_pscan` — new; five kernels, reuses `kla_scan_bwd`. **Untested.**
- [x] `triton_unfused_recurrent` — new implementation in `unfused_kla_scan.py`,
      which `tiled_mobius_scan.py` was renamed to. **Untested.**
- [x] `triton_unfused_pscan` — new; reuses the fused cell's doubling rounds
      rather than repeating them. **Untested.**

**cuda** — `sm_80–90`, ≤48 KB smem, `d_state <= 32` fast path. Sources under
`kernels/cuda/scan/`; `v2_*` untouched beside them.

- [x] `cuda_recurrent` — new, exact backward. **Untested; needs nvcc.**
- [x] `cuda_chunk` — new forward; reuses the backward above. **Untested.**
- [x] `cuda_pscan` — new; the Metal pscan transcribed. **Untested.**
- [ ] `cuda_v2_1` / `cuda_v2_2` — leave untouched

## Where the matrix stands

All fifteen cells exist, plus the two `cuda_v2_*` kept for comparison. Every
cell has a forward, the exact backward its backend shares, state carry and
prior decode.

What has actually *run* is torch and mps — six cells, on this machine, against
the sequential reference in both directions. The triton and cuda cells are
written and not yet executed; see below for what that leaves unverified and in
what order to suspect it.

Two structural rules held everywhere, and are worth keeping when tuning:

**One backward per backend, not one per implementation.** Each backend has exactly
one adjoint, and adding a implementation needed no change to it. The adjoint reads
the state at the checkpoints (fused) or the values λ and η (unfused) — neither
depends on the order a forward produced them in.

**`pscan` computes its own checkpoints.** "The state entering chunk c" is
exactly what the scan over aggregates produces, so the checkpoints are free
there rather than an extra store, and the shared backward runs behind it
unchanged.

## Validation of blind work

`tests/test_adjoint_formulas.py` pins the blind work against autograd in
float64, on CPU, in two layers:

1. **The formulas** — the scalar gain, both source terms, the `da`/`dp`
   factorings, the boundary grads. Matches to ~4e-16 with and without prior.
2. **The chunked replay** — checkpoint before the step, constant-trip-count
   replay, masked tail, reverse chunk walk. This is the structure `cuda` and
   `mps` both implement literally, and it is checked at every alignment:
   `L = 1, 15, 16, 17, 32, 40` against a stride of 16.

So the mathematics *and* the structure are verified even where the kernels
cannot run. What remains unverified is per-backend mechanics: indexing, thread
masking, `tl.flip`, atomics, shared-memory reductions, the launch geometry. Run
this file first when a kernel's gradients look wrong — it separates a mistake in
the maths from a mistake in the indexing.

Known risks in the untested triton kernels, in likely order:

- `tl.associative_scan` inside `_pscan_*_reduce_kernel` and
  `_pscan_*_apply_kernel`, where the aggregate is read as the last row of the
  inclusive scan (`t == BLOCK_L - 1`). That is correct only because masked-off
  rows load the identity — check that the `other=` values are right first.
- Scalar `tl.where(live, vector, scalar)` in `pscan_kla_scan.py`: a scalar
  condition broadcast over a `[BLOCK_S]` tile.
- The clamped-address-plus-mask stores (`tc` rather than `t`) in the pscan
  apply kernel.
- The ping-pong in `_doubling` returning the wrong buffer when `n_ck == 1`
  (zero rounds, so the reduce kernel's output is the answer).

Known risks in the untested triton backward, in likely order:

- `tl.flip(x, 0)` on a 2-D tile — API shape and axis semantics.
- `tl.associative_scan` inside the `_reverse_affine` helper rather than inline.
- The partial final chunk: leading flipped positions must pass the zero carry
  through untouched.
- `tl.atomic_add` on the `[B,L,S]` and `[M,S]` outputs under masking.

Known risks in the untested CUDA kernels, in likely order:

- The `float4` / `float2` views of the aggregate tensors in `pscan_fwd`
  (`reinterpret_cast` over a `{B,M,NCK,S,4}` float tensor) — alignment holds
  because torch allocations are 512-byte aligned, but nothing checks it.
- `std::swap` on the ping-pong pointers, and which buffer holds the inclusive
  prefix once the loop ends.

- The `KLA_LAUNCH` / `KLA_DISPATCH_BLOCK_S` macro pair and the `LAUNCH_BODY`
  it expands around — it is the least type-checked thing in the tree.
- `__shfl_xor_sync(..., width=BLOCK_S)` segmenting the warp for `BLOCK_S < 32`.
- The `BLOCK_S > 32` shared-memory reduction: every thread must reach every
  `__syncthreads()`, which holds only because `n` is uniform across the block.
- `#pragma unroll` actually keeping `lam_h`/`eta_h` in registers rather than
  spilling to local memory. Check with `-Xptxas -v`.

## Phase 3 — Docs and measurement

- [x] `docs/implementations.md` — the scheme and the matrix
- [x] Rewrite `docs/backends.md` against the new names
- [x] Update `docs/usage.md` and `readme.md`
- [x] Fold completed items out of `docs/todo.md`
- [ ] Measure `cuda_chunk` vs `cuda_v2_2`: accuracy **and** wall-clock
- [x] Confirm `chunk` is the right default per device, or change the alias.
      It was not: `torch` and `mps` both moved to `recurrent`, which won every
      shape either is realistically used at. `chunk` turned out to be *never*
      optimal for torch — beaten by `pscan` in training and `recurrent` in
      inference, on CPU and on Metal alike. triton and cuda keep `chunk` until
      someone runs the same sweep on a CUDA device — `docs/benchmarks/mps.md`
      states the method, and nothing in it is Metal-specific except the
      synchronize call.
