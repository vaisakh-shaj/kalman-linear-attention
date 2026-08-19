# Todo

Backend and implementation coverage is done — all fifteen cells exist, each with an
exact backward, state carry and prior decode. See `docs/rework-plan.md` for what has been
run on real hardware and what is still only written, and
`docs/implementations.md` for the matrix.

## Measurement

- [x] Apple silicon — `docs/benchmarks/mps.md`. Moved the `torch` and `mps`
      aliases to `recurrent`; `chunk` turned out to be never optimal for torch.
- [ ] The same sweep on a CUDA device, for the `triton` and `cuda` aliases
- [ ] `cuda_chunk` vs `cuda_v2_2`: accuracy **and** wall-clock
- [ ] Tune `KLA_CHUNK` / `KLA_ITEMS` / `BLOCK_L` per backend once the above runs

## Experiments

- [ ] `mad/` submodule — MAD synthetics
- [ ] `nanochat/` submodule — FineWeb-Edu pretraining
- [ ] `experiments/` + `main.py` to run both
