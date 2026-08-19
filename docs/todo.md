# Todo

Backend and schedule coverage is done — all fifteen cells exist, each with an
exact backward, state carry and prior decode. See `PLAN.md` for what has been
run on real hardware and what is still only written, and
`docs/implementations.md` for the matrix.

## Measurement

- [ ] `cuda_chunk` vs `cuda_v2_2`: accuracy **and** wall-clock, on a CUDA device
- [ ] Confirm `chunk` is the right default per device, or change the alias
- [ ] Tune `KLA_CHUNK` / `KLA_ITEMS` / `BLOCK_L` per backend once the above runs

## Experiments

- [ ] `mad/` submodule — MAD synthetics
- [ ] `nanochat/` submodule — FineWeb-Edu pretraining
- [ ] `experiments/` + `main.py` to run both
