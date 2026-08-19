# Todo

## MPS backend

- [x] Tiled parallel-scan forward (`mps_tiled`) — wins below ~6k lanes, i.e. batch-1 prefill on a narrow model
- [ ] Backward for `mps_tiled`, if the tiled forward turns out to matter in practice

## CUDA backend

- [ ] Exact gradients — replace the 4x4 Jacobian adjoint with a scalar reverse scan over lambda, as triton does
- [ ] Carry the filter state — accept an initial state, return the final one, differentiate both ways

## Experiments

- [ ] `mad/` submodule — MAD synthetics
- [ ] `nanochat/` submodule — FineWeb-Edu pretraining
- [ ] `experiments/` + `main.py` to run both
