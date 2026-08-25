# Implementations

The scan can be implemented in several ways. Each one is named:

```
<backend>_<fusion>_<implementation>
```

## Fusion

How much of the scan happens inside a single kernel.

- **`fused`**: the whole scan in one kernel.
- **`unfused`**: torch code with standalone scans between it. Slower, but simple
  to read, and the reference the others are checked against.
- **`merged`**: fused, and doing in one pass what the others take two to do.

## Implementation

How the scan gets through the sequence.

- **`recurrent`**: one timestep at a time, parallel over batch, channel and
  state instead of time.
- **`chunk`**: the sequence split into chunks, parallel within a chunk and
  serial across them.

## Support matrix

`torch` is always unfused, and offers `unfused_recurrent`\*, `unfused_chunk` and
`merged_chunk`. It runs anywhere, has no `d_state` limit, and is the only
backend with float64.

The GPU backends:

| | `fused_recurrent` | `fused_chunk` | `merged_chunk` | max `d_state` |
|---|---|---|---|---|
| **triton** | yes | yes | yes\* | none |
| **cuda** | yes | yes\* | yes | 64 |
| **mps** | yes\* | yes | yes | 128 |

\* is the backend default, selected by a bare backend name (`backend="mps"`).
Every default was picked by measurement; see [benchmarks/](benchmarks/).

`backend="auto"` is the overall default and picks by device: triton on NVIDIA,
mps on Apple silicon, torch otherwise. It never picks `cuda`, which needs a
compiler that `auto` cannot assume is present.

Run `python -m kla --check-backends` for what a given machine supports.
