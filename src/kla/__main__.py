"""``python -m kla`` - print what this machine can run, then prove it works.

A first-run sanity check. It reports the resolved backend rather than making you
guess why ``backend="auto"`` picked what it picked, and runs one tiny forward so
a broken install fails here instead of in your training loop.
"""

from __future__ import annotations


def main() -> int:
    import torch

    import kla

    info = kla.backend_info()
    print(f"kla {kla.__version__}   torch {torch.__version__}   device: {info['device']}")
    print(f"backend='auto' resolves to: {info['resolved_auto']}\n")
    for name, (ok, why) in info["backends"].items():
        print(f"  [{'x' if ok else ' '}] {name:<7} {why}")
    print(f"\n  log-space scan on: {', '.join(info['log_space_on'])}")

    layer = kla.KLALayer(d_model=64, config=kla.KLAConfig(d_state=8))
    y = layer(torch.randn(2, 16, 64))
    assert y.shape == (2, 16, 64) and torch.isfinite(y).all()
    n = sum(p.numel() for p in layer.parameters())
    print(f"\n  forward check: KLALayer(64) -> {tuple(y.shape)}, {n:,} params   OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
