"""``python -m kla`` - what this machine can run, and whether it runs correctly.

    python -m kla                          # version, device, and the "auto" pick
    python -m kla --check-backends         # every backend, and which are usable
    python -m kla --test-backends all      # run every one against the reference
    python -m kla --test-backends triton   # ...or just the ones you name

``--check-backends`` is a cheap capability probe: it looks for the device, the
package and ``nvcc``, and imports the backend, but compiles nothing - so ``[x]``
on a kernel backend does not mean "the kernel builds". ``[X]`` marks the one
``auto`` resolves to here. ``--test-backends`` is the authoritative answer - it
runs the forward and the backward of every implementation named and reports each
against :func:`kla.ops.kla_scan_reference`.

Naming backends asserts they work: ``--test-backends triton`` exits non-zero if
triton is unusable here, so it is a CI smoke test. ``all`` is a survey and
always exits 0 - on a CPU-only box the GPU ones are expected to fail.

``--test-backends`` takes either a family (``mps``, which runs every Metal
implementation) or one exact backend (``mps_fused``), so a single implementation
can be pinned and checked on its own.
"""

from __future__ import annotations

import tyro

# Prose keyed by backend family (see _families). A family added to the
# dispatcher still lists; it just gets no description.
_DESCRIPTIONS = {
    "torch": "the portable reference implementation",
    "triton": "fused triton kernels",
    "cuda": "JIT-compiled CUDA kernel; forward/training only",
    "mps": "Metal kernels; no toolchain needed",
}

# Module that must import for a backend family to be usable at all.
_MODULES = {
    "triton": "kla.ops.triton_backend",
    "cuda": "kla.ops.cuda_backend",
    "mps": "kla.ops.mps_backend",
}


def _families(names: tuple[str, ...]) -> dict[str, list[str]]:
    """Backend families, mapped to the implementations behind each.

    A family is everything sharing a ``<family>_<impl>`` prefix, because what a
    reader wants from ``--check-backends`` is "can this machine run Metal / CUDA
    kernels", which is one answer per device. ``--test-backends`` still accepts
    the exact names (see :func:`main`).
    """
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault(name.split("_", 1)[0], []).append(name)
    return grouped


# Accuracy contract, condensed from the parity profiles in tests/test_backends.py
# (the tests are not importable from the installed package, so the numbers are
# restated here; keep them in step).
_ATOL = _RTOL = 5e-4
_GRAD_TOL = 1e-2
# The CUDA backward is a knowingly approximate adjoint on the precision-scan
# path, so d(lambda_v), d(k), d(a) and d(p) are held to a documented looser
# budget - see the parity notes in kla.ops.cuda_backend. Tightening these would
# fail a *correct* build.
# Keyed by the group prefix, so every cuda kernel version gets the same budget.
_LOOSE_GRADS = {"cuda": ("lambda_v", "k", "a", "p")}
_LOOSE_GRAD_TOL = 0.25

_INPUT_NAMES = ("v", "lambda_v", "k", "q", "a", "p")


def _nvcc() -> str | None:
    """The compiler the CUDA backend would build with, if there is one."""
    import os
    import shutil

    home = os.environ.get("KLA_CUDA_HOME") or os.environ.get("CUDA_HOME")
    if home:
        nvcc = os.path.join(home, "bin", "nvcc")
        return nvcc if os.path.exists(nvcc) else None
    return shutil.which("nvcc")


def _import_error(backend: str) -> str:
    """One line on why ``backend``'s module will not import, or ``""``.

    Only reached once a backend's requirements are met, so importing it here
    costs nothing on a machine that could not run it anyway. Still compiles
    nothing - both GPU backends build their kernels on first call.
    """
    import importlib

    module = _MODULES[backend]
    try:
        importlib.import_module(module)
    except Exception as exc:
        return _oneline(exc)
    return ""


def _requirements() -> dict[str, list[tuple[bool, str]]]:
    """What each family needs before it can run: ``(holds, what is missing)``.

    One row per family rather than a branch per device, so a new backend is a
    new row. Every check here is cheap - a device query, a spec lookup, a
    ``which`` - and nothing compiles.
    """
    from importlib.util import find_spec

    import torch

    cuda = (torch.cuda.is_available(), "needs a CUDA device")
    return {
        "torch": [],
        "triton": [
            cuda,
            (
                find_spec("triton") is not None,
                "needs the triton package: pip install 'kla[triton]'",
            ),
        ],
        "cuda": [
            cuda,
            (
                _nvcc() is not None,
                "needs nvcc to build; not on PATH or under CUDA_HOME",
            ),
        ],
        "mps": [(torch.backends.mps.is_available(), "needs an Apple-silicon GPU")],
    }


def _probe(family: str, requirements: list[tuple[bool, str]]) -> tuple[bool, str]:
    """Can ``family`` plausibly run here? Cheap; no compute, no compilation.

    Returns ``(usable, why)``. ``why`` always says what the backend *is*, so an
    unusable one still describes itself, and appends what is missing - including
    the case where the requirements are met but the backend itself is broken.
    """
    described = _DESCRIPTIONS.get(family, "")

    def no(reason: str) -> tuple[bool, str]:
        return False, f"{described} - {reason}" if described else reason

    for holds, missing in requirements:
        if not holds:
            return no(missing)
    if family not in _MODULES:  # torch: nothing to import, nothing to build
        return True, described
    if broken := _import_error(family):
        return no(f"installed, but fails to import - {broken}")
    return True, f"{described} (compiled on first use)"


def _inputs(device: str, requires_grad: bool = False):
    """The probe batch every backend is run on.

    Well-conditioned the same way ``tests/test_backends.py`` conditions its
    inputs - lambda_v and p strictly positive, a inside (0, 1) - so a failure
    here means the backend, not a pathological batch.
    """
    import torch

    B, L, M, S = 1, 32, 8, 8
    gen = torch.Generator().manual_seed(0)

    def to(t):
        return t.to(device).requires_grad_(requires_grad)

    return (
        to(torch.randn(B, L, M, generator=gen)),  # v
        to(torch.rand(B, L, M, generator=gen) + 0.5),  # lambda_v (Λ^v > 0)
        to(torch.randn(B, L, S, generator=gen) * 0.3),  # k
        to(torch.randn(B, L, S, generator=gen) * 0.3),  # q
        to(torch.rand(M, S, generator=gen) * 0.5 + 0.4),  # a
        to(torch.rand(M, S, generator=gen) * 0.05 + 0.01),  # p
    )


def _rel_err(got, ref) -> float:
    """Max deviation normalized by the reference's own scale.

    Gradients of the static a/p accumulate over batch x time, so their absolute
    magnitude says nothing on its own; normalizing keeps one budget meaningful
    across all six inputs.
    """
    return ((got - ref).abs().max() / (ref.abs().max() + 1.0)).item()


def _oneline(exc: Exception) -> str:
    """First line of an exception - JIT build logs run to hundreds."""
    line = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {line[0]}" if line else type(exc).__name__


def _forward(backend: str, device: str) -> tuple[str, str]:
    """Forward parity against the reference. Returns ``(status, detail)``."""
    import torch

    from kla.ops import kla_scan, kla_scan_reference

    args = _inputs(device)
    try:
        y, y_var, _ = kla_scan(*args, backend=backend)
    except NotImplementedError as exc:
        return "skipped", _oneline(exc)
    except Exception as exc:
        return "FAILED", _oneline(exc)

    if not (torch.isfinite(y).all() and torch.isfinite(y_var).all()):
        return "FAILED", "produced non-finite output"

    y_ref, y_var_ref, _ = kla_scan_reference(*args)
    dy = (y - y_ref).abs().max().item()
    dvar = (y_var - y_var_ref).abs().max().item()
    detail = f"max|dy| {dy:.1e}   max|dvar| {dvar:.1e}   (atol {_ATOL:g})"
    matches = torch.allclose(y, y_ref, atol=_ATOL, rtol=_RTOL) and torch.allclose(
        y_var, y_var_ref, atol=_ATOL, rtol=_RTOL
    )
    return ("ok" if matches else "FAILED"), detail


def _gradients(backend: str, device: str) -> tuple[str, str]:
    """Per-input gradient parity against the reference. ``(status, detail)``."""
    import torch

    from kla.ops import kla_scan, kla_scan_reference

    args = _inputs(device, requires_grad=True)
    refs = tuple(t.detach().clone().requires_grad_(True) for t in args)
    try:
        y, y_var, _ = kla_scan(*args, backend=backend)
        # Squaring y keeps the read-out path in the loss; summing y_var keeps the
        # precision path in it. Both are needed to give every input a gradient.
        (y.square().sum() + y_var.sum()).backward()
    except NotImplementedError as exc:
        return "skipped", _oneline(exc)
    except Exception as exc:
        return "FAILED", _oneline(exc)

    y_ref, y_var_ref, _ = kla_scan_reference(*refs)
    (y_ref.square().sum() + y_var_ref.sum()).backward()

    loose = next((v for k, v in _LOOSE_GRADS.items() if backend.startswith(k)), ())
    worst_name, worst_ratio, worst_err = "", 0.0, 0.0
    for name, got, ref in zip(_INPUT_NAMES, args, refs):
        if got.grad is None:
            return "FAILED", f"no gradient reached d{name}"
        if not torch.isfinite(got.grad).all():
            return "FAILED", f"non-finite d{name}"
        err = _rel_err(got.grad, ref.grad)
        budget = _LOOSE_GRAD_TOL if name in loose else _GRAD_TOL
        if err / budget > worst_ratio:
            worst_name, worst_ratio, worst_err = name, err / budget, err

    budget = _LOOSE_GRAD_TOL if worst_name in loose else _GRAD_TOL
    detail = f"worst d{worst_name} {worst_err:.1e}   (budget {budget:g})"
    if loose:
        detail += f"   [d{', d'.join(loose)} approximate by design]"
    return ("ok" if worst_ratio < 1.0 else "FAILED"), detail


_CHECKS = (("forward", _forward), ("gradients", _gradients))


def _test(backend: str, device: str, indent: str) -> bool:
    """Run every check for one implementation, printing a row each.

    Returns whether all of them passed. Set ``KLA_JIT_VERBOSE=1`` for the CUDA
    backend's full build log.
    """
    ok, last = True, None
    for label, run in _CHECKS:
        status, detail = run(backend, device)
        ok &= status == "ok"
        # A skip reason is the dispatcher's, so it repeats across every check.
        if (status, detail) == last:
            detail = "(same reason)"
        else:
            last = (status, detail)
        print(f"{indent}{label:<9}  {status:<7}  {detail}")
    return ok


def main(
    check_backends: bool = False,
    test_backends: tuple[str, ...] | None = None,
) -> int:
    """Report and test the KLA scan backends on this machine.

    Args:
        check_backends: Show every backend and whether it is usable here,
            without running it.
        test_backends: Run the named backends against the reference. Takes a
            family ('mps'), an exact backend ('mps_fused'), or 'all'.
    """
    import torch

    import kla
    from kla.ops import default_device

    families = _families(kla.backend_names())
    width = max(len(name) for name in families)

    if check_backends:
        requirements = _requirements()
        auto = kla.resolve_backend()
        for name in families:
            ok, why = _probe(name, requirements[name])
            mark = "X" if name == auto else "x" if ok else " "
            print(f"  [{mark}] {name:<{width}}  {why}")
        print("\n  [X] = what backend='auto' resolves to here")
        return 0

    if test_backends is not None:
        # A family runs every implementation behind it; an exact name runs just
        # that one, so a single kernel can be pinned and asserted. Family names
        # last: a family is also a dispatcher name, and naming it means the set.
        targets = {name: [name] for name in kla.backend_names()} | dict(families)
        if not test_backends:
            raise SystemExit(
                f"--test-backends needs 'all', or one of {', '.join(targets)}"
            )
        survey = "all" in test_backends
        selected = list(families) if survey else list(test_backends)
        unknown = [name for name in selected if name not in targets]
        if unknown:
            raise SystemExit(
                f"unknown backend(s) {', '.join(unknown)}; "
                f"expected 'all' or {', '.join(targets)}"
            )
        device = default_device()
        failed = 0
        for name in selected:
            print(f"\n{name}")
            impls = targets[name]
            # Name each implementation only when the entry holds more than one;
            # otherwise the header above already said which kernel is running.
            nested = len(impls) > 1
            ok = True
            for impl in impls:
                if nested:
                    print(f"  {impl}")
                ok &= _test(impl, device, "    " if nested else "  ")
            failed += not ok
        return 1 if failed and not survey else 0

    device = kla.ops.default_device()
    if device == "cuda":
        device = torch.cuda.get_device_name(0)
    print(f"kla {kla.__version__}   torch {torch.__version__}   device: {device}")
    print(f"backend='auto' resolves to: {kla.resolve_backend()}")
    print("\n  --check-backends to see them all, --test-backends to run them")
    return 0


if __name__ == "__main__":
    raise SystemExit(tyro.cli(main, prog="python -m kla"))
