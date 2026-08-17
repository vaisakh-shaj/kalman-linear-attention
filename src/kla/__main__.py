"""``python -m kla`` - what this machine can run, and whether it runs correctly.

    python -m kla                          # version, device, and the "auto" pick
    python -m kla --check-backends         # every backend, and which are usable
    python -m kla --test-backends all      # run every one against the reference
    python -m kla --test-backends triton   # ...or just the ones you name

``--check-backends`` is a cheap capability probe: it looks for the device, the
package and ``nvcc``, and imports the backend, but compiles nothing - so ``[x]``
on a CUDA backend does not mean "the kernel builds". ``[X]`` marks the one
``auto`` resolves to here. ``--test-backends`` is the authoritative
answer - per backend it runs the forward and the backward and reports each
against :func:`kla.ops.kla_scan_reference`.

Naming backends asserts they work: ``--test-backends triton`` exits non-zero if
triton is unusable here, so it is a CI smoke test. ``all`` is a survey and
always exits 0 - on a CPU-only box the GPU ones are expected to fail.
"""

from __future__ import annotations

import tyro

# Prose keyed by CLI-facing backend name (see _groups). A backend added to the
# dispatcher still lists; it just gets no description.
_DESCRIPTIONS = {
    "torch": "the portable reference implementation",
    "triton": "fused triton kernels",
    "cuda": "JIT-compiled CUDA kernel; forward/training only",
}


def _groups(names: tuple[str, ...]) -> dict[str, list[str]]:
    """CLI-facing backends, mapped to the dispatcher names behind each.

    There is one ``cuda`` backend. The dispatcher carries pinned kernel variants
    under ``cuda*`` names for reproducing old runs; they compile from the same
    toolchain and are folded into the single ``cuda`` entry here.
    """
    grouped: dict[str, list[str]] = {}
    for name in names:
        grouped.setdefault("cuda" if name.startswith("cuda") else name, []).append(name)
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

    module = "kla.ops.triton_backend" if backend == "triton" else "kla.ops.cuda_backend"
    try:
        importlib.import_module(module)
    except Exception as exc:
        return _oneline(exc)
    return ""


def _probe(backend: str, cuda: bool, triton: bool) -> tuple[bool, str]:
    """Can ``backend`` plausibly run here? Cheap; no compute, no compilation.

    Returns ``(usable, why)``. ``why`` always says what the backend *is*, so an
    unusable one still describes itself, and appends what is missing - including
    the case where the requirements are met but the backend itself is broken.
    """
    described = _DESCRIPTIONS.get(backend, "")

    def no(reason: str) -> tuple[bool, str]:
        return False, f"{described} - {reason}" if described else reason

    if backend == "torch":
        return True, described
    if not cuda:
        return no("needs a CUDA device")
    if backend == "triton" and not triton:
        return no("needs the triton package: pip install 'kla[triton]'")
    if broken := _import_error(backend):
        return no(f"installed, but fails to import - {broken}")
    if backend.startswith("cuda"):
        if _nvcc() is None:
            return no("needs nvcc to build; not on PATH or under CUDA_HOME")
        return True, f"{described} (compiled on first use)"
    return True, described


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


def _forward(backend: str, device: str) -> tuple[str, str, float]:
    """Forward parity against the reference. Returns (status, detail, severity).

    ``severity`` is the deviation as a fraction of its budget, so results from
    different kernels are comparable when a grouped entry picks one to report.
    """
    import torch

    from kla.ops import kla_scan, kla_scan_reference

    args = _inputs(device)
    try:
        y, y_var, _ = kla_scan(*args, backend=backend)
    except NotImplementedError as exc:
        return "skipped", _oneline(exc), 0.0
    except Exception as exc:
        return "FAILED", _oneline(exc), float("inf")

    if not (torch.isfinite(y).all() and torch.isfinite(y_var).all()):
        return "FAILED", "produced non-finite output", float("inf")

    y_ref, y_var_ref, _ = kla_scan_reference(*args)
    dy = (y - y_ref).abs().max().item()
    dvar = (y_var - y_var_ref).abs().max().item()
    detail = f"max|dy| {dy:.1e}   max|dvar| {dvar:.1e}   (atol {_ATOL:g})"
    matches = torch.allclose(y, y_ref, atol=_ATOL, rtol=_RTOL) and torch.allclose(
        y_var, y_var_ref, atol=_ATOL, rtol=_RTOL
    )
    return ("ok" if matches else "FAILED"), detail, max(dy, dvar) / _ATOL


def _gradients(backend: str, device: str) -> tuple[str, str, float]:
    """Per-input gradient parity against the reference.

    Returns ``(status, detail, severity)``; see :func:`_forward`.
    """
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
        return "skipped", _oneline(exc), 0.0
    except Exception as exc:
        return "FAILED", _oneline(exc), float("inf")

    y_ref, y_var_ref, _ = kla_scan_reference(*refs)
    (y_ref.square().sum() + y_var_ref.sum()).backward()

    loose = next((v for k, v in _LOOSE_GRADS.items() if backend.startswith(k)), ())
    worst_name, worst_ratio, worst_err = "", 0.0, 0.0
    for name, got, ref in zip(_INPUT_NAMES, args, refs):
        if got.grad is None:
            return "FAILED", f"no gradient reached d{name}", float("inf")
        if not torch.isfinite(got.grad).all():
            return "FAILED", f"non-finite d{name}", float("inf")
        err = _rel_err(got.grad, ref.grad)
        budget = _LOOSE_GRAD_TOL if name in loose else _GRAD_TOL
        if err / budget > worst_ratio:
            worst_name, worst_ratio, worst_err = name, err / budget, err

    budget = _LOOSE_GRAD_TOL if worst_name in loose else _GRAD_TOL
    detail = f"worst d{worst_name} {worst_err:.1e}   (budget {budget:g})"
    if loose:
        detail += f"   [d{', d'.join(loose)} approximate by design]"
    return ("ok" if worst_ratio < 1.0 else "FAILED"), detail, worst_ratio


# Worst-first, so a group reports its most serious kernel result.
_RANK = {"FAILED": 0, "skipped": 1, "ok": 2}


def _test(group: str, members: list[str]) -> tuple[bool, list[tuple[str, str, str]]]:
    """Run forward and gradient checks for one CLI entry.

    Returns ``(ok, rows)`` with ``rows`` as ``(label, status, detail)`` - one row
    per check, not per kernel, so a grouped entry stays one report. It passes
    only if every kernel behind it passes, and the detail names the kernel
    responsible whenever the group holds more than one. Set
    ``KLA_JIT_VERBOSE=1`` for the CUDA backend's full build log.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows, ok = [], True
    for label, run in (("forward", _forward), ("gradients", _gradients)):
        results = [(name, *run(name, device)) for name in members]
        name, status, detail, _ = min(results, key=lambda r: (_RANK[r[1]], -r[3]))
        ok &= all(r[1] == "ok" for r in results)
        if len(members) > 1 and status != "skipped":
            detail = f"{name}: {detail}"
        # The skip reason is the dispatcher's, identical for every check here.
        if rows and (status, detail) == (rows[-1][1], rows[-1][2]):
            detail = "(same reason)"
        rows.append((label, status, detail))
    return ok, rows


def main(
    check_backends: bool = False,
    test_backends: tuple[str, ...] | None = None,
) -> int:
    """Report and test the KLA scan backends on this machine.

    Args:
        check_backends: Show every backend and whether it is usable here,
            without running it.
        test_backends: Run the named backends against the reference. Pass 'all'
            for every backend.
    """
    from importlib.util import find_spec

    import torch

    import kla

    groups = _groups(kla.backend_names())
    width = max(len(name) for name in groups)

    if check_backends:
        cuda = torch.cuda.is_available()
        triton = find_spec("triton") is not None
        resolved = kla.resolve_backend()
        auto = "cuda" if resolved.startswith("cuda") else resolved
        for name in groups:
            ok, why = _probe(name, cuda, triton)
            mark = "X" if name == auto else "x" if ok else " "
            print(f"  [{mark}] {name:<{width}}  {why}")
        print("\n  [X] = what backend='auto' resolves to here")
        return 0

    if test_backends is not None:
        if not test_backends:
            raise SystemExit(
                f"--test-backends needs 'all', or one of {', '.join(groups)}"
            )
        survey = "all" in test_backends
        selected = list(groups) if survey else list(test_backends)
        unknown = [name for name in selected if name not in groups]
        if unknown:
            raise SystemExit(
                f"unknown backend(s) {', '.join(unknown)}; "
                f"expected 'all' or {', '.join(groups)}"
            )
        failed = 0
        for name in selected:
            ok, rows = _test(name, groups[name])
            failed += not ok
            label_width = max(len(row[0]) for row in rows)
            print(f"\n{name}")
            for label, status, detail in rows:
                print(f"  {label:<{label_width}}  {status:<7}  {detail}")
        return 1 if failed and not survey else 0

    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"kla {kla.__version__}   torch {torch.__version__}   device: {device}")
    print(f"backend='auto' resolves to: {kla.resolve_backend()}")
    print("\n  --check-backends to see them all, --test-backends to run them")
    return 0


if __name__ == "__main__":
    raise SystemExit(tyro.cli(main, prog="python -m kla"))
