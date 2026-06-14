"""Backend-mode contract for the single-stage continuation warm-start donor.

The donor build (``benchmarks/perlmutter/single_stage_continuation_donor.slurm``)
produces a warm-start surface by running ``run_single_stage_continuation.py``,
whose outer optimization is driven through the host-SciPy *reference* lane
(``optimizer_jax_reference.reference_minimize``, ``method='lbfgs'``). That lane
requires a native (non-jax) backend: ``_require_native_cpu_reference_backend_for_
scipy_adapter`` raises a ``RuntimeError`` unless ``backend != 'jax'``. The donor
passes no ``--backend`` / ``--optimizer-backend`` override, so the exported
``SIMSOPT_BACKEND_MODE`` alone decides whether that lane is permitted.

Perlmutter job 54390541 (2026-06-13) crashed 21 s into the first rung because the
launcher had copy-pasted the production parity env
``SIMSOPT_BACKEND_MODE=jax_cpu_parity`` (a jax backend). This test pins the donor
to a backend mode that resolves to a non-jax backend so that regression cannot
recur, and resolves the mode through the production SSOT map rather than a literal
string so it survives a future rename of the cpu mode.
"""
from __future__ import annotations

import re
from pathlib import Path

from simsopt_jax.backend.runtime import _MODE_TO_RUNTIME

REPO_ROOT = Path(__file__).resolve().parents[2]
DONOR_LAUNCHER = (
    REPO_ROOT / "benchmarks" / "perlmutter" / "single_stage_continuation_donor.slurm"
)


def _single_exported_value(script: str, name: str) -> str:
    matches = re.findall(
        rf"^\s*export\s+{re.escape(name)}=([^\s#]+)\s*(?:#.*)?$", script, re.MULTILINE
    )
    assert matches, f"{name} is not exported by the donor launcher"
    assert len(matches) == 1, (
        f"{name} is exported {len(matches)} times; expected exactly one source of truth"
    )
    return matches[0].strip().strip('"')


def test_donor_launcher_backend_mode_permits_host_scipy_reference_lane():
    script = DONOR_LAUNCHER.read_text(encoding="utf-8")

    # Premise anchor: the contract below only matters because the donor drives the
    # continuation, whose outer loop uses the host-SciPy reference lane.
    assert "run_single_stage_continuation.py" in script, (
        "donor launcher no longer runs run_single_stage_continuation.py; the "
        "backend-mode contract premise may no longer hold"
    )

    # The donor relies on the env-default optimizer lane. If it ever passes an
    # explicit backend/optimizer override the constraint changes, so this guard is
    # only valid while no override is present. Inspect command lines only, so
    # documenting the constraint in a comment does not trip this guard.
    command_lines = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )
    assert "--optimizer-backend" not in command_lines, (
        "donor launcher now passes --optimizer-backend; revisit this contract"
    )
    assert "--backend" not in command_lines, (
        "donor launcher now passes --backend; revisit this contract"
    )

    mode = _single_exported_value(script, "SIMSOPT_BACKEND_MODE")
    assert mode in _MODE_TO_RUNTIME, f"unknown SIMSOPT_BACKEND_MODE={mode!r}"
    backend, _platform = _MODE_TO_RUNTIME[mode]
    assert backend != "jax", (
        f"donor SIMSOPT_BACKEND_MODE={mode!r} resolves to the jax backend, but the "
        "continuation host-SciPy reference lane requires a native backend "
        "(_require_native_cpu_reference_backend_for_scipy_adapter raises under jax). "
        "See Perlmutter job 54390541."
    )
