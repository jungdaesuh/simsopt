"""Regression panel pytest configuration.

Strict env / platform gate for the colleague-artifact regression panel.
The panel asserts SHA-equality on snapshots generated under specific
conditions (Darwin/arm64, Accelerate BLAS, numpy version recorded in the
snapshot `_meta`, `OMP_NUM_THREADS=1`). Running with mismatched
conditions would produce spurious failures that have nothing to do with
simsopt-core correctness.

Behavior: when any gate condition is violated, every test in
`tests/regression/test_colleague_artifact.py` and
`tests/regression/test_negative_control.py` is skipped with a clear
reason. The gate does **not** modify the operator's environment (no
`os.environ.setdefault`) — silent env mutation is a footgun.

See ``docs/regression_panel_colleague_artifacts_2026-05-11.md`` §5.5 and
§6.4. Tests/regression/README.md documents the local-only acceptance
line.
"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import numpy as np
import pytest

_SNAPSHOT_SYSTEM = "darwin"
_SNAPSHOT_MACHINE = "arm64"
_SNAPSHOT_DIR = Path(__file__).resolve().parent / "colleague_artifact_snapshots"


def _load_snapshot_meta():
    """Return a representative snapshot's _meta, or None if no snapshot exists."""
    snaps = sorted(_SNAPSHOT_DIR.glob("*.snapshot.json"))
    if not snaps:
        return None
    with open(snaps[0]) as f:
        return json.load(f).get("_meta", {})


def _gate_reasons():
    """Collect every condition under which the panel must skip. Returns a list of human-readable strings; empty means gate is open."""
    reasons = []

    sys_name = platform.system().lower()
    machine = platform.machine()
    if (sys_name, machine) != (_SNAPSHOT_SYSTEM, _SNAPSHOT_MACHINE):
        reasons.append(
            f"platform: got {sys_name}/{machine}, snapshots pinned to "
            f"{_SNAPSHOT_SYSTEM}/{_SNAPSHOT_MACHINE} (see plan §6.4)"
        )

    omp = os.environ.get("OMP_NUM_THREADS")
    if omp != "1":
        reasons.append(
            f"OMP_NUM_THREADS={omp!r}, required '1' for ULP-tight BLAS reduction order"
        )

    meta = _load_snapshot_meta()
    if meta is not None:
        expected_numpy = meta.get("numpy_version")
        if expected_numpy and np.__version__ != expected_numpy:
            reasons.append(
                f"numpy: got {np.__version__}, snapshot generated under {expected_numpy}"
            )

    return reasons


def pytest_collection_modifyitems(config, items):
    """Skip every panel + negative-control test when any gate condition fails."""
    reasons = _gate_reasons()
    if not reasons:
        return

    skip = pytest.mark.skip(
        reason="regression panel gate: " + "; ".join(reasons)
    )
    for item in items:
        path_str = str(item.fspath)
        if "tests/regression/test_colleague_artifact" in path_str:
            item.add_marker(skip)
        elif "tests/regression/test_negative_control" in path_str:
            item.add_marker(skip)
