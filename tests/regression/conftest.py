"""Regression panel pytest configuration.

Pins ``OMP_NUM_THREADS=1`` for determinism (if not set externally) and
auto-skips the snapshot-dependent tests when the runtime platform does not
match the platform the snapshots were generated on.

See ``docs/regression_panel_colleague_artifacts_2026-05-11.md`` §6.4 (platform
pinning) and §11 R2 (cross-platform mitigation).
"""

from __future__ import annotations

import os
import platform
import warnings

import pytest

# Set OMP threads to 1 as early as possible — before BLAS is initialized by
# any heavy import in the regression-dir tests. This affects subprocesses and
# any libraries that read the env at first use. If a parallel BLAS is already
# initialized in this process (e.g., numpy was imported by tests/conftest.py
# first), this set has no effect — see the pytest_configure check below for
# the runtime warning.
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Snapshot-pinned platform — must match the one used by
# tests/regression/_generate_colleague_snapshots.py at generation time.
_SNAPSHOT_SYSTEM = "darwin"
_SNAPSHOT_MACHINE = "arm64"


def pytest_configure(config):
    if os.environ.get("OMP_NUM_THREADS") != "1":
        warnings.warn(
            f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', '<unset>')!r}; "
            "regression panel requires '1' for ULP-tight snapshot reproducibility. "
            "Re-run with OMP_NUM_THREADS=1 in the environment.",
            UserWarning,
        )


def pytest_collection_modifyitems(config, items):
    """Auto-skip snapshot-dependent tests on a non-baseline platform.

    The negative-control and colleague-artifact panels rely on the snapshots
    under ``colleague_artifact_snapshots/`` which are Darwin/arm64-pinned by
    construction (see plan §6.4). Running them elsewhere produces spurious
    SHA mismatches that have nothing to do with simsopt-core correctness.
    """
    sys_name = platform.system().lower()
    machine = platform.machine()
    if (sys_name, machine) == (_SNAPSHOT_SYSTEM, _SNAPSHOT_MACHINE):
        return

    skip = pytest.mark.skip(
        reason=(
            f"Regression panel snapshots are pinned to "
            f"{_SNAPSHOT_SYSTEM}/{_SNAPSHOT_MACHINE}; running on "
            f"{sys_name}/{machine}. To enable on a second platform, generate a "
            f"platform-keyed snapshot (see plan §6.4) — do not relax tolerances."
        )
    )
    for item in items:
        path_str = str(item.fspath)
        if "tests/regression/test_colleague_artifact" in path_str:
            item.add_marker(skip)
        elif "tests/regression/test_negative_control" in path_str:
            item.add_marker(skip)
