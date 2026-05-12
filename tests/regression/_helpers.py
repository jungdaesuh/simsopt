"""Shared utilities for the colleague-artifact regression panel.

Used by:
  - tests/regression/_smoke_colleague_artifacts.py   (diagnostic script)
  - tests/regression/_generate_colleague_snapshots.py (snapshot writer)
  - tests/regression/test_colleague_artifact.py       (pytest module)
"""

from __future__ import annotations

import hashlib
import os
import platform as _platform
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
_EXAMPLES = REPO / "examples" / "single_stage_optimization"
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

from simsopt.field.coil import Current, ScaledCurrent  # noqa: E402

try:
    from simsopt.field.coil import CurrentSum
except ImportError:
    CurrentSum = None

MU0 = 4.0 * np.pi * 1e-7

ARTIFACT_DIR = Path("/Users/suhjungdae/code/columbia/banana_drivers/inputs")
ARTIFACT_KEYS = ("01", "02", "10", "20")
ARTIFACTS = {kA: ARTIFACT_DIR / f"bsurf_opt_{kA}kA.json" for kA in ARTIFACT_KEYS}

SNAPSHOT_DIR = Path(__file__).resolve().parent / "colleague_artifact_snapshots"

EVAL_POINTS_SEED = 1234
EVAL_POINTS_N = 100
LINEARITY_PROBE_SEED = 5678
LINEARITY_PROBE_N = 30
BOOZER_KERNEL_IOTA = 0.27
CCD_MIN_DISTANCE = 0.05
CCD_DOWNSAMPLE = 2


def sha_full(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def sha_short(arr) -> str:
    return sha_full(arr)[:16]


def eval_points(surface, *, seed: int, n: int) -> np.ndarray:
    """Deterministic eval points in a box around the surface."""
    pts = surface.gamma().reshape(-1, 3)
    centroid = pts.mean(axis=0)
    half = 0.5 * (pts.max(axis=0) - pts.min(axis=0))
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(centroid + rng.uniform(-half, half, (n, 3)))


def unique_leaf_currents(biotsavart):
    """Walk BS coil graph and return unique leaf Current DOFs."""
    leaves = {}
    pending = [coil.current for coil in reversed(biotsavart.coils)]
    while pending:
        c = pending.pop()
        if isinstance(c, Current):
            leaves[id(c)] = c
        elif isinstance(c, ScaledCurrent):
            pending.append(c.current_to_scale)
        elif CurrentSum is not None and isinstance(c, CurrentSum):
            pending.append(c.current_b)
            pending.append(c.current_a)
    return list(leaves.values())


def scale_leaf_currents_in_memory(biotsavart, factor: float):
    """Multiply every unique leaf Current value by ``factor``.

    Handles the case where all leaf currents are fixed (the colleague's
    artifact ships them fixed). Returns a restore-callable that returns
    the BS to its previous fixed/free state and value.
    """
    leaves = unique_leaf_currents(biotsavart)
    saved = []
    for c in leaves:
        saved.append((c.dofs.all_fixed(), c.local_full_x.copy()))
        c.unfix_all()
        c.x = factor * c.x

    def restore():
        for c, (was_fixed, vals) in zip(leaves, saved):
            c.unfix_all()
            c.x = vals.copy()
            if was_fixed:
                c.fix_all()
        biotsavart.clear_cached_properties()

    return leaves, restore


def env_summary() -> dict:
    """Diagnostic record of the current environment, for the snapshot _meta."""
    try:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        head = "<unknown>"

    blas = "<unknown>"
    try:
        cfg = np.show_config(mode="dicts")
        blas_info = cfg.get("Build Dependencies", {}).get("blas", {})
        blas = f"{blas_info.get('name', '?')}/{blas_info.get('version', '?')}"
    except Exception:
        pass

    try:
        import scipy
        scipy_v = scipy.__version__
    except Exception:
        scipy_v = "<unavailable>"

    return {
        "head_sha": head,
        "numpy_version": np.__version__,
        "scipy_version": scipy_v,
        "blas": blas,
        "platform_system": _platform.system().lower(),
        "platform_machine": _platform.machine(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
    }
