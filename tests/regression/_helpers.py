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
from simsopt.geo.curveobjectives import CurveCurveDistance  # noqa: E402
import simsoptpp as _sopp  # noqa: E402  — patchable by negative-control tests
from banana_opt import boozer_finite_current as _bfc  # noqa: E402  — patchable

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


_FIRST_N = 10


def _flat_prefix(arr):
    return np.ascontiguousarray(arr).reshape(-1)[:_FIRST_N]


# ---------------------------------------------------------------------------
# Panel assertion helpers — shared between the regression panel
# (test_colleague_artifact.py) and the end-to-end negative-control tests
# (test_negative_control.py). Both call exactly these functions so the
# negative controls exercise the same comparison logic, not duplicates.
#
# Each helper takes the loaded artifact and its snapshot dict, evaluates the
# math layer, and asserts SHA-equality + per-category tolerance.
# ---------------------------------------------------------------------------


def assert_surface_geometry_matches_snapshot(loaded, snapshot) -> None:
    gamma = loaded.surface.gamma()
    normal = loaded.surface.normal()
    expected = snapshot["surface_geometry"]

    assert list(gamma.shape) == expected["gamma_shape"]
    assert sha_full(gamma) == expected["gamma_sha256"], "gamma SHA mismatch"
    assert sha_full(normal) == expected["normal_sha256"], "normal SHA mismatch"
    np.testing.assert_allclose(
        _flat_prefix(gamma),
        expected["gamma_sample_first10_flat"],
        rtol=1e-14, atol=0,
    )


def assert_volume_matches_snapshot(loaded, snapshot) -> None:
    np.testing.assert_allclose(
        loaded.label.J(), snapshot["volume"], rtol=1e-14, atol=0
    )


def assert_biot_savart_eval_matches_snapshot(loaded, snapshot) -> None:
    bs = loaded.biotsavart
    bs.clear_cached_properties()
    pts = eval_points(loaded.surface, seed=EVAL_POINTS_SEED, n=EVAL_POINTS_N)
    bs.set_points(pts)
    B = bs.B().copy()
    dB = bs.dB_by_dX().copy()

    expected = snapshot["biot_savart_eval"]
    assert sha_full(pts) == expected["eval_points_sha256"], (
        "eval-points fixture diverged — regenerate snapshots"
    )
    assert sha_full(B) == expected["B_sha256"], "B SHA mismatch"
    assert sha_full(dB) == expected["dB_sha256"], "dB SHA mismatch"

    np.testing.assert_allclose(
        _flat_prefix(B), expected["B_sample_first10_flat"], rtol=1e-13, atol=0
    )
    np.testing.assert_allclose(
        _flat_prefix(dB), expected["dB_sample_first10_flat"], rtol=1e-12, atol=0
    )


def assert_coil0_geometry_matches_snapshot(loaded, snapshot) -> None:
    curve = loaded.biotsavart.coils[0].curve
    cgamma = curve.gamma()
    cdg = curve.dgamma_by_dcoeff()
    expected = snapshot["coil0_geometry"]

    assert type(curve).__name__ == expected["type"]
    assert list(cgamma.shape) == expected["gamma_shape"]
    assert list(cdg.shape) == expected["dgamma_dcoeff_shape"]
    assert sha_full(cgamma) == expected["gamma_sha256"]
    assert sha_full(cdg) == expected["dgamma_dcoeff_sha256"]


def assert_curve_curve_distance_matches_snapshot(loaded, snapshot) -> None:
    curves = [c.curve for c in loaded.biotsavart.coils]
    ccd = CurveCurveDistance(
        curves,
        minimum_distance=snapshot["curve_curve_distance"]["minimum_distance"],
        downsample=snapshot["curve_curve_distance"]["downsample"],
    )
    np.testing.assert_allclose(
        ccd.J(), snapshot["curve_curve_distance"]["value"], rtol=1e-12, atol=0
    )


def assert_boozer_kernel_path_b_matches_snapshot(loaded, snapshot) -> None:
    surface = loaded.surface
    bs = loaded.biotsavart
    I = float(loaded.I)
    iota = snapshot["boozer_kernel_path_b"]["iota"]
    sum_abs_I = float(sum(abs(c.current.get_value()) for c in bs.coils))
    G0 = MU0 * sum_abs_I
    G_eff = G0 + iota * I

    np.testing.assert_allclose(
        sum_abs_I, snapshot["boozer_kernel_path_b"]["sum_abs_I_coil"], rtol=1e-13
    )
    np.testing.assert_allclose(
        G_eff, snapshot["boozer_kernel_path_b"]["G_eff"], rtol=1e-13
    )

    # Wrapper path — call through the patchable module attribute.
    residual = _bfc.boozer_surface_residual_finite_I(
        surface, iota, G0, bs, derivatives=0, weight_inv_modB=False, I=I
    )[0]
    wrapper_val = 0.5 * float(np.sum(residual ** 2))

    # Raw kernel path — call through the patchable module attribute.
    x = surface.gamma()
    bs.set_points(x.reshape(-1, 3).copy())
    B_at_surf = bs.B().reshape(x.shape)
    kernel_val = float(_sopp.boozer_residual(
        G_eff, iota, surface.gammadash1(), surface.gammadash2(), B_at_surf, False
    ))
    kernel_at_G0 = float(_sopp.boozer_residual(
        G0, iota, surface.gammadash1(), surface.gammadash2(), B_at_surf, False
    ))

    np.testing.assert_allclose(
        wrapper_val,
        snapshot["boozer_kernel_path_b"]["wrapper_residual_sqsum_half"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        kernel_val,
        snapshot["boozer_kernel_path_b"]["raw_kernel_value"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(wrapper_val, kernel_val, rtol=1e-12)
    assert not np.isclose(kernel_val, kernel_at_G0), (
        "kernel(G_eff) == kernel(G0): finite-I term is silently being ignored"
    )


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
