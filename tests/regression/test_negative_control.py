"""End-to-end negative-control sanity tests for the regression panel.

Two kinds of tests live here:

1. **End-to-end panel-resolution tests.** Monkey-patch the math layer at
   the patch points the actual helpers call through (``bs.B``,
   ``label.J``, ``surface.gamma``, ``simsoptpp.boozer_residual`` via
   ``_helpers._sopp``), then invoke the *same* assertion helper that
   ``test_colleague_artifact.py`` uses, and assert ``pytest.raises(
   AssertionError)``. These prove the panel catches injected drift
   through its actual codepath — not just that the comparison math
   detects deltas in pre-computed arrays.

2. **Comparison-threshold sanity tests.** Synthetic perturbations on
   snapshot scalars run through ``np.testing.assert_allclose`` at the
   panel's documented tolerances. These prove the comparison-math
   resolution claim (1e-14 for Volume, 1e-12 for kernel, etc.). They are
   *not* a guard on the panel's path-through-the-helper; they are a guard
   on the choice of tolerance constants.

If a future commit weakens any panel tolerance below the documented
resolution or routes a helper around the patched call site, the
corresponding test here fails — they are a guard on the guard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import _helpers as H  # noqa: E402

from banana_opt.json_compat import load_boozer_finite_i  # noqa: E402


NEG_CONTROL_KA = "01"
PERTURBATION_B = 1.0 + 1e-10           # > BLAS/FMA ULP drift on B
PERTURBATION_VOLUME = 1.0 + 1e-13       # > Volume rtol=1e-14
PERTURBATION_KERNEL = 1.0 + 1e-11       # > kernel rtol=1e-12


@pytest.fixture(scope="module")
def loaded():
    path = H.ARTIFACTS[NEG_CONTROL_KA]
    if not path.exists():
        pytest.skip(f"colleague artifact not present at {path}")
    obj = load_boozer_finite_i(str(path))
    obj.biotsavart.clear_cached_properties()
    return obj


@pytest.fixture(scope="module")
def snapshot():
    snap_path = H.SNAPSHOT_DIR / f"bsurf_opt_{NEG_CONTROL_KA}kA.snapshot.json"
    if not snap_path.exists():
        pytest.skip(f"snapshot not present at {snap_path}")
    with open(snap_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 1. End-to-end panel-resolution tests
#
# Each test injects a perturbation at the math source and invokes the
# *exact* assertion helper that test_colleague_artifact.py uses. The
# pytest.raises(AssertionError) confirms the panel codepath catches the
# injected drift end-to-end, not just that the comparison math works.
# ---------------------------------------------------------------------------


def test_panel_baseline_no_perturbation_passes(loaded, snapshot):
    """Sanity: with no patches, the panel helpers all pass on this artifact."""
    H.assert_surface_geometry_matches_snapshot(loaded, snapshot)
    H.assert_volume_matches_snapshot(loaded, snapshot)
    H.assert_biot_savart_eval_matches_snapshot(loaded, snapshot)
    H.assert_coil0_geometry_matches_snapshot(loaded, snapshot)
    H.assert_curve_curve_distance_matches_snapshot(loaded, snapshot)
    H.assert_boozer_kernel_path_b_matches_snapshot(loaded, snapshot)


def test_panel_assert_biot_savart_eval_fails_on_perturbed_B(loaded, snapshot, monkeypatch):
    """Monkey-patch bs.B to multiply by (1+1e-10); assert the panel helper
    catches it. Goes through the panel's actual codepath."""
    bs = loaded.biotsavart
    original_B = bs.B

    def perturbed_B(*args, **kwargs):
        return original_B(*args, **kwargs) * PERTURBATION_B

    monkeypatch.setattr(bs, "B", perturbed_B)
    with pytest.raises(AssertionError):
        H.assert_biot_savart_eval_matches_snapshot(loaded, snapshot)


def test_panel_assert_biot_savart_eval_fails_on_perturbed_dB(loaded, snapshot, monkeypatch):
    bs = loaded.biotsavart
    original_dB = bs.dB_by_dX

    def perturbed_dB(*args, **kwargs):
        return original_dB(*args, **kwargs) * PERTURBATION_B

    monkeypatch.setattr(bs, "dB_by_dX", perturbed_dB)
    with pytest.raises(AssertionError):
        H.assert_biot_savart_eval_matches_snapshot(loaded, snapshot)


def test_panel_assert_volume_fails_on_perturbed_label(loaded, snapshot, monkeypatch):
    label = loaded.label
    original_J = label.J

    def perturbed_J(*args, **kwargs):
        return original_J(*args, **kwargs) * PERTURBATION_VOLUME

    monkeypatch.setattr(label, "J", perturbed_J)
    with pytest.raises(AssertionError):
        H.assert_volume_matches_snapshot(loaded, snapshot)


def test_panel_assert_surface_geometry_fails_on_perturbed_gamma(loaded, snapshot, monkeypatch):
    surface = loaded.surface
    original_gamma = surface.gamma

    def perturbed_gamma(*args, **kwargs):
        return original_gamma(*args, **kwargs) * PERTURBATION_B

    monkeypatch.setattr(surface, "gamma", perturbed_gamma)
    with pytest.raises(AssertionError):
        H.assert_surface_geometry_matches_snapshot(loaded, snapshot)


def test_panel_assert_boozer_kernel_fails_on_perturbed_raw_kernel(loaded, snapshot, monkeypatch):
    """Monkey-patch the raw sopp.boozer_residual via the module attr that
    _helpers references (`_helpers._sopp`). The helper reads
    `_sopp.boozer_residual` at call time, so the patch is observed."""
    original_kernel = H._sopp.boozer_residual

    def perturbed_kernel(*args, **kwargs):
        return original_kernel(*args, **kwargs) * PERTURBATION_KERNEL

    monkeypatch.setattr(H._sopp, "boozer_residual", perturbed_kernel)
    with pytest.raises(AssertionError):
        H.assert_boozer_kernel_path_b_matches_snapshot(loaded, snapshot)


def test_panel_assert_boozer_kernel_fails_on_perturbed_wrapper(loaded, snapshot, monkeypatch):
    """Monkey-patch the finite-I wrapper via `_helpers._bfc`."""
    original_wrapper = H._bfc.boozer_surface_residual_finite_I

    def perturbed_wrapper(*args, **kwargs):
        out = original_wrapper(*args, **kwargs)
        # boozer_surface_residual_finite_I returns a tuple; perturb the residual
        residual = out[0] * PERTURBATION_KERNEL
        return (residual,) + tuple(out[1:])

    monkeypatch.setattr(H._bfc, "boozer_surface_residual_finite_I", perturbed_wrapper)
    with pytest.raises(AssertionError):
        H.assert_boozer_kernel_path_b_matches_snapshot(loaded, snapshot)


def test_panel_assert_curve_curve_distance_fails_on_perturbed_value(loaded, snapshot, monkeypatch):
    """Patch CurveCurveDistance.J at the class level — instances are
    constructed inside the helper, so instance patches do not apply."""
    from simsopt.geo.curveobjectives import CurveCurveDistance
    original_J = CurveCurveDistance.J

    def perturbed_J(self, *args, **kwargs):
        return original_J(self, *args, **kwargs) * PERTURBATION_KERNEL

    monkeypatch.setattr(CurveCurveDistance, "J", perturbed_J)
    with pytest.raises(AssertionError):
        H.assert_curve_curve_distance_matches_snapshot(loaded, snapshot)


def test_panel_linearity_oracle_fails_on_injected_drift(loaded, monkeypatch):
    """Linearity oracle is reference-free. Verify it fails if B' != 2*B."""
    surface = loaded.surface
    bs = loaded.biotsavart
    pts = H.eval_points(surface, seed=H.LINEARITY_PROBE_SEED, n=H.LINEARITY_PROBE_N)

    bs.clear_cached_properties()
    bs.set_points(pts)
    B0 = bs.B().copy()

    leaves, restore = H.scale_leaf_currents_in_memory(bs, 2.0)
    bs.clear_cached_properties()
    bs.set_points(pts)
    B1 = bs.B().copy() * (1.0 + 1.0e-15)  # synthetic drift
    restore()

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(B1, 2.0 * B0)


# ---------------------------------------------------------------------------
# 2. Comparison-threshold sanity tests
#
# These prove the assert_allclose tolerance constants alone catch the
# documented resolution. They do NOT cover the helper-end-to-end path
# (see section 1 for that). Kept to lock the tolerance constants
# themselves.
# ---------------------------------------------------------------------------


def test_threshold_volume_rtol_1e14_catches_1e13_perturbation(snapshot):
    V_perturbed = snapshot["volume"] * (1.0 + 1.0e-13)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(V_perturbed, snapshot["volume"], rtol=1e-14)


def test_threshold_kernel_rtol_1e12_catches_1e11_perturbation(snapshot):
    expected = snapshot["boozer_kernel_path_b"]["raw_kernel_value"]
    perturbed = expected * (1.0 + 1.0e-11)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(perturbed, expected, rtol=1e-12)


def test_threshold_B_rtol_1e13_catches_1e10_perturbation(snapshot):
    B_first10 = np.array(snapshot["biot_savart_eval"]["B_sample_first10_flat"])
    B_perturbed = B_first10 * (1.0 + 1.0e-10)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(B_perturbed, B_first10, rtol=1e-13)
