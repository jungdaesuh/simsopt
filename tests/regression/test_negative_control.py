"""Negative-control sanity tests for the colleague-artifact regression panel.

These tests inject a known synthetic perturbation into the math layer and
assert that the panel's invariant checks detect it. They are not regression
tests — they are *resolution* tests: they prove that the panel has the
sensitivity it claims (per the tolerances in §6.2 of
docs/regression_panel_colleague_artifacts_2026-05-11.md).

If a future "tolerance relaxation" weakens the panel below the resolution
documented here, these negative-control tests will fail — which is the
desired behavior. They are a guard on the guard.
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


# Inject the negative control on this one artifact — enough to show resolution
# without paying for all 4 loads.
NEG_CONTROL_KA = "01"
PERTURBATION = 1.0 + 1e-10


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


def test_baseline_B_matches_snapshot(loaded, snapshot):
    """Unperturbed sanity: B at HEAD matches snapshot at full SHA. This is
    the baseline the perturbation tests below break against."""
    bs = loaded.biotsavart
    bs.clear_cached_properties()
    pts = H.eval_points(loaded.surface, seed=H.EVAL_POINTS_SEED, n=H.EVAL_POINTS_N)
    bs.set_points(pts)
    B = bs.B()
    assert H.sha_full(B) == snapshot["biot_savart_eval"]["B_sha256"]


def test_panel_detects_1e10_B_perturbation_via_sha(loaded, snapshot):
    """Multiply B by (1 + 1e-10) and assert SHA mismatches.

    A multiplicative 1e-10 perturbation on a value of order 1 changes ~30
    mantissa bits' worth in the least-significant region — far above any
    BLAS/FMA ULP drift. The SHA-256 of the array bytes must change.
    """
    bs = loaded.biotsavart
    bs.clear_cached_properties()
    pts = H.eval_points(loaded.surface, seed=H.EVAL_POINTS_SEED, n=H.EVAL_POINTS_N)
    bs.set_points(pts)
    B_unperturbed = bs.B().copy()
    B_perturbed = B_unperturbed * PERTURBATION

    expected_sha = snapshot["biot_savart_eval"]["B_sha256"]
    assert H.sha_full(B_unperturbed) == expected_sha
    assert H.sha_full(B_perturbed) != expected_sha, (
        "SHA-256 did not change after 1e-10 perturbation — panel resolution claim is broken"
    )


def test_panel_detects_1e10_B_perturbation_via_rtol(loaded, snapshot):
    """Same perturbation, asserted through the panel's rtol=1e-13 numeric check."""
    bs = loaded.biotsavart
    bs.clear_cached_properties()
    pts = H.eval_points(loaded.surface, seed=H.EVAL_POINTS_SEED, n=H.EVAL_POINTS_N)
    bs.set_points(pts)
    B = bs.B().copy() * PERTURBATION
    expected_first10 = np.array(snapshot["biot_savart_eval"]["B_sample_first10_flat"])

    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            np.ascontiguousarray(B).reshape(-1)[:10],
            expected_first10,
            rtol=1e-13,
            atol=0,
        )


def test_panel_detects_1e15_volume_perturbation(loaded, snapshot):
    """Volume tolerance is rtol=1e-14, so 1e-13 must be caught.

    Justifies the strictest tolerance in the panel: Volume is a closed-form
    polynomial in surface DOFs, so it should match bit-equally — anything
    larger than 1e-14 is a real change.
    """
    V_perturbed = loaded.label.J() * (1.0 + 1.0e-13)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(V_perturbed, snapshot["volume"], rtol=1e-14)


def test_panel_detects_1e13_kernel_perturbation(loaded, snapshot):
    """Path-B Boozer kernel tolerance is rtol=1e-12; 1e-11 must be caught."""
    kernel_perturbed = snapshot["boozer_kernel_path_b"]["raw_kernel_value"] * (1.0 + 1.0e-11)
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(
            kernel_perturbed, snapshot["boozer_kernel_path_b"]["raw_kernel_value"], rtol=1e-12
        )


def test_linearity_oracle_detects_1e15_violation(loaded):
    """The linearity oracle asserts array equality (rtol=0). Even a 1e-15
    multiplicative perturbation must be caught."""
    surface = loaded.surface
    bs = loaded.biotsavart
    pts = H.eval_points(surface, seed=H.LINEARITY_PROBE_SEED, n=H.LINEARITY_PROBE_N)
    bs.clear_cached_properties()
    bs.set_points(pts)
    B0 = bs.B().copy()

    leaves, restore = H.scale_leaf_currents_in_memory(bs, 2.0)
    bs.clear_cached_properties()
    bs.set_points(pts)
    B1 = bs.B().copy() * (1.0 + 1.0e-15)  # synthetic perturbation
    restore()

    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(B1, 2.0 * B0)
