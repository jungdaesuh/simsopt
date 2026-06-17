import math

import numpy as np
import pytest

from geo._frontier_test_helpers import ensure_examples_import_path

ensure_examples_import_path()

from banana_opt.trace_transit_normalization import (
    TRANSIT_NORMALIZATION_SCHEMA_VERSION,
    measure_axis_field,
    predicted_transits,
    tmax_for_transits,
    transit_normalization_metadata,
)

from simsopt.field import ToroidalField, compute_fieldlines


B0 = 0.5
R0 = 0.9


def test_predicted_transits_formula_and_roundtrip():
    n = predicted_transits(500.0, b_axis_T=B0, r_axis_m=R0)
    assert n == pytest.approx(B0 * 500.0 / (2.0 * math.pi * R0))
    assert tmax_for_transits(n, b_axis_T=B0, r_axis_m=R0) == pytest.approx(500.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tmax": -1.0, "b_axis_T": B0, "r_axis_m": R0},
        {"tmax": 0.0, "b_axis_T": B0, "r_axis_m": R0},
        {"tmax": 500.0, "b_axis_T": 0.0, "r_axis_m": R0},
        {"tmax": 500.0, "b_axis_T": B0, "r_axis_m": -0.9},
        {"tmax": float("nan"), "b_axis_T": B0, "r_axis_m": R0},
    ],
)
def test_predicted_transits_rejects_nonpositive_inputs(kwargs):
    with pytest.raises(ValueError):
        predicted_transits(kwargs["tmax"], b_axis_T=kwargs["b_axis_T"], r_axis_m=kwargs["r_axis_m"])


def test_measure_axis_field_exact_on_toroidal_field():
    field = ToroidalField(R0, B0)
    axis = measure_axis_field(field, r_axis_m=R0)
    # ToroidalField: B = B0 * R0 / R * phi-hat, so |B_phi| == B0 exactly at R0.
    assert axis["b_axis_T"] == pytest.approx(B0, rel=1e-12)
    assert axis["b_mod_mean_T"] == pytest.approx(B0, rel=1e-12)
    assert axis["r_axis_m"] == pytest.approx(R0)
    assert axis["schema_version"] == TRANSIT_NORMALIZATION_SCHEMA_VERSION


def test_measure_axis_field_requires_exactly_one_radius_source():
    field = ToroidalField(R0, B0)
    with pytest.raises(ValueError):
        measure_axis_field(field)


def test_measure_axis_field_restores_field_points():
    field = ToroidalField(R0, B0)
    probe = np.array([[1.1, 0.2, 0.05]])
    field.set_points(probe)
    measure_axis_field(field, r_axis_m=R0)
    np.testing.assert_allclose(field.get_points_cart_ref(), probe)


def test_predicted_transits_matches_traced_phi_crossings():
    """End-to-end: on a pure toroidal field the formula is exact.

    dphi/dt = B_phi / R = B0 R0 / R^2; a field line seeded at R0 stays on
    the circle R = R0, so transits = B0 * tmax / (2 pi R0) exactly.
    """
    field = ToroidalField(R0, B0)
    tmax = 100.0
    expected = predicted_transits(tmax, b_axis_T=B0, r_axis_m=R0)
    _, phi_hits = compute_fieldlines(
        field,
        [R0],
        [0.0],
        tmax=tmax,
        tol=1e-9,
        phis=[0.0],
        stopping_criteria=[],
    )
    measured = int(np.sum(phi_hits[0][:, 1] >= 0))
    assert measured == int(expected)


def test_transit_normalization_metadata_contents():
    field = ToroidalField(R0, B0)

    class _Surf:
        @staticmethod
        def major_radius():
            return R0

    meta = transit_normalization_metadata(field, _Surf(), 500.0)
    assert meta["tmax"] == 500.0
    assert meta["implied_transits"] == pytest.approx(
        B0 * 500.0 / (2.0 * math.pi * R0)
    )
    assert meta["transits_per_unit_tmax"] == pytest.approx(
        B0 / (2.0 * math.pi * R0)
    )
