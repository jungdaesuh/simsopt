import math

import numpy as np

from geo._frontier_test_helpers import ensure_examples_import_path

ensure_examples_import_path()

import topology_scorer


class RingSurface:
    nfp = 1

    def cross_section(self, *, phi, thetas):
        theta = np.linspace(0.0, 2.0 * np.pi, int(thetas), endpoint=False)
        return np.column_stack(
            (
                1.1 + 0.1 * np.cos(theta),
                np.zeros_like(theta) + float(phi),
                0.1 * np.sin(theta),
            )
        )


class ShiftedCentroidSurface:
    nfp = 1

    def cross_section(self, *, phi, thetas):
        theta = np.linspace(0.0, 2.0 * np.pi, int(thetas), endpoint=False)
        return np.column_stack(
            (
                1.22 + 0.14 * np.cos(theta),
                np.zeros_like(theta) + float(phi),
                0.0 + 0.1 * np.sin(theta),
            )
        )


class ShiftedAxisField:
    def __init__(self, *, axis_r, axis_z, iota):
        self.axis_r = float(axis_r)
        self.axis_z = float(axis_z)
        self.iota = float(iota)
        self._points = np.zeros((0, 3), dtype=float)

    def set_points(self, points):
        self._points = np.asarray(points, dtype=float)
        return self

    def B(self):
        vectors = []
        for point in self._points:
            x, y, z = point
            radius = math.hypot(float(x), float(y))
            phi = math.atan2(float(y), float(x))
            e_r = np.asarray([math.cos(phi), math.sin(phi), 0.0], dtype=float)
            e_phi = np.asarray([-math.sin(phi), math.cos(phi), 0.0], dtype=float)
            e_z = np.asarray([0.0, 0.0, 1.0], dtype=float)

            d_radius_d_phi = -self.iota * (float(z) - self.axis_z)
            d_z_d_phi = self.iota * (radius - self.axis_r)
            b_phi = 1.0
            b_radius = d_radius_d_phi * b_phi / radius
            b_z = d_z_d_phi * b_phi / radius
            vectors.append(b_radius * e_r + b_phi * e_phi + b_z * e_z)
        return np.asarray(vectors, dtype=float)


def test_invariant_torus_classification_reports_not_evaluated_for_short_traces():
    short_hits = np.array(
        [[float(step), 0.0, 1.1, 0.0, 0.0] for step in range(60)],
        dtype=float,
    )

    result = topology_scorer.invariant_torus_classification(
        [short_hits],
        RingSurface(),
        axis_point={"r": 1.1, "z": 0.0, "source": "test_magnetic_axis"},
    )

    assert result["invariant_torus_fraction"] is None
    assert result["wba_classified_seed_count"] == 0
    assert result["wba_classification_counts"]["insufficient_returns"] == 1
    assert result["wba_evaluation_state"] == "not_evaluated_insufficient_returns"


def test_wba_reports_not_evaluated_without_magnetic_axis_reference():
    short_hits = np.array(
        [[float(step), 0.0, 1.1, 0.0, 0.0] for step in range(60)],
        dtype=float,
    )

    result = topology_scorer.invariant_torus_classification(
        [short_hits],
        RingSurface(),
    )

    assert result["invariant_torus_fraction"] is None
    assert result["wba_classified_seed_count"] == 0
    assert result["wba_axis"] is None
    assert result["wba_classification_counts"]["missing_magnetic_axis"] == 1
    assert result["wba_evaluation_state"] == "not_evaluated_missing_magnetic_axis"


def test_wba_uses_magnetic_axis_instead_of_boundary_centroid():
    axis_r = 1.05
    axis_z = 0.16
    rotation_number = (math.sqrt(5.0) - 1.0) / 10.0
    theta = np.arange(300, dtype=float) * 2.0 * math.pi * rotation_number
    radius = axis_r + 0.08 * np.cos(theta)
    z = axis_z + 0.08 * np.sin(theta)
    hits = np.column_stack(
        (
            np.arange(theta.size, dtype=float),
            np.zeros(theta.size, dtype=float),
            radius,
            np.zeros(theta.size, dtype=float),
            z,
        )
    )

    result = topology_scorer.invariant_torus_classification(
        [hits],
        ShiftedCentroidSurface(),
        bfield=ShiftedAxisField(axis_r=axis_r, axis_z=axis_z, iota=rotation_number),
    )

    surface_centroid = topology_scorer._surface_phi0_bounds(ShiftedCentroidSurface())[2]
    assert result["wba_axis"]["source"] == "magnetic_axis_fieldline_fixed_point"
    assert abs(result["wba_axis"]["r"] - axis_r) < 1.0e-6
    assert abs(result["wba_axis"]["z"] - axis_z) < 1.0e-6
    assert abs(result["wba_axis"]["r"] - surface_centroid[0]) > 1.0e-2
    assert abs(result["wba_axis"]["z"] - surface_centroid[1]) > 1.0e-2
    assert result["wba_evaluation_state"] == "evaluated"
    assert result["invariant_torus_fraction"] == 1.0


def test_empty_topology_score_does_not_promote_not_evaluated_wba_to_kam_fraction():
    result = topology_scorer.empty_topology_score_result(12, 50.0)

    assert result["invariant_torus_fraction"] is None
    assert result["kam_fraction"] is None
    assert result["kam_fraction_semantics"] is None
    assert result["wba_evaluation_state"] == "not_evaluated_no_classified_seeds"
