"""GROC <-> hinge cross-check regression tests (SE-V-05).

Validates the Gonzalez-Maddocks thickness identity (PNAS 96:4769)

    shortest_groc == min(1/kappa_max, d_dc / 2)

on synthetic geometry, where ``d_dc`` is the doubly-critical
self-distance computed by
``banana_opt.self_intersect.doubly_critical_self_distance_np``, and
pins the rev-1 cross-check failure mode: substituting the 60 mm
arc-windowed hinge minimum for ``d_dc`` flips the proxy onto a
non-critical window-edge pair and overstates the error on
curvature-limited curves near the crossover
(``reduced_m12_presolved``, 8.61% apparent error at a 5% bound).
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.self_intersect import (  # noqa: E402
    _global_radius_values_np,
    _periodic_arc_distance_mask_np,
    doubly_critical_self_distance_np,
)

RELATIVE_TOLERANCE = 0.05
SELF_WINDOW_M = 0.060


def _central_difference_tangents(points: np.ndarray) -> np.ndarray:
    chords = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    return chords / np.linalg.norm(chords, axis=1)[:, None]


def _shortest_groc(points: np.ndarray, tangents: np.ndarray) -> float:
    return float(np.min(_global_radius_values_np(points, tangents)))


def _curvature_max(points: np.ndarray) -> float:
    previous_points = np.roll(points, 1, axis=0)
    next_points = np.roll(points, -1, axis=0)
    in_vec = points - previous_points
    out_vec = next_points - points
    in_len = np.linalg.norm(in_vec, axis=1)
    out_len = np.linalg.norm(out_vec, axis=1)
    in_tangent = in_vec / in_len[:, None]
    out_tangent = out_vec / out_len[:, None]
    arc_step = 0.5 * (in_len + out_len)
    curvature = np.linalg.norm(out_tangent - in_tangent, axis=1) / arc_step
    return float(np.max(curvature))


def _windowed_hinge_min(points: np.ndarray, window_m: float) -> float:
    """Point-pair hinge minimum (CurveSelfDistance.shortest_self_distance)."""
    mask = _periodic_arc_distance_mask_np(points, window_m)
    dist = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    return float(np.min(np.where(mask > 0.0, dist, np.inf)))


def _relative_error(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(value), abs(reference), 1.0e-12)


def _rounded_rectangle(
    *,
    width: float,
    length: float,
    corner_radius: float,
    spacing: float,
    bump_amplitude: float = 0.0,
    bump_wavelength: float = 0.24,
) -> np.ndarray:
    """Closed rounded-rectangle polyline, optional inward bump on long sides.

    Long sides run along x at y = +-width/2; corners are quarter circles
    of ``corner_radius``. ``bump_amplitude`` deflects both long sides
    inward by a C^1 cos^2 bump centred at x = 0, producing a
    doubly-critical pair at distance ``width - 2*bump_amplitude``.
    """
    half_length = length / 2.0
    half_width = width / 2.0
    straight_x = half_length - corner_radius
    straight_y = half_width - corner_radius

    def bump(x: np.ndarray) -> np.ndarray:
        inside = np.abs(x) <= bump_wavelength / 2.0
        return np.where(
            inside,
            bump_amplitude * np.cos(np.pi * x / bump_wavelength) ** 2,
            0.0,
        )

    pieces: list[np.ndarray] = []
    n_long = max(2, int(round(2.0 * straight_x / spacing)))
    x_bottom = np.linspace(-straight_x, straight_x, n_long, endpoint=False)
    pieces.append(
        np.column_stack(
            (x_bottom, -half_width + bump(x_bottom), np.zeros_like(x_bottom))
        )
    )
    n_corner = max(2, int(round((np.pi / 2.0) * corner_radius / spacing)))
    theta = np.linspace(-np.pi / 2.0, 0.0, n_corner, endpoint=False)
    pieces.append(
        np.column_stack(
            (
                straight_x + corner_radius * np.cos(theta),
                -straight_y + corner_radius * np.sin(theta),
                np.zeros_like(theta),
            )
        )
    )
    n_short = max(2, int(round(2.0 * straight_y / spacing)))
    y_right = np.linspace(-straight_y, straight_y, n_short, endpoint=False)
    pieces.append(
        np.column_stack(
            (np.full_like(y_right, half_length), y_right, np.zeros_like(y_right))
        )
    )
    theta = np.linspace(0.0, np.pi / 2.0, n_corner, endpoint=False)
    pieces.append(
        np.column_stack(
            (
                straight_x + corner_radius * np.cos(theta),
                straight_y + corner_radius * np.sin(theta),
                np.zeros_like(theta),
            )
        )
    )
    x_top = np.linspace(straight_x, -straight_x, n_long, endpoint=False)
    pieces.append(
        np.column_stack((x_top, half_width - bump(x_top), np.zeros_like(x_top)))
    )
    theta = np.linspace(np.pi / 2.0, np.pi, n_corner, endpoint=False)
    pieces.append(
        np.column_stack(
            (
                -straight_x + corner_radius * np.cos(theta),
                straight_y + corner_radius * np.sin(theta),
                np.zeros_like(theta),
            )
        )
    )
    y_left = np.linspace(straight_y, -straight_y, n_short, endpoint=False)
    pieces.append(
        np.column_stack(
            (np.full_like(y_left, -half_length), y_left, np.zeros_like(y_left))
        )
    )
    theta = np.linspace(np.pi, 1.5 * np.pi, n_corner, endpoint=False)
    pieces.append(
        np.column_stack(
            (
                -straight_x + corner_radius * np.cos(theta),
                -straight_y + corner_radius * np.sin(theta),
                np.zeros_like(theta),
            )
        )
    )
    return np.concatenate(pieces, axis=0)


class DoublyCriticalSelfDistanceTests(unittest.TestCase):
    def test_circle_has_no_doubly_critical_pair(self):
        """On a circle the pairwise distance has no interior local minimum,
        so d_dc = inf and the identity reduces to the curvature branch."""
        radius = 0.5
        theta = np.linspace(0.0, 2.0 * np.pi, 256, endpoint=False)
        points = radius * np.column_stack(
            (np.cos(theta), np.sin(theta), np.zeros_like(theta))
        )
        tangents = np.column_stack(
            (-np.sin(theta), np.cos(theta), np.zeros_like(theta))
        )
        distance, index_i, index_j, arc_sep = doubly_critical_self_distance_np(points)
        self.assertEqual(distance, np.inf)
        self.assertEqual((index_i, index_j), (-1, -1))
        self.assertEqual(arc_sep, np.inf)
        groc = _shortest_groc(points, tangents)
        # Point-tangent radius of any chord on a circle is exactly R.
        self.assertLess(_relative_error(groc, radius), 1.0e-12)

    def test_parallel_legs_doubly_critical_distance(self):
        """Opposite long sides of a rounded rectangle are doubly critical
        at exactly the side separation."""
        width = 0.060
        points = _rounded_rectangle(
            width=width, length=0.36, corner_radius=0.0275, spacing=0.001
        )
        distance, index_i, index_j, _ = doubly_critical_self_distance_np(points)
        self.assertLess(_relative_error(distance, width), 0.01)
        chord = points[index_j] - points[index_i]
        chord /= np.linalg.norm(chord)
        tangents = _central_difference_tangents(points)
        self.assertLess(abs(float(chord @ tangents[index_i])), 0.05)
        self.assertLess(abs(float(chord @ tangents[index_j])), 0.05)

    def test_crossover_curve_identity_holds_where_windowed_hinge_proxy_fails(self):
        """The reduced_m12_presolved failure class: 1/kappa_max sits between
        the window-edge hinge d/2 and the true d_dc/2. The corrected
        identity holds; the rev-1 windowed-hinge proxy breaks the 5% bound."""
        width = 0.060
        corner_radius = 0.0275
        points = _rounded_rectangle(
            width=width, length=0.36, corner_radius=corner_radius, spacing=0.001
        )
        tangents = _central_difference_tangents(points)
        groc = _shortest_groc(points, tangents)
        local_radius = 1.0 / _curvature_max(points)
        distance, _, _, _ = doubly_critical_self_distance_np(points)
        hinge_min = _windowed_hinge_min(points, SELF_WINDOW_M)

        # Crossover ordering: hinge/2 < 1/kappa_max < d_dc/2.
        self.assertLess(hinge_min / 2.0, local_radius)
        self.assertLess(local_radius, distance / 2.0)

        corrected_proxy = min(local_radius, distance / 2.0)
        legacy_proxy = min(local_radius, hinge_min / 2.0)
        self.assertLessEqual(
            _relative_error(groc, corrected_proxy), RELATIVE_TOLERANCE
        )
        self.assertGreater(_relative_error(groc, legacy_proxy), RELATIVE_TOLERANCE)
        # The binding mechanism is curvature: GROC tracks 1/kappa_max.
        self.assertLess(_relative_error(groc, corner_radius), 0.01)

    def test_distance_limited_curve_identity_holds(self):
        """Inward bumps pull the long sides to 48 mm so d_dc/2 < 1/kappa_max:
        the identity must follow the distance branch."""
        width = 0.060
        bump_amplitude = 0.006
        points = _rounded_rectangle(
            width=width,
            length=0.36,
            corner_radius=0.0275,
            spacing=0.001,
            bump_amplitude=bump_amplitude,
        )
        tangents = _central_difference_tangents(points)
        groc = _shortest_groc(points, tangents)
        local_radius = 1.0 / _curvature_max(points)
        distance, _, _, _ = doubly_critical_self_distance_np(points)
        expected_distance = width - 2.0 * bump_amplitude
        self.assertLess(_relative_error(distance, expected_distance), 0.01)
        self.assertLess(distance / 2.0, local_radius)
        corrected_proxy = min(local_radius, distance / 2.0)
        self.assertLessEqual(
            _relative_error(groc, corrected_proxy), RELATIVE_TOLERANCE
        )
        self.assertLess(_relative_error(groc, expected_distance / 2.0), 0.01)

    def test_non_uniform_sampling_invariance(self):
        """d_dc and the identity are stable under a ~3x sampling-density
        variation (the discretisation hypothesis for the 8.61% is wrong)."""
        width = 0.060
        uniform = _rounded_rectangle(
            width=width, length=0.36, corner_radius=0.0275, spacing=0.001
        )
        n = uniform.shape[0]
        s = np.linspace(0.0, 1.0, n, endpoint=False)
        warped = s + 0.08 * np.sin(2.0 * np.pi * s)
        chords = np.linalg.norm(np.roll(uniform, -1, axis=0) - uniform, axis=1)
        arc = np.concatenate(([0.0], np.cumsum(chords[:-1])))
        arc /= arc[-1] + chords[-1]
        non_uniform = np.column_stack(
            [np.interp(warped, arc, uniform[:, k], period=1.0) for k in range(3)]
        )
        d_uniform, _, _, _ = doubly_critical_self_distance_np(uniform)
        d_non_uniform, _, _, _ = doubly_critical_self_distance_np(non_uniform)
        self.assertLess(_relative_error(d_uniform, d_non_uniform), 0.01)
        tangents = _central_difference_tangents(non_uniform)
        groc = _shortest_groc(non_uniform, tangents)
        local_radius = 1.0 / _curvature_max(non_uniform)
        corrected_proxy = min(local_radius, d_non_uniform / 2.0)
        self.assertLessEqual(
            _relative_error(groc, corrected_proxy), RELATIVE_TOLERANCE
        )


if __name__ == "__main__":
    unittest.main()
