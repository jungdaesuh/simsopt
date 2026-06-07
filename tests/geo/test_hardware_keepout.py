"""Tests for the hardware keep-out penalty (``banana_opt.hardware_keepout``).

Covers the objective contract the single-stage driver relies on:
gradient correctness against finite differences, exact zero away from the
cloud, chunking/padding invariance, per-curve attribution over a multi-curve
set, and the keep-out JSON loader's frame/schema validation.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
EXAMPLES_ROOT_STR = str(EXAMPLES_ROOT)
if EXAMPLES_ROOT_STR not in sys.path:
    sys.path.insert(0, EXAMPLES_ROOT_STR)

from banana_opt.hardware_keepout import (  # noqa: E402
    CurveHardwareKeepout,
    load_hardware_keepout,
)
from banana_opt.hardware_contracts import (  # noqa: E402
    HARDWARE_KEEPOUT_MIN_DISTANCE_M,
)

from simsopt.geo import CurveXYZFourier  # noqa: E402


def _circle_curve(radius=1.0, order=3, quadpoints=64, seed=None):
    """A unit-ish circular CurveXYZFourier in the z=0 plane, optionally with a
    small random Fourier perturbation so gradients are generic."""
    curve = CurveXYZFourier(quadpoints, order)
    dofs = np.zeros(curve.dof_size)
    curve.x = dofs
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    if seed is not None:
        rng = np.random.default_rng(seed)
        curve.x = curve.x + 1e-3 * rng.standard_normal(curve.dof_size)
    return curve


class HardwareKeepoutObjectiveTests(unittest.TestCase):
    def test_far_cloud_gives_exact_zero_value_and_gradient(self):
        curve = _circle_curve(seed=1)
        # Cloud 10 m away: no quadpoint can be within the threshold.
        points = np.array([[10.0, 0.0, 0.0], [10.0, 0.1, 0.0]])
        objective = CurveHardwareKeepout(
            [curve], points, HARDWARE_KEEPOUT_MIN_DISTANCE_M, point_weight=1e-4)
        self.assertEqual(objective.J(), 0.0)
        self.assertTrue(np.all(objective.dJ() == 0.0))
        # The AABB pruning should have discarded every (curve, chunk) pair.
        self.assertEqual(objective.candidates, [])

    def test_near_cloud_activates_and_decreases_with_distance(self):
        curve = _circle_curve(seed=2)
        # Point just inside the threshold of the curve's +x crossing (1, 0, 0).
        near = np.array([[1.0 + 0.5 * HARDWARE_KEEPOUT_MIN_DISTANCE_M, 0.0, 0.0]])
        farther = np.array([[1.0 + 0.9 * HARDWARE_KEEPOUT_MIN_DISTANCE_M, 0.0, 0.0]])
        j_near = CurveHardwareKeepout(
            [curve], near, HARDWARE_KEEPOUT_MIN_DISTANCE_M, point_weight=1e-4).J()
        j_farther = CurveHardwareKeepout(
            [curve], farther, HARDWARE_KEEPOUT_MIN_DISTANCE_M, point_weight=1e-4).J()
        self.assertGreater(j_near, 0.0)
        self.assertGreater(j_farther, 0.0)
        self.assertGreater(j_near, j_farther)

    def test_gradient_matches_finite_differences(self):
        # Taylor test at a configuration with an ACTIVE hinge: a small cloud
        # near the curve. Central differences along a fixed random direction.
        curve = _circle_curve(seed=3)
        rng = np.random.default_rng(7)
        points = np.array([
            [1.0 + 0.4 * HARDWARE_KEEPOUT_MIN_DISTANCE_M, 0.0, 0.0],
            [0.0, 1.0 + 0.6 * HARDWARE_KEEPOUT_MIN_DISTANCE_M, 0.0],
        ])
        objective = CurveHardwareKeepout(
            [curve], points, HARDWARE_KEEPOUT_MIN_DISTANCE_M, point_weight=1e-4)
        x0 = np.asarray(curve.x, dtype=float).copy()
        direction = rng.standard_normal(x0.size)
        direction /= np.linalg.norm(direction)
        analytic = float(np.dot(objective.dJ(), direction))
        self.assertNotEqual(analytic, 0.0)

        eps = 1e-7
        curve.x = x0 + eps * direction
        j_plus = objective.J()
        curve.x = x0 - eps * direction
        j_minus = objective.J()
        curve.x = x0
        fd = (j_plus - j_minus) / (2.0 * eps)
        self.assertAlmostEqual(
            analytic, fd, delta=1e-5 * max(1.0, abs(analytic)),
            msg=f"analytic {analytic} vs finite-difference {fd}")

    def test_chunking_and_padding_do_not_change_the_value(self):
        curve = _circle_curve(seed=4)
        rng = np.random.default_rng(11)
        # A ring of points straddling the curve, many inside the threshold.
        theta = rng.uniform(0.0, 2.0 * np.pi, size=333)
        radius = 1.0 + rng.uniform(-0.5, 0.5, size=333) * HARDWARE_KEEPOUT_MIN_DISTANCE_M
        points = np.column_stack([
            radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])
        j_one_chunk = CurveHardwareKeepout(
            [curve], points, HARDWARE_KEEPOUT_MIN_DISTANCE_M,
            point_weight=1e-4, chunk_size=100000).J()
        j_many_chunks = CurveHardwareKeepout(
            [curve], points, HARDWARE_KEEPOUT_MIN_DISTANCE_M,
            point_weight=1e-4, chunk_size=64).J()
        self.assertGreater(j_one_chunk, 0.0)
        self.assertAlmostEqual(j_one_chunk, j_many_chunks, places=12)

    def test_multi_curve_attribution(self):
        # Two curves; the cloud sits near curve B only. J must equal the
        # single-curve value for B, and A must receive zero gradient.
        curve_a = _circle_curve(radius=1.0, seed=5)
        curve_b = _circle_curve(radius=1.0, seed=6)
        curve_b.set("zc(0)", 0.5)  # lift B half a metre
        points = np.array([[1.0, 0.0, 0.5 + 0.4 * HARDWARE_KEEPOUT_MIN_DISTANCE_M]])

        both = CurveHardwareKeepout(
            [curve_a, curve_b], points, HARDWARE_KEEPOUT_MIN_DISTANCE_M,
            point_weight=1e-4)
        only_b = CurveHardwareKeepout(
            [curve_b], points, HARDWARE_KEEPOUT_MIN_DISTANCE_M, point_weight=1e-4)
        self.assertGreater(both.J(), 0.0)
        self.assertAlmostEqual(both.J(), only_b.J(), places=12)
        # Gradient attribution: the dof vector is [curve_a dofs, curve_b dofs];
        # A is untouched by the cloud, B matches the single-curve gradient.
        grad_both = np.asarray(both.dJ(), dtype=float)
        n_a = curve_a.dof_size
        self.assertTrue(
            np.all(grad_both[:n_a] == 0.0),
            msg="curve A is far from the cloud but received gradient",
        )
        np.testing.assert_allclose(
            grad_both[n_a:],
            np.asarray(only_b.dJ(), dtype=float),
            rtol=0.0,
            atol=1e-15,
            err_msg="curve B gradient differs between joint and single-curve objectives",
        )

    def test_shortest_distance_diagnostic(self):
        curve = _circle_curve(seed=8)
        points = np.array([[1.5, 0.0, 0.0]])
        objective = CurveHardwareKeepout(
            [curve], points, HARDWARE_KEEPOUT_MIN_DISTANCE_M, point_weight=1e-4)
        # Quadpoint nearest (1.5, 0, 0) is ~(1, 0, 0): distance ~0.5 up to the
        # 64-quadpoint discretization of the circle (and the tiny seeded
        # Fourier perturbation), which shifts the nearest sample by ~1e-2.
        self.assertAlmostEqual(objective.shortest_distance(), 0.5, delta=0.02)


class HardwareKeepoutLoaderTests(unittest.TestCase):
    def _write(self, payload):
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False)
        json.dump(payload, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_loads_schema_v1_and_concatenates_groups(self):
        path = self._write({
            "schema_version": 1,
            "frame": "machine_metres_zup",
            "units": "m",
            "spacing_m": 0.006,
            "recommended_min_distance_m": 0.0226,
            "groups": [
                {"label": "sensors", "points": [[1.0, 0.0, 0.0]]},
                {"label": "solenoid", "points": [[0.0, 1.0, 0.0], [0.0, 1.1, 0.0]]},
            ],
            "provenance": {"glb_sha256": "abc"},
        })
        points, weight, d_min, provenance = load_hardware_keepout(path)
        self.assertEqual(points.shape, (3, 3))
        self.assertAlmostEqual(weight, 0.006 ** 2)
        self.assertEqual(d_min, 0.0226)
        self.assertEqual(d_min, HARDWARE_KEEPOUT_MIN_DISTANCE_M)
        self.assertEqual(provenance["glb_sha256"], "abc")

    def test_rejects_wrong_frame(self):
        path = self._write({
            "schema_version": 1,
            "frame": "render_yup",
            "units": "m",
            "spacing_m": 0.006,
            "recommended_min_distance_m": 0.0226,
            "groups": [{"label": "sensors", "points": [[1.0, 0.0, 0.0]]}],
            "provenance": {},
        })
        with self.assertRaises(ValueError):
            load_hardware_keepout(path)

    def test_rejects_unknown_schema_version(self):
        path = self._write({"schema_version": 2})
        with self.assertRaises(ValueError):
            load_hardware_keepout(path)


if __name__ == "__main__":
    unittest.main()
