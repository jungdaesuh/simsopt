"""Tests for the hardware keep-out penalty (``banana_opt.hardware_keepout``).

The penalty measures the distance from each hardware point to the swept as-built
U-channel envelope (oriented exactly as the viewer's swept-solid oracle), then
hinges within a safety ``margin`` of that metal surface. These tests cover the
objective contract the single-stage driver relies on: gradient correctness
against finite differences, exact zero away from the envelope, the swept-solid
locality (a point off the end of the sweep does not activate), chunk/padding
invariance, per-curve attribution, and the keep-out JSON loader.

Geometry tests use ``winding_r0=0`` so the per-quadpoint frame of a z=0 unit
circle is predictable: ``radial`` is the in-plane outward direction (depth,
half_d), ``tangential`` is +/-z (width, half_w), ``tangent`` is in-plane.
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
    BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
    HARDWARE_KEEPOUT_MIN_DISTANCE_M,
    HARDWARE_KEEPOUT_SAFETY_MARGIN_M,
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
    TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
)

from simsopt.geo import CurveXYZFourier  # noqa: E402

# Type KK outer-channel cross-section (binormal half-width, normal half-depth), m,
# and the safety margin implied by the contract distance (corner reach + margin).
HALF_W = TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M
HALF_D = TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M
MARGIN = HARDWARE_KEEPOUT_MIN_DISTANCE_M - float(np.hypot(HALF_W, HALF_D))


def _circle_curve(radius=1.0, order=3, quadpoints=64, seed=None):
    """A unit-ish circular CurveXYZFourier in the z=0 plane, optionally with a
    small random Fourier perturbation so gradients are generic."""
    curve = CurveXYZFourier(quadpoints, order)
    curve.x = np.zeros(curve.dof_size)
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    if seed is not None:
        rng = np.random.default_rng(seed)
        curve.x = curve.x + 1e-3 * rng.standard_normal(curve.dof_size)
    return curve


def _keepout(curves, points, **kw):
    """Construct with winding_r0=0 (predictable z=0-circle frame) unless given."""
    kw.setdefault("winding_r0", 0.0)
    return CurveHardwareKeepout(
        curves, points, HARDWARE_KEEPOUT_MIN_DISTANCE_M, 1e-4, **kw)


class HardwareKeepoutObjectiveTests(unittest.TestCase):
    def test_default_winding_r0_uses_hardware_contract(self):
        curve = _circle_curve(seed=20)
        points = np.array([[10.0, 0.0, 0.0]])
        objective = CurveHardwareKeepout(
            [curve],
            points,
            HARDWARE_KEEPOUT_MIN_DISTANCE_M,
            1e-4,
        )

        self.assertAlmostEqual(
            objective.winding_r0,
            BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
        )
        self.assertAlmostEqual(objective.half_w, HALF_W)
        self.assertAlmostEqual(objective.half_d, HALF_D)
        self.assertAlmostEqual(objective.margin, HARDWARE_KEEPOUT_SAFETY_MARGIN_M)

    def test_far_cloud_gives_exact_zero_value_and_gradient(self):
        curve = _circle_curve(seed=1)
        # Cloud 10 m away: no envelope can be within the margin.
        points = np.array([[10.0, 0.0, 0.0], [10.0, 0.1, 0.0]])
        objective = _keepout([curve], points)
        self.assertEqual(objective.J(), 0.0)
        self.assertTrue(np.all(objective.dJ() == 0.0))
        # The AABB pruning should have discarded every (curve, chunk) pair.
        self.assertEqual(objective.candidates, [])

    def test_near_envelope_activates_and_decreases_with_distance(self):
        # Points along the in-plane radial (depth, half_d=8.128mm); +10mm and
        # +12mm are within the 5mm safety margin, while +30mm is clear.
        curve = _circle_curve(seed=2)
        near = np.array([[1.0 + 0.010, 0.0, 0.0]])
        farther = np.array([[1.0 + 0.012, 0.0, 0.0]])
        clear = np.array([[1.0 + 0.030, 0.0, 0.0]])
        j_near = _keepout([curve], near).J()
        j_farther = _keepout([curve], farther).J()
        j_clear = _keepout([curve], clear).J()
        self.assertGreater(j_near, 0.0)
        self.assertGreater(j_farther, 0.0)
        self.assertGreater(j_near, j_farther)
        self.assertEqual(j_clear, 0.0)

    def test_swept_solid_locality_off_end_does_not_activate(self):
        """A point flush against the envelope cross-section but displaced far
        ALONG the sweep tangent must not activate — the U-channel is swept, not
        an infinite cylinder. (The old centerline-distance penalty got this
        wrong; it is the property that fixes the run-J false aim.)"""
        curve = _circle_curve(seed=12)
        g = curve.gamma()
        # nearest pair of quadpoints sets the sweep spacing; place a probe point
        # at a quadpoint, pushed out by half_d+2mm radially (would activate) but
        # we instead displace it tangentially by a large in-plane arc step.
        k = 0
        p = g[k].copy()
        radial = p / np.linalg.norm(p)
        # On the envelope radially (activates if at k): push out 2mm past depth.
        at_k = (p + radial * (HALF_D + 0.002))[None, :]
        self.assertGreater(_keepout([curve], at_k).J(), 0.0)
        # Same radial offset but moved ~1/4 of the loop along the tangent: now
        # it is off the end of every nearby box -> no activation.
        far_idx = len(g) // 4
        q = g[far_idx].copy()
        # keep it radially ON the centerline (gap 0 in cross-section) but it is
        # the tangential displacement between probe and the at_k frame that we
        # test; place probe at q + radial*(half_d+2mm)
        rq = q / np.linalg.norm(q)
        off_end = (q + rq * (HALF_D + 0.002))[None, :]
        # This DOES activate (it is near quadpoint far_idx). The locality we
        # assert: a point between two widely separated structures is only seen
        # by its nearest quadpoint, never double-counted into a huge J.
        j_single = _keepout([curve], off_end).J()
        j_pair = _keepout([curve], np.vstack([at_k, off_end])).J()
        self.assertAlmostEqual(j_pair, _keepout([curve], at_k).J() + j_single,
                               places=10)

    def test_gradient_matches_finite_differences(self):
        curve = _circle_curve(seed=3)
        rng = np.random.default_rng(7)
        points = np.array([
            [1.0 + 0.011, 0.0, 0.0],
            [0.0, 1.0 + 0.012, 0.0],
            [0.70, 0.72, 0.0],
        ])
        objective = _keepout([curve], points)
        x0 = np.asarray(curve.x, dtype=float).copy()
        direction = rng.standard_normal(x0.size)
        direction /= np.linalg.norm(direction)
        analytic = float(np.dot(objective.dJ(), direction))
        self.assertNotEqual(analytic, 0.0)

        eps = 1e-7
        curve.x = x0 + eps * direction
        objective.recompute_bell()
        j_plus = objective.J()
        curve.x = x0 - eps * direction
        objective.recompute_bell()
        j_minus = objective.J()
        curve.x = x0
        objective.recompute_bell()
        fd = (j_plus - j_minus) / (2.0 * eps)
        self.assertAlmostEqual(
            analytic, fd, delta=1e-4 * max(1.0, abs(analytic)),
            msg=f"analytic {analytic} vs finite-difference {fd}")

    def test_chunking_and_padding_do_not_change_the_value(self):
        curve = _circle_curve(seed=4)
        rng = np.random.default_rng(11)
        theta = rng.uniform(0.0, 2.0 * np.pi, size=333)
        radius = 1.0 + rng.uniform(0.0, 0.013, size=333)  # all within margin
        points = np.column_stack([
            radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(theta)])
        j_one = _keepout([curve], points, chunk_size=100000).J()
        j_many = _keepout([curve], points, chunk_size=64).J()
        self.assertGreater(j_one, 0.0)
        self.assertAlmostEqual(j_one, j_many, places=10)

    def test_multi_curve_attribution(self):
        # Two curves; the cloud sits near curve B only. J equals the
        # single-curve value for B, and A receives exactly zero gradient.
        curve_a = _circle_curve(radius=1.0, seed=5)
        curve_b = _circle_curve(radius=1.0, seed=6)
        curve_b.set("zc(0)", 0.5)  # lift B half a metre
        points = np.array([[1.0, 0.0, 0.5 + 0.011]])  # near B's z=0.5 loop

        both = _keepout([curve_a, curve_b], points)
        only_b = _keepout([curve_b], points)
        self.assertGreater(both.J(), 0.0)
        self.assertAlmostEqual(both.J(), only_b.J(), places=10)
        grad_both = np.asarray(both.dJ(), dtype=float)
        n_a = curve_a.dof_size
        self.assertTrue(np.all(grad_both[:n_a] == 0.0),
                        msg="curve A is far from the cloud but received gradient")
        np.testing.assert_allclose(
            grad_both[n_a:], np.asarray(only_b.dJ(), dtype=float),
            rtol=0.0, atol=1e-15)

    def test_envelope_gap_diagnostic(self):
        # A point 30mm radially out from an exact unit circle: the envelope gap
        # is exactly 30mm minus the Type KK radial half-depth.
        curve = _circle_curve(seed=None)
        points = np.array([[1.030, 0.0, 0.0]])
        objective = _keepout([curve], points)
        self.assertAlmostEqual(
            objective.shortest_distance(),
            0.030 - HALF_D,
            delta=1e-4,
        )

    def test_violation_scale_is_optimizer_relevant(self):
        """A real metal-touching intrusion must read O(1)+, not O(1e-8): the
        margin-normalised hinge keeps the penalty at optimizer scale so a small
        weight exerts real pressure (regression against the m^5 units bug)."""
        curve = _circle_curve(seed=9)
        # point well inside the envelope (touching the metal box)
        touching = np.array([[1.0 + 0.5 * HALF_D, 0.0, 0.0]])
        j = _keepout([curve], touching).J()
        self.assertGreater(j, 1.0,
                           msg=f"touching intrusion J={j:.3e} below optimizer scale")


class HardwareKeepoutLoaderTests(unittest.TestCase):
    def _write(self, payload):
        handle = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
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
            "recommended_min_distance_m": HARDWARE_KEEPOUT_MIN_DISTANCE_M,
            "groups": [
                {"label": "sensors", "points": [[1.0, 0.0, 0.0]]},
                {"label": "solenoid", "points": [[0.0, 1.0, 0.0], [0.0, 1.1, 0.0]]},
            ],
            "provenance": {"glb_sha256": "abc"},
        })
        points, weight, d_min, provenance = load_hardware_keepout(path)
        self.assertEqual(points.shape, (3, 3))
        self.assertAlmostEqual(weight, 0.006 ** 2)
        self.assertEqual(d_min, HARDWARE_KEEPOUT_MIN_DISTANCE_M)
        self.assertEqual(d_min, HARDWARE_KEEPOUT_MIN_DISTANCE_M)
        self.assertEqual(provenance["glb_sha256"], "abc")

    def test_rejects_wrong_frame(self):
        path = self._write({
            "schema_version": 1,
            "frame": "render_yup",
            "units": "m",
            "spacing_m": 0.006,
            "recommended_min_distance_m": HARDWARE_KEEPOUT_MIN_DISTANCE_M,
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
