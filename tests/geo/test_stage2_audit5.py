"""Unit tests for the 2026-06-11 formulation-audit Stage-2 fixes (5a/5b/5c).

These pin the three fixes audited/completed by agent C2:

* 5a (default-on for finite-build): the frame-aware winding-pack curvature
  limit and the in-run tightening helper. Proves the explicit off path returns
  the caller threshold byte-identically, that the enabled path only ever
  tightens, and that a broken contract (enabled without finite-build) raises
  loudly rather than silently no-opping.
* 5b (always-on, additive): the exact closed-chord-polyline segment minima.
  Proves the segment minimum is <= the point-cloud minimum on a constructed
  near-miss pair (cc) and near-miss curve/surface pair (cs), and that the
  additive results keys are emitted with their method labels.
* 5c (opt-in, weight-gated): the vessel-envelope keep-out wiring. Proves the
  CLI gate raises loudly on a negative weight, and that the realized CWS
  winding torus (not the 0.903 spec constant) orients the term's frame.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXAMPLE_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
if str(EXAMPLE_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_ROOT))

from banana_opt.coil_order_upgrade import realized_cws_winding_radii  # noqa: E402
from banana_opt.hardware_contracts import (  # noqa: E402
    BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
    TYPE_KK_INNER_RADIUS_MARGIN_M,
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
    TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
    VACUUM_VESSEL_MAJOR_RADIUS_M,
)
from banana_opt.hardware_keepout import CurveVesselEnvelopeKeepout  # noqa: E402
from banana_opt.stage2_geometry import (  # noqa: E402
    FiniteBuildSettings,
    closed_polyline_segments,
    curve_curve_min_distance_segments_m,
    curve_surface_min_distance_segments_m,
    finite_build_frame_aware_curvature_limit_inv_m,
    segment_segment_distance,
)

from simsopt.field import Coil, Current, coils_via_symmetries  # noqa: E402
from simsopt.geo import CurveCWSFourierCPP, SurfaceRZFourier  # noqa: E402

SOLVER_PATH = EXAMPLE_ROOT / "STAGE_2" / "banana_coil_solver.py"
from STAGE_2 import banana_coil_solver as _SOLVER  # noqa: E402


def _settings(frame="surface_tangent", numfilaments_n=2, numfilaments_b=7,
              gapsize_n=0.02, gapsize_b=0.04, rotation_order=1):
    return FiniteBuildSettings(
        numfilaments_n=numfilaments_n,
        numfilaments_b=numfilaments_b,
        gapsize_n=gapsize_n,
        gapsize_b=gapsize_b,
        rotation_order=rotation_order,
        frame=frame,
    )


class _StubGamma:
    """Minimal stand-in exposing only ``gamma()`` (all the segment-distance
    helpers consume)."""

    def __init__(self, points):
        self._g = np.ascontiguousarray(np.asarray(points, dtype=np.float64))

    def gamma(self):
        return self._g


def _point_cloud_min(points_a, points_b):
    a = np.asarray(points_a, dtype=np.float64)
    b = np.asarray(points_b, dtype=np.float64)
    return float(np.min(np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)))


def _build_cws_banana_family(major_radius, minor_radius=0.142):
    surface = SurfaceRZFourier(nfp=5, stellsym=True)
    surface.set_rc(0, 0, major_radius)
    surface.set_rc(1, 0, minor_radius)
    surface.set_zs(1, 0, minor_radius)
    quadpoints = np.linspace(0.0, 1.0, 64, endpoint=False)
    curve = CurveCWSFourierCPP(quadpoints, order=2, surf=surface, G=1, H=0)
    curve.set("phic(0)", 0.05)
    curve.set("phic(1)", 0.03)
    return list(
        coils_via_symmetries(
            [curve], [Current(1.1e4)], surface.nfp, surface.stellsym
        )
    )


# ───────────────────────── 5a ─────────────────────────


class FrameAwareCurvatureLimitTest(unittest.TestCase):
    def test_limit_is_reciprocal_of_margin_plus_outer_edgewise_reach(self) -> None:
        # Adopted self-intersection model: cap = 1/(inner-radius margin +
        # outer-channel edgewise reach max(half_depth, half_width)), independent
        # of the conductor-pack grid. The bench measurement ruled only the
        # flatwise/edgewise axis limits, so the conservative cap is the wider
        # edgewise axis -- not a stricter diagonal corner that was never measured.
        fb = _settings()
        margin = TYPE_KK_INNER_RADIUS_MARGIN_M
        outer_edgewise_reach = max(
            TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
            TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
        )
        limit = finite_build_frame_aware_curvature_limit_inv_m(fb, margin)
        self.assertAlmostEqual(
            limit, 1.0 / (margin + outer_edgewise_reach), places=12
        )
        # ~43.31/m for the Type-KK outer channel.
        self.assertAlmostEqual(limit, 43.31, places=1)
        # Stricter than the centerline cap (smaller inv_m = larger required radius).
        self.assertLess(limit, 100.0)

    def test_nonpositive_required_radius_is_infinite(self) -> None:
        # The outer-channel reach is a fixed positive constant, so the required
        # radius is non-positive only when the margin cancels it; the helper
        # returns inf at that degenerate boundary.
        fb = _settings()
        outer_corner_reach = float(
            np.hypot(
                TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
                TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
            )
        )
        self.assertTrue(
            np.isinf(
                finite_build_frame_aware_curvature_limit_inv_m(
                    fb, -outer_corner_reach
                )
            )
        )


class FrameAwareTighteningTest(unittest.TestCase):
    def test_resolver_defaults_on_for_finite_build_only(self) -> None:
        self.assertTrue(
            _SOLVER.stage2_frame_aware_curvature_threshold_enabled(
                SimpleNamespace(
                    finite_build=True,
                    finitebuild_frame_aware_curvature_threshold=None,
                )
            )
        )
        self.assertFalse(
            _SOLVER.stage2_frame_aware_curvature_threshold_enabled(
                SimpleNamespace(
                    finite_build=False,
                    finitebuild_frame_aware_curvature_threshold=None,
                )
            )
        )
        self.assertFalse(
            _SOLVER.stage2_frame_aware_curvature_threshold_enabled(
                SimpleNamespace(
                    finite_build=True,
                    finitebuild_frame_aware_curvature_threshold=False,
                )
            )
        )

    def test_opt_in_off_is_byte_identical(self) -> None:
        fb = _settings()
        threshold, pack_limit, applied = _SOLVER.stage2_frame_aware_curvature_tightening(
            100.0, fb, False
        )
        self.assertEqual(threshold, 100.0)
        self.assertIsNone(pack_limit)
        self.assertFalse(applied)

    def test_opt_in_tightens_when_pack_limit_stricter(self) -> None:
        fb = _settings()
        expected = finite_build_frame_aware_curvature_limit_inv_m(
            fb, TYPE_KK_INNER_RADIUS_MARGIN_M
        )
        threshold, pack_limit, applied = _SOLVER.stage2_frame_aware_curvature_tightening(
            100.0, fb, True
        )
        self.assertTrue(applied)
        self.assertAlmostEqual(threshold, expected, places=12)
        self.assertAlmostEqual(pack_limit, expected, places=12)

    def test_opt_in_never_loosens_an_already_stricter_threshold(self) -> None:
        fb = _settings()
        expected_limit = finite_build_frame_aware_curvature_limit_inv_m(
            fb, TYPE_KK_INNER_RADIUS_MARGIN_M
        )
        # 5.0 m^-1 (radius 0.2 m) is already stricter than the pack limit.
        threshold, pack_limit, applied = _SOLVER.stage2_frame_aware_curvature_tightening(
            5.0, fb, True
        )
        self.assertFalse(applied)
        self.assertEqual(threshold, 5.0)
        self.assertAlmostEqual(pack_limit, expected_limit, places=12)

    def test_opt_in_without_finite_build_raises_loudly(self) -> None:
        with self.assertRaises(ValueError):
            _SOLVER.stage2_frame_aware_curvature_tightening(100.0, None, True)

    def test_off_without_finite_build_is_identity(self) -> None:
        threshold, pack_limit, applied = _SOLVER.stage2_frame_aware_curvature_tightening(
            100.0, None, False
        )
        self.assertEqual(threshold, 100.0)
        self.assertIsNone(pack_limit)
        self.assertFalse(applied)

    def test_finite_build_cs_clearance_threshold_is_used_for_hardware_status(self) -> None:
        source = SOLVER_PATH.read_text(encoding="utf-8")
        self.assertNotIn("coil_surface_threshold=CS_THRESHOLD", source)
        self.assertIn("coil_surface_threshold=cs_clearance_threshold", source)

        nominal_threshold = 0.010
        finite_build_threshold = 0.01813
        measured_clearance = 0.012
        nominal_status = _SOLVER._evaluate_stage2_hardware_constraints(
            coil_length=1.0,
            length_target=2.0,
            curve_curve_min_dist=0.050,
            cc_threshold=0.0462,
            max_curvature=20.0,
            curvature_threshold=40.0,
            curve_surface_min_dist=measured_clearance,
            coil_surface_threshold=nominal_threshold,
        )
        finite_build_status = _SOLVER._evaluate_stage2_hardware_constraints(
            coil_length=1.0,
            length_target=2.0,
            curve_curve_min_dist=0.050,
            cc_threshold=0.0462,
            max_curvature=20.0,
            curvature_threshold=40.0,
            curve_surface_min_dist=measured_clearance,
            coil_surface_threshold=finite_build_threshold,
        )

        self.assertTrue(nominal_status["success"])
        self.assertFalse(finite_build_status["success"])
        self.assertIn("coil_surface_spacing", finite_build_status["constraints"])


class ProjectedBendHalfExtentTest(unittest.TestCase):
    def test_non_surface_frame_uses_conservative_outer_corner_reach(self) -> None:
        fb = _settings(frame="centroid")
        curve = _StubGamma([[1.0, 0.0, 0.0]])
        # centroid frame has no surface normal -> conservative outer-channel corner
        # reach for every quadpoint. Provide kappa via a small stub.
        curve.kappa = lambda: np.array([50.0])  # type: ignore[attr-defined]
        extent = _SOLVER._finite_build_projected_bend_half_extent_m(fb, curve)
        outer_corner_reach = float(
            np.hypot(
                TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
                TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
            )
        )
        self.assertTrue(np.allclose(extent, outer_corner_reach))

    def test_surface_tangent_projection_never_exceeds_outer_corner_reach(self) -> None:
        fb = _settings(frame="surface_tangent")
        family = _build_cws_banana_family(BANANA_WINDING_SURFACE_MAJOR_RADIUS_M)
        curve = family[0].curve
        extent = _SOLVER._finite_build_projected_bend_half_extent_m(fb, curve)
        self.assertEqual(extent.shape[0], np.asarray(curve.kappa()).shape[0])
        self.assertTrue(np.all(extent >= 0.0))
        # a*half_n + b*half_b with a^2+b^2=1, a,b>=0 -> bounded by the outer-channel
        # corner reach hypot(half_depth_normal, half_width_binormal).
        outer_corner_reach = float(
            np.hypot(
                TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
                TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
            )
        )
        self.assertTrue(np.all(extent <= outer_corner_reach + 1e-9))


# ───────────────────────── 5b ─────────────────────────


class SegmentSegmentExactnessTest(unittest.TestCase):
    def test_segment_cc_minimum_below_point_cloud_on_near_miss(self) -> None:
        # Curve A: x-axis chord [-1, 1]; Curve B: vertical chord at (0, 0.5).
        # True closest approach is at the un-sampled midpoints (distance 0.5),
        # while the nearest sampled endpoints are 1.5 apart.
        gamma_a = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        gamma_b = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, -1.0]])
        point_cloud = _point_cloud_min(gamma_a, gamma_b)
        seg = curve_curve_min_distance_segments_m(
            [_StubGamma(gamma_a), _StubGamma(gamma_b)]
        )
        self.assertAlmostEqual(point_cloud, 1.5, places=12)
        self.assertAlmostEqual(seg, 0.5, places=9)
        self.assertLess(seg, point_cloud)

    def test_segment_cs_minimum_below_point_cloud_on_near_miss(self) -> None:
        gamma_a = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        surface = np.array([[0.0, 0.5, 0.0]])
        point_cloud = _point_cloud_min(gamma_a, surface)
        seg = curve_surface_min_distance_segments_m([_StubGamma(gamma_a)], surface)
        self.assertAlmostEqual(seg, 0.5, places=9)
        self.assertLess(seg, point_cloud)

    def test_segment_kernel_matches_brute_force(self) -> None:
        # Independent brute-force min over a dense sampling of two skew segments.
        p1, p2 = np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
        q1, q2 = np.array([0.3, 0.2, 0.4]), np.array([0.3, 0.2, -0.6])
        ts = np.linspace(0.0, 1.0, 4001)
        pa = p1[None, :] + ts[:, None] * (p2 - p1)[None, :]
        qb = q1[None, :] + ts[:, None] * (q2 - q1)[None, :]
        brute = float(np.min(np.linalg.norm(pa[:, None, :] - qb[None, :, :], axis=2)))
        kernel = segment_segment_distance(p1, p2, q1, q2)
        self.assertAlmostEqual(kernel, brute, places=4)

    def test_closed_polyline_segments_rejects_bad_shape(self) -> None:
        with self.assertRaises(ValueError):
            closed_polyline_segments(np.zeros((1, 3)))  # need N>=2
        with self.assertRaises(ValueError):
            closed_polyline_segments(np.zeros((4, 2)))  # need 3 columns

    def test_curve_surface_rejects_empty_point_cloud(self) -> None:
        with self.assertRaises(ValueError):
            curve_surface_min_distance_segments_m(
                [_StubGamma(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))],
                np.zeros((0, 3)),
            )

    def test_artifact_fields_are_additive_with_method_labels(self) -> None:
        gamma_a = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        gamma_b = np.array([[0.0, 0.5, 1.0], [0.0, 0.5, -1.0]])
        surf = _StubGamma(np.array([[0.0, 0.5, 0.0]]))
        fields = _SOLVER._segment_exact_clearance_artifact_fields(
            [_StubGamma(gamma_a), _StubGamma(gamma_b)], surf
        )
        self.assertEqual(
            set(fields),
            {
                "CURVE_CURVE_MIN_DIST_SEGMENT_EXACT",
                "CURVE_CURVE_MIN_DIST_SEGMENT_EXACT_METHOD",
                "CURVE_SURFACE_MIN_DIST_SEGMENT_EXACT",
                "CURVE_SURFACE_MIN_DIST_SEGMENT_EXACT_METHOD",
            },
        )
        self.assertAlmostEqual(
            fields["CURVE_CURVE_MIN_DIST_SEGMENT_EXACT"], 0.5, places=9
        )
        # The surface sample (0, 0.5, 0) lies exactly on curve B's chord, so the
        # exact segment-to-surface minimum is 0 (and < the point-cloud minimum,
        # since no curve sample sits on it): the additive cs key tracks the
        # true between-sample approach.
        self.assertAlmostEqual(
            fields["CURVE_SURFACE_MIN_DIST_SEGMENT_EXACT"], 0.0, places=9
        )
        self.assertLess(
            fields["CURVE_SURFACE_MIN_DIST_SEGMENT_EXACT"],
            _point_cloud_min(
                np.vstack([gamma_a, gamma_b]), np.array([[0.0, 0.5, 0.0]])
            ),
        )
        self.assertEqual(
            fields["CURVE_CURVE_MIN_DIST_SEGMENT_EXACT_METHOD"],
            "closed_chord_polyline_segment_segment",
        )


# ───────────────────────── 5c ─────────────────────────


class VesselKeepoutCliGateTest(unittest.TestCase):
    def test_negative_weight_raises_loudly(self) -> None:
        with self.assertRaises(ValueError):
            _SOLVER.validate_stage2_vessel_keepout_cli_args(
                SimpleNamespace(
                    stage2_vessel_keepout_weight=-1.0,
                    stage2_available_envelope_reward_weight=0.0,
                )
            )
        with self.assertRaises(ValueError):
            _SOLVER.validate_stage2_vessel_keepout_cli_args(
                SimpleNamespace(
                    stage2_vessel_keepout_weight=0.0,
                    stage2_available_envelope_reward_weight=-1.0,
                )
            )

    def test_zero_and_positive_weight_accepted(self) -> None:
        _SOLVER.validate_stage2_vessel_keepout_cli_args(
            SimpleNamespace(
                stage2_vessel_keepout_weight=0.0,
                stage2_available_envelope_reward_weight=0.0,
            )
        )
        _SOLVER.validate_stage2_vessel_keepout_cli_args(
            SimpleNamespace(
                stage2_vessel_keepout_weight=2.5,
                stage2_available_envelope_reward_weight=1.5,
            )
        )


class VesselKeepoutWindingFrameTest(unittest.TestCase):
    def test_realized_winding_torus_orients_the_term(self) -> None:
        realized_r0 = 0.920
        family = _build_cws_banana_family(realized_r0)
        # The solver resolves winding_r0 from the realized CWS torus, not 0.903.
        resolved = realized_cws_winding_radii(family)
        self.assertIsNotNone(resolved)
        self.assertAlmostEqual(resolved[0], realized_r0, places=12)
        term = CurveVesselEnvelopeKeepout(
            [coil.curve for coil in family], winding_r0=resolved[0]
        )
        self.assertAlmostEqual(term.winding_r0, realized_r0, places=12)
        self.assertNotAlmostEqual(
            term.winding_r0, BANANA_WINDING_SURFACE_MAJOR_RADIUS_M, places=6
        )

    def test_clear_coils_score_zero(self) -> None:
        # A banana family well inside the vessel scores exactly 0, so adding
        # weight * Jvessel is the identity when the lineage clears the wall
        # (the additive-zero baseline behind the default-OFF wiring). A small
        # loop centered on the vessel major radius clears the wall by a wide
        # margin.
        family = _build_cws_banana_family(
            VACUUM_VESSEL_MAJOR_RADIUS_M, minor_radius=0.05
        )
        term = CurveVesselEnvelopeKeepout(
            [coil.curve for coil in family],
            winding_r0=VACUUM_VESSEL_MAJOR_RADIUS_M,
        )
        self.assertGreater(term.shortest_clearance(), term.minimum_clearance)
        self.assertEqual(term.J(), 0.0)


if __name__ == "__main__":
    unittest.main()
