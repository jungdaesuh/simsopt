"""Unit tests for the 2026-06-11 formulation-audit fixes in the Stage 2 solver.

Covers the finite-build default-on and additive changes:

5a. ``--finitebuild-frame-aware-curvature-threshold``: drives finite-build
    optimizer curvature threshold with the conservative frame-aware
    winding-pack limit ``1 / (single-filament bend floor + pack corner reach)``
    instead of the centerline cap. Default-on for finite-build, never loosens,
    explicitly opt-outable, and incompatible with ``--filament-only`` when
    explicitly requested.
5b. Exact segment-based cc/cs minima recorded at artifact capture
    (``CURVE_CURVE_MIN_DIST_SEGMENT_EXACT`` /
    ``CURVE_SURFACE_MIN_DIST_SEGMENT_EXACT``): always <= the point-cloud
    minima on the same quadrature samples, proven here on constructed
    near-miss geometries whose closest approach lies between samples.
5c. ``--stage2-vessel-keepout-weight`` / ``--stage2-hardware-keepout-weight``:
    wire the single-stage-proven ``CurveVesselEnvelopeKeepout`` /
    ``CurveHardwareKeepout`` terms into Stage 2 default-ON at single-stage
    parity (weight ``STAGE2_{VESSEL,HARDWARE}_KEEPOUT_WEIGHT_DEFAULT``); an
    explicit 0 weight restores the legacy objective for reproduction.
    ``--stage2-available-envelope-reward-weight`` is default-OFF and can steer
    usable vessel-envelope fill without replacing the hard promotion gates.
"""

import math
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from STAGE_2 import banana_coil_solver as stage2_solver  # noqa: E402
from banana_opt.hardware_contracts import (  # noqa: E402
    BANANA_CC_OBJECTIVE_MARGIN_M,
    COIL_COIL_MIN_DIST_M,
    SINGLE_STAGE_HARDWARE_KEEPOUT_WEIGHT_DEFAULT,
    SINGLE_STAGE_VESSEL_KEEPOUT_WEIGHT_DEFAULT,
    STAGE2_HARDWARE_KEEPOUT_WEIGHT_DEFAULT,
    STAGE2_VESSEL_KEEPOUT_WEIGHT_DEFAULT,
    TYPE_KK_FINITE_BUILD_GAPSIZE_B_M,
    TYPE_KK_FINITE_BUILD_GAPSIZE_N_M,
    TYPE_KK_FINITE_BUILD_NUMFILAMENTS_B,
    TYPE_KK_FINITE_BUILD_NUMFILAMENTS_N,
    TYPE_KK_INNER_RADIUS_MARGIN_M,
    TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
    TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
    VACUUM_VESSEL_MAJOR_RADIUS_M,
    VACUUM_VESSEL_MINOR_RADIUS_M,
)
from banana_opt.hardware_keepout import CurveVesselEnvelopeKeepout  # noqa: E402
from banana_opt.stage2_geometry import (  # noqa: E402
    FiniteBuildSettings,
    closed_polyline_segments,
    curve_curve_min_distance_segments_m,
    curve_surface_min_distance_segments_m,
    finite_build_frame_aware_curvature_limit_inv_m,
    finite_build_rotation_aware_curvature_limit_inv_m,
    pack_projected_reach_m,
)
from simsopt.geo import CurveXYZFourier  # noqa: E402
from simsopt.geo import CurveCurveDistance, CurveLength  # noqa: E402


def _parse(argv_tail):
    argv = ["banana_coil_solver.py", *argv_tail]
    # Neutralize inherited env participating in the argparse defaults under test.
    env = {
        "STAGE2_VESSEL_KEEPOUT_WEIGHT": "0.0",
        "STAGE2_AVAILABLE_ENVELOPE_REWARD_WEIGHT": "0.0",
        "STAGE2_HARDWARE_SDF_FREE_SPACE_REWARD_WEIGHT": "0.0",
        "CURVATURE_THRESHOLD": "100.0",
    }
    with mock.patch.object(sys, "argv", argv), mock.patch.dict(
        "os.environ", env, clear=False
    ):
        return stage2_solver.parse_args()


def _type_kk_finite_build_settings():
    return FiniteBuildSettings(
        numfilaments_n=TYPE_KK_FINITE_BUILD_NUMFILAMENTS_N,
        numfilaments_b=TYPE_KK_FINITE_BUILD_NUMFILAMENTS_B,
        gapsize_n=TYPE_KK_FINITE_BUILD_GAPSIZE_N_M,
        gapsize_b=TYPE_KK_FINITE_BUILD_GAPSIZE_B_M,
        rotation_order=1,
        frame="surface_tangent",
    )


class _PolylineCurveStub:
    """Minimal curve double for the segment-distance helpers (gamma only)."""

    def __init__(self, points):
        self._points = np.asarray(points, dtype=float)

    def gamma(self):
        return self._points


class _SurfaceStub:
    """Minimal surface double exposing the (nphi, ntheta, 3) gamma grid."""

    def __init__(self, points):
        self._points = np.asarray(points, dtype=float).reshape((1, -1, 3))

    def gamma(self):
        return self._points


def _poloidal_circle_curve(center_major_radius_m, circle_radius_m, numquadpoints=64):
    """Closed circle in the phi=0 poloidal plane (x = R, y = 0, z vertical)."""
    curve = CurveXYZFourier(numquadpoints, order=1)
    curve.set("xc(0)", center_major_radius_m)
    curve.set("xc(1)", circle_radius_m)
    curve.set("zs(1)", circle_radius_m)
    return curve


class FrameAwareCurvatureThresholdTests(unittest.TestCase):
    """Audit 5a: frame-aware pack curvature limit drives the optimizer."""

    def test_pack_limit_matches_type_kk_constants(self):
        # Adopted self-intersection model: cap = 1/(inner-radius margin +
        # outer-channel edgewise reach max(half_depth, half_width)), independent
        # of the conductor-pack grid.
        settings = _type_kk_finite_build_settings()
        limit = finite_build_frame_aware_curvature_limit_inv_m(
            settings,
            TYPE_KK_INNER_RADIUS_MARGIN_M,
        )
        expected = 1.0 / (
            TYPE_KK_INNER_RADIUS_MARGIN_M
            + max(
                TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
                TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
            )
        )
        self.assertAlmostEqual(limit, expected, places=12)
        # External anchor: the conservative edgewise cap for the Type KK outer
        # channel is ~43.31 1/m, far below the 100 centerline cap.
        self.assertAlmostEqual(limit, 43.31, delta=0.01)
        self.assertLess(limit, 100.0)

    def test_pack_limit_is_conservative_for_every_bend_direction(self):
        # The per-point frame-aware bound uses the axis-interpolated projected
        # half-extent cos^2*half_n + sin^2*half_b for a bend direction over the
        # OUTER channel. That convex interpolation never exceeds the edgewise
        # reach max(half_n, half_b), so a curve at the conservative cap satisfies
        # every per-point bound.
        settings = _type_kk_finite_build_settings()
        limit = finite_build_frame_aware_curvature_limit_inv_m(
            settings,
            TYPE_KK_INNER_RADIUS_MARGIN_M,
        )
        half_n = TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M
        half_b = TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M
        rng = np.random.default_rng(20260611)
        angles = rng.uniform(0.0, 2.0 * np.pi, size=256)
        projected = np.cos(angles) ** 2 * half_n + np.sin(angles) ** 2 * half_b
        per_point_limits = 1.0 / (TYPE_KK_INNER_RADIUS_MARGIN_M + projected)
        self.assertLessEqual(
            limit,
            float(np.min(per_point_limits)) + 1e-12,
            "conservative cap must be at most the per-point frame-aware limit "
            "for every bend direction",
        )

    def test_degenerate_zero_radius_returns_inf(self):
        # The outer-channel reach is a fixed positive constant, so the required
        # radius is non-positive only when the margin cancels it; the helper
        # returns inf at that degenerate boundary.
        settings = _type_kk_finite_build_settings()
        outer_corner_reach = math.hypot(
            TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
            TYPE_KK_OUTER_CHANNEL_HALF_WIDTH_BINORMAL_M,
        )
        self.assertEqual(
            finite_build_frame_aware_curvature_limit_inv_m(
                settings, -outer_corner_reach
            ),
            float("inf"),
        )

    def test_projected_reach_endpoints_and_maximum_is_edgewise_axis(self):
        # T3.2: the axis-interpolated projected reach is half_n along the normal
        # (angle 0), half_b along the binormal (angle pi/2), and -- being a convex
        # interpolation cos^2*half_n + sin^2*half_b -- maxes out at the edgewise
        # axis reach max(half_n, half_b), with no stricter diagonal-corner reach.
        settings = _type_kk_finite_build_settings()
        half_n = settings.pack_half_extent_n_m
        half_b = settings.pack_half_extent_b_m
        self.assertAlmostEqual(pack_projected_reach_m(half_n, half_b, 0.0), half_n)
        self.assertAlmostEqual(
            pack_projected_reach_m(half_n, half_b, np.pi / 2.0), half_b
        )
        # The edgewise axis reach max(half_n, half_b) is attained at pi/2 (asserted
        # above) and -- by convexity -- bounds the reach for every other angle.
        edgewise_reach = max(half_n, half_b)
        angles = np.linspace(0.0, 2.0 * np.pi, 720)
        projected = [pack_projected_reach_m(half_n, half_b, a) for a in angles]
        self.assertLessEqual(float(np.max(projected)), edgewise_reach + 1e-12)

    def test_rotation_aware_limit_matches_conservative_at_edgewise_bend_angle(self):
        # At the edgewise (wide-axis) bend angle the rotation-aware cap reproduces
        # the conservative cap exactly (SSOT: same required-radius formula). The
        # conservative cap uses the edgewise reach max(half_n, half_b) = half_b for
        # the Type-KK outer channel, realized at the binormal axis (pi/2).
        settings = _type_kk_finite_build_settings()
        margin = TYPE_KK_INNER_RADIUS_MARGIN_M
        conservative = finite_build_frame_aware_curvature_limit_inv_m(settings, margin)
        edgewise_angle = np.pi / 2.0
        self.assertAlmostEqual(
            finite_build_rotation_aware_curvature_limit_inv_m(
                settings, margin, edgewise_angle
            ),
            conservative,
            places=12,
        )

    def test_rotation_aware_limit_lifts_cap_when_bend_favours_narrow_extent(self):
        # T3.2 cap-lift: aligning the bend plane with the thin (flatwise/normal)
        # outer-channel extent lifts the Type KK cap from ~43.31 to ~123.03.
        settings = _type_kk_finite_build_settings()
        margin = TYPE_KK_INNER_RADIUS_MARGIN_M
        conservative = finite_build_frame_aware_curvature_limit_inv_m(settings, margin)
        flatwise = finite_build_rotation_aware_curvature_limit_inv_m(
            settings, margin, 0.0
        )
        self.assertGreater(flatwise, conservative)
        self.assertAlmostEqual(flatwise, 123.03, delta=0.01)
        # Independent closed form: 1 / (margin + outer half_depth_normal).
        self.assertAlmostEqual(
            flatwise,
            1.0 / (margin + TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M),
            places=12,
        )

    def test_rotation_aware_limit_degenerate_zero_radius_returns_inf(self):
        # Required radius is non-positive only when the margin cancels the fixed
        # outer-channel reach; the helper returns inf at that boundary.
        settings = _type_kk_finite_build_settings()
        self.assertEqual(
            finite_build_rotation_aware_curvature_limit_inv_m(
                settings,
                -TYPE_KK_OUTER_CHANNEL_HALF_DEPTH_NORMAL_M,
                0.0,
            ),
            float("inf"),
        )

    def test_cli_default_auto_enables_for_finite_build(self):
        args = _parse([])
        self.assertIsNone(args.finitebuild_frame_aware_curvature_threshold)
        self.assertTrue(
            stage2_solver.stage2_frame_aware_curvature_threshold_enabled(args)
        )

    def test_type_kk_finite_build_is_default_and_filament_only_opts_out(self):
        default_args = _parse([])
        filament_only_args = _parse(["--filament-only"])
        self.assertTrue(default_args.finite_build)
        self.assertFalse(filament_only_args.finite_build)
        self.assertTrue(
            stage2_solver.stage2_frame_aware_curvature_threshold_enabled(default_args)
        )
        self.assertFalse(
            stage2_solver.stage2_frame_aware_curvature_threshold_enabled(
                filament_only_args
            )
        )

    def test_cli_flag_rejects_filament_only(self):
        with self.assertRaises(SystemExit):
            _parse(["--filament-only", "--finitebuild-frame-aware-curvature-threshold"])

    def test_cli_flag_parses_with_default_finite_build(self):
        args = _parse(["--finitebuild-frame-aware-curvature-threshold"])
        self.assertTrue(args.finitebuild_frame_aware_curvature_threshold)
        self.assertTrue(
            stage2_solver.stage2_frame_aware_curvature_threshold_enabled(args)
        )

    def test_cli_no_flag_disables_default_finite_build(self):
        args = _parse(["--no-finitebuild-frame-aware-curvature-threshold"])
        self.assertFalse(args.finitebuild_frame_aware_curvature_threshold)
        self.assertFalse(
            stage2_solver.stage2_frame_aware_curvature_threshold_enabled(args)
        )

    def test_tightening_resolver_tightens_and_never_loosens(self):
        settings = _type_kk_finite_build_settings()
        threshold, pack_limit, applied = (
            stage2_solver.stage2_frame_aware_curvature_tightening(
                100.0, settings, True
            )
        )
        self.assertTrue(applied)
        self.assertAlmostEqual(threshold, pack_limit, places=12)
        self.assertLess(threshold, 100.0)
        # A user threshold already stricter than the pack limit is kept.
        threshold, pack_limit, applied = (
            stage2_solver.stage2_frame_aware_curvature_tightening(
                20.0, settings, True
            )
        )
        self.assertFalse(applied)
        self.assertEqual(threshold, 20.0)
        self.assertLess(pack_limit, 100.0)

    def test_tightening_resolver_noops_only_when_opted_out(self):
        settings = _type_kk_finite_build_settings()
        self.assertEqual(
            stage2_solver.stage2_frame_aware_curvature_tightening(
                100.0, settings, False
            ),
            (100.0, None, False),
        )

    def test_tightening_resolver_raises_when_opted_in_without_finite_build(self):
        with self.assertRaises(ValueError):
            stage2_solver.stage2_frame_aware_curvature_tightening(100.0, None, True)


class CoilCoilObjectiveMarginTests(unittest.TestCase):
    """Plan C: keep the hard CC gate fixed while steering with a margin."""

    def test_cli_defaults_to_type_kk_objective_margin(self):
        args = _parse([])
        self.assertAlmostEqual(args.cc_objective_margin, BANANA_CC_OBJECTIVE_MARGIN_M)
        self.assertAlmostEqual(args.cc_threshold, COIL_COIL_MIN_DIST_M)

    def test_cli_accepts_nonnegative_objective_margin(self):
        args = _parse(["--cc-objective-margin", "0.003"])
        self.assertAlmostEqual(args.cc_objective_margin, 0.003)
        self.assertAlmostEqual(args.cc_threshold, COIL_COIL_MIN_DIST_M)

    def test_cli_rejects_negative_objective_margin(self):
        with self.assertRaises(SystemExit):
            _parse(["--cc-objective-margin", "-0.001"])


class SegmentExactDistanceTests(unittest.TestCase):
    """Audit 5b: exact segment minima vs point-cloud minima at capture."""

    def test_near_miss_pair_segment_min_below_point_cloud_min(self):
        # Two coarse rectangles whose long edges cross at a 0.05 m vertical
        # gap mid-segment; the sampled corners are >1.2 m apart, so the
        # point-cloud metric misses the real 0.05 m approach entirely.
        rect_a = _PolylineCurveStub(
            [
                (-1.0, -0.1, 0.0),
                (1.0, -0.1, 0.0),
                (1.0, 0.1, 0.0),
                (-1.0, 0.1, 0.0),
            ]
        )
        rect_b = _PolylineCurveStub(
            [
                (-0.1, -1.0, 0.05),
                (0.1, -1.0, 0.05),
                (0.1, 1.0, 0.05),
                (-0.1, 1.0, 0.05),
            ]
        )
        point_cloud_min = float(
            np.min(
                np.linalg.norm(
                    rect_a.gamma()[:, None, :] - rect_b.gamma()[None, :, :],
                    axis=2,
                )
            )
        )
        segment_min = curve_curve_min_distance_segments_m([rect_a, rect_b])
        self.assertLessEqual(segment_min, point_cloud_min)
        self.assertAlmostEqual(segment_min, 0.05, places=12)
        self.assertGreater(point_cloud_min, 1.0)

    def test_segment_min_leq_point_cloud_min_on_real_curves(self):
        # Two real simsopt circles with phase-offset quadratures: every
        # sampled point pair is farther than the true 0.3 m plane gap, the
        # chord polylines recover (nearly) the true gap, and the exact value
        # is never above the point-cloud value (the recorded metric).
        numquadpoints = 16
        circle_a = CurveXYZFourier(
            np.linspace(0.0, 1.0, numquadpoints, endpoint=False), order=1
        )
        circle_a.set("xc(1)", 1.0)
        circle_a.set("ys(1)", 1.0)
        circle_b = CurveXYZFourier(
            np.linspace(0.0, 1.0, numquadpoints, endpoint=False)
            + 0.5 / numquadpoints,
            order=1,
        )
        circle_b.set("xc(1)", 1.0)
        circle_b.set("ys(1)", 1.0)
        circle_b.set("zc(0)", 0.3)
        point_cloud_min = float(
            CurveCurveDistance(
                [circle_a, circle_b], minimum_distance=1.0
            ).shortest_distance()
        )
        segment_min = curve_curve_min_distance_segments_m([circle_a, circle_b])
        self.assertLessEqual(segment_min, point_cloud_min)
        self.assertGreater(point_cloud_min, 0.3)
        # Chord sagitta error at 16 samples on a unit circle is ~2e-2.
        self.assertAlmostEqual(segment_min, 0.3, delta=0.03)

    def test_curve_surface_segment_min_sees_between_sample_approach(self):
        square = _PolylineCurveStub(
            [
                (-1.0, -1.0, 0.0),
                (1.0, -1.0, 0.0),
                (1.0, 1.0, 0.0),
                (-1.0, 1.0, 0.0),
            ]
        )
        # One surface point hovering 0.02 m above the middle of the bottom
        # edge: nearest sampled curve point is ~1 m away.
        surface_points = np.array([[0.0, -1.0, 0.02], [5.0, 5.0, 5.0]])
        point_cloud_min = float(
            np.min(
                np.linalg.norm(
                    square.gamma()[:, None, :] - surface_points[None, :, :],
                    axis=2,
                )
            )
        )
        segment_min = curve_surface_min_distance_segments_m(
            [square], surface_points
        )
        self.assertLessEqual(segment_min, point_cloud_min)
        self.assertAlmostEqual(segment_min, 0.02, places=12)
        self.assertGreater(point_cloud_min, 0.9)

    def test_closed_polyline_includes_wraparound_chord(self):
        triangle = np.array(
            [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        )
        segments = closed_polyline_segments(triangle)
        self.assertEqual(segments.shape, (3, 2, 3))
        np.testing.assert_allclose(segments[-1, 0], triangle[-1])
        np.testing.assert_allclose(segments[-1, 1], triangle[0])

    def test_artifact_fields_keys_methods_and_consistency(self):
        rect_a = _PolylineCurveStub(
            [
                (-1.0, -0.1, 0.0),
                (1.0, -0.1, 0.0),
                (1.0, 0.1, 0.0),
                (-1.0, 0.1, 0.0),
            ]
        )
        rect_b = _PolylineCurveStub(
            [
                (-0.1, -1.0, 0.05),
                (0.1, -1.0, 0.05),
                (0.1, 1.0, 0.05),
                (-0.1, 1.0, 0.05),
            ]
        )
        surf = _SurfaceStub(np.array([[0.0, -1.0, 0.02]]))
        fields = stage2_solver._segment_exact_clearance_artifact_fields(
            [rect_a, rect_b], surf
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
            fields["CURVE_CURVE_MIN_DIST_SEGMENT_EXACT"], 0.05, places=12
        )
        # The surface point (0, -1, 0.02) is closest to the middle of
        # rect_b's bottom edge (y=-1, z=0.05): |0.05 - 0.02| = 0.03.
        self.assertAlmostEqual(
            fields["CURVE_SURFACE_MIN_DIST_SEGMENT_EXACT"], 0.03, places=12
        )
        self.assertEqual(
            fields["CURVE_CURVE_MIN_DIST_SEGMENT_EXACT_METHOD"],
            "closed_chord_polyline_segment_segment",
        )
        self.assertEqual(
            fields["CURVE_SURFACE_MIN_DIST_SEGMENT_EXACT_METHOD"],
            "closed_chord_polyline_segment_to_surface_point_cloud",
        )


class VesselKeepoutStage2Tests(unittest.TestCase):
    """Audit 5c: vessel + hardware keep-out terms default-ON at single-stage
    parity; an explicit weight 0 restores the legacy objective."""

    @staticmethod
    def _parse_without_keepout_env(argv_tail):
        # Surface the true argparse default literals by removing any inherited
        # keep-out weight env vars that would otherwise win over the default.
        argv = ["banana_coil_solver.py", *argv_tail]
        env = {"CURVATURE_THRESHOLD": "100.0"}
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            "os.environ", env, clear=False
        ):
            os.environ.pop("STAGE2_VESSEL_KEEPOUT_WEIGHT", None)
            os.environ.pop("STAGE2_HARDWARE_KEEPOUT_WEIGHT", None)
            return stage2_solver.parse_args()

    def test_cli_vessel_weight_defaults_to_single_stage_parity(self):
        args = self._parse_without_keepout_env([])
        # Parity claim: Stage-2's default IS the single-stage production weight.
        # Asserting against the single-stage SSOT (not the Stage-2 alias the
        # source default reads) catches an alias-to-wrong-constant re-point.
        self.assertEqual(
            args.stage2_vessel_keepout_weight,
            SINGLE_STAGE_VESSEL_KEEPOUT_WEIGHT_DEFAULT,
        )
        self.assertEqual(
            STAGE2_VESSEL_KEEPOUT_WEIGHT_DEFAULT,
            SINGLE_STAGE_VESSEL_KEEPOUT_WEIGHT_DEFAULT,
        )
        self.assertGreater(args.stage2_vessel_keepout_weight, 0.0)

    def test_cli_hardware_weight_defaults_to_single_stage_parity(self):
        args = self._parse_without_keepout_env([])
        self.assertEqual(
            args.stage2_hardware_keepout_weight,
            SINGLE_STAGE_HARDWARE_KEEPOUT_WEIGHT_DEFAULT,
        )
        self.assertEqual(
            STAGE2_HARDWARE_KEEPOUT_WEIGHT_DEFAULT,
            SINGLE_STAGE_HARDWARE_KEEPOUT_WEIGHT_DEFAULT,
        )
        self.assertGreater(args.stage2_hardware_keepout_weight, 0.0)
        self.assertEqual(
            args.stage2_hardware_keepout_glb,
            stage2_solver.DEFAULT_HARDWARE_KEEPOUT_GLB_PATH,
        )

    def test_cli_hardware_keepout_glb_parses_explicit_path(self):
        args = self._parse_without_keepout_env(
            ["--stage2-hardware-keepout-glb", "/tmp/hbt_assembly.glb"]
        )

        self.assertEqual(args.stage2_hardware_keepout_glb, "/tmp/hbt_assembly.glb")

    def test_cli_vessel_weight_env_override_to_zero_disables(self):
        # The legacy/byte-identical reproduction escape hatch: an explicit 0
        # weight (here via env) is honored over the default-on parity value.
        args = _parse([])  # _parse pins STAGE2_VESSEL_KEEPOUT_WEIGHT=0.0
        self.assertEqual(args.stage2_vessel_keepout_weight, 0.0)

    def test_cli_hardware_weight_env_override_to_zero_disables(self):
        # The hardware term carries the unique weight>0-requires-JSON failure
        # mode, so its disable path needs its own coverage: an explicit 0 weight
        # (via env) must escape that requirement, mirroring the vessel override.
        argv = ["banana_coil_solver.py"]
        env = {
            "STAGE2_HARDWARE_KEEPOUT_WEIGHT": "0.0",
            "CURVATURE_THRESHOLD": "100.0",
        }
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            "os.environ", env, clear=False
        ):
            args = stage2_solver.parse_args()
        self.assertEqual(args.stage2_hardware_keepout_weight, 0.0)

    def test_cli_weights_argv_zero_disables(self):
        # The documented `pass 0` argv disable route yields 0.0 for both terms,
        # overriding the default-on parity value regardless of inherited env.
        args = self._parse_without_keepout_env(
            [
                "--stage2-vessel-keepout-weight",
                "0",
                "--stage2-hardware-keepout-weight",
                "0",
            ]
        )
        self.assertEqual(args.stage2_vessel_keepout_weight, 0.0)
        self.assertEqual(args.stage2_hardware_keepout_weight, 0.0)

    def test_cli_hardware_keepout_alm_scale_default_none_and_parses(self):
        # The ALM-row hardness knob: defaults None (the schema scale applies),
        # and parses an explicit value when given.
        argv = ["banana_coil_solver.py"]
        env = {"CURVATURE_THRESHOLD": "100.0"}
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            "os.environ", env, clear=False
        ):
            os.environ.pop("STAGE2_HARDWARE_KEEPOUT_ALM_SCALE", None)
            default_args = stage2_solver.parse_args()
        self.assertIsNone(default_args.stage2_hardware_keepout_alm_scale)
        args = _parse(["--stage2-hardware-keepout-alm-scale", "3.0"])
        self.assertEqual(args.stage2_hardware_keepout_alm_scale, 3.0)

    def test_cli_hardware_keepout_tolerance_default_none_and_parses(self):
        # The ALM-row activity-tolerance knob: defaults None (the 1e-6 producer
        # default applies), and parses an explicit value when given.
        argv = ["banana_coil_solver.py"]
        env = {"CURVATURE_THRESHOLD": "100.0"}
        with mock.patch.object(sys, "argv", argv), mock.patch.dict(
            "os.environ", env, clear=False
        ):
            os.environ.pop("STAGE2_HARDWARE_KEEPOUT_TOLERANCE", None)
            default_args = stage2_solver.parse_args()
        self.assertIsNone(default_args.stage2_hardware_keepout_tolerance)
        args = _parse(["--stage2-hardware-keepout-tolerance", "5e-5"])
        self.assertEqual(args.stage2_hardware_keepout_tolerance, 5e-5)

    def test_cli_weight_rejects_negative(self):
        with self.assertRaises(SystemExit):
            _parse(["--stage2-vessel-keepout-weight", "-1.0"])
        with self.assertRaises(SystemExit):
            _parse(["--stage2-available-envelope-reward-weight", "-1.0"])
        with self.assertRaises(SystemExit):
            _parse(["--stage2-hardware-sdf-free-space-reward-weight", "-1.0"])

    def test_cli_weight_parses_positive(self):
        args = _parse(["--stage2-vessel-keepout-weight", "250.0"])
        self.assertEqual(args.stage2_vessel_keepout_weight, 250.0)
        args = _parse(["--stage2-available-envelope-reward-weight", "2.5"])
        self.assertEqual(args.stage2_available_envelope_reward_weight, 2.5)
        args = _parse(
            [
                "--stage2-hardware-keepout-backend",
                "sdf",
                "--stage2-hardware-keepout-sdf-manifest",
                "/tmp/hardware_sdf.json",
                "--stage2-hardware-sdf-free-space-reward-weight",
                "3.5",
            ]
        )
        self.assertEqual(args.stage2_hardware_sdf_free_space_reward_weight, 3.5)

    def test_cli_hardware_sdf_reward_requires_sdf_backend_and_manifest(self):
        with self.assertRaises(SystemExit):
            _parse(["--stage2-hardware-sdf-free-space-reward-weight", "1.0"])
        with self.assertRaises(SystemExit):
            _parse(
                [
                    "--stage2-hardware-keepout-backend",
                    "sdf",
                    "--stage2-hardware-sdf-free-space-reward-weight",
                    "1.0",
                ]
            )

    def test_term_zero_inside_vessel_positive_outside(self):
        # Compliant: poloidal circle on the winding axis, deep inside the
        # vessel tube (clearance >> the 5 mm margin) -> exactly zero penalty.
        inside = _poloidal_circle_curve(0.903, 0.05)
        term_inside = CurveVesselEnvelopeKeepout([inside])
        self.assertEqual(term_inside.J(), 0.0)
        self.assertGreater(term_inside.shortest_clearance(), 0.0)

        # Violating: circle centered well outboard so envelope corners poke
        # through the vessel wall at R0 + a.
        poking_center = VACUUM_VESSEL_MAJOR_RADIUS_M + VACUUM_VESSEL_MINOR_RADIUS_M
        outside = _poloidal_circle_curve(poking_center, 0.05)
        term_outside = CurveVesselEnvelopeKeepout([outside])
        self.assertGreater(term_outside.J(), 0.0)
        self.assertLess(term_outside.shortest_clearance(), 0.0)
        gradient = term_outside.dJ()
        self.assertGreater(float(np.max(np.abs(gradient))), 0.0)

    def test_weighted_term_composes_into_objective_sum(self):
        # The Stage 2 wiring adds `weight * Jvessel` to the existing objective
        # sums; verify the weighted Optimizable composition evaluates J and dJ.
        poking_center = VACUUM_VESSEL_MAJOR_RADIUS_M + VACUUM_VESSEL_MINOR_RADIUS_M
        curve = _poloidal_circle_curve(poking_center, 0.05)
        keepout = CurveVesselEnvelopeKeepout([curve])
        weight = 250.0
        objective = CurveLength(curve) + weight * keepout
        expected = float(CurveLength(curve).J()) + weight * float(keepout.J())
        self.assertAlmostEqual(float(objective.J()), expected, places=10)
        self.assertEqual(
            np.asarray(objective.dJ()).shape,
            np.asarray(CurveLength(curve).dJ()).shape,
        )


if __name__ == "__main__":
    unittest.main()
