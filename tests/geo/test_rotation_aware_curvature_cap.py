"""Unit tests for the realized rotation-aware curvature measurement (T3.2 / G1).

These cover the pure helpers added to ``banana_opt.stage2_geometry``:
``rotation_aware_projected_half_extent_m`` and ``rotation_aware_curvature_report``.
They are frame-independent: the pack twist ``alpha`` is scanned and the realized
projected reach must sweep the full physical band ``[min(half_n, half_b),
hypot(half_n, half_b)]`` -- i.e. the curvature cap sweeps ``[32.77, 66.88]/m`` for
the Type-KK 2x7 pack -- and the rotation-aware cap is never below the conservative
worst-case corner cap.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt.stage2_geometry import (  # noqa: E402
    FiniteBuildSettings,
    finite_build_frame_aware_curvature_limit_inv_m,
    finite_build_rotation_aware_curvature_limit_inv_m,
    rotation_aware_curvature_report,
    rotation_aware_projected_half_extent_m,
)
from simsopt.geo import (  # noqa: E402
    CurveXYZFourier,
    FrameRotation,
    FramedCurveSurfaceTangent,
    ZeroRotation,
)

# Type-KK 2x7 pack geometry (matches hardware_contracts Type-KK conductor pack).
WINDING_R0 = 0.903
FLOOR_M = 0.01  # TYPE_KK_SINGLE_FILAMENT_MIN_BEND_RADIUS_M = 1/100
GAPSIZE_N = 0.390 * 0.0254  # conductor depth (normal), inches -> m
GAPSIZE_B = (1.568 * 0.0254) / (7 - 1)  # conductor width / (nfil_b - 1), m


def _type_kk_settings(rotation_order=1):
    return FiniteBuildSettings(
        numfilaments_n=2,
        numfilaments_b=7,
        gapsize_n=GAPSIZE_N,
        gapsize_b=GAPSIZE_B,
        rotation_order=rotation_order,
        frame="surface_tangent",
    )


def _poloidal_circle(npoints=64, radius=0.1):
    """A poloidal loop near the winding torus: constant |kappa| = 1/radius, and a
    surface-tangent frame whose normal rotates around the loop, so scanning the
    pack twist sweeps the bend angle through a full period."""
    curve = CurveXYZFourier(npoints, 1)
    dofs = {name: 0.0 for name in curve.local_dof_names}
    # gamma(phi) = (R0 + r cos phi, 0, r sin phi)
    dofs["xc(0)"] = WINDING_R0
    dofs["xc(1)"] = radius
    dofs["zs(1)"] = radius
    curve.x = np.array([dofs[name] for name in curve.local_dof_names])
    return curve


def _set_constant_alpha(rotation, alpha):
    """Set a constant pack twist alpha(theta) == alpha via the 0th Fourier dof."""
    x = np.zeros_like(np.asarray(rotation.x, dtype=float))
    x[0] = alpha / rotation.scale
    rotation.x = x
    return rotation


class RotationAwareProjectedReachTest(unittest.TestCase):
    def setUp(self):
        self.fb = _type_kk_settings()
        self.half_n = self.fb.pack_half_extent_n_m
        self.half_b = self.fb.pack_half_extent_b_m
        self.reach_min = min(self.half_n, self.half_b)
        self.reach_max = float(np.hypot(self.half_n, self.half_b))

    def test_type_kk_scalar_caps_match_expected(self):
        # Sanity-pin the conservative (worst corner) and best (narrow-aligned) caps.
        conservative = finite_build_frame_aware_curvature_limit_inv_m(self.fb, FLOOR_M)
        narrow = finite_build_rotation_aware_curvature_limit_inv_m(
            self.fb, FLOOR_M, bend_angle_rad=0.0  # bend along the narrow (normal) axis
        )
        self.assertAlmostEqual(conservative, 32.77, places=1)
        self.assertAlmostEqual(narrow, 66.88, places=1)

    def test_constant_alpha_helper_is_constant(self):
        # Guards the 0th-dof-is-constant assumption used to set a known twist.
        curve = _poloidal_circle()
        rotation = FrameRotation(curve.quadpoints, 1)
        _set_constant_alpha(rotation, 0.37)
        alpha = np.asarray(rotation.alpha(curve.quadpoints), dtype=float)
        self.assertTrue(np.allclose(alpha, 0.37, atol=1e-9))

    def test_reach_stays_within_physical_band_for_any_twist(self):
        curve = _poloidal_circle()
        for alpha in np.linspace(0.0, np.pi, 13):
            rotation = FrameRotation(curve.quadpoints, 1)
            _set_constant_alpha(rotation, alpha)
            fc = FramedCurveSurfaceTangent(curve, WINDING_R0, 0.0, rotation)
            reach = rotation_aware_projected_half_extent_m(self.fb, curve, fc)
            self.assertGreaterEqual(float(np.min(reach)), self.reach_min - 1e-9)
            self.assertLessEqual(float(np.max(reach)), self.reach_max + 1e-9)

    def test_twist_scan_sweeps_full_reach_band(self):
        # Across the alpha scan the realized reach must reach both the narrow-axis
        # minimum and the worst-case corner maximum -> the cap spans 32.77..66.88.
        curve = _poloidal_circle()
        global_min = np.inf
        global_max = -np.inf
        for alpha in np.linspace(0.0, np.pi, 73):
            rotation = FrameRotation(curve.quadpoints, 1)
            _set_constant_alpha(rotation, alpha)
            fc = FramedCurveSurfaceTangent(curve, WINDING_R0, 0.0, rotation)
            reach = rotation_aware_projected_half_extent_m(self.fb, curve, fc)
            global_min = min(global_min, float(np.min(reach)))
            global_max = max(global_max, float(np.max(reach)))
        self.assertAlmostEqual(global_min, self.reach_min, places=4)
        self.assertAlmostEqual(global_max, self.reach_max, places=4)

    def test_zero_rotation_matches_frame_rotation_at_alpha_zero(self):
        curve = _poloidal_circle()
        fc_zero = FramedCurveSurfaceTangent(curve, WINDING_R0, 0.0, ZeroRotation(curve.quadpoints))
        rotation = FrameRotation(curve.quadpoints, 1)
        _set_constant_alpha(rotation, 0.0)
        fc_rot = FramedCurveSurfaceTangent(curve, WINDING_R0, 0.0, rotation)
        reach_zero = rotation_aware_projected_half_extent_m(self.fb, curve, fc_zero)
        reach_rot = rotation_aware_projected_half_extent_m(self.fb, curve, fc_rot)
        self.assertTrue(np.allclose(reach_zero, reach_rot, atol=1e-12))


class RotationAwareReportTest(unittest.TestCase):
    def setUp(self):
        self.fb = _type_kk_settings()

    def _report(self, alpha, radius=0.1):
        curve = _poloidal_circle(radius=radius)
        rotation = FrameRotation(curve.quadpoints, 1)
        _set_constant_alpha(rotation, alpha)
        fc = FramedCurveSurfaceTangent(curve, WINDING_R0, 0.0, rotation)
        return rotation_aware_curvature_report(self.fb, curve, fc, FLOOR_M)

    def test_report_keys_and_invariants(self):
        report = self._report(alpha=0.3)
        for key in (
            "FINITEBUILD_ROTATION_AWARE_CAP_MIN_INV_M",
            "FINITEBUILD_ROTATION_AWARE_CAP_MEAN_INV_M",
            "FINITEBUILD_CONSERVATIVE_CORNER_CAP_INV_M",
            "FINITEBUILD_ROTATION_AWARE_HEADROOM_ARCLEN_FRACTION",
            "FINITEBUILD_ROTATION_AWARE_RESIDUAL_INFEASIBLE_FRACTION",
        ):
            self.assertIn(key, report)
        conservative = report["FINITEBUILD_CONSERVATIVE_CORNER_CAP_INV_M"]
        cap_min = report["FINITEBUILD_ROTATION_AWARE_CAP_MIN_INV_M"]
        cap_mean = report["FINITEBUILD_ROTATION_AWARE_CAP_MEAN_INV_M"]
        self.assertAlmostEqual(conservative, 32.77, places=1)
        # The rotation-aware cap is never tighter than the conservative corner cap
        # and never looser than the narrow-axis best case (~66.88/m).
        self.assertGreaterEqual(cap_min, conservative - 1e-6)
        self.assertLessEqual(cap_mean, 66.88 + 1e-6)
        self.assertGreaterEqual(cap_mean, cap_min - 1e-9)

    def test_fractions_are_well_formed(self):
        report = self._report(alpha=0.3)
        cashed = report["FINITEBUILD_ROTATION_AWARE_HEADROOM_ARCLEN_FRACTION"]
        residual = report["FINITEBUILD_ROTATION_AWARE_RESIDUAL_INFEASIBLE_FRACTION"]
        for frac in (cashed, residual):
            self.assertGreaterEqual(frac, 0.0)
            self.assertLessEqual(frac, 1.0)

    def test_gentle_curve_has_no_residual_infeasible(self):
        # A loose loop (|kappa| = 1/0.3 ~ 3.3/m) is far below every cap, so no
        # arclength is over-cap: both cashed and residual fractions are zero.
        report = self._report(alpha=0.3, radius=0.3)
        self.assertEqual(
            report["FINITEBUILD_ROTATION_AWARE_RESIDUAL_INFEASIBLE_FRACTION"], 0.0
        )
        self.assertEqual(
            report["FINITEBUILD_ROTATION_AWARE_HEADROOM_ARCLEN_FRACTION"], 0.0
        )


if __name__ == "__main__":
    unittest.main()
