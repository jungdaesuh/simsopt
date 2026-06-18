"""Tests for the Phase 2/3 pack-rotation coupling levers (T3.2 G2/G3).

Covers the new Stage-2 CLI contract (`validate_finite_build_cli_args`) and the
core G2 behavior: feeding the SHARED finite-build pack frame to the fold /
geodesic-curvature penalty (and to the torsional-strain regularizer) makes those
objectives actually depend on the pack twist ``alpha(theta)`` — whereas the
legacy fresh-``ZeroRotation`` fold frame is decoupled.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from STAGE_2 import banana_coil_solver as solver  # noqa: E402
from banana_opt.fold_buildability import CurveSurfaceGeodesicCurvature  # noqa: E402
from simsopt.geo import (  # noqa: E402
    CurveXYZFourier,
    FrameRotation,
    FramedCurveSurfaceTangent,
    ZeroRotation,
    create_multifilament_grid,
)
from simsopt.geo.strain_optimization import LPTorsionalStrainPenalty  # noqa: E402

WINDING_R0 = 0.903


def _finite_build_args(**overrides):
    """A minimal args namespace that passes validate_finite_build_cli_args."""
    base = dict(
        finite_build=True,
        finitebuild_frame_aware_curvature_threshold=None,
        stage2_bs_path=None,
        finitebuild_numfilaments_n=2,
        finitebuild_numfilaments_b=7,
        finitebuild_gapsize_n=0.0099,
        finitebuild_gapsize_b=0.0066,
        finitebuild_pin_current=False,
        constraint_method="penalty",
        stage2_couple_pack_rotation_to_fold=False,
        fold_material_binormal_curvature_limit=None,
        stage2_pack_twist_strain_weight=0.0,
        stage2_rotation_aware_curvature_cap=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _poloidal_circle(npoints=48, radius=0.1):
    curve = CurveXYZFourier(npoints, 1)
    dofs = {name: 0.0 for name in curve.local_dof_names}
    dofs["xc(0)"] = WINDING_R0
    dofs["xc(1)"] = radius
    dofs["zs(1)"] = radius
    curve.x = np.array([dofs[name] for name in curve.local_dof_names])
    return curve


def _shared_pack_frame(curve, rotation_order=1):
    """The single shared FramedCurveSurfaceTangent the finite-build pack rides."""
    filaments = create_multifilament_grid(
        curve, 2, 7, 0.0099, 0.0066,
        rotation_order=rotation_order, frame="surface_tangent",
        surface_major_radius=WINDING_R0,
    )
    return filaments[0].framedcurve


class ValidateCliArgsTest(unittest.TestCase):
    def test_couple_without_finite_build_raises(self):
        args = _finite_build_args(
            finite_build=False, stage2_couple_pack_rotation_to_fold=True
        )
        with self.assertRaisesRegex(ValueError, "requires --finite-build"):
            solver.validate_finite_build_cli_args(args)

    def test_strain_weight_without_finite_build_raises(self):
        args = _finite_build_args(
            finite_build=False, stage2_pack_twist_strain_weight=1.0
        )
        with self.assertRaisesRegex(ValueError, "requires --finite-build"):
            solver.validate_finite_build_cli_args(args)

    def test_cap_without_finite_build_raises(self):
        args = _finite_build_args(
            finite_build=False, stage2_rotation_aware_curvature_cap=True
        )
        with self.assertRaisesRegex(ValueError, "requires --finite-build"):
            solver.validate_finite_build_cli_args(args)

    def test_cap_without_couple_raises(self):
        args = _finite_build_args(
            stage2_rotation_aware_curvature_cap=True,
            stage2_couple_pack_rotation_to_fold=False,
        )
        with self.assertRaisesRegex(
            ValueError, "requires --stage2-couple-pack-rotation-to-fold"
        ):
            solver.validate_finite_build_cli_args(args)

    def test_cap_with_couple_passes(self):
        args = _finite_build_args(
            stage2_rotation_aware_curvature_cap=True,
            stage2_couple_pack_rotation_to_fold=True,
        )
        solver.validate_finite_build_cli_args(args)  # must not raise

    def test_material_binormal_fold_limit_must_be_positive_when_set(self):
        args = _finite_build_args(fold_material_binormal_curvature_limit=0.0)
        with self.assertRaisesRegex(
            ValueError, "--fold-material-binormal-curvature-limit"
        ):
            solver.validate_stage2_buildability_objective_cli_args(args)

    def test_all_levers_off_passes(self):
        solver.validate_finite_build_cli_args(_finite_build_args())  # must not raise


class FoldCouplingTest(unittest.TestCase):
    def setUp(self):
        self.curve = _poloidal_circle()
        self.shared_frame = _shared_pack_frame(self.curve)
        self.rotation = self.shared_frame.rotation
        self.assertIsInstance(self.rotation, FrameRotation)

    def test_coupled_fold_depends_on_rotation_dofs(self):
        coupled = CurveSurfaceGeodesicCurvature(self.shared_frame, p=2, threshold=0.0)
        decoupled = CurveSurfaceGeodesicCurvature(
            FramedCurveSurfaceTangent(self.curve, WINDING_R0, 0.0), p=2, threshold=0.0
        )
        rot_dofs = set(self.rotation.dof_names)
        self.assertTrue(rot_dofs)  # the FrameRotation exposes dofs
        # Coupled fold carries the pack-rotation dofs; the legacy ZeroRotation
        # fold frame does not.
        self.assertTrue(rot_dofs <= set(coupled.dof_names))
        self.assertTrue(rot_dofs.isdisjoint(set(decoupled.dof_names)))

    def test_coupled_fold_value_responds_to_twist(self):
        coupled = CurveSurfaceGeodesicCurvature(self.shared_frame, p=2, threshold=0.0)
        x0 = np.asarray(self.rotation.x, dtype=float).copy()
        j0 = coupled.J()
        twisted = x0.copy()
        twisted[0] = x0[0] + 0.05
        self.rotation.x = twisted
        j1 = coupled.J()
        self.rotation.x = x0
        self.assertFalse(np.isclose(j0, j1))

    def test_decoupled_fold_is_byte_identical_baseline(self):
        # The legacy fold frame is a fresh ZeroRotation frame -> no rotation dofs,
        # so the off-path objective term is unchanged by this work.
        decoupled_frame = FramedCurveSurfaceTangent(self.curve, WINDING_R0, 0.0)
        self.assertIsInstance(decoupled_frame.rotation, ZeroRotation)
        decoupled = CurveSurfaceGeodesicCurvature(decoupled_frame, p=2, threshold=0.0)
        self.assertEqual(decoupled.dof_size, self.curve.dof_size)


class TorsionalStrainRegularizerTest(unittest.TestCase):
    def setUp(self):
        self.curve = _poloidal_circle()
        self.shared_frame = _shared_pack_frame(self.curve)
        self.rotation = self.shared_frame.rotation

    def test_strain_penalty_depends_on_rotation_dofs(self):
        jtwist = LPTorsionalStrainPenalty(self.shared_frame, width=0.04, p=2)
        rot_dofs = set(self.rotation.dof_names)
        self.assertTrue(rot_dofs <= set(jtwist.dof_names))

    def test_strain_penalty_responds_to_twist(self):
        jtwist = LPTorsionalStrainPenalty(self.shared_frame, width=0.04, p=2)
        x0 = np.asarray(self.rotation.x, dtype=float).copy()
        j0 = jtwist.J()
        twisted = x0.copy()
        twisted[-1] = x0[-1] + 0.1  # excite a non-constant (theta-dependent) mode
        self.rotation.x = twisted
        j1 = jtwist.J()
        self.rotation.x = x0
        # A non-constant twist profile has nonzero torsional strain.
        self.assertGreater(j1, j0)


if __name__ == "__main__":
    unittest.main()
