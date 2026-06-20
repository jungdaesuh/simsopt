"""Tests for the single-stage B2 finite-build-field pack-rotation levers.

Mirrors the Stage-2 ``tests/geo/test_pack_rotation_coupling.py`` for the
single-stage entrypoint. Covers the new CLI contract
(``validate_single_stage_pack_rotation_field_cli_args`` /
``validate_single_stage_finite_build_cli_args``) and the core B2 behavior:

* the finite-build FIELD swap rebuilds the BiotSavart source from the
  multi-filament pack (coil count grows by the filament factor), preserves the
  NET banana current exactly (R1), and keeps the SHARED current state at one
  control current;
* the SHARED finite-build pack frame feeds the fold / geodesic-curvature penalty
  and the torsional-strain regularizer, so both depend on the pack twist
  ``alpha(theta)`` (whereas a fresh ``ZeroRotation`` frame is decoupled);
* the rotation-aware curvature cap relaxes the in-loop LpCurveCurvature steering
  threshold (when the pack twist favours the thin axis) without moving the
  conservative honest gate;
* with the field off (default) no pack rotation DOFs appear and no terms are
  constructed -- the objective graph is byte-identical to today.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
SINGLE_STAGE_ROOT = EXAMPLES_ROOT / "SINGLE_STAGE"
for _p in (str(EXAMPLES_ROOT), str(SINGLE_STAGE_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import single_stage_banana_example as sse  # noqa: E402
from banana_opt.fold_buildability import CurveSurfaceGeodesicCurvature  # noqa: E402
from banana_opt.hardware_contracts import (  # noqa: E402
    TYPE_KK_INNER_RADIUS_MARGIN_M,
)
from banana_opt.reference_surfaces import build_banana_reference_surfaces  # noqa: E402
from banana_opt.single_stage_banana_current_mode import (  # noqa: E402
    BANANA_CURRENT_MODE_SHARED,
    build_single_stage_banana_current_state,
)
from banana_opt.stage2_geometry import (  # noqa: E402
    FiniteBuildSettings,
    finite_build_frame_aware_curvature_limit_inv_m,
    rotation_aware_projected_half_extent_m,
)
from banana_opt.stage2_single_stage_handoff import Stage2CoilPartitions  # noqa: E402
from simsopt.field import (  # noqa: E402
    BiotSavart,
    Current,
    apply_symmetries_to_curves,
    coils_via_symmetries,
)
from simsopt.field.coil import Coil  # noqa: E402
from simsopt.geo import CurveCWSFourierCPP, CurveXYZFourier  # noqa: E402
from simsopt.geo.framedcurve import (  # noqa: E402
    FrameRotation,
    FramedCurveSurfaceTangent,
    ZeroRotation,
)
from simsopt.geo.strain_optimization import LPTorsionalStrainPenalty  # noqa: E402

NFP = 5
NET_BANANA_CURRENT_A = -16000.0
BANANA_SURF_RADIUS_M = 0.21
WINDING_R0 = 0.903


def _finite_build_args(**overrides):
    """A minimal args namespace passing the B2 dependency-chain validators."""
    base = dict(
        finite_build=True,
        single_stage_finite_build_field=False,
        single_stage_pack_rotation_fold_weight=0.0,
        single_stage_pack_twist_strain_weight=0.0,
        single_stage_rotation_aware_curvature_cap=False,
        single_stage_banana_current_mode="shared",
        finitebuild_numfilaments_n=2,
        finitebuild_numfilaments_b=7,
        finitebuild_gapsize_n=0.0099,
        finitebuild_gapsize_b=0.0066,
        finitebuild_frame_aware_curvature_threshold=None,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _winding_surface():
    return build_banana_reference_surfaces(
        NFP, BANANA_SURF_RADIUS_M
    ).coil_winding_surface


def _master_curve(surf_coils, order=4, num_quadpoints=96):
    curve = CurveCWSFourierCPP(
        np.linspace(0, 1, num_quadpoints, endpoint=False), order=order, surf=surf_coils
    )
    curve.set("phic(0)", 0.0)
    curve.set("thetac(0)", 0.0)
    curve.set("phic(1)", 0.05)
    curve.set("thetas(1)", 0.5)
    return curve


def _settings(rotation_order=1):
    return FiniteBuildSettings(
        numfilaments_n=2,
        numfilaments_b=7,
        gapsize_n=0.0099,
        gapsize_b=0.0066,
        rotation_order=rotation_order,
        frame="surface_tangent",
    )


def _thin_centerline_partitions(surf_coils, master_curve):
    """A Stage2CoilPartitions whose banana coils are the thin master symmetry
    images (one shared net Current), plus a couple of fixed TF field sources."""
    banana_coils = coils_via_symmetries(
        [master_curve],
        [Current(NET_BANANA_CURRENT_A)],
        surf_coils.nfp,
        surf_coils.stellsym,
    )
    tf_curve = CurveXYZFourier(32, 1)
    tf_dofs = {name: 0.0 for name in tf_curve.local_dof_names}
    tf_dofs["xc(0)"] = 1.0
    tf_dofs["xc(1)"] = 0.3
    tf_dofs["zs(1)"] = 0.3
    tf_curve.x = np.array([tf_dofs[name] for name in tf_curve.local_dof_names])
    tf_coils = [Coil(tf_curve, Current(-80000.0))]
    return Stage2CoilPartitions(
        tf_coils=tuple(tf_coils),
        banana_coils=tuple(banana_coils),
        proxy_coils=(),
        vf_coils=(),
        num_tf_coils=len(tf_coils),
        num_banana_coils=len(banana_coils),
        num_proxy_coils=0,
        num_vf_coils=0,
        finite_current_mode="vacuum",
    )


def _shared_pack_frame(master_curve, surf_coils, rotation_order=1):
    """The single shared FramedCurveSurfaceTangent the finite-build pack rides,
    recovered from the swapped pack (so it is exactly what the field uses)."""
    biot_savart = BiotSavart(
        coils_via_symmetries(
            [master_curve],
            [Current(NET_BANANA_CURRENT_A)],
            surf_coils.nfp,
            surf_coils.stellsym,
        )
    )
    partitions = Stage2CoilPartitions(
        tf_coils=(),
        banana_coils=tuple(biot_savart.coils),
        proxy_coils=(),
        vf_coils=(),
        num_tf_coils=0,
        num_banana_coils=len(biot_savart.coils),
        num_proxy_coils=0,
        num_vf_coils=0,
        finite_current_mode="vacuum",
    )
    _, pack_partitions, _, _ = sse.swap_single_stage_finite_build_field(
        biot_savart,
        partitions,
        surf_coils,
        finite_build_settings=_settings(rotation_order=rotation_order),
        free_pack_alpha=True,
    )
    return pack_partitions.banana_coils[0].curve.framedcurve


def _set_constant_alpha(rotation, alpha):
    x = np.zeros_like(np.asarray(rotation.x, dtype=float))
    x[0] = alpha / rotation.scale
    rotation.x = x
    return rotation


class ValidateCliArgsTest(unittest.TestCase):
    def test_field_without_finite_build_raises(self):
        args = _finite_build_args(finite_build=False, single_stage_finite_build_field=True)
        with self.assertRaisesRegex(ValueError, "requires --finite-build"):
            sse.validate_single_stage_finite_build_cli_args(args)

    def test_field_with_independent_current_mode_raises(self):
        args = _finite_build_args(
            single_stage_finite_build_field=True,
            single_stage_banana_current_mode="independent",
        )
        with self.assertRaisesRegex(ValueError, "current-mode shared"):
            sse.validate_single_stage_finite_build_cli_args(args)

    def test_fold_weight_without_field_raises(self):
        args = _finite_build_args(
            single_stage_finite_build_field=False,
            single_stage_pack_rotation_fold_weight=10.0,
        )
        with self.assertRaisesRegex(ValueError, "requires --single-stage-finite-build-field"):
            sse.validate_single_stage_finite_build_cli_args(args)

    def test_twist_weight_without_fold_raises(self):
        args = _finite_build_args(
            single_stage_finite_build_field=True,
            single_stage_pack_rotation_fold_weight=0.0,
            single_stage_pack_twist_strain_weight=1.0,
        )
        with self.assertRaisesRegex(ValueError, "requires --single-stage-pack-rotation-fold-weight"):
            sse.validate_single_stage_finite_build_cli_args(args)

    def test_cap_without_fold_raises(self):
        args = _finite_build_args(
            single_stage_finite_build_field=True,
            single_stage_pack_rotation_fold_weight=0.0,
            single_stage_rotation_aware_curvature_cap=True,
        )
        with self.assertRaisesRegex(ValueError, "requires --single-stage-pack-rotation-fold-weight"):
            sse.validate_single_stage_finite_build_cli_args(args)

    def test_field_with_fold_twist_cap_passes(self):
        args = _finite_build_args(
            single_stage_finite_build_field=True,
            single_stage_pack_rotation_fold_weight=10.0,
            single_stage_pack_twist_strain_weight=1.0,
            single_stage_rotation_aware_curvature_cap=True,
        )
        sse.validate_single_stage_finite_build_cli_args(args)  # must not raise

    def test_all_levers_off_passes(self):
        sse.validate_single_stage_finite_build_cli_args(_finite_build_args())

    def test_finite_build_off_passes(self):
        sse.validate_single_stage_finite_build_cli_args(
            _finite_build_args(finite_build=False)
        )


class FiniteBuildFieldSwapTest(unittest.TestCase):
    def setUp(self):
        self.surf_coils = _winding_surface()
        self.master = _master_curve(self.surf_coils)
        self.partitions = _thin_centerline_partitions(self.surf_coils, self.master)
        self.thin_bs = BiotSavart(
            [*self.partitions.tf_coils, *self.partitions.banana_coils]
        )
        self.symmetry_copies = NFP * (2 if self.surf_coils.stellsym else 1)

    def test_field_swap_grows_coils_by_filament_factor(self):
        settings = _settings()
        new_bs, new_partitions, _, _ = sse.swap_single_stage_finite_build_field(
            self.thin_bs,
            self.partitions,
            self.surf_coils,
            finite_build_settings=settings,
            free_pack_alpha=True,
        )
        expected_banana = settings.nfilaments * self.symmetry_copies
        self.assertEqual(len(new_partitions.banana_coils), expected_banana)
        self.assertEqual(
            len(new_bs.coils), len(self.partitions.tf_coils) + expected_banana
        )

    def test_field_swap_preserves_net_current_exactly(self):
        seed_net = float(self.partitions.banana_coils[0].current.get_value())
        _, _, _, net_current = sse.swap_single_stage_finite_build_field(
            self.thin_bs,
            self.partitions,
            self.surf_coils,
            finite_build_settings=_settings(),
            free_pack_alpha=True,
        )
        # R1: the shared NET optimizable recovers the seed net to ~1e-9.
        self.assertAlmostEqual(float(net_current.get_value()), seed_net, delta=1e-9)

    def test_shared_current_state_has_one_control(self):
        _, _, master_curve, net_current = sse.swap_single_stage_finite_build_field(
            self.thin_bs,
            self.partitions,
            self.surf_coils,
            finite_build_settings=_settings(),
            free_pack_alpha=True,
        )
        state = build_single_stage_banana_current_state(
            [Coil(master_curve, net_current)], mode=BANANA_CURRENT_MODE_SHARED
        )
        self.assertEqual(state.num_control_currents(), 1)
        seed_net = float(self.partitions.banana_coils[0].current.get_value())
        # R1: the recorded compatibility current is the NET, not the filament scale.
        self.assertAlmostEqual(state.compatibility_current_A(), seed_net, delta=1e-9)

    def test_free_alpha_adds_rotation_dofs_to_field(self):
        new_bs, new_partitions, _, _ = sse.swap_single_stage_finite_build_field(
            self.thin_bs,
            self.partitions,
            self.surf_coils,
            finite_build_settings=_settings(rotation_order=1),
            free_pack_alpha=True,
        )
        rotation = new_partitions.banana_coils[0].curve.framedcurve.rotation
        self.assertIsInstance(rotation, FrameRotation)
        rot_dofs = set(rotation.dof_names)
        self.assertTrue(rot_dofs)
        # R2: the freed alpha DOFs are referenced by the pack field (not dangling).
        self.assertTrue(rot_dofs <= set(new_bs.dof_names))

    def test_zero_alpha_leaves_no_rotation_dofs(self):
        new_bs, new_partitions, _, _ = sse.swap_single_stage_finite_build_field(
            self.thin_bs,
            self.partitions,
            self.surf_coils,
            finite_build_settings=_settings(rotation_order=1),
            free_pack_alpha=False,
        )
        rotation = new_partitions.banana_coils[0].curve.framedcurve.rotation
        # R2: no alpha consumer -> ZeroRotation, so the field has zero rotation DOFs.
        self.assertIsInstance(rotation, ZeroRotation)
        self.assertEqual(len(list(rotation.dof_names)), 0)
        self.assertFalse([n for n in new_bs.dof_names if "Rotation" in n])

    def test_master_symmetry_images_match_thin_family_count(self):
        # The per-banana objective curves are the MASTER symmetry images, not the
        # filaments, so their count equals the thin-centerline banana family.
        banana_curves = list(
            apply_symmetries_to_curves(
                [self.master], self.surf_coils.nfp, self.surf_coils.stellsym
            )
        )
        self.assertEqual(len(banana_curves), len(self.partitions.banana_coils))

    def test_pack_filaments_lack_second_derivative_master_provides_it(self):
        # The coil-force regularizer needs gammadashdash on the acted-on coil. Pack
        # filaments (CurveFilament) implement none, so the force term must act on the
        # master centerlines (which do) -- this is why the field-on force-target coils
        # are rebuilt from the master, not taken from the pack filaments.
        _, new_partitions, master_curve, net_current = (
            sse.swap_single_stage_finite_build_field(
                self.thin_bs,
                self.partitions,
                self.surf_coils,
                finite_build_settings=_settings(),
                free_pack_alpha=True,
            )
        )
        filament_curve = new_partitions.banana_coils[0].curve
        with self.assertRaises(RuntimeError):
            filament_curve.gammadashdash()
        force_target_coils = list(
            coils_via_symmetries(
                [master_curve], [net_current], self.surf_coils.nfp, self.surf_coils.stellsym
            )
        )
        # Master force-target coils evaluate the second derivative cleanly.
        self.assertEqual(
            np.asarray(force_target_coils[0].curve.gammadashdash()).shape[1], 3
        )


class FoldCouplingTest(unittest.TestCase):
    def setUp(self):
        self.surf_coils = _winding_surface()
        self.master = _master_curve(self.surf_coils)
        self.shared_frame = _shared_pack_frame(self.master, self.surf_coils)
        self.rotation = self.shared_frame.rotation
        self.assertIsInstance(self.rotation, FrameRotation)

    def test_coupled_fold_depends_on_rotation_dofs(self):
        coupled = CurveSurfaceGeodesicCurvature(self.shared_frame, p=2, threshold=0.0)
        decoupled = CurveSurfaceGeodesicCurvature(
            FramedCurveSurfaceTangent(self.master, WINDING_R0, 0.0), p=2, threshold=0.0
        )
        rot_dofs = set(self.rotation.dof_names)
        self.assertTrue(rot_dofs)
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
        decoupled_frame = FramedCurveSurfaceTangent(self.master, WINDING_R0, 0.0)
        self.assertIsInstance(decoupled_frame.rotation, ZeroRotation)
        decoupled = CurveSurfaceGeodesicCurvature(decoupled_frame, p=2, threshold=0.0)
        self.assertEqual(decoupled.dof_size, self.master.dof_size)


class TorsionalStrainRegularizerTest(unittest.TestCase):
    def setUp(self):
        self.surf_coils = _winding_surface()
        self.master = _master_curve(self.surf_coils)
        self.shared_frame = _shared_pack_frame(self.master, self.surf_coils)
        self.rotation = self.shared_frame.rotation

    def test_strain_penalty_depends_on_rotation_dofs(self):
        jtwist = LPTorsionalStrainPenalty(self.shared_frame, width=0.04, p=2)
        rot_dofs = set(self.rotation.dof_names)
        self.assertTrue(rot_dofs <= set(jtwist.dof_names))

    def test_strain_penalty_responds_to_non_constant_twist(self):
        jtwist = LPTorsionalStrainPenalty(self.shared_frame, width=0.04, p=2)
        x0 = np.asarray(self.rotation.x, dtype=float).copy()
        j0 = jtwist.J()
        twisted = x0.copy()
        twisted[-1] = x0[-1] + 0.1  # excite a theta-dependent mode
        self.rotation.x = twisted
        j1 = jtwist.J()
        self.rotation.x = x0
        self.assertGreater(j1, j0)


class RotationAwareCurvatureCapTest(unittest.TestCase):
    """The relaxed cap (the LpCurveCurvature steering target) rises when the pack
    twist favours the thin axis; the conservative honest gate is independent."""

    def setUp(self):
        self.settings = _settings()
        self.conservative_inv_m = finite_build_frame_aware_curvature_limit_inv_m(
            self.settings, TYPE_KK_INNER_RADIUS_MARGIN_M
        )

    def _cap_for_constant_alpha(self, alpha):
        # A poloidal loop near the winding torus: its surface-tangent frame normal
        # rotates around the loop, so a constant pack twist sweeps the bend angle.
        curve = CurveXYZFourier(64, 1)
        dofs = {name: 0.0 for name in curve.local_dof_names}
        dofs["xc(0)"] = WINDING_R0
        dofs["xc(1)"] = 0.1
        dofs["zs(1)"] = 0.1
        curve.x = np.array([dofs[name] for name in curve.local_dof_names])
        rotation = FrameRotation(curve.quadpoints, 1)
        _set_constant_alpha(rotation, alpha)
        framedcurve = FramedCurveSurfaceTangent(curve, WINDING_R0, 0.0, rotation)
        reach = rotation_aware_projected_half_extent_m(
            self.settings, curve, framedcurve
        )
        return 1.0 / (TYPE_KK_INNER_RADIUS_MARGIN_M + float(np.max(reach)))

    def test_conservative_edgewise_cap_pins_expected_value(self):
        # Conservative cap = 1/(margin + outer edgewise reach) ~43.31/m.
        self.assertAlmostEqual(self.conservative_inv_m, 43.31, places=1)

    def test_relaxed_cap_can_exceed_conservative_for_favorable_twist(self):
        # Scan the twist; at least one orientation must lift the cap above the
        # conservative edgewise cap (toward the ~123.03/m flatwise best case).
        caps = [self._cap_for_constant_alpha(a) for a in np.linspace(0.0, np.pi, 37)]
        self.assertGreater(max(caps), self.conservative_inv_m + 1.0)
        # The flatwise/thin-axis best case is ~123.03/m (allow a small numerical band).
        self.assertLessEqual(max(caps), 123.1)

    def test_relaxed_cap_never_below_conservative(self):
        caps = [self._cap_for_constant_alpha(a) for a in np.linspace(0.0, np.pi, 37)]
        self.assertGreaterEqual(min(caps), self.conservative_inv_m - 1e-6)


if __name__ == "__main__":
    unittest.main()
