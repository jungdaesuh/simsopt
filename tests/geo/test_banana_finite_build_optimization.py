"""Tests for finite-build banana Stage-2 optimization.

Covers the construction helper (`build_finite_build_banana_coils`), the
`FiniteBuildSettings` contract, the solver CLI/metadata helpers
(`resolve_finite_build_settings`, `validate_finite_build_cli_args`,
`_finite_build_artifact_metadata`), and the `initialize_coils` jhalpern30 guard.

Run:
  PYTHONNOUSERSITE=1 PYTHONPATH=examples/single_stage_optimization \
    .conda-env/bin/python -m pytest -q tests/geo/test_banana_finite_build_optimization.py
"""

import importlib.util
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.coil import ScaledCurrent
from simsopt.geo import CurveCWSFourierCPP, SurfaceRZFourier
from simsopt.objectives import SquaredFlux

from banana_opt.finite_current_profiles import JHALPERN30_FINITE_CURRENT_MODE
from banana_opt.reference_surfaces import build_banana_reference_surfaces
from banana_opt.stage2_geometry import (
    FiniteBuildSettings,
    build_finite_build_banana_coils,
    initialize_coils,
)


NFP = 5
NET_BANANA_CURRENT_A = -16000.0
BANANA_SURF_RADIUS_M = 0.21
SOLVER_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "banana_coil_solver.py"
)


def _load_solver_module():
    spec = importlib.util.spec_from_file_location(
        f"banana_coil_solver_{uuid.uuid4().hex}", SOLVER_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _winding_surface():
    return build_banana_reference_surfaces(NFP, BANANA_SURF_RADIUS_M).coil_winding_surface


def _master_curve(surf_coils, order=4, num_quadpoints=96):
    curve = CurveCWSFourierCPP(
        np.linspace(0, 1, num_quadpoints, endpoint=False), order=order, surf=surf_coils
    )
    curve.set("phic(0)", 0.0)
    curve.set("thetac(0)", 0.0)
    curve.set("phic(1)", 0.05)
    curve.set("thetas(1)", 0.5)
    return curve


def _settings(rotation_order=1, frame="centroid"):
    return FiniteBuildSettings(
        numfilaments_n=2,
        numfilaments_b=3,
        gapsize_n=0.02,
        gapsize_b=0.04,
        rotation_order=rotation_order,
        frame=frame,
    )


class FiniteBuildSettingsTest(unittest.TestCase):
    def test_nfilaments_is_product_of_counts(self):
        self.assertEqual(_settings().nfilaments, 6)

    def test_pack_half_extents_use_count_minus_one_times_gap(self):
        settings = _settings()
        self.assertAlmostEqual(settings.pack_half_extent_n_m, 0.5 * (2 - 1) * 0.02)
        self.assertAlmostEqual(settings.pack_half_extent_b_m, 0.5 * (3 - 1) * 0.04)

    def test_pack_reach_is_hypot_of_half_extents(self):
        settings = _settings()  # half_n=0.01, half_b=0.04
        self.assertAlmostEqual(settings.pack_reach_m, np.hypot(0.01, 0.04))


class BuildFiniteBuildBananaCoilsTest(unittest.TestCase):
    def setUp(self):
        self.surf_coils = _winding_surface()
        self.symmetry_copies = NFP * (2 if self.surf_coils.stellsym else 1)

    def test_coil_count_is_filaments_times_symmetry_copies(self):
        settings = _settings()
        coils = build_finite_build_banana_coils(
            _master_curve(self.surf_coils),
            NET_BANANA_CURRENT_A,
            settings,
            self.surf_coils,
        )
        self.assertEqual(len(coils), settings.nfilaments * self.symmetry_copies)

    def test_net_current_conserved_across_one_pack(self):
        settings = _settings()
        coils = build_finite_build_banana_coils(
            _master_curve(self.surf_coils),
            NET_BANANA_CURRENT_A,
            settings,
            self.surf_coils,
        )
        # The first `nfilaments` coils are the identity-symmetry pack copy.
        one_pack = coils[: settings.nfilaments]
        net = sum(c.current.get_value() for c in one_pack)
        self.assertAlmostEqual(net, NET_BANANA_CURRENT_A, places=6)
        for coil in one_pack:
            self.assertAlmostEqual(
                coil.current.get_value(), NET_BANANA_CURRENT_A / settings.nfilaments
            )

    def test_net_current_optimizable_recoverable_from_first_coil(self):
        coils = build_finite_build_banana_coils(
            _master_curve(self.surf_coils),
            NET_BANANA_CURRENT_A,
            _settings(),
            self.surf_coils,
        )
        net_current = coils[0].current.current_to_scale
        self.assertAlmostEqual(net_current.get_value(), NET_BANANA_CURRENT_A)

    def test_rotation_order_adds_only_rotation_dofs_one_shared_current(self):
        master = _master_curve(self.surf_coils)
        thin = coils_via_symmetries(
            [master],
            [ScaledCurrent(Current(1), NET_BANANA_CURRENT_A)],
            self.surf_coils.nfp,
            self.surf_coils.stellsym,
        )
        fb_norot = build_finite_build_banana_coils(
            master, NET_BANANA_CURRENT_A, _settings(rotation_order=None), self.surf_coils
        )
        fb_rot1 = build_finite_build_banana_coils(
            master, NET_BANANA_CURRENT_A, _settings(rotation_order=1), self.surf_coils
        )
        n_thin = len(BiotSavart(thin).x)
        n_norot = len(BiotSavart(fb_norot).x)
        n_rot1 = len(BiotSavart(fb_rot1).x)
        # No rotation: filaments share the master curve + a single current DOF, so
        # the finite-build pack has the same DOF count as the thin coil.
        self.assertEqual(n_norot, n_thin)
        # rotation_order=1 -> FrameRotation adds 2*order+1 = 3 DOFs, nothing else.
        self.assertEqual(n_rot1, n_thin + 3)

    def test_field_evaluates_finite(self):
        coils = build_finite_build_banana_coils(
            _master_curve(self.surf_coils),
            NET_BANANA_CURRENT_A,
            _settings(),
            self.surf_coils,
        )
        bs = BiotSavart(coils)
        points = self.surf_coils.gamma().reshape((-1, 3))
        bs.set_points(points)
        field = bs.B()
        self.assertEqual(field.shape, (points.shape[0], 3))
        self.assertTrue(np.all(np.isfinite(field)))

    def test_field_objective_and_gradient_finite_through_rotation_dofs(self):
        surfs = build_banana_reference_surfaces(NFP, BANANA_SURF_RADIUS_M)
        surf_coils = surfs.coil_winding_surface
        plasma = SurfaceRZFourier(nfp=NFP, stellsym=True, mpol=2, ntor=2)
        plasma.set_rc(0, 0, 0.9)
        plasma.set_rc(1, 0, 0.1)
        plasma.set_zs(1, 0, 0.1)
        master = _master_curve(surf_coils)
        coils = build_finite_build_banana_coils(
            master, NET_BANANA_CURRENT_A, _settings(rotation_order=1), surf_coils
        )
        # The finite-build field objective and its gradient — flowing through the
        # filament pack and the FrameRotation DOFs, the genuinely new optimization
        # path — are finite.
        Jf = SquaredFlux(plasma, BiotSavart(coils))
        self.assertTrue(np.isfinite(Jf.J()))
        self.assertTrue(np.all(np.isfinite(Jf.dJ())))
        # The pack-rotation DOFs are live in the field-objective DOF vector
        # (rotation_order=1 -> 3 extra DOFs over the no-rotation pack).
        coils_norot = build_finite_build_banana_coils(
            master, NET_BANANA_CURRENT_A, _settings(rotation_order=None), surf_coils
        )
        Jf_norot = SquaredFlux(plasma, BiotSavart(coils_norot))
        self.assertEqual(len(Jf.x), len(Jf_norot.x) + 3)


class InitializeCoilsFiniteBuildGuardTest(unittest.TestCase):
    def test_jhalpern30_with_finite_build_raises(self):
        surf_coils = _winding_surface()
        with self.assertRaises(ValueError):
            initialize_coils(
                surf_coils,
                surf_coils,
                [],
                96,
                4,
                NET_BANANA_CURRENT_A,
                0.0,
                0.0,
                0.05,
                0.5,
                "/tmp/finite_build_guard_test/",
                equilibrium_file="unused.nc",
                surface_scale_factor=1.0,
                toroidal_flux=0.3,
                nphi=8,
                ntheta=8,
                finite_current_mode=JHALPERN30_FINITE_CURRENT_MODE,
                finite_build=_settings(),
            )


class SolverFiniteBuildHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = _load_solver_module()

    def test_resolve_returns_none_when_off(self):
        args = SimpleNamespace(finite_build=False)
        self.assertIsNone(self.module.resolve_finite_build_settings(args))

    def test_resolve_builds_settings_when_on(self):
        args = SimpleNamespace(
            finite_build=True,
            finitebuild_numfilaments_n=2,
            finitebuild_numfilaments_b=3,
            finitebuild_gapsize_n=0.02,
            finitebuild_gapsize_b=0.04,
            finitebuild_rotation_order=2,
            finitebuild_frame="centroid",
        )
        settings = self.module.resolve_finite_build_settings(args)
        self.assertIsInstance(settings, FiniteBuildSettings)
        self.assertEqual(settings.nfilaments, 6)
        self.assertEqual(settings.rotation_order, 2)

    def test_resolve_negative_rotation_order_maps_to_none(self):
        args = SimpleNamespace(
            finite_build=True,
            finitebuild_numfilaments_n=2,
            finitebuild_numfilaments_b=3,
            finitebuild_gapsize_n=0.02,
            finitebuild_gapsize_b=0.04,
            finitebuild_rotation_order=-1,
            finitebuild_frame="centroid",
        )
        self.assertIsNone(self.module.resolve_finite_build_settings(args).rotation_order)

    def _valid_args(self, **overrides):
        base = dict(
            finite_build=True,
            stage2_bs_path=None,
            finitebuild_numfilaments_n=2,
            finitebuild_numfilaments_b=3,
            finitebuild_gapsize_n=0.02,
            finitebuild_gapsize_b=0.04,
        )
        base.update(overrides)
        return SimpleNamespace(**base)

    def test_validate_passes_when_off(self):
        self.module.validate_finite_build_cli_args(SimpleNamespace(finite_build=False))

    def test_validate_passes_for_valid_fresh_config(self):
        self.module.validate_finite_build_cli_args(self._valid_args())

    def test_validate_rejects_seeded_path(self):
        with self.assertRaises(ValueError):
            self.module.validate_finite_build_cli_args(
                self._valid_args(stage2_bs_path="seed.json")
            )

    def test_validate_rejects_nonpositive_filament_count(self):
        with self.assertRaises(ValueError):
            self.module.validate_finite_build_cli_args(
                self._valid_args(finitebuild_numfilaments_n=0)
            )

    def test_validate_rejects_nonpositive_gap(self):
        with self.assertRaises(ValueError):
            self.module.validate_finite_build_cli_args(
                self._valid_args(finitebuild_gapsize_b=0.0)
            )

    def test_metadata_reports_counts_currents_and_buildable_diagnostic(self):
        banana_curve = SimpleNamespace(kappa=lambda: np.array([1.0]))  # radius 1.0 m
        metadata = self.module._finite_build_artifact_metadata(
            _settings(), banana_curve, NET_BANANA_CURRENT_A
        )
        self.assertTrue(metadata["FINITE_BUILD_ENABLED"])
        self.assertEqual(metadata["FINITEBUILD_FILAMENTS_PER_BANANA"], 6)
        self.assertAlmostEqual(
            metadata["BANANA_FILAMENT_CURRENT_A"], NET_BANANA_CURRENT_A / 6
        )
        self.assertAlmostEqual(metadata["FINITEBUILD_MIN_CURVATURE_RADIUS_M"], 1.0)
        # radius 1.0 m >> pack half-extent 0.04 m -> buildable.
        self.assertTrue(metadata["FINITEBUILD_CURVATURE_OK"])

    def test_metadata_flags_unbuildable_tight_curvature(self):
        # max curvature 1000 m^-1 -> radius 0.001 m < pack half-extent 0.04 m.
        banana_curve = SimpleNamespace(kappa=lambda: np.array([1000.0]))
        metadata = self.module._finite_build_artifact_metadata(
            _settings(), banana_curve, NET_BANANA_CURRENT_A
        )
        self.assertFalse(metadata["FINITEBUILD_CURVATURE_OK"])

    def test_curvature_gate_uses_the_larger_half_build(self):
        # half_n = 0.5*(5-1)*0.02 = 0.04 (binding); half_b = 0.5*(3-1)*0.01 = 0.01.
        settings = FiniteBuildSettings(
            numfilaments_n=5,
            numfilaments_b=3,
            gapsize_n=0.02,
            gapsize_b=0.01,
            rotation_order=1,
            frame="centroid",
        )
        # radius 0.02 m: above the binormal half-build (0.01) but BELOW the binding
        # normal half-build (0.04) -> the old binormal-only gate would pass, the
        # corrected max(half_n,half_b) gate must fail.
        banana_curve = SimpleNamespace(kappa=lambda: np.array([50.0]))  # radius 0.02
        metadata = self.module._finite_build_artifact_metadata(
            settings, banana_curve, NET_BANANA_CURRENT_A
        )
        self.assertAlmostEqual(metadata["FINITEBUILD_BINDING_HALF_BUILD_M"], 0.04)
        self.assertAlmostEqual(metadata["FINITEBUILD_INNER_EDGE_RADIUS_M"], 0.02 - 0.04)
        self.assertFalse(metadata["FINITEBUILD_CURVATURE_OK"])

    def test_curvature_margin_tightens_gate(self):
        # radius 0.05 m, binding half-build 0.04 -> inner-edge radius 0.01 m.
        banana_curve = SimpleNamespace(kappa=lambda: np.array([20.0]))
        ok = self.module._finite_build_artifact_metadata(
            _settings(), banana_curve, NET_BANANA_CURRENT_A, curvature_margin_m=0.0
        )
        self.assertTrue(ok["FINITEBUILD_CURVATURE_OK"])  # 0.01 >= 0
        tight = self.module._finite_build_artifact_metadata(
            _settings(), banana_curve, NET_BANANA_CURRENT_A, curvature_margin_m=0.02
        )
        self.assertFalse(tight["FINITEBUILD_CURVATURE_OK"])  # 0.01 < 0.02

    def test_envelope_clearance_verdicts_use_corner_for_cc_normal_for_cs(self):
        # CC (pack-to-pack) uses the worst-case corner reach hypot(0.01,0.04)~=0.0412;
        # CS (pack-to-plasma) uses the NORMAL half-build only (half_n = 0.01).
        cc_reach = float(np.hypot(0.01, 0.04))
        cs_reach = 0.01
        banana_curve = SimpleNamespace(kappa=lambda: np.array([1.0]))
        m = self.module._finite_build_artifact_metadata(
            _settings(),
            banana_curve,
            NET_BANANA_CURRENT_A,
            cc_min_dist_m=0.20,
            cs_min_dist_m=0.08,
            cc_nominal_m=0.0462,
            cs_nominal_m=0.010,
        )
        self.assertAlmostEqual(m["FINITEBUILD_PACK_REACH_M"], cc_reach)
        self.assertAlmostEqual(m["FINITEBUILD_CS_REACH_M"], cs_reach)
        self.assertAlmostEqual(
            m["FINITEBUILD_CC_ENVELOPE_MIN_DIST_M"], 0.20 - 2 * cc_reach
        )
        self.assertTrue(m["FINITEBUILD_CC_ENVELOPE_OK"])  # 0.117 >= 0.0462
        self.assertAlmostEqual(m["FINITEBUILD_CS_ENVELOPE_MIN_DIST_M"], 0.08 - cs_reach)
        self.assertTrue(m["FINITEBUILD_CS_ENVELOPE_OK"])  # 0.07 >= 0.010
        # Tight gaps fail once the (different) reaches are subtracted.
        tight = self.module._finite_build_artifact_metadata(
            _settings(),
            banana_curve,
            NET_BANANA_CURRENT_A,
            cc_min_dist_m=0.10,
            cs_min_dist_m=0.012,
            cc_nominal_m=0.0462,
            cs_nominal_m=0.010,
        )
        self.assertFalse(tight["FINITEBUILD_CC_ENVELOPE_OK"])  # 0.0176 < 0.0462
        self.assertFalse(tight["FINITEBUILD_CS_ENVELOPE_OK"])  # 0.002 < 0.010

    def test_metadata_omits_envelope_keys_without_distances(self):
        banana_curve = SimpleNamespace(kappa=lambda: np.array([1.0]))
        m = self.module._finite_build_artifact_metadata(
            _settings(), banana_curve, NET_BANANA_CURRENT_A
        )
        self.assertNotIn("FINITEBUILD_CC_ENVELOPE_OK", m)
        self.assertNotIn("FINITEBUILD_CS_ENVELOPE_OK", m)


if __name__ == "__main__":
    unittest.main()
