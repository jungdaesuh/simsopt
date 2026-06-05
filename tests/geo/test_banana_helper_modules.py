import importlib.util
import os
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
REFERENCE_SURFACES_PATH = EXAMPLES_ROOT / "banana_opt" / "reference_surfaces.py"
STAGE2_GEOMETRY_PATH = EXAMPLES_ROOT / "banana_opt" / "stage2_geometry.py"
HARDWARE_CONTRACTS_PATH = EXAMPLES_ROOT / "banana_opt" / "hardware_contracts.py"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from banana_opt import proxy_current_coils  # noqa: E402


def _load_module(module_path: Path, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.path.insert(0, str(EXAMPLES_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        del sys.path[0]
    return module


class _FakeSurfaceRZFourier:
    def __init__(self, nfp, stellsym):
        self.nfp = nfp
        self.stellsym = stellsym
        self.rc = {}
        self.zs = {}

    def set_rc(self, m, n, value):
        self.rc[(m, n)] = float(value)

    def set_zs(self, m, n, value):
        self.zs[(m, n)] = float(value)


class BananaReferenceSurfaceTests(unittest.TestCase):
    def test_build_banana_reference_surfaces_sets_expected_modes(self):
        module = _load_module(REFERENCE_SURFACES_PATH, "banana_reference_surfaces")

        with mock.patch.object(module, "SurfaceRZFourier", _FakeSurfaceRZFourier):
            surfaces = module.build_banana_reference_surfaces(
                nfp=5,
                banana_surf_radius=0.23,
            )

        self.assertEqual(surfaces.vessel.nfp, 5)
        self.assertTrue(surfaces.vessel.stellsym)
        self.assertEqual(surfaces.vessel.rc, {(0, 0): 0.976, (1, 0): 0.222})
        self.assertEqual(surfaces.vessel.zs, {(1, 0): 0.222})

        self.assertEqual(
            surfaces.lcfs_clearance_reference.rc,
            {
                (0, 0): module.LCFS_CLEARANCE_REFERENCE_MAJOR_RADIUS_M,
                (1, 0): module.LCFS_CLEARANCE_REFERENCE_MINOR_RADIUS_M,
            },
        )
        self.assertEqual(
            surfaces.lcfs_clearance_reference.zs,
            {(1, 0): module.LCFS_CLEARANCE_REFERENCE_MINOR_RADIUS_M},
        )

        self.assertEqual(
            surfaces.coil_winding_surface.rc,
            {
                (0, 0): module.BANANA_WINDING_SURFACE_MAJOR_RADIUS_M,
                (1, 0): 0.23,
            },
        )
        self.assertEqual(
            surfaces.coil_winding_surface.zs,
            {(1, 0): 0.23},
        )


class Stage2GeometryHelperTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module(STAGE2_GEOMETRY_PATH, "banana_stage2_geometry")
        self.hardware_contracts = _load_module(
            HARDWARE_CONTRACTS_PATH,
            "banana_hardware_contracts",
        )

    def test_load_plasma_geometry_scales_lcfs_to_target_and_working_with_same_factor(self):
        class FakeSurface:
            def __init__(self, major_radius, minor_radius):
                self._major_radius = float(major_radius)
                self._minor_radius = float(minor_radius)
                self._dofs = np.array([1.0])
                self.nfp = 22

            def major_radius(self):
                return self._major_radius

            def minor_radius(self):
                return self._minor_radius

            def get_dofs(self):
                return self._dofs.copy()

            def set_dofs(self, dofs):
                dofs = np.asarray(dofs, dtype=float)
                scale = dofs[0] / self._dofs[0]
                self._dofs = dofs
                self._major_radius *= scale
                self._minor_radius *= scale

        def fake_from_wout(*_args, **kwargs):
            if kwargs["s"] == 0.24:
                return FakeSurface(1.20, 0.10)
            if kwargs["s"] == 1.0:
                return FakeSurface(1.45, 0.19)
            raise AssertionError(f"unexpected VMEC surface label {kwargs['s']}")

        with mock.patch.object(
            self.module.SurfaceRZFourier,
            "from_wout",
            side_effect=fake_from_wout,
        ):
            geometry = self.module.load_plasma_geometry(
                target_lcfs_major_radius_m=(
                    self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M
                ),
                s_working=0.24,
                file_loc="/tmp/demo.nc",
                nphi=8,
                ntheta=8,
            )

        expected_scale = (
            self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M / 1.45
        )
        self.assertAlmostEqual(geometry.scale_factor, expected_scale)
        self.assertAlmostEqual(
            geometry.working_surface.major_radius(),
            1.20 * expected_scale,
        )
        self.assertAlmostEqual(
            geometry.working_surface.minor_radius(),
            0.10 * expected_scale,
        )
        self.assertAlmostEqual(
            geometry.lcfs_major_radius_m,
            self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M,
        )
        self.assertAlmostEqual(geometry.lcfs_minor_radius_m, 0.19 * expected_scale)

    def test_load_plasma_geometry_real_wout_uses_scaled_lcfs_boundary(self):
        equilibrium_path = (
            EXAMPLES_ROOT
            / "equilibria"
            / "wout_nfp22ginsburg_000_014417_iota15.nc"
        )
        nphi = 91
        ntheta = 32
        working_label = 0.24
        target_lcfs_major_radius = (
            self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M
        )
        lcfs_surface = self.module.SurfaceRZFourier.from_wout(
            str(equilibrium_path),
            range="full torus",
            nphi=nphi,
            ntheta=ntheta,
            s=1.0,
        )
        expected_scale = (
            target_lcfs_major_radius / float(lcfs_surface.major_radius())
        )
        expected_lcfs_surface = self.module.SurfaceRZFourier.from_wout(
            str(equilibrium_path),
            range="full torus",
            nphi=nphi,
            ntheta=ntheta,
            s=1.0,
        )
        expected_lcfs_surface.set_dofs(
            expected_lcfs_surface.get_dofs() * expected_scale
        )

        geometry = self.module.load_plasma_geometry(
            target_lcfs_major_radius_m=target_lcfs_major_radius,
            s_working=working_label,
            file_loc=str(equilibrium_path),
            nphi=nphi,
            ntheta=ntheta,
        )

        self.assertAlmostEqual(
            geometry.lcfs_major_radius_m,
            target_lcfs_major_radius,
            places=12,
        )
        self.assertAlmostEqual(geometry.scale_factor, expected_scale, places=12)
        self.assertAlmostEqual(
            geometry.lcfs_minor_radius_m,
            expected_lcfs_surface.minor_radius(),
            places=12,
        )
        self.assertAlmostEqual(
            geometry.lcfs_major_radius_m,
            expected_lcfs_surface.major_radius(),
            places=12,
        )
        self.assertAlmostEqual(
            geometry.lcfs_minor_radius_m,
            expected_lcfs_surface.minor_radius(),
            places=12,
        )

    def test_geometry_preflight_prefers_requested_candidate_when_it_fits(self):
        class FakeSurface:
            def __init__(self, major_radius, minor_radius, gamma_value):
                self._major_radius = float(major_radius)
                self._minor_radius = float(minor_radius)
                self._gamma_value = float(gamma_value)

            def major_radius(self):
                return self._major_radius

            def minor_radius(self):
                return self._minor_radius

            def gamma(self):
                return np.ones((2, 2, 3), dtype=float) * self._gamma_value

        result = self.module.select_plasma_geometry_preflight_candidate(
            lcfs_surface=FakeSurface(1.84, 0.26, 0.30),
            requested_s=0.24,
            target_lcfs_major_radius_m=(
                self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M
            ),
            target_lcfs_minor_radius_m=(
                self.hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M
            ),
            vessel_surface=FakeSurface(0.0, 0.0, 0.10),
            s_candidates=(0.24, 0.50),
            target_lcfs_major_radius_candidates_m=(
                self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M,
                0.90,
            ),
        )

        self.assertTrue(result.selected.success)
        self.assertEqual(result.selected.s_working, 0.24)
        self.assertEqual(
            result.selected.target_lcfs_major_radius_m,
            self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M,
        )
        self.assertAlmostEqual(
            result.selected.lcfs_minor_radius_m,
            (
                0.26
                * self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M
                / 1.84
            ),
        )
        self.assertEqual(len(result.candidates), 4)

    def test_geometry_preflight_records_surface_vessel_distance_without_gating(self):
        class FakeSurface:
            def __init__(self, major_radius, minor_radius):
                self._major_radius = float(major_radius)
                self._minor_radius = float(minor_radius)

            def major_radius(self):
                return self._major_radius

            def minor_radius(self):
                return self._minor_radius

        def fake_distance(_surface, scale_factor, _vessel):
            return 0.039 if scale_factor > 0.95 else 0.045

        result = self.module.select_plasma_geometry_preflight_candidate(
            lcfs_surface=FakeSurface(
                self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M,
                0.12,
            ),
            requested_s=0.24,
            target_lcfs_major_radius_m=(
                self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M
            ),
            target_lcfs_minor_radius_m=(
                self.hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M
            ),
            vessel_surface=object(),
            s_candidates=(0.24,),
            target_lcfs_major_radius_candidates_m=(
                self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M,
                0.87,
            ),
            distance_fn=fake_distance,
        )

        self.assertEqual(result.selected.s_working, 0.24)
        self.assertEqual(
            result.selected.target_lcfs_major_radius_m,
            self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M,
        )
        self.assertEqual(result.selected.violations, ())
        self.assertAlmostEqual(result.selected.plasma_vessel_min_dist_m, 0.039)

    def test_geometry_preflight_includes_requested_target_below_scan_floor(self):
        class FakeSurface:
            def major_radius(self):
                return 1.0

            def minor_radius(self):
                return 0.12

        result = self.module.select_plasma_geometry_preflight_candidate(
            lcfs_surface=FakeSurface(),
            requested_s=0.24,
            target_lcfs_major_radius_m=0.79,
            target_lcfs_minor_radius_m=(
                self.hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M
            ),
            vessel_surface=object(),
            s_candidates=(0.24,),
            distance_fn=lambda *_args: 0.05,
        )

        self.assertEqual(result.selected.target_lcfs_major_radius_m, 0.79)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.selected.violations, ())

    def test_geometry_preflight_rejects_when_no_candidate_fits(self):
        class FakeSurface:
            def major_radius(self):
                return 1.0

            def minor_radius(self):
                return 0.30

        with self.assertRaisesRegex(ValueError, "No Stage 2 plasma geometry"):
            self.module.select_plasma_geometry_preflight_candidate(
                lcfs_surface=FakeSurface(),
                requested_s=0.24,
                target_lcfs_major_radius_m=(
                    self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M
                ),
                target_lcfs_minor_radius_m=(
                    self.hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M
                ),
                vessel_surface=object(),
                s_candidates=(0.24,),
                target_lcfs_major_radius_candidates_m=(
                    self.hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M,
                ),
                distance_fn=lambda *_args: 0.03,
            )

    def test_build_proxy_plasma_current_coils_rescales_vmec_axis(self):
        class FakeNetcdfFile:
            def __init__(self):
                self.variables = {
                    "raxis_cc": np.array([2.0]),
                    "zaxis_cs": np.array([0.3]),
                }

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        class FakeCurveXYZFourier:
            def __init__(self, nquadpoints, order):
                self.nquadpoints = nquadpoints
                self.order = order
                self.assignments = {}
                self.fixed = False

            def set(self, name, value):
                self.assignments[name] = value

            def fix_all(self):
                self.fixed = True

        class FakeCurrent:
            def __init__(self, value):
                self.value = float(value)
                self.fixed = False

            def fix_all(self):
                self.fixed = True

        class FakeCoil:
            def __init__(self, curve, current):
                self.curve = curve
                self.current = current

        with mock.patch.object(
            proxy_current_coils,
            "netcdf_file",
            return_value=FakeNetcdfFile(),
        ), mock.patch.object(
            proxy_current_coils,
            "CurveXYZFourier",
            FakeCurveXYZFourier,
        ), mock.patch.object(
            proxy_current_coils,
            "Current",
            FakeCurrent,
        ), mock.patch.object(
            proxy_current_coils,
            "Coil",
            FakeCoil,
        ):
            coils = self.module.build_proxy_plasma_current_coils(
                equilibrium_file="/tmp/demo.nc",
                surface_scale_factor=2.0,
                nphi=91,
                ntheta=32,
                toroidal_flux=0.24,
                plasma_current_A=9000.0,
            )

        self.assertEqual(len(coils), 1)
        proxy_coil = coils[0]
        self.assertEqual(proxy_coil.curve.assignments["xc(1)"], 4.0)
        self.assertEqual(proxy_coil.curve.assignments["ys(1)"], 4.0)
        self.assertEqual(proxy_coil.curve.assignments["zc(0)"], 0.6)
        self.assertTrue(proxy_coil.curve.fixed)
        self.assertEqual(proxy_coil.current.value, 9000.0)
        self.assertTrue(proxy_coil.current.fixed)

    def test_build_vf_coils_preserves_template_current_signs(self):
        class FakeCurve:
            def __init__(self):
                self.fixed = False

            def fix_all(self):
                self.fixed = True

        class FakeTemplateCurrent:
            def __init__(self, value):
                self.value = float(value)

            def get_value(self):
                return self.value

        class FakeCurrent:
            def __init__(self, value):
                self.value = float(value)
                self.fixed = False

            def fix_all(self):
                self.fixed = True

        class FakeCoil:
            def __init__(self, curve, current):
                self.curve = curve
                self.current = current

        template = SimpleNamespace(
            coils=[
                FakeCoil(FakeCurve(), FakeTemplateCurrent(3.0)),
                FakeCoil(FakeCurve(), FakeTemplateCurrent(-7.0)),
            ]
        )

        with mock.patch.object(self.module, "Current", FakeCurrent), mock.patch.object(
            self.module,
            "Coil",
            FakeCoil,
        ):
            result = self.module.build_vf_coil_build_result(
                vf_current_A=500.0,
                vf_template_path="/tmp/vf_template.json",
                load_fn=lambda _path: template,
            )
            coils = self.module.build_vf_coils(
                vf_current_A=500.0,
                vf_template_path="/tmp/vf_template.json",
                load_fn=lambda _path: template,
            )

        self.assertIsNone(result.current_control)
        self.assertEqual([coil.current.value for coil in result.coils], [500.0, -500.0])
        self.assertEqual([coil.current.value for coil in coils], [500.0, -500.0])
        self.assertTrue(all(coil.curve.fixed for coil in coils))
        self.assertTrue(all(coil.current.fixed for coil in coils))

    def test_build_vf_coils_rejects_zero_sign_template_current(self):
        class FakeTemplateCurrent:
            def get_value(self):
                return 0.0

        class FakeCoil:
            def __init__(self):
                self.curve = object()
                self.current = FakeTemplateCurrent()

        with self.assertRaisesRegex(ValueError, "must carry non-zero signed currents"):
            self.module.build_vf_coils(
                vf_current_A=500.0,
                vf_template_path="/tmp/vf_template.json",
                load_fn=lambda _path: SimpleNamespace(coils=[FakeCoil()]),
            )

    def test_shared_vf_current_control_requires_one_leaf_current(self):
        shared_current = self.module.Current(1.0) * 120.0
        shared_control = self.module.shared_vf_current_control_for_coils(
            [
                SimpleNamespace(current=shared_current * 1.0),
                SimpleNamespace(current=shared_current * -1.0),
            ]
        )

        self.assertIs(shared_control, shared_current)

        with self.assertRaisesRegex(ValueError, "share one leaf Current"):
            self.module.shared_vf_current_control_for_coils(
                [
                    SimpleNamespace(current=self.module.Current(120.0)),
                    SimpleNamespace(current=self.module.Current(-120.0)),
                ]
            )

    def test_coerce_vf_build_result_rejects_raw_list_when_control_required(self):
        with self.assertRaisesRegex(ValueError, "requires VFCoilBuildResult"):
            self.module.coerce_vf_coil_build_result(
                [object()],
                requires_current_control=True,
            )

    def test_check_all_pairs_skips_adjacent_segments(self):
        segments = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            dtype=float,
        )

        self.assertFalse(
            self.module.check_all_pairs(segments, tol=1e-6, neighbor_skip=1)
        )

    def test_check_all_pairs_detects_non_neighbor_intersection(self):
        segments = np.array(
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[0.5, -0.2, 0.0], [0.5, 0.2, 0.0]],
                [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            ],
            dtype=float,
        )

        self.assertTrue(
            self.module.check_all_pairs(segments, tol=1e-3, neighbor_skip=1)
        )

    def test_is_self_intersecting_detects_crossing_polygon(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        with mock.patch.object(self.module, "gamma_at_t", return_value=points):
            self.assertTrue(
                self.module.is_self_intersecting(
                    curve=object(),
                    npts=4,
                    tol_factor=0.01,
                    neighbor_skip=1,
                )
            )

    def test_is_self_intersecting_rejects_simple_loop(self):
        points = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        with mock.patch.object(self.module, "gamma_at_t", return_value=points):
            self.assertFalse(
                self.module.is_self_intersecting(
                    curve=object(),
                    npts=4,
                    tol_factor=0.01,
                    neighbor_skip=1,
                )
            )


class HardwareContractsPlasmaVesselClearanceTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module(HARDWARE_CONTRACTS_PATH, "banana_hardware_contracts")

    def test_fixed_stage2_clearance_contract_excludes_diagnostic_vessel_metric(self):
        self.assertNotIn(
            "PLASMA_VESSEL_MIN_DIST_M",
            self.module.fixed_stage2_clearance_contract(),
        )


class HardwareContractsEnvFlagTests(unittest.TestCase):
    def setUp(self):
        self.module = _load_module(HARDWARE_CONTRACTS_PATH, "banana_hardware_contracts")

    def test_validate_target_lcfs_major_radius_accepts_roundoff(self):
        roundoff_major_radius = (
            self.module.TARGET_LCFS_MAX_MAJOR_RADIUS_M
            + self.module.LCFS_RADIUS_ABS_TOL_M * 0.5
        )

        self.assertEqual(
            self.module.validate_target_lcfs_major_radius(roundoff_major_radius),
            roundoff_major_radius,
        )

    def test_validate_target_lcfs_major_radius_rejects_real_excess(self):
        real_excess_major_radius = (
            self.module.TARGET_LCFS_MAX_MAJOR_RADIUS_M
            + self.module.LCFS_RADIUS_ABS_TOL_M * 10.0
        )

        with self.assertRaisesRegex(ValueError, "target LCFS major radius"):
            self.module.validate_target_lcfs_major_radius(real_excess_major_radius)

    def test_env_flag_recognizes_truthy_values(self):
        flag_name = "BANANA_OPT_TEST_FLAG"
        original = os.environ.get(flag_name)
        try:
            for truthy in ("1", "true", "TRUE", "yes"):
                os.environ[flag_name] = truthy
                self.assertTrue(self.module.env_flag(flag_name))
            for falsy in ("0", "false", "", "no"):
                os.environ[flag_name] = falsy
                self.assertFalse(self.module.env_flag(flag_name))
        finally:
            if original is None:
                os.environ.pop(flag_name, None)
            else:
                os.environ[flag_name] = original
