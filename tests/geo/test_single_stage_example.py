import importlib.util
from contextlib import ExitStack
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simsopt.geo.surfaceobjectives import boozer_surface_residual, boozer_surface_residual_dB
from simsopt.objectives.utilities import forward_backward


EXAMPLE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)
BOOZER_SURFACE_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "simsopt"
    / "geo"
    / "boozersurface.py"
)
TOPOLOGY_SCORER_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "topology_scorer.py"
)
ALM_UTILS_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "alm_utils.py"
)
TEST_MPOL = 8
TEST_NTOR = 6
TEST_VOL_TARGET = 0.1
TEST_IOTA = 0.15
TEST_G0 = 1.0
TEST_BOOZER_I = 0.37


def load_single_stage_example_module():
    spec = importlib.util.spec_from_file_location(
        f"single_stage_banana_example_{uuid.uuid4().hex}",
        EXAMPLE_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_topology_scorer_module():
    spec = importlib.util.spec_from_file_location(
        f"topology_scorer_{uuid.uuid4().hex}",
        TOPOLOGY_SCORER_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_alm_utils_module():
    spec = importlib.util.spec_from_file_location(
        f"alm_utils_{uuid.uuid4().hex}",
        ALM_UTILS_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakeSurfPrev:
    def __init__(self):
        self.nfp = 5
        self.quadpoints_phi = np.linspace(0, 1 / self.nfp, 13, endpoint=False)
        self.quadpoints_theta = np.linspace(0, 1, 17, endpoint=False)

    def gamma(self):
        return np.zeros((self.quadpoints_phi.size, self.quadpoints_theta.size, 3))


class FakeSurfaceXYZTensorFourier:
    instances = []

    def __init__(
        self,
        *,
        mpol,
        ntor,
        nfp,
        stellsym,
        quadpoints_theta,
        quadpoints_phi,
        dofs=None,
    ):
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.stellsym = stellsym
        self.quadpoints_theta = np.asarray(quadpoints_theta)
        self.quadpoints_phi = np.asarray(quadpoints_phi)
        self.dofs = np.array([1.0]) if dofs is None else np.asarray(dofs)
        FakeSurfaceXYZTensorFourier.instances.append(self)

    def least_squares_fit(self, gamma):
        self.fitted_gamma = gamma

    def is_self_intersecting(self):
        return False

    def volume(self):
        return 1.0


class FakeVolume:
    def __init__(self, surface):
        self.surface = surface


class FakeBoozerSurface:
    def __init__(self, bs, surface, label, targetlabel, constraint_weight, options=None, I=0.0):
        self.bs = bs
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.constraint_weight = constraint_weight
        self.options = options or {}
        self.I = I
        self.res = {"success": True, "iota": 0.15, "G": 1.0, "I": I}

    def run_code(self, iota, G):
        return self.res


class FakeResolvedSurface:
    def __init__(
        self,
        *,
        mpol,
        ntor,
        stellsym,
        nfp,
        quadpoints_phi,
        quadpoints_theta,
    ):
        self.mpol = mpol
        self.ntor = ntor
        self.stellsym = stellsym
        self.nfp = nfp
        self.quadpoints_phi = np.asarray(quadpoints_phi)
        self.quadpoints_theta = np.asarray(quadpoints_theta)
        self._dofs = np.zeros(2)

    def set_dofs(self, dofs):
        self._dofs = np.asarray(dofs, dtype=float)

    def get_dofs(self):
        return self._dofs.copy()

    def gamma(self):
        return np.zeros((self.quadpoints_phi.size, self.quadpoints_theta.size, 3))


class FakeLabel:
    def J(self):
        return 0.0

    def dJ_by_dsurfacecoefficients(self):
        return np.zeros(2)


class FakeObjectiveBiotSavart:
    def __init__(self):
        self.points = None
        self.last_vjp_input = None

    def set_points(self, points):
        self.points = np.asarray(points, dtype=float)

    def B_vjp(self, dJ_by_dB):
        self.last_vjp_input = np.asarray(dJ_by_dB, dtype=float)
        return np.array([3.0, -1.0])


class FakeParentBoozerSurface:
    def __init__(self, *, surface, label, targetlabel, need_to_run_code, res):
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.need_to_run_code = need_to_run_code
        self.res = res
        self.ancestors = []
        self.name = "FakeBoozerSurface"
        self.dofs = object()
        self.local_full_dof_size = 0
        self.local_dof_size = 0
        self._id = SimpleNamespace(id=0)

    def _add_child(self, child):
        del child


class FakeAlgebraicObjective:
    def __init__(self, value, gradient):
        self._value = float(value)
        self._gradient = np.asarray(gradient, dtype=float)

    def J(self):
        return self._value

    def dJ(self):
        return self._gradient.copy()

    def __add__(self, other):
        if other == 0:
            return self
        return FakeAlgebraicObjective(self._value + other._value, self._gradient + other._gradient)

    __radd__ = __add__

    def __mul__(self, scalar):
        return FakeAlgebraicObjective(self._value * scalar, self._gradient * scalar)

    __rmul__ = __mul__


class SingleStageExampleTests(unittest.TestCase):
    def setUp(self):
        FakeSurfaceXYZTensorFourier.instances = []

    def load_module(self):
        return load_single_stage_example_module()

    def initialize_boozer_surface(self, module, surf_prev, *, constraint_weight):
        with patch.object(module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier), patch.object(
            module, "Volume", FakeVolume
        ), patch.object(module, "BoozerSurface", FakeBoozerSurface):
            return module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=constraint_weight,
                iota=TEST_IOTA,
                G0=TEST_G0,
                boozer_I=TEST_BOOZER_I,
            )

    def run_exact_boozer_objective(self, module, *, current_I):
        input_surface = FakeResolvedSurface(
            mpol=2,
            ntor=2,
            stellsym=True,
            nfp=5,
            quadpoints_phi=np.linspace(0, 1 / 5, 2, endpoint=False),
            quadpoints_theta=np.linspace(0, 1, 3, endpoint=False),
        )
        input_surface.set_dofs(np.array([1.0, -2.0]))
        fake_boozer_surface = FakeParentBoozerSurface(
            surface=input_surface,
            label=FakeLabel(),
            targetlabel=0.0,
            need_to_run_code=False,
            res={
                "iota": -0.4,
                "G": 1.2,
                "I": current_I,
                "PLU": (None, None, None),
                "vjp": lambda adj, booz_surf, iota, G: np.zeros(2),
            },
        )
        fake_bs = FakeObjectiveBiotSavart()
        nsurfdofs = input_surface.get_dofs().size
        residual_calls = []
        residual_dB_calls = []

        def fake_residual(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=False, I=0.0):
            num_points = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
            residual_calls.append((derivatives, weight_inv_modB, I))
            return np.ones(num_points), np.zeros((num_points, nsurfdofs + 2))

        def fake_residual_dB(surface, iota, G, biotsavart, derivatives=0, weight_inv_modB=False, I=0.0):
            num_points = 3 * surface.quadpoints_phi.size * surface.quadpoints_theta.size
            residual_dB_calls.append((derivatives, weight_inv_modB, I))
            return np.ones(num_points), np.ones((num_points, 3))

        with patch.object(module, "SurfaceXYZTensorFourier", FakeResolvedSurface), patch.object(
            module, "boozer_surface_residual", side_effect=fake_residual
        ), patch.object(
            module, "boozer_surface_residual_dB", side_effect=fake_residual_dB
        ), patch.object(
            module, "forward_backward", return_value=np.zeros(nsurfdofs + 2)
        ):
            objective = module.BoozerResidualExact(fake_boozer_surface, fake_bs)
            value = objective.J()
            gradient = objective.dJ(partials=True)

        return objective, fake_bs, residual_calls, residual_dB_calls, value, gradient

    def test_exact_boozer_helpers_are_imported(self):
        module = self.load_module()

        self.assertIs(module.boozer_surface_residual, boozer_surface_residual)
        self.assertIs(module.boozer_surface_residual_dB, boozer_surface_residual_dB)
        self.assertIs(module.forward_backward, forward_backward)

    def test_save_surface_artifacts_writes_boozer_surface_jsons(self):
        module = self.load_module()

        class _Surface:
            def __init__(self):
                self.saved_paths = []
                self.vtk_paths = []

            def gamma(self):
                return np.zeros((1, 1, 3))

            def unitnormal(self):
                return np.ones((1, 1, 3))

            def to_vtk(self, path, extra_data=None):
                self.vtk_paths.append(path)

            def save(self, path):
                self.saved_paths.append(path)
                Path(path).write_text("surface", encoding="utf-8")

        class _BoozerSurface:
            def __init__(self, surface):
                self.surface = surface
                self.saved_paths = []

            def save(self, path):
                self.saved_paths.append(path)
                Path(path).write_text("boozer", encoding="utf-8")

        class _BiotSavart:
            def set_points(self, points):
                self.points = np.asarray(points)

            def B(self):
                return np.ones((1, 3))

        inner = _BoozerSurface(_Surface())
        outer = _BoozerSurface(_Surface())
        surface_data = [
            {"name": "inner", "boozer_surface": inner},
            {"name": "outer", "boozer_surface": outer},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            module.save_surface_artifacts(
                surface_data,
                _BiotSavart(),
                tmpdir,
                "surf_opt",
                also_write_outer_legacy=True,
            )

            tmp_path = Path(tmpdir)
            for filename in (
                "surf_opt_inner.json",
                "surf_opt_outer.json",
                "surf_opt.json",
                "surf_opt_inner_boozer_surface.json",
                "surf_opt_outer_boozer_surface.json",
                "surf_opt_boozer_surface.json",
            ):
                self.assertTrue((tmp_path / filename).exists())
            self.assertEqual(
                inner.saved_paths,
                [str(tmp_path / "surf_opt_inner_boozer_surface.json")],
            )
            self.assertEqual(
                outer.saved_paths,
                [
                    str(tmp_path / "surf_opt_outer_boozer_surface.json"),
                    str(tmp_path / "surf_opt_boozer_surface.json"),
                ],
            )

    def test_initialize_boozer_surface_exact_uses_ntor_phi_quadrature(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        boozer_surface = self.initialize_boozer_surface(module, surf_prev, constraint_weight=None)

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 2)

        exact_surface = FakeSurfaceXYZTensorFourier.instances[1]
        expected_phi = np.linspace(0, 1 / surf_prev.nfp, 2 * TEST_NTOR + 1, endpoint=False)

        self.assertEqual(exact_surface.quadpoints_theta.size, 2 * TEST_MPOL + 1)
        self.assertEqual(exact_surface.quadpoints_phi.size, 2 * TEST_NTOR + 1)
        np.testing.assert_allclose(exact_surface.quadpoints_phi, expected_phi)

    def test_initialize_boozer_surface_zero_constraint_weight_keeps_least_squares_path(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        boozer_surface = self.initialize_boozer_surface(module, surf_prev, constraint_weight=0.0)

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 1)
        self.assertIs(boozer_surface.surface, FakeSurfaceXYZTensorFourier.instances[0])
        self.assertEqual(boozer_surface.I, TEST_BOOZER_I)

    def test_initialize_boozer_surface_exact_threads_negative_current(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()

        with patch.object(module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier), patch.object(
            module, "Volume", FakeVolume
        ), patch.object(module, "BoozerSurface", FakeBoozerSurface):
            boozer_surface = module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=None,
                iota=TEST_IOTA,
                G0=TEST_G0,
                boozer_I=-TEST_BOOZER_I,
            )

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(boozer_surface.I, -TEST_BOOZER_I)

    def test_real_boozersurface_source_treats_zero_constraint_weight_as_least_squares(self):
        source = BOOZER_SURFACE_PATH.read_text()
        self.assertIn("self.boozer_type = 'ls' if constraint_weight is not None else 'exact'", source)

    def test_boozer_residual_exact_threads_fixed_current_into_example_adjoint_path(self):
        module = self.load_module()
        objective, fake_bs, residual_calls, residual_dB_calls, value, gradient = self.run_exact_boozer_objective(
            module,
            current_I=TEST_BOOZER_I,
        )

        self.assertIsInstance(value, float)
        np.testing.assert_allclose(gradient, np.array([3.0, -1.0]))
        self.assertEqual(residual_calls, [(1, True, TEST_BOOZER_I)])
        self.assertEqual(residual_dB_calls, [(0, True, TEST_BOOZER_I)])
        expected_point_count = objective.surface.quadpoints_phi.size * objective.surface.quadpoints_theta.size
        self.assertEqual(fake_bs.points.shape, (expected_point_count, 3))
        self.assertEqual(fake_bs.last_vjp_input.shape, (expected_point_count, 3))

    def test_boozer_residual_exact_no_longer_accepts_unused_constraint_weight(self):
        module = self.load_module()

        with self.assertRaises(TypeError):
            module.BoozerResidualExact(object(), object(), constraint_weight=0.0)

    def test_resolve_plasma_current_settings_accepts_physical_amps(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=8000.0,
            )
        )

        self.assertEqual(settings["input_source"], "physical_A")
        self.assertEqual(settings["mode"], "boozer_surrogate")
        self.assertEqual(settings["effective_mode"], "boozer_surrogate")
        self.assertEqual(settings["plasma_current_A"], 8000.0)
        self.assertAlmostEqual(settings["boozer_I"], 0.0016)

    def test_resolve_plasma_current_settings_zero_physical_amps_reports_vacuum_effective_mode(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=0.0,
            )
        )

        self.assertEqual(settings["input_source"], "physical_A")
        self.assertEqual(settings["mode"], "boozer_surrogate")
        self.assertEqual(settings["effective_mode"], "vacuum")
        self.assertEqual(settings["plasma_current_A"], 0.0)
        self.assertEqual(settings["boozer_I"], 0.0)

    def test_resolve_plasma_current_settings_accepts_negative_physical_amps(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=-35200.0,
            )
        )

        self.assertEqual(settings["plasma_current_A"], -35200.0)
        self.assertAlmostEqual(settings["boozer_I"], -0.00704)
        self.assertEqual(settings["effective_mode"], "boozer_surrogate")

    def test_resolve_plasma_current_settings_rejects_mixed_raw_and_physical_inputs(self):
        module = self.load_module()

        with self.assertRaisesRegex(ValueError, "--plasma-current-A"):
            module.resolve_plasma_current_settings(
                SimpleNamespace(
                    boozer_I=0.5,
                    plasma_current_A=8000.0,
                )
            )

    def test_resolve_plasma_current_settings_defaults_to_disabled_zero(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=None,
            )
        )

        self.assertEqual(settings["input_source"], "default_zero")
        self.assertEqual(settings["mode"], "disabled")
        self.assertEqual(settings["effective_mode"], "vacuum")
        self.assertEqual(settings["plasma_current_A"], 0.0)
        self.assertEqual(settings["boozer_I"], 0.0)

    def test_build_stage2_bs_path_uses_unique_globbed_current_match(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent = root / "local" / "outputs-demo.nc"
            parent.mkdir(parents=True)
            matched = (
                parent
                / "R0=0.915-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-TFC=80000-Order=2-CM=penalty-BH=3"
                / "biot_savart_opt.json"
            )
            matched.parent.mkdir(parents=True)
            matched.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.915,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
                stage2_source="local",
                local_stage2_root=str(root / "local"),
                database_stage2_root=str(root / "database"),
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(matched))

    def test_build_stage2_bs_path_rejects_ambiguous_globbed_current_matches(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent = root / "local" / "outputs-demo.nc"
            parent.mkdir(parents=True)
            for suffix in ("-CM=penalty-BH=3", "-CM=penalty-BH=4"):
                candidate = (
                    parent
                    / (
                        "R0=0.915-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-"
                        f"SR=0.220-TFC=80000-Order=2{suffix}"
                    )
                    / "biot_savart_opt.json"
                )
                candidate.parent.mkdir(parents=True)
                candidate.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.915,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
                stage2_source="local",
                local_stage2_root=str(root / "local"),
                database_stage2_root=str(root / "database"),
            )

            with self.assertRaisesRegex(
                FileNotFoundError,
                "Multiple Stage 2 outputs match the requested seed specification",
            ):
                module.build_stage2_bs_path(args)

    def test_fun_fallback_returns_elevated_j_and_same_sign_gradient(self):
        """Issue #2: failed Boozer must return elevated J + same-sign gradient,
        not (J_old, -dJ_old)."""
        module = self.load_module()

        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            x = np.ones(3)
            def is_self_intersecting(self):
                return False
            def volume(self):
                return 1.0
            def gamma(self):
                return np.zeros((1, 1, 3))

        class _BoozerSurface:
            surface = _Surface()
            res = {"success": False, "iota": TEST_IOTA, "G": TEST_G0}
            def run_code(self, iota, G):
                return self.res

        class _JF:
            x = np.zeros(5)

        surface_data = [{"boozer_surface": _BoozerSurface()}]
        module.run_dict = {
            "x_prev": np.zeros(5), "lscount": 0,
            "surface_state": {"sdofs": [np.ones(3)], "iota": [TEST_IOTA], "G": [TEST_G0]},
            "J": last_J, "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "accepted_x": np.zeros(5),
        }
        module.surface_data = surface_data
        module.VV = object()
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.0
        module.JF = _JF()

        J_out, dJ_out = module.fun(np.ones(5))

        self.assertGreater(J_out, last_J)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        self.assertIsNot(dJ_out, module.run_dict["dJ"])

    def test_evaluate_topology_gate_reports_early_surface_exit(self):
        module = self.load_module()

        class _Surface:
            def cross_section(self, phi, thetas):
                theta = np.asarray(thetas)
                return np.column_stack(
                    [
                        1.0 + 0.1 * np.cos(2 * np.pi * theta),
                        np.zeros_like(theta),
                        0.1 * np.sin(2 * np.pi * theta),
                    ]
                )

        class _Stop:
            def __init__(self, *args, **kwargs):
                pass

        fake_hits = [
            np.array([[0.8, -1.0, 1.0, 0.0, 0.0]]),
            np.array([]),
        ]

        with patch.object(module, "SurfaceClassifier", return_value=SimpleNamespace(dist=lambda xyz: 1.0)), patch.object(
            module, "LevelsetStoppingCriterion", _Stop
        ), patch.object(module, "MaxZStoppingCriterion", _Stop), patch.object(
            module, "MinZStoppingCriterion", _Stop
        ), patch.object(module, "MinRStoppingCriterion", _Stop), patch.object(
            module, "MaxRStoppingCriterion", _Stop
        ), patch.object(module, "compute_fieldlines", return_value=([], fake_hits)):
            status = module.evaluate_topology_gate(_Surface(), object(), 2, 2.0, 1e-7, 0.75)

        self.assertTrue(status["enabled"])
        self.assertFalse(status["success"])
        self.assertEqual(status["survived_lines"], 1)
        self.assertAlmostEqual(status["survival_fraction"], 0.5)
        self.assertEqual(status["first_exit_reason"], "surface_exit")
        self.assertAlmostEqual(status["first_exit_time"], 0.8)

    def test_topology_gate_rejection_increment_scales_with_deficit(self):
        module = self.load_module()

        status = {
            "enabled": True,
            "survival_fraction": 0.5,
            "survival_threshold": 0.75,
        }

        self.assertAlmostEqual(module.topology_gate_deficit(status), 0.25)
        self.assertAlmostEqual(
            module.topology_gate_rejection_increment(42.0, status, 4.0),
            84.0,
        )

    def test_fun_rejects_candidate_on_topology_gate_failure(self):
        module = self.load_module()

        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            x = np.ones(3)

            def is_self_intersecting(self):
                return False

            def volume(self):
                return 1.0

            def gamma(self):
                return np.zeros((1, 1, 3))

        class _BoozerSurface:
            surface = _Surface()
            res = {"success": True, "iota": TEST_IOTA, "G": TEST_G0}

            def run_code(self, iota, G):
                return self.res

        class _JF:
            x = np.zeros(5)

        surface_data = [{"boozer_surface": _BoozerSurface()}, {"boozer_surface": _BoozerSurface()}]
        module.run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "surface_state": {"sdofs": [np.ones(3), np.ones(3)], "iota": [TEST_IOTA, TEST_IOTA], "G": [TEST_G0, TEST_G0]},
            "J": last_J,
            "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "accepted_x": np.zeros(5),
        }
        module.surface_data = surface_data
        module.outer_surface_data = surface_data[-1]
        module.surface_iota_terms = [SimpleNamespace(J=lambda: TEST_IOTA), SimpleNamespace(J=lambda: TEST_IOTA)]
        module.VV = object()
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.0
        module.JF = _JF()
        module.bs = object()
        module.nonQSs = []
        module.brs = []
        module.Jiota = object()
        module.IOTAS_WEIGHT = 1.0
        module.JCurveLength = object()
        module.LENGTH_WEIGHT = 1.0
        module.JCurveCurve = object()
        module.CC_WEIGHT = 1.0
        module.JCurveSurface = object()
        module.CS_WEIGHT = 1.0
        module.JCurvature = object()
        module.CURVATURE_WEIGHT = 1.0
        module.JSurfSurf = None
        module.SURF_DIST_WEIGHT = 0.0
        module.RES_WEIGHT = 1.0
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.TOPOLOGY_GATE_FIELDLINES = 4
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1e-7
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.25
        module.TOPOLOGY_GATE_PENALTY_SCALE = 4.0

        with patch.object(
            module,
            "solve_surface_stack_at_dofs",
            return_value={
                "success": True,
                "solve_success": [True, True],
                "self_intersections": [False, False],
                "volumes_ordered": True,
                "gap_ok": True,
                "vessel_gap_ok": True,
                "nesting_ok": True,
                "adjacent_gaps": [0.01],
                "outer_vessel_gap": 0.1,
                "bad_nesting_phis": [],
            },
        ), patch.object(
            module,
            "evaluate_total_objective",
            return_value={"total": 7.0, "grad": np.arange(5, dtype=float), "surface_weights": np.array([1.0, 1.0])},
        ), patch.object(module, "restore_surface_states") as restore_mock, patch.object(
            module,
            "evaluate_topology_gate",
            return_value={
                "enabled": True,
                "success": False,
                "nfieldlines": 4,
                "survived_lines": 0,
                "survival_fraction": 0.0,
                "survival_threshold": 0.25,
                "tmax": 2.0,
                "tol": 1e-7,
                "stop_reason_counts": {"surface_exit": 4},
                "first_exit_time": 0.4,
                "first_exit_angle": 0.2,
                "first_exit_reason": "surface_exit",
            },
        ):
            J_out, dJ_out = module.fun(np.ones(5))

        self.assertEqual(J_out, 126.0)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        restore_mock.assert_called_once()

    def test_build_surface_configs_two_surface_mode_derives_inner_target_volume(self):
        module = self.load_module()

        class _FakeRZSurface:
            def __init__(self, label):
                self.label = label
                self._major_radius = 2.0
                self._volume = 10.0 * label
                self.nfp = 5
                self._dofs = np.array([1.0])

            def major_radius(self):
                return self._major_radius

            def get_dofs(self):
                return self._dofs.copy()

            def set_dofs(self, dofs):
                dofs = np.asarray(dofs)
                scale = dofs[0] / self._dofs[0]
                self._dofs = dofs
                self._major_radius *= scale
                self._volume *= scale ** 3

            def volume(self):
                return self._volume

        fake_factory = SimpleNamespace(
            from_wout=lambda *_args, **kwargs: _FakeRZSurface(kwargs["s"]),
        )

        with patch.object(module, "SurfaceRZFourier", fake_factory):
            configs = module.build_surface_configs(
                "dummy.nc",
                nphi=11,
                ntheta=13,
                seed_label=0.25,
                major_radius=1.0,
                outer_target_volume=0.10,
                num_surfaces=2,
                inner_surface_ratio=0.8,
            )

        self.assertEqual([config["name"] for config in configs], ["inner", "outer"])
        self.assertAlmostEqual(configs[0]["seed_label"], 0.20)
        self.assertAlmostEqual(configs[1]["seed_label"], 0.25)
        self.assertAlmostEqual(configs[0]["target_volume"], 0.08)
        self.assertAlmostEqual(configs[1]["target_volume"], 0.10)

    def test_build_surface_configs_single_surface_mode_keeps_outer_only_contract(self):
        module = self.load_module()

        class _FakeRZSurface:
            def __init__(self, label):
                self.label = label
                self._major_radius = 2.0
                self._volume = 10.0 * label
                self.nfp = 5
                self._dofs = np.array([1.0])

            def major_radius(self):
                return self._major_radius

            def get_dofs(self):
                return self._dofs.copy()

            def set_dofs(self, dofs):
                dofs = np.asarray(dofs)
                scale = dofs[0] / self._dofs[0]
                self._dofs = dofs
                self._major_radius *= scale
                self._volume *= scale ** 3

            def volume(self):
                return self._volume

        fake_factory = SimpleNamespace(
            from_wout=lambda *_args, **kwargs: _FakeRZSurface(kwargs["s"]),
        )

        with patch.object(module, "SurfaceRZFourier", fake_factory):
            configs = module.build_surface_configs(
                "dummy.nc",
                nphi=11,
                ntheta=13,
                seed_label=0.25,
                major_radius=1.0,
                outer_target_volume=0.10,
                num_surfaces=1,
                inner_surface_ratio=0.8,
            )

        self.assertEqual(len(configs), 1)
        self.assertEqual(configs[0]["name"], "outer")
        self.assertAlmostEqual(configs[0]["seed_label"], 0.25)
        self.assertAlmostEqual(configs[0]["target_volume"], 0.10)

    def test_average_surface_objectives_uses_mean_and_preserves_single_surface_scale(self):
        module = self.load_module()

        single = FakeAlgebraicObjective(2.0, [2.0, -1.0])
        single_avg = module.average_surface_objectives([single])
        self.assertAlmostEqual(single_avg.J(), 2.0)
        np.testing.assert_allclose(single_avg.dJ(), [2.0, -1.0])

        left = FakeAlgebraicObjective(2.0, [2.0, -1.0])
        right = FakeAlgebraicObjective(6.0, [4.0, 3.0])
        pair_avg = module.average_surface_objectives([left, right])
        self.assertAlmostEqual(pair_avg.J(), 4.0)
        np.testing.assert_allclose(pair_avg.dJ(), [3.0, 1.0])

    def test_build_surface_search_weights_ramps_inner_surface_only(self):
        module = self.load_module()

        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=1,
                accepted_iterations=0,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [1.0],
        )
        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=2,
                accepted_iterations=0,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [0.0, 1.0],
        )
        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=2,
                accepted_iterations=2,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [0.4, 1.0],
        )
        np.testing.assert_allclose(
            module.build_surface_search_weights(
                num_surfaces=2,
                accepted_iterations=5,
                ramp_iterations=5,
                initial_inner_weight=0.0,
            ),
            [1.0, 1.0],
        )

    def test_build_surface_search_gate_ramps_thresholds_and_nesting(self):
        module = self.load_module()

        single = module.build_surface_search_gate(
            num_surfaces=1,
            accepted_iterations=0,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
            vessel_gap_threshold=0.04,
        )
        self.assertEqual(single["surface_gap_threshold"], 0.005)
        self.assertEqual(single["vessel_gap_threshold"], 0.04)
        self.assertTrue(single["enforce_nesting"])
        self.assertEqual(single["gate_scale"], 1.0)

        start = module.build_surface_search_gate(
            num_surfaces=2,
            accepted_iterations=0,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
            vessel_gap_threshold=0.04,
        )
        self.assertEqual(start["surface_gap_threshold"], 0.0)
        self.assertEqual(start["vessel_gap_threshold"], 0.0)
        self.assertFalse(start["enforce_nesting"])
        self.assertEqual(start["gate_scale"], 0.0)

        mid = module.build_surface_search_gate(
            num_surfaces=2,
            accepted_iterations=2,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
            vessel_gap_threshold=0.04,
        )
        self.assertAlmostEqual(mid["surface_gap_threshold"], 0.002)
        self.assertAlmostEqual(mid["vessel_gap_threshold"], 0.016)
        self.assertFalse(mid["enforce_nesting"])
        self.assertAlmostEqual(mid["gate_scale"], 0.4)

        done = module.build_surface_search_gate(
            num_surfaces=2,
            accepted_iterations=5,
            ramp_iterations=5,
            initial_inner_weight=0.0,
            surface_gap_threshold=0.005,
            vessel_gap_threshold=0.04,
        )
        self.assertAlmostEqual(done["surface_gap_threshold"], 0.005)
        self.assertAlmostEqual(done["vessel_gap_threshold"], 0.04)
        self.assertTrue(done["enforce_nesting"])
        self.assertAlmostEqual(done["gate_scale"], 1.0)


class HardwareConstraintTests(unittest.TestCase):
    def load_module(self):
        return load_single_stage_example_module()

    def _run_fun_with_hardware_violation(
        self,
        *,
        hardware_search_mode="hard",
        hardware_search_soft_iterations=0,
        accepted_iterations=0,
    ):
        module = load_single_stage_example_module()

        last_J = 12.0
        last_dJ = np.array([1.0, -1.0, 2.0])

        class _Surface:
            x = np.ones(2)

            def is_self_intersecting(self):
                return False

            def volume(self):
                return 1.0

            def gamma(self):
                return np.zeros((1, 1, 3))

        class _BoozerSurface:
            surface = _Surface()
            res = {"success": True, "iota": TEST_IOTA, "G": TEST_G0}

            def run_code(self, iota, G):
                return self.res

        class _JF:
            x = np.zeros(3)

        class _DistanceObjective:
            def __init__(self, distance):
                self.distance = distance

            def shortest_distance(self):
                return self.distance

        class _CurvatureObjective:
            def J(self):
                return 1.0

            def dJ(self):
                return np.zeros(3)

        class _LengthObjective:
            def J(self):
                return 1.0

            def dJ(self):
                return np.zeros(3)

        class _Curve:
            def kappa(self):
                return np.array([41.0])

            def gamma(self):
                return np.zeros((2, 3))

        surface_data = [{"boozer_surface": _BoozerSurface()}, {"boozer_surface": _BoozerSurface()}]
        module.run_dict = {
            "x_prev": np.zeros(3),
            "lscount": 0,
            "surface_state": {
                "sdofs": [np.ones(2), np.ones(2)],
                "iota": [TEST_IOTA, TEST_IOTA],
                "G": [TEST_G0, TEST_G0],
            },
            "J": last_J,
            "dJ": last_dJ.copy(),
            "accepted_iterations": accepted_iterations,
            "accepted_x": np.zeros(3),
            "trial_hardware_status": None,
            "accepted_hardware_status": None,
        }
        module.surface_data = surface_data
        module.outer_surface_data = surface_data[-1]
        module.surface_iota_terms = [SimpleNamespace(J=lambda: TEST_IOTA), SimpleNamespace(J=lambda: TEST_IOTA)]
        module.VV = object()
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.04
        module.JF = _JF()
        module.bs = object()
        module.nonQSs = []
        module.brs = []
        module.Jiota = object()
        module.IOTAS_WEIGHT = 1.0
        module.JCurveLength = _LengthObjective()
        module.LENGTH_WEIGHT = 1.0
        module.JCurveCurve = _DistanceObjective(0.04)
        module.CC_WEIGHT = 1.0
        module.CC_DIST = 0.05
        module.JCurveSurface = _DistanceObjective(0.03)
        module.CS_WEIGHT = 1.0
        module.CS_DIST = 0.02
        module.JCurvature = _CurvatureObjective()
        module.CURVATURE_WEIGHT = 1.0
        module.CURVATURE_THRESHOLD = 40.0
        module.JSurfSurf = _DistanceObjective(0.05)
        module.SURF_DIST_WEIGHT = 1.0
        module.RES_WEIGHT = 1.0
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.TOPOLOGY_GATE_FIELDLINES = 0
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1e-7
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.25
        module.TOPOLOGY_GATE_PENALTY_SCALE = 4.0
        module.HARDWARE_SEARCH_MODE = hardware_search_mode
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = hardware_search_soft_iterations
        module.banana_curve = _Curve()

        with patch.object(
            module,
            "solve_surface_stack_at_dofs",
            return_value={
                "success": True,
                "solve_success": [True, True],
                "self_intersections": [False, False],
                "volumes_ordered": True,
                "gap_ok": True,
                "vessel_gap_ok": True,
                "nesting_ok": True,
                "adjacent_gaps": [0.1],
                "outer_vessel_gap": 0.05,
                "bad_nesting_phis": [],
            },
        ), patch.object(
            module,
            "evaluate_total_objective",
            return_value={
                "total": 7.0,
                "grad": np.arange(3, dtype=float),
                "surface_weights": np.array([1.0, 1.0]),
                "J_QS": 0.0,
                "dJ_QS": np.zeros(3),
                "J_Boozer": 0.0,
                "dJ_Boozer": np.zeros(3),
                "J_iota": 0.0,
                "dJ_iota": np.zeros(3),
                "J_surf": 0.0,
                "dJ_surf": np.zeros(3),
                "J_curvature": 0.0,
                "dJ_curvature": np.zeros(3),
            },
        ), patch.object(module, "restore_surface_states") as restore_mock, patch.object(
            module,
            "evaluate_search_topology_gate",
            return_value={
                "enabled": False,
                "success": True,
                "nfieldlines": 0,
                "survived_lines": 0,
                "survival_fraction": 1.0,
                "survival_threshold": 0.25,
                "tmax": 2.0,
                "tol": 1e-7,
                "stop_reason_counts": {},
                "first_exit_time": None,
                "first_exit_angle": None,
                "first_exit_reason": None,
            },
        ):
            J_out, dJ_out = module.fun(np.ones(3))

        return module, J_out, dJ_out, last_dJ, restore_mock

    def test_stage2_hardware_constraints_report_each_violation(self):
        module = load_stage2_module()

        status = module.evaluate_stage2_hardware_constraints(
            coil_length=1.8,
            length_target=1.75,
            curve_curve_min_dist=0.04,
            cc_threshold=0.05,
            max_curvature=41.0,
            curvature_threshold=40.0,
        )

        self.assertFalse(status["success"])
        self.assertEqual(len(status["violations"]), 3)
        self.assertIn("coil_length", status["violations"][0])
        self.assertIn("coil_coil_min_dist", status["violations"][1])
        self.assertIn("max_curvature", status["violations"][2])

    def test_single_stage_hardware_constraints_report_each_violation(self):
        module = load_single_stage_example_module()

        status = module.evaluate_single_stage_hardware_constraints(
            curve_curve_min_dist=0.04,
            cc_dist=0.05,
            curve_surface_min_dist=0.01,
            cs_dist=0.02,
            surface_vessel_min_dist=0.03,
            ss_dist=0.04,
            max_curvature=41.0,
            curvature_threshold=40.0,
        )

        self.assertFalse(status["success"])
        self.assertEqual(len(status["violations"]), 4)
        self.assertIn("coil_coil_min_dist", status["violations"][0])
        self.assertIn("coil_surface_min_dist", status["violations"][1])
        self.assertIn("surface_vessel_min_dist", status["violations"][2])
        self.assertIn("max_curvature", status["violations"][3])

    def test_surface_vessel_min_dist_uses_single_source_for_results(self):
        module = load_single_stage_example_module()

        class _DistanceObjective:
            def shortest_distance(self):
                return 0.123

        class _Surface:
            def __init__(self, points):
                self._points = np.asarray(points, dtype=float).reshape((-1, 1, 3))

            def gamma(self):
                return self._points

        self.assertEqual(
            module.compute_single_stage_surface_vessel_min_dist(
                _DistanceObjective(),
                {"outer_vessel_gap": 0.456},
            ),
            0.123,
        )
        self.assertEqual(
            module.compute_single_stage_surface_vessel_min_dist(
                None,
                {"outer_vessel_gap": 0.456},
            ),
            0.456,
        )
        self.assertEqual(
            module.compute_single_stage_surface_vessel_min_dist(
                None,
                {"outer_vessel_gap": None},
                _Surface([[0.0, 0.0, 0.0]]),
                _Surface([[0.0, 0.3, 0.4]]),
            ),
            0.5,
        )

    def test_refinement_eligible_incumbent_requires_accepted_hardware_pass(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 4.0},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": False, "violations": ["coil_coil_min_dist"]},
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
        }

        self.assertFalse(module.refinement_eligible_incumbent(run_dict))

        run_dict["accepted_hardware_status"] = {"success": True, "violations": []}
        self.assertTrue(module.refinement_eligible_incumbent(run_dict))

    def test_maybe_update_best_feasible_incumbent_uses_search_total_metric(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 4.0, "surface_weights": np.array([1.0])},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
        }

        self.assertTrue(module.maybe_update_best_feasible_incumbent(run_dict, "initial"))
        self.assertEqual(run_dict["best_feasible_metric"], 4.0)
        self.assertEqual(run_dict["best_feasible_stage"], "initial")
        np.testing.assert_allclose(run_dict["best_feasible_incumbent"].x, [1.0, 2.0])

        run_dict["search_eval"] = {"total": 5.0, "surface_weights": np.array([1.0])}
        run_dict["J"] = 5.0
        self.assertFalse(module.maybe_update_best_feasible_incumbent(run_dict, "final"))
        self.assertEqual(run_dict["best_feasible_metric"], 4.0)
        self.assertEqual(run_dict["best_feasible_stage"], "initial")

        run_dict["search_eval"] = {"total": 3.0, "surface_weights": np.array([1.0])}
        run_dict["J"] = 3.0
        self.assertTrue(module.maybe_update_best_feasible_incumbent(run_dict, "final"))
        self.assertEqual(run_dict["best_feasible_metric"], 3.0)
        self.assertEqual(run_dict["best_feasible_stage"], "final")

    def test_validate_boozer_stage_refinement_args_rejects_unsupported_scope(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            boozer_stage_refinement=True,
            constraint_method="alm",
            num_surfaces=1,
            basin_hops=0,
            boozer_stage="initial",
            refinement_boozer_stage="final",
            refinement_maxiter=20,
            refinement_chunk_maxiter=10,
            refinement_max_stalled_chunks=2,
        )

        with self.assertRaisesRegex(ValueError, "--constraint-method=penalty"):
            module.validate_boozer_stage_refinement_args(args, constraint_weight=1.0)

        args.constraint_method = "penalty"
        args.num_surfaces = 2
        with self.assertRaisesRegex(ValueError, "--num-surfaces=1"):
            module.validate_boozer_stage_refinement_args(args, constraint_weight=1.0)

        args.num_surfaces = 1
        args.refinement_chunk_maxiter = 0
        with self.assertRaisesRegex(ValueError, "--refinement-chunk-maxiter must be positive"):
            module.validate_boozer_stage_refinement_args(args, constraint_weight=1.0)

    def test_refinement_improves_phase1_metric_uses_phase1_stage_basis(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 6.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 6.0, "surface_weights": np.array([1.0])},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "accepted_iterations": 0,
        }
        refinement_incumbent = module.snapshot_single_stage_incumbent_state(run_dict)
        rebuild_calls = []

        def fake_rebuild(stage_name):
            rebuild_calls.append(stage_name)

        with patch.object(
            module,
            "refresh_accepted_search_state",
            autospec=True,
            side_effect=lambda current_run_dict, stage_name: 2.5 if stage_name == "initial" else 9.0,
        ):
            refinement_metric, refinement_improved = module.refinement_improves_phase1_metric(
                3.0,
                "initial",
                run_dict,
                refinement_incumbent,
                fake_rebuild,
            )

        self.assertEqual(refinement_metric, 2.5)
        self.assertTrue(refinement_improved)
        self.assertEqual(rebuild_calls, ["initial"])

    def test_reported_boozer_stage_follows_saved_final_source(self):
        module = load_single_stage_example_module()
        self.assertEqual(module.reported_boozer_stage("initial", "final"), "final")
        self.assertEqual(module.reported_boozer_stage("initial", None), "initial")

    def test_run_chunked_refinement_aborts_after_stalled_chunks_without_improvement(self):
        module = load_single_stage_example_module()
        phase1_incumbent = SimpleNamespace(name="phase1")
        stalled_incumbent = SimpleNamespace(name="stalled")
        chunk_results = [
            (SimpleNamespace(nit=5, success=False, message="chunk1"), stalled_incumbent),
            (SimpleNamespace(nit=4, success=False, message="chunk2"), stalled_incumbent),
        ]

        with patch.object(module, "run_refinement_chunk", side_effect=chunk_results), patch.object(
            module,
            "refinement_improves_phase1_metric",
            side_effect=[(6.0, False), (6.0, False)],
        ):
            result = module.run_chunked_refinement(
                {},
                phase1_incumbent,
                5.0,
                "initial",
                "final",
                lambda stage_name: None,
                20,
                10,
                2,
                300,
                1e-9,
                1e-9,
            )

        self.assertIsNone(result["best_incumbent"])
        self.assertEqual(result["iterations"], 9)
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["abort_reason"], "stalled_without_improvement")
        self.assertEqual(result["termination_message"], "chunk2; stalled_without_improvement")

    def test_run_chunked_refinement_keeps_best_improvement_after_later_stall(self):
        module = load_single_stage_example_module()
        phase1_incumbent = SimpleNamespace(name="phase1")
        improved_incumbent = SimpleNamespace(name="improved")
        chunk_results = [
            (SimpleNamespace(nit=6, success=False, message="chunk1"), improved_incumbent),
            (SimpleNamespace(nit=3, success=False, message="chunk2"), improved_incumbent),
        ]

        with patch.object(module, "run_refinement_chunk", side_effect=chunk_results), patch.object(
            module,
            "refinement_improves_phase1_metric",
            side_effect=[(4.0, True), (4.0, False)],
        ):
            result = module.run_chunked_refinement(
                {},
                phase1_incumbent,
                5.0,
                "initial",
                "final",
                lambda stage_name: None,
                20,
                10,
                1,
                300,
                1e-9,
                1e-9,
            )

        self.assertIs(result["best_incumbent"], improved_incumbent)
        self.assertEqual(result["best_metric"], 4.0)
        self.assertEqual(result["iterations"], 9)
        self.assertEqual(result["chunks"], 2)
        self.assertEqual(result["abort_reason"], "stalled_after_improvement")
        self.assertEqual(result["termination_message"], "chunk2; stalled_after_improvement")

    def test_run_chunked_refinement_reports_budget_exhaustion_after_improvement(self):
        module = load_single_stage_example_module()
        phase1_incumbent = SimpleNamespace(name="phase1")
        improved_incumbent = SimpleNamespace(name="improved")

        with patch.object(
            module,
            "run_refinement_chunk",
            return_value=(SimpleNamespace(nit=5, success=False, message="chunk1"), improved_incumbent),
        ), patch.object(
            module,
            "refinement_improves_phase1_metric",
            return_value=(4.0, True),
        ):
            result = module.run_chunked_refinement(
                {},
                phase1_incumbent,
                5.0,
                "initial",
                "final",
                lambda stage_name: None,
                5,
                5,
                2,
                300,
                1e-9,
                1e-9,
            )

        self.assertIs(result["best_incumbent"], improved_incumbent)
        self.assertEqual(result["best_metric"], 4.0)
        self.assertEqual(result["abort_reason"], "budget_exhausted_after_improvement")
        self.assertEqual(result["termination_message"], "chunk1; budget_exhausted_after_improvement")

    def test_summarize_refinement_result_uses_refinement_status(self):
        module = load_single_stage_example_module()
        accepted_x = np.array([1.0, 2.0])
        refinement_result = {
            "termination_message": "stalled_without_improvement",
            "success": False,
        }

        termination_message, optimizer_success, result = module.summarize_refinement_result(
            refinement_result,
            total_iterations=13,
            accepted_x=accepted_x,
        )

        self.assertEqual(termination_message, "stalled_without_improvement")
        self.assertFalse(optimizer_success)
        self.assertEqual(result.nit, 13)
        self.assertEqual(result.message, "stalled_without_improvement")
        self.assertFalse(result.success)
        np.testing.assert_array_equal(result.x, accepted_x)

    def test_fun_rejects_candidate_on_hardware_constraint_failure(self):
        module, J_out, dJ_out, last_dJ, restore_mock = self._run_fun_with_hardware_violation(
            hardware_search_mode="hard",
        )

        self.assertEqual(J_out, 24.0)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        restore_mock.assert_called_once()
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])
        self.assertIsNone(module.run_dict["accepted_hardware_status"])

    def test_fun_warns_only_on_hardware_constraint_failure_in_warn_mode(self):
        module, J_out, dJ_out, _last_dJ, restore_mock = self._run_fun_with_hardware_violation(
            hardware_search_mode="warn",
        )

        self.assertEqual(J_out, 7.0)
        np.testing.assert_array_equal(dJ_out, np.arange(3, dtype=float))
        restore_mock.assert_not_called()
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])
        self.assertIsNone(module.run_dict["accepted_hardware_status"])

    def test_fun_rejects_hardware_violation_in_adaptive_mode_when_gate_not_relaxed(self):
        module, J_out, dJ_out, _last_dJ, restore_mock = self._run_fun_with_hardware_violation(
            hardware_search_mode="adaptive",
            hardware_search_soft_iterations=1,
            accepted_iterations=0,
        )

        self.assertEqual(J_out, 24.0)
        np.testing.assert_array_equal(dJ_out, np.array([1.0, -1.0, 2.0]))
        restore_mock.assert_called_once()
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])
        self.assertIsNone(module.run_dict["accepted_hardware_status"])

    def test_fun_warns_in_adaptive_mode_only_while_gate_is_relaxed(self):
        module, J_out, dJ_out, _last_dJ, restore_mock = self._run_fun_with_hardware_violation(
            hardware_search_mode="adaptive",
            hardware_search_soft_iterations=1,
            accepted_iterations=0,
        )

        module.MULTISURFACE_RAMP_ITERATIONS = 5
        module.INNER_SURFACE_INITIAL_WEIGHT = 0.0
        module.run_dict["accepted_iterations"] = 0

        with patch.object(module, "restore_surface_states") as adaptive_restore_mock, patch.object(
            module,
            "solve_surface_stack_at_dofs",
            return_value={
                "success": True,
                "solve_success": [True, True],
                "self_intersections": [False, False],
                "volumes_ordered": True,
                "gap_ok": True,
                "vessel_gap_ok": True,
                "nesting_ok": True,
                "adjacent_gaps": [0.1],
                "outer_vessel_gap": 0.05,
                "bad_nesting_phis": [],
            },
        ), patch.object(
            module,
            "evaluate_total_objective",
            return_value={
                "total": 7.0,
                "grad": np.arange(3, dtype=float),
                "surface_weights": np.array([0.0, 1.0]),
                "J_QS": 0.0,
                "dJ_QS": np.zeros(3),
                "J_Boozer": 0.0,
                "dJ_Boozer": np.zeros(3),
                "J_iota": 0.0,
                "dJ_iota": np.zeros(3),
                "J_surf": 0.0,
                "dJ_surf": np.zeros(3),
                "J_curvature": 0.0,
                "dJ_curvature": np.zeros(3),
            },
        ):
            J_out, dJ_out = module.fun(np.ones(3))

        self.assertEqual(J_out, 7.0)
        np.testing.assert_array_equal(dJ_out, np.arange(3, dtype=float))
        restore_mock.assert_called_once()
        adaptive_restore_mock.assert_not_called()
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])

    def test_callback_records_accepted_invalid_hardware_status_after_warn_mode_step(self):
        module, J_out, _dJ_out, _last_dJ, restore_mock = self._run_fun_with_hardware_violation(
            hardware_search_mode="warn",
        )

        self.assertEqual(J_out, 7.0)
        restore_mock.assert_not_called()

        class _Surface:
            nfp = 5

            def __init__(self):
                self.x = np.array([0.1])

            def volume(self):
                return 1.0

            def gamma(self):
                return np.array([[[0.0, 0.0, 0.0]]])

            def unitnormal(self):
                return np.array([[[1.0, 0.0, 0.0]]])

        class _ScalarObjective:
            def __init__(self, value):
                self._value = value

            def J(self):
                return self._value

            def dJ(self):
                return np.array([self._value, -self._value, 0.0])

        class _DistanceObjective(_ScalarObjective):
            def __init__(self, value, min_distance):
                super().__init__(value)
                self._min_distance = min_distance

            def shortest_distance(self):
                return self._min_distance

        class _Curve:
            def gamma(self):
                return np.array([[1.0, 0.0, 0.0]])

            def kappa(self):
                return np.array([41.0])

        class _CurveLength:
            def J(self):
                return 1.7

        class _BS:
            def set_points(self, pts):
                self._points = pts

            def B(self):
                return np.array([[1.0, 0.0, 0.0]])

        surface = _Surface()
        surface_entry = {
            "name": "outer",
            "seed_label": 0.16,
            "target_volume": 1.0,
            "boozer_surface": SimpleNamespace(
                surface=surface,
                res={"success": True, "iota": TEST_IOTA, "G": TEST_G0},
            ),
        }
        objective_eval = {
            "total": 7.0,
            "grad": np.arange(3, dtype=float),
            "surface_weights": np.array([1.0]),
            "J_QS": 0.0,
            "dJ_QS": np.zeros(3),
            "J_Boozer": 0.0,
            "dJ_Boozer": np.zeros(3),
            "J_iota": 0.0,
            "dJ_iota": np.zeros(3),
            "J_surf": 0.0,
            "dJ_surf": np.zeros(3),
            "J_curvature": 0.0,
            "dJ_curvature": np.zeros(3),
        }
        stack_status = {
            "success": True,
            "solve_success": [True],
            "self_intersections": [False],
            "volumes_ordered": True,
            "gap_ok": True,
            "vessel_gap_ok": True,
            "nesting_ok": True,
            "adjacent_gaps": [],
            "outer_vessel_gap": None,
            "bad_nesting_phis": [],
        }
        accepted_surface_state = {
            "sdofs": [np.array([0.1])],
            "iota": [TEST_IOTA],
            "G": [TEST_G0],
        }
        hardware_snapshot = {
            "curve_curve_min_dist": 0.04,
            "curve_surface_min_dist": 0.03,
            "surface_vessel_min_dist": 0.0,
            "max_curvature": 41.0,
            "status": {
                "success": False,
                "violations": ["coil_coil_min_dist=0.040000 < threshold=0.050000"],
            },
        }

        module.surface_data = [surface_entry]
        module.outer_surface_data = surface_entry
        module.surface_iota_terms = [SimpleNamespace(J=lambda: TEST_IOTA)]
        module.JCurveLength = _ScalarObjective(0.44)
        module.JCurveCurve = _DistanceObjective(0.55, 0.04)
        module.JCurveSurface = _DistanceObjective(0.77, 0.03)
        module.JCurvature = _ScalarObjective(0.99)
        module.JSurfSurf = None
        module.banana_curve = _Curve()
        module.curvelength = _CurveLength()
        module.bs = _BS()
        module.VV = object()
        module.CHECKPOINT_EVERY = 0
        module.TOPOLOGY_SCORER_EVERY = 0
        module.CONSTRAINT_METHOD = "penalty"
        module.run_dict["surface_state"] = accepted_surface_state
        module.run_dict["it"] = 1

        with tempfile.TemporaryDirectory() as tmpdir:
            module.OUT_DIR_ITER = tmpdir

            with patch.object(
                module,
                "evaluate_search_objective",
                return_value=objective_eval,
            ), patch.object(
                module,
                "snapshot_surface_states",
                return_value=accepted_surface_state,
            ), patch.object(
                module,
                "evaluate_surface_stack",
                return_value=stack_status,
            ), patch.object(
                module,
                "evaluate_single_stage_hardware_snapshot",
                return_value=hardware_snapshot,
            ):
                module.callback(np.ones(3))

            self.assertFalse(module.run_dict["accepted_hardware_status"]["success"])
            log_text = (Path(tmpdir) / "log.txt").read_text()

        self.assertIn("Hardware Constraints OK", log_text)
        self.assertIn("Hardware Violations", log_text)
        self.assertIn("coil_coil_min_dist=0.040000 < threshold=0.050000", log_text)

    def test_alm_rejection_preserves_constraint_metadata_for_outer_updates(self):
        module = load_single_stage_example_module()
        module.CONSTRAINT_METHOD = "alm"
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.04
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1e-7
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.25
        module.JF = SimpleNamespace(x=np.zeros(2))
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=object())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {
                "constraint_values": np.array([0.4, 0.1, 0.0]),
                "max_violation": 0.4,
                "stationarity_norm": 2.5,
                "constraint_names": ["coil_coil_spacing", "coil_surface_spacing", "max_curvature"],
                "base_total": 5.0,
            },
        }

        with patch.object(
            module,
            "solve_surface_stack_at_dofs",
            return_value={
                "success": False,
                "solve_success": [False],
                "self_intersections": [False],
                "volumes_ordered": True,
                "gap_ok": True,
                "vessel_gap_ok": True,
                "nesting_ok": True,
                "adjacent_gaps": [],
                "outer_vessel_gap": None,
                "bad_nesting_phis": [],
            },
        ), patch.object(module, "restore_surface_states") as restore_mock:
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertEqual(evaluation["total"], 14.0)
        np.testing.assert_array_equal(evaluation["grad"], np.array([3.0, -1.0]))
        np.testing.assert_array_equal(
            evaluation["constraint_values"],
            np.array([0.4, 0.1, 0.0]),
        )
        self.assertAlmostEqual(evaluation["max_violation"], 0.4)
        self.assertAlmostEqual(evaluation["stationarity_norm"], 2.5)
        self.assertEqual(
            evaluation["constraint_names"],
            ["coil_coil_spacing", "coil_surface_spacing", "max_curvature"],
        )
        self.assertAlmostEqual(evaluation["base_total"], 5.0)
        restore_mock.assert_called_once()

    def test_build_scaled_outer_problem_scales_coordinates_gradients_and_callback(self):
        module = self.load_module()
        seen = {"fun": [], "callback": []}

        def base_fun(x):
            seen["fun"].append(np.asarray(x, dtype=float).copy())
            return 7.5, np.array([3.0, -4.0])

        def base_callback(x):
            seen["callback"].append(np.asarray(x, dtype=float).copy())

        scaled_fun, scaled_callback = module.build_scaled_outer_problem(
            base_fun,
            base_callback,
            np.array([10.0, 20.0]),
            0.1,
        )

        J, dJ = scaled_fun(np.array([1.0, -2.0]))
        self.assertAlmostEqual(J, 7.5)
        np.testing.assert_allclose(dJ, [0.3, -0.4])
        np.testing.assert_allclose(seen["fun"][0], [10.1, 19.8])

        scaled_callback(np.array([1.0, -2.0]))
        np.testing.assert_allclose(seen["callback"][0], [10.1, 19.8])

    def test_resolve_initial_step_phase_maxiter(self):
        module = self.load_module()

        self.assertEqual(module.resolve_initial_step_phase_maxiter(40, 1.0, 10), 0)
        self.assertEqual(module.resolve_initial_step_phase_maxiter(40, 0.5, 0), 0)
        self.assertEqual(module.resolve_initial_step_phase_maxiter(40, 0.5, 10), 10)
        self.assertEqual(module.resolve_initial_step_phase_maxiter(5, 0.5, 10), 5)

    def test_evaluate_total_objective_uses_surface_weights_for_qs_and_boozer_terms(self):
        module = self.load_module()

        nonqs = [
            FakeAlgebraicObjective(2.0, [2.0, 0.0]),
            FakeAlgebraicObjective(6.0, [4.0, 0.0]),
        ]
        brs = [
            FakeAlgebraicObjective(10.0, [1.0, 1.0]),
            FakeAlgebraicObjective(20.0, [3.0, 3.0]),
        ]
        zero = FakeAlgebraicObjective(0.0, [0.0, 0.0])

        outer_only = module.evaluate_total_objective(
            np.array([0.0, 1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=zero,
            IOTAS_WEIGHT=3.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=4.0,
            JCurveCurve=zero,
            CC_WEIGHT=5.0,
            JCurveSurface=zero,
            CS_WEIGHT=6.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=7.0,
        )
        self.assertAlmostEqual(outer_only["J_QS"], 6.0)
        self.assertAlmostEqual(outer_only["J_Boozer"], 20.0)
        np.testing.assert_allclose(outer_only["dJ_QS"], [4.0, 0.0])
        np.testing.assert_allclose(outer_only["dJ_Boozer"], [3.0, 3.0])
        self.assertAlmostEqual(outer_only["total"], 46.0)
        np.testing.assert_allclose(outer_only["grad"], [10.0, 6.0])

        ramped = module.evaluate_total_objective(
            np.array([0.5, 1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=zero,
            IOTAS_WEIGHT=3.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=4.0,
            JCurveCurve=zero,
            CC_WEIGHT=5.0,
            JCurveSurface=zero,
            CS_WEIGHT=6.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=7.0,
        )
        self.assertAlmostEqual(ramped["J_QS"], (0.5 * 2.0 + 6.0) / 1.5)
        self.assertAlmostEqual(ramped["J_Boozer"], (0.5 * 10.0 + 20.0) / 1.5)
        np.testing.assert_allclose(ramped["dJ_QS"], [10.0 / 3.0, 0.0])
        np.testing.assert_allclose(ramped["dJ_Boozer"], [7.0 / 3.0, 7.0 / 3.0])
        self.assertAlmostEqual(ramped["total"], 38.0)
        np.testing.assert_allclose(ramped["grad"], [8.0, 14.0 / 3.0])

    def test_build_total_objective_skips_missing_surface_vessel_term_without_injecting_int(self):
        module = self.load_module()

        total = module.build_total_objective(
            FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            FakeAlgebraicObjective(7.0, [2.0, 0.0]),
            8.0,
            FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            SURF_DIST_WEIGHT=1000.0,
            JSurfSurf=None,
        )

        self.assertAlmostEqual(total.J(), 1 + 2*3 + 4*5 + 6*7 + 8*9 + 10*11 + 12*13)
        np.testing.assert_allclose(total.dJ(), [33.0, 28.0])

    def test_evaluate_surface_stack_rejects_unordered_or_too_close_surfaces(self):
        module = self.load_module()

        class _FakeSurface:
            def __init__(self, volume, points, self_intersecting=False):
                self._volume = volume
                self._points = np.asarray(points, dtype=float).reshape((-1, 1, 3))
                self._self_intersecting = self_intersecting

            def volume(self):
                return self._volume

            def gamma(self):
                return self._points

            def is_self_intersecting(self):
                return self._self_intersecting

        good_stack = [
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.08, [[0.0, 0.0, 0.0]]), res={"success": True, "iota": 0.12})},
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.10, [[0.4, 0.0, 0.0]]), res={"success": True, "iota": 0.15})},
        ]
        vessel = _FakeSurface(0.2, [[1.0, 0.0, 0.0]])
        good_status = module.evaluate_surface_stack(good_stack, vessel_surface=vessel, surface_gap_threshold=0.1, vessel_gap_threshold=0.1)
        self.assertTrue(good_status["success"])
        self.assertEqual(good_status["adjacent_gaps"], [0.4])

        bad_order = [
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.11, [[0.0, 0.0, 0.0]]), res={"success": True, "iota": 0.12})},
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.10, [[0.4, 0.0, 0.0]]), res={"success": True, "iota": 0.15})},
        ]
        order_status = module.evaluate_surface_stack(bad_order)
        self.assertFalse(order_status["success"])
        self.assertFalse(order_status["volumes_ordered"])

        bad_gap = [
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.08, [[0.0, 0.0, 0.0]]), res={"success": True, "iota": 0.12})},
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.10, [[0.05, 0.0, 0.0]]), res={"success": True, "iota": 0.15})},
        ]
        gap_status = module.evaluate_surface_stack(bad_gap, surface_gap_threshold=0.1)
        self.assertFalse(gap_status["success"])
        self.assertFalse(gap_status["gap_ok"])

    def test_evaluate_surface_stack_rejects_cross_section_nesting_failure(self):
        module = self.load_module()

        class _FakeSurface:
            nfp = 5

            def __init__(self, volume, cross_section):
                self._volume = volume
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._cross_section.reshape((-1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        inner_crossing = [
            [0.9, 0.0, -0.2],
            [1.4, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [0.6, 0.0, 0.0],
        ]
        outer_box = [
            [0.7, 0.0, -0.3],
            [1.3, 0.0, -0.3],
            [1.3, 0.0, 0.3],
            [0.7, 0.0, 0.3],
        ]
        surface_data = [
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.08, inner_crossing), res={"success": True, "iota": 0.12})},
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.10, outer_box), res={"success": True, "iota": 0.15})},
        ]

        status = module.evaluate_surface_stack(surface_data)
        self.assertFalse(status["success"])
        self.assertFalse(status["nesting_ok"])
        self.assertTrue(status["bad_nesting_phis"])

    def test_evaluate_surface_stack_can_skip_nesting_during_search_continuation(self):
        module = self.load_module()

        class _FakeSurface:
            nfp = 5

            def __init__(self, volume, point, cross_section):
                self._volume = volume
                self._point = np.asarray(point, dtype=float)
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        inner_crossing = [
            [0.9, 0.0, -0.2],
            [1.4, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [0.6, 0.0, 0.0],
        ]
        outer_box = [
            [0.7, 0.0, -0.3],
            [1.3, 0.0, -0.3],
            [1.3, 0.0, 0.3],
            [0.7, 0.0, 0.3],
        ]
        surface_data = [
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.08, [0.0, 0.0, 0.0], inner_crossing), res={"success": True, "iota": 0.12})},
            {"boozer_surface": SimpleNamespace(surface=_FakeSurface(0.10, [0.4, 0.0, 0.0], outer_box), res={"success": True, "iota": 0.15})},
        ]

        relaxed = module.evaluate_surface_stack(surface_data, enforce_nesting=False)
        self.assertTrue(relaxed["success"])
        self.assertTrue(relaxed["nesting_ok"])
        self.assertEqual(relaxed["bad_nesting_phis"], [])

        strict = module.evaluate_surface_stack(surface_data, enforce_nesting=True)
        self.assertFalse(strict["success"])
        self.assertFalse(strict["nesting_ok"])

    def test_fun_multisurface_fallback_restores_all_surface_state(self):
        module = self.load_module()
        last_J = 42.0
        last_dJ = np.array([1.0, -2.0, 3.0, -4.0, 5.0])

        class _Surface:
            nfp = 5

            def __init__(self, volume, point):
                self.x = np.array([volume])
                self._volume = volume
                self._point = np.asarray(point, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return np.array([[1.0, 0.0, -0.1], [1.1, 0.0, 0.0], [1.0, 0.0, 0.1], [0.9, 0.0, 0.0]])

        class _BoozerSurface:
            def __init__(self, surface, success):
                self.surface = surface
                self.res = {"success": success, "iota": TEST_IOTA, "G": TEST_G0}

            def run_code(self, iota, G):
                self.res["iota"] = iota + 0.1
                self.res["G"] = G + 0.2
                return self.res

        class _JF:
            x = np.zeros(5)

        inner = _BoozerSurface(_Surface(0.08, [0.0, 0.0, 0.0]), True)
        outer = _BoozerSurface(_Surface(0.10, [0.4, 0.0, 0.0]), False)
        module.surface_data = [
            {"boozer_surface": inner},
            {"boozer_surface": outer},
        ]
        module.run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "surface_state": {
                "sdofs": [np.array([0.08]), np.array([0.10])],
                "iota": [0.12, 0.15],
                "G": [1.0, 1.1],
            },
            "J": last_J,
            "dJ": last_dJ.copy(),
            "accepted_iterations": 0,
            "accepted_x": np.zeros(5),
        }
        module.VV = SimpleNamespace(gamma=lambda: np.array([[[1.0, 0.0, 0.0]]]))
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.0
        module.JF = _JF()

        J_out, dJ_out = module.fun(np.ones(5))

        self.assertGreater(J_out, last_J)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        np.testing.assert_array_equal(inner.surface.x, np.array([0.08]))
        np.testing.assert_array_equal(outer.surface.x, np.array([0.10]))
        self.assertEqual(inner.res["iota"], 0.12)
        self.assertEqual(outer.res["iota"], 0.15)

    def test_callback_multisurface_records_status_and_log(self):
        module = self.load_module()

        class _Surface:
            nfp = 5

            def __init__(self, volume, point, cross_section):
                self.x = np.array([volume])
                self._volume = volume
                self._point = np.asarray(point, dtype=float)
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def unitnormal(self):
                return np.array([[[1.0, 0.0, 0.0]]])

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        class _ScalarObjective:
            def __init__(self, value):
                self._value = value

            def J(self):
                return self._value

            def dJ(self):
                return np.array([self._value, -self._value])

            def __add__(self, other):
                if other == 0:
                    return self
                return FakeAlgebraicObjective(self._value + other.J(), self.dJ() + other.dJ())

            __radd__ = __add__

            def __mul__(self, scalar):
                return FakeAlgebraicObjective(self._value * scalar, self.dJ() * scalar)

            __rmul__ = __mul__

        class _DistanceObjective(_ScalarObjective):
            def __init__(self, value, min_distance):
                super().__init__(value)
                self._min_distance = min_distance

            def shortest_distance(self):
                return self._min_distance

        class _Curve:
            def gamma(self):
                return np.array([[1.0, 0.0, 0.0]])

            def kappa(self):
                return np.array([2.0, 3.0])

        class _CurveLength:
            def J(self):
                return 4.2

        class _BS:
            def set_points(self, pts):
                self._points = pts

            def B(self):
                return np.array([[1.0, 0.0, 0.0]])

        inner_cs = [[0.85, 0.0, -0.1], [1.05, 0.0, -0.1], [1.05, 0.0, 0.1], [0.85, 0.0, 0.1]]
        outer_cs = [[0.7, 0.0, -0.3], [1.3, 0.0, -0.3], [1.3, 0.0, 0.3], [0.7, 0.0, 0.3]]
        inner = SimpleNamespace(surface=_Surface(0.08, [0.0, 0.0, 0.0], inner_cs), res={"success": True, "iota": 0.12, "G": 1.0})
        outer = SimpleNamespace(surface=_Surface(0.10, [0.4, 0.0, 0.0], outer_cs), res={"success": True, "iota": 0.15, "G": 1.1})

        with tempfile.TemporaryDirectory() as tmpdir:
            module.surface_data = [
                {"name": "inner", "seed_label": 0.16, "target_volume": 0.08, "boozer_surface": inner},
                {"name": "outer", "seed_label": 0.20, "target_volume": 0.10, "boozer_surface": outer},
            ]
            module.outer_surface_data = module.surface_data[-1]
            module.surface_iota_terms = [_ScalarObjective(0.12), _ScalarObjective(0.15)]
            module.nonQSs = [_ScalarObjective(0.10), _ScalarObjective(0.12)]
            module.brs = [_ScalarObjective(0.20), _ScalarObjective(0.24)]
            module.VV = SimpleNamespace(gamma=lambda: np.array([[[1.0, 0.0, 0.0]]]))
            module.SURFACE_GAP_THRESHOLD = 0.0
            module.SS_DIST = 0.0
            module.MULTISURFACE_RAMP_ITERATIONS = 5
            module.INNER_SURFACE_INITIAL_WEIGHT = 0.0
            module.JF = _ScalarObjective(1.23)
            module.Jiota = _ScalarObjective(0.33)
            module.JCurveLength = _ScalarObjective(0.44)
            module.JCurveCurve = _DistanceObjective(0.55, 0.66)
            module.JCurveSurface = _DistanceObjective(0.77, 0.88)
            module.JSurfSurf = None
            module.JCurvature = _ScalarObjective(0.99)
            module.RES_WEIGHT = 1000.0
            module.IOTAS_WEIGHT = 200.0
            module.LENGTH_WEIGHT = 1.0
            module.CC_WEIGHT = 100.0
            module.CC_DIST = 0.05
            module.CS_WEIGHT = 1.0
            module.CS_DIST = 0.02
            module.CURVATURE_WEIGHT = 0.1
            module.CURVATURE_THRESHOLD = 40.0
            module.SURF_DIST_WEIGHT = 1000.0
            module.banana_curve = _Curve()
            module.curvelength = _CurveLength()
            module.bs = _BS()
            module.OUT_DIR_ITER = tmpdir
            module.run_dict = {
                "surface_state": {
                    "sdofs": [np.array([0.08]), np.array([0.10])],
                    "iota": [0.12, 0.15],
                    "G": [1.0, 1.1],
                },
                "J": 1.0,
                "dJ": np.array([1.0, -1.0]),
                "it": 1,
                "accepted_iterations": 0,
                "lscount": 0,
                "x_prev": np.zeros(2),
                "intersecting": False,
                "topology_gate_status": {"enabled": False, "success": True, "nfieldlines": 0, "survived_lines": 0, "survival_fraction": 1.0, "survival_threshold": 0.25, "tmax": 2.0, "tol": 1e-7, "stop_reason_counts": {}, "first_exit_time": None, "first_exit_angle": None, "first_exit_reason": None},
            }

            module.callback(np.zeros(2))

            self.assertEqual(module.run_dict["surface_status"]["adjacent_gaps"], [0.4])
            self.assertEqual(module.run_dict["search_surface_status"]["adjacent_gaps"], [0.4])
            self.assertTrue(module.run_dict["surface_status"]["nesting_ok"])
            log_path = Path(tmpdir) / "log.txt"
            self.assertTrue(log_path.exists())
            log_text = log_path.read_text()
            self.assertIn("Adjacent surface gaps", log_text)
            self.assertIn("Surfaces nested", log_text)
            self.assertIn("Surface gate scale", log_text)

    def test_callback_tracks_relaxed_search_status_separately_from_full_status(self):
        module = self.load_module()

        class _Surface:
            nfp = 5

            def __init__(self, volume, point, cross_section):
                self.x = np.array([volume])
                self._volume = volume
                self._point = np.asarray(point, dtype=float)
                self._cross_section = np.asarray(cross_section, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def unitnormal(self):
                return np.array([[[1.0, 0.0, 0.0]]])

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                return self._cross_section

        class _ScalarObjective:
            def __init__(self, value):
                self._value = value

            def J(self):
                return self._value

            def dJ(self):
                return np.array([self._value, -self._value])

            def __add__(self, other):
                if other == 0:
                    return self
                return FakeAlgebraicObjective(self._value + other.J(), self.dJ() + other.dJ())

            __radd__ = __add__

            def __mul__(self, scalar):
                return FakeAlgebraicObjective(self._value * scalar, self.dJ() * scalar)

            __rmul__ = __mul__

        class _DistanceObjective(_ScalarObjective):
            def __init__(self, value, min_distance):
                super().__init__(value)
                self._min_distance = min_distance

            def shortest_distance(self):
                return self._min_distance

        class _Curve:
            def gamma(self):
                return np.array([[1.0, 0.0, 0.0]])

            def kappa(self):
                return np.array([2.0, 3.0])

        class _CurveLength:
            def J(self):
                return 4.2

        class _BS:
            def set_points(self, pts):
                self._points = pts

            def B(self):
                return np.array([[1.0, 0.0, 0.0]])

        inner_crossing = [
            [0.9, 0.0, -0.2],
            [1.4, 0.0, 0.0],
            [0.9, 0.0, 0.2],
            [0.6, 0.0, 0.0],
        ]
        outer_box = [
            [0.7, 0.0, -0.3],
            [1.3, 0.0, -0.3],
            [1.3, 0.0, 0.3],
            [0.7, 0.0, 0.3],
        ]
        inner = SimpleNamespace(surface=_Surface(0.08, [0.0, 0.0, 0.0], inner_crossing), res={"success": True, "iota": 0.12, "G": 1.0})
        outer = SimpleNamespace(surface=_Surface(0.10, [0.4, 0.0, 0.0], outer_box), res={"success": True, "iota": 0.15, "G": 1.1})

        with tempfile.TemporaryDirectory() as tmpdir:
            module.surface_data = [
                {"name": "inner", "seed_label": 0.16, "target_volume": 0.08, "boozer_surface": inner},
                {"name": "outer", "seed_label": 0.20, "target_volume": 0.10, "boozer_surface": outer},
            ]
            module.outer_surface_data = module.surface_data[-1]
            module.surface_iota_terms = [_ScalarObjective(0.12), _ScalarObjective(0.15)]
            module.nonQSs = [_ScalarObjective(0.10), _ScalarObjective(0.12)]
            module.brs = [_ScalarObjective(0.20), _ScalarObjective(0.24)]
            module.VV = SimpleNamespace(gamma=lambda: np.array([[[1.0, 0.0, 0.0]]]))
            module.SURFACE_GAP_THRESHOLD = 0.005
            module.SS_DIST = 0.0
            module.MULTISURFACE_RAMP_ITERATIONS = 5
            module.INNER_SURFACE_INITIAL_WEIGHT = 0.0
            module.JF = _ScalarObjective(1.23)
            module.Jiota = _ScalarObjective(0.33)
            module.JCurveLength = _ScalarObjective(0.44)
            module.JCurveCurve = _DistanceObjective(0.55, 0.66)
            module.JCurveSurface = _DistanceObjective(0.77, 0.88)
            module.JSurfSurf = None
            module.JCurvature = _ScalarObjective(0.99)
            module.RES_WEIGHT = 1000.0
            module.IOTAS_WEIGHT = 200.0
            module.LENGTH_WEIGHT = 1.0
            module.CC_WEIGHT = 100.0
            module.CC_DIST = 0.05
            module.CS_WEIGHT = 1.0
            module.CS_DIST = 0.02
            module.CURVATURE_WEIGHT = 0.1
            module.CURVATURE_THRESHOLD = 40.0
            module.SURF_DIST_WEIGHT = 1000.0
            module.banana_curve = _Curve()
            module.curvelength = _CurveLength()
            module.bs = _BS()
            module.OUT_DIR_ITER = tmpdir
            module.run_dict = {
                "surface_state": {
                    "sdofs": [np.array([0.08]), np.array([0.10])],
                    "iota": [0.12, 0.15],
                    "G": [1.0, 1.1],
                },
                "J": 1.0,
                "dJ": np.array([1.0, -1.0]),
                "it": 1,
                "accepted_iterations": 0,
                "lscount": 0,
                "x_prev": np.zeros(2),
                "intersecting": False,
                "topology_gate_status": {"enabled": False, "success": True, "nfieldlines": 0, "survived_lines": 0, "survival_fraction": 1.0, "survival_threshold": 0.25, "tmax": 2.0, "tol": 1e-7, "stop_reason_counts": {}, "first_exit_time": None, "first_exit_angle": None, "first_exit_reason": None},
            }

            module.callback(np.zeros(2))

            self.assertTrue(module.run_dict["search_surface_status"]["success"])
            self.assertTrue(module.run_dict["search_surface_status"]["nesting_ok"])
            self.assertFalse(module.run_dict["surface_status"]["success"])
            self.assertFalse(module.run_dict["surface_status"]["nesting_ok"])

    def test_finalize_surface_stack_reverts_to_last_accepted_state_when_final_endpoint_is_invalid(self):
        module = self.load_module()

        class _Objective:
            def __init__(self):
                self.x = np.array([0.0])

            def J(self):
                return float(self.x[0] + 10.0)

            def dJ(self):
                return np.array([self.x[0] + 1.0])

        class _Surface:
            nfp = 5

            def __init__(self, accepted_x, accepted_volume, point):
                self.x = np.array([accepted_x], dtype=float)
                self._volume = accepted_volume
                self._point = np.asarray(point, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                radius = self._point[0]
                return np.array([
                    [radius - 0.05, 0.0, -0.05],
                    [radius + 0.05, 0.0, -0.05],
                    [radius + 0.05, 0.0, 0.05],
                    [radius - 0.05, 0.0, 0.05],
                ])

        class _BoozerSurface:
            def __init__(self, surface, objective, valid_limit, success_iota):
                self.surface = surface
                self._objective = objective
                self._valid_limit = valid_limit
                self._success_iota = success_iota
                self.res = {"success": True, "iota": success_iota, "G": 1.0}

            def run_code(self, iota, G):
                current = float(self._objective.x[0])
                self.surface.x = np.array([current], dtype=float)
                self.surface._volume = 0.08 if self._success_iota < 0.2 else 0.10
                self.res["success"] = current <= self._valid_limit
                self.res["iota"] = self._success_iota if self.res["success"] else -1.0
                self.res["G"] = G
                return self.res

        objective = _Objective()
        inner_surface = _Surface(1.0, 0.08, [0.2, 0.0, 0.0])
        outer_surface = _Surface(1.0, 0.10, [0.6, 0.0, 0.0])
        surface_data = [
            {"boozer_surface": _BoozerSurface(inner_surface, objective, valid_limit=1.5, success_iota=0.12)},
            {"boozer_surface": _BoozerSurface(outer_surface, objective, valid_limit=1.5, success_iota=0.15)},
        ]
        run_state = {
            "surface_state": {
                "sdofs": [np.array([1.0]), np.array([1.0])],
                "iota": [0.12, 0.15],
                "G": [1.0, 1.1],
            },
            "accepted_x": np.array([1.0]),
            "J": 11.0,
            "dJ": np.array([2.0]),
            "intersecting": False,
        }

        status = module.finalize_surface_stack(np.array([2.0]), objective, surface_data, run_state)

        self.assertFalse(status["success"])
        np.testing.assert_allclose(objective.x, [1.0])
        np.testing.assert_allclose(run_state["accepted_x"], [1.0])
        np.testing.assert_allclose(surface_data[0]["boozer_surface"].surface.x, [1.0])
        np.testing.assert_allclose(surface_data[1]["boozer_surface"].surface.x, [1.0])
        self.assertEqual(run_state["surface_state"]["iota"], [0.12, 0.15])

    def test_minimize_alm_restores_single_stage_incumbent_state_before_finalization(self):
        module = self.load_module()
        alm_globals = module.minimize_alm.__globals__
        augmented_inequality_objective = alm_globals["augmented_inequality_objective"]
        grad = np.array([1.0], dtype=float)
        search_weights = np.array([1.0], dtype=float)

        class _Objective:
            def __init__(self, x):
                self.x = np.array([x], dtype=float)

            def J(self):
                return float(self.x[0])

            def dJ(self):
                return np.array([1.0], dtype=float)

        class _Surface:
            nfp = 5

            def __init__(self, x, volume, point):
                self.x = np.array([x], dtype=float)
                self._volume = volume
                self._point = np.asarray(point, dtype=float)

            def volume(self):
                return self._volume

            def gamma(self):
                return self._point.reshape((1, 1, 3))

            def is_self_intersecting(self):
                return False

            def cross_section(self, phi, thetas=None, tol=1e-13):
                radius = self._point[0]
                return np.array([
                    [radius - 0.05, 0.0, -0.05],
                    [radius + 0.05, 0.0, -0.05],
                    [radius + 0.05, 0.0, 0.05],
                    [radius - 0.05, 0.0, 0.05],
                ])

        class _BoozerSurface:
            def __init__(self, surface, objective, success_iota):
                self.surface = surface
                self._objective = objective
                self._success_iota = success_iota
                self.res = {"success": True, "iota": success_iota, "G": 1.0}

            def run_code(self, iota, G):
                current = float(self._objective.x[0])
                self.surface.x = np.array([current], dtype=float)
                self.res["success"] = True
                self.res["iota"] = self._success_iota
                self.res["G"] = G
                return self.res

        objective = _Objective(1.0)
        surface = _Surface(1.0, 0.08, [0.4, 0.0, 0.0])
        surface_data = [
            {"boozer_surface": _BoozerSurface(surface, objective, success_iota=0.12)}
        ]
        topology_status = {
            "enabled": False,
            "success": True,
            "nfieldlines": 0,
            "survived_lines": 0,
            "survival_fraction": 1.0,
            "survival_threshold": 0.25,
            "tmax": 2.0,
            "tol": 1e-7,
            "stop_reason_counts": {},
            "first_exit_time": None,
            "first_exit_angle": None,
            "first_exit_reason": None,
        }

        def make_stack_status():
            return module.evaluate_surface_stack(surface_data)

        def make_search_eval(base_value, total):
            return {
                "total": float(total),
                "base_value": float(base_value),
                "grad": grad.copy(),
            }

        run_dict = {
            "accepted_x": np.array([1.0], dtype=float),
            "surface_state": module.snapshot_surface_states(surface_data),
            "J": 9.0,
            "dJ": grad.copy(),
            "search_eval": make_search_eval(9.0, 9.0),
            "surface_status": make_stack_status(),
            "search_surface_status": make_stack_status(),
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": topology_status,
            "last_successful_eval": {"total": 999.0},
            "last_successful_eval_weights": search_weights.copy(),
        }
        settings = module.ALMSettings(
            max_outer_iterations=2,
            max_subproblem_continuations=0,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-12,
        )
        minimize_targets = iter([2.0, 3.0])

        def evaluation_profile(x):
            point = float(np.asarray(x, dtype=float)[0])
            if point >= 2.5:
                return 5.0, 0.5
            if point >= 1.5:
                return 1.0, 1.0
            return 9.0, 9.0

        def update_single_stage_state(x):
            base_value, total = evaluation_profile(x)
            x_array = np.asarray(x, dtype=float).copy()
            objective.x = x_array.copy()
            surface_data[0]["boozer_surface"].surface.x = x_array.copy()
            surface_data[0]["boozer_surface"].res["success"] = True
            surface_data[0]["boozer_surface"].res["iota"] = 0.12
            surface_data[0]["boozer_surface"].res["G"] = 1.0
            run_dict["accepted_x"] = x_array.copy()
            run_dict["surface_state"] = module.snapshot_surface_states(surface_data)
            run_dict["J"] = float(total)
            run_dict["dJ"] = grad.copy()
            run_dict["search_eval"] = make_search_eval(base_value, total)
            run_dict["surface_status"] = make_stack_status()
            run_dict["search_surface_status"] = make_stack_status()
            run_dict["accepted_hardware_status"] = {"success": True, "violations": []}
            run_dict["topology_gate_status"] = dict(topology_status)
            run_dict["last_successful_eval"] = {"total": float(total)}
            run_dict["last_successful_eval_weights"] = search_weights.copy()

        def evaluate_problem(x, multipliers, penalty):
            base_value, total = evaluation_profile(x)
            evaluation = augmented_inequality_objective(
                base_value=base_value,
                base_grad=grad.copy(),
                constraint_values=np.array([-1.0], dtype=float),
                constraint_grads=[np.zeros(1, dtype=float)],
                multipliers=np.asarray(multipliers, dtype=float),
                penalty=float(penalty),
            )
            evaluation["total"] = float(total)
            evaluation["base_value"] = float(base_value)
            evaluation["stationarity_norm"] = 1.0
            return evaluation

        def accepted_callback(x):
            update_single_stage_state(x)

        def snapshot_accepted_state():
            return module.snapshot_single_stage_incumbent_state(run_dict)

        def restore_incumbent_state(incumbent_state):
            module.restore_single_stage_incumbent_state(run_dict, incumbent_state)
            objective.x = run_dict["accepted_x"].copy()
            module.restore_surface_states(surface_data, run_dict["surface_state"])

        def fake_minimize(fun, x, jac, method, bounds, callback, options):
            del x, jac, method, bounds, callback, options
            target = np.array([next(minimize_targets)], dtype=float)
            fun(target)
            return SimpleNamespace(
                x=target,
                nit=1,
                success=True,
                message="synthetic",
            )

        with patch.dict(alm_globals, {"minimize": fake_minimize}):
            result = module.minimize_alm(
                np.array([0.0], dtype=float),
                ["demo_constraint"],
                evaluate_problem,
                settings,
                {"maxiter": 5, "ftol": 1e-12, "gtol": 1e-12},
                accepted_callback=accepted_callback,
                snapshot_accepted_state_fn=snapshot_accepted_state,
                restore_incumbent_state_fn=restore_incumbent_state,
            )

        np.testing.assert_allclose(result.x, [2.0])
        np.testing.assert_allclose(run_dict["accepted_x"], [2.0])
        np.testing.assert_allclose(objective.x, [2.0])
        np.testing.assert_allclose(surface_data[0]["boozer_surface"].surface.x, [2.0])
        self.assertNotIn("last_successful_eval", run_dict)
        self.assertNotIn("last_successful_eval_weights", run_dict)

        status = module.finalize_surface_stack(result.x, objective, surface_data, run_dict)

        self.assertTrue(status["success"])
        np.testing.assert_allclose(run_dict["accepted_x"], [2.0])
        np.testing.assert_allclose(surface_data[0]["boozer_surface"].surface.x, [2.0])

    def test_collect_surface_run_metadata_serializes_multisurface_fields(self):
        module = self.load_module()
        surface_data = [
            {"name": "inner", "seed_label": 0.16, "target_volume": 0.08},
            {"name": "outer", "seed_label": 0.20, "target_volume": 0.10},
        ]
        run_status = {
            "self_intersections": [False, False],
            "adjacent_gaps": [0.4],
            "outer_vessel_gap": 0.6,
            "nesting_ok": True,
            "bad_nesting_phis": [],
        }
        payload = module.collect_surface_run_metadata(
            surface_data,
            run_status,
            initial_surface_volumes=[0.08, 0.10],
            initial_surface_iotas=[0.12, 0.15],
            final_surface_volumes=[0.081, 0.101],
            final_surface_iotas=[0.121, 0.151],
        )

        self.assertEqual(payload["SURFACE_NAMES"], ["inner", "outer"])
        self.assertEqual(payload["ADJACENT_SURFACE_GAPS"], [0.4])
        self.assertTrue(payload["SURFACES_NESTED"])
        self.assertEqual(payload["FINAL_SURFACE_VOLUMES"], [0.081, 0.101])


class BoozerFallbackLBFGSBTests(unittest.TestCase):
    """Issue #2: elevated-J fallback must not flush L-BFGS-B Hessian memory."""

    def test_elevated_j_stale_gradient_preserves_bfgs_memory(self):
        from scipy.optimize import minimize

        def rosenbrock(x):
            f = sum(100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
                    for i in range(len(x) - 1))
            g = np.zeros_like(x)
            for i in range(len(x) - 1):
                g[i] += -400*x[i]*(x[i+1] - x[i]**2) - 2*(1 - x[i])
                g[i+1] += 200*(x[i+1] - x[i]**2)
            return f, g

        rng = np.random.RandomState(42)
        x0 = rng.randn(10) * 0.5
        state = {"x_good": x0.copy(), "J": None, "dJ": None}

        def fun_with_fallback(x):
            f, g = rosenbrock(x)
            if np.linalg.norm(x - state["x_good"]) > 0.5 and state["J"] is not None:
                return state["J"] + max(abs(state["J"]), 1.0), state["dJ"].copy()
            state["J"] = f
            state["dJ"] = g.copy()
            state["x_good"] = x.copy()
            return f, g

        res = minimize(fun_with_fallback, x0, jac=True, method="L-BFGS-B",
                       options={"maxiter": 500, "maxcor": 10})

        self.assertTrue(res.success, f"L-BFGS-B did not converge: {res.message}")
        self.assertGreater(res.hess_inv.n_corrs, 0)


STAGE2_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "banana_coil_solver.py"
)
STAGE2_GEOMETRY_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "stage2_geometry.py"
)


def load_stage2_module():
    spec = importlib.util.spec_from_file_location(
        f"banana_coil_solver_{uuid.uuid4().hex}",
        STAGE2_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_segment_distance_from_source():
    """Extract the deployed segment-distance kernel from the SSOT stage2 module via AST.

    Parses the source file, extracts just the _clamp01 and segment_segment_distance
    function definitions (stripping @njit decorators), and compiles them in an
    isolated namespace. This executes the real deployed algorithm without requiring
    numba or importing the full Stage 2 workflow module.
    """
    import ast
    source = STAGE2_GEOMETRY_MODULE_PATH.read_text()
    tree = ast.parse(source)
    func_nodes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name in ("_clamp01", "segment_segment_distance"):
            node.decorator_list = []
            func_nodes.append(node)
    extracted = ast.Module(body=func_nodes, type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace = {"np": np}
    exec(compile(extracted, str(STAGE2_GEOMETRY_MODULE_PATH), "exec"), namespace)
    return namespace["segment_segment_distance"]


_segment_segment_distance = _load_segment_distance_from_source()


def _brute_force_segment_distance(P1, P2, Q1, Q2):
    """Reference distance via interior + 4 edge candidates on [0,1]^2."""
    u, v, w0 = P2 - P1, Q2 - Q1, P1 - Q1
    a, bv, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
    d_val, e = np.dot(u, w0), np.dot(v, w0)
    cands = []
    denom = a * c - bv * bv
    if denom > 1e-30:
        sn = (bv * e - c * d_val) / denom
        tn = (a * e - bv * d_val) / denom
        if 0.0 <= sn <= 1.0 and 0.0 <= tn <= 1.0:
            dp = w0 + sn * u - tn * v
            cands.append(np.dot(dp, dp))
    for sf in [0.0, 1.0]:
        to = max(0.0, min(1.0, (e + sf * bv) / c)) if c > 1e-30 else 0.0
        dp = w0 + sf * u - to * v
        cands.append(np.dot(dp, dp))
    for tf in [0.0, 1.0]:
        so = max(0.0, min(1.0, (tf * bv - d_val) / a)) if a > 1e-30 else 0.0
        dp = w0 + so * u - tf * v
        cands.append(np.dot(dp, dp))
    return np.sqrt(min(cands))


class SegmentDistanceTests(unittest.TestCase):
    """Issue #5/#6: segment-segment distance with Sunday/Lumelsky re-projection."""

    def _d(self, p1, p2, q1, q2):
        return _segment_segment_distance(
            np.array(p1, dtype=float), np.array(p2, dtype=float),
            np.array(q1, dtype=float), np.array(q2, dtype=float),
        )

    def test_skew_segments_reprojection(self):
        """Issue #5: buggy=1.414, correct=sqrt(1.8) after re-projection."""
        d = self._d([0, 0, 0], [2, 1, 0], [-1, 3, 0], [1, 2, 0])
        self.assertAlmostEqual(d, np.sqrt(1.8), places=10)

    def test_parallel_overlapping_segments(self):
        """Issue #6: buggy=8.06, correct=1.0 for overlapping parallel segments."""
        d = self._d([0, 0, 0], [10, 0, 0], [8, 1, 0], [20, 1, 0])
        self.assertAlmostEqual(d, 1.0, places=10)

    def test_collinear_gap(self):
        d = self._d([0, 0, 0], [1, 0, 0], [3, 0, 0], [5, 0, 0])
        self.assertAlmostEqual(d, 2.0, places=10)

    def test_perpendicular_touching(self):
        d = self._d([0, 0, 0], [1, 0, 0], [0.5, 0, 0], [0.5, 1, 0])
        self.assertAlmostEqual(d, 0.0, places=10)

    def test_point_to_segment(self):
        d = self._d([0, 2, 0], [0, 2, 0], [0, 0, 0], [1, 0, 0])
        self.assertAlmostEqual(d, 2.0, places=10)

    def test_parallel_non_overlapping(self):
        d = self._d([0, 0, 0], [3, 0, 0], [5, 1, 0], [8, 1, 0])
        self.assertAlmostEqual(d, np.sqrt(5.0), places=10)

    def test_t_shaped(self):
        d = self._d([0, 0, 0], [2, 0, 0], [1, 0.5, 0], [1, 3, 0])
        self.assertAlmostEqual(d, 0.5, places=10)

    def test_both_degenerate(self):
        d = self._d([1, 2, 3], [1, 2, 3], [4, 5, 6], [4, 5, 6])
        self.assertAlmostEqual(d, np.linalg.norm([3, 3, 3]), places=10)

    def test_near_parallel_interior_minimum(self):
        """Near-parallel segments where the true minimum is at an interior point.

        P along x-axis, Q nearly parallel with tiny z-tilt and small y-offset.
        Endpoint projections all return sqrt(d^2 + eps^2) but the true minimum
        at (s=0.5, t=0.5) is just d (z components cancel at the midpoint).
        """
        eps = 9e-6
        d_offset = 1e-6
        d = self._d([-1, 0, 0], [1, 0, 0], [-1, d_offset, -eps], [1, d_offset, eps])
        self.assertAlmostEqual(d, d_offset, places=10)

    def test_near_parallel_brute_force(self):
        """Stress-test the near-parallel branch with 1000 adversarial pairs."""
        rng = np.random.RandomState(77777)
        PAR_EPS = 1e-10
        n_parallel = 0
        for _ in range(1000):
            base = rng.randn(3)
            base /= np.linalg.norm(base)
            P1 = rng.randn(3)
            P2 = P1 + rng.uniform(0.5, 5.0) * base
            angle = rng.uniform(1e-7, 1e-5)
            perp = rng.randn(3)
            perp -= np.dot(perp, base) * base
            perp /= np.linalg.norm(perp)
            q_dir = base + angle * perp
            q_dir /= np.linalg.norm(q_dir)
            Q1 = P1 + rng.randn(3) * rng.uniform(1e-7, 1e-4)
            Q2 = Q1 + rng.uniform(0.5, 5.0) * q_dir
            u, v, _ = P2 - P1, Q2 - Q1, P1 - Q1
            a, bv, c = np.dot(u, u), np.dot(u, v), np.dot(v, v)
            denom = a * c - bv * bv
            if denom < PAR_EPS * a * c:
                n_parallel += 1
            d_algo = _segment_segment_distance(P1, P2, Q1, Q2)
            d_brute = _brute_force_segment_distance(P1, P2, Q1, Q2)
            self.assertAlmostEqual(d_algo, d_brute, places=9,
                                   msg=f"Near-parallel mismatch: algo={d_algo}, brute={d_brute}")
        self.assertGreater(n_parallel, 900, "Not enough pairs hit the near-parallel branch")

    def test_random_brute_force(self):
        """Verify against exhaustive interior + edge search on 1000 random pairs."""
        rng = np.random.RandomState(12345)
        for _ in range(1000):
            P1, P2, Q1, Q2 = rng.randn(4, 3)
            d_algo = _segment_segment_distance(P1, P2, Q1, Q2)
            d_brute = _brute_force_segment_distance(P1, P2, Q1, Q2)
            self.assertAlmostEqual(d_algo, d_brute, places=9,
                                   msg=f"Mismatch: algo={d_algo}, brute={d_brute}")


class CrossSectionNormalizationTests(unittest.TestCase):
    """Issue #8/#9: cross_section phi argument must be normalized to [0,1]."""

    PLOTTING_UTILS_PATH = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "single_stage_optimization"
        / "plotting_utils.py"
    )

    def test_plotting_utils_source_divides_by_2pi(self):
        """Verify the shared plotting_utils uses phi_slice / (2 * np.pi), not * 2 * np.pi."""
        source = self.PLOTTING_UTILS_PATH.read_text()
        self.assertIn("phi_slice / (2 * np.pi)", source)
        self.assertNotIn("phi_slice * 2 * np.pi", source)


class FtolGtolDefaultTests(unittest.TestCase):
    """ftol/gtol should be explicit tight defaults, not mpol-dependent."""

    def test_no_mpol_based_tolerance_table(self):
        """ftol/gtol are tight defaults, not mpol-dependent lookups."""
        source = EXAMPLE_MODULE_PATH.read_text()
        self.assertNotIn("ftol_by_mpol", source)
        self.assertNotIn("gtol_by_mpol", source)


class ConfinementSurrogateTests(unittest.TestCase):
    def test_topology_scorer_surrogate_emphasizes_tail_failures(self):
        topology_module = load_topology_scorer_module()

        line_metrics = [
            {"survived": True, "first_exit_time": None},
            {"survived": False, "first_exit_time": 80.0},
            {"survived": False, "first_exit_time": 20.0},
            {"survived": False, "first_exit_time": 10.0},
        ]

        surrogate = topology_module.summarize_confinement_surrogate(
            line_metrics,
            tmax=100.0,
            worst_k=2,
            early_exit_threshold=0.2,
            mean_weight=0.2,
            worst_weight=0.6,
            early_weight=0.2,
        )

        self.assertAlmostEqual(surrogate["mean_line_loss"], 0.475)
        self.assertAlmostEqual(surrogate["worst_k_line_loss"], 0.85)
        self.assertAlmostEqual(surrogate["early_exit_fraction"], 0.25)
        self.assertAlmostEqual(surrogate["confinement_loss"], 0.655)
        self.assertEqual(surrogate["confinement_surrogate_k"], 2)

    def test_checkpoint_confinement_objective_adds_weighted_loss(self):
        module = load_single_stage_example_module()

        objective = module.checkpoint_confinement_objective(
            0.125,
            {"confinement_loss": 0.4},
            3.0,
        )

        self.assertAlmostEqual(objective, 1.325)


class RunIdentityTests(unittest.TestCase):
    def _make_identity_args(self):
        return SimpleNamespace(
            boozer_stage_refinement=False,
            refinement_boozer_stage="final",
            refinement_maxiter=100,
            refinement_chunk_maxiter=20,
            refinement_max_stalled_chunks=2,
            cc_dist=0.05,
            cc_weight=100.0,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            constraint_method="penalty",
            init_only=False,
            basin_hops=0,
            basin_stepsize=0.01,
            ftol=None,
            gtol=None,
            alm_max_outer_iters=10,
            alm_penalty_init=1.0,
            alm_penalty_scale=10.0,
            alm_feas_tol=1e-6,
            alm_stationarity_tol=1e-6,
            alm_trust_radius_init=0.0,
            alm_trust_radius_min=1e-9,
            alm_trust_radius_shrink=0.5,
            alm_trust_radius_grow=2.0,
            alm_max_inner_attempts=4,
            alm_max_subproblem_continuations=0,
            alm_distance_smoothing=1e-3,
            alm_curvature_smoothing=1e-3,
            num_surfaces=1,
            inner_surface_ratio=0.8,
            surface_gap_threshold=0.0,
            multisurface_ramp_iterations=0,
            inner_surface_initial_weight=1.0,
            multisurface_initial_step_scale=1.0,
            multisurface_initial_step_maxiter=0,
            topology_gate_fieldlines=4,
            topology_gate_tmax=2.0,
            topology_gate_tol=1e-7,
            topology_gate_survival_threshold=0.25,
            topology_gate_penalty_scale=4.0,
            hardware_search_mode="hard",
            hardware_search_soft_iterations=0,
            topology_scorer_every=10,
            topology_scorer_nfieldlines=12,
            topology_scorer_tmax=50.0,
            confinement_objective_weight=0.0,
            confinement_surrogate_worst_k=3,
            confinement_surrogate_early_threshold=0.2,
            confinement_surrogate_mean_weight=0.2,
            confinement_surrogate_worst_weight=0.6,
            confinement_surrogate_early_weight=0.2,
        )

    def _make_identity_config(self, module, args, boozer_I=0.37, plasma_current_A=1850000.0):
        return module.make_run_identity_config(
            args,
            "stage2-seed.json",
            "final",
            0.1,
            args.constraint_method,
            0.15,
            0.15,
            boozer_I,
            plasma_current_A,
            0.22,
            80,
            80,
            None,
        )

    def _build_identity(self, module, args, boozer_I=0.37, plasma_current_A=1850000.0):
        return module.build_run_identity_config(
            self._make_identity_config(
                module,
                args,
                boozer_I=boozer_I,
                plasma_current_A=plasma_current_A,
            )
        )

    def test_run_identity_config_is_frozen(self):
        module = load_single_stage_example_module()
        args = self._make_identity_args()
        config = self._make_identity_config(module, args)

        with self.assertRaisesRegex(Exception, "cannot assign to field"):
            config.stage = "other"

    def test_run_identity_changes_when_only_confinement_settings_change(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.confinement_objective_weight = 5.0

        base_config = self._build_identity(module, base_args)
        weighted_config = self._build_identity(module, weighted_args)

        self.assertNotEqual(base_config, weighted_config)

    def test_run_identity_changes_when_constraint_method_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        alm_args = self._make_identity_args()
        alm_args.constraint_method = "alm"

        penalty_config = self._build_identity(module, base_args)
        alm_config = self._build_identity(module, alm_args)

        self.assertNotEqual(penalty_config, alm_config)

    def test_run_identity_changes_when_physical_plasma_current_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()

        base_config = self._build_identity(module, base_args, boozer_I=0.0, plasma_current_A=0.0)
        physical_config = self._build_identity(module, base_args, boozer_I=0.0016, plasma_current_A=8000.0)

        self.assertNotEqual(base_config, physical_config)

    def test_run_identity_ignores_plasma_current_input_source_when_realized_current_matches(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()

        physical_config = self._build_identity(module, base_args, boozer_I=0.0016, plasma_current_A=8000.0)
        raw_config = self._build_identity(module, base_args, boozer_I=0.0016, plasma_current_A=8000.0)

        self.assertEqual(physical_config, raw_config)

    def test_run_identity_changes_when_topology_gate_penalty_scale_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        changed_args = self._make_identity_args()
        changed_args.topology_gate_penalty_scale = 9.0

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, changed_args),
        )

    def test_run_identity_changes_when_hardware_search_policy_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        changed_args = self._make_identity_args()
        changed_args.hardware_search_mode = "adaptive"
        changed_args.hardware_search_soft_iterations = 3

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, changed_args),
        )

    def test_run_identity_changes_when_refinement_policy_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        changed_args = self._make_identity_args()
        changed_args.boozer_stage_refinement = True
        changed_args.refinement_maxiter = 25

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, changed_args),
        )

    def test_run_identity_changes_when_refinement_chunk_policy_changes(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        changed_args = self._make_identity_args()
        changed_args.refinement_chunk_maxiter = 8

        self.assertNotEqual(
            self._build_identity(module, base_args),
            self._build_identity(module, changed_args),
        )

    def test_run_identity_does_not_depend_on_module_globals(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()

        base_config = self._build_identity(module, base_args)
        module.MULTISURFACE_RAMP_ITERATIONS = 17
        module.INNER_SURFACE_INITIAL_WEIGHT = 0.25
        module.TOPOLOGY_GATE_FIELDLINES = 99
        module.TOPOLOGY_GATE_TMAX = 9.0
        module.TOPOLOGY_GATE_TOL = 1e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.9
        module.TOPOLOGY_SCORER_EVERY = 77
        module.TOPOLOGY_SCORER_NFIELDLINES = 42
        module.TOPOLOGY_SCORER_TMAX = 88.0
        module.CONFINEMENT_OBJECTIVE_WEIGHT = 9.0
        module.CONFINEMENT_SURROGATE_WORST_K = 11
        module.CONFINEMENT_SURROGATE_EARLY_THRESHOLD = 0.9
        module.CONFINEMENT_SURROGATE_MEAN_WEIGHT = 0.7
        module.CONFINEMENT_SURROGATE_WORST_WEIGHT = 0.2
        module.CONFINEMENT_SURROGATE_EARLY_WEIGHT = 0.1

        self.assertEqual(base_config, self._build_identity(module, base_args))


class CurrentBaselineContractTests(unittest.TestCase):
    def test_stage2_seed_dir_formats_include_tf_current_segment(self):
        module = load_single_stage_example_module()
        seed_spec = module.Stage2SeedSpec(
            plasma_surf_filename="dummy.nc",
            major_radius=0.915,
            toroidal_flux=0.24,
            length_weight=0.0005,
            cc_weight=100.0,
            cc_threshold=0.05,
            curvature_weight=0.0001,
            curvature_threshold=40.0,
            banana_surf_radius=0.22,
            tf_current_A=8.0e4,
            order=2,
        )
        local_dir = module.format_local_stage2_seed_dir(seed_spec)
        database_dir = module.format_database_stage2_seed_dir(seed_spec)

        self.assertIn("TFC=80000", local_dir)
        self.assertIn("TFC=80000", database_dir)

    def test_resolve_stage2_tf_current_prefers_recorded_stage2_result(self):
        module = load_single_stage_example_module()

        tf_coils = [
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: 1.0e5)),
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: 1.0e5)),
        ]

        self.assertEqual(
            module.resolve_stage2_tf_current_A({"TF_CURRENT_A": 8.0e4}, tf_coils),
            8.0e4,
        )

    def test_resolve_stage2_num_tf_coils_prefers_recorded_artifact_count(self):
        module = load_single_stage_example_module()
        stage2_results = {"NUM_TF_COILS": 20}

        self.assertEqual(
            module.resolve_stage2_num_tf_coils(stage2_results, requested_num_tf_coils=20),
            20,
        )

    def test_resolve_stage2_num_tf_coils_rejects_cli_mismatch(self):
        module = load_single_stage_example_module()
        stage2_results = {"NUM_TF_COILS": 18}

        with self.assertRaisesRegex(ValueError, "NUM_TF_COILS=18.*--num-tf-coils=20"):
            module.resolve_stage2_num_tf_coils(stage2_results, requested_num_tf_coils=20)

    def test_validate_loaded_stage2_coils_partition_rejects_too_few_loaded_coils(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "has only 19 coils.*NUM_TF_COILS=20"):
            module.validate_loaded_stage2_coils_partition([object()] * 19, num_tf_coils=20)

    def test_validate_loaded_stage2_coils_partition_rejects_missing_banana_coils(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "leaving no banana coils"):
            module.validate_loaded_stage2_coils_partition([object()] * 20, num_tf_coils=20)

    def test_build_stage2_bs_path_prefers_current_penalty_dir(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            current_dir = (
                outputs_dir
                / "R0=0.915-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-TFC=80000-Order=2-CM=penalty"
            )
            current_dir.mkdir(parents=True)
            expected_path = current_dir / "biot_savart_opt.json"
            expected_path.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                stage2_source="local",
                local_stage2_root=tmpdir,
                database_stage2_root="/unused",
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.915,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_build_stage2_bs_path_falls_back_to_legacy_basin_hop_without_tf_segment(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            legacy_dir = (
                outputs_dir
                / "R0=0.915-s=0.24-LW=0.0005-CCW=100-CW=0.0001-SR=0.220-Order=2-BH=3-BS=0.01-BSeed=7"
            )
            legacy_dir.mkdir(parents=True)
            expected_path = legacy_dir / "biot_savart_opt.json"
            expected_path.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                stage2_source="local",
                local_stage2_root=tmpdir,
                database_stage2_root="/unused",
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.915,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_single_stage_parse_args_accepts_hardware_search_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--hardware-search-mode",
                "adaptive",
                "--hardware-search-soft-iterations",
                "3",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.hardware_search_mode, "adaptive")
        self.assertEqual(args.hardware_search_soft_iterations, 3)

    def test_single_stage_parse_args_accepts_boozer_stage_refinement_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--boozer-stage-refinement",
                "--refinement-boozer-stage",
                "final",
                "--refinement-maxiter",
                "25",
                "--refinement-chunk-maxiter",
                "7",
                "--refinement-max-stalled-chunks",
                "3",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.boozer_stage_refinement)
        self.assertEqual(args.refinement_boozer_stage, "final")
        self.assertEqual(args.refinement_maxiter, 25)
        self.assertEqual(args.refinement_chunk_maxiter, 7)
        self.assertEqual(args.refinement_max_stalled_chunks, 3)

    def test_stage2_parse_args_accepts_tf_current_A(self):
        module = load_stage2_module()

        with patch.object(sys, "argv", ["banana_coil_solver.py", "--tf-current-A", "80000"]):
            args = module.parse_args()

        self.assertEqual(args.tf_current_A, 80000.0)


class Stage2RuntimeSmokeTests(unittest.TestCase):
    _EXPECTED_BASIN_TELEMETRY = {
        "basin_accepted_hops": 1,
        "basin_rejected_hops": 1,
        "basin_best_objective": 0.42,
        "basin_accept_test_rejections": 1,
        "basin_accept_test_triggered": True,
    }

    def _make_stage2_args(self, output_root, **overrides):
        defaults = {
            "plasma_surf_filename": "demo.nc",
            "equilibria_dir": str(output_root),
            "equilibrium_path": str(Path(output_root) / "demo.nc"),
            "output_root": str(output_root),
            "stage2_bs_path": str(Path(output_root) / "seed.json"),
            "nphi": 8,
            "ntheta": 8,
            "init_only": True,
            "banana_surf_radius": 0.22,
            "tf_current_A": 8.0e4,
            "major_radius": 0.915,
            "toroidal_flux": 0.24,
            "order": 2,
            "maxiter": 30,
            "ftol": 1e-15,
            "gtol": 1e-15,
            "constraint_method": "penalty",
            "alm_max_outer_iters": 7,
            "alm_penalty_init": 2.0,
            "alm_penalty_scale": 3.0,
            "alm_feas_tol": 1e-4,
            "alm_stationarity_tol": 2e-4,
            "alm_trust_radius_init": 0.15,
            "alm_trust_radius_min": 1e-3,
            "alm_trust_radius_shrink": 0.4,
            "alm_trust_radius_grow": 1.8,
            "alm_max_inner_attempts": 5,
            "alm_max_subproblem_continuations": 9,
            "alm_distance_smoothing": 0.005,
            "alm_curvature_smoothing": 0.05,
            "alm_taylor_test": False,
            "alm_taylor_test_seed": 123,
            "length_weight": 5e-4,
            "length_target": 1.75,
            "cc_threshold": 0.05,
            "cc_weight": 100.0,
            "curvature_weight": 1e-4,
            "curvature_threshold": 40.0,
            "curvature_p_norm": 2,
            "squared_flux_weight": 1.0,
            "basin_hops": 0,
            "basin_stepsize": 0.01,
            "basin_temperature": 2.5,
            "basin_niter_success": 6,
            "basin_seed": 7,
            "theta_center": np.pi,
            "phi_center": np.pi / 4.0,
            "theta_width": np.pi / 6.0,
            "phi_width": np.pi / 8.0,
            "num_quadpoints": 16,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def _run_stage2_main(self, *, init_only, constraint_method, use_seed, basin_hops=0):
        module = load_stage2_module()
        runtime = {
            "seed_loads": 0,
            "initialize_calls": 0,
            "minimize_calls": 0,
            "minimize_alm_calls": 0,
            "run_basin_hopping_calls": 0,
            "results": None,
        }

        class FakeStage2Objective:
            def __init__(self, value, gradient, x=None):
                self._value = float(value)
                self._gradient = np.asarray(gradient, dtype=float)
                self.x = np.zeros(2, dtype=float) if x is None else np.asarray(x, dtype=float)

            def J(self):
                return self._value

            def dJ(self, partials=False):
                if partials:
                    return lambda _objective: self._gradient.copy()
                return self._gradient.copy()

            def __add__(self, other):
                if other == 0:
                    return self
                return FakeStage2Objective(
                    self._value + other.J(),
                    self._gradient + other.dJ(),
                    self.x.copy(),
                )

            __radd__ = __add__

            def __mul__(self, scalar):
                return FakeStage2Objective(
                    self._value * float(scalar),
                    self._gradient * float(scalar),
                    self.x.copy(),
                )

            __rmul__ = __mul__

        class FakeCurrent:
            def __init__(self, value):
                self._value = float(value)

            def __mul__(self, scalar):
                return FakeCurrent(self._value * float(scalar))

            __rmul__ = __mul__

            def fix_all(self):
                return None

            def get_value(self):
                return self._value

        class FakeCurve:
            def fix_all(self):
                return None

        class FakeSurface:
            def __init__(self):
                self.nfp = 22

            def gamma(self):
                return np.zeros((2, 2, 3), dtype=float)

            def unitnormal(self):
                return np.ones((2, 2, 3), dtype=float)

            def to_vtk(self, *_args, **_kwargs):
                return None

            def volume(self):
                return 0.12

        class FakeBiotSavart:
            def __init__(self):
                self.points = np.zeros((4, 3), dtype=float)

            def set_points(self, points):
                self.points = np.asarray(points, dtype=float)

            def B(self):
                return np.ones_like(self.points)

            def save(self, *_args, **_kwargs):
                return None

        class FakeCurveDistance(FakeStage2Objective):
            def __init__(self):
                super().__init__(0.25, [0.3, 0.4])
                self.minimum_distance = 0.05
                self.curves = ["curve_a", "curve_b"]

            def shortest_distance(self):
                return 0.04

        class FakeCurvatureObjective(FakeStage2Objective):
            def __init__(self):
                super().__init__(0.35, [0.2, 0.3])
                self.threshold = 40.0
                self.curve = SimpleNamespace(kappa=lambda: np.array([39.0, 41.0], dtype=float))

        fake_bs = FakeBiotSavart()
        fake_surface = FakeSurface()
        fake_curve_names = ["curve_a", "curve_b", "curve_c"]
        fake_banana_curve = SimpleNamespace(kappa=lambda: np.array([39.0, 41.0], dtype=float))
        fake_banana_coils = [SimpleNamespace(curve=fake_banana_curve, current=FakeCurrent(9500.0))]
        fake_tf_coils = [
            SimpleNamespace(curve=FakeCurve(), current=FakeCurrent(8.0e4)),
            SimpleNamespace(curve=FakeCurve(), current=FakeCurrent(8.0e4)),
        ]

        def fake_seed_loader(seed_bs_path, surf, num_tf_coils, out_dir):
            runtime["seed_loads"] += 1
            self.assertEqual(num_tf_coils, 20)
            self.assertIs(surf, fake_surface)
            return (
                fake_bs,
                fake_curve_names,
                fake_banana_curve,
                fake_banana_coils,
                fake_tf_coils,
            )

        def fake_initialize_coils(
            surf,
            surf_coils,
            tf_coils,
            num_quadpoints,
            order,
            phi_center,
            theta_center,
            phi_width,
            theta_width,
            out_dir,
        ):
            runtime["initialize_calls"] += 1
            self.assertIs(surf, fake_surface)
            self.assertEqual(surf_coils, "surf_coils")
            self.assertEqual(len(tf_coils), 20)
            self.assertEqual(num_quadpoints, 16)
            self.assertEqual(order, 2)
            self.assertEqual(phi_center, np.pi / 4.0)
            self.assertEqual(theta_center, np.pi)
            self.assertEqual(phi_width, np.pi / 8.0)
            self.assertEqual(theta_width, np.pi / 6.0)
            self.assertTrue(str(out_dir).endswith("outputs-demo.nc/"))
            return (
                fake_bs,
                fake_curve_names,
                fake_banana_curve,
                fake_banana_coils,
            )

        def fake_minimize(*_args, **_kwargs):
            runtime["minimize_calls"] += 1
            return SimpleNamespace(
                x=np.array([0.3, -0.2], dtype=float),
                nit=4,
                message="penalty_ok",
                success=True,
            )

        def fake_minimize_alm(*_args, **_kwargs):
            runtime["minimize_alm_calls"] += 1
            return SimpleNamespace(
                x=np.array([0.1, 0.2], dtype=float),
                nit=5,
                message="alm_ok",
                success=True,
                outer_iterations=2,
                penalty=3.5,
                multipliers=np.array([0.1, 0.2, 0.3], dtype=float),
                constraint_values=np.array([0.0, 0.01, 0.0], dtype=float),
                solver_constraint_values=np.array([0.0, 0.2, 0.0], dtype=float),
                trust_radius=0.1,
                history=[{"outer_iteration": 1}],
            )

        def fake_run_basin_hopping(*_args, **_kwargs):
            runtime["run_basin_hopping_calls"] += 1
            self.assertEqual(_kwargs["basin_temperature"], 2.5)
            self.assertEqual(_kwargs["basin_niter_success"], 6)
            return (
                SimpleNamespace(
                    x=np.array([0.6, -0.1], dtype=float),
                    fun=0.42,
                    nit=2,
                    minimization_failures=1,
                    lowest_optimization_result=SimpleNamespace(
                        nit=6,
                        message="basin_ok",
                        success=True,
                    ),
                ),
                self._EXPECTED_BASIN_TELEMETRY.copy(),
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            stage2_bs_path = str(Path(tmpdir) / "seed.json") if use_seed else None
            args = self._make_stage2_args(
                tmpdir,
                init_only=init_only,
                constraint_method=constraint_method,
                stage2_bs_path=stage2_bs_path,
                equilibrium_path=str(Path(tmpdir) / "demo.nc"),
                basin_hops=basin_hops,
            )

            with ExitStack() as stack:
                common_patches = [
                    patch.object(module, "validate_alm_cli_args", lambda *_args: None),
                    patch.object(module, "build_equilibrium_path", lambda _args: args.equilibrium_path),
                    patch.object(
                        module,
                        "create_equally_spaced_curves",
                        lambda *_args, **_kwargs: [FakeCurve() for _ in range(20)],
                    ),
                    patch.object(module, "Current", FakeCurrent),
                    patch.object(
                        module,
                        "Coil",
                        lambda curve, current: SimpleNamespace(curve=curve, current=current),
                    ),
                    patch.object(module, "_init_surface", lambda *_args, **_kwargs: fake_surface),
                    patch.object(
                        module,
                        "build_hbt_reference_surfaces",
                        lambda *_args, **_kwargs: (
                            "hbt",
                            "surf_coils",
                            SimpleNamespace(to_vtk=lambda *_a, **_k: None),
                        ),
                    ),
                    patch.object(
                        module,
                        "SquaredFlux",
                        lambda *_args, **_kwargs: FakeStage2Objective(0.5, [1.0, 1.0]),
                    ),
                    patch.object(
                        module,
                        "CurveLength",
                        lambda *_args, **_kwargs: FakeStage2Objective(1.8, [0.1, 0.2]),
                    ),
                    patch.object(
                        module,
                        "CurveCurveDistance",
                        lambda *_args, **_kwargs: FakeCurveDistance(),
                    ),
                    patch.object(
                        module,
                        "LpCurveCurvature",
                        lambda *_args, **_kwargs: FakeCurvatureObjective(),
                    ),
                    patch.object(
                        module,
                        "QuadraticPenalty",
                        lambda *_args, **_kwargs: FakeStage2Objective(0.05, [0.01, 0.02]),
                    ),
                    patch.object(module, "format_local_stage2_run_dir", lambda *_args, **_kwargs: "runtime-smoke"),
                    patch.object(module, "curves_to_vtk", lambda *_args, **_kwargs: None),
                    patch.object(module, "cross_section_plot", lambda *_args, **_kwargs: None),
                    patch.object(module, "_magnetic_field_plots", lambda *_args, **_kwargs: 0.03),
                    patch.object(module, "is_self_intersecting", lambda *_args, **_kwargs: False),
                    patch.object(
                        module,
                        "_evaluate_stage2_hardware_constraints",
                        lambda *_args, **_kwargs: {"success": True, "violations": []},
                    ),
                    patch.object(module, "minimize", side_effect=fake_minimize),
                    patch.object(module, "minimize_alm", side_effect=fake_minimize_alm),
                    patch.object(module, "run_basin_hopping", side_effect=fake_run_basin_hopping),
                    patch.object(
                        module.json,
                        "dump",
                        side_effect=lambda data, _outfile, indent=2: runtime.__setitem__("results", data),
                    ),
                ]
                for patcher in common_patches:
                    stack.enter_context(patcher)
                if use_seed:
                    stack.enter_context(patch.object(module, "load_stage2_seed_configuration", side_effect=fake_seed_loader))
                    stack.enter_context(patch.object(module, "_initialize_coils", side_effect=AssertionError("unexpected fresh initialization")))
                else:
                    stack.enter_context(patch.object(module, "load_stage2_seed_configuration", side_effect=AssertionError("unexpected seed load")))
                    stack.enter_context(patch.object(module, "_initialize_coils", side_effect=fake_initialize_coils))
                module.main(args)

        return runtime

    def _assert_runtime_counts(
        self,
        runtime,
        *,
        seed_loads,
        initialize_calls,
        minimize_calls,
        minimize_alm_calls,
    ):
        self.assertEqual(runtime["seed_loads"], seed_loads)
        self.assertEqual(runtime["initialize_calls"], initialize_calls)
        self.assertEqual(runtime["minimize_calls"], minimize_calls)
        self.assertEqual(runtime["minimize_alm_calls"], minimize_alm_calls)
        expected_basin_calls = 1 if runtime["results"]["basin_hops"] > 0 else 0
        self.assertEqual(runtime["run_basin_hopping_calls"], expected_basin_calls)

    def test_stage2_main_init_only_loads_seed_and_writes_results(self):
        runtime = self._run_stage2_main(init_only=True, constraint_method="penalty", use_seed=True)

        self._assert_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
            minimize_calls=0,
            minimize_alm_calls=0,
        )
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "init_only")
        self.assertTrue(runtime["results"]["OPTIMIZER_SUCCESS"])
        self.assertEqual(runtime["results"]["iterations"], 0)
        self.assertTrue(runtime["results"]["HARDWARE_CONSTRAINTS_OK"])
        self.assertTrue(runtime["results"]["STAGE2_BS_PATH"].endswith("seed.json"))

    def test_stage2_main_alm_path_uses_minimize_alm(self):
        runtime = self._run_stage2_main(init_only=False, constraint_method="alm", use_seed=True)

        self._assert_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
            minimize_calls=0,
            minimize_alm_calls=1,
        )
        self.assertEqual(runtime["results"]["CONSTRAINT_METHOD"], "alm")
        self.assertEqual(runtime["results"]["ALM_OUTER_ITERATIONS"], 2)
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "alm_ok")

    def test_stage2_main_penalty_path_uses_lbfgsb(self):
        runtime = self._run_stage2_main(init_only=False, constraint_method="penalty", use_seed=True)

        self._assert_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
            minimize_calls=1,
            minimize_alm_calls=0,
        )
        self.assertEqual(runtime["results"]["CONSTRAINT_METHOD"], "penalty")
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "penalty_ok")

    def test_stage2_main_basin_hopping_persists_telemetry(self):
        runtime = self._run_stage2_main(
            init_only=False,
            constraint_method="penalty",
            use_seed=True,
            basin_hops=2,
        )

        self._assert_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
            minimize_calls=0,
            minimize_alm_calls=0,
        )
        self.assertEqual(runtime["results"]["basin_hops"], 2)
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "basin_ok")
        self.assertEqual(runtime["results"]["basin_iterations"], 2)
        self.assertEqual(runtime["results"]["basin_minimization_failures"], 1)
        self.assertEqual(runtime["results"]["basin_temperature"], 2.5)
        self.assertEqual(runtime["results"]["basin_niter_success"], 6)
        for key, expected in self._EXPECTED_BASIN_TELEMETRY.items():
            self.assertEqual(runtime["results"][key], expected)

    def test_stage2_main_fresh_init_path_uses_initialize_coils(self):
        runtime = self._run_stage2_main(init_only=True, constraint_method="penalty", use_seed=False)

        self._assert_runtime_counts(
            runtime,
            seed_loads=0,
            initialize_calls=1,
            minimize_calls=0,
            minimize_alm_calls=0,
        )
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "init_only")
        self.assertIsNone(runtime["results"]["STAGE2_BS_PATH"])


class AlmUtilsTests(unittest.TestCase):
    def test_upper_bound_residual_clamps_negative_values(self):
        module = load_alm_utils_module()

        self.assertEqual(module.upper_bound_residual(1.0, 2.0), 0.0)
        self.assertEqual(module.upper_bound_residual(2.5, 2.0), 0.5)

    def test_augmented_objective_combines_base_and_constraints(self):
        module = load_alm_utils_module()

        evaluation = module.augmented_objective(
            base_value=3.0,
            base_grad=np.array([1.0, -1.0]),
            constraint_values=[0.5, 0.0],
            constraint_grads=[np.array([2.0, 0.0]), np.array([0.0, 0.0])],
            multipliers=np.array([1.0, 7.0]),
            penalty=10.0,
        )

        self.assertAlmostEqual(evaluation["total"], 4.75)
        np.testing.assert_allclose(evaluation["grad"], np.array([13.0, -1.0]))
        self.assertAlmostEqual(evaluation["max_violation"], 0.5)

    def test_minimize_alm_solves_simple_quadratic_with_signed_upper_bound_constraint(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=6,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-8,
            stationarity_tol=1e-8,
        )

        def evaluate_problem(x, multipliers, penalty):
            value = 0.5 * (x[0] - 2.0) ** 2
            grad = np.array([x[0] - 2.0])
            signed_constraint_value = np.array([x[0] - 1.0])
            constraint_grad = [np.array([1.0])]
            return module.augmented_inequality_objective(
                value,
                grad,
                signed_constraint_value,
                constraint_grad,
                multipliers,
                penalty,
            )

        result = module.minimize_alm(
            np.array([0.0]),
            ["x_upper_bound"],
            evaluate_problem,
            settings,
            {"maxiter": 50, "maxcor": 20, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertTrue(result.success)
        self.assertLessEqual(result.x[0], 1.0 + 1e-6)
        self.assertLessEqual(result.constraint_values[0], 1e-6)

    def test_minimize_alm_failure_reports_last_solved_subproblem_state(self):
        module = load_alm_utils_module()
        settings = module.ALMSettings(
            max_outer_iterations=1,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-12,
            stationarity_tol=1e-12,
        )

        def evaluate_problem(x, multipliers, penalty):
            value = 0.5 * (x[0] - 2.0) ** 2
            grad = np.array([x[0] - 2.0])
            constraint_value = module.upper_bound_residual(x[0], 1.0)
            constraint_grad = np.array([1.0]) if constraint_value > 0.0 else np.array([0.0])
            return module.augmented_objective(
                value,
                grad,
                [constraint_value],
                [constraint_grad],
                multipliers,
                penalty,
            )

        result = module.minimize_alm(
            np.array([0.0]),
            ["x_upper_bound"],
            evaluate_problem,
            settings,
            {"maxiter": 50, "maxcor": 20, "ftol": 1e-12, "gtol": 1e-12},
        )

        self.assertFalse(result.success)
        self.assertAlmostEqual(result.penalty, 1.0)
        self.assertEqual(result.multipliers, [0.0])


class InitOnlyResultTests(unittest.TestCase):
    def test_final_topology_gate_for_results_skips_expensive_probe_in_init_only(self):
        module = load_single_stage_example_module()

        with patch.object(module, "evaluate_search_topology_gate", side_effect=AssertionError("should not run")):
            status = module.final_topology_gate_for_results(True, 2, object(), object())

        self.assertFalse(status["evaluated"])
        self.assertIsNone(status["success"])
        self.assertIsNone(status["stop_reason_counts"])


if __name__ == "__main__":
    unittest.main()
