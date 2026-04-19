import copy
import importlib.util
from contextlib import ExitStack
import json
import os
import re
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
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
TOPOLOGY_FIDELITY_LADDER_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "topology_fidelity_ladder.py"
)
ALM_UTILS_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "alm_utils.py"
)
WORKFLOW_HELPERS_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "workflow_helpers.py"
)
WORKFLOW_RUNNER_COMMON_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "workflow_runner_common.py"
)
TEST_MPOL = 8
TEST_NTOR = 6
TEST_VOL_TARGET = 0.1
TEST_IOTA = 0.15
TEST_G0 = 1.0
TEST_BOOZER_I = 0.37


def _load_module_from_path(module_path, name_prefix, *, register_in_sys_modules=False):
    spec = importlib.util.spec_from_file_location(
        f"{name_prefix}_{uuid.uuid4().hex}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    if register_in_sys_modules:
        sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_single_stage_example_module():
    return _load_module_from_path(
        EXAMPLE_MODULE_PATH,
        "single_stage_banana_example",
    )


def load_topology_scorer_module():
    return _load_module_from_path(
        TOPOLOGY_SCORER_MODULE_PATH,
        "topology_scorer",
    )


def load_topology_fidelity_ladder_module():
    return _load_module_from_path(
        TOPOLOGY_FIDELITY_LADDER_MODULE_PATH,
        "topology_fidelity_ladder",
        register_in_sys_modules=True,
    )


def load_alm_utils_module():
    return _load_module_from_path(
        ALM_UTILS_MODULE_PATH,
        "alm_utils",
    )


def load_workflow_helpers_module():
    return _load_module_from_path(
        WORKFLOW_HELPERS_MODULE_PATH,
        "workflow_helpers",
        register_in_sys_modules=True,
    )


def load_workflow_runner_common_module():
    return _load_module_from_path(
        WORKFLOW_RUNNER_COMMON_MODULE_PATH,
        "workflow_runner_common",
        register_in_sys_modules=True,
    )


def _mock_topology_score_result(
    *,
    stop_reason,
    first_exit_time,
    survival_fraction=0.5,
    survived_lines=1,
    seed_mode="midplane_radial_sweep",
    field_mode="native",
):
    return {
        "survival_fraction": float(survival_fraction),
        "survived_lines": int(survived_lines),
        "stop_reason_counts": {str(stop_reason): 1},
        "first_exit": {
            "first_exit_time": float(first_exit_time),
            "first_exit_angle": 0.0,
            "stop_reason": str(stop_reason),
        },
        "seed_contract": {"mode": str(seed_mode)},
        "field_model": {"selected_mode": str(field_mode)},
    }


def phase1_runtime_kwargs(module, *, phase1_config=None):
    resolved_phase1_config = (
        module.build_phase1_config()
        if phase1_config is None
        else phase1_config
    )
    return {
        "phase1_config": resolved_phase1_config,
        "refinement_eligible_fn": module.refinement_eligible_incumbent,
        "repair_progress_state_fn": module.repair_progress_state,
    }


def make_frontier_goal_config(module, **overrides):
    config = {
        "iota_reference": 0.10,
        "iota_scale": 0.05,
        "volume_reference": 0.10,
        "volume_scale": 0.01,
        "qs_reference": 1.0e-4,
        "boozer_reference": 1.0e-6,
        "boozer_trust_threshold": 1.0e-5,
        "boozer_trust_penalty_scale": 5.0e-5,
        "effective_qs_weight": 1.0,
        "effective_boozer_weight": 1.0,
        "effective_iota_weight": 1.0,
        "effective_volume_weight": 1.0,
        "scalarization_type": "weight_schedule_v1",
        "chebyshev_rho": 1.0e-3,
        "chebyshev_weight_iota": 1.0,
        "chebyshev_weight_volume": 1.0,
        "chebyshev_weight_qa": 1.0,
        "chebyshev_weight_boozer": 1.0,
        "epsilon_constraint_qa_max": None,
        "epsilon_constraint_boozer_max": None,
    }
    config.update(overrides)
    return module.FrontierGoalConfig(**config)


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


class FakeProjectedObjective(FakeAlgebraicObjective):
    def __init__(self, value, gradient, projected_gradient):
        super().__init__(value, gradient)
        self._projected_gradient = np.asarray(projected_gradient, dtype=float)

    def dJ(self, partials=False):
        if not partials:
            return super().dJ()
        return lambda objective_optimizable: self._projected_gradient.copy()

    def __add__(self, other):
        if other == 0:
            return self
        return FakeProjectedObjective(
            self._value + other._value,
            self._gradient + other._gradient,
            self._projected_gradient + other._projected_gradient,
        )

    __radd__ = __add__

    def __mul__(self, scalar):
        return FakeProjectedObjective(
            self._value * scalar,
            self._gradient * scalar,
            self._projected_gradient * scalar,
        )

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
        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["effective_mode"], "wataru_proxy_field")
        self.assertEqual(settings["plasma_current_A"], 8000.0)
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * 8000.0)
        self.assertEqual(settings["boozer_current_convention"], "mu0")

    def test_resolve_plasma_current_settings_zero_physical_amps_reports_vacuum_effective_mode(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=0.0,
            )
        )

        self.assertEqual(settings["input_source"], "physical_A")
        self.assertEqual(settings["mode"], "wataru_proxy_field")
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
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * -35200.0)
        self.assertEqual(settings["effective_mode"], "wataru_proxy_field")

    def test_resolve_plasma_current_settings_rejects_mixed_raw_and_physical_inputs(self):
        module = self.load_module()

        with self.assertRaisesRegex(ValueError, "--plasma-current-A"):
            module.resolve_plasma_current_settings(
                SimpleNamespace(
                    boozer_I=0.5,
                    plasma_current_A=8000.0,
                )
            )

    def test_resolve_plasma_current_settings_defaults_to_surrogate_zero(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=None,
            )
        )

        self.assertEqual(settings["input_source"], "default_zero")
        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["effective_mode"], "vacuum")
        self.assertEqual(settings["plasma_current_A"], 0.0)
        self.assertEqual(settings["boozer_I"], 0.0)
        self.assertEqual(settings["boozer_current_convention"], "mu0")

    def test_resolve_plasma_current_settings_single_surface_normalizes_legacy_mode(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=9100.0,
                finite_current_mode=None,
            ),
            finite_current_mode="boozer_surrogate",
            num_surfaces=1,
        )

        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["effective_mode"], "wataru_proxy_field")
        self.assertEqual(settings["input_source"], "physical_A")
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * 9100.0)

    def test_resolve_plasma_current_settings_single_surface_allows_raw_boozer_override(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=0.125,
                plasma_current_A=None,
                finite_current_mode=None,
            ),
            finite_current_mode="boozer_surrogate",
            num_surfaces=1,
        )

        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["input_source"], "raw_boozer_I")
        self.assertAlmostEqual(settings["boozer_I"], 0.125)
        self.assertAlmostEqual(settings["plasma_current_A"], 0.125 / (4.0e-7 * np.pi))

    def test_resolve_plasma_current_settings_single_surface_rejects_conflicting_requested_mode(self):
        module = self.load_module()

        with self.assertRaisesRegex(ValueError, "Single-surface mode is locked to"):
            module.resolve_plasma_current_settings(
                SimpleNamespace(
                    boozer_I=None,
                    plasma_current_A=9100.0,
                    finite_current_mode="boozer_surrogate",
                ),
                finite_current_mode="boozer_surrogate",
                num_surfaces=1,
            )

    def test_resolve_plasma_current_settings_multisurface_preserves_requested_mode(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=6400.0,
                finite_current_mode="boozer_surrogate",
            ),
            finite_current_mode="boozer_surrogate",
            num_surfaces=2,
        )

        self.assertEqual(settings["mode"], "boozer_surrogate")
        self.assertEqual(settings["effective_mode"], "boozer_surrogate")
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * 6400.0)

    def test_resolve_plasma_current_settings_uses_artifact_default_in_wataru_mode(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=None,
            ),
            finite_current_mode="wataru_proxy_field",
            default_plasma_current_A=8000.0,
        )

        self.assertEqual(settings["input_source"], "artifact_default_A")
        self.assertEqual(settings["mode"], "wataru_proxy_field")
        self.assertEqual(settings["effective_mode"], "wataru_proxy_field")
        self.assertEqual(settings["plasma_current_A"], 8000.0)
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * 8000.0)
        self.assertEqual(settings["boozer_current_convention"], "mu0")

    def test_wataru_mode_preserves_mu0_no_2pi_convention(self):
        module = self.load_module()

        settings = module.resolve_plasma_current_settings(
            SimpleNamespace(
                boozer_I=None,
                plasma_current_A=9000.0,
            ),
            finite_current_mode="wataru_proxy_field",
        )

        self.assertEqual(settings["boozer_current_convention"], "mu0")
        self.assertAlmostEqual(settings["boozer_I"], 4.0e-7 * np.pi * 9000.0)
        self.assertNotAlmostEqual(settings["boozer_I"], 2.0e-7 * 9000.0)

    def test_stage2_resolve_finite_current_mode_accepts_explicit_wataru_without_artifact(self):
        module = load_stage2_module()

        self.assertEqual(
            module.resolve_finite_current_mode(
                "wataru_proxy_field",
                artifact_mode=None,
            ),
            "wataru_proxy_field",
        )

    def test_stage2_resolve_finite_current_mode_explains_legacy_assumed_default(self):
        module = load_stage2_module()

        with self.assertRaisesRegex(
            ValueError,
            "recorded no finite-current mode, so that value was assumed as the legacy default",
        ):
            module.resolve_finite_current_mode(
                "wataru_proxy_field",
                artifact_mode="boozer_surrogate",
                artifact_mode_source="legacy_assumed_default",
            )

    def test_build_stage2_bs_path_uses_unique_globbed_current_match(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            parent = root / "local" / "outputs-demo.nc"
            parent.mkdir(parents=True)
            matched = (
                parent
                / "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-INITC=10000-MAXC=16000-TFC=80000-Order=2-CM=penalty-BH=3"
                / "biot_savart_opt.json"
            )
            matched.parent.mkdir(parents=True)
            matched.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=1.0e4,
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
                        "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-"
                        f"SR=0.220-INITC=10000-MAXC=16000-TFC=80000-Order=2{suffix}"
                    )
                    / "biot_savart_opt.json"
                )
                candidate.parent.mkdir(parents=True)
                candidate.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=1.0e4,
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
        status = module._evaluate_topology_gate_impl(
            object(),
            object(),
            2,
            2.0,
            1e-7,
            0.75,
            score_topology_fn=lambda *_args, **_kwargs: _mock_topology_score_result(
                stop_reason="surface_exit",
                first_exit_time=0.8,
            ),
        )

        self.assertTrue(status["enabled"])
        self.assertFalse(status["success"])
        self.assertEqual(status["state"], "modeled_infeasible")
        self.assertEqual(status["survived_lines"], 1)
        self.assertAlmostEqual(status["survival_fraction"], 0.5)
        self.assertEqual(status["first_exit_reason"], "surface_exit")
        self.assertAlmostEqual(status["first_exit_time"], 0.8)

    def test_evaluate_topology_gate_threads_transport_diagnostics(self):
        module = self.load_module()
        transport_diagnostics = {
            "schema_version": "single_stage_topology_transport_diagnostics_v1",
            "status": "partial",
            "gamma_c": {"status": "unavailable"},
            "effective_ripple": {"status": "unavailable", "aliases": ["epsilon_eff"]},
        }

        status = module._evaluate_topology_gate_impl(
            object(),
            object(),
            2,
            2.0,
            1e-7,
            0.75,
            score_topology_fn=lambda *_args, **_kwargs: {
                **_mock_topology_score_result(
                    stop_reason="surface_exit",
                    first_exit_time=0.8,
                ),
                "transport_diagnostics": transport_diagnostics,
            },
        )

        self.assertEqual(status["transport_diagnostics"], transport_diagnostics)

    def test_evaluate_topology_gate_wrapper_uses_shared_impl_signature(self):
        module = self.load_module()

        with patch.object(
            module,
            "_evaluate_topology_gate_impl",
            return_value={"ok": True},
        ) as gate_impl:
            result = module.evaluate_topology_gate(
                "surface",
                "field",
                2,
                2.0,
                1e-7,
                0.75,
            )

        self.assertEqual(result, {"ok": True})
        gate_impl.assert_called_once_with(
            "surface",
            "field",
            2,
            2.0,
            1e-7,
            0.75,
        )

    def test_evaluate_topology_gate_marks_iteration_limit_as_broken(self):
        module = self.load_module()
        status = module._evaluate_topology_gate_impl(
            object(),
            object(),
            2,
            2.0,
            1e-7,
            0.5,
            score_topology_fn=lambda *_args, **_kwargs: _mock_topology_score_result(
                stop_reason="iteration_limit",
                first_exit_time=0.4,
            ),
        )

        self.assertFalse(status["success"])
        self.assertEqual(status["state"], "broken")
        self.assertTrue(status["broken"])
        self.assertEqual(status["survived_lines"], 1)
        self.assertAlmostEqual(status["survival_fraction"], 0.5)

    def test_disabled_topology_gate_status_does_not_claim_feasible_state(self):
        module = self.load_module()

        status = module.disabled_topology_gate_status(2.0, 1e-7, 0.25)

        self.assertFalse(status["enabled"])
        self.assertTrue(status["success"])
        self.assertIsNone(status["state"])
        self.assertFalse(status["broken"])
        diagnostics = module.build_topology_gate_diagnostics(
            status,
            artifact_role="final_topology_gate",
        )
        self.assertEqual(diagnostics["outcome"], "disabled")

    def test_topology_gate_and_scorer_share_trace_metrics(self):
        module = self.load_module()
        topology_module = load_topology_scorer_module()

        class _Surface:
            nfp = 1

        fieldlines_tys = [
            np.array([[0.0, 1.0, 0.0, 0.0]]),
            np.array([[0.0, 1.1, 0.0, 0.0]]),
            np.array([[0.0, 1.2, 0.0, 0.0]]),
        ]
        fieldlines_phi_hits = [
            np.array([[0.4, 0.0, 1.0, 0.0, 0.0], [0.7, -1.0, 1.0, 0.0, 0.0]]),
            np.array([[0.5, 0.0, 1.1, 0.0, 0.0]]),
            np.array([]),
        ]
        stop_labels = [
            "surface_exit",
            "max_z_guardrail",
            "min_z_guardrail",
            "min_r_guardrail",
            "max_r_guardrail",
            "iteration_limit",
        ]

        with patch.object(
            topology_module,
            "build_stopping_criteria",
            return_value=([object()], stop_labels),
        ), patch.object(
            topology_module,
            "midplane_seed_radii",
            return_value=np.array([1.0, 1.1, 1.2]),
        ), patch.object(
            topology_module,
            "prepare_topology_field",
            return_value=(object(), {"selected_mode": "native"}),
        ), patch.object(
            topology_module,
            "cross_section_span",
            return_value=1.0,
        ), patch(
            "simsopt.field.compute_fieldlines",
            return_value=(fieldlines_tys, fieldlines_phi_hits),
        ):
            scorer_result = topology_module.score_topology(
                _Surface(),
                object(),
                nfieldlines=3,
                tmax=2.0,
                tol=1e-7,
                nphis=1,
                field_policy="never",
            )
        gate_status = module._evaluate_topology_gate_impl(
            _Surface(),
            object(),
            3,
            2.0,
            1e-7,
            0.60,
            score_topology_fn=lambda *_args, **_kwargs: scorer_result,
        )

        self.assertAlmostEqual(
            gate_status["survival_fraction"],
            scorer_result["survival_fraction"],
        )
        self.assertEqual(
            gate_status["survived_lines"],
            scorer_result["survived_lines"],
        )
        self.assertEqual(
            gate_status["stop_reason_counts"],
            scorer_result["stop_reason_counts"],
        )
        self.assertAlmostEqual(
            gate_status["first_exit_time"],
            scorer_result["first_exit"]["first_exit_time"],
        )
        self.assertAlmostEqual(
            gate_status["first_exit_angle"],
            scorer_result["first_exit"]["first_exit_angle"],
        )

    def test_topology_scorer_safe_wrapper_returns_broken_result_on_exception(self):
        module = load_topology_scorer_module()

        with patch.object(
            module,
            "score_topology",
            side_effect=RuntimeError("trace exploded"),
        ):
            result = module.safe_score_topology(
                object(),
                object(),
                nfieldlines=4,
                tmax=2.0,
            )

        self.assertTrue(result["broken"])
        self.assertEqual(result["evaluation_state"], "broken")
        self.assertIn("trace exploded", result["evaluation_error"])
        self.assertEqual(result["evaluation_error_type"], "RuntimeError")
        self.assertEqual(result["nfieldlines"], 4)
        self.assertEqual(result["survived_lines"], 0)
        self.assertTrue(np.isinf(result["confinement_loss"]))

    def test_trace_metrics_rejects_malformed_empty_hit_rows(self):
        topology_module = load_topology_scorer_module()

        with self.assertRaises(ValueError):
            topology_module.trace_metrics(
                [np.array([[0.0, 1.0, 0.0, 0.0]])],
                [np.empty((0, 0))],
                [],
                ["surface_exit"],
            )

    def test_trace_metrics_marks_iteration_limit_as_broken_validation(self):
        topology_module = load_topology_scorer_module()

        metrics = topology_module.trace_metrics(
            [np.array([[0.0, 1.0, 0.0, 0.0]])],
            [np.array([[0.1, -6.0, 1.0, 0.0, 0.0]])],
            [0.0],
            [
                "surface_exit",
                "max_z_guardrail",
                "min_z_guardrail",
                "min_r_guardrail",
                "max_r_guardrail",
                "iteration_limit",
            ],
            mode="validation",
        )

        self.assertEqual(metrics["validation_status"], "broken")
        self.assertEqual(metrics["stop_reason_counts"]["iteration_limit"], 1)

    def test_midplane_seed_radii_produces_inset_radial_sweep(self):
        topology_module = load_topology_scorer_module()

        class _Surface:
            nfp = 5

            def cross_section(self, phi, thetas):
                angles = np.linspace(0.0, 2.0 * np.pi, int(thetas), endpoint=False)
                R = 1.0 + 0.2 * np.cos(angles)
                Z = 0.15 * np.sin(angles)
                phi_abs = 2.0 * np.pi * float(phi)
                return np.column_stack(
                    [
                        R * np.cos(phi_abs),
                        R * np.sin(phi_abs),
                        Z,
                    ]
                )

        radii = topology_module.midplane_seed_radii(_Surface(), 12, inset_fraction=0.05)
        self.assertEqual(radii.shape, (12,))
        # With R = 1 + 0.2 cos(theta) near the midplane, R ranges over ~[0.8, 1.2].
        # The 0.05 inset takes ~5% of that span off each end.
        span = 1.2 - 0.8
        expected_inset = max(0.05 * span, 0.01)
        self.assertGreaterEqual(radii[0], 0.8 + 0.9 * expected_inset)
        self.assertLessEqual(radii[-1], 1.2 - 0.9 * expected_inset)
        self.assertTrue(np.all(np.diff(radii) > 0))

    def test_extended_surface_seed_radii_spans_extended_surface_without_mutating_input(self):
        topology_module = load_topology_scorer_module()

        class _Surface:
            def __init__(self, delta=0.0):
                self.delta = float(delta)
                self.extend_calls = []

            def copy(self):
                return _Surface(self.delta)

            def extend_via_normal(self, distance):
                self.extend_calls.append(float(distance))
                self.delta += float(distance)

            def gamma(self):
                radii = np.array([1.0 - self.delta, 1.0 + self.delta], dtype=float)
                return np.array(
                    [
                        [[radii[0], 0.0, 0.0], [radii[1], 0.0, 0.0]],
                    ],
                    dtype=float,
                )

        surface = _Surface(delta=0.2)
        radii = topology_module.extended_surface_seed_radii(
            surface,
            5,
            extend_distance=0.05,
        )

        self.assertEqual(radii.shape, (5,))
        self.assertAlmostEqual(radii[0], 0.75)
        self.assertAlmostEqual(radii[-1], 1.25)
        self.assertTrue(np.all(np.diff(radii) > 0))
        self.assertEqual(surface.extend_calls, [])

        contract = topology_module.build_extended_surface_seed_contract(
            5,
            0.05,
            radii,
        )
        self.assertEqual(contract["mode"], "extended_surface_radial_sweep")
        self.assertEqual(contract["nfieldlines"], 5)
        self.assertAlmostEqual(contract["extend_distance"], 0.05)
        self.assertEqual(contract["radial_sampling_source"], "global_extended_surface_bounds")
        self.assertAlmostEqual(contract["r_min_seed"], 0.75)
        self.assertAlmostEqual(contract["r_max_seed"], 1.25)

    def test_extended_surface_seed_radii_clones_real_surface_xyztensorfourier(self):
        topology_module = load_topology_scorer_module()
        from simsopt.geo import SurfaceXYZTensorFourier

        surface = SurfaceXYZTensorFourier(
            nfp=5,
            stellsym=True,
            mpol=2,
            ntor=1,
            quadpoints_phi=np.linspace(0.0, 1.0 / 5.0, 9, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 11, endpoint=False),
        )
        dofs = surface.get_dofs().copy()
        dofs[0] = 1.23
        surface.set_dofs(dofs)
        surface.fix(0)
        original_x = np.asarray(surface.x, dtype=float).copy()
        original_full_x = np.asarray(surface.get_dofs(), dtype=float).copy()
        original_gamma = surface.gamma().copy()

        radii = topology_module.extended_surface_seed_radii(
            surface,
            8,
            extend_distance=0.02,
        )

        self.assertEqual(radii.shape, (8,))
        self.assertTrue(np.all(np.diff(radii) > 0))
        self.assertLess(original_x.size, original_full_x.size)
        clone = topology_module._clone_surface_for_extension(surface)
        np.testing.assert_allclose(clone.get_dofs(), original_full_x)
        np.testing.assert_allclose(np.asarray(surface.x, dtype=float), original_x)
        np.testing.assert_allclose(np.asarray(surface.get_dofs(), dtype=float), original_full_x)
        np.testing.assert_allclose(surface.gamma(), original_gamma)

    def test_prepare_topology_field_auto_policy_switches_at_threshold(self):
        topology_module = load_topology_scorer_module()

        class _Surface:
            nfp = 5
            stellsym = True

            def gamma(self):
                return np.array(
                    [
                        [[1.0, 0.0, 0.1], [1.1, 0.0, -0.1]],
                        [[0.9, 0.1, 0.05], [1.05, -0.1, -0.05]],
                    ],
                    dtype=float,
                )

        class _BField:
            def __init__(self):
                self.points = None

            def set_points(self, points):
                self.points = np.asarray(points, dtype=float)

            def B(self):
                assert self.points is not None
                return np.ones((self.points.shape[0], 3), dtype=float)

        class _InterpolatedField:
            def __init__(
                self,
                source_field,
                degree,
                rrange,
                phirange,
                zrange,
                extrapolate,
                *,
                nfp,
                stellsym,
            ):
                self.source_field = source_field
                self.degree = degree
                self.rrange = rrange
                self.phirange = phirange
                self.zrange = zrange
                self.extrapolate = extrapolate
                self.nfp = nfp
                self.stellsym = stellsym
                self.points = None

            def set_points(self, points):
                self.points = np.asarray(points, dtype=float)

            def B(self):
                assert self.points is not None
                return np.ones((self.points.shape[0], 3), dtype=float)

        surface = _Surface()
        native_field = _BField()
        below_threshold_field, below_threshold_model = topology_module.prepare_topology_field(
            surface,
            native_field,
            49.9,
            field_policy="auto",
        )
        self.assertIs(below_threshold_field, native_field)
        self.assertEqual(below_threshold_model["selected_mode"], "native")
        self.assertEqual(below_threshold_model["reason"], "below_threshold")

        with patch("simsopt.field.InterpolatedField", _InterpolatedField):
            threshold_field = _BField()
            interpolated_field, interpolated_model = topology_module.prepare_topology_field(
                surface,
                threshold_field,
                50.0,
                field_policy="auto",
                interpolation_grid={
                    "degree": 5,
                    "nr": 10,
                    "nphi": 11,
                    "nz": 12,
                },
            )

        self.assertIsInstance(interpolated_field, _InterpolatedField)
        self.assertEqual(interpolated_model["selected_mode"], "interpolated")
        self.assertEqual(interpolated_model["reason"], "tmax_threshold")
        self.assertEqual(
            interpolated_model["grid"],
            {"degree": 5, "nr": 10, "nphi": 11, "nz": 12},
        )
        self.assertEqual(interpolated_model["max_abs_error"], 0.0)
        self.assertEqual(interpolated_model["mean_abs_error"], 0.0)
        self.assertEqual(interpolated_model["max_rel_error"], 0.0)

    def test_compute_topology_transport_diagnostics_reports_surface_structure(self):
        topology_module = load_topology_scorer_module()

        class _Surface:
            def gamma(self):
                return np.array(
                    [
                        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                        [[3.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
                    ],
                    dtype=float,
                )

        class _Field:
            def set_points(self, points):
                self._points = np.asarray(points, dtype=float)

            def AbsB(self):
                return self._points[:, 0] + 1.0

        diagnostics = topology_module.compute_topology_transport_diagnostics(
            _Surface(),
            _Field(),
        )

        self.assertEqual(
            diagnostics["schema_version"],
            "single_stage_topology_transport_diagnostics_v1",
        )
        self.assertEqual(diagnostics["status"], "partial")
        self.assertEqual(
            diagnostics["surface_field_structure"]["status"],
            "evaluated",
        )
        self.assertEqual(
            diagnostics["surface_field_structure"]["grid_shape"],
            [2, 2],
        )
        self.assertAlmostEqual(
            diagnostics["surface_field_structure"]["modB_min"],
            2.0,
        )
        self.assertAlmostEqual(
            diagnostics["surface_field_structure"]["modB_max"],
            5.0,
        )
        self.assertAlmostEqual(
            diagnostics["surface_field_structure"]["mirror_ratio"],
            2.5,
        )
        self.assertAlmostEqual(
            diagnostics["surface_field_structure"]["effective_inverse_aspect_ratio_epsilon"],
            3.0 / 7.0,
        )
        self.assertEqual(diagnostics["gamma_c"]["status"], "unavailable")
        self.assertEqual(
            diagnostics["effective_ripple"]["aliases"],
            ["epsilon_eff"],
        )

    def test_topology_fidelity_report_summarizes_tier_agreement(self):
        ladder_module = load_topology_fidelity_ladder_module()

        report = ladder_module.build_topology_fidelity_report(
            [
                {
                    "label": "case_a",
                    "cheap": {"passed": True, "confinement_score": 0.80},
                    "medium": {"passed": True, "confinement_score": 0.82},
                    "strict": {"passed": False, "confinement_score": 0.20},
                },
                {
                    "label": "case_b",
                    "cheap": {"passed": False, "confinement_score": 0.30},
                    "medium": {"passed": False, "confinement_score": 0.35},
                    "strict": {"passed": True, "confinement_score": 0.70},
                },
                {
                    "label": "case_c",
                    "cheap": {"passed": True, "confinement_score": 0.60},
                    "medium": {"passed": True, "confinement_score": 0.62},
                    "strict": {"passed": True, "confinement_score": 0.65},
                },
            ]
        )

        cheap_vs_strict = report["agreements"]["cheap_vs_strict"]
        medium_vs_strict = report["agreements"]["medium_vs_strict"]
        self.assertEqual(cheap_vs_strict["false_pass_count"], 1)
        self.assertEqual(cheap_vs_strict["false_reject_count"], 1)
        self.assertEqual(cheap_vs_strict["false_pass_labels"], ["case_a"])
        self.assertEqual(cheap_vs_strict["false_reject_labels"], ["case_b"])
        self.assertEqual(medium_vs_strict["false_pass_count"], 1)
        self.assertEqual(medium_vs_strict["false_reject_count"], 1)
        self.assertIsNotNone(cheap_vs_strict["spearman_rank_correlation"])
        self.assertEqual(report["schema_version"], "topology_fidelity_ladder_v2")
        cheap_spec = report["tier_specs"]["cheap"]
        self.assertEqual(
            cheap_spec["seed_mode"],
            "midplane_radial_sweep",
        )
        self.assertEqual(cheap_spec["inset_fraction"], 0.05)
        self.assertEqual(report["tier_specs"]["medium"]["field_policy"], "auto")

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
                "state": "modeled_infeasible",
                "broken": False,
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
                "evaluation_error": None,
                "evaluation_error_type": None,
            },
        ):
            J_out, dJ_out = module.fun(np.ones(5))

        self.assertEqual(J_out, 126.0)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        restore_mock.assert_called_once()
        self.assertEqual(module.run_dict["topology_gate_status"]["state"], "modeled_infeasible")
        self.assertEqual(module.run_dict["topology_gate_rejects"], 1)
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)

    def test_fun_rejects_candidate_on_broken_topology_evaluation(self):
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
            side_effect=RuntimeError("trace exploded"),
        ):
            J_out, dJ_out = module.fun(np.ones(5))

        self.assertEqual(J_out, last_J * 2.0)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        restore_mock.assert_called_once()
        self.assertEqual(module.run_dict["topology_gate_status"]["state"], "broken")
        self.assertTrue(module.run_dict["topology_gate_status"]["broken"])
        self.assertIn("trace exploded", module.run_dict["topology_gate_status"]["evaluation_error"])
        self.assertEqual(module.run_dict["topology_gate_status"]["evaluation_error_type"], "RuntimeError")
        self.assertEqual(module.run_dict["topology_gate_rejects"], 0)
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 1)

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
        self.assertIn("coil_coil_spacing", status["violations"][0])
        self.assertIn("max_curvature", status["violations"][1])
        self.assertIn("coil_length", status["violations"][2])

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
        self.assertIn("coil_coil_spacing", status["violations"][0])
        self.assertIn("coil_surface_spacing", status["violations"][1])
        self.assertIn("surface_vessel_spacing", status["violations"][2])
        self.assertIn("max_curvature", status["violations"][3])

    def test_single_stage_hardware_constraints_report_current_and_length_violations(self):
        module = load_single_stage_example_module()

        status = module.evaluate_single_stage_hardware_constraints(
            curve_curve_min_dist=0.05,
            cc_dist=0.05,
            curve_surface_min_dist=0.02,
            cs_dist=0.02,
            surface_vessel_min_dist=0.04,
            ss_dist=0.04,
            max_curvature=40.0,
            curvature_threshold=40.0,
            coil_length=1.8,
            length_target=1.7,
            tf_current_A=9.0e4,
            tf_current_limit_A=8.0e4,
            banana_current_A=1.7e4,
            banana_current_max_A=1.6e4,
        )
        search_status = status["search_hardware_status"]
        artifact_status = status["artifact_hardware_status"]

        self.assertFalse(search_status["success"])
        self.assertEqual(search_status["violations"], [
            "|banana_current| 17000.000000 exceeds threshold 16000.000000"
        ])
        self.assertEqual(
            search_status["allowed_traversal_status"]["violations"],
            [],
        )
        self.assertEqual(
            search_status["forbidden_traversal_status"]["violations"],
            ["|banana_current| 17000.000000 exceeds threshold 16000.000000"],
        )
        self.assertFalse(artifact_status["success"])
        self.assertEqual(len(artifact_status["violations"]), 3)
        self.assertIn("coil_length", artifact_status["violations"][0])
        self.assertIn("banana_current", artifact_status["violations"][1])
        self.assertIn("tf_current", artifact_status["violations"][2])
        self.assertEqual(
            artifact_status["allowed_traversal_status"]["violations"],
            ["coil_length 1.800000 exceeds threshold 1.700000"],
        )
        self.assertEqual(
            artifact_status["forbidden_traversal_status"]["violations"],
            [
                "|banana_current| 17000.000000 exceeds threshold 16000.000000",
                "|tf_current| 90000.000000 exceeds threshold 80000.000000",
            ],
        )
        self.assertEqual(
            artifact_status["constraints"]["coil_length"]["threshold"],
            1.7,
        )
        self.assertEqual(
            artifact_status["constraints"]["tf_current"]["threshold"],
            8.0e4,
        )
        self.assertEqual(
            search_status["constraints"]["banana_current"]["threshold"],
            1.6e4,
        )

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

    def test_refinement_eligible_incumbent_returns_python_bool(self):
        module = load_single_stage_example_module()
        run_dict = {
            "search_eval": {"total": 4.0},
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "intersecting": False,
        }

        result = module.refinement_eligible_incumbent(run_dict)

        self.assertIs(type(result), bool)
        self.assertTrue(result)

    def test_write_json_artifact_normalizes_numpy_scalars(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "results.json"
            module.write_json_artifact(
                str(artifact_path),
                {
                    "FINAL_FEASIBILITY_OK": np.bool_(True),
                    "SEARCH_OBJECTIVE_J": np.float64(1.25),
                    "INVALID_STATE_REJECTS_TOTAL": np.int64(3),
                    "FINAL_SEARCH_SURFACE_WEIGHTS": np.array([1.0, 0.5]),
                    "nested": {
                        "success": np.bool_(False),
                        "violations": [np.str_("cc")],
                    },
                },
            )

            payload = json.loads(artifact_path.read_text(encoding="utf-8"))

        self.assertIs(payload["FINAL_FEASIBILITY_OK"], True)
        self.assertEqual(payload["SEARCH_OBJECTIVE_J"], 1.25)
        self.assertEqual(payload["INVALID_STATE_REJECTS_TOTAL"], 3)
        self.assertEqual(payload["FINAL_SEARCH_SURFACE_WEIGHTS"], [1.0, 0.5])
        self.assertIs(payload["nested"]["success"], False)
        self.assertEqual(payload["nested"]["violations"], ["cc"])

    def test_append_jsonl_artifact_rewrites_archive_without_partial_lines(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            archive_path = Path(tmpdir) / "topology_archive.jsonl"
            archive_path.write_text('{"accepted_iteration": 0}\n', encoding="utf-8")

            module.append_jsonl_artifact(
                str(archive_path),
                {"accepted_iteration": np.int64(1), "weights": np.array([1.0, 0.5])},
            )

            payloads = [
                json.loads(line)
                for line in archive_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(
            payloads,
            [
                {"accepted_iteration": 0},
                {"accepted_iteration": 1, "weights": [1.0, 0.5]},
            ],
        )

    def test_normalize_optimizer_termination_message_enriches_blank_abnormal(self):
        module = load_single_stage_example_module()

        message = module.normalize_optimizer_termination_message(
            "ABNORMAL: ",
            success=False,
            status=np.int64(8),
            invalid_state_rejects_total=np.int64(29),
            surface_solve_rejects=np.int64(29),
            hardware_rejects=np.int64(0),
            topology_gate_rejects=np.int64(0),
        )

        self.assertEqual(
            message,
            "ABNORMAL: empty SciPy L-BFGS-B task; status=8; "
            "invalid_state_rejects=29; surface_solve_rejects=29; "
            "hardware_rejects=0; topology_gate_rejects=0",
        )

    def test_normalize_optimizer_termination_message_decodes_bytes_abnormal(self):
        module = load_single_stage_example_module()

        message = module.normalize_optimizer_termination_message(
            b"ABNORMAL: ",
            success=False,
            status=8,
            invalid_state_rejects_total=11,
        )

        self.assertEqual(
            message,
            "ABNORMAL: empty SciPy L-BFGS-B task; status=8; invalid_state_rejects=11",
        )

    def test_normalize_optimizer_termination_message_preserves_non_abnormal_text(self):
        module = load_single_stage_example_module()

        message = module.normalize_optimizer_termination_message(
            "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
            success=False,
            status=5,
        )

        self.assertEqual(message, "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT")

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

    def test_maybe_update_best_accepted_incumbent_tracks_valid_nonself_intersecting_states(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {"total": 4.0, "surface_weights": np.array([1.0])},
            "surface_status": {"success": True},
            "search_surface_status": {"success": True},
            "accepted_hardware_status": {"success": False, "violations": ["max_curvature"]},
            "topology_gate_status": {"enabled": False, "success": True},
            "intersecting": False,
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
        }

        self.assertTrue(module.maybe_update_best_accepted_incumbent(run_dict, "initial"))
        self.assertEqual(run_dict["best_accepted_metric"], 4.0)
        self.assertEqual(run_dict["best_accepted_stage"], "initial")
        np.testing.assert_allclose(run_dict["best_accepted_incumbent"].x, [1.0, 2.0])

        run_dict["search_eval"] = {"total": 5.0, "surface_weights": np.array([1.0])}
        run_dict["J"] = 5.0
        self.assertFalse(module.maybe_update_best_accepted_incumbent(run_dict, "middle"))
        self.assertEqual(run_dict["best_accepted_metric"], 4.0)

        run_dict["intersecting"] = True
        run_dict["search_eval"] = {"total": 3.0, "surface_weights": np.array([1.0])}
        run_dict["J"] = 3.0
        self.assertFalse(module.maybe_update_best_accepted_incumbent(run_dict, "final"))
        self.assertEqual(run_dict["best_accepted_metric"], 4.0)

    def test_frontier_preserved_incumbents_require_trust_and_frontier_rank_metric(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        run_dict = {
            "accepted_x": np.array([1.0, 2.0]),
            "surface_state": {"sdofs": [np.array([1.0])], "iota": [0.15], "G": [1.0]},
            "J": 4.0,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {
                "total": 100.0,
                "frontier_rank_total": 4.0,
                "frontier_trust_ok": True,
                "surface_weights": np.array([1.0]),
            },
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

        run_dict["search_eval"]["frontier_trust_ok"] = False
        run_dict["search_eval"]["frontier_rank_total"] = 3.0
        run_dict["J"] = 3.0
        self.assertFalse(module.maybe_update_best_feasible_incumbent(run_dict, "final"))
        self.assertEqual(run_dict["best_feasible_metric"], 4.0)

        run_dict["search_eval"]["frontier_trust_ok"] = True
        run_dict["search_eval"]["finite_eval_ok"] = False
        run_dict["search_eval"]["frontier_rank_total"] = 2.0
        run_dict["J"] = 2.0
        self.assertFalse(module.maybe_update_best_feasible_incumbent(run_dict, "final"))
        self.assertEqual(run_dict["best_feasible_metric"], 4.0)

    def test_refinement_eligible_incumbent_requires_topology_success(self):
        module = load_single_stage_example_module()
        run_dict = {
            "accepted_hardware_status": {"success": True},
            "topology_gate_status": {"enabled": True, "success": False},
            "surface_status": {"success": True},
            "search_eval": {"total": 1.0},
            "intersecting": False,
        }

        self.assertFalse(module.refinement_eligible_incumbent(run_dict))

        run_dict["topology_gate_status"] = {"enabled": True, "success": True}
        self.assertTrue(module.refinement_eligible_incumbent(run_dict))

    def test_write_preserved_timeout_artifacts_uses_kind_specific_filenames(self):
        module = load_single_stage_example_module()

        class FakeBiotSavart:
            def __init__(self):
                self.saved = []

            def save(self, path):
                self.saved.append(path)

        class FakeSurface:
            def __init__(self):
                self.saved = []

            def save(self, path):
                self.saved.append(path)

        class FakeBoozerSurface:
            def __init__(self):
                self.surface = FakeSurface()
                self.saved = []

            def save(self, path):
                self.saved.append(path)

        fake_bs = FakeBiotSavart()
        fake_outer = FakeBoozerSurface()

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            payload = {"FIELD_ERROR": 0.01}
            module.write_preserved_timeout_artifacts(
                out_dir,
                preservation_kind="best_feasible",
                results_payload=payload,
                biotsavart=fake_bs,
                surface_data=[{"name": "outer", "boozer_surface": fake_outer}],
            )

            partial_results = out_dir / "results_best_feasible.partial.json"
            self.assertTrue(partial_results.exists())
            self.assertEqual(json.loads(partial_results.read_text(encoding="utf-8")), payload)
            self.assertEqual(fake_bs.saved, [str(out_dir / "biot_savart_best_feasible.json")])
            self.assertEqual(
                fake_outer.surface.saved,
                [str(out_dir / "surf_best_feasible_outer.json")],
            )
            self.assertEqual(
                fake_outer.saved,
                [str(out_dir / "surf_best_feasible_outer_boozer_surface.json")],
            )

    def test_build_topology_gate_diagnostics_distinguishes_pass_reject_and_broken(self):
        module = load_single_stage_example_module()

        passed = module.build_topology_gate_diagnostics(
            {
                "enabled": True,
                "evaluated": True,
                "success": True,
                "state": "feasible",
                "survived_lines": 4,
                "nfieldlines": 4,
                "survival_fraction": 1.0,
                "survival_threshold": 0.5,
            },
            artifact_role="final_topology_gate",
        )
        rejected = module.build_topology_gate_diagnostics(
            {
                "enabled": True,
                "evaluated": True,
                "success": False,
                "state": "modeled_infeasible",
                "survived_lines": 1,
                "nfieldlines": 4,
                "survival_fraction": 0.25,
                "survival_threshold": 0.5,
                "first_exit_reason": "surface_exit",
                "first_exit_time": 0.2,
                "first_exit_angle": 0.1,
            },
            artifact_role="final_topology_gate",
        )
        broken = module.build_topology_gate_diagnostics(
            {
                "enabled": True,
                "evaluated": True,
                "success": False,
                "state": "broken",
                "broken": True,
                "evaluation_error": "trace exploded",
                "evaluation_error_type": "RuntimeError",
            },
            artifact_role="final_topology_gate",
        )

        self.assertEqual(passed["outcome"], "pass")
        self.assertEqual(passed["reason"], "survival_threshold_met")
        self.assertIn("Topology gate pass", passed["summary"])
        self.assertEqual(rejected["outcome"], "reject")
        self.assertEqual(rejected["reason"], "surface_exit")
        self.assertIn("surface_exit", rejected["summary"])
        self.assertEqual(broken["outcome"], "broken")
        self.assertEqual(broken["reason"], "RuntimeError")
        self.assertIn("trace exploded", broken["summary"])

    def test_build_topology_gate_diagnostics_keeps_skipped_gate_disabled(self):
        module = load_single_stage_example_module()

        diagnostics = module.build_topology_gate_diagnostics(
            module.skipped_topology_gate_status(),
            artifact_role="final_topology_gate",
        )

        self.assertEqual(diagnostics["outcome"], "not_evaluated")
        self.assertFalse(diagnostics["enabled"])
        self.assertFalse(diagnostics["evaluated"])
        self.assertEqual(diagnostics["reason"], "not_evaluated")

    def test_write_topology_checkpoint_artifacts_writes_diagnostics(self):
        module = load_single_stage_example_module()

        class FakeBiotSavart:
            def __init__(self):
                self.saved = []

            def save(self, path):
                self.saved.append(path)

        class FakeBoozerSurface:
            def __init__(self):
                self.surface = object()

        fake_bs = FakeBiotSavart()
        fake_outer = FakeBoozerSurface()
        topology_entry = {
            "accepted_iteration": 7,
            "topology_state": "evaluated",
            "topology_broken": False,
            "survived_lines": 10,
            "nfieldlines": 12,
            "confinement_score": 0.91,
            "confinement_loss": 0.08,
            "transport_diagnostics": {
                "schema_version": "single_stage_topology_transport_diagnostics_v1",
                "status": "partial",
                "effective_ripple": {
                    "status": "unavailable",
                    "aliases": ["epsilon_eff"],
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "best_topology"
            with patch.object(module, "save_surface_artifacts") as save_surface_artifacts:
                module.write_topology_checkpoint_artifacts(
                    out_dir,
                    artifact_role="best_topology_checkpoint",
                    topology_entry=topology_entry,
                    biotsavart=fake_bs,
                    surface_data=[{"name": "outer", "boozer_surface": fake_outer}],
                )

            diagnostics = json.loads(
                (out_dir / "topology_diagnostics.json").read_text(encoding="utf-8")
            )
            self.assertEqual(diagnostics["kind"], "score")
            self.assertEqual(diagnostics["artifact_role"], "best_topology_checkpoint")
            self.assertEqual(diagnostics["outcome"], "scored")
            self.assertEqual(diagnostics["entry"]["accepted_iteration"], 7)
            self.assertEqual(
                diagnostics["entry"]["transport_diagnostics"]["effective_ripple"]["aliases"],
                ["epsilon_eff"],
            )
            self.assertEqual(fake_bs.saved, [str(out_dir / "biot_savart.json")])
            save_surface_artifacts.assert_called_once_with(
                [{"name": "outer", "boozer_surface": fake_outer}],
                fake_bs,
                out_dir,
                "surf",
                also_write_outer_legacy=False,
            )

    def test_build_preserved_timeout_results_payload_includes_replay_metadata(self):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path="/equilibria/wout_test.nc",
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": False},
            "topology_gate_status": {"success": True},
        }
        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_accepted",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {
                    "success": False,
                    "violations": ["coil_coil_spacing 0.049600 below threshold 0.050000"],
                },
                "artifact_hardware_status": {
                    "success": False,
                    "violations": ["coil_coil_spacing 0.049600 below threshold 0.050000"],
                },
                "max_curvature": 19.8,
                "length_target": 1.7,
                "tf_current_A": 8.0e4,
                "tf_current_limit_A": 8.0e4,
                "banana_current_A": 1.4e4,
                "banana_current_max_A": 1.6e4,
                "curve_curve_min_dist": 0.0496,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=2.91,
            accepted_iteration=1,
        )

        self.assertEqual(payload["PLASMA_SURF_PATH"], "/equilibria/wout_test.nc")
        self.assertEqual(payload["STAGE2_BS_PATH"], "/seeds/biot_savart_opt.json")
        self.assertEqual(payload["STAGE2_RESULTS_PATH"], "/seeds/results.json")
        self.assertEqual(payload["mpol"], 8)
        self.assertEqual(payload["ntor"], 6)
        self.assertEqual(payload["nphi"], 127)
        self.assertEqual(payload["ntheta"], 32)
        self.assertEqual(payload["CONSTRAINT_WEIGHT"], 1.0)
        self.assertEqual(payload["max_iterations"], 30)
        self.assertEqual(payload["TARGET_VOLUME"], 0.10)
        self.assertEqual(payload["TARGET_IOTA"], 0.15)
        self.assertEqual(payload["FINAL_SOURCE_STAGE"], payload["PRESERVED_TIMEOUT_SALVAGE_STAGE"])
        self.assertEqual(payload["FINAL_TOPOLOGY_GATE_DIAGNOSTICS"]["kind"], "gate")
        self.assertEqual(
            payload["FINAL_TOPOLOGY_GATE_DIAGNOSTICS"]["outcome"],
            "pass",
        )
        self.assertEqual(payload["MAX_CURVATURE"], 19.8)
        self.assertEqual(payload["COIL_LENGTH"], 2.91)
        self.assertEqual(payload["LENGTH_TARGET"], 1.7)
        self.assertEqual(payload["TF_CURRENT_A"], 8.0e4)
        self.assertEqual(payload["TF_CURRENT_LIMIT_A"], 8.0e4)
        self.assertEqual(payload["BANANA_CURRENT_A"], 1.4e4)
        self.assertEqual(payload["BANANA_CURRENT_MAX_A"], 1.6e4)
        self.assertIsNone(payload["FINAL_TOPOLOGY_TRANSPORT_DIAGNOSTICS"])

    def test_build_preserved_timeout_results_payload_includes_alm_runtime_state(self):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path="/equilibria/wout_test.nc",
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="alm",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
        )
        run_dict = {
            "search_eval": {
                "total": 9.5e-4,
                "base_total": 8.1e-4,
                "max_feasibility_violation": 4.0e-3,
                "metric_stationarity_norm": 7.5e-5,
                "constraint_values": np.array([1.0e-3, -2.0e-4]),
            },
            "J": 9.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
            "alm_outer_iteration": 3,
            "alm_feasibility_tolerance": 1.0e-4,
            "alm_stationarity_tolerance": 2.0e-4,
        }
        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_feasible",
            incumbent_stage="final",
            run_dict=run_dict,
            objective_eval={"J_QS": 1.7e-4, "J_Boozer": 8.0e-7},
            field_error=2.5e-4,
            final_iota=0.151,
            final_volume=0.101,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "max_curvature": 18.2,
                "curve_curve_min_dist": 0.051,
                "curve_surface_min_dist": 0.068,
                "surface_vessel_min_dist": 0.084,
            },
            coil_length=2.85,
            accepted_iteration=4,
            alm_runtime_state=module.build_preserved_timeout_alm_state(
                constraint_method="alm",
                penalty=12.5,
                multipliers=np.array([0.25, -0.75]),
            ),
        )

        self.assertEqual(payload["ALM_FORMULATION"], "weighted_sum")
        self.assertEqual(payload["ALM_OUTER_ITERATIONS"], 3)
        self.assertEqual(payload["ALM_FINAL_PENALTY"], 12.5)
        self.assertEqual(payload["ALM_FINAL_MULTIPLIERS"], [0.25, -0.75])
        self.assertEqual(payload["ALM_FINAL_CONSTRAINT_VALUES"], [1.0e-3, -2.0e-4])
        self.assertEqual(payload["ALM_FINAL_FEASIBILITY_TOL"], 1.0e-4)
        self.assertEqual(payload["ALM_FINAL_STATIONARITY_TOL"], 2.0e-4)
        self.assertEqual(payload["ALM_FINAL_MAX_FEASIBILITY_VIOLATION"], 4.0e-3)
        self.assertEqual(payload["ALM_FINAL_STATIONARITY_NORM"], 7.5e-5)

    def test_build_preserved_timeout_results_payload_uses_artifact_hardware_status_for_final_feasibility(self):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path="/equilibria/wout_test.nc",
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }

        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_accepted",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {
                    "success": False,
                    "violations": ["coil_length 1.800000 exceeds threshold 1.700000"],
                },
                "max_curvature": 19.8,
                "curve_curve_min_dist": 0.0501,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=1.8,
            accepted_iteration=1,
        )

        self.assertIs(payload["FINAL_FEASIBILITY_OK"], False)
        self.assertIs(payload["HARDWARE_CONSTRAINTS_OK"], False)
        self.assertEqual(
            payload["HARDWARE_CONSTRAINT_VIOLATIONS"],
            ["coil_length 1.800000 exceeds threshold 1.700000"],
        )

    def test_build_preserved_timeout_results_payload_backfills_missing_coil_length(self):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path="/equilibria/wout_test.nc",
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }

        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_accepted",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "coil_length": None,
                "max_curvature": 19.8,
                "curve_curve_min_dist": 0.0501,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=1.83,
            accepted_iteration=1,
        )

        self.assertEqual(payload["COIL_LENGTH"], 1.83)

    def test_build_preserved_timeout_results_payload_frontier_uses_reference_metadata(self):
        module = load_single_stage_example_module()
        replay_config = module.PreservedTimeoutReplayConfig(
            plasma_surf_filename="wout_test.nc",
            plasma_surf_path="/equilibria/wout_test.nc",
            stage2_bs_path="/seeds/biot_savart_opt.json",
            stage2_results_path="/seeds/results.json",
            mpol=8,
            ntor=6,
            nphi=127,
            ntheta=32,
            constraint_weight=1.0,
            constraint_method="penalty",
            alm_formulation="weighted_sum",
            max_iterations=30,
            target_volume=0.10,
            target_iota=0.15,
            single_stage_goal_mode="frontier",
            single_stage_goal_mode_impl="frontier_tradeoff_score_v2",
            boozer_surface_target_volumes=(0.10,),
            frontier_iota_reference=0.15,
            frontier_iota_scale=0.05,
            frontier_volume_reference=0.10,
            frontier_volume_scale=0.01,
            frontier_qs_reference=2.0e-4,
            frontier_boozer_reference=1.0e-6,
            frontier_boozer_trust_threshold=1.0e-5,
            frontier_boozer_trust_penalty_scale=5.0e-5,
            frontier_effective_qs_weight=1.0,
            frontier_effective_boozer_weight=1.0,
            frontier_effective_iota_weight=1.0,
            frontier_effective_volume_weight=1.0,
        )
        run_dict = {
            "search_eval": {
                "total": 7.5e-4,
                "base_total": 7.4e-4,
                "frontier_rank_total": 7.5e-4,
                "frontier_trust_ok": True,
                "frontier_boozer_trust_threshold": 1.0e-5,
                "frontier_boozer_trust_excess": 0.0,
                "frontier_boozer_trust_excess_ratio": 0.0,
                "frontier_boozer_trust_penalty_scale": 5.0e-5,
                "frontier_trust_penalty": 0.0,
                "J_volume": -0.2,
            },
            "J": 7.5e-4,
            "intersecting": False,
            "surface_status": {"success": True},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": True},
        }
        payload = module.build_preserved_timeout_results_payload(
            replay_config=replay_config,
            preservation_kind="best_feasible",
            incumbent_stage="initial",
            run_dict=run_dict,
            objective_eval={"J_QS": 2.7e-4, "J_Boozer": 4.8e-7},
            field_error=3.5e-4,
            final_iota=0.14997,
            final_volume=0.09998,
            hardware_snapshot={
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "max_curvature": 19.8,
                "curve_curve_min_dist": 0.0496,
                "curve_surface_min_dist": 0.067,
                "surface_vessel_min_dist": 0.082,
            },
            coil_length=2.91,
            accepted_iteration=1,
        )

        self.assertIsNone(payload["TARGET_VOLUME"])
        self.assertIsNone(payload["TARGET_IOTA"])
        self.assertEqual(payload["SINGLE_STAGE_GOAL_MODE_IMPL"], "frontier_tradeoff_score_v2")
        self.assertEqual(payload["BOOZER_SURFACE_TARGET_VOLUMES"], [0.10])
        self.assertEqual(payload["FRONTIER_REFERENCE_IOTA"], 0.15)
        self.assertEqual(payload["FRONTIER_REFERENCE_VOLUME"], 0.10)
        self.assertTrue(payload["FRONTIER_TRUST_OK"])
        self.assertEqual(payload["FRONTIER_BOOZER_TRUST_PENALTY_SCALE"], 5.0e-5)
        self.assertEqual(payload["FRONTIER_BOOZER_TRUST_EXCESS_RATIO"], 0.0)
        self.assertEqual(payload["FRONTIER_TRUST_PENALTY"], 0.0)

    def test_build_best_feasible_results_summary_emits_schema_backed_hardware_fields(self):
        module = load_single_stage_example_module()
        run_dict = {
            "best_feasible_incumbent": {"surface_state": "saved"},
            "best_feasible_stage": "accepted",
            "J": 7.5e-4,
            "dJ": np.array([1.0, -1.0]),
            "search_eval": {
                "total": 7.5e-4,
                "physics_total": 7.4e-4,
                "J_QS": 2.7e-4,
                "J_Boozer": 4.8e-7,
                "frontier_rank_total": None,
                "frontier_trust_ok": True,
            },
            "surface_status": {
                "iotas": [0.14997],
                "volumes": [0.09998],
                "success": True,
                "self_intersections": [False],
            },
            "topology_gate_status": {
                "success": True,
                "state": "pass",
                "evaluation_error": None,
                "transport_diagnostics": None,
            },
        }
        hardware_snapshot = {
            "artifact_hardware_status": {
                "success": False,
                "violations": ["coil_length 1.800000 exceeds threshold 1.700000"],
            },
            "curve_curve_min_dist": 0.0501,
            "curve_surface_min_dist": 0.067,
            "surface_vessel_min_dist": 0.082,
            "max_curvature": 19.8,
            "coil_length": 1.8,
            "length_target": 1.7,
            "tf_current_A": 8.0e4,
            "tf_current_limit_A": 8.0e4,
            "banana_current_A": 1.4e4,
            "banana_current_max_A": 1.6e4,
        }

        with patch.object(
            module,
            "snapshot_single_stage_incumbent_state",
            return_value={"surface_state": "current"},
        ), patch.object(
            module,
            "restore_single_stage_incumbent_state",
        ), patch.object(
            module,
            "evaluate_single_stage_hardware_snapshot",
            return_value=hardware_snapshot,
        ), patch.object(
            module,
            "build_topology_gate_diagnostics",
            return_value={"kind": "gate", "outcome": "pass"},
        ):
            summary = module.build_best_feasible_results_summary(
                run_dict,
                curve_curve_distance_obj=object(),
                curve_surface_distance_obj=object(),
                surface_surface_distance_obj=object(),
                banana_curve=object(),
                curvelength_obj=SimpleNamespace(J=lambda: 1.8),
                cc_dist=0.05,
                cs_dist=0.015,
                ss_dist=0.04,
                curvature_threshold=100.0,
                length_target=1.7,
                tf_current_A=8.0e4,
                banana_coils=[SimpleNamespace(current=SimpleNamespace(get_value=lambda: 1.4e4))],
                banana_current_max_A=1.6e4,
                outer_surface=object(),
                vessel_surface=object(),
            )

        self.assertTrue(summary["BEST_FEASIBLE_AVAILABLE"])
        self.assertEqual(summary["BEST_FEASIBLE_STAGE"], "accepted")
        self.assertEqual(summary["BEST_FEASIBLE_CURVE_CURVE_MIN_DIST"], 0.0501)
        self.assertEqual(summary["BEST_FEASIBLE_CURVE_SURFACE_MIN_DIST"], 0.067)
        self.assertEqual(summary["BEST_FEASIBLE_SURFACE_VESSEL_MIN_DIST"], 0.082)
        self.assertEqual(summary["BEST_FEASIBLE_MAX_CURVATURE"], 19.8)
        self.assertEqual(summary["BEST_FEASIBLE_COIL_LENGTH"], 1.8)
        self.assertEqual(summary["BEST_FEASIBLE_LENGTH_TARGET"], 1.7)
        self.assertEqual(summary["BEST_FEASIBLE_TF_CURRENT_A"], 8.0e4)
        self.assertEqual(summary["BEST_FEASIBLE_TF_CURRENT_LIMIT_A"], 8.0e4)
        self.assertEqual(summary["BEST_FEASIBLE_BANANA_CURRENT_A"], 1.4e4)
        self.assertEqual(summary["BEST_FEASIBLE_BANANA_CURRENT_MAX_A"], 1.6e4)
        self.assertFalse(summary["BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK"])
        self.assertEqual(
            summary["BEST_FEASIBLE_HARDWARE_CONSTRAINT_VIOLATIONS"],
            ["coil_length 1.800000 exceeds threshold 1.700000"],
        )

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

            def save(self, path):
                self._saved_path = path

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

            def save(self, path):
                self._saved_path = path

        surface = _Surface()
        surface_entry = {
            "name": "outer",
            "seed_label": 0.16,
            "target_volume": 1.0,
            "boozer_surface": SimpleNamespace(
                surface=surface,
                res={"success": True, "iota": TEST_IOTA, "G": TEST_G0},
                save=lambda path: None,
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
            "success": False,
            "violations": ["coil_coil_min_dist=0.040000 < threshold=0.050000"],
            "search_hardware_status": {
                "success": False,
                "violations": ["coil_coil_min_dist=0.040000 < threshold=0.050000"],
            },
            "artifact_hardware_status": {
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

    def test_evaluate_search_step_frontier_trust_excess_remains_search_penalty(self):
        module = self.load_module()

        class _Surface:
            def volume(self):
                return 0.10

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = SimpleNamespace(
            boozer_trust_threshold=1.0e-5,
            boozer_trust_penalty_scale=5.0e-5,
        )
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.04
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1.0e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.5
        module.HARDWARE_SEARCH_MODE = "hard"
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = 0
        module.CC_DIST = 0.05
        module.CS_DIST = 0.015
        module.CURVATURE_THRESHOLD = 40.0
        module.bs = object()
        module.JCurveCurve = object()
        module.JCurveSurface = object()
        module.JSurfSurf = None
        module.banana_curve = object()
        module.JF = SimpleNamespace(x=np.zeros(2))
        module.surface_iota_terms = [SimpleNamespace(J=lambda: 0.15)]
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=_Surface())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {"total": 7.0},
            "invalid_state_rejects_total": 0,
            "topology_gate_rejects": 0,
            "hardware_rejects": 0,
            "surface_solve_rejects": 0,
            "frontier_trust_rejects": 0,
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
        objective_eval = {
            "total": 2.09,
            "grad": np.array([1.024, -2.012]),
            "J_Boozer": 2.5e-5,
            "dJ_Boozer": np.array([2.0e-6, -1.0e-6]),
            "frontier_trust_penalty": 0.09,
            "frontier_boozer_trust_excess_ratio": 0.3,
            "surface_weights": np.array([1.0]),
        }

        with patch.object(
            module,
            "solve_surface_stack_at_dofs",
            return_value=stack_status,
        ), patch.object(
            module,
            "evaluate_search_objective",
            return_value=objective_eval,
        ), patch.object(
            module,
            "evaluate_search_topology_gate",
            return_value={"enabled": False, "success": True},
        ), patch.object(
            module,
            "evaluate_single_stage_hardware_snapshot",
            return_value={
                "success": True,
                "violations": [],
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "curve_curve_min_dist": 0.06,
                "curve_surface_min_dist": 0.02,
                "surface_vessel_min_dist": None,
                "max_curvature": 15.0,
            },
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertAlmostEqual(evaluation["total"], 2.09)
        np.testing.assert_allclose(evaluation["grad"], [1.024, -2.012])
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)
        self.assertEqual(module.run_dict["frontier_trust_rejects"], 1)
        self.assertTrue(module.run_dict["topology_gate_status"]["success"])

    def test_evaluate_search_step_frontier_topology_reject_becomes_penalty(self):
        module = self.load_module()

        class _Surface:
            def volume(self):
                return 0.10

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = SimpleNamespace(
            boozer_trust_threshold=1.0e-5,
            boozer_trust_penalty_scale=5.0e-5,
        )
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.04
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1.0e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.5
        module.TOPOLOGY_GATE_PENALTY_SCALE = 4.0
        module.HARDWARE_SEARCH_MODE = "hard"
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = 0
        module.CC_DIST = 0.05
        module.CS_DIST = 0.015
        module.CURVATURE_THRESHOLD = 40.0
        module.bs = object()
        module.JCurveCurve = object()
        module.JCurveSurface = object()
        module.JSurfSurf = None
        module.banana_curve = object()
        module.JF = SimpleNamespace(x=np.zeros(2))
        module.surface_iota_terms = [SimpleNamespace(J=lambda: 0.15)]
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=_Surface())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {"total": 7.0},
            "invalid_state_rejects_total": 0,
            "topology_gate_rejects": 0,
            "hardware_rejects": 0,
            "surface_solve_rejects": 0,
            "frontier_trust_rejects": 0,
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
        objective_eval = {
            "total": 2.0,
            "grad": np.array([1.0, -2.0]),
            "J_Boozer": 5.0e-6,
            "dJ_Boozer": np.array([0.0, 0.0]),
            "surface_weights": np.array([1.0]),
        }

        with patch.object(
            module,
            "solve_surface_stack_at_dofs",
            return_value=stack_status,
        ), patch.object(
            module,
            "evaluate_search_objective",
            return_value=objective_eval,
        ), patch.object(
            module,
            "evaluate_search_topology_gate",
            return_value={
                "enabled": True,
                "success": False,
                "survived_lines": 1,
                "nfieldlines": 4,
                "survival_fraction": 0.5,
                "survival_threshold": 0.75,
                "first_exit_time": None,
                "first_exit_angle": None,
                "first_exit_reason": None,
            },
        ), patch.object(
            module,
            "evaluate_single_stage_hardware_snapshot",
            return_value={
                "success": True,
                "violations": [],
                "search_hardware_status": {"success": True, "violations": []},
                "artifact_hardware_status": {"success": True, "violations": []},
                "curve_curve_min_dist": 0.06,
                "curve_surface_min_dist": 0.02,
                "surface_vessel_min_dist": None,
                "max_curvature": 15.0,
            },
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertAlmostEqual(evaluation["total"], 9.0)
        np.testing.assert_allclose(evaluation["grad"], [1.0, -2.0])
        self.assertAlmostEqual(module.run_dict["last_successful_eval"]["frontier_topology_penalty"], 7.0)
        self.assertAlmostEqual(module.run_dict["last_successful_eval"]["frontier_contract_penalty"], 7.0)
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)
        self.assertEqual(module.run_dict["topology_gate_rejects"], 0)
        self.assertFalse(module.run_dict["topology_gate_status"]["success"])

    def test_evaluate_search_step_frontier_hardware_reject_becomes_penalty(self):
        module = self.load_module()

        class _Surface:
            def volume(self):
                return 0.10

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = SimpleNamespace(
            boozer_trust_threshold=1.0e-5,
            boozer_trust_penalty_scale=5.0e-5,
        )
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.04
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1.0e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.5
        module.HARDWARE_SEARCH_MODE = "hard"
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = 0
        module.HARDWARE_SEARCH_PENALTY_SCALE = 4.0
        module.CC_DIST = 0.05
        module.CS_DIST = 0.015
        module.CURVATURE_THRESHOLD = 40.0
        module.bs = object()
        module.JCurveCurve = object()
        module.JCurveSurface = object()
        module.JSurfSurf = None
        module.banana_curve = object()
        module.JF = SimpleNamespace(x=np.zeros(2))
        module.surface_iota_terms = [SimpleNamespace(J=lambda: 0.15)]
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=_Surface())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {"total": 7.0},
            "invalid_state_rejects_total": 0,
            "topology_gate_rejects": 0,
            "hardware_rejects": 0,
            "surface_solve_rejects": 0,
            "frontier_trust_rejects": 0,
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
        objective_eval = {
            "total": 2.0,
            "grad": np.array([1.0, -2.0]),
            "J_Boozer": 5.0e-6,
            "dJ_Boozer": np.array([0.0, 0.0]),
            "surface_weights": np.array([1.0]),
        }

        with patch.object(
            module,
            "solve_surface_stack_at_dofs",
            return_value=stack_status,
        ), patch.object(
            module,
            "evaluate_search_objective",
            return_value=objective_eval,
        ), patch.object(
            module,
            "evaluate_search_topology_gate",
            return_value={"enabled": False, "success": True},
        ), patch.object(
            module,
            "evaluate_single_stage_hardware_snapshot",
            return_value={
                "success": False,
                "violations": ["coil_coil_min_dist low"],
                "search_hardware_status": {
                    "success": False,
                    "violations": ["coil_coil_min_dist low"],
                    "curve_curve_min_dist": 0.06,
                    "cc_dist": 0.08,
                    "curve_surface_min_dist": 0.02,
                    "cs_dist": 0.015,
                    "surface_vessel_min_dist": None,
                    "ss_dist": None,
                    "max_curvature": 15.0,
                    "curvature_threshold": 40.0,
                },
                "artifact_hardware_status": {
                    "success": False,
                    "violations": ["coil_coil_min_dist low"],
                },
                "curve_curve_min_dist": 0.06,
                "curve_surface_min_dist": 0.02,
                "surface_vessel_min_dist": None,
                "max_curvature": 15.0,
            },
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertAlmostEqual(evaluation["total"], 9.0)
        np.testing.assert_allclose(evaluation["grad"], [1.0, -2.0])
        self.assertAlmostEqual(module.run_dict["last_successful_eval"]["frontier_hardware_penalty"], 7.0)
        self.assertAlmostEqual(module.run_dict["last_successful_eval"]["frontier_contract_penalty"], 7.0)
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)
        self.assertEqual(module.run_dict["hardware_rejects"], 0)
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])

    def test_evaluate_search_step_repair_phase1_keeps_valid_hardware_bad_candidate_live(self):
        module = self.load_module()

        class _Surface:
            def volume(self):
                return 0.10

        module.SINGLE_STAGE_GOAL_MODE = "target"
        module.MULTISURFACE_RAMP_ITERATIONS = 0
        module.INNER_SURFACE_INITIAL_WEIGHT = 1.0
        module.SURFACE_GAP_THRESHOLD = 0.0
        module.SS_DIST = 0.04
        module.TOPOLOGY_GATE_TMAX = 2.0
        module.TOPOLOGY_GATE_TOL = 1.0e-3
        module.TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.5
        module.HARDWARE_SEARCH_MODE = "hard"
        module.HARDWARE_SEARCH_SOFT_ITERATIONS = 0
        module.CC_DIST = 0.05
        module.CS_DIST = 0.015
        module.CURVATURE_THRESHOLD = 40.0
        module.bs = object()
        module.JCurveCurve = object()
        module.JCurveSurface = object()
        module.JSurfSurf = None
        module.banana_curve = object()
        module.JF = SimpleNamespace(x=np.zeros(2))
        module.surface_iota_terms = [SimpleNamespace(J=lambda: 0.15)]
        module.surface_data = [{"boozer_surface": SimpleNamespace(surface=_Surface())}]
        module.run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 0,
            "accepted_iterations": 0,
            "surface_state": {"sdofs": [], "iota": [], "G": []},
            "accepted_x": np.zeros(2),
            "J": 7.0,
            "dJ": np.array([3.0, -1.0]),
            "search_eval": {"total": 7.0},
            "invalid_state_rejects_total": 0,
            "topology_gate_rejects": 0,
            "hardware_rejects": 0,
            "surface_solve_rejects": 0,
            "frontier_trust_rejects": 0,
            "phase1_repair_mode_active": True,
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
        objective_eval = {
            "total": 2.0,
            "grad": np.array([1.0, -2.0]),
            "J_cc": 1.5,
            "dJ_cc": np.array([0.5, -0.5]),
            "J_cs": 0.5,
            "dJ_cs": np.array([0.25, -0.25]),
            "J_curvature": 0.25,
            "dJ_curvature": np.array([0.1, -0.1]),
            "J_surf": 0.0,
            "dJ_surf": np.zeros(2),
            "surface_weights": np.array([1.0]),
        }

        with patch.object(
            module,
            "solve_surface_stack_at_dofs",
            return_value=stack_status,
        ), patch.object(
            module,
            "evaluate_search_objective",
            return_value=objective_eval,
        ), patch.object(
            module,
            "evaluate_search_topology_gate",
            return_value={"enabled": False, "success": True},
        ), patch.object(
            module,
            "evaluate_single_stage_hardware_snapshot",
            return_value={
                "success": False,
                "violations": ["coil_coil_min_dist low"],
                "search_hardware_status": {
                    "success": False,
                    "violations": ["coil_coil_min_dist low"],
                    "curve_curve_min_dist": 0.06,
                    "cc_dist": 0.08,
                    "curve_surface_min_dist": 0.02,
                    "cs_dist": 0.015,
                    "surface_vessel_min_dist": None,
                    "ss_dist": None,
                    "max_curvature": 15.0,
                    "curvature_threshold": 40.0,
                },
                "artifact_hardware_status": {
                    "success": False,
                    "violations": ["coil_coil_min_dist low"],
                },
                "curve_curve_min_dist": 0.06,
                "curve_surface_min_dist": 0.02,
                "surface_vessel_min_dist": None,
                "max_curvature": 15.0,
            },
        ):
            evaluation = module.evaluate_search_step(np.ones(2))

        self.assertAlmostEqual(evaluation["total"], 2.0)
        np.testing.assert_allclose(evaluation["grad"], [1.0, -2.0])
        self.assertEqual(module.run_dict["invalid_state_rejects_total"], 0)
        self.assertEqual(module.run_dict["hardware_rejects"], 0)
        self.assertFalse(module.run_dict["trial_hardware_status"]["success"])
        self.assertEqual(module.run_dict["last_successful_eval"], objective_eval)

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

    def test_build_scipy_bounds_returns_none_when_unbounded(self):
        module = self.load_module()

        bounds = module.build_scipy_bounds(
            np.array([-np.inf, -np.inf]),
            np.array([np.inf, np.inf]),
        )

        self.assertIsNone(bounds)

    def test_build_scaled_outer_bounds_transforms_to_scaled_coordinates(self):
        module = self.load_module()

        bounds = module.build_scaled_outer_bounds(
            np.array([10.0, 20.0]),
            0.1,
            np.array([9.0, -np.inf]),
            np.array([10.5, 25.0]),
        )

        self.assertEqual(bounds, [(-10.0, 5.0), (-np.inf, 50.0)])

    def test_build_local_relative_bounds_clips_to_anchor_box_and_global_bounds(self):
        module = self.load_module()

        bounds = module.build_local_relative_bounds(
            np.array([10.0, -2.0]),
            0.1,
            np.array([9.5, -10.0]),
            np.array([12.0, -1.5]),
        )

        self.assertEqual(bounds, [(9.5, 11.0), (-2.2, -1.8)])

    def test_build_scaled_local_outer_bounds_transforms_local_box(self):
        module = self.load_module()

        bounds = module.build_scaled_local_outer_bounds(
            np.array([10.0, 20.0]),
            0.5,
            np.array([8.0, 18.0]),
            np.array([15.0, 30.0]),
            0.1,
        )

        self.assertEqual(bounds, [(-2.0, 2.0), (-4.0, 4.0)])

    def test_resolve_initial_step_phase_maxiter(self):
        module = self.load_module()

        self.assertEqual(module.resolve_initial_step_phase_maxiter(40, 1.0, 10), 0)
        self.assertEqual(module.resolve_initial_step_phase_maxiter(40, 0.5, 0), 0)
        self.assertEqual(module.resolve_initial_step_phase_maxiter(40, 0.5, 10), 10)
        self.assertEqual(module.resolve_initial_step_phase_maxiter(5, 0.5, 10), 5)

    def test_penalty_feasible_start_local_preservation_enabled(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_hardware_status": {"success": True},
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
        }

        self.assertTrue(
            module.penalty_feasible_start_local_preservation_enabled(
                run_dict,
                constraint_method="penalty",
                num_surfaces=1,
                basin_hops=0,
                init_only=False,
            )
        )
        self.assertFalse(
            module.penalty_feasible_start_local_preservation_enabled(
                run_dict,
                constraint_method="penalty",
                num_surfaces=2,
                basin_hops=0,
                init_only=False,
            )
        )

    def test_resolve_single_stage_seed_regime_auto_routes_by_incumbent_state(self):
        module = self.load_module()
        good_run_dict = {
            "accepted_hardware_status": {"success": True},
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
        }
        bridge_run_dict = {
            "accepted_hardware_status": {"success": False},
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
        }
        bad_run_dict = {
            "accepted_hardware_status": {"success": False},
            "surface_status": {"success": False},
            "intersecting": True,
            "search_eval": {"total": 1.0},
        }

        self.assertEqual(
            module.resolve_single_stage_seed_regime(
                "auto",
                good_run_dict,
                constraint_method="penalty",
                num_surfaces=1,
                basin_hops=0,
                init_only=False,
            ),
            "preserve_first",
        )
        self.assertEqual(
            module.resolve_single_stage_seed_regime(
                "auto",
                bridge_run_dict,
                constraint_method="penalty",
                num_surfaces=1,
                basin_hops=0,
                init_only=False,
            ),
            "bridge_only",
        )
        self.assertEqual(
            module.resolve_single_stage_seed_regime(
                "auto",
                bad_run_dict,
                constraint_method="penalty",
                num_surfaces=1,
                basin_hops=0,
                init_only=False,
            ),
            "repair_first",
        )
        self.assertEqual(
            module.resolve_single_stage_seed_regime(
                "bridge_only",
                good_run_dict,
                constraint_method="alm",
                num_surfaces=1,
                basin_hops=0,
                init_only=False,
            ),
            "global_search",
        )

    def test_resolve_penalty_phase1_settings_auto_enables_local_preservation(self):
        module = self.load_module()

        settings = module.resolve_penalty_phase1_settings(
            40,
            1.0,
            0,
            enable_local_preservation=True,
        )

        self.assertTrue(settings["use_phase1"])
        self.assertTrue(settings["auto_enabled"])
        self.assertTrue(settings["use_local_bounds"])
        self.assertEqual(
            settings["phase1_maxiter"],
            min(40, module._PENALTY_FEASIBLE_START_LOCAL_MAXITER),
        )
        self.assertEqual(settings["phase1_scale"], 1.0)
        self.assertEqual(
            settings["local_relative_radius"],
            module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
        )

    def test_resolve_penalty_phase1_settings_frontier_auto_contracts_phase1(self):
        module = self.load_module()

        settings = module.resolve_penalty_phase1_settings(
            40,
            1.0,
            0,
            enable_local_preservation=True,
            is_frontier_mode=True,
        )

        self.assertTrue(settings["use_phase1"])
        self.assertTrue(settings["auto_enabled"])
        self.assertTrue(settings["use_local_bounds"])
        self.assertEqual(
            settings["phase1_maxiter"],
            min(40, module._PENALTY_FEASIBLE_START_LOCAL_MAXITER),
        )
        self.assertEqual(
            settings["phase1_scale"],
            module._FRONTIER_FEASIBLE_START_PHASE1_SCALE,
        )
        self.assertEqual(
            settings["local_relative_radius"],
            module._FRONTIER_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
        )

    def test_resolve_penalty_phase1_settings_frontier_does_not_auto_contract_when_local_preservation_disabled(self):
        module = self.load_module()

        settings = module.resolve_penalty_phase1_settings(
            40,
            1.0,
            0,
            enable_local_preservation=False,
            is_frontier_mode=True,
        )

        self.assertFalse(settings["use_phase1"])
        self.assertFalse(settings["auto_enabled"])
        self.assertFalse(settings["use_local_bounds"])
        self.assertEqual(settings["phase1_scale"], 1.0)
        self.assertIsNone(settings["local_relative_radius"])

    def test_resolve_penalty_phase1_settings_frontier_respects_explicit_initial_step_scale(self):
        module = self.load_module()

        settings = module.resolve_penalty_phase1_settings(
            40,
            0.5,
            5,
            enable_local_preservation=True,
            is_frontier_mode=True,
        )

        self.assertTrue(settings["use_phase1"])
        self.assertFalse(settings["auto_enabled"])
        self.assertTrue(settings["use_local_bounds"])
        self.assertEqual(settings["phase1_maxiter"], 5)
        self.assertEqual(settings["phase1_scale"], 0.5)
        self.assertEqual(
            settings["local_relative_radius"],
            module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
        )

    def test_resolve_penalty_phase1_settings_repair_first_uses_extra_local_attempt(self):
        module = self.load_module()

        settings = module.resolve_penalty_phase1_settings(
            40,
            1.0,
            0,
            enable_local_preservation=True,
            seed_regime="repair_first",
        )

        self.assertTrue(settings["use_phase1"])
        self.assertTrue(settings["use_local_bounds"])
        self.assertEqual(
            settings["local_max_attempts"],
            module._PENALTY_FEASIBLE_START_LOCAL_MAX_ATTEMPTS + 1,
        )

    def test_run_penalty_phase1_preserves_feasible_start_when_no_safe_step_exists(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }
        restore_calls = []

        def fake_restore():
            restore_calls.append(True)

        def fake_minimize(*args, **kwargs):
            return SimpleNamespace(nit=2, success=False, message="ABNORMAL_TERMINATION", status=2)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=lambda x: None,
            normalize_message_fn=lambda *args, **kwargs: "phase1_reject",
            restore_accepted_state_fn=fake_restore,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertTrue(result["used_phase1"])
        self.assertFalse(result["continue_search"])
        self.assertTrue(result["local_preservation_used"])
        self.assertTrue(result["local_preservation_preserved_start"])
        self.assertGreaterEqual(result["local_preservation_attempts"], 1)
        self.assertEqual(result["phase1_outcome"], "preserved_start_no_safe_step")
        self.assertIsNone(result["phase1_first_accepted_step_rms"])
        self.assertIsNone(result["phase1_max_accepted_step_rms"])
        self.assertFalse(result["phase1_anchor_restore_used"])
        self.assertEqual(result["phase1_unsafe_accept_rollbacks"], 0)
        self.assertEqual(result["phase1_invalid_reject_attempts"], 0)
        self.assertFalse(result["phase1_recovery_used"])
        self.assertEqual(result["next_dofs"].tolist(), [1.0, -1.0])
        self.assertGreaterEqual(len(restore_calls), 1)

    def test_run_penalty_phase1_continues_after_local_acceptance(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.01, -0.99]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertTrue(result["continue_search"])
        self.assertFalse(result["local_preservation_preserved_start"])
        self.assertEqual(result["phase1_outcome"], "safe_local_accept")
        self.assertFalse(result["phase1_anchor_restore_used"])
        self.assertEqual(result["phase1_unsafe_accept_rollbacks"], 0)
        self.assertEqual(result["phase1_invalid_reject_attempts"], 0)
        self.assertFalse(result["phase1_recovery_used"])
        np.testing.assert_allclose(result["next_dofs"], [1.01, -0.99])
        expected_step_rms = module.basin_normalized_step_rms(
            np.array([1.0, -1.0]),
            np.array([1.01, -0.99]),
        )
        self.assertAlmostEqual(
            result["phase1_first_accepted_step_rms"],
            expected_step_rms,
        )
        self.assertAlmostEqual(
            result["phase1_max_accepted_step_rms"],
            expected_step_rms,
        )
        self.assertLessEqual(
            result["local_preservation_radius"],
            module._PENALTY_FEASIBLE_START_SAFE_STEP_RMS_LIMIT
            * module._PENALTY_FEASIBLE_START_PHASE2_RADIUS_SCALE,
        )

    def test_run_penalty_phase1_zero_move_accept_does_not_graduate(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.0, -1.0]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(
                module,
                phase1_config=module.build_phase1_config(local_max_attempts=1),
            ),
        )

        self.assertFalse(result["continue_search"])
        self.assertTrue(result["local_preservation_preserved_start"])
        self.assertEqual(result["phase1_outcome"], "preserved_start_no_safe_step")
        self.assertTrue(result["phase1_anchor_restore_used"])
        self.assertEqual(result["phase1_unsafe_accept_rollbacks"], 1)
        self.assertEqual(result["phase1_invalid_reject_attempts"], 0)
        self.assertTrue(result["phase1_recovery_used"])
        self.assertEqual(result["phase1_first_accepted_step_rms"], 0.0)
        self.assertEqual(result["phase1_max_accepted_step_rms"], 0.0)
        self.assertIn("unsafe_local_accept", result["phase1_termination_message"])

    def test_run_penalty_phase1_rolls_back_unsafe_accept_and_uses_reject_shrink(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }
        seen_bounds = []
        attempts = {"count": 0}
        refresh_calls = []

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            if attempts["count"] == 1:
                run_dict["accepted_hardware_status"] = {"success": False}
                run_dict["invalid_state_rejects_total"] += 1
            else:
                run_dict["accepted_hardware_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            attempts["count"] += 1
            seen_bounds.append(kwargs["bounds"])
            if attempts["count"] == 1:
                kwargs["callback"](np.array([1.04, -0.96]))
            else:
                kwargs["callback"](np.array([1.01, -0.99]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            refresh_preserved_timeout_artifacts_fn=lambda: refresh_calls.append("refresh"),
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(refresh_calls, ["refresh"])
        self.assertTrue(result["continue_search"])
        self.assertFalse(result["local_preservation_preserved_start"])
        self.assertEqual(result["phase1_outcome"], "safe_local_accept_after_recovery")
        self.assertTrue(result["phase1_anchor_restore_used"])
        self.assertEqual(result["phase1_unsafe_accept_rollbacks"], 1)
        self.assertEqual(result["phase1_invalid_reject_attempts"], 1)
        self.assertTrue(result["phase1_recovery_used"])
        np.testing.assert_allclose(result["next_dofs"], [1.01, -0.99])
        first_step_rms = module.basin_normalized_step_rms(
            np.array([1.0, -1.0]),
            np.array([1.04, -0.96]),
        )
        second_step_rms = module.basin_normalized_step_rms(
            np.array([1.0, -1.0]),
            np.array([1.01, -0.99]),
        )
        self.assertAlmostEqual(
            result["phase1_first_accepted_step_rms"],
            first_step_rms,
        )
        self.assertAlmostEqual(
            result["phase1_max_accepted_step_rms"],
            max(first_step_rms, second_step_rms),
        )
        self.assertEqual(seen_bounds[0], [(0.95, 1.05), (-1.05, -0.95)])
        self.assertEqual(
            seen_bounds[1],
            [
                (
                    1.0 - module._PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK
                    * module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
                    1.0 + module._PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK
                    * module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
                ),
                (
                    -1.0 - module._PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK
                    * module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
                    -1.0 + module._PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK
                    * module._PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
                ),
            ],
        )

    def test_run_penalty_phase1_tracks_first_and_max_accepted_step_across_callbacks(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": True},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.015, -0.985]))
            kwargs["callback"](np.array([1.01, -0.99]))
            return SimpleNamespace(nit=2, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertEqual(result["phase1_outcome"], "safe_local_accept")
        self.assertAlmostEqual(result["phase1_first_accepted_step_rms"], 0.015)
        self.assertAlmostEqual(result["phase1_max_accepted_step_rms"], 0.015)
        np.testing.assert_allclose(result["next_dofs"], [1.01, -0.99])

    def test_run_penalty_phase1_repair_first_accepts_local_violation_reduction_not_preserve_gate(self):
        module = self.load_module()
        anchor_hardware = {
            "success": False,
            "violations": [
                "coil_coil_min_dist 0.030000 below threshold 0.050000",
                "max_curvature 120.000000 exceeds threshold 100.000000",
            ],
            "curve_curve_min_dist": 0.03,
            "cc_dist": 0.05,
            "curve_surface_min_dist": 0.02,
            "cs_dist": 0.015,
            "surface_vessel_min_dist": 0.05,
            "ss_dist": 0.04,
            "max_curvature": 120.0,
            "curvature_threshold": 100.0,
        }
        repaired_hardware = {
            "success": False,
            "violations": [
                "coil_coil_min_dist 0.040000 below threshold 0.050000",
                "max_curvature 110.000000 exceeds threshold 100.000000",
            ],
            "curve_curve_min_dist": 0.04,
            "cc_dist": 0.05,
            "curve_surface_min_dist": 0.02,
            "cs_dist": 0.015,
            "surface_vessel_min_dist": 0.05,
            "ss_dist": 0.04,
            "max_curvature": 110.0,
            "curvature_threshold": 100.0,
        }

        repair_run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": dict(anchor_hardware),
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }
        preserve_run_dict = copy.deepcopy(repair_run_dict)

        def make_callback(run_dict):
            def fake_callback(x):
                run_dict["accepted_iterations"] += 1
                run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
                run_dict["accepted_hardware_status"] = dict(repaired_hardware)
                run_dict["surface_status"] = {"success": True}

            return fake_callback

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.03, -0.97]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        repair_result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=False,
            seed_regime="repair_first",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=repair_run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=make_callback(repair_run_dict),
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        preserve_result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=True,
            seed_regime="preserve_first",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=preserve_run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=make_callback(preserve_run_dict),
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertTrue(repair_result["continue_search"])
        self.assertEqual(repair_result["phase1_outcome"], "repair_local_recovery")
        self.assertEqual(repair_result["startup_local_phase_regime"], "repair_first")
        self.assertTrue(repair_result["startup_local_recovery_achieved"])
        self.assertFalse(repair_result["bridge_local_donor_ready"])
        self.assertFalse(repair_result["local_preservation_preserved_start"])
        self.assertAlmostEqual(repair_result["local_preservation_radius"], 0.015)
        self.assertFalse(preserve_result["continue_search"])
        self.assertEqual(preserve_result["phase1_outcome"], "preserved_start_no_safe_step")
        self.assertFalse(preserve_result["startup_local_recovery_achieved"])
        self.assertFalse(preserve_result["bridge_local_donor_ready"])

    def test_run_penalty_phase1_repair_first_uses_repair_objective_not_generic_total(self):
        module = self.load_module()
        module.CC_WEIGHT = 101.0
        module.CS_WEIGHT = 103.0
        module.CURVATURE_WEIGHT = 107.0
        module.SURF_DIST_WEIGHT = 109.0
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": True},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": False, "violations": ["coil_coil"]},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }
        captured = {}

        def fake_objective_eval(x):
            captured["repair_mode_during_eval"] = run_dict.get("phase1_repair_mode_active")
            return {
                "total": 999.0,
                "grad": np.array([9.0, 9.0]),
                "J_cc": 2.0,
                "dJ_cc": np.array([1.0, 0.0]),
                "J_cs": 3.0,
                "dJ_cs": np.array([0.0, 1.0]),
                "J_curvature": 4.0,
                "dJ_curvature": np.array([-1.0, 2.0]),
                "J_surf": 5.0,
                "dJ_surf": np.array([3.0, -2.0]),
            }

        def fake_minimize(fun, x0, **kwargs):
            total, grad = fun(x0)
            captured["total"] = total
            captured["grad"] = grad
            return SimpleNamespace(nit=1, success=False, message="ABNORMAL_TERMINATION", status=2)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=1,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=1,
            enable_local_preservation=False,
            seed_regime="repair_first",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (111.0, np.array([7.0, 7.0])),
            callback_fn=lambda x: None,
            objective_eval_fn=fake_objective_eval,
            normalize_message_fn=lambda *args, **kwargs: "phase1_stop",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(
                module,
                phase1_config=module.build_phase1_config(
                    cc_weight=2.0,
                    cs_weight=3.0,
                    curvature_weight=5.0,
                    surf_dist_weight=7.0,
                ),
            ),
        )

        self.assertEqual(
            captured["total"],
            2.0 * 2.0 + 3.0 * 3.0 + 5.0 * 4.0 + 7.0 * 5.0,
        )
        np.testing.assert_allclose(
            captured["grad"],
            2.0 * np.array([1.0, 0.0])
            + 3.0 * np.array([0.0, 1.0])
            + 5.0 * np.array([-1.0, 2.0])
            + 7.0 * np.array([3.0, -2.0]),
        )
        self.assertTrue(captured["repair_mode_during_eval"])
        self.assertFalse(run_dict["phase1_repair_mode_active"])
        self.assertEqual(result["phase1_outcome"], "repair_first_no_local_recovery")

    def test_run_penalty_phase1_bridge_only_requires_safe_step_for_donor_ready(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": False},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": False},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

        def fake_callback(x):
            run_dict["accepted_iterations"] += 1
            run_dict["accepted_x"] = np.asarray(x, dtype=float).copy()
            run_dict["accepted_hardware_status"] = {"success": True}
            run_dict["surface_status"] = {"success": True}

        def fake_minimize(fun, x0, **kwargs):
            kwargs["callback"](np.array([1.01, -0.99]))
            return SimpleNamespace(nit=1, success=True, message="CONVERGENCE", status=0)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=False,
            seed_regime="bridge_only",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=fake_callback,
            normalize_message_fn=lambda *args, **kwargs: "phase1_ok",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertTrue(result["continue_search"])
        self.assertEqual(result["phase1_outcome"], "bridge_local_donor_ready")
        self.assertEqual(result["startup_local_phase_regime"], "bridge_only")
        self.assertTrue(result["startup_local_recovery_achieved"])
        self.assertTrue(result["bridge_local_donor_ready"])

    def test_run_penalty_phase1_bridge_only_stops_without_preserved_closeout(self):
        module = self.load_module()
        run_dict = {
            "accepted_iterations": 0,
            "accepted_x": np.array([1.0, -1.0]),
            "invalid_state_rejects_total": 0,
            "surface_solve_rejects": 0,
            "hardware_rejects": 0,
            "topology_gate_rejects": 0,
            "surface_status": {"success": False},
            "intersecting": False,
            "search_eval": {"total": 1.0},
            "accepted_hardware_status": {"success": False},
            "surface_state": {"seed": "anchor"},
            "J": 1.0,
            "dJ": np.zeros(2),
            "search_surface_status": {"success": True},
            "topology_gate_status": {"enabled": False},
            "x_prev": np.array([1.0, -1.0]),
            "best_accepted_incumbent": None,
            "best_accepted_metric": None,
            "best_accepted_stage": None,
            "best_feasible_incumbent": None,
            "best_feasible_metric": None,
            "best_feasible_stage": None,
            "it": 0,
        }

        def fake_minimize(*args, **kwargs):
            return SimpleNamespace(nit=1, success=False, message="ABNORMAL_TERMINATION", status=2)

        result = module.run_penalty_phase1(
            np.array([1.0, -1.0]),
            total_maxiter=4,
            maxcor=5,
            ftol=1e-15,
            gtol=1e-15,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            enable_local_preservation=False,
            seed_regime="bridge_only",
            lower_bounds=np.array([-5.0, -5.0]),
            upper_bounds=np.array([5.0, 5.0]),
            run_dict=run_dict,
            objective_fn=lambda x: (0.0, np.zeros_like(x)),
            callback_fn=lambda x: None,
            normalize_message_fn=lambda *args, **kwargs: "phase1_reject",
            restore_accepted_state_fn=lambda: None,
            minimize_fn=fake_minimize,
            **phase1_runtime_kwargs(module),
        )

        self.assertFalse(result["continue_search"])
        self.assertEqual(result["phase1_outcome"], "bridge_only_no_local_donor")
        self.assertFalse(result["local_preservation_preserved_start"])
        self.assertEqual(result["startup_local_phase_regime"], "bridge_only")
        self.assertFalse(result["bridge_local_donor_ready"])

    def test_build_penalty_phase2_bounds_keeps_local_preservation_radius(self):
        module = self.load_module()

        bounds = module.build_penalty_phase2_bounds(
            np.array([2.0, -4.0]),
            lower_bounds=np.array([-10.0, -10.0]),
            upper_bounds=np.array([10.0, 10.0]),
            phase1_result={
                "local_preservation_used": True,
                "local_preservation_preserved_start": False,
                "local_preservation_radius": 0.05,
            },
        )

        self.assertEqual(bounds, [(1.9, 2.1), (-4.2, -3.8)])

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

    def test_build_total_objective_skips_missing_volume_and_surface_vessel_terms(self):
        module = self.load_module()

        total = module.build_total_objective(
            FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            None,
            8.0,
            FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            14.0,
            FakeAlgebraicObjective(15.0, [2.0, -2.0]),
            SURF_DIST_WEIGHT=1000.0,
            JSurfSurf=None,
        )

        self.assertAlmostEqual(
            total.J(),
            1 + 2 * 3 + 4 * 5 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15,
        )
        np.testing.assert_allclose(total.dJ(), [49.0, 0.0])

    def test_build_total_objective_includes_volume_term_when_present(self):
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
            14.0,
            FakeAlgebraicObjective(15.0, [2.0, -2.0]),
            SURF_DIST_WEIGHT=0.0,
            JSurfSurf=None,
        )

        self.assertAlmostEqual(
            total.J(),
            1 + 2 * 3 + 4 * 5 + 6 * 7 + 8 * 9 + 10 * 11 + 12 * 13 + 14 * 15,
        )
        np.testing.assert_allclose(total.dJ(), [61.0, 0.0])

    def test_build_single_stage_iota_objective_target_mode_uses_quadratic_penalty(self):
        module = self.load_module()
        surface_iota = FakeAlgebraicObjective(0.15, [1.0, -2.0])
        target_objective = object()

        with patch.object(module, "QuadraticPenalty", return_value=target_objective) as quadratic_penalty:
            result = module.build_single_stage_iota_objective(
                surface_iota,
                0.17,
                goal_mode="target",
            )

        self.assertIs(result, target_objective)
        quadratic_penalty.assert_called_once_with(surface_iota, 0.17)

    def test_build_single_stage_iota_objective_frontier_mode_uses_bounded_reward(self):
        module = self.load_module()
        surface_iota = FakeAlgebraicObjective(0.15, [1.0, -2.0])
        frontier_goal_config = make_frontier_goal_config(module)

        result = module.build_single_stage_iota_objective(
            surface_iota,
            0.17,
            goal_mode="frontier",
            frontier_goal_config=frontier_goal_config,
        )

        self.assertAlmostEqual(result.J(), -np.tanh(1.0))
        np.testing.assert_allclose(result.dJ(), [-8.39948683, 16.79897366])

    def test_build_single_stage_volume_objective_frontier_mode_rewards_larger_volume(self):
        module = self.load_module()
        surface_volume = FakeAlgebraicObjective(0.12, [2.0, -1.0])
        frontier_goal_config = make_frontier_goal_config(module)

        result = module.build_single_stage_volume_objective(
            surface_volume,
            goal_mode="frontier",
            frontier_goal_config=frontier_goal_config,
        )

        self.assertAlmostEqual(result.J(), -np.tanh(2.0))
        np.testing.assert_allclose(result.dJ(), [-14.13016434, 7.06508217])

    def test_bounded_improvement_reward_partials_wrap_callable_optimizable_gradients(self):
        module = self.load_module()

        class DummyMetricObjective(Optimizable):
            def __init__(self):
                super().__init__(x0=np.array([0.0, 0.0]))

            def J(self):
                return 0.12

            def dJ(self, partials=False):
                if partials:
                    return lambda _objective: np.array([2.0, -1.0])
                return np.array([2.0, -1.0])

        metric_objective = DummyMetricObjective()
        reward = module.BoundedImprovementReward(metric_objective, reference=0.10, scale=0.01)

        partial_gradient = reward.dJ(partials=True)

        self.assertIsInstance(partial_gradient, Derivative)
        np.testing.assert_allclose(partial_gradient(metric_objective), [-14.13016497, 7.06508249])

    def test_build_frontier_goal_config_derives_normalized_weights_and_trust_threshold(self):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
        )

        self.assertAlmostEqual(config.iota_reference, 0.15)
        self.assertAlmostEqual(config.iota_scale, 0.05)
        self.assertAlmostEqual(config.volume_reference, 0.10)
        self.assertAlmostEqual(config.volume_scale, 0.01)
        self.assertAlmostEqual(config.qs_reference, 2.0e-4)
        self.assertAlmostEqual(config.boozer_reference, 1.0e-6)
        self.assertAlmostEqual(config.boozer_trust_threshold, 1.0e-5)
        self.assertAlmostEqual(config.boozer_trust_penalty_scale, 5.0e-5)
        self.assertAlmostEqual(config.effective_boozer_weight, 1.0)
        self.assertAlmostEqual(config.effective_iota_weight, 1.0)
        self.assertAlmostEqual(config.effective_volume_weight, 1.0)

    def test_build_frontier_goal_config_volume_weight_independent_of_iota(self):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=500.0,
            volume_weight=200.0,
        )

        self.assertAlmostEqual(config.effective_iota_weight, 5.0)
        self.assertAlmostEqual(config.effective_volume_weight, 2.0)

    def test_build_frontier_goal_config_volume_weight_defaults_to_iotas_weight(self):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=300.0,
        )

        self.assertAlmostEqual(config.effective_iota_weight, 3.0)
        self.assertAlmostEqual(config.effective_volume_weight, 3.0)

    def test_build_frontier_goal_config_applies_reference_and_trust_overrides(self):
        module = self.load_module()

        config = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
            volume_weight=150.0,
            iota_reference_override=0.17,
            iota_scale_override=0.02,
            volume_reference_override=0.105,
            volume_scale_override=0.015,
            qs_reference_override=0.011,
            boozer_reference_override=0.007,
            boozer_trust_threshold_override=0.009,
            boozer_trust_penalty_scale_override=0.045,
        )

        self.assertAlmostEqual(config.iota_reference, 0.17)
        self.assertAlmostEqual(config.iota_scale, 0.02)
        self.assertAlmostEqual(config.volume_reference, 0.105)
        self.assertAlmostEqual(config.volume_scale, 0.015)
        self.assertAlmostEqual(config.qs_reference, 0.011)
        self.assertAlmostEqual(config.boozer_reference, 0.007)
        self.assertAlmostEqual(config.boozer_trust_threshold, 0.009)
        self.assertAlmostEqual(config.boozer_trust_penalty_scale, 0.045)
        self.assertAlmostEqual(config.effective_iota_weight, 1.0)
        self.assertAlmostEqual(config.effective_volume_weight, 1.5)

    def test_annotate_frontier_search_eval_adds_threshold_relative_trust_penalty(self):
        module = self.load_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = module.build_frontier_goal_config(
            initial_iota=0.15,
            initial_volume=0.10,
            initial_qs_objective=2.0e-4,
            initial_boozer_objective=6.0e-7,
            res_weight=1000.0,
            iotas_weight=100.0,
        )
        search_eval = {
            "total": 2.0,
            "grad": np.array([1.0, -2.0]),
            "J_Boozer": 2.5e-5,
            "dJ_Boozer": np.array([2.0e-6, -1.0e-6]),
        }

        annotated = module.annotate_frontier_search_eval(search_eval)

        self.assertEqual(annotated["frontier_base_total"], 2.0)
        self.assertFalse(annotated["frontier_trust_ok"])
        self.assertAlmostEqual(annotated["frontier_boozer_trust_threshold"], 1.0e-5)
        self.assertAlmostEqual(annotated["frontier_boozer_trust_penalty_scale"], 5.0e-5)
        self.assertAlmostEqual(annotated["frontier_boozer_trust_excess"], 1.5e-5)
        self.assertAlmostEqual(annotated["frontier_boozer_trust_excess_ratio"], 0.3)
        self.assertAlmostEqual(annotated["frontier_trust_penalty"], 0.09)
        self.assertAlmostEqual(annotated["frontier_rank_total"], 2.09)
        self.assertAlmostEqual(annotated["total"], 2.09)
        self.assertTrue(annotated["finite_eval_ok"])
        np.testing.assert_allclose(
            annotated["grad"],
            np.array([1.024, -2.012]),
        )

    def test_build_single_stage_iota_objective_rejects_invalid_goal_mode(self):
        module = self.load_module()
        surface_iota = FakeAlgebraicObjective(0.15, [1.0, -2.0])

        with self.assertRaisesRegex(ValueError, "Unsupported single-stage goal mode"):
            module.build_single_stage_iota_objective(
                surface_iota,
                0.17,
                goal_mode="not-a-mode",
            )

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

            def save(self, path):
                self._saved_path = path

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

            def save(self, path):
                self._saved_path = path

        inner_cs = [[0.85, 0.0, -0.1], [1.05, 0.0, -0.1], [1.05, 0.0, 0.1], [0.85, 0.0, 0.1]]
        outer_cs = [[0.7, 0.0, -0.3], [1.3, 0.0, -0.3], [1.3, 0.0, 0.3], [0.7, 0.0, 0.3]]
        inner = SimpleNamespace(
            surface=_Surface(0.08, [0.0, 0.0, 0.0], inner_cs),
            res={"success": True, "iota": 0.12, "G": 1.0},
            save=lambda path: None,
        )
        outer = SimpleNamespace(
            surface=_Surface(0.10, [0.4, 0.0, 0.0], outer_cs),
            res={"success": True, "iota": 0.15, "G": 1.1},
            save=lambda path: None,
        )

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


def _make_unbounded_scaled_current():
    leaf_current = SimpleNamespace(
        local_lower_bounds=np.array([-np.inf], dtype=float),
        local_upper_bounds=np.array([np.inf], dtype=float),
    )
    return leaf_current, SimpleNamespace(current_to_scale=leaf_current, scale=1.0)


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
            single_stage_goal_mode="target",
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
            alm_penalty_max=1.0e8,
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

    def test_run_identity_omits_default_target_goal_mode(self):
        module = load_single_stage_example_module()
        args = self._make_identity_args()
        config = self._make_identity_config(module, args)

        self.assertIsNone(config.single_stage_goal_mode)

    def test_run_identity_distinguishes_explicit_preserve_first_from_auto(self):
        module = load_single_stage_example_module()
        auto_args = self._make_identity_args()
        auto_args.seed_regime = "auto"
        preserve_args = self._make_identity_args()
        preserve_args.seed_regime = "preserve_first"

        auto_config = self._make_identity_config(module, auto_args)
        preserve_config = self._make_identity_config(module, preserve_args)

        self.assertIsNone(auto_config.seed_regime)
        self.assertEqual(preserve_config.seed_regime, "preserve_first")
        self.assertNotEqual(
            module.build_run_identity_config(auto_config),
            module.build_run_identity_config(preserve_config),
        )

    def test_run_identity_changes_when_only_confinement_settings_change(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()
        weighted_args = self._make_identity_args()
        weighted_args.confinement_objective_weight = 5.0

        base_config = self._build_identity(module, base_args)
        weighted_config = self._build_identity(module, weighted_args)

        self.assertNotEqual(base_config, weighted_config)

    def test_run_identity_changes_when_goal_mode_changes(self):
        module = load_single_stage_example_module()
        target_args = self._make_identity_args()
        frontier_args = self._make_identity_args()
        frontier_args.single_stage_goal_mode = "frontier"

        target_config = self._build_identity(module, target_args)
        frontier_config = self._build_identity(module, frontier_args)

        self.assertNotEqual(target_config, frontier_config)

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
        physical_config = self._build_identity(
            module,
            base_args,
            boozer_I=4.0e-7 * np.pi * 8000.0,
            plasma_current_A=8000.0,
        )

        self.assertNotEqual(base_config, physical_config)

    def test_run_identity_ignores_plasma_current_input_source_when_realized_current_matches(self):
        module = load_single_stage_example_module()
        base_args = self._make_identity_args()

        physical_config = self._build_identity(
            module,
            base_args,
            boozer_I=4.0e-7 * np.pi * 8000.0,
            plasma_current_A=8000.0,
        )
        raw_config = self._build_identity(
            module,
            base_args,
            boozer_I=4.0e-7 * np.pi * 8000.0,
            plasma_current_A=8000.0,
        )

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
            major_radius=0.976,
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
        self.assertIn("INITC=10000", local_dir)
        self.assertIn("INITC=10000", database_dir)

    def test_resolve_stage2_tf_current_rejects_metadata_mismatch_against_loaded_seed(self):
        module = load_single_stage_example_module()

        tf_coils = [
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: 1.0e5)),
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: 1.0e5)),
        ]

        with self.assertRaisesRegex(ValueError, "does not match the artifact metadata TF_CURRENT_A"):
            module.resolve_stage2_tf_current_A({"TF_CURRENT_A": 8.0e4}, tf_coils)

    def test_resolve_stage2_tf_current_accepts_matching_loaded_seed_value(self):
        module = load_single_stage_example_module()

        tf_coils = [
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: 8.0e4)),
            SimpleNamespace(current=SimpleNamespace(get_value=lambda: 8.0e4)),
        ]

        self.assertEqual(
            module.resolve_stage2_tf_current_A({"TF_CURRENT_A": 8.0e4}, tf_coils),
            8.0e4,
        )

    def test_infer_uniform_tf_current_returns_none_when_coils_are_missing(self):
        module = load_single_stage_example_module()

        self.assertIsNone(module.infer_uniform_tf_current_A([]))

    def test_validate_stage2_seed_contract_rejects_missing_tf_current_metadata(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "missing TF_CURRENT_A even after legacy-contract upgrade"):
            module.validate_stage2_seed_contract(
                {
                    "banana_surf_radius": module.BANANA_WINDING_MINOR_RADIUS_M,
                    "CURVATURE_THRESHOLD": module.MAX_CURVATURE_INV_M,
                }
            )

    def test_validate_stage2_seed_contract_accepts_upgraded_legacy_tf_current(self):
        module = load_single_stage_example_module()
        stage2_results = module.upgrade_legacy_stage2_artifact_results(
            {
                "banana_surf_radius": 0.22,
                "CURVATURE_THRESHOLD": module.MAX_CURVATURE_INV_M,
            },
            known_tf_current_A=8.0e4,
        )

        module.validate_stage2_seed_contract(stage2_results)

    def test_resolve_single_stage_banana_surf_radius_defaults_to_loaded_artifact(self):
        module = load_single_stage_example_module()

        self.assertEqual(
            module.resolve_single_stage_banana_surf_radius(
                {"banana_surf_radius": 0.22},
                None,
            ),
            0.22,
        )

    def test_resolve_single_stage_banana_surf_radius_rejects_cli_override_mismatch(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "must match the loaded Stage 2 artifact radius 0.220000 m"):
            module.resolve_single_stage_banana_surf_radius(
                {"banana_surf_radius": 0.22},
                0.21,
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
            module.validate_loaded_stage2_coils_partition(
                [object()] * 19,
                stage2_results={"NUM_TF_COILS": 20},
                requested_num_tf_coils=20,
            )

    def test_validate_loaded_stage2_coils_partition_rejects_missing_banana_coils(self):
        module = load_single_stage_example_module()

        with self.assertRaisesRegex(ValueError, "leaving no banana coils"):
            module.validate_loaded_stage2_coils_partition(
                [object()] * 20,
                stage2_results={"NUM_TF_COILS": 20},
                requested_num_tf_coils=20,
            )

    def test_build_stage2_bs_path_prefers_current_penalty_dir(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            current_dir = (
                outputs_dir
                / "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-INITC=10000-MAXC=16000-TFC=80000-Order=2-CM=penalty"
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
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=1.0e4,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_build_stage2_bs_path_falls_back_to_legacy_basin_hop_without_tf_segment(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            legacy_dir = (
                outputs_dir
                / "R0=0.976-s=0.24-LW=0.0005-CCW=100-CW=0.0001-SR=0.220-Order=2-BH=3-BS=0.01-BSeed=7"
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
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=1.0e4,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_build_stage2_bs_path_falls_back_to_legacy_radius_for_local_lookup(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            legacy_dir = (
                outputs_dir
                / "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=100-SR=0.220-INITC=10000-MAXC=16000-TFC=80000-Order=2-CM=penalty"
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
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=100.0,
                stage2_seed_banana_surf_radius=module.BANANA_WINDING_MINOR_RADIUS_M,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=1.0e4,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_build_stage2_bs_path_falls_back_to_legacy_radius_for_database_lookup(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            legacy_dir = outputs_dir / "MR=0.976-TF=0.24-LW=0.0005-CCW=100-CW=0.0001-SR=0.22-INITC=10000-TFC=80000-Order=2"
            legacy_dir.mkdir(parents=True)
            expected_path = legacy_dir / "biot_savart_opt.json"
            expected_path.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                stage2_source="database",
                local_stage2_root="/unused",
                database_stage2_root=tmpdir,
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=100.0,
                stage2_seed_banana_surf_radius=module.BANANA_WINDING_MINOR_RADIUS_M,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=1.0e4,
            )

            self.assertEqual(module.build_stage2_bs_path(args), str(expected_path))

    def test_build_stage2_bs_path_discovers_unique_wataru_local_output(self):
        module = load_single_stage_example_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs_dir = Path(tmpdir) / "outputs-demo.nc"
            wataru_dir = (
                outputs_dir
                / "R0=0.976-s=0.24-LW=0.0005-CCW=100-CCT=0.05-CW=0.0001-CT=40-SR=0.220-INITC=10000-MAXC=16000-TFC=80000-Order=2-FCM=wataru_proxy_field-PPC=9000-VFC=500-VFT=wataru_vf_template-CM=penalty"
            )
            wataru_dir.mkdir(parents=True)
            expected_path = wataru_dir / "biot_savart_opt.json"
            expected_path.write_text("{}", encoding="utf-8")

            args = SimpleNamespace(
                stage2_bs_path=None,
                stage2_source="local",
                local_stage2_root=tmpdir,
                database_stage2_root="/unused",
                plasma_surf_filename="demo.nc",
                stage2_seed_major_radius=0.976,
                stage2_seed_toroidal_flux=0.24,
                stage2_seed_length_weight=0.0005,
                stage2_seed_cc_weight=100.0,
                stage2_seed_cc_threshold=0.05,
                stage2_seed_curvature_weight=0.0001,
                stage2_seed_curvature_threshold=40.0,
                stage2_seed_banana_surf_radius=0.22,
                stage2_seed_tf_current_A=8.0e4,
                stage2_seed_order=2,
                stage2_seed_banana_init_current_A=1.0e4,
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

    def test_single_stage_parse_args_accepts_goal_mode_flag(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--single-stage-goal-mode",
                "frontier",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_goal_mode, "frontier")

    def test_single_stage_parse_args_accepts_seed_regime_flag(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--seed-regime",
                "bridge_only",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.seed_regime, "bridge_only")

    def test_single_stage_parse_args_accepts_frontier_volume_weight_flag(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--frontier-volume-weight",
                "200",
            ],
        ):
            args = module.parse_args()

        self.assertAlmostEqual(args.frontier_volume_weight, 200.0)

    def test_single_stage_parse_args_accepts_frontier_scalarization_flags(self):
        module = load_single_stage_example_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--frontier-scalarization-type",
                "achievement_chebyshev_sweep_v1",
                "--frontier-chebyshev-rho",
                "0.02",
                "--frontier-chebyshev-weight-iota",
                "2.0",
                "--epsilon-constraint-qa-max",
                "0.011",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(
            args.frontier_scalarization_type,
            "achievement_chebyshev_sweep_v1",
        )
        self.assertAlmostEqual(args.frontier_chebyshev_rho, 0.02)
        self.assertAlmostEqual(args.frontier_chebyshev_weight_iota, 2.0)
        self.assertAlmostEqual(args.epsilon_constraint_qa_max, 0.011)

    def test_single_stage_parse_args_reads_goal_mode_from_environment(self):
        module = load_single_stage_example_module()

        with patch.dict(os.environ, {"SINGLE_STAGE_GOAL_MODE": "frontier"}, clear=False), patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py"],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_goal_mode, "frontier")

    def test_frontier_goal_mode_warning_message_reports_scale_and_unsaturated_reward(self):
        module = load_single_stage_example_module()
        frontier_goal_config = make_frontier_goal_config(
            module,
            iota_reference=0.15,
            qs_reference=2.0e-4,
        )

        warning = module.frontier_goal_mode_warning_message(frontier_goal_config)

        self.assertIn("normalized tradeoff score", warning)
        self.assertIn("iota_ref=0.150000", warning)
        self.assertIn("1.000000e-05", warning)

    def test_apply_frontier_scalarization_override_uses_chebyshev_lane(self):
        module = load_single_stage_example_module()

        class _ScalarObjective:
            def __init__(self, value, grad):
                self._value = value
                self._grad = np.asarray(grad, dtype=float)

            def J(self):
                return self._value

            def dJ(self):
                return self._grad

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = make_frontier_goal_config(
            module,
            scalarization_type="achievement_chebyshev_sweep_v1",
            chebyshev_rho=0.02,
            chebyshev_weight_iota=2.0,
            chebyshev_weight_volume=1.5,
            chebyshev_weight_qa=1.0,
            chebyshev_weight_boozer=0.5,
        )
        module.surface_iota_terms = [_ScalarObjective(0.13, [1.0, 0.0])]
        module.surface_volume_term = _ScalarObjective(0.09, [0.0, 1.0])
        module.EFFECTIVE_RES_WEIGHT = 1.0
        module.EFFECTIVE_IOTAS_WEIGHT = 1.0
        module.EFFECTIVE_VOLUME_WEIGHT = 1.0
        module.LENGTH_WEIGHT = 1.0
        module.CC_WEIGHT = 0.0
        module.CS_WEIGHT = 0.0
        module.CURVATURE_WEIGHT = 0.0
        module.SURF_DIST_WEIGHT = 0.0

        objective_eval = {
            "total": 0.0,
            "grad": np.zeros(2),
            "J_QS": 1.2e-4,
            "dJ_QS": np.array([0.5, 0.0]),
            "J_QS_objective": 1.2,
            "dJ_QS_objective": np.array([0.5, 0.0]),
            "J_Boozer": 2.0e-6,
            "dJ_Boozer": np.array([0.0, 0.4]),
            "J_Boozer_objective": 2.0,
            "dJ_Boozer_objective": np.array([0.0, 0.4]),
            "J_iota": -0.1,
            "dJ_iota": np.array([-0.3, 0.0]),
            "J_volume": -0.2,
            "dJ_volume": np.array([0.0, -0.2]),
            "J_len": 0.05,
            "dJ_len": np.array([0.1, 0.1]),
            "J_cc": 0.0,
            "dJ_cc": np.zeros(2),
            "J_cs": 0.0,
            "dJ_cs": np.zeros(2),
            "J_curvature": 0.0,
            "dJ_curvature": np.zeros(2),
            "J_surf": 0.0,
            "dJ_surf": np.zeros(2),
        }

        scalarized = module.apply_frontier_scalarization_override(objective_eval)

        self.assertEqual(
            scalarized["frontier_scalarization_type"],
            "achievement_chebyshev_sweep_v1",
        )
        self.assertIn("frontier_chebyshev_deltas", scalarized)
        self.assertNotAlmostEqual(
            scalarized["frontier_goal_total"],
            1.2 + 2.0 - 0.1 - 0.2,
        )

    def test_apply_frontier_scalarization_override_adds_epsilon_search_penalty(self):
        module = load_single_stage_example_module()

        class _ScalarObjective:
            def __init__(self, value, grad):
                self._value = value
                self._grad = np.asarray(grad, dtype=float)

            def J(self):
                return self._value

            def dJ(self):
                return self._grad

        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = make_frontier_goal_config(
            module,
            scalarization_type="epsilon_constraint_sweep_v1",
            epsilon_constraint_qa_max=1.0e-4,
            epsilon_constraint_boozer_max=1.0e-6,
        )
        module.surface_iota_terms = [_ScalarObjective(0.13, [1.0, 0.0])]
        module.surface_volume_term = _ScalarObjective(0.09, [0.0, 1.0])
        module.EFFECTIVE_RES_WEIGHT = 1.0
        module.EFFECTIVE_IOTAS_WEIGHT = 1.0
        module.EFFECTIVE_VOLUME_WEIGHT = 1.0
        module.LENGTH_WEIGHT = 1.0
        module.CC_WEIGHT = 0.0
        module.CS_WEIGHT = 0.0
        module.CURVATURE_WEIGHT = 0.0
        module.SURF_DIST_WEIGHT = 0.0

        objective_eval = {
            "total": 0.0,
            "grad": np.zeros(2),
            "J_QS": 4.0e-4,
            "dJ_QS": np.array([0.5, 0.0]),
            "J_QS_objective": 1.2,
            "dJ_QS_objective": np.array([0.5, 0.0]),
            "J_Boozer": 5.0e-6,
            "dJ_Boozer": np.array([0.0, 0.4]),
            "J_Boozer_objective": 2.0,
            "dJ_Boozer_objective": np.array([0.0, 0.4]),
            "J_iota": -0.1,
            "dJ_iota": np.array([-0.3, 0.0]),
            "J_volume": -0.2,
            "dJ_volume": np.array([0.0, -0.2]),
            "J_len": 0.05,
            "dJ_len": np.array([0.1, 0.1]),
            "J_cc": 0.0,
            "dJ_cc": np.zeros(2),
            "J_cs": 0.0,
            "dJ_cs": np.zeros(2),
            "J_curvature": 0.0,
            "dJ_curvature": np.zeros(2),
            "J_surf": 0.0,
            "dJ_surf": np.zeros(2),
        }

        scalarized = module.apply_frontier_scalarization_override(objective_eval)

        self.assertEqual(
            scalarized["frontier_scalarization_type"],
            "epsilon_constraint_sweep_v1",
        )
        self.assertGreater(scalarized["frontier_epsilon_penalty"], 0.0)
        self.assertIn("qa_error", scalarized["frontier_epsilon_constraints"])

    def test_apply_frontier_scalarization_override_is_noop_outside_frontier_mode(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "target"
        module.FRONTIER_GOAL_CONFIG = None
        if hasattr(module, "surface_iota_terms"):
            delattr(module, "surface_iota_terms")
        if hasattr(module, "surface_volume_term"):
            delattr(module, "surface_volume_term")

        objective_eval = {
            "total": 1.23,
            "grad": np.array([0.1, -0.2]),
            "J_QS": 1.2e-4,
            "dJ_QS": np.array([0.5, 0.0]),
            "J_Boozer": 2.0e-6,
            "dJ_Boozer": np.array([0.0, 0.4]),
            "J_iota": -0.1,
            "dJ_iota": np.array([-0.3, 0.0]),
        }

        scalarized = module.apply_frontier_scalarization_override(objective_eval)

        self.assertEqual(set(scalarized.keys()), set(objective_eval.keys()))
        self.assertIsNot(scalarized, objective_eval)
        np.testing.assert_allclose(scalarized["grad"], objective_eval["grad"])

    def test_apply_frontier_scalarization_override_projects_metric_gradients(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "frontier"
        module.FRONTIER_GOAL_CONFIG = make_frontier_goal_config(
            module,
            scalarization_type="achievement_chebyshev_sweep_v1",
            chebyshev_rho=0.02,
        )
        module.JF = object()
        module.surface_iota_terms = [
            FakeProjectedObjective(0.13, [1.0, 0.0], [0.0, 0.0, 1.0, 0.0])
        ]
        module.surface_volume_term = FakeProjectedObjective(
            0.09,
            [0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        )
        module.EFFECTIVE_RES_WEIGHT = 1.0
        module.EFFECTIVE_IOTAS_WEIGHT = 1.0
        module.EFFECTIVE_VOLUME_WEIGHT = 1.0
        module.LENGTH_WEIGHT = 1.0
        module.CC_WEIGHT = 0.0
        module.CS_WEIGHT = 0.0
        module.CURVATURE_WEIGHT = 0.0
        module.SURF_DIST_WEIGHT = 0.0

        objective_eval = {
            "total": 0.0,
            "grad": np.zeros(4),
            "J_QS": 1.2e-4,
            "dJ_QS": np.array([0.5, 0.0, 0.0, 0.0]),
            "J_QS_objective": 1.2,
            "dJ_QS_objective": np.array([0.5, 0.0, 0.0, 0.0]),
            "J_Boozer": 2.0e-6,
            "dJ_Boozer": np.array([0.0, 0.4, 0.0, 0.0]),
            "J_Boozer_objective": 2.0,
            "dJ_Boozer_objective": np.array([0.0, 0.4, 0.0, 0.0]),
            "J_iota": -0.1,
            "dJ_iota": np.array([-0.3, 0.0, 0.0, 0.0]),
            "J_volume": -0.2,
            "dJ_volume": np.array([0.0, -0.2, 0.0, 0.0]),
            "J_len": 0.05,
            "dJ_len": np.array([0.1, 0.1, 0.0, 0.0]),
            "J_cc": 0.0,
            "dJ_cc": np.zeros(4),
            "J_cs": 0.0,
            "dJ_cs": np.zeros(4),
            "J_curvature": 0.0,
            "dJ_curvature": np.zeros(4),
            "J_surf": 0.0,
            "dJ_surf": np.zeros(4),
        }

        scalarized = module.apply_frontier_scalarization_override(objective_eval)

        self.assertEqual(scalarized["frontier_goal_grad"].shape, (4,))
        np.testing.assert_allclose(
            scalarized["dJ_iota_metric"],
            [0.0, 0.0, 1.0, 0.0],
        )
        np.testing.assert_allclose(
            scalarized["dJ_volume_metric"],
            [0.0, 0.0, 0.0, 1.0],
        )

    def test_evaluate_total_objective_matches_raw_impl_outside_frontier_mode(self):
        module = load_single_stage_example_module()
        module.SINGLE_STAGE_GOAL_MODE = "target"
        module.FRONTIER_GOAL_CONFIG = None

        surface_weights = np.array([1.0])
        non_qs = [FakeAlgebraicObjective(1.2, [0.5, 0.0])]
        boozer = [FakeAlgebraicObjective(2.0, [0.0, 0.4])]
        jiota = FakeAlgebraicObjective(-0.1, [-0.3, 0.0])
        curve_length = FakeAlgebraicObjective(0.05, [0.1, 0.1])
        curve_curve = FakeAlgebraicObjective(0.25, [0.3, 0.4])
        curve_surface = FakeAlgebraicObjective(0.15, [0.2, -0.1])
        curvature = FakeAlgebraicObjective(0.35, [0.2, 0.3])
        resolved_terms = {
            "effective_res_weight": 7.0,
            "effective_iotas_weight": 11.0,
            "effective_volume_weight": 0.0,
            "JNonQSObjective": None,
            "JBoozerObjective": None,
            "JVolume": None,
        }

        with patch.object(
            module,
            "resolve_current_surface_objective_terms",
            return_value=resolved_terms,
        ):
            wrapped = module.evaluate_total_objective(
                surface_weights,
                non_qs,
                boozer,
                RES_WEIGHT=999.0,
                Jiota=jiota,
                IOTAS_WEIGHT=888.0,
                JCurveLength=curve_length,
                LENGTH_WEIGHT=0.5,
                JCurveCurve=curve_curve,
                CC_WEIGHT=2.0,
                JCurveSurface=curve_surface,
                CS_WEIGHT=3.0,
                JCurvature=curvature,
                CURVATURE_WEIGHT=4.0,
            )
        raw = module._evaluate_total_objective_impl(
            surface_weights,
            non_qs,
            boozer,
            resolved_terms["effective_res_weight"],
            jiota,
            resolved_terms["effective_iotas_weight"],
            curve_length,
            0.5,
            curve_curve,
            2.0,
            curve_surface,
            3.0,
            curvature,
            4.0,
            JNonQSObjective=resolved_terms["JNonQSObjective"],
            JBoozerObjective=resolved_terms["JBoozerObjective"],
            JVolume=resolved_terms["JVolume"],
            VOLUME_WEIGHT=resolved_terms["effective_volume_weight"],
        )

        self.assertEqual(set(wrapped.keys()), set(raw.keys()))
        for key, raw_value in raw.items():
            wrapped_value = wrapped[key]
            if isinstance(raw_value, np.ndarray):
                np.testing.assert_allclose(wrapped_value, raw_value)
            else:
                self.assertEqual(wrapped_value, raw_value)

    def test_evaluate_total_objective_projects_component_gradients_to_search_space(self):
        module = load_single_stage_example_module()

        surface_weights = np.array([1.0])
        objective_optimizable = object()
        non_qs = [
            FakeProjectedObjective(1.2, [0.5, 0.0], [0.5, 0.0, 0.0, 0.0])
        ]
        boozer = [
            FakeProjectedObjective(2.0, [0.0, 0.4], [0.0, 0.4, 0.0, 0.0])
        ]
        jiota = FakeProjectedObjective(-0.1, [-0.3, 0.0], [0.0, 0.0, -0.3, 0.0])
        volume = FakeProjectedObjective(-0.2, [0.0, -0.2], [0.0, 0.0, 0.0, -0.2])
        curve_length = FakeProjectedObjective(0.05, [0.1, 0.1], [0.1, 0.1, 0.0, 0.0])
        curve_curve = FakeProjectedObjective(0.0, [0.0, 0.0], np.zeros(4))
        curve_surface = FakeProjectedObjective(0.0, [0.0, 0.0], np.zeros(4))
        curvature = FakeProjectedObjective(0.0, [0.0, 0.0], np.zeros(4))

        objective_eval = module._evaluate_total_objective_impl(
            surface_weights,
            non_qs,
            boozer,
            RES_WEIGHT=1.0,
            Jiota=jiota,
            IOTAS_WEIGHT=1.0,
            JCurveLength=curve_length,
            LENGTH_WEIGHT=1.0,
            JCurveCurve=curve_curve,
            CC_WEIGHT=0.0,
            JCurveSurface=curve_surface,
            CS_WEIGHT=0.0,
            JCurvature=curvature,
            CURVATURE_WEIGHT=0.0,
            JVolume=volume,
            VOLUME_WEIGHT=1.0,
            objective_optimizable=objective_optimizable,
        )

        self.assertEqual(objective_eval["grad"].shape, (4,))
        self.assertEqual(objective_eval["dJ_QS_objective"].shape, (4,))
        self.assertEqual(objective_eval["dJ_Boozer_objective"].shape, (4,))
        self.assertEqual(objective_eval["dJ_iota"].shape, (4,))
        self.assertEqual(objective_eval["dJ_volume"].shape, (4,))
        np.testing.assert_allclose(objective_eval["dJ_QS"], [0.5, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(objective_eval["dJ_Boozer"], [0.0, 0.4, 0.0, 0.0])
        np.testing.assert_allclose(objective_eval["dJ_iota"], [0.0, 0.0, -0.3, 0.0])
        np.testing.assert_allclose(objective_eval["dJ_volume"], [0.0, 0.0, 0.0, -0.2])

    def test_validate_single_stage_alm_formulation_args_rejects_frontier_thresholded_physics(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            alm_formulation="thresholded_physics",
            single_stage_goal_mode="frontier",
            constraint_method="alm",
            alm_qs_threshold=0.1,
            alm_boozer_threshold=0.2,
            alm_iota_penalty_threshold=0.3,
            alm_length_penalty_threshold=0.4,
        )

        with self.assertRaisesRegex(
            ValueError,
            "frontier is not compatible with --alm-formulation=thresholded_physics",
        ):
            module.validate_single_stage_alm_formulation_args(args)

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

    def test_single_stage_parse_args_preserve_wrapper_default_hardware_thresholds(self):
        module = load_single_stage_example_module()

        with patch.object(sys, "argv", ["single_stage_banana_example.py"]):
            args = module.parse_args()

        self.assertEqual(args.cs_dist, 0.015)
        self.assertEqual(args.curvature_threshold, 100.0)
        self.assertEqual(args.banana_current_max_A, 16000.0)

    def test_apply_default_stage2_seed_args_uses_legacy_seed_defaults(self):
        module = load_single_stage_example_module()
        args = SimpleNamespace(
            plasma_surf_filename="wout_nfp22ginsburg_000_014417_iota15.nc",
            stage2_seed_major_radius=None,
            stage2_seed_toroidal_flux=None,
            stage2_seed_length_weight=None,
            stage2_seed_cc_weight=None,
            stage2_seed_curvature_weight=None,
            stage2_seed_cc_threshold=None,
            stage2_seed_curvature_threshold=None,
            stage2_seed_banana_surf_radius=None,
            stage2_seed_tf_current_A=None,
            stage2_seed_order=None,
            stage2_seed_banana_init_current_A=None,
        )

        module.apply_default_stage2_seed_args(args)

        self.assertEqual(args.stage2_seed_curvature_threshold, 100.0)
        self.assertEqual(args.stage2_seed_banana_surf_radius, 0.21)
        self.assertEqual(args.stage2_seed_tf_current_A, 8.0e4)

    def test_stage2_parse_args_accepts_banana_current_controls(self):
        module = load_stage2_module()

        with patch.object(
            sys,
            "argv",
            [
                "banana_coil_solver.py",
                "--banana-init-current-A",
                "12000",
                "--banana-current-max-A",
                "16000",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.banana_init_current_A, 12000.0)
        self.assertEqual(args.banana_current_max_A, 16000.0)

    def test_penalty_traversal_helper_applies_symmetric_box_bound(self):
        module = load_stage2_module()
        leaf_current, scaled_current = _make_unbounded_scaled_current()

        resolved = module.apply_penalty_traversal_forbidden_box_bounds(
            bound_targets={"banana_current": scaled_current},
            requested_thresholds={"banana_current": 16000.0},
        )

        self.assertEqual(resolved, {"banana_current": 16000.0})
        np.testing.assert_allclose(leaf_current.local_lower_bounds, [-16000.0])
        np.testing.assert_allclose(leaf_current.local_upper_bounds, [16000.0])

    def test_shared_penalty_traversal_helper_uses_schema_bound(self):
        module = load_stage2_module()
        leaf_current, scaled_current = _make_unbounded_scaled_current()

        resolved = module.apply_penalty_traversal_forbidden_box_bounds(
            bound_targets={"banana_current": scaled_current},
            requested_thresholds={"banana_current": 20000.0},
        )

        self.assertEqual(resolved, {"banana_current": 16000.0})
        np.testing.assert_allclose(leaf_current.local_lower_bounds, [-16000.0])
        np.testing.assert_allclose(leaf_current.local_upper_bounds, [16000.0])

    def test_shared_penalty_traversal_helper_rejects_missing_target(self):
        module = load_stage2_module()

        with self.assertRaisesRegex(
            KeyError,
            "Missing penalty box-bound target for hardware constraint 'banana_current'",
        ):
            module.apply_penalty_traversal_forbidden_box_bounds(
                bound_targets={},
                requested_thresholds={"banana_current": 16000.0},
            )


class Stage2ArtifactWriterTests(unittest.TestCase):
    def test_materialize_stage2_artifact_results_rejects_biot_savart_partition_mismatch(self):
        module = load_stage2_module()

        fake_bs = SimpleNamespace(
            coils=[object(), object(), object()],
            save=lambda *_args, **_kwargs: None,
        )

        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaisesRegex(
            ValueError,
            "Stage 2 artifact writer partition metadata does not match the loaded BiotSavart coil count",
        ):
            module.materialize_stage2_artifact_results(
                args=SimpleNamespace(),
                stage2_bs_artifact_path=str(Path(tmpdir) / "biot_savart_opt.json"),
                results_kwargs={
                    "num_tf_coils": 1,
                    "num_banana_coils": 1,
                    "num_proxy_coils": 0,
                    "num_vf_coils": 0,
                },
                stage2_iota_runtime=None,
                new_bs=fake_bs,
                new_surf=SimpleNamespace(),
            )

    def test_materialize_stage2_artifact_results_emits_matching_checksum(self):
        module = load_stage2_module()

        def _save(path):
            Path(path).write_text('{"coils": [1, 2]}', encoding="utf-8")

        fake_bs = SimpleNamespace(
            coils=[object(), object()],
            save=_save,
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            module,
            "_magnetic_field_plots",
            return_value=0.125,
        ), patch.object(
            module,
            "_build_stage2_results_impl",
            return_value={"FIELD_ERROR": 0.125},
        ), patch.object(
            module,
            "build_stage2_iota_report_payload",
            return_value={},
        ):
            artifact_path = Path(tmpdir) / "biot_savart_opt.json"
            results = module.materialize_stage2_artifact_results(
                args=SimpleNamespace(),
                stage2_bs_artifact_path=str(artifact_path),
                results_kwargs={
                    "num_tf_coils": 1,
                    "num_banana_coils": 1,
                    "num_proxy_coils": 0,
                    "num_vf_coils": 0,
                },
                stage2_iota_runtime=None,
                new_bs=fake_bs,
                new_surf=SimpleNamespace(),
            )
            expected_digest = module.compute_stage2_bs_sha256(artifact_path)

        self.assertEqual(
            results["STAGE2_BS_SHA256"],
            expected_digest,
        )


class Stage2RuntimeSmokeTests(unittest.TestCase):
    _EXPECTED_BASIN_TELEMETRY = {
        "basin_accepted_hops": 1,
        "basin_rejected_hops": 1,
        "basin_completed_hops": 2,
        "basin_best_objective": 0.42,
        "basin_initial_objective": 0.55,
        "basin_best_hop_objective": 0.42,
        "basin_best_hop_index": 1,
        "basin_best_result_source": "hop",
        "basin_objective_improvement": 0.13,
        "basin_accept_test_rejections": 1,
        "basin_accept_test_triggered": True,
        "basin_nonfinite_rejections": 0,
        "basin_normalized_step_rejections": 1,
    }

    @staticmethod
    def _make_fake_tf_coils(curve_cls, current_cls, *, count=20, current_A=8.0e4):
        return [
            SimpleNamespace(curve=curve_cls(), current=current_cls(current_A))
            for _ in range(count)
        ]

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
            "banana_surf_radius": 0.21,
            "tf_current_A": 8.0e4,
            "banana_init_current_A": 1.0e4,
            "banana_current_max_A": 1.6e4,
            "major_radius": 0.976,
            "accept_offspec_r0_seed": False,
            "toroidal_flux": 0.24,
            "order": 2,
            "maxiter": 30,
            "ftol": 1e-15,
            "gtol": 1e-15,
            "constraint_method": "penalty",
            "alm_max_outer_iters": 7,
            "alm_penalty_init": 2.0,
            "alm_penalty_scale": 3.0,
            "alm_penalty_max": 50.0,
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
            "stage2_iota_mode": "off",
            "stage2_iota_target": None,
            "stage2_iota_tolerance": 5.0e-3,
            "stage2_iota_weight": 1.0,
            "stage2_iota_vol_target": 0.10,
            "stage2_iota_constraint_weight": 1.0,
            "stage2_iota_num_tf_coils": 20,
            "stage2_iota_nphi": 91,
            "stage2_iota_ntheta": 32,
            "stage2_iota_mpol": 8,
            "stage2_iota_ntor": 6,
            "length_weight": 5e-4,
            "length_target": 1.7,
            "cc_threshold": 0.05,
            "cc_weight": 100.0,
            "curvature_weight": 1e-4,
            "curvature_threshold": 100.0,
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

    def _run_stage2_main(
        self,
        *,
        init_only,
        constraint_method,
        use_seed,
        # Seeded smoke tests emulate a valid donor sidecar by default. Opt out
        # only when exercising the explicit sidecar-required rejection path.
        seed_has_results_sidecar=True,
        basin_hops=0,
        banana_current_A=9500.0,
        alm_accepted_candidate_x=None,
        artifact_state_by_x=None,
        seed_stage2_results=None,
        arg_overrides=None,
        missing_attr_names=(),
    ):
        module = load_stage2_module()
        runtime = {
            "seed_loads": 0,
            "initialize_calls": 0,
            "minimize_calls": 0,
            "minimize_alm_calls": 0,
            "run_basin_hopping_calls": 0,
            "minimize_bounds": None,
            "basin_bounds": None,
            "initialize_extra_kwargs": None,
            "curve_curve_curves": None,
            "curve_surface_curves": None,
            "results": None,
        }

        class FakeStage2Objective:
            def __init__(self, value, gradient, x=None):
                self._value = float(value)
                self._gradient = np.asarray(gradient, dtype=float)
                self.x = np.zeros(2, dtype=float) if x is None else np.asarray(x, dtype=float)
                self.lower_bounds = np.full(self.x.shape, -np.inf, dtype=float)
                self.upper_bounds = np.full(self.x.shape, np.inf, dtype=float)

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
                self.local_lower_bounds = np.array([-np.inf], dtype=float)
                self.local_upper_bounds = np.array([np.inf], dtype=float)

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
                self.coils = []

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
                return 0.06

        class FakeCurvatureObjective(FakeStage2Objective):
            def __init__(self):
                super().__init__(0.35, [0.2, 0.3])
                self.threshold = 40.0
                self.curve = SimpleNamespace(kappa=lambda: np.array([39.0, 41.0], dtype=float))

        class FakeCurveSurfaceDistance(FakeStage2Objective):
            def __init__(self):
                super().__init__(0.15, [0.05, 0.06])
                self.minimum_distance = 0.015
                self.curves = ["curve_a", "curve_b"]
                self.surface = fake_surface

            def shortest_distance(self):
                return 0.02

        fake_bs = FakeBiotSavart()
        fake_surface = FakeSurface()
        fake_curve_names = ["curve_a", "curve_b", "curve_c"]
        fake_banana_curve = SimpleNamespace(kappa=lambda: np.array([39.0, 41.0], dtype=float))
        fake_banana_coils = [
            SimpleNamespace(curve=fake_banana_curve, current=FakeCurrent(banana_current_A))
        ]
        fake_tf_coils = self._make_fake_tf_coils(FakeCurve, FakeCurrent)

        def build_coil_bundle(*, wataru_proxy_field):
            proxy_coils = []
            vf_coils = []
            curves = list(fake_curve_names)
            if wataru_proxy_field:
                proxy_coils = [
                    SimpleNamespace(curve="proxy_curve", current=FakeCurrent(9000.0))
                ]
                vf_coils = [
                    SimpleNamespace(curve="vf_curve", current=FakeCurrent(-500.0))
                ]
                curves = [
                    *(f"tf_curve_{index}" for index in range(20)),
                    fake_banana_curve,
                    "proxy_curve",
                    "vf_curve",
                ]
            return curves, proxy_coils, vf_coils

        def fake_seed_loader(seed_bs_path, surf, num_tf_coils, out_dir, **_kwargs):
            runtime["seed_loads"] += 1
            self.assertEqual(num_tf_coils, 20)
            self.assertIs(surf, fake_surface)
            effective_seed_stage2_results = (
                seed_stage2_results
                if seed_stage2_results is not None
                else {"FINITE_CURRENT_MODE": "wataru_proxy_field"}
            )
            curves, proxy_coils, vf_coils = build_coil_bundle(
                wataru_proxy_field=(
                    effective_seed_stage2_results.get("FINITE_CURRENT_MODE")
                    == "wataru_proxy_field"
                ),
            )
            fake_bs.coils = [*fake_tf_coils, *fake_banana_coils, *proxy_coils, *vf_coils]
            return (
                fake_bs,
                curves,
                fake_banana_curve,
                fake_banana_coils,
                fake_tf_coils,
                proxy_coils,
                vf_coils,
            )

        def fake_initialize_coils(
            surf,
            surf_coils,
            tf_coils,
            num_quadpoints,
            order,
            banana_init_current_A,
            phi_center,
            theta_center,
            phi_width,
            theta_width,
            out_dir,
            **extra_kwargs,
        ):
            runtime["initialize_calls"] += 1
            runtime["initialize_extra_kwargs"] = dict(extra_kwargs)
            self.assertIs(surf, fake_surface)
            self.assertEqual(surf_coils, "surf_coils")
            self.assertEqual(len(tf_coils), 20)
            self.assertEqual(num_quadpoints, 16)
            self.assertEqual(order, 2)
            self.assertEqual(banana_init_current_A, 1.0e4)
            self.assertEqual(phi_center, np.pi / 4.0)
            self.assertEqual(theta_center, np.pi)
            self.assertEqual(phi_width, np.pi / 8.0)
            self.assertEqual(theta_width, np.pi / 6.0)
            self.assertTrue(str(out_dir).endswith("outputs-demo.nc/"))
            # Fix #4: _initialize_coils no longer takes finite_current_mode.
            # Proxy is always built; VF is built iff vf_template_path is set —
            # that kwarg is the SSOT the mock must mirror.
            curves, proxy_coils, vf_coils = build_coil_bundle(
                wataru_proxy_field=bool(extra_kwargs.get("vf_template_path")),
            )
            fake_bs.coils = [*fake_tf_coils, *fake_banana_coils, *proxy_coils, *vf_coils]
            return (
                fake_bs,
                curves,
                fake_banana_curve,
                fake_banana_coils,
                proxy_coils,
                vf_coils,
            )

        def fake_curve_curve_distance(curves, *_args, **_kwargs):
            runtime["curve_curve_curves"] = tuple(curves)
            return FakeCurveDistance()

        def fake_curve_surface_distance(curves, *_args, **_kwargs):
            runtime["curve_surface_curves"] = tuple(curves)
            return FakeCurveSurfaceDistance()

        def fake_minimize(*_args, **_kwargs):
            runtime["minimize_calls"] += 1
            runtime["minimize_bounds"] = _kwargs.get("bounds")
            return SimpleNamespace(
                x=np.array([0.3, -0.2], dtype=float),
                nit=4,
                message="penalty_ok",
                success=True,
            )

        def fake_minimize_alm(*_args, **_kwargs):
            runtime["minimize_alm_calls"] += 1
            accepted_callback = _kwargs.get("accepted_callback")
            if accepted_callback is not None and alm_accepted_candidate_x is not None:
                accepted_callback(np.asarray(alm_accepted_candidate_x, dtype=float))
            return SimpleNamespace(
                x=np.array([0.1, 0.2], dtype=float),
                nit=5,
                message="alm_ok",
                success=True,
                outer_iterations=2,
                penalty=3.5,
                multipliers=np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=float),
                constraint_values=np.array([0.0, 0.01, 0.0, 0.0, 0.0], dtype=float),
                solver_constraint_values=np.array([0.0, 0.2, 0.0, 0.0, 0.0], dtype=float),
                hard_signed_constraint_values=np.array([0.0, 0.02, 0.0, 0.0, 0.0], dtype=float),
                hard_violation_values=np.array([0.0, 0.01, 0.0, 0.0, 0.0], dtype=float),
                surrogate_signed_constraint_values=np.array([0.0, 0.2, 0.0, 0.0, 0.0], dtype=float),
                trust_radius=0.1,
                multiplier_cap_binding=True,
                multiplier_cap_binding_indices=[1],
                termination_reason="max_outer_after_subproblem_limit",
                converged_to_tolerances=False,
                restored_best_feasible=True,
                restored_best_feasible_reason="final_iterate_worse_than_best_feasible",
                optimizer_success=False,
                optimizer_message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
                final_max_feasibility_violation=0.01,
                final_stationarity_norm=0.02,
                final_raw_stationarity_norm=0.03,
                final_kkt_stationarity_norm=0.025,
                final_hard_max_violation=0.01,
                final_surrogate_max_value=0.2,
                hard_positive_shift_zero=True,
                signal_mismatch_active=False,
                final_penalty_gradient_norm=0.4,
                final_feasibility_tolerance=1.0e-3,
                final_stationarity_tolerance=5.0e-3,
                history=[{"outer_iteration": 1}],
            )

        def fake_capture_stage2_artifact_state(**kwargs):
            dofs = tuple(np.asarray(kwargs["dofs"], dtype=float).tolist())
            if artifact_state_by_x is None or dofs not in artifact_state_by_x:
                raise AssertionError(f"unexpected artifact-state request for dofs {dofs}")
            state = artifact_state_by_x[dofs]
            return {
                "x": np.asarray(kwargs["dofs"], dtype=float).copy(),
                "field_objective": float(state["field_objective"]),
                "coil_length": float(state["coil_length"]),
                "curve_curve_min_dist": float(state["curve_curve_min_dist"]),
                "curve_surface_min_dist": float(state["curve_surface_min_dist"]),
                "max_curvature": float(state["max_curvature"]),
                "banana_current_A": float(state["banana_current_A"]),
                "tf_current_A": float(state["tf_current_A"]),
                "hardware_status": {
                    "success": bool(state["hardware_status"]["success"]),
                    "violations": list(state["hardware_status"]["violations"]),
                },
            }

        def fake_run_basin_hopping(*_args, **_kwargs):
            runtime["run_basin_hopping_calls"] += 1
            self.assertEqual(_kwargs["basin_temperature"], 2.5)
            self.assertEqual(_kwargs["basin_niter_success"], 6)
            runtime["basin_bounds"] = _kwargs["minimizer_kwargs"].get("bounds")
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
                **({} if arg_overrides is None else dict(arg_overrides)),
            )
            for attr_name in missing_attr_names:
                delattr(args, attr_name)

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
                            SimpleNamespace(
                                gamma=lambda: np.ones((2, 2, 3), dtype=float) * 0.1,
                                to_vtk=lambda *_a, **_k: None,
                            ),
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
                        lambda *_args, **_kwargs: FakeStage2Objective(1.6, [0.1, 0.2]),
                    ),
                    patch.object(
                        module,
                        "CurveCurveDistance",
                        side_effect=fake_curve_curve_distance,
                    ),
                    patch.object(
                        module,
                        "CurveSurfaceDistance",
                        side_effect=fake_curve_surface_distance,
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
                if use_seed and seed_has_results_sidecar:
                    effective_seed_stage2_results = (
                        seed_stage2_results
                        if seed_stage2_results is not None
                        else {"FINITE_CURRENT_MODE": "wataru_proxy_field"}
                    )
                    stack.enter_context(
                        patch.object(
                            module,
                            "load_stage2_seed_results",
                            return_value=(
                                Path(stage2_bs_path).with_name("results.json"),
                                dict(effective_seed_stage2_results),
                            ),
                        )
                    )
                if artifact_state_by_x is not None:
                    stack.enter_context(
                        patch.object(
                            module,
                            "_capture_stage2_artifact_state",
                            side_effect=fake_capture_stage2_artifact_state,
                        )
                    )
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

    def _assert_banana_current_cap_rejected(self, runtime):
        self.assertFalse(runtime["results"]["OPTIMIZER_SUCCESS"])
        self.assertFalse(runtime["results"]["HARDWARE_CONSTRAINTS_OK"])
        self.assertTrue(
            any(
                "banana_current" in violation
                for violation in runtime["results"]["HARDWARE_CONSTRAINT_VIOLATIONS"]
            )
        )

    def _assert_init_only_runtime_counts(self, runtime, *, seed_loads, initialize_calls):
        self._assert_runtime_counts(
            runtime,
            seed_loads=seed_loads,
            initialize_calls=initialize_calls,
            minimize_calls=0,
            minimize_alm_calls=0,
        )
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "init_only")

    def test_stage2_main_init_only_loads_seed_and_writes_results(self):
        runtime = self._run_stage2_main(init_only=True, constraint_method="penalty", use_seed=True)

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
        )
        self.assertTrue(runtime["results"]["OPTIMIZER_SUCCESS"])
        self.assertEqual(runtime["results"]["iterations"], 0)
        self.assertTrue(runtime["results"]["HARDWARE_CONSTRAINTS_OK"])
        self.assertTrue(runtime["results"]["STAGE2_BS_PATH"].endswith("seed.json"))

    def test_stage2_main_injected_args_without_accept_offspec_flag_use_parser_default(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=True,
            missing_attr_names=("accept_offspec_r0_seed",),
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
        )

    def test_stage2_main_rejects_wataru_seed_without_results_sidecar(self):
        workflow_runner_common = load_workflow_runner_common_module()

        with self.assertRaisesRegex(
            ValueError,
            re.escape(workflow_runner_common.STAGE2_SIDECAR_REQUIRED_ERROR),
        ):
            self._run_stage2_main(
                init_only=True,
                constraint_method="penalty",
                use_seed=True,
                seed_has_results_sidecar=False,
                arg_overrides={"finite_current_mode": "wataru_proxy_field"},
            )

    def test_stage2_main_init_only_wataru_proxy_field_uses_repo_default_vf_and_banana_only_penalties(self):
        workflow_helpers = load_workflow_helpers_module()
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=False,
            arg_overrides={
                "finite_current_mode": "wataru_proxy_field",
                "proxy_plasma_current_A": 9000.0,
                "vf_current_A": 500.0,
            },
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=0,
            initialize_calls=1,
        )
        self.assertEqual(runtime["results"]["FINITE_CURRENT_MODE"], "wataru_proxy_field")
        self.assertEqual(runtime["results"]["NUM_PROXY_COILS"], 1)
        self.assertEqual(runtime["results"]["NUM_VF_COILS"], 1)
        self.assertEqual(runtime["results"]["PROXY_PLASMA_CURRENT_A"], 9000.0)
        self.assertEqual(runtime["results"]["VF_CURRENT_A"], 500.0)
        self.assertEqual(
            runtime["results"]["VF_TEMPLATE_PATH"],
            workflow_helpers.default_wataru_vf_template_path(),
        )
        self.assertEqual(len(runtime["curve_curve_curves"]), 1)
        self.assertEqual(len(runtime["curve_surface_curves"]), 1)
        self.assertEqual(
            runtime["curve_surface_curves"][0],
            runtime["curve_curve_curves"][0],
        )
        self.assertEqual(
            runtime["initialize_extra_kwargs"]["vf_template_path"],
            workflow_helpers.default_wataru_vf_template_path(),
        )

    def test_stage2_main_init_only_wataru_seed_restart_uses_banana_only_penalties(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=True,
            arg_overrides={"finite_current_mode": "wataru_proxy_field"},
            seed_stage2_results={
                "PLASMA_SURF_FILENAME": "demo.nc",
                "TF_CURRENT_A": 8.0e4,
                "NUM_TF_COILS": 20,
                "NUM_BANANA_COILS": 1,
                "NUM_PROXY_COILS": 1,
                "NUM_VF_COILS": 1,
                "FINITE_CURRENT_MODE": "wataru_proxy_field",
                "BOOZER_CURRENT_CONVENTION": "mu0",
                "PROXY_PLASMA_CURRENT_A": 9000.0,
                "VF_CURRENT_A": 500.0,
                "VF_TEMPLATE_PATH": "/tmp/vf_template.json",
            },
        )

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
        )
        self.assertEqual(runtime["results"]["FINITE_CURRENT_MODE"], "wataru_proxy_field")
        self.assertEqual(runtime["results"]["NUM_PROXY_COILS"], 1)
        self.assertEqual(runtime["results"]["NUM_VF_COILS"], 1)
        self.assertEqual(len(runtime["curve_curve_curves"]), 1)
        self.assertEqual(len(runtime["curve_surface_curves"]), 1)
        self.assertEqual(
            runtime["curve_surface_curves"][0],
            runtime["curve_curve_curves"][0],
        )

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
        self.assertEqual(
            runtime["results"]["ALM_TERMINATION_REASON"],
            "max_outer_after_subproblem_limit",
        )
        self.assertFalse(runtime["results"]["ALM_CONVERGED"])
        self.assertTrue(runtime["results"]["ALM_RESTORED_BEST_FEASIBLE"])
        self.assertEqual(
            runtime["results"]["ALM_RESTORED_BEST_FEASIBLE_REASON"],
            "final_iterate_worse_than_best_feasible",
        )
        self.assertFalse(runtime["results"]["ALM_INNER_OPTIMIZER_SUCCESS"])
        self.assertEqual(
            runtime["results"]["ALM_INNER_OPTIMIZER_MESSAGE"],
            "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
        )
        self.assertEqual(
            runtime["results"]["ALM_FINAL_MAX_FEASIBILITY_VIOLATION"],
            0.01,
        )
        self.assertEqual(runtime["results"]["ALM_FINAL_STATIONARITY_NORM"], 0.02)
        self.assertEqual(runtime["results"]["ALM_FINAL_RAW_STATIONARITY_NORM"], 0.03)
        self.assertEqual(runtime["results"]["ALM_FINAL_KKT_STATIONARITY_NORM"], 0.025)
        np.testing.assert_allclose(
            runtime["results"]["ALM_FINAL_HARD_SIGNED_CONSTRAINT_VALUES"],
            [0.0, 0.02, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            runtime["results"]["ALM_FINAL_HARD_VIOLATION_VALUES"],
            [0.0, 0.01, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            runtime["results"]["ALM_FINAL_SURROGATE_SIGNED_CONSTRAINT_VALUES"],
            [0.0, 0.2, 0.0, 0.0, 0.0],
        )
        self.assertEqual(runtime["results"]["ALM_FINAL_HARD_MAX_VIOLATION"], 0.01)
        self.assertEqual(runtime["results"]["ALM_FINAL_SURROGATE_MAX_VALUE"], 0.2)
        self.assertTrue(runtime["results"]["ALM_FINAL_HARD_POSITIVE_SHIFT_ZERO"])
        self.assertFalse(runtime["results"]["ALM_FINAL_SIGNAL_MISMATCH_ACTIVE"])
        self.assertEqual(runtime["results"]["ALM_FINAL_PENALTY_GRADIENT_NORM"], 0.4)
        self.assertEqual(runtime["results"]["ALM_FINAL_FEASIBILITY_TOL"], 1.0e-3)
        self.assertEqual(runtime["results"]["ALM_FINAL_STATIONARITY_TOL"], 5.0e-3)
        self.assertTrue(runtime["results"]["ALM_MULTIPLIER_CAP_BINDING"])
        self.assertEqual(runtime["results"]["ALM_MULTIPLIER_CAP_BINDING_INDICES"], [1])
        self.assertEqual(runtime["results"]["TERMINATION_MESSAGE"], "alm_ok")

    def test_stage2_main_alm_restores_best_exact_hardware_pass_for_artifact_output(self):
        def make_artifact_state(
            field_objective,
            coil_length,
            *,
            success,
            curve_curve_min_dist=0.06,
            max_curvature=41.0,
        ):
            violations = [] if success else [f"coil_length {coil_length:.6f} > 1.700000"]
            return {
                "field_objective": float(field_objective),
                "coil_length": float(coil_length),
                "curve_curve_min_dist": float(curve_curve_min_dist),
                "curve_surface_min_dist": 0.02,
                "max_curvature": float(max_curvature),
                "banana_current_A": 9500.0,
                "tf_current_A": 8.0e4,
                "hardware_status": {
                    "success": bool(success),
                    "violations": violations,
                },
            }

        runtime = self._run_stage2_main(
            init_only=False,
            constraint_method="alm",
            use_seed=True,
            alm_accepted_candidate_x=np.array([0.9, 0.8], dtype=float),
            artifact_state_by_x={
                (0.0, 0.0): make_artifact_state(0.9, 1.7004, success=False),
                (0.9, 0.8): make_artifact_state(
                    0.4,
                    1.69,
                    success=True,
                    curve_curve_min_dist=0.07,
                    max_curvature=39.0,
                ),
                (0.1, 0.2): make_artifact_state(0.6, 1.7002, success=False),
            },
        )

        self._assert_runtime_counts(
            runtime,
            seed_loads=1,
            initialize_calls=0,
            minimize_calls=0,
            minimize_alm_calls=1,
        )
        self.assertEqual(
            runtime["results"]["TERMINATION_MESSAGE"],
            "alm_ok; restored_best_exact_hardware_pass",
        )
        self.assertFalse(runtime["results"]["OPTIMIZER_SUCCESS"])
        self.assertTrue(runtime["results"]["HARDWARE_CONSTRAINTS_OK"])
        self.assertEqual(runtime["results"]["HARDWARE_CONSTRAINT_VIOLATIONS"], [])
        self.assertEqual(runtime["results"]["COIL_LENGTH"], 1.69)

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
        self.assertEqual(runtime["results"]["BANANA_INIT_CURRENT_A"], 9500.0)
        self.assertEqual(runtime["results"]["BANANA_CURRENT_MAX_A"], 1.6e4)
        self.assertIsNotNone(runtime["minimize_bounds"])

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
        self.assertEqual(runtime["results"]["basin_completed_hops"], 2)
        self.assertEqual(runtime["results"]["basin_initial_objective"], 0.55)
        self.assertEqual(runtime["results"]["basin_best_hop_objective"], 0.42)
        self.assertEqual(runtime["results"]["basin_best_hop_index"], 1)
        self.assertEqual(runtime["results"]["basin_best_result_source"], "hop")
        self.assertEqual(runtime["results"]["basin_objective_improvement"], 0.13)
        self.assertEqual(runtime["results"]["basin_nonfinite_rejections"], 0)
        self.assertEqual(runtime["results"]["basin_normalized_step_rejections"], 1)
        self.assertIsNotNone(runtime["basin_bounds"])

    def test_stage2_main_rejects_final_banana_current_above_cap(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=False,
            banana_current_A=17000.0,
        )

        self._assert_banana_current_cap_rejected(runtime)

    def test_stage2_main_rejects_negative_final_banana_current_above_cap_magnitude(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=False,
            banana_current_A=-17000.0,
        )

        self._assert_banana_current_cap_rejected(runtime)

    def test_stage2_main_records_loaded_seed_current_as_initial_current(self):
        runtime = self._run_stage2_main(
            init_only=True,
            constraint_method="penalty",
            use_seed=True,
            banana_current_A=12345.0,
        )

        self.assertEqual(runtime["results"]["BANANA_INIT_CURRENT_A"], 12345.0)

    def test_stage2_main_fresh_init_path_uses_initialize_coils(self):
        runtime = self._run_stage2_main(init_only=True, constraint_method="penalty", use_seed=False)

        self._assert_init_only_runtime_counts(
            runtime,
            seed_loads=0,
            initialize_calls=1,
        )
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
