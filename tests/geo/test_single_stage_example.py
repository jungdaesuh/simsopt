from contextlib import contextmanager
import importlib.util
import os
import json
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt.geo.surfaceobjectives import (
    SurfaceSurfaceDistance,
    boozer_surface_residual,
    boozer_surface_residual_dB,
)
from simsopt.jax_core.specs import (
    CoilDofExtractionSpec,
    CoilSetDofExtractionSpec,
    CoilSymmetrySpec,
    CurveXYZFourierSpec,
    OptimizableDofMapSpec,
)
from simsopt.objectives.utilities import forward_backward


EXAMPLE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)
STAGE2_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "banana_coil_solver.py"
)
TEST_MPOL = 8
TEST_NTOR = 6
TEST_VOL_TARGET = 0.1
TEST_IOTA = 0.15
TEST_G0 = 1.0
_SINGLE_STAGE_JAX_ONLY_ONDEVICE = (
    "the single-stage outer loop with backend='jax' requires "
    "optimizer_backend='ondevice'"
)
_SINGLE_STAGE_CPU_ONLY_SCIPY = (
    "single-stage outer loop CPU/reference lane only supports optimizer_backend='scipy'"
)
_OPTIMIZER_BACKEND_INVALID = "optimizer_backend must be one of: scipy, ondevice."


def load_single_stage_example_module():
    spec = importlib.util.spec_from_file_location(
        f"single_stage_banana_example_{uuid.uuid4().hex}",
        EXAMPLE_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_stage2_module():
    spec = importlib.util.spec_from_file_location(
        f"banana_coil_solver_{uuid.uuid4().hex}",
        STAGE2_MODULE_PATH,
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

    def set_dofs(self, dofs):
        self.dofs = np.asarray(dofs)

    def get_dofs(self):
        return np.asarray(self.dofs)

    def is_self_intersecting(self):
        return False

    def volume(self):
        return 1.0


class FakeVolume:
    def __init__(self, surface):
        self.surface = surface


class FakeBoozerSurface:
    def __init__(
        self, bs, surface, label, targetlabel, constraint_weight, options=None
    ):
        self.bs = bs
        self.surface = surface
        self.label = label
        self.targetlabel = targetlabel
        self.constraint_weight = constraint_weight
        self.options = options or {}
        self.res = {"success": True, "iter": 1, "iota": 0.15, "G": 1.0}

    def run_code(self, iota, G, *, sdofs=None):
        return self.res


class RecordingCPUBoozerSurface(FakeBoozerSurface):
    instances = []

    def __init__(
        self, bs, surface, label, targetlabel, constraint_weight, options=None
    ):
        super().__init__(bs, surface, label, targetlabel, constraint_weight, options)
        self.run_code_calls = []
        RecordingCPUBoozerSurface.instances.append(self)

    def run_code(self, iota, G=None):
        self.run_code_calls.append((iota, G))
        return self.res


class FailingCPUBoozerSurface:
    def __init__(self, *args, **kwargs):
        raise AssertionError("CPU BoozerSurface should not be constructed")


class SingleStageExampleTests(unittest.TestCase):
    def setUp(self):
        FakeSurfaceXYZTensorFourier.instances = []
        RecordingCPUBoozerSurface.instances = []

    @staticmethod
    def _make_candidate_run_dict(sdofs):
        return {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "sdofs": np.asarray(sdofs).copy(),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.zeros(5),
        }

    def load_module(self):
        return load_single_stage_example_module()

    @staticmethod
    def _make_reporting_runtime_summary(*, include_distance_metrics):
        return {
            "solver_success": True,
            "has_G": True,
            "final_G": 1.75,
            "final_non_qs": 0.11,
            "final_boozer_residual": 0.22,
            "final_iota_penalty": 0.33,
            "final_length_penalty": 0.44,
            "final_curve_curve_penalty": 0.55,
            "final_curve_surface_penalty": 0.66,
            "final_surface_vessel_penalty": 0.77,
            "final_curvature_penalty": 0.88,
            "coil_length": 4.25,
            "max_curvature": 12.5,
            "final_volume": 6.5,
            "final_iota": 0.21,
            "curve_curve_min_dist": 0.8 if include_distance_metrics else None,
            "curve_surface_min_dist": 0.9 if include_distance_metrics else None,
            "surface_vessel_min_dist": 1.0 if include_distance_metrics else None,
        }

    @staticmethod
    def _make_reporting_runtime_builder(captured, runtime_summary):
        def _runtime_builder(
            boozer_surface,
            bs,
            iota_target,
            *,
            include_profile_suite=False,
            include_host_wrappers=False,
            outer_objective_config=None,
            success_filter=None,
        ):
            del boozer_surface, bs, iota_target
            captured["include_profile_suite"] = include_profile_suite
            captured["include_host_wrappers"] = include_host_wrappers
            captured["outer_objective_config"] = outer_objective_config
            captured["success_filter"] = success_filter

            def _reporting_metrics(coil_dofs, **kwargs):
                del coil_dofs
                captured["reporting_metrics_kwargs"] = kwargs
                return runtime_summary

            return {"reporting_metrics": _reporting_metrics}

        return _runtime_builder

    def resolve_benchmark_target_lane_sync(
        self,
        module,
        *,
        sync_policy="per-accept",
    ):
        return module.resolve_effective_target_lane_accepted_step_sync(
            sync_policy,
            benchmark_mode=True,
        )

    def initialize_boozer_surface(self, module, surf_prev, *, constraint_weight):
        with patch.object(
            module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier
        ), patch.object(module, "Volume", FakeVolume), patch.object(
            module, "BoozerSurface", FakeBoozerSurface
        ), patch.object(
            module,
            "project_surface_dofs_to_resolution",
            return_value=np.array([1.0], dtype=np.float64),
        ):
            return module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=constraint_weight,
                iota=TEST_IOTA,
                G0=TEST_G0,
            )

    def build_fake_boozer_surface_jax_class(self, *, record_run_calls):
        class FakeBoozerSurfaceJAX:
            instances = []

            def __init__(
                self,
                bs,
                surface,
                label,
                targetlabel,
                constraint_weight,
                options=None,
                surface_runtime_state=None,
            ):
                self.bs = bs
                self.surface = surface
                self.label = label
                self.targetlabel = targetlabel
                self.constraint_weight = constraint_weight
                self.options = options or {}
                self.surface_runtime_state = surface_runtime_state
                self.res = {
                    "success": True,
                    "iter": 1,
                    "iota": TEST_IOTA,
                    "G": TEST_G0,
                }
                self.run_code_calls = [] if record_run_calls else None
                FakeBoozerSurfaceJAX.instances.append(self)

            def run_code(self, iota, G, *, sdofs=None):
                if self.run_code_calls is not None:
                    self.run_code_calls.append((iota, G, sdofs))
                return self.res

        return FakeBoozerSurfaceJAX

    @contextmanager
    def patch_initialize_boozer_surface_jax(self, module, fake_boozer_surface_jax):
        fake_jax_module = types.ModuleType("simsopt.geo.boozersurface_jax")
        fake_jax_module.BoozerSurfaceJAX = fake_boozer_surface_jax
        fake_jax_module.build_boozer_surface_runtime_state = (
            lambda surface: {
                "mpol": surface.mpol,
                "ntor": surface.ntor,
                "nfp": surface.nfp,
                "stellsym": surface.stellsym,
                "quadpoints_phi": np.asarray(surface.quadpoints_phi).copy(),
                "quadpoints_theta": np.asarray(surface.quadpoints_theta).copy(),
            }
        )
        with patch.object(
            module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier
        ), patch.object(module, "Volume", FakeVolume), patch.object(
            module, "BoozerSurface", FailingCPUBoozerSurface
        ), patch.object(
            module,
            "project_surface_dofs_to_resolution",
            return_value=np.array([1.0], dtype=np.float64),
        ), patch.dict(
            sys.modules,
            {"simsopt.geo.boozersurface_jax": fake_jax_module},
        ):
            yield

    @contextmanager
    def patch_optimizer_jax_module(
        self,
        *,
        require_target_backend_x64,
        jax_minimize,
        scipy_minimize_side_effect=None,
    ):
        class ReferenceOptimizerContract:
            def __init__(self, method):
                self.method = method

        class TargetOptimizerContract:
            def __init__(self, method, *, use_least_squares_objective=False):
                self.method = method
                self.use_least_squares_objective = use_least_squares_objective

        def resolve_reference_outer_loop_optimizer_contract(
            field_backend,
            optimizer_backend,
            *,
            component_label,
        ):
            del component_label
            if optimizer_backend not in {"scipy", "ondevice"}:
                raise ValueError(_OPTIMIZER_BACKEND_INVALID)
            if field_backend == "jax":
                raise ValueError(f"the {_SINGLE_STAGE_JAX_ONLY_ONDEVICE}.")
            if optimizer_backend != "scipy":
                raise ValueError(f"the {_SINGLE_STAGE_CPU_ONLY_SCIPY}.")
            return ReferenceOptimizerContract("lbfgs")

        def resolve_target_outer_loop_optimizer_contract(
            field_backend,
            optimizer_backend,
            *,
            component_label,
            least_squares_algorithm="quasi-newton",
        ):
            del component_label, least_squares_algorithm
            if optimizer_backend not in {"scipy", "ondevice"}:
                raise ValueError(_OPTIMIZER_BACKEND_INVALID)
            if field_backend != "jax" or optimizer_backend != "ondevice":
                raise ValueError(f"the {_SINGLE_STAGE_JAX_ONLY_ONDEVICE}.")
            require_target_backend_x64(optimizer_backend)
            return TargetOptimizerContract("lbfgs-ondevice")

        fake_optimizer_module = types.ModuleType("simsopt.geo.optimizer_jax")
        fake_optimizer_module.ReferenceOptimizerContract = ReferenceOptimizerContract
        fake_optimizer_module.TargetOptimizerContract = TargetOptimizerContract
        fake_optimizer_module.require_target_backend_x64 = require_target_backend_x64
        fake_optimizer_module.reference_minimize = jax_minimize
        fake_optimizer_module.target_minimize = jax_minimize
        fake_optimizer_module.resolve_reference_outer_loop_optimizer_contract = (
            resolve_reference_outer_loop_optimizer_contract
        )
        fake_optimizer_module.resolve_target_outer_loop_optimizer_contract = (
            resolve_target_outer_loop_optimizer_contract
        )
        scipy_patch = patch(
            "scipy.optimize.minimize", side_effect=scipy_minimize_side_effect
        )
        with scipy_patch, patch.dict(
            sys.modules, {"simsopt.geo.optimizer_jax": fake_optimizer_module}
        ):
            yield

    @contextmanager
    def patch_surface_self_intersection_backend_unavailable(self, module):
        with patch.object(module.surface_module, "get_context", None), patch.object(
            module.surface_module, "contour_self_intersects", None
        ), patch.object(module.surface_module, "LineString", None, create=True):
            yield

    def test_exact_boozer_helpers_are_imported(self):
        module = self.load_module()

        self.assertIs(module.boozer_surface_residual, boozer_surface_residual)
        self.assertIs(module.boozer_surface_residual_dB, boozer_surface_residual_dB)
        self.assertIs(module.forward_backward, forward_backward)

    def test_initialize_boozer_surface_exact_uses_ntor_phi_quadrature(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        boozer_surface = self.initialize_boozer_surface(
            module, surf_prev, constraint_weight=None
        )

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 2)

        exact_surface = FakeSurfaceXYZTensorFourier.instances[1]
        expected_phi = np.linspace(
            0, 1 / surf_prev.nfp, 2 * TEST_NTOR + 1, endpoint=False
        )

        self.assertEqual(exact_surface.quadpoints_theta.size, 2 * TEST_MPOL + 1)
        self.assertEqual(exact_surface.quadpoints_phi.size, 2 * TEST_NTOR + 1)
        np.testing.assert_allclose(exact_surface.quadpoints_phi, expected_phi)

    def test_initialize_boozer_surface_zero_constraint_weight_keeps_least_squares_path(
        self,
    ):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        boozer_surface = self.initialize_boozer_surface(
            module, surf_prev, constraint_weight=0.0
        )

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 1)
        self.assertIs(boozer_surface.surface, FakeSurfaceXYZTensorFourier.instances[0])

    def test_initialize_boozer_surface_jax_backend_routes_to_jax_solver(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        fake_boozer_surface_jax = self.build_fake_boozer_surface_jax_class(
            record_run_calls=True
        )

        with self.patch_initialize_boozer_surface_jax(module, fake_boozer_surface_jax):
            boozer_surface = module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=1.0,
                iota=TEST_IOTA,
                G0=TEST_G0,
                backend="jax",
            )

        self.assertIsInstance(boozer_surface, fake_boozer_surface_jax)
        self.assertEqual(len(boozer_surface.run_code_calls), 1)
        call_iota, call_G, call_sdofs = boozer_surface.run_code_calls[0]
        self.assertEqual(call_iota, TEST_IOTA)
        self.assertEqual(call_G, TEST_G0)
        self.assertIsNone(call_sdofs)
        self.assertEqual(boozer_surface.constraint_weight, 1.0)
        self.assertEqual(boozer_surface.options["verbose"], True)
        self.assertEqual(boozer_surface.options["optimizer_backend"], "ondevice")
        self.assertIs(boozer_surface.surface, FakeSurfaceXYZTensorFourier.instances[0])
        self.assertEqual(boozer_surface.surface_runtime_state["mpol"], TEST_MPOL)
        self.assertEqual(boozer_surface.surface_runtime_state["ntor"], TEST_NTOR)
        np.testing.assert_allclose(
            boozer_surface.surface_runtime_state["quadpoints_phi"],
            FakeSurfaceXYZTensorFourier.instances[0].quadpoints_phi,
        )

    def test_initialize_boozer_surface_cpu_warm_start_does_not_pass_sdofs(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        surface_override = np.array([4.0, 5.0, 6.0])

        with patch.object(
            module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier
        ), patch.object(module, "Volume", FakeVolume), patch.object(
            module, "BoozerSurface", RecordingCPUBoozerSurface
        ):
            boozer_surface = module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=1.0,
                iota=TEST_IOTA,
                G0=TEST_G0,
                backend="cpu",
                surface_dofs_override=surface_override,
                iota_override=0.21,
                G_override=1.7,
            )

        self.assertIsInstance(boozer_surface, RecordingCPUBoozerSurface)
        self.assertEqual(len(boozer_surface.run_code_calls), 1)
        self.assertEqual(boozer_surface.run_code_calls[0], (0.21, 1.7))
        np.testing.assert_array_equal(
            boozer_surface.surface.get_dofs(), surface_override
        )
        self.assertFalse(hasattr(boozer_surface.surface, "fitted_gamma"))

    def test_initialize_boozer_surface_skips_fit_when_surface_override_present(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        surface_override = np.array([4.0, 5.0, 6.0])

        with patch.object(
            module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier
        ), patch.object(module, "Volume", FakeVolume), patch.object(
            module, "BoozerSurface", RecordingCPUBoozerSurface
        ):
            boozer_surface = module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=1.0,
                iota=TEST_IOTA,
                G0=TEST_G0,
                backend="cpu",
                surface_dofs_override=surface_override,
            )

        self.assertFalse(hasattr(boozer_surface.surface, "fitted_gamma"))
        np.testing.assert_array_equal(
            boozer_surface.surface.get_dofs(),
            surface_override,
        )

    def test_initialize_boozer_surface_threads_nondefault_optimizer_backend(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        fake_boozer_surface_jax = self.build_fake_boozer_surface_jax_class(
            record_run_calls=False
        )

        with self.patch_initialize_boozer_surface_jax(module, fake_boozer_surface_jax):
            module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=1.0,
                iota=TEST_IOTA,
                G0=TEST_G0,
                backend="jax",
                optimizer_backend="ondevice",
            )

        self.assertEqual(len(fake_boozer_surface_jax.instances), 1)
        self.assertEqual(
            fake_boozer_surface_jax.instances[0].options["optimizer_backend"],
            "ondevice",
        )

    def test_initialize_boozer_surface_threads_nondefault_least_squares_algorithm(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        fake_boozer_surface_jax = self.build_fake_boozer_surface_jax_class(
            record_run_calls=False
        )

        with self.patch_initialize_boozer_surface_jax(module, fake_boozer_surface_jax):
            module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=1.0,
                iota=TEST_IOTA,
                G0=TEST_G0,
                backend="jax",
                boozer_least_squares_algorithm="lm",
            )

        self.assertEqual(len(fake_boozer_surface_jax.instances), 1)
        self.assertEqual(
            fake_boozer_surface_jax.instances[0].options["least_squares_algorithm"],
            "lm",
        )

    def test_initialize_boozer_surface_threads_solver_budget_overrides(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        fake_boozer_surface_jax = self.build_fake_boozer_surface_jax_class(
            record_run_calls=False
        )

        with self.patch_initialize_boozer_surface_jax(module, fake_boozer_surface_jax):
            module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=1.0,
                iota=TEST_IOTA,
                G0=TEST_G0,
                backend="jax",
                bfgs_tol_override=3.0e-6,
                bfgs_maxiter_override=32,
                newton_tol_override=1.0e-7,
                newton_maxiter_override=9,
            )

        options = fake_boozer_surface_jax.instances[0].options
        self.assertEqual(options["bfgs_tol"], 3.0e-6)
        self.assertEqual(options["bfgs_maxiter"], 32)
        self.assertEqual(options["newton_tol"], 1.0e-7)
        self.assertEqual(options["newton_maxiter"], 9)

    def test_resolve_warm_start_boozer_init_overrides_is_empty_without_warm_start(self):
        module = self.load_module()

        overrides = module.resolve_warm_start_boozer_init_overrides(
            warm_start_state=None,
            explicit_surface_warm_start=False,
            field_backend="jax",
            optimizer_backend="ondevice",
            boozer_optimizer_backend="ondevice",
            boozer_least_squares_algorithm="lm",
            boozer_least_squares_algorithm_explicit=False,
            target_lane_boozer_bfgs_tol=3.0e-6,
            target_lane_boozer_bfgs_maxiter=32,
        )

        self.assertEqual(
            overrides,
            {
                "least_squares_algorithm_override": None,
                "bfgs_tol_override": None,
                "bfgs_maxiter_override": None,
                "newton_tol_override": None,
                "newton_maxiter_override": None,
            },
        )

    def test_resolve_warm_start_boozer_init_overrides_keeps_explicit_surface_algorithm(self):
        module = self.load_module()

        overrides = module.resolve_warm_start_boozer_init_overrides(
            warm_start_state={"surface": object()},
            explicit_surface_warm_start=True,
            field_backend="jax",
            optimizer_backend="ondevice",
            boozer_optimizer_backend="ondevice",
            boozer_least_squares_algorithm="lm",
            boozer_least_squares_algorithm_explicit=False,
            target_lane_boozer_bfgs_tol=3.0e-6,
            target_lane_boozer_bfgs_maxiter=32,
        )

        self.assertEqual(
            overrides,
            {
                "least_squares_algorithm_override": None,
                "bfgs_tol_override": 1.0e-8,
                "bfgs_maxiter_override": 128,
                "newton_tol_override": None,
                "newton_maxiter_override": None,
            },
        )

    def test_resolve_warm_start_boozer_init_overrides_uses_quasi_newton_for_legacy_path(self):
        module = self.load_module()

        overrides = module.resolve_warm_start_boozer_init_overrides(
            warm_start_state={"surface": object()},
            explicit_surface_warm_start=False,
            field_backend="jax",
            optimizer_backend="ondevice",
            boozer_optimizer_backend="ondevice",
            boozer_least_squares_algorithm="lm",
            boozer_least_squares_algorithm_explicit=False,
            target_lane_boozer_bfgs_tol=3.0e-6,
            target_lane_boozer_bfgs_maxiter=32,
        )

        self.assertEqual(overrides["least_squares_algorithm_override"], "quasi-newton")
        self.assertEqual(overrides["bfgs_tol_override"], 1.0e-8)
        self.assertEqual(overrides["bfgs_maxiter_override"], 128)

    def test_resolve_warm_start_boozer_init_overrides_preserves_explicit_algorithm(self):
        module = self.load_module()

        overrides = module.resolve_warm_start_boozer_init_overrides(
            warm_start_state={"surface": object()},
            explicit_surface_warm_start=False,
            field_backend="jax",
            optimizer_backend="ondevice",
            boozer_optimizer_backend="ondevice",
            boozer_least_squares_algorithm="lm",
            boozer_least_squares_algorithm_explicit=True,
            target_lane_boozer_bfgs_tol=3.0e-6,
            target_lane_boozer_bfgs_maxiter=32,
        )

        self.assertIsNone(overrides["least_squares_algorithm_override"])
        self.assertEqual(overrides["bfgs_tol_override"], 1.0e-8)
        self.assertEqual(overrides["bfgs_maxiter_override"], 128)

    def test_initialize_boozer_surface_emits_stage_callbacks(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        stage_events = []

        class FakeBoozerSurfaceJAX:
            def __init__(
                self,
                bs,
                surface,
                label,
                targetlabel,
                constraint_weight,
                options=None,
                surface_runtime_state=None,
            ):
                del bs, label, targetlabel, constraint_weight, surface_runtime_state
                self.surface = surface
                self.options = options or {}
                self.res = {
                    "success": True,
                    "iter": 3,
                    "iota": TEST_IOTA,
                    "G": TEST_G0,
                }

            def run_code(self, iota, G, *, sdofs=None):
                del iota, G, sdofs
                stage_callback = self.options["stage_callback"]
                stage_callback("before_boozer_lbfgs", method="lbfgs-ondevice")
                stage_callback(
                    "after_boozer_lbfgs",
                    solve_success="true",
                    iterations=2.0,
                    method="lbfgs-ondevice",
                )
                stage_callback(
                    "before_boozer_newton",
                    method="newton-polish",
                    ls_method="lbfgs-ondevice",
                )
                stage_callback(
                    "after_boozer_newton",
                    solve_success="true",
                    iterations=1.0,
                )
                return self.res

        with self.patch_initialize_boozer_surface_jax(module, FakeBoozerSurfaceJAX):
            module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=1.0,
                iota=TEST_IOTA,
                G0=TEST_G0,
                backend="jax",
                on_stage=lambda label, **extra: stage_events.append((label, extra)),
            )

        self.assertEqual(
            [label for label, _ in stage_events],
            [
                "after_boozer_surface_fit",
                "after_boozer_setup",
                "before_boozer_solve",
                "before_boozer_lbfgs",
                "after_boozer_lbfgs",
                "before_boozer_newton",
                "after_boozer_newton",
                "after_boozer_solve",
                "after_boozer_postprocess",
            ],
        )

    def test_initialize_boozer_surface_skips_jax_solver_stage_callback_without_cpu(
        self,
    ):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        stage_events = []

        class FakeBoozerSurfaceJAX:
            instances = []

            def __init__(
                self,
                bs,
                surface,
                label,
                targetlabel,
                constraint_weight,
                options=None,
                surface_runtime_state=None,
            ):
                del bs, label, targetlabel, constraint_weight, surface_runtime_state
                self.surface = surface
                self.options = options or {}
                self.res = {
                    "success": True,
                    "iter": 1,
                    "iota": TEST_IOTA,
                    "G": TEST_G0,
                }
                FakeBoozerSurfaceJAX.instances.append(self)

            def run_code(self, iota, G, *, sdofs=None):
                del iota, G, sdofs
                self.options.get("stage_callback")
                return self.res

        with patch.object(module, "jax_solver_stage_callback_supported", return_value=False):
            with self.patch_initialize_boozer_surface_jax(module, FakeBoozerSurfaceJAX):
                module.initialize_boozer_surface(
                    surf_prev,
                    mpol=TEST_MPOL,
                    ntor=TEST_NTOR,
                    bs=object(),
                    vol_target=TEST_VOL_TARGET,
                    constraint_weight=1.0,
                    iota=TEST_IOTA,
                    G0=TEST_G0,
                    backend="jax",
                    on_stage=lambda label, **extra: stage_events.append((label, extra)),
                )

        self.assertNotIn(
            "stage_callback",
            FakeBoozerSurfaceJAX.instances[0].options,
        )
        self.assertEqual(
            [label for label, _ in stage_events],
            [
                "after_boozer_surface_fit",
                "after_boozer_setup",
                "before_boozer_solve",
                "after_boozer_solve",
                "after_boozer_postprocess",
            ],
        )

    def test_initialize_boozer_surface_force_enables_jax_solver_stage_callback_without_cpu(
        self,
    ):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        stage_events = []

        class FakeBoozerSurfaceJAX:
            instances = []

            def __init__(
                self,
                bs,
                surface,
                label,
                targetlabel,
                constraint_weight,
                options=None,
                surface_runtime_state=None,
            ):
                del bs, label, targetlabel, constraint_weight, surface_runtime_state
                self.surface = surface
                self.options = options or {}
                self.res = {
                    "success": True,
                    "iter": 1,
                    "iota": TEST_IOTA,
                    "G": TEST_G0,
                }
                FakeBoozerSurfaceJAX.instances.append(self)

            def run_code(self, iota, G, *, sdofs=None):
                del iota, G, sdofs
                stage_callback = self.options.get("stage_callback")
                assert stage_callback is not None
                stage_callback("before_boozer_lbfgs", method="lm-ondevice")
                return self.res

        with patch.dict(
            os.environ,
            {"SIMSOPT_FORCE_JAX_SOLVER_STAGE_CALLBACK": "1"},
            clear=False,
        ):
            with patch.object(module, "jax_solver_stage_callback_supported", wraps=module.jax_solver_stage_callback_supported):
                with self.patch_initialize_boozer_surface_jax(module, FakeBoozerSurfaceJAX):
                    module.initialize_boozer_surface(
                        surf_prev,
                        mpol=TEST_MPOL,
                        ntor=TEST_NTOR,
                        bs=object(),
                        vol_target=TEST_VOL_TARGET,
                        constraint_weight=1.0,
                        iota=TEST_IOTA,
                        G0=TEST_G0,
                        backend="jax",
                        on_stage=lambda label, **extra: stage_events.append((label, extra)),
                    )

        self.assertIn(
            "stage_callback",
            FakeBoozerSurfaceJAX.instances[0].options,
        )
        self.assertIn(
            "before_boozer_lbfgs",
            [label for label, _ in stage_events],
        )

    def test_build_stage_progress_recorder_writes_stage_history(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, "boozer_init_progress.json")
            record_stage = module.build_stage_progress_recorder(progress_path)
            record_stage("starting", backend="jax")
            record_stage("after_boozer_setup", boozer_type="ls")

            with open(progress_path, encoding="utf-8") as infile:
                payload = json.load(infile)

        self.assertEqual(payload["current_stage"], "after_boozer_setup")
        self.assertEqual(
            payload["completed_stages"],
            ["starting", "after_boozer_setup"],
        )
        self.assertEqual(payload["stages"]["starting"]["backend"], "jax")
        self.assertEqual(payload["stages"]["after_boozer_setup"]["boozer_type"], "ls")

    def test_resolve_boozer_optimizer_backend_defaults_and_overrides(self):
        module = self.load_module()

        self.assertIsNone(
            module.resolve_boozer_optimizer_backend(
                "cpu",
                "ondevice",
                "scipy",
            )
        )
        self.assertEqual(
            module.resolve_boozer_optimizer_backend("jax", "ondevice", None),
            "ondevice",
        )
        with self.assertRaisesRegex(
            ValueError, "requires boozer_optimizer_backend='ondevice'"
        ):
            module.resolve_boozer_optimizer_backend("jax", "ondevice", "scipy")
        with self.assertRaisesRegex(
            ValueError, "requires boozer_optimizer_backend='ondevice'"
        ):
            module.resolve_boozer_optimizer_backend("jax", "scipy", None)

    def test_resolve_boozer_least_squares_algorithm_defaults_follow_effective_backend(
        self,
    ):
        module = self.load_module()

        self.assertIsNone(
            module.resolve_single_stage_default_boozer_least_squares_algorithm(
                "cpu",
                "scipy",
            )
        )
        self.assertEqual(
            module.resolve_single_stage_default_boozer_least_squares_algorithm(
                "jax",
                "ondevice",
            ),
            "lm",
        )
        with self.assertRaisesRegex(
            ValueError, "requires boozer_optimizer_backend='ondevice'"
        ):
            module.resolve_single_stage_default_boozer_least_squares_algorithm(
                "jax",
                "ondevice",
                "scipy",
            )
        with self.assertRaisesRegex(
            ValueError, "requires boozer_optimizer_backend='ondevice'"
        ):
            module.resolve_single_stage_default_boozer_least_squares_algorithm(
                "jax",
                "scipy",
            )
        self.assertEqual(
            module.resolve_single_stage_default_boozer_least_squares_algorithm(
                "jax",
                "ondevice",
                None,
                "quasi-newton",
            ),
            "quasi-newton",
        )

    def test_parse_args_does_not_treat_optimizer_env_as_explicit_boozer_override(self):
        module = self.load_module()

        with patch.dict(os.environ, {"OPTIMIZER_BACKEND": "ondevice"}), patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py"],
        ):
            args = module.parse_args()

        self.assertEqual(args.optimizer_backend, "ondevice")
        self.assertIsNone(args.boozer_optimizer_backend)
        self.assertIsNone(args.boozer_least_squares_algorithm)

    def test_parse_args_defaults_jax_backend_to_ondevice_optimizer_lane(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py", "--backend", "jax"],
        ):
            args = module.parse_args()

        self.assertEqual(args.optimizer_backend, "ondevice")
        self.assertIsNone(args.boozer_optimizer_backend)
        self.assertEqual(args.boozer_least_squares_algorithm, "lm")
        self.assertFalse(args.boozer_least_squares_algorithm_explicit)

    def test_parse_args_preserves_cpu_default_reference_lane(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py"],
        ):
            args = module.parse_args()

        self.assertEqual(args.backend, "cpu")
        self.assertEqual(args.optimizer_backend, "scipy")
        self.assertIsNone(args.boozer_optimizer_backend)
        self.assertIsNone(args.boozer_least_squares_algorithm)

    def test_parse_args_defaults_boozer_algorithm_from_explicit_inner_backend(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--boozer-optimizer-backend",
                "ondevice",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.optimizer_backend, "ondevice")
        self.assertEqual(args.boozer_optimizer_backend, "ondevice")
        self.assertEqual(args.boozer_least_squares_algorithm, "lm")
        self.assertFalse(args.boozer_least_squares_algorithm_explicit)

    def test_parse_args_defaults_target_lane_sync_to_final_only(self):
        module = self.load_module()

        with patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py"],
        ):
            args = module.parse_args()

        self.assertEqual(args.target_lane_accepted_step_sync, "final-only")
        self.assertFalse(args.profile_target_lane)
        self.assertFalse(args.experimental_target_lane_value_and_grad)
        self.assertFalse(args.disable_target_lane_success_filter)

    def test_parse_args_defaults_target_lane_outer_maxls_to_tighter_budget(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py", "--backend", "jax"],
        ):
            args = module.parse_args()

        self.assertEqual(args.outer_maxls, 8)
        self.assertEqual(args.maxcor, 20)
        self.assertEqual(args.initial_step_scale, 1.0)
        self.assertEqual(args.initial_step_maxiter, 0)
        self.assertEqual(args.target_lane_boozer_bfgs_tol, 1e-8)
        self.assertIsNone(args.target_lane_boozer_bfgs_maxiter)
        self.assertIsNone(args.target_lane_boozer_newton_tol)
        self.assertIsNone(args.target_lane_boozer_newton_maxiter)

    def test_parse_args_benchmark_mode_uses_target_lane_trial_defaults(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--benchmark-mode",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.outer_maxls, 4)
        self.assertEqual(args.target_lane_outer_initial_step_size, 1.0e-4)
        self.assertEqual(args.target_lane_boozer_bfgs_tol, 1e-6)
        self.assertEqual(args.target_lane_boozer_bfgs_maxiter, 64)
        self.assertIsNone(args.target_lane_boozer_newton_tol)
        self.assertIsNone(args.target_lane_boozer_newton_maxiter)

    def test_parse_args_preserves_reference_outer_maxls_default(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py"],
        ):
            args = module.parse_args()

        self.assertEqual(args.outer_maxls, 20)
        self.assertEqual(args.maxcor, 300)
        self.assertIsNone(args.target_lane_outer_initial_step_size)
        self.assertEqual(args.initial_step_scale, 1.0)
        self.assertEqual(args.initial_step_maxiter, 0)
        self.assertIsNone(args.target_lane_boozer_bfgs_tol)
        self.assertIsNone(args.target_lane_boozer_bfgs_maxiter)
        self.assertIsNone(args.target_lane_boozer_newton_tol)
        self.assertIsNone(args.target_lane_boozer_newton_maxiter)

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

        value, grad = scaled_fun(np.array([1.0, -2.0]))
        self.assertAlmostEqual(value, 7.5)
        np.testing.assert_allclose(grad, [0.3, -0.4])
        np.testing.assert_allclose(seen["fun"][0], [10.1, 19.8])

        scaled_callback(np.array([1.0, -2.0]))
        np.testing.assert_allclose(seen["callback"][0], [10.1, 19.8])

    def test_build_scaled_outer_problem_omits_callback_when_base_none(self):
        module = self.load_module()

        def base_fun(x):
            return 7.5, np.array([3.0, -4.0])

        scaled_fun, scaled_callback = module.build_scaled_outer_problem(
            base_fun,
            None,
            np.array([10.0, 20.0]),
            0.1,
        )

        value, grad = scaled_fun(np.array([1.0, -2.0]))

        self.assertIsNone(scaled_callback)
        self.assertAlmostEqual(value, 7.5)
        np.testing.assert_allclose(grad, [0.3, -0.4])

    def test_build_scaled_outer_scalar_problem_scales_coordinates_and_callback(self):
        module = self.load_module()
        seen = {"fun": [], "callback": []}

        def base_fun(x):
            seen["fun"].append(np.asarray(x, dtype=float).copy())
            return float(np.dot(x, x))

        def base_callback(x):
            seen["callback"].append(np.asarray(x, dtype=float).copy())

        scaled_fun, scaled_callback = module.build_scaled_outer_scalar_problem(
            base_fun,
            base_callback,
            np.array([10.0, 20.0]),
            0.25,
        )

        value = scaled_fun(np.array([2.0, -4.0]))
        self.assertAlmostEqual(value, 10.5**2 + 19.0**2)
        np.testing.assert_allclose(seen["fun"][0], [10.5, 19.0])

        scaled_callback(np.array([2.0, -4.0]))
        np.testing.assert_allclose(seen["callback"][0], [10.5, 19.0])

    def test_build_scaled_outer_scalar_problem_omits_callback_when_base_none(self):
        module = self.load_module()

        def base_fun(x):
            return float(np.dot(x, x))

        scaled_fun, scaled_callback = module.build_scaled_outer_scalar_problem(
            base_fun,
            None,
            np.array([10.0, 20.0]),
            0.25,
        )

        self.assertIsNone(scaled_callback)
        self.assertAlmostEqual(scaled_fun(np.array([2.0, -4.0])), 10.5**2 + 19.0**2)

    def test_build_scaled_outer_phase_initial_dofs_target_lane_is_transfer_safe(self):
        module = self.load_module()
        dofs = jax.device_put(np.array([1.0, -2.0, 3.0], dtype=np.float64))

        with jax.transfer_guard("disallow"):
            zeros = module.build_scaled_outer_phase_initial_dofs(
                dofs,
                use_target_lane=True,
            )

        np.testing.assert_allclose(
            np.asarray(jax.device_get(zeros)),
            np.zeros(3, dtype=np.float64),
        )

    def test_resolve_scaled_outer_phase_final_dofs_target_lane_is_transfer_safe(self):
        module = self.load_module()
        anchor_dofs = jax.device_put(np.array([10.0, 20.0], dtype=np.float64))
        step_dofs = np.array([2.0, -4.0], dtype=np.float64)

        with jax.transfer_guard("disallow"):
            final_dofs = module.resolve_scaled_outer_phase_final_dofs(
                anchor_dofs,
                step_dofs,
                0.25,
                use_target_lane=True,
            )

        np.testing.assert_allclose(
            np.asarray(jax.device_get(final_dofs), dtype=np.float64),
            [10.5, 19.0],
        )

    def test_resolve_scaled_outer_phase_final_dofs_target_lane_host_anchor_device_step_is_transfer_safe(
        self,
    ):
        module = self.load_module()
        anchor_dofs = np.array([10.0, 20.0], dtype=np.float64)
        step_dofs = jax.device_put(np.array([2.0, -4.0], dtype=np.float64))

        with jax.transfer_guard("disallow"):
            final_dofs = module.resolve_scaled_outer_phase_final_dofs(
                anchor_dofs,
                step_dofs,
                0.25,
                use_target_lane=True,
            )

        np.testing.assert_allclose(
            np.asarray(jax.device_get(final_dofs), dtype=np.float64),
            [10.5, 19.0],
        )

    def test_resolve_scaled_outer_phase_final_dofs_target_lane_host_anchor_host_step_is_transfer_safe(
        self,
    ):
        module = self.load_module()
        anchor_dofs = np.array([10.0, 20.0], dtype=np.float64)
        step_dofs = np.array([2.0, -4.0], dtype=np.float64)

        with jax.transfer_guard("disallow"):
            final_dofs = module.resolve_scaled_outer_phase_final_dofs(
                anchor_dofs,
                step_dofs,
                0.25,
                use_target_lane=True,
            )

        self.assertIsInstance(final_dofs, jax.Array)
        np.testing.assert_allclose(
            np.asarray(jax.device_get(final_dofs), dtype=np.float64),
            [10.5, 19.0],
        )

    def test_resolve_scaled_outer_phase_final_dofs_target_lane_scaled_state_host_anchor_is_transfer_safe(
        self,
    ):
        module = self.load_module()
        scaled_state = module.ScaledOuterPhaseOptimizerState(
            step_dofs=jax.device_put(np.array([2.0, -4.0], dtype=np.float64)),
            anchor_dofs=np.array([10.0, 20.0], dtype=np.float64),
        )

        with jax.transfer_guard("disallow"):
            final_dofs = module.resolve_scaled_outer_phase_final_dofs(
                np.array([0.0, 0.0], dtype=np.float64),
                scaled_state,
                0.25,
                use_target_lane=True,
            )

        np.testing.assert_allclose(
            np.asarray(jax.device_get(final_dofs), dtype=np.float64),
            [10.5, 19.0],
        )

    def test_resolve_scaled_outer_phase_final_dofs_target_lane_scaled_state_host_anchor_host_step_is_transfer_safe(
        self,
    ):
        module = self.load_module()
        scaled_state = module.ScaledOuterPhaseOptimizerState(
            step_dofs=np.array([2.0, -4.0], dtype=np.float64),
            anchor_dofs=np.array([10.0, 20.0], dtype=np.float64),
        )

        with jax.transfer_guard("disallow"):
            final_dofs = module.resolve_scaled_outer_phase_final_dofs(
                np.array([0.0, 0.0], dtype=np.float64),
                scaled_state,
                0.25,
                use_target_lane=True,
            )

        self.assertIsInstance(final_dofs, jax.Array)
        np.testing.assert_allclose(
            np.asarray(jax.device_get(final_dofs), dtype=np.float64),
            [10.5, 19.0],
        )

    def test_build_scaled_outer_problem_target_lane_is_transfer_safe(self):
        module = self.load_module()
        seen = {"fun": [], "callback": []}

        def base_fun(x):
            seen["fun"].append(np.asarray(jax.device_get(x), dtype=np.float64))
            return jnp.sum(x * x), x + x

        def base_callback(x):
            seen["callback"].append(np.asarray(jax.device_get(x), dtype=np.float64))

        anchor_x = jax.device_put(np.array([10.0, 20.0], dtype=np.float64))
        z = jax.device_put(np.array([2.0, -4.0], dtype=np.float64))

        scaled_fun, scaled_callback = module.build_scaled_outer_problem(
            base_fun,
            base_callback,
            anchor_x,
            0.25,
        )

        with jax.transfer_guard("disallow"):
            value, grad = scaled_fun(z)
            scaled_callback(z)

        self.assertAlmostEqual(float(jax.device_get(value)), 10.5**2 + 19.0**2)
        np.testing.assert_allclose(
            np.asarray(jax.device_get(grad), dtype=np.float64),
            [5.25, 9.5],
        )
        np.testing.assert_allclose(seen["fun"][0], [10.5, 19.0])
        np.testing.assert_allclose(seen["callback"][0], [10.5, 19.0])

    def test_build_scaled_outer_problem_target_lane_jit_is_transfer_safe(self):
        module = self.load_module()

        def base_fun(x):
            return jnp.sum(x * x), x + x

        anchor_x = jax.device_put(np.array([10.0, 20.0], dtype=np.float64))
        z = jax.device_put(np.array([2.0, -4.0], dtype=np.float64))
        scaled_fun, _ = module.build_scaled_outer_problem(
            base_fun,
            None,
            anchor_x,
            0.25,
        )

        with jax.transfer_guard("disallow"):
            value, grad = jax.jit(scaled_fun)(z)

        self.assertAlmostEqual(float(jax.device_get(value)), 10.5**2 + 19.0**2)
        np.testing.assert_allclose(
            np.asarray(jax.device_get(grad), dtype=np.float64),
            [5.25, 9.5],
        )

    def test_build_scaled_outer_problem_scaled_state_target_lane_is_transfer_safe(
        self,
    ):
        module = self.load_module()
        seen = {"callback": []}

        def base_fun(x):
            return jnp.sum(x * x), x + x

        def base_callback(x):
            seen["callback"].append(np.asarray(jax.device_get(x), dtype=np.float64))

        anchor_x = jax.device_put(np.array([10.0, 20.0], dtype=np.float64))
        step_dofs = jax.device_put(np.array([2.0, -4.0], dtype=np.float64))
        scaled_state = module.build_target_lane_scaled_outer_phase_state(
            anchor_x,
            step_dofs,
        )
        scaled_fun, scaled_callback = module.build_scaled_outer_problem(
            base_fun,
            base_callback,
            anchor_x,
            0.25,
        )

        with jax.transfer_guard("disallow"):
            value, grad = scaled_fun(scaled_state)
            scaled_callback(scaled_state)

        self.assertAlmostEqual(float(jax.device_get(value)), 10.5**2 + 19.0**2)
        self.assertIsInstance(grad, module.ScaledOuterPhaseOptimizerState)
        np.testing.assert_allclose(
            np.asarray(jax.device_get(grad.step_dofs), dtype=np.float64),
            [5.25, 9.5],
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(grad.anchor_dofs), dtype=np.float64),
            [0.0, 0.0],
        )
        np.testing.assert_allclose(seen["callback"][0], [10.5, 19.0])

    def test_build_scaled_outer_problem_scaled_state_target_lane_jit_is_transfer_safe(
        self,
    ):
        module = self.load_module()

        def base_fun(x):
            return jnp.sum(x * x), x + x

        anchor_x = jax.device_put(np.array([10.0, 20.0], dtype=np.float64))
        step_dofs = jax.device_put(np.array([2.0, -4.0], dtype=np.float64))
        scaled_state = module.build_target_lane_scaled_outer_phase_state(
            anchor_x,
            step_dofs,
        )
        scaled_fun, _ = module.build_scaled_outer_problem(
            base_fun,
            None,
            anchor_x,
            0.25,
            anchor_in_state=True,
        )

        with jax.transfer_guard("disallow"):
            value, grad = jax.jit(scaled_fun)(scaled_state)

        self.assertAlmostEqual(float(jax.device_get(value)), 10.5**2 + 19.0**2)
        self.assertIsInstance(grad, module.ScaledOuterPhaseOptimizerState)
        np.testing.assert_allclose(
            np.asarray(jax.device_get(grad.step_dofs), dtype=np.float64),
            [5.25, 9.5],
        )
        np.testing.assert_allclose(
            np.asarray(jax.device_get(grad.anchor_dofs), dtype=np.float64),
            [0.0, 0.0],
        )

    def test_build_scaled_outer_scalar_problem_scaled_state_target_lane_jit_is_transfer_safe(
        self,
    ):
        module = self.load_module()

        def base_fun(x):
            return jnp.sum(x * x)

        anchor_x = jax.device_put(np.array([10.0, 20.0], dtype=np.float64))
        step_dofs = jax.device_put(np.array([2.0, -4.0], dtype=np.float64))
        scaled_state = module.build_target_lane_scaled_outer_phase_state(
            anchor_x,
            step_dofs,
        )
        scaled_fun, scaled_callback = module.build_scaled_outer_scalar_problem(
            base_fun,
            lambda _x: None,
            anchor_x,
            0.25,
            anchor_in_state=True,
        )

        with jax.transfer_guard("disallow"):
            value = jax.jit(scaled_fun)(scaled_state)
            scaled_callback(scaled_state)

        self.assertAlmostEqual(float(jax.device_get(value)), 10.5**2 + 19.0**2)

    def test_resolve_single_stage_warm_start_paths_requires_artifacts(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                FileNotFoundError,
                "single-stage warm start run directory is missing required artifacts",
            ):
                module.resolve_single_stage_warm_start_paths(tmpdir)

    def test_load_single_stage_warm_start_state_reads_surface_and_metrics(self):
        module = self.load_module()
        surface_marker = object()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
            (run_dir / "results.json").write_text(
                json.dumps({"FINAL_IOTA": 0.123, "FINAL_G": 4.5}),
                encoding="utf-8",
            )

            with patch.object(module, "load", return_value=surface_marker):
                warm_start = module.load_single_stage_warm_start_state(str(run_dir))

        self.assertIs(warm_start["surface"], surface_marker)
        self.assertEqual(warm_start["iota"], 0.123)
        self.assertEqual(warm_start["G"], 4.5)
        self.assertTrue(warm_start["surface_path"].endswith("surf_opt.json"))
        self.assertTrue(warm_start["results_path"].endswith("results.json"))

    def test_load_single_stage_warm_start_state_reads_serialized_surface_payload(self):
        module = self.load_module()
        surface = module.SurfaceXYZTensorFourier(
            mpol=2,
            ntor=1,
            nfp=5,
            stellsym=True,
            quadpoints_phi=np.linspace(0.0, 0.2, 4, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
        )
        surface_dofs = surface.get_dofs().copy()
        surface_dofs[:] = np.linspace(0.01, 0.01 * surface_dofs.size, surface_dofs.size)
        surface.set_dofs(surface_dofs)

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            surface.save(run_dir / "surf_opt.json")
            (run_dir / "results.json").write_text(
                json.dumps({"FINAL_IOTA": 0.123, "FINAL_G": 4.5}),
                encoding="utf-8",
            )

            with patch.object(
                module,
                "load",
                side_effect=AssertionError("serialized warm start should not call load()"),
            ):
                warm_start = module.load_single_stage_warm_start_state(str(run_dir))

        self.assertIsInstance(warm_start["surface"], module.SerializedSurfaceState)
        self.assertEqual(warm_start["surface"].surface_class, "SurfaceXYZTensorFourier")
        np.testing.assert_allclose(warm_start["surface"].dofs, surface_dofs)
        self.assertEqual(warm_start["iota"], 0.123)
        self.assertEqual(warm_start["G"], 4.5)

    def test_project_single_stage_warm_start_surface_dofs_delegates_to_resolution_projector(
        self,
    ):
        module = self.load_module()
        surface = object()
        quadpoints_phi = np.linspace(0.0, 0.2, 5, endpoint=False)
        quadpoints_theta = np.linspace(0.0, 1.0, 7, endpoint=False)
        with patch.object(
            module,
            "project_surface_dofs_to_resolution",
            return_value=np.array([1.0]),
        ) as projector:
            projected_dofs = module.project_single_stage_warm_start_surface_dofs(
                surface,
                mpol=8,
                ntor=6,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )

        np.testing.assert_allclose(projected_dofs, np.array([1.0]))
        projector.assert_called_once_with(
            surface,
            mpol=8,
            ntor=6,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )

    def test_project_surface_dofs_to_resolution_reprojects_to_target_resolution(self):
        module = self.load_module()
        captured = {}

        class FakeSurface:
            nfp = 5
            stellsym = True

            def gamma(self):
                raise AssertionError("projection should resample on the target grid")

            def cross_section(self, phi, thetas=None):
                theta_values = np.asarray(thetas, dtype=float)
                return np.column_stack(
                    (
                        np.full(theta_values.shape, float(phi)),
                        theta_values,
                        theta_values + float(phi),
                    )
                )

        quadpoints_phi = np.linspace(0.0, 0.2, 5, endpoint=False)
        quadpoints_theta = np.linspace(0.0, 1.0, 7, endpoint=False)

        def _fit_surface_xyz_tensor_dofs_to_gamma(target_gamma, **kwargs):
            captured["target_gamma"] = np.asarray(target_gamma)
            captured["kwargs"] = kwargs
            return np.array([1.0]), True

        with patch.object(
            module,
            "_fit_surface_xyz_tensor_dofs_to_gamma",
            side_effect=_fit_surface_xyz_tensor_dofs_to_gamma,
        ):
            projected_dofs = module.project_surface_dofs_to_resolution(
                FakeSurface(),
                mpol=8,
                ntor=6,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )

        np.testing.assert_allclose(projected_dofs, np.array([1.0]))
        expected_gamma = np.stack(
            [
                np.column_stack(
                    (
                        np.full(quadpoints_theta.shape, float(phi)),
                        quadpoints_theta,
                        quadpoints_theta + float(phi),
                    )
                )
                for phi in quadpoints_phi
            ],
            axis=0,
        )
        np.testing.assert_allclose(captured["target_gamma"], expected_gamma)
        self.assertEqual(captured["kwargs"]["mpol"], 8)
        self.assertEqual(captured["kwargs"]["ntor"], 6)
        self.assertEqual(captured["kwargs"]["nfp"], 5)
        self.assertTrue(captured["kwargs"]["stellsym"])

    def test_project_single_stage_warm_start_surface_dofs_uses_gamma_fast_path_for_xyz_surface(
        self,
    ):
        module = self.load_module()
        captured = {}
        source_surface = module.SurfaceXYZTensorFourier(
            mpol=2,
            ntor=2,
            nfp=5,
            stellsym=True,
            quadpoints_theta=np.linspace(0.0, 1.0, 3, endpoint=False),
            quadpoints_phi=np.linspace(0.0, 0.2, 2, endpoint=False),
        )
        source_dofs = source_surface.get_dofs().copy()
        source_dofs[:] = np.linspace(0.05, 0.05 * source_dofs.size, source_dofs.size)
        source_surface.set_dofs(source_dofs)
        quadpoints_phi = np.linspace(0.0, 0.2, 5, endpoint=False)
        quadpoints_theta = np.linspace(0.0, 1.0, 7, endpoint=False)

        expected_gamma = module.surface_gamma_from_dofs(
            jnp.asarray(source_surface.get_dofs(), dtype=jnp.float64),
            jnp.asarray(quadpoints_phi, dtype=jnp.float64),
            jnp.asarray(quadpoints_theta, dtype=jnp.float64),
            source_surface.mpol,
            source_surface.ntor,
            source_surface.nfp,
            source_surface.stellsym,
            scatter_indices=module.stellsym_scatter_indices(
                source_surface.mpol, source_surface.ntor
            ),
        )

        def _fit_surface_xyz_tensor_dofs_to_gamma(target_gamma, **kwargs):
            captured["target_gamma"] = np.asarray(target_gamma)
            captured["kwargs"] = kwargs
            return np.array([1.0]), True

        with patch.object(
            module,
            "_fit_surface_xyz_tensor_dofs_to_gamma",
            side_effect=_fit_surface_xyz_tensor_dofs_to_gamma,
        ):
            projected_dofs = module.project_single_stage_warm_start_surface_dofs(
                source_surface,
                mpol=8,
                ntor=6,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )

        np.testing.assert_allclose(projected_dofs, np.array([1.0]))
        np.testing.assert_allclose(captured["target_gamma"], np.asarray(expected_gamma))
        self.assertEqual(captured["kwargs"]["mpol"], 8)
        self.assertEqual(captured["kwargs"]["ntor"], 6)

    def test_project_surface_dofs_to_resolution_uses_dof_fast_path_for_rz_surface(
        self,
    ):
        module = self.load_module()
        captured = {}
        source_surface = module.SurfaceRZFourier(
            mpol=2,
            ntor=1,
            nfp=5,
            stellsym=True,
            quadpoints_phi=np.linspace(0.0, 0.2, 4, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
        )
        source_dofs = source_surface.get_dofs().copy()
        source_dofs[:] = np.linspace(0.03, 0.03 * source_dofs.size, source_dofs.size)
        source_surface.set_dofs(source_dofs)
        quadpoints_phi = np.linspace(0.0, 0.2, 6, endpoint=False)
        quadpoints_theta = np.linspace(0.0, 1.0, 7, endpoint=False)

        source_spec = module.surface_rz_fourier_spec_from_dofs(
            jnp.asarray(source_surface.get_dofs(), dtype=jnp.float64),
            quadpoints_phi=jnp.asarray(quadpoints_phi, dtype=jnp.float64),
            quadpoints_theta=jnp.asarray(quadpoints_theta, dtype=jnp.float64),
            mpol=source_surface.mpol,
            ntor=source_surface.ntor,
            nfp=source_surface.nfp,
            stellsym=source_surface.stellsym,
        )
        expected_gamma = module.surface_rz_fourier_gamma_from_dofs(
            source_spec,
            jnp.asarray(source_surface.get_dofs(), dtype=jnp.float64),
        )

        def _fit_surface_xyz_tensor_dofs_to_gamma(target_gamma, **kwargs):
            captured["target_gamma"] = np.asarray(target_gamma)
            captured["kwargs"] = kwargs
            return np.array([2.0]), True

        with patch.object(
            module,
            "_fit_surface_xyz_tensor_dofs_to_gamma",
            side_effect=_fit_surface_xyz_tensor_dofs_to_gamma,
        ):
            projected_dofs = module.project_surface_dofs_to_resolution(
                source_surface,
                mpol=8,
                ntor=6,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )

        np.testing.assert_allclose(projected_dofs, np.array([2.0]))
        np.testing.assert_allclose(captured["target_gamma"], np.asarray(expected_gamma))
        self.assertEqual(captured["kwargs"]["nfp"], source_surface.nfp)
        self.assertTrue(captured["kwargs"]["stellsym"])

    def test_project_surface_dofs_to_resolution_matches_host_for_serialized_xyz_surface(
        self,
    ):
        module = self.load_module()
        source_surface = module.SurfaceXYZTensorFourier(
            mpol=2,
            ntor=2,
            nfp=5,
            stellsym=True,
            quadpoints_phi=np.linspace(0.0, 0.2, 7, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 9, endpoint=False),
        )
        source_dofs = source_surface.get_dofs().copy()
        source_dofs[:] = np.linspace(
            0.02,
            0.02 * source_dofs.size,
            source_dofs.size,
        )
        source_surface.set_dofs(source_dofs)
        quadpoints_phi = np.linspace(0.0, 0.2, 11, endpoint=False)
        quadpoints_theta = np.linspace(0.0, 1.0, 13, endpoint=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            surface_path = Path(tmpdir) / "surf_opt.json"
            source_surface.save(surface_path)
            serialized_surface = module.load_serialized_surface_state(surface_path)

        host_projected_dofs = module.project_surface_dofs_to_resolution(
            source_surface,
            mpol=4,
            ntor=3,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        serialized_projected_dofs = module.project_surface_dofs_to_resolution(
            serialized_surface,
            mpol=4,
            ntor=3,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )

        np.testing.assert_allclose(
            serialized_projected_dofs,
            host_projected_dofs,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_project_surface_dofs_to_resolution_matches_host_for_rank_deficient_serialized_xyz_surface(
        self,
    ):
        module = self.load_module()
        source_surface = module.SurfaceXYZTensorFourier(
            mpol=2,
            ntor=2,
            nfp=5,
            stellsym=False,
            quadpoints_phi=np.linspace(0.0, 1.0, 7, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 9, endpoint=False),
        )
        source_dofs = source_surface.get_dofs().copy()
        source_dofs[:] = np.linspace(
            0.02,
            0.02 * source_dofs.size,
            source_dofs.size,
        )
        source_surface.set_dofs(source_dofs)
        quadpoints_phi = np.linspace(0.0, 1.0, 5, endpoint=False)
        quadpoints_theta = np.linspace(0.0, 1.0, 7, endpoint=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            surface_path = Path(tmpdir) / "surf_opt.json"
            source_surface.save(surface_path)
            serialized_surface = module.load_serialized_surface_state(surface_path)

        host_projected_dofs = module.project_surface_dofs_to_resolution(
            source_surface,
            mpol=1,
            ntor=1,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        serialized_projected_dofs = module.project_surface_dofs_to_resolution(
            serialized_surface,
            mpol=1,
            ntor=1,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )

        np.testing.assert_allclose(
            serialized_projected_dofs,
            host_projected_dofs,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_project_surface_dofs_to_resolution_matches_host_for_rz_nonstellsym_alias_convention(
        self,
    ):
        module = self.load_module()
        source_surface = module.SurfaceRZFourier(
            mpol=2,
            ntor=2,
            nfp=3,
            stellsym=False,
            quadpoints_phi=np.linspace(0.0, 1.0, 8, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 7, endpoint=False),
        )
        source_dofs = source_surface.get_dofs().copy()
        source_dofs[:] = np.linspace(
            0.015,
            0.015 * source_dofs.size,
            source_dofs.size,
        )
        source_surface.set_dofs(source_dofs)
        quadpoints_phi = np.linspace(0.0, 1.0, 9, endpoint=False)
        quadpoints_theta = np.linspace(0.0, 1.0, 8, endpoint=False)

        projected_dofs = module.project_surface_dofs_to_resolution(
            source_surface,
            mpol=3,
            ntor=2,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )

        legacy_surface = module.SurfaceXYZTensorFourier(
            mpol=3,
            ntor=2,
            nfp=source_surface.nfp,
            stellsym=source_surface.stellsym,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        target_gamma = np.stack(
            [
                np.asarray(
                    source_surface.cross_section(float(phi), thetas=quadpoints_theta),
                    dtype=float,
                )
                for phi in np.asarray(quadpoints_phi, dtype=float)
            ],
            axis=0,
        )
        legacy_surface.least_squares_fit(target_gamma)

        np.testing.assert_allclose(
            projected_dofs,
            np.asarray(legacy_surface.get_dofs(), dtype=float),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_target_lane_hardware_success_filter_keeps_closure_constants_on_host(self):
        module = self.load_module()
        banana_curve = object()

        class FakeBS:
            def __init__(self):
                self.coils = [types.SimpleNamespace(curve=banana_curve)]

            def coil_dof_extraction_spec(self):
                return {
                    "offsets": jax.device_put(np.asarray([1.0, 2.0], dtype=np.float64))
                }

        class FakeBoozerSurface:
            def __init__(self):
                self.res = {"G": None}
                self.surface = types.SimpleNamespace(
                    mpol=2,
                    ntor=2,
                    nfp=5,
                    stellsym=True,
                    quadpoints_phi=np.linspace(0.0, 0.1, 3, endpoint=False),
                    quadpoints_theta=np.linspace(0.0, 1.0, 4, endpoint=False),
                )
                self.quadpoints_phi = jax.device_put(
                    np.linspace(0.0, 0.1, 3, endpoint=False, dtype=np.float64)
                )
                self.quadpoints_theta = jax.device_put(
                    np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64)
                )
                self.mpol = 2
                self.ntor = 2
                self.nfp = 5
                self.stellsym = True
                self.scatter_indices = jax.device_put(np.asarray([0], dtype=np.int32))
                self._surface_geometry_kind = "rzfourier"

            def _unpack_decision_vector_jax(self, solved_x, optimize_G, coil_set_spec=None):
                del solved_x, optimize_G, coil_set_spec
                return (
                    jax.device_put(np.zeros(2, dtype=np.float64)),
                    jax.device_put(np.asarray(0.0, dtype=np.float64)),
                    None,
                )

        vessel_surface = types.SimpleNamespace(surface_spec=lambda: {"rc": 1.0})
        with patch(
            "simsopt.jax_core.surface_rzfourier.surface_rz_fourier_gamma_from_spec",
            return_value=jax.device_put(np.zeros((2, 3, 3), dtype=np.float64)),
        ):
            success_filter = module.build_single_stage_target_lane_hardware_success_filter(
                FakeBoozerSurface(),
                FakeBS(),
                banana_curve,
                vessel_surface,
                cc_dist=0.05,
                cs_dist=0.05,
                ss_dist=0.05,
                curvature_threshold=40.0,
            )

        def contains_jax_array(value):
            return any(
                isinstance(leaf, jax.Array)
                for leaf in jax.tree_util.tree_leaves(value)
            )

        captured_values = [
            cell.cell_contents for cell in (success_filter.__closure__ or ())
        ]
        self.assertFalse(any(contains_jax_array(value) for value in captured_values))
        self.assertIsInstance(
            getattr(success_filter, "_traceable_runtime_cache_signature", None),
            tuple,
        )

    def test_target_lane_hardware_success_filter_signature_is_stable_for_equivalent_filters(
        self,
    ):
        module = self.load_module()
        banana_curve = object()

        class FakeBS:
            def __init__(self):
                self.coils = [types.SimpleNamespace(curve=banana_curve)]

            def coil_dof_extraction_spec(self):
                return {"offsets": np.asarray([1.0, 2.0], dtype=np.float64)}

        class FakeBoozerSurface:
            def __init__(self):
                self.res = {"G": None}
                self.surface = types.SimpleNamespace(
                    mpol=2,
                    ntor=2,
                    nfp=5,
                    stellsym=True,
                    quadpoints_phi=np.linspace(0.0, 0.1, 3, endpoint=False),
                    quadpoints_theta=np.linspace(0.0, 1.0, 4, endpoint=False),
                )
                self.quadpoints_phi = np.linspace(0.0, 0.1, 3, endpoint=False)
                self.quadpoints_theta = np.linspace(0.0, 1.0, 4, endpoint=False)
                self.mpol = 2
                self.ntor = 2
                self.nfp = 5
                self.stellsym = True
                self.scatter_indices = np.asarray([0], dtype=np.int32)
                self._surface_geometry_kind = "rzfourier"

            def _unpack_decision_vector_jax(self, solved_x, optimize_G, coil_set_spec=None):
                del solved_x, optimize_G, coil_set_spec
                return (
                    jax.device_put(np.zeros(2, dtype=np.float64)),
                    jax.device_put(np.asarray(0.0, dtype=np.float64)),
                    None,
                )

        vessel_surface = types.SimpleNamespace(surface_spec=lambda: {"rc": 1.0})
        with patch(
            "simsopt.jax_core.surface_rzfourier.surface_rz_fourier_gamma_from_spec",
            return_value=np.zeros((2, 3, 3), dtype=np.float64),
        ):
            success_filter_a = module.build_single_stage_target_lane_hardware_success_filter(
                FakeBoozerSurface(),
                FakeBS(),
                banana_curve,
                vessel_surface,
                cc_dist=0.05,
                cs_dist=0.05,
                ss_dist=0.05,
                curvature_threshold=40.0,
            )
            success_filter_b = module.build_single_stage_target_lane_hardware_success_filter(
                FakeBoozerSurface(),
                FakeBS(),
                banana_curve,
                vessel_surface,
                cc_dist=0.05,
                cs_dist=0.05,
                ss_dist=0.05,
                curvature_threshold=40.0,
            )

        self.assertEqual(
            success_filter_a._traceable_runtime_cache_signature,
            success_filter_b._traceable_runtime_cache_signature,
        )

    def test_profile_target_lane_only_forces_profile_collection(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--profile-target-lane-only",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.profile_target_lane_only)
        self.assertTrue(args.profile_target_lane)

    def test_parse_args_accepts_diagnose_target_lane_gradient(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--diagnose-target-lane-gradient",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.diagnose_target_lane_gradient)
        self.assertFalse(args.profile_target_lane_only)

    def test_parse_args_accepts_diagnose_target_lane_scaled_phase1(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--diagnose-target-lane-scaled-phase1",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.diagnose_target_lane_scaled_phase1)
        self.assertFalse(args.profile_target_lane_only)

    def test_parse_args_accepts_record_target_lane_invalid_state_events(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--record-target-lane-invalid-state-events",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.record_target_lane_invalid_state_events)

    def test_parse_args_accepts_minimal_artifacts(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py", "--minimal-artifacts"],
        ):
            args = module.parse_args()

        self.assertTrue(args.minimal_artifacts)

    def test_extract_optimizer_diagnostics_flags_nonfinite_state(self):
        module = self.load_module()
        result = types.SimpleNamespace(
            fun=np.inf,
            jac=np.asarray([1.0, np.nan], dtype=np.float64),
            x=np.asarray([0.0, 1.0], dtype=np.float64),
        )

        diagnostics = module.extract_optimizer_diagnostics(result, ran_optimizer=True)

        self.assertIsNone(diagnostics["fun"])
        self.assertFalse(diagnostics["fun_finite"])
        self.assertFalse(diagnostics["jac_finite"])
        self.assertIsNone(diagnostics["jac_inf_norm"])
        self.assertTrue(diagnostics["x_finite"])
        self.assertTrue(diagnostics["invalid_state"])

    def test_extract_optimizer_diagnostics_uses_nonfinite_message_fallback(self):
        module = self.load_module()

        diagnostics = module.extract_optimizer_diagnostics(
            None,
            ran_optimizer=True,
            termination_message="Optimization failed with non-finite objective or gradient.",
        )

        self.assertIsNone(diagnostics["fun"])
        self.assertFalse(diagnostics["fun_finite"])
        self.assertFalse(diagnostics["jac_finite"])
        self.assertIsNone(diagnostics["jac_inf_norm"])
        self.assertIsNone(diagnostics["x_finite"])
        self.assertTrue(diagnostics["invalid_state"])

    def test_build_target_lane_invalid_state_failure_callback_records_payload(self):
        module = self.load_module()
        events = []

        callback = module.build_target_lane_invalid_state_failure_callback(
            events,
            phase="phase2",
        )
        callback(
            3,
            np.array([0.0, 1.0], dtype=np.float64),
            np.nan,
            np.array([1.0, np.nan], dtype=np.float64),
            np.array([-1.0, 2.0], dtype=np.float64),
            np.array([-0.5, 1.0], dtype=np.float64),
            0.5,
            False,
            True,
            False,
            True,
            False,
            0,
        )

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["phase"], "phase2")
        self.assertEqual(event["iteration"], 3)
        self.assertEqual(event["step_scale"]["value"], 0.5)
        self.assertEqual(event["trial_value"]["classification"], "nan")
        self.assertFalse(event["trial_grad"]["all_finite"])
        self.assertEqual(event["trial_grad"]["first_nonfinite_index"], 1)
        self.assertEqual(event["search_direction"]["values"], [-1.0, 2.0])
        self.assertEqual(event["step_vector"]["values"], [-0.5, 1.0])

    def test_record_target_lane_invalid_state_events_enabled_defaults_false(self):
        module = self.load_module()

        self.assertFalse(
            module.record_target_lane_invalid_state_events_enabled(
                types.SimpleNamespace()
            )
        )

    def test_resolve_target_lane_invalid_state_failure_callback_requires_opt_in(self):
        module = self.load_module()
        events = []

        callback = module.resolve_target_lane_invalid_state_failure_callback(
            events,
            phase="phase2",
            use_target_lane=True,
            args=types.SimpleNamespace(record_target_lane_invalid_state_events=False),
        )

        self.assertIsNone(callback)

    def test_resolve_target_lane_invalid_state_failure_callback_builds_callback_when_enabled(
        self,
    ):
        module = self.load_module()
        events = []

        callback = module.resolve_target_lane_invalid_state_failure_callback(
            events,
            phase="phase2",
            use_target_lane=True,
            args=types.SimpleNamespace(record_target_lane_invalid_state_events=True),
        )

        self.assertIsNotNone(callback)
        callback(
            1,
            np.array([0.0], dtype=np.float64),
            1.0,
            np.array([0.5], dtype=np.float64),
            np.array([-0.25], dtype=np.float64),
            np.array([-0.25], dtype=np.float64),
            0.5,
            False,
            False,
            False,
            True,
            False,
            0,
        )
        self.assertEqual(len(events), 1)

    def test_resolve_single_stage_outer_maxls_rejects_nonpositive_budget(self):
        module = self.load_module()

        with self.assertRaisesRegex(ValueError, "outer_maxls must be at least 1"):
            module.resolve_single_stage_outer_maxls("jax", "ondevice", 0)

    def test_resolve_single_stage_outer_maxls_uses_benchmark_budget_for_target_lane(self):
        module = self.load_module()

        self.assertEqual(
            module.resolve_single_stage_outer_maxls(
                "jax",
                "ondevice",
                benchmark_mode=True,
            ),
            4,
        )

    def test_resolve_target_lane_outer_initial_step_size_uses_benchmark_default(self):
        module = self.load_module()

        self.assertEqual(
            module.resolve_target_lane_outer_initial_step_size(
                "jax",
                "ondevice",
                benchmark_mode=True,
            ),
            1.0e-4,
        )
        self.assertIsNone(
            module.resolve_target_lane_outer_initial_step_size(
                "cpu",
                "scipy",
                benchmark_mode=False,
            )
        )

    def test_resolve_target_lane_outer_initial_step_size_rejects_nonpositive_override(
        self,
    ):
        module = self.load_module()

        with self.assertRaisesRegex(
            ValueError,
            "target_lane_outer_initial_step_size must be positive",
        ):
            module.resolve_target_lane_outer_initial_step_size(
                "jax",
                "ondevice",
                0.0,
            )

    def test_resolve_single_stage_outer_maxcor_rejects_nonpositive_budget(self):
        module = self.load_module()

        with self.assertRaisesRegex(ValueError, "maxcor must be at least 1"):
            module.resolve_single_stage_outer_maxcor("jax", "ondevice", 0)

    def test_resolve_target_lane_boozer_bfgs_tol_rejects_nonpositive_override(self):
        module = self.load_module()

        with self.assertRaisesRegex(
            ValueError, "target_lane_boozer_bfgs_tol must be positive"
        ):
            module.resolve_target_lane_boozer_bfgs_tol("jax", "ondevice", 0.0)

    def test_resolve_target_lane_boozer_bfgs_maxiter_rejects_nonpositive_override(
        self,
    ):
        module = self.load_module()

        with self.assertRaisesRegex(
            ValueError, "target_lane_boozer_bfgs_maxiter must be at least 1"
        ):
            module.resolve_target_lane_boozer_bfgs_maxiter("jax", "ondevice", 0)

    def test_resolve_target_lane_boozer_newton_tol_rejects_nonpositive_override(self):
        module = self.load_module()

        with self.assertRaisesRegex(
            ValueError, "target_lane_boozer_newton_tol must be positive"
        ):
            module.resolve_target_lane_boozer_newton_tol("jax", "ondevice", 0.0)

    def test_resolve_target_lane_boozer_newton_maxiter_rejects_nonpositive_override(
        self,
    ):
        module = self.load_module()

        with self.assertRaisesRegex(
            ValueError, "target_lane_boozer_newton_maxiter must be at least 1"
        ):
            module.resolve_target_lane_boozer_newton_maxiter("jax", "ondevice", 0)

    def test_should_write_single_stage_full_artifacts_respects_minimal_mode(self):
        module = self.load_module()

        self.assertTrue(
            module.should_write_single_stage_full_artifacts(False, False)
        )
        self.assertFalse(
            module.should_write_single_stage_full_artifacts(False, True)
        )
        self.assertFalse(
            module.should_write_single_stage_full_artifacts(True, False)
        )
        self.assertFalse(
            module.should_write_single_stage_full_artifacts(True, True)
        )

    def test_should_write_single_stage_restart_artifacts_only_skips_benchmark(self):
        module = self.load_module()

        self.assertTrue(module.should_write_single_stage_restart_artifacts(False))
        self.assertFalse(module.should_write_single_stage_restart_artifacts(True))

    def test_temporary_boozer_surface_option_overrides_restores_original_values(self):
        module = self.load_module()
        boozer_surface = types.SimpleNamespace(options={"bfgs_tol": 1e-10, "verbose": True})

        with module.temporary_boozer_surface_option_overrides(
            boozer_surface,
            bfgs_tol=1e-8,
            verbose=None,
        ):
            self.assertEqual(boozer_surface.options["bfgs_tol"], 1e-8)
            self.assertEqual(boozer_surface.options["verbose"], True)

        self.assertEqual(boozer_surface.options["bfgs_tol"], 1e-10)
        self.assertEqual(boozer_surface.options["verbose"], True)

    def test_parse_args_accepts_boozer_least_squares_algorithm_override(self):
        module = self.load_module()

        with patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--boozer-least-squares-algorithm",
                "lm",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.boozer_least_squares_algorithm, "lm")
        self.assertTrue(args.boozer_least_squares_algorithm_explicit)

    def test_use_experimental_target_lane_value_and_grad_only_on_jax_ondevice(self):
        module = self.load_module()

        self.assertFalse(
            module.use_experimental_target_lane_value_and_grad(
                backend="cpu",
                optimizer_backend=None,
                enabled=True,
            )
        )
        self.assertFalse(
            module.use_experimental_target_lane_value_and_grad(
                backend="jax",
                optimizer_backend="scipy",
                enabled=True,
            )
        )
        self.assertFalse(
            module.use_experimental_target_lane_value_and_grad(
                backend="jax",
                optimizer_backend="ondevice",
                enabled=False,
            )
        )
        self.assertTrue(
            module.use_experimental_target_lane_value_and_grad(
                backend="jax",
                optimizer_backend="ondevice",
                enabled=True,
            )
        )

    def test_build_target_lane_outer_objectives_uses_runtime_bundle_for_target_lane(
        self,
    ):
        module = self.load_module()
        value_and_grad_marker = object()
        runtime_calls = []

        def _scalar_builder(*args):
            raise AssertionError(
                "default target-lane build should not recreate the scalar closure"
            )

        def _runtime_builder(
            *args,
            include_profile_suite=False,
            include_host_wrappers=False,
            outer_objective_config=None,
            success_filter=None,
        ):
            runtime_calls.append(
                (
                    include_profile_suite,
                    include_host_wrappers,
                    outer_objective_config,
                    success_filter,
                )
            )
            return {
                "objective": object(),
                "value_and_grad": value_and_grad_marker,
            }

        with patch.object(
            module,
            "get_traceable_single_stage_objective_builder",
            return_value=_scalar_builder,
        ), patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=_runtime_builder,
        ):
            scalar_fun, value_and_grad_fun, target_lane_profile = (
                module.build_target_lane_outer_objectives(
                    object(),
                    object(),
                    object(),
                    use_value_and_grad=True,
                    profile_target_lane=False,
                    outer_objective_config=None,
                )
            )

        self.assertIsNone(scalar_fun)
        self.assertIs(value_and_grad_fun, value_and_grad_marker)
        self.assertIsNone(target_lane_profile)
        self.assertEqual(runtime_calls, [(False, False, None, None)])

    def test_build_target_lane_outer_objectives_threads_runtime_bundle_options(
        self,
    ):
        module = self.load_module()
        objective_marker = object()
        outer_objective_config_marker = object()
        success_filter_marker = object()
        runtime_calls = []

        def _runtime_builder(
            *args,
            include_profile_suite=False,
            include_host_wrappers=False,
            outer_objective_config=None,
            success_filter=None,
        ):
            runtime_calls.append(
                (
                    include_profile_suite,
                    include_host_wrappers,
                    outer_objective_config,
                    success_filter,
                )
            )
            return {
                "objective": objective_marker,
                "value_and_grad": object(),
            }

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=_runtime_builder,
        ):
            scalar_fun, value_and_grad_fun, target_lane_profile = (
                module.build_target_lane_outer_objectives(
                    object(),
                    object(),
                    object(),
                    use_value_and_grad=False,
                    profile_target_lane=False,
                    outer_objective_config=outer_objective_config_marker,
                    success_filter=success_filter_marker,
                )
            )

        self.assertIs(scalar_fun, objective_marker)
        self.assertIsNone(value_and_grad_fun)
        self.assertIsNone(target_lane_profile)
        self.assertEqual(
            runtime_calls,
            [
                (
                    False,
                    False,
                    outer_objective_config_marker,
                    success_filter_marker,
                )
            ],
        )

    def test_resolve_single_stage_final_penalty_metrics_prefers_target_lane_runtime_summary(
        self,
    ):
        module = self.load_module()
        captured = {}
        runtime_summary = self._make_reporting_runtime_summary(
            include_distance_metrics=True
        )

        class RejectingPenalty:
            def J(self):
                raise AssertionError("host-side penalty wrapper should not be used")

        class RejectingDistance(RejectingPenalty):
            def shortest_distance(self):
                raise AssertionError("host-side distance wrapper should not be used")

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=self._make_reporting_runtime_builder(captured, runtime_summary),
        ):
            metrics = module.resolve_single_stage_final_penalty_metrics(
                use_target_lane=True,
                benchmark_mode=False,
                skip_outer_optimizer=False,
                boozer_surface=object(),
                bs=object(),
                iota_target=0.21,
                coil_dofs=jax.device_put(np.array([1.0, -2.0], dtype=np.float64)),
                outer_objective_config="config-marker",
                success_filter="success-filter-marker",
                curvelength=RejectingPenalty(),
                j_non_qs=RejectingPenalty(),
                j_boozer_residual=RejectingPenalty(),
                j_iota=RejectingPenalty(),
                j_curve_length=RejectingPenalty(),
                j_curve_curve=RejectingDistance(),
                j_curve_surface=RejectingDistance(),
                j_surface_surface=RejectingDistance(),
                j_curvature=RejectingPenalty(),
                cc_dist=0.05,
                cs_dist=0.02,
                ss_dist=0.04,
                curvature_threshold=40.0,
                init_only=False,
                termination_message="ok",
                optimizer_success=True,
            )

        self.assertEqual(captured["include_profile_suite"], False)
        self.assertEqual(captured["include_host_wrappers"], False)
        self.assertEqual(captured["outer_objective_config"], "config-marker")
        self.assertEqual(captured["success_filter"], "success-filter-marker")
        self.assertEqual(
            captured["reporting_metrics_kwargs"],
            {"include_distance_metrics": True},
        )
        for metric_name, expected_value in runtime_summary.items():
            self.assertEqual(metrics[metric_name], expected_value)
        self.assertTrue(metrics["hardware_status"]["success"])
        self.assertEqual(metrics["hardware_status"]["violations"], [])

    def test_resolve_single_stage_final_penalty_metrics_skips_target_lane_distances_in_benchmark_mode(
        self,
    ):
        module = self.load_module()
        captured = {}
        runtime_summary = self._make_reporting_runtime_summary(
            include_distance_metrics=False
        )

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=self._make_reporting_runtime_builder(captured, runtime_summary),
        ):
            metrics = module.resolve_single_stage_final_penalty_metrics(
                use_target_lane=True,
                benchmark_mode=True,
                skip_outer_optimizer=False,
                boozer_surface=object(),
                bs=object(),
                iota_target=0.21,
                coil_dofs=jax.device_put(np.array([1.0, -2.0], dtype=np.float64)),
                outer_objective_config="config-marker",
                success_filter="success-filter-marker",
                curvelength=object(),
                j_non_qs=object(),
                j_boozer_residual=object(),
                j_iota=object(),
                j_curve_length=object(),
                j_curve_curve=object(),
                j_curve_surface=object(),
                j_surface_surface=object(),
                j_curvature=object(),
                cc_dist=0.05,
                cs_dist=0.02,
                ss_dist=0.04,
                curvature_threshold=40.0,
            )

        self.assertEqual(
            captured["reporting_metrics_kwargs"],
            {"include_distance_metrics": False},
        )
        self.assertIsNone(metrics["curve_curve_min_dist"])
        self.assertIsNone(metrics["curve_surface_min_dist"])
        self.assertIsNone(metrics["surface_vessel_min_dist"])
        self.assertIsNone(metrics["hardware_status"]["success"])
        self.assertEqual(
            metrics["hardware_status"]["violations"],
            ["skipped_in_benchmark_mode"],
        )

    def test_build_single_stage_target_lane_accepted_step_sync_uses_pure_reporting_metrics(
        self,
    ):
        module = self.load_module()
        captured = {}
        runtime_summary = self._make_reporting_runtime_summary(
            include_distance_metrics=True
        )
        fake_boozer_surface = types.SimpleNamespace(
            run_code_traceable=lambda *_args: {
                "success": jnp.asarray(True, dtype=bool),
                "sdofs": jnp.asarray([0.4, -0.2], dtype=jnp.float64),
                "iota": jnp.asarray(0.21, dtype=jnp.float64),
                "G": jnp.asarray(1.75, dtype=jnp.float64),
            }
        )
        fake_bs = types.SimpleNamespace(coil_set_spec_from_dofs=lambda coil_dofs: coil_dofs)

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=self._make_reporting_runtime_builder(captured, runtime_summary),
        ), patch.object(module, "CC_DIST", 0.05, create=True), patch.object(
            module, "CS_DIST", 0.02, create=True
        ), patch.object(module, "SS_DIST", 0.04, create=True), patch.object(
            module, "CURVATURE_THRESHOLD", 40.0, create=True
        ):
            sync = module.build_single_stage_target_lane_accepted_step_sync(
                fake_boozer_surface,
                fake_bs,
                0.21,
                outer_objective_config="config-marker",
                success_filter="success-filter-marker",
            )
            run_dict = {
                "sdofs": np.array([0.1, -0.05], dtype=np.float64),
                "iota": 0.2,
                "G": 1.0,
                "J": 1.0,
                "dJ": np.zeros(2, dtype=np.float64),
            }
            summary = sync(
                run_dict,
                jax.device_put(np.array([1.0, -2.0], dtype=np.float64)),
                benchmark_mode=False,
            )

        self.assertEqual(captured["include_profile_suite"], False)
        self.assertEqual(captured["include_host_wrappers"], False)
        self.assertEqual(captured["outer_objective_config"], "config-marker")
        self.assertEqual(captured["success_filter"], "success-filter-marker")
        self.assertEqual(
            captured["reporting_metrics_kwargs"],
            {"include_distance_metrics": True},
        )
        self.assertIn("objective_value", summary)
        self.assertEqual(
            summary["reporting_metrics"]["final_non_qs"],
            runtime_summary["final_non_qs"],
        )
        self.assertTrue(run_dict["hardware_constraint_status"]["success"])

    def test_build_target_lane_outer_objectives_profiles_with_jax_coil_dofs(self):
        module = self.load_module()
        bs = types.SimpleNamespace(x=np.array([1.0, -2.0], dtype=np.float64))
        profiled_calls = []

        def _runtime_builder(
            *args,
            include_profile_suite=False,
            include_host_wrappers=False,
            outer_objective_config=None,
            success_filter=None,
        ):
            self.assertFalse(include_host_wrappers)
            return {
                "objective": object(),
                "value_and_grad": object(),
                "profile_suite": "profile-suite-marker",
            }

        def _profile(profile_suite, coil_dofs):
            profiled_calls.append((profile_suite, coil_dofs))
            return {"ok": True}

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=_runtime_builder,
        ), patch.object(
            module,
            "profile_traceable_target_lane_objective",
            side_effect=_profile,
        ):
            _, _, target_lane_profile = module.build_target_lane_outer_objectives(
                object(),
                bs,
                object(),
                use_value_and_grad=True,
                profile_target_lane=True,
                profile_batch_size=1,
                outer_objective_config=None,
            )

        self.assertEqual(target_lane_profile["ok"], True)
        self.assertEqual(len(profiled_calls), 1)
        self.assertEqual(profiled_calls[0][0], "profile-suite-marker")
        self.assertIsInstance(profiled_calls[0][1], jax.Array)
        self.assertNotEqual(
            tuple(np.asarray(profiled_calls[0][1], dtype=np.float64)),
            tuple(bs.x),
        )
        self.assertEqual(target_lane_profile["profile_point_kind"], "baseline_perturbed")

    def test_build_target_lane_profile_coil_dofs_avoids_exact_baseline_fast_path(self):
        module = self.load_module()

        profiled = module.build_target_lane_profile_coil_dofs(
            np.array([0.0, -2.0], dtype=np.float64)
        )

        self.assertIsInstance(profiled, jax.Array)
        np.testing.assert_allclose(
            np.asarray(profiled, dtype=np.float64)[1:],
            np.array([-2.0], dtype=np.float64),
        )
        self.assertNotEqual(float(np.asarray(profiled, dtype=np.float64)[0]), 0.0)

    def test_build_target_lane_profile_batch_coil_dofs_returns_perturbed_batch(self):
        module = self.load_module()

        profiled_batch = module.build_target_lane_profile_batch_coil_dofs(
            np.array([0.0, -2.0], dtype=np.float64),
            batch_size=3,
        )

        self.assertIsInstance(profiled_batch, jax.Array)
        host_batch = np.asarray(profiled_batch, dtype=np.float64)
        self.assertEqual(host_batch.shape, (3, 2))
        self.assertTrue(np.all(np.isfinite(host_batch)))
        self.assertTrue(np.all(host_batch[:, 1] != 0.0))
        self.assertTrue(
            any(
                not np.array_equal(row, np.array([0.0, -2.0]))
                for row in host_batch
            )
        )

    def test_build_target_lane_outer_objectives_profiles_seed_batch_when_requested(
        self,
    ):
        module = self.load_module()
        bs = types.SimpleNamespace(x=np.array([1.0, -2.0], dtype=np.float64))
        batched_calls = []

        def _runtime_builder(
            *args,
            include_profile_suite=False,
            include_host_wrappers=False,
            outer_objective_config=None,
            success_filter=None,
        ):
            self.assertFalse(include_host_wrappers)
            return {
                "objective": object(),
                "value_and_grad": object(),
                "profile_suite": "profile-suite-marker",
            }

        def _profile(_profile_suite, _coil_dofs):
            return {"ok": True}

        def _profile_batch(profile_suite, coil_dofs_batch):
            batched_calls.append((profile_suite, coil_dofs_batch))
            return {"batch_ok": True}

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=_runtime_builder,
        ), patch.object(
            module,
            "profile_traceable_target_lane_objective",
            side_effect=_profile,
        ), patch.object(
            module,
            "profile_traceable_target_lane_seed_batch",
            side_effect=_profile_batch,
        ):
            _, _, target_lane_profile = module.build_target_lane_outer_objectives(
                object(),
                bs,
                object(),
                use_value_and_grad=True,
                profile_target_lane=True,
                profile_batch_size=3,
                outer_objective_config=None,
            )

        self.assertEqual(target_lane_profile["batched_seed_profile"]["batch_ok"], True)
        self.assertEqual(
            target_lane_profile["batched_seed_profile"]["profile_point_kind"],
            "baseline_perturbed_batch",
        )
        self.assertEqual(len(batched_calls), 1)
        self.assertEqual(batched_calls[0][0], "profile-suite-marker")
        self.assertIsInstance(batched_calls[0][1], jax.Array)
        self.assertEqual(np.asarray(batched_calls[0][1]).shape, (3, 2))

    def test_prepare_target_lane_outer_objectives_still_builds_objective_when_filter_disabled(
        self,
    ):
        module = self.load_module()
        objective_marker = object()
        profile_marker = object()

        with patch.object(
            module,
            "build_single_stage_target_lane_hardware_success_filter",
            side_effect=AssertionError(
                "success filter should not be built when explicitly disabled"
            ),
        ), patch.object(
            module,
            "build_target_lane_outer_objective_config",
            return_value="config-marker",
        ) as build_objective_config, patch.object(
            module,
            "build_target_lane_outer_objectives",
            return_value=(objective_marker, None, profile_marker),
        ) as build_objectives:
            (
                scalar_fun,
                value_and_grad_fun,
                target_lane_profile,
                success_filter,
            ) = module.prepare_target_lane_outer_objectives(
                object(),
                object(),
                object(),
                object(),
                object(),
                use_target_lane=True,
                use_value_and_grad=False,
                profile_target_lane=True,
                profile_batch_size=3,
                disable_success_filter=True,
                non_qs_weight=1.0,
                residual_weight=10.0,
                iota_weight=20.0,
                length_weight=30.0,
                length_target=4.0,
                cc_dist=0.05,
                cc_weight=40.0,
                cs_dist=0.01,
                cs_weight=50.0,
                ss_dist=0.02,
                surf_dist_weight=60.0,
                curvature_threshold=40.0,
                curvature_weight=70.0,
            )

        self.assertIs(scalar_fun, objective_marker)
        self.assertIsNone(value_and_grad_fun)
        self.assertIs(target_lane_profile, profile_marker)
        self.assertIsNone(success_filter)
        build_objective_config.assert_called_once()
        build_objectives.assert_called_once_with(
            unittest.mock.ANY,
            unittest.mock.ANY,
            unittest.mock.ANY,
            use_value_and_grad=False,
            profile_target_lane=True,
            profile_batch_size=3,
            outer_objective_config="config-marker",
            success_filter=None,
        )

    def test_prepare_target_lane_outer_objectives_threads_enabled_success_filter(
        self,
    ):
        module = self.load_module()
        success_filter_marker = object()
        value_and_grad_marker = object()

        with patch.object(
            module,
            "build_single_stage_target_lane_hardware_success_filter",
            return_value=success_filter_marker,
        ) as build_success_filter, patch.object(
            module,
            "build_target_lane_outer_objective_config",
            return_value="config-marker",
        ) as build_objective_config, patch.object(
            module,
            "build_target_lane_outer_objectives",
            return_value=(None, value_and_grad_marker, None),
        ) as build_objectives:
            (
                scalar_fun,
                value_and_grad_fun,
                target_lane_profile,
                success_filter,
            ) = module.prepare_target_lane_outer_objectives(
                object(),
                object(),
                object(),
                object(),
                object(),
                use_target_lane=True,
                use_value_and_grad=True,
                profile_target_lane=False,
                profile_batch_size=1,
                disable_success_filter=False,
                non_qs_weight=1.0,
                residual_weight=10.0,
                iota_weight=20.0,
                length_weight=30.0,
                length_target=4.0,
                cc_dist=0.05,
                cc_weight=40.0,
                cs_dist=0.01,
                cs_weight=50.0,
                ss_dist=0.02,
                surf_dist_weight=60.0,
                curvature_threshold=40.0,
                curvature_weight=70.0,
            )

        self.assertIsNone(scalar_fun)
        self.assertIs(value_and_grad_fun, value_and_grad_marker)
        self.assertIsNone(target_lane_profile)
        self.assertIs(success_filter, success_filter_marker)
        build_success_filter.assert_called_once()
        build_objective_config.assert_called_once()
        build_objectives.assert_called_once_with(
            unittest.mock.ANY,
            unittest.mock.ANY,
            unittest.mock.ANY,
            use_value_and_grad=True,
            profile_target_lane=False,
            profile_batch_size=1,
            outer_objective_config="config-marker",
            success_filter=success_filter_marker,
        )

    def test_build_target_lane_gradient_diagnosis_threads_config_and_filter(self):
        module = self.load_module()
        success_filter_marker = object()
        diagnostic_builder_calls = []

        def _diagnostic_builder(
            *args,
            outer_objective_config=None,
            success_filter=None,
        ):
            diagnostic_builder_calls.append(
                (outer_objective_config, success_filter)
            )
            return {
                "all_finite": False,
                "first_nonfinite_term": "curve_surface",
            }

        with patch.object(
            module,
            "build_target_lane_outer_objective_config",
            return_value="config-marker",
        ) as build_config, patch.object(
            module,
            "get_traceable_single_stage_runtime_diagnostic_builder",
            return_value=_diagnostic_builder,
        ):
            diagnosis = module.build_target_lane_gradient_diagnosis(
                object(),
                object(),
                object(),
                object(),
                object(),
                success_filter=success_filter_marker,
                non_qs_weight=1.0,
                residual_weight=10.0,
                iota_weight=20.0,
                length_weight=30.0,
                length_target=4.0,
                cc_dist=0.05,
                cc_weight=40.0,
                cs_dist=0.01,
                cs_weight=50.0,
                ss_dist=0.02,
                surf_dist_weight=60.0,
                curvature_threshold=40.0,
                curvature_weight=70.0,
            )

        self.assertEqual(
            diagnosis,
            {"all_finite": False, "first_nonfinite_term": "curve_surface"},
        )
        build_config.assert_called_once()
        self.assertEqual(
            diagnostic_builder_calls,
            [("config-marker", success_filter_marker)],
        )

    def test_build_target_lane_scaled_phase1_diagnosis_threads_runtime_and_optimizer(
        self,
    ):
        module = self.load_module()
        success_filter_marker = object()
        runtime_builder_calls = []
        optimizer_calls = []

        def _runtime_builder(
            *args,
            include_host_wrappers=False,
            outer_objective_config=None,
            success_filter=None,
        ):
            runtime_builder_calls.append(
                (include_host_wrappers, outer_objective_config, success_filter)
            )
            self.assertFalse(include_host_wrappers)

            def _value_and_grad(x):
                x = np.asarray(x, dtype=np.float64)
                return np.dot(x, x), 2.0 * x

            return {
                "value_and_grad": _value_and_grad,
            }

        def _run_single_stage_optimizer(
            fun,
            dofs,
            *,
            contract,
            maxiter,
            ftol,
            gtol,
            maxcor,
            outer_maxls,
            callback,
            scalar_fun,
        ):
            recorded_dofs = getattr(dofs, "step_dofs", dofs)
            optimizer_calls.append(
                {
                    "fun": fun,
                    "dofs": np.asarray(recorded_dofs, dtype=np.float64),
                    "contract_method": contract.method,
                    "maxiter": maxiter,
                    "ftol": ftol,
                    "gtol": gtol,
                    "maxcor": maxcor,
                    "outer_maxls": outer_maxls,
                    "callback": callback,
                    "scalar_fun": scalar_fun,
                }
            )
            return types.SimpleNamespace(
                x=np.array([0.2, -0.4], dtype=np.float64),
                nit=1,
                success=False,
                message="phase1 failed",
                status=5,
                nfev=3,
                njev=3,
                ls_status=2,
            )

        with patch.object(
            module,
            "build_target_lane_outer_objective_config",
            return_value="config-marker",
        ), patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=_runtime_builder,
        ), patch.object(
            module,
            "build_scaled_outer_problem",
            return_value=("scaled-fun", "scaled-callback"),
        ) as build_scaled_problem, patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=_run_single_stage_optimizer,
        ):
            diagnosis = module.build_target_lane_scaled_phase1_diagnosis(
                object(),
                object(),
                object(),
                object(),
                object(),
                anchor_dofs=np.array([1.0, -2.0], dtype=np.float64),
                contract=types.SimpleNamespace(method="lbfgs-ondevice"),
                phase1_maxiter=4,
                step_scale=0.25,
                ftol=1e-8,
                gtol=1e-6,
                maxcor=9,
                outer_maxls=7,
                callback="callback-marker",
                success_filter=success_filter_marker,
                non_qs_weight=1.0,
                residual_weight=10.0,
                iota_weight=20.0,
                length_weight=30.0,
                length_target=4.0,
                cc_dist=0.05,
                cc_weight=40.0,
                cs_dist=0.01,
                cs_weight=50.0,
                ss_dist=0.02,
                surf_dist_weight=60.0,
                curvature_threshold=40.0,
                curvature_weight=70.0,
            )

        build_scaled_problem.assert_called_once()
        self.assertEqual(
            runtime_builder_calls,
            [(False, "config-marker", success_filter_marker)],
        )
        self.assertEqual(len(optimizer_calls), 1)
        self.assertEqual(optimizer_calls[0]["contract_method"], "lbfgs-ondevice")
        self.assertEqual(optimizer_calls[0]["maxiter"], 4)
        self.assertEqual(optimizer_calls[0]["callback"], "scaled-callback")
        self.assertIsNone(optimizer_calls[0]["scalar_fun"])
        self.assertEqual(diagnosis["contract_method"], "lbfgs-ondevice")
        self.assertTrue(diagnosis["all_finite"])
        self.assertIsNone(diagnosis["first_nonfinite_stage"])
        self.assertEqual(diagnosis["optimizer"]["message"], "phase1 failed")
        self.assertEqual(diagnosis["optimizer"]["status"], 5)
        np.testing.assert_allclose(
            diagnosis["anchor"]["mapped_coil_dofs"],
            np.array([1.0, -2.0], dtype=np.float64),
        )
        np.testing.assert_allclose(
            diagnosis["scaled_origin"]["scaled_dofs"],
            np.zeros(2, dtype=np.float64),
        )
        np.testing.assert_allclose(
            diagnosis["steepest_descent_trial"]["scaled_dofs"],
            np.array([-0.5, 1.0], dtype=np.float64),
        )
        np.testing.assert_allclose(
            diagnosis["optimizer_mapped_state"]["mapped_coil_dofs"],
            np.array([1.05, -2.1], dtype=np.float64),
        )

    def test_build_target_lane_scaled_phase1_diagnosis_is_transfer_safe(self):
        module = self.load_module()
        value_and_grad_calls = []

        def _runtime_builder(
            *args,
            include_host_wrappers=False,
            outer_objective_config=None,
            success_filter=None,
        ):
            del args, outer_objective_config, success_filter
            self.assertFalse(include_host_wrappers)

            def _value_and_grad(x):
                value_and_grad_calls.append(x)
                self.assertIsInstance(x, jax.Array)
                x_host = np.asarray(jax.device_get(x), dtype=np.float64)
                return np.dot(x_host, x_host), 2.0 * x_host

            return {
                "value_and_grad": _value_and_grad,
            }

        def _run_single_stage_optimizer(
            fun,
            dofs,
            *,
            contract,
            maxiter,
            ftol,
            gtol,
            maxcor,
            outer_maxls,
            callback,
            scalar_fun,
        ):
            del (
                fun,
                dofs,
                contract,
                maxiter,
                ftol,
                gtol,
                maxcor,
                outer_maxls,
                callback,
                scalar_fun,
            )
            return types.SimpleNamespace(
                x=jax.device_put(np.array([0.2, -0.4], dtype=np.float64)),
                nit=1,
                success=True,
                message="ok",
                status=0,
                nfev=1,
                njev=1,
                ls_status=0,
            )

        with patch.object(
            module,
            "build_target_lane_outer_objective_config",
            return_value="config-marker",
        ), patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=_runtime_builder,
        ), patch.object(
            module,
            "build_scaled_outer_problem",
            return_value=("scaled-fun", None),
        ), patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=_run_single_stage_optimizer,
        ):
            with jax.transfer_guard("disallow"):
                diagnosis = module.build_target_lane_scaled_phase1_diagnosis(
                    object(),
                    object(),
                    object(),
                    object(),
                    object(),
                    anchor_dofs=jax.device_put(np.array([1.0, -2.0], dtype=np.float64)),
                    contract=types.SimpleNamespace(method="lbfgs-ondevice"),
                    phase1_maxiter=4,
                    step_scale=0.25,
                    ftol=1e-8,
                    gtol=1e-6,
                    maxcor=9,
                    outer_maxls=7,
                    callback=None,
                    success_filter=None,
                    non_qs_weight=1.0,
                    residual_weight=10.0,
                    iota_weight=20.0,
                    length_weight=30.0,
                    length_target=4.0,
                    cc_dist=0.05,
                    cc_weight=40.0,
                    cs_dist=0.01,
                    cs_weight=50.0,
                    ss_dist=0.02,
                    surf_dist_weight=60.0,
                    curvature_threshold=40.0,
                    curvature_weight=70.0,
                )

        self.assertTrue(diagnosis["all_finite"])
        self.assertGreaterEqual(len(value_and_grad_calls), 4)

    def test_build_target_lane_scaled_phase1_diagnosis_writes_incremental_checkpoints(
        self,
    ):
        module = self.load_module()
        checkpoint_payloads = []

        def _runtime_builder(
            *args,
            include_host_wrappers=False,
            outer_objective_config=None,
            success_filter=None,
        ):
            del args, outer_objective_config, success_filter
            self.assertFalse(include_host_wrappers)

            def _value_and_grad(x):
                x = np.asarray(x, dtype=np.float64)
                return np.dot(x, x), 2.0 * x

            return {
                "value_and_grad": _value_and_grad,
            }

        def _run_single_stage_optimizer(
            fun,
            dofs,
            *,
            contract,
            maxiter,
            ftol,
            gtol,
            maxcor,
            outer_maxls,
            callback,
            scalar_fun,
        ):
            del (
                fun,
                dofs,
                contract,
                maxiter,
                ftol,
                gtol,
                maxcor,
                outer_maxls,
                callback,
                scalar_fun,
            )
            return types.SimpleNamespace(
                x=np.array([0.2, -0.4], dtype=np.float64),
                nit=1,
                success=False,
                message="phase1 failed",
                status=5,
                nfev=3,
                njev=3,
                ls_status=2,
            )

        def _write_json_file(path, payload):
            checkpoint_payloads.append(
                (
                    path,
                    json.loads(json.dumps(module.sanitize_json_payload(payload))),
                )
            )

        with patch.object(
            module,
            "build_target_lane_outer_objective_config",
            return_value="config-marker",
        ), patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=_runtime_builder,
        ), patch.object(
            module,
            "build_scaled_outer_problem",
            return_value=("scaled-fun", "scaled-callback"),
        ), patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=_run_single_stage_optimizer,
        ), patch.object(
            module,
            "write_json_file",
            side_effect=_write_json_file,
        ):
            diagnosis = module.build_target_lane_scaled_phase1_diagnosis(
                object(),
                object(),
                object(),
                object(),
                object(),
                anchor_dofs=np.array([1.0, -2.0], dtype=np.float64),
                contract=types.SimpleNamespace(method="lbfgs-ondevice"),
                phase1_maxiter=4,
                step_scale=0.25,
                ftol=1e-8,
                gtol=1e-6,
                maxcor=9,
                outer_maxls=7,
                callback="callback-marker",
                success_filter=None,
                non_qs_weight=1.0,
                residual_weight=10.0,
                iota_weight=20.0,
                length_weight=30.0,
                length_target=4.0,
                cc_dist=0.05,
                cc_weight=40.0,
                cs_dist=0.01,
                cs_weight=50.0,
                ss_dist=0.02,
                surf_dist_weight=60.0,
                curvature_threshold=40.0,
                curvature_weight=70.0,
                checkpoint_path="/tmp/target_lane_scaled_phase1_diagnosis.json",
            )

        self.assertEqual(
            [payload["checkpoint_stage"] for _, payload in checkpoint_payloads],
            [
                "starting",
                "runtime_bundle_ready",
                "anchor",
                "scaled_origin",
                "steepest_descent_trial",
                "scaled_origin_after_trial",
                "optimizer_finished",
                "optimizer_scaled_state",
                "optimizer_mapped_state",
                "scaled_origin_after_optimizer",
                "completed",
            ],
        )
        final_payload = checkpoint_payloads[-1][1]
        self.assertTrue(final_payload["diagnosis_complete"])
        self.assertTrue(final_payload["all_finite"])
        self.assertEqual(
            final_payload["completed_stages"][-1],
            "scaled_origin_after_optimizer",
        )
        self.assertEqual(
            final_payload["optimizer"]["message"],
            "phase1 failed",
        )
        self.assertEqual(final_payload["optimizer"]["status"], 5)
        np.testing.assert_allclose(
            final_payload["optimizer_mapped_state"]["mapped_coil_dofs"],
            diagnosis["optimizer_mapped_state"]["mapped_coil_dofs"],
        )

    def test_resolve_effective_target_lane_sync_forces_final_only_in_benchmark_mode(
        self,
    ):
        module = self.load_module()

        self.assertEqual(
            module.resolve_effective_target_lane_accepted_step_sync(
                "per-accept",
                benchmark_mode=True,
            ),
            "final-only",
        )
        self.assertEqual(
            module.resolve_effective_target_lane_accepted_step_sync(
                "final-only",
                benchmark_mode=False,
            ),
            "final-only",
        )

    def test_resolve_target_lane_accepted_step_sync_record_only_when_effective(self):
        module = self.load_module()

        self.assertIsNone(
            module.resolve_target_lane_accepted_step_sync_record(
                backend="cpu",
                optimizer_backend=None,
                maxiter=10,
                sync_policy="final-only",
            )
        )
        self.assertIsNone(
            module.resolve_target_lane_accepted_step_sync_record(
                backend="jax",
                optimizer_backend="scipy",
                maxiter=10,
                sync_policy="final-only",
            )
        )
        self.assertIsNone(
            module.resolve_target_lane_accepted_step_sync_record(
                backend="jax",
                optimizer_backend="ondevice",
                maxiter=0,
                sync_policy="final-only",
            )
        )
        self.assertEqual(
            module.resolve_target_lane_accepted_step_sync_record(
                backend="jax",
                optimizer_backend="ondevice",
                maxiter=3,
                sync_policy="final-only",
            ),
            "final-only",
        )
        self.assertEqual(
            module.resolve_target_lane_accepted_step_sync_record(
                backend="jax",
                optimizer_backend="ondevice",
                maxiter=3,
                sync_policy=self.resolve_benchmark_target_lane_sync(module),
            ),
            "final-only",
        )

    def test_resolve_target_lane_accepted_step_callback_skips_per_accept_in_benchmark_mode(
        self,
    ):
        module = self.load_module()
        adapter = types.SimpleNamespace(callback=object())

        callback = module.resolve_target_lane_accepted_step_callback(
            adapter,
            use_target_lane=True,
            sync_policy=self.resolve_benchmark_target_lane_sync(module),
        )

        self.assertIsNone(callback)

    def test_should_force_strict_target_lane_final_sync(self):
        module = self.load_module()

        self.assertFalse(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=False,
                res_nit=3,
                accepted_step_callback=None,
                trial_boozer_override_active=True,
            )
        )
        self.assertFalse(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=True,
                res_nit=0,
                accepted_step_callback=None,
                trial_boozer_override_active=True,
            )
        )
        self.assertFalse(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=True,
                res_nit=2,
                accepted_step_callback=object(),
                trial_boozer_override_active=False,
            )
        )
        self.assertTrue(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=True,
                res_nit=2,
                accepted_step_callback=None,
                trial_boozer_override_active=False,
            )
        )
        self.assertTrue(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=True,
                res_nit=2,
                accepted_step_callback=object(),
                trial_boozer_override_active=True,
            )
        )

    def test_run_single_stage_optimizer_prefers_fused_target_lane_contract(self):
        module = self.load_module()
        captured = {}
        explicit_fun = lambda x: (
            jnp.asarray(jnp.dot(x, x), dtype=jnp.float64),
            jnp.asarray(2.0 * x, dtype=jnp.float64),
        )
        scalar_fun = lambda x: float(np.dot(x, x))

        def fake_require_target_backend_x64(optimizer_backend):
            captured["x64_backend"] = optimizer_backend

        def fake_jax_minimize(
            fun, x0, *, method, tol, maxiter, options, value_and_grad, callback
        ):
            captured["fun"] = fun
            captured["method"] = method
            captured["x0"] = x0
            captured["tol"] = tol
            captured["maxiter"] = maxiter
            captured["options"] = dict(options)
            captured["value_and_grad"] = value_and_grad
            captured["callback"] = callback
            return types.SimpleNamespace(x=x0, nit=0, message="ok")

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=fake_require_target_backend_x64,
            jax_minimize=fake_jax_minimize,
        ):
            callback = object()
            contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
            result = module.run_single_stage_optimizer(
                explicit_fun,
                np.array([1.0, -2.0]),
                contract=contract,
                maxiter=7,
                ftol=1e-8,
                gtol=1e-6,
                maxcor=9,
                outer_maxls=7,
                callback=callback,
                scalar_fun=scalar_fun,
            )

        self.assertEqual(captured["x64_backend"], "ondevice")
        self.assertIsInstance(captured["x0"], jax.Array)
        self.assertEqual(captured["method"], "lbfgs-ondevice")
        np.testing.assert_allclose(
            module._single_stage_optimizer_dofs_array(captured["x0"]),
            np.array([1.0, -2.0]),
        )
        self.assertEqual(captured["tol"], 1e-6)
        self.assertEqual(captured["maxiter"], 7)
        self.assertEqual(captured["options"], {"maxcor": 9, "ftol": 1e-8, "maxls": 7})
        self.assertTrue(captured["value_and_grad"])
        self.assertIs(captured["callback"], callback)
        value, grad = captured["fun"](captured["x0"])
        self.assertEqual(value, 5.0)
        self.assertIsInstance(grad, jax.Array)
        np.testing.assert_allclose(
            module._single_stage_optimizer_dofs_array(grad),
            np.array([2.0, -4.0]),
        )
        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_threads_target_lane_failure_callback(self):
        module = self.load_module()
        captured = {}
        explicit_fun = lambda x: (
            jnp.asarray(jnp.dot(x, x), dtype=jnp.float64),
            jnp.asarray(2.0 * x, dtype=jnp.float64),
        )

        def fake_require_target_backend_x64(_optimizer_backend):
            return None

        def fake_jax_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad,
            callback,
            failure_callback=None,
        ):
            del fun, x0, method, tol, maxiter, options, value_and_grad, callback
            captured["failure_callback"] = failure_callback
            return types.SimpleNamespace(x=np.zeros(2), nit=0, message="ok")

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=fake_require_target_backend_x64,
            jax_minimize=fake_jax_minimize,
        ):
            failure_callback = object()
            contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
            result = module.run_single_stage_optimizer(
                explicit_fun,
                np.array([0.0, 0.0]),
                contract=contract,
                maxiter=1,
                ftol=0.0,
                gtol=1e-6,
                maxcor=5,
                outer_maxls=6,
                callback=None,
                scalar_fun=None,
                failure_callback=failure_callback,
            )

        self.assertIs(captured["failure_callback"], failure_callback)
        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_threads_target_lane_initial_step_size(self):
        module = self.load_module()
        captured = {}
        explicit_fun = lambda x: (
            jnp.asarray(jnp.dot(x, x), dtype=jnp.float64),
            jnp.asarray(2.0 * x, dtype=jnp.float64),
        )

        def fake_require_target_backend_x64(_optimizer_backend):
            return None

        def fake_jax_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad,
            callback,
            failure_callback=None,
        ):
            del fun, x0, method, tol, maxiter, value_and_grad, callback, failure_callback
            captured["options"] = dict(options)
            return types.SimpleNamespace(x=np.zeros(2), nit=0, message="ok")

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=fake_require_target_backend_x64,
            jax_minimize=fake_jax_minimize,
        ):
            contract = module.resolve_single_stage_optimizer_contract(
                "jax", "ondevice"
            )
            result = module.run_single_stage_optimizer(
                explicit_fun,
                np.array([0.0, 0.0]),
                contract=contract,
                maxiter=1,
                ftol=0.0,
                gtol=1e-6,
                maxcor=5,
                outer_maxls=6,
                callback=None,
                scalar_fun=None,
                target_lane_initial_step_size=1.0e-4,
            )

        self.assertEqual(captured["options"]["initial_step_size"], 1.0e-4)
        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_target_lane_requires_objective_contract(self):
        module = self.load_module()

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=lambda _optimizer_backend: None,
            jax_minimize=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("jax_minimize should not run without an objective")
            ),
        ):
            target_contract = module.resolve_single_stage_optimizer_contract(
                "jax", "ondevice"
            )
            with self.assertRaisesRegex(
                RuntimeError,
                "Single-stage target-lane optimization requires either the fused "
                "value-and-gradient objective or a scalar JAX objective",
            ):
                module.run_single_stage_optimizer(
                    None,
                    np.array([1.0, -2.0]),
                    contract=target_contract,
                    maxiter=7,
                    ftol=1e-8,
                    gtol=1e-6,
                    maxcor=9,
                    outer_maxls=7,
                    callback=None,
                )

    def test_run_single_stage_optimizer_ondevice_does_not_enter_scipy_minimize(self):
        module = self.load_module()
        explicit_fun = lambda x: (
            jnp.asarray(jnp.dot(x, x), dtype=jnp.float64),
            jnp.asarray(2.0 * x, dtype=jnp.float64),
        )

        def fake_require_target_backend_x64(_optimizer_backend):
            return None

        def fake_jax_minimize(
            fun, x0, *, method, tol, maxiter, options, value_and_grad, callback
        ):
            value, grad = fun(x0)
            self.assertEqual(value, 0.0)
            self.assertIsInstance(grad, jax.Array)
            np.testing.assert_allclose(
                module._single_stage_optimizer_dofs_array(grad),
                np.zeros(2),
            )
            self.assertIsInstance(x0, jax.Array)
            del x0, tol, maxiter, options, callback
            self.assertEqual(method, "lbfgs-ondevice")
            self.assertTrue(value_and_grad)
            return types.SimpleNamespace(x=np.zeros(2), nit=0, message="ok")

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=fake_require_target_backend_x64,
            jax_minimize=fake_jax_minimize,
            scipy_minimize_side_effect=AssertionError,
        ):
            contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
            result = module.run_single_stage_optimizer(
                explicit_fun,
                np.array([0.0, 0.0]),
                contract=contract,
                maxiter=1,
                ftol=0.0,
                gtol=1e-6,
                maxcor=5,
                outer_maxls=4,
                callback=lambda _x: None,
                scalar_fun=None,
            )

        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_allows_explicit_experimental_target_lane(self):
        module = self.load_module()
        captured = {}
        explicit_fun = lambda x: (
            jnp.asarray(jnp.dot(x, x), dtype=jnp.float64),
            jnp.asarray(2.0 * x, dtype=jnp.float64),
        )

        def fake_require_target_backend_x64(_optimizer_backend):
            return None

        def fake_jax_minimize(
            fun, x0, *, method, tol, maxiter, options, value_and_grad, callback
        ):
            captured["fun"] = fun
            captured["x0"] = x0
            captured["value_and_grad"] = value_and_grad
            return types.SimpleNamespace(x=x0, nit=0, message="ok")

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=fake_require_target_backend_x64,
            jax_minimize=fake_jax_minimize,
        ):
            contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
            result = module.run_single_stage_optimizer(
                explicit_fun,
                np.array([0.0, 0.0]),
                contract=contract,
                maxiter=1,
                ftol=0.0,
                gtol=1e-6,
                maxcor=5,
                outer_maxls=6,
                callback=None,
                scalar_fun=None,
            )

        self.assertIsInstance(captured["x0"], jax.Array)
        self.assertTrue(captured["value_and_grad"])
        value, grad = captured["fun"](captured["x0"])
        self.assertEqual(value, 0.0)
        self.assertIsInstance(grad, jax.Array)
        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_rejects_unknown_outer_lane(self):
        module = self.load_module()

        def fake_require_target_backend_x64(optimizer_backend):
            raise AssertionError(
                f"x64 check should not run for unsupported lane: {optimizer_backend}"
            )

        def fake_jax_minimize(
            fun, x0, *, method, tol, maxiter, options, value_and_grad, callback
        ):
            raise AssertionError(
                "Unsupported single-stage lane must fail before jax_minimize."
            )

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=fake_require_target_backend_x64,
            jax_minimize=fake_jax_minimize,
        ):
            with self.assertRaisesRegex(
                ValueError,
                _OPTIMIZER_BACKEND_INVALID,
            ):
                module.resolve_single_stage_optimizer_contract("jax", "bogus")

    def test_resolve_single_stage_outer_optimizer_method_rejects_cpu_ondevice(self):
        module = self.load_module()

        with self.assertRaisesRegex(
            ValueError,
            _SINGLE_STAGE_CPU_ONLY_SCIPY,
        ):
            module.resolve_single_stage_outer_optimizer_method("cpu", "ondevice")

    def test_resolve_single_stage_outer_optimizer_method_rejects_unknown_backend(self):
        module = self.load_module()

        with self.assertRaisesRegex(
            ValueError,
            _OPTIMIZER_BACKEND_INVALID,
        ):
            module.resolve_single_stage_outer_optimizer_method("jax", "bogus")

    def test_single_stage_adapter_callback_reevaluates_before_accept_in_target_lane(
        self,
    ):
        module = self.load_module()

        class _JF:
            def __init__(self):
                self._x = None

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = np.asarray(value)

        jf = _JF()
        run_dict = {"x_prev": np.zeros(2), "lscount": 3}
        captured = {}

        def fake_eval(
            x,
            state,
            booz,
            objective,
            objectives=None,
            diagnostics=None,
        ):
            captured["eval"] = {
                "x": np.asarray(x),
                "state": state,
                "booz": booz,
                "objective": objective,
                "objectives": objectives,
                "diagnostics": diagnostics,
            }
            return 1.0, np.zeros(2)

        def fake_accept_step(
            state, booz, objective, bs, objectives, diagnostics, log_path
        ):
            captured["accept"] = {
                "state": state,
                "booz": booz,
                "objective": objective,
                "bs": bs,
                "objectives": objectives,
                "diagnostics": diagnostics,
                "log_path": log_path,
            }

        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface="booz",
            JF=jf,
            bs="bs",
            objectives={"qs": "obj"},
            diagnostics={"iota": "diag"},
            log_path="/tmp/log.txt",
            reevaluate_before_accept=True,
        )

        with patch.object(
            module,
            "_evaluate_candidate_impl",
            side_effect=fake_eval,
        ), patch.object(module, "accept_step", side_effect=fake_accept_step):
            adapter.callback(
                module.SingleStageOuterOptimizerState(
                    coil_dofs=np.array([2.0, -1.0]),
                )
            )

        np.testing.assert_allclose(jf.x, np.array([2.0, -1.0]))
        np.testing.assert_allclose(run_dict["x_prev"], np.array([2.0, -1.0]))
        # Line-search counter must not increment during reevaluation.
        self.assertEqual(run_dict["lscount"], 3)
        self.assertIs(captured["accept"]["state"], run_dict)
        self.assertEqual(captured["accept"]["log_path"], "/tmp/log.txt")
        self.assertEqual(captured["eval"]["objectives"], {"qs": "obj"})
        self.assertEqual(captured["eval"]["diagnostics"], {"iota": "diag"})

    def test_single_stage_adapter_sync_accepted_step_refreshes_target_lane_state(
        self,
    ):
        module = self.load_module()

        class _JF:
            def __init__(self):
                self._x = None

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = np.asarray(value)

        jf = _JF()
        run_dict = {"x_prev": np.zeros(2), "lscount": 5}
        captured = {}

        def fake_eval(
            x,
            state,
            booz,
            objective,
            objectives=None,
            diagnostics=None,
        ):
            captured["eval"] = {
                "x": np.asarray(x),
                "state": state,
                "booz": booz,
                "objective": objective,
                "objectives": objectives,
                "diagnostics": diagnostics,
            }
            return 1.0, np.zeros(2)

        def fake_accept_step(
            state, booz, objective, bs, objectives, diagnostics, log_path
        ):
            captured["accept"] = {
                "state": state,
                "booz": booz,
                "objective": objective,
                "bs": bs,
                "objectives": objectives,
                "diagnostics": diagnostics,
                "log_path": log_path,
            }

        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface="booz",
            JF=jf,
            bs="bs",
            objectives={"qs": "obj"},
            diagnostics={"iota": "diag"},
            log_path="/tmp/log.txt",
            reevaluate_before_accept=True,
        )

        with patch.object(
            module,
            "_evaluate_candidate_impl",
            side_effect=fake_eval,
        ), patch.object(module, "accept_step", side_effect=fake_accept_step):
            adapter.sync_accepted_step(
                module.SingleStageOuterOptimizerState(
                    coil_dofs=np.array([3.0, -4.0]),
                )
            )

        np.testing.assert_allclose(jf.x, np.array([3.0, -4.0]))
        np.testing.assert_allclose(run_dict["x_prev"], np.array([3.0, -4.0]))
        self.assertEqual(run_dict["lscount"], 5)
        self.assertIs(captured["accept"]["state"], run_dict)
        self.assertEqual(captured["accept"]["log_path"], "/tmp/log.txt")
        self.assertEqual(captured["eval"]["objectives"], {"qs": "obj"})
        self.assertEqual(captured["eval"]["diagnostics"], {"iota": "diag"})

    def test_single_stage_adapter_uses_custom_target_lane_dof_setter(self):
        module = self.load_module()

        class _JF:
            def __init__(self):
                self._x = np.array([1.0, 2.0])

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = np.asarray(value)

        jf = _JF()
        run_dict = {"x_prev": np.zeros(2), "lscount": 0}
        captured = {"setter": []}

        def fake_setter(value):
            captured["setter"].append(np.asarray(value))

        def fake_eval(
            x,
            state,
            booz,
            objective,
            objectives=None,
            diagnostics=None,
        ):
            captured["eval_x"] = np.asarray(x)
            captured["eval_objectives"] = objectives
            captured["eval_diagnostics"] = diagnostics
            return 1.0, np.zeros(2)

        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface="booz",
            JF=jf,
            bs="bs",
            objectives={"qs": "obj"},
            diagnostics={"iota": "diag"},
            log_path="/tmp/log.txt",
            reevaluate_before_accept=True,
            apply_coil_dofs=fake_setter,
        )

        with patch.object(
            module, "_evaluate_candidate_impl", side_effect=fake_eval
        ), patch.object(module, "accept_step", return_value=None):
            adapter.sync_accepted_step(np.array([7.0, -8.0]))

        assert len(captured["setter"]) == 1
        np.testing.assert_allclose(captured["setter"][0], np.array([7.0, -8.0]))
        np.testing.assert_allclose(captured["eval_x"], np.array([7.0, -8.0]))
        self.assertEqual(captured["eval_objectives"], {"qs": "obj"})
        self.assertEqual(captured["eval_diagnostics"], {"iota": "diag"})
        np.testing.assert_allclose(jf.x, np.array([1.0, 2.0]))

    def test_single_stage_adapter_sync_accepted_step_prefers_array_native_target_lane_sync(
        self,
    ):
        module = self.load_module()

        class _JF:
            def __init__(self):
                self._x = np.array([1.0, 2.0])

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = np.asarray(value)

        jf = _JF()
        run_dict = {"x_prev": np.zeros(2), "lscount": 4, "it": 3}
        captured = {"setter": []}

        def fake_setter(value):
            captured["setter"].append(np.asarray(value))

        def fake_sync(state, x, *, benchmark_mode):
            captured["sync"] = {
                "state": state,
                "x": np.asarray(x),
                "benchmark_mode": benchmark_mode,
            }
            state["sdofs"] = np.array([9.0, -2.0])
            state["iota"] = 0.21
            state["G"] = 1.8
            return {
                "objective_value": 1.25,
                "reporting_metrics": {
                    "final_iota": 0.21,
                    "final_volume": 0.75,
                    "coil_length": 2.0,
                    "max_curvature": 3.0,
                    "curve_curve_min_dist": 0.11,
                    "curve_surface_min_dist": 0.12,
                    "surface_vessel_min_dist": 0.13,
                    "hardware_status": {
                        "success": True,
                        "violations": [],
                    },
                },
            }

        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface="booz",
            JF=jf,
            bs="bs",
            objectives={"qs": "obj"},
            diagnostics={"iota": "diag"},
            log_path="/tmp/log.txt",
            reevaluate_before_accept=True,
            apply_coil_dofs=fake_setter,
            accepted_step_state_sync=fake_sync,
        )

        with patch.object(
            module,
            "log_single_stage_target_lane_accepted_step",
            return_value=None,
        ) as log_summary, patch.object(
            module,
            "_evaluate_candidate_impl",
            side_effect=AssertionError(
                "array-native target-lane sync should not reevaluate the mutable graph"
            ),
        ), patch.object(
            module,
            "accept_step",
            side_effect=AssertionError(
                "array-native target-lane sync should not call accept_step"
            ),
        ):
            adapter.sync_accepted_step(np.array([3.0, -4.0]))

        self.assertIs(captured["sync"]["state"], run_dict)
        np.testing.assert_allclose(captured["sync"]["x"], np.array([3.0, -4.0]))
        self.assertFalse(captured["sync"]["benchmark_mode"])
        log_summary.assert_called_once()
        np.testing.assert_allclose(run_dict["x_prev"], np.array([3.0, -4.0]))
        self.assertEqual(captured["setter"], [])
        np.testing.assert_allclose(jf.x, np.array([1.0, 2.0]))

    def test_single_stage_adapter_sync_accepted_step_uses_benchmark_snapshot_path(
        self,
    ):
        module = self.load_module()

        class _JF:
            def __init__(self):
                self._x = None

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = np.asarray(value)

        jf = _JF()
        run_dict = {"x_prev": np.zeros(2), "lscount": 4}
        captured = {}

        def fake_snapshot(
            state,
            booz,
            objective,
            *,
            objective_value=None,
            objective_grad=None,
            store_objective_grad=True,
        ):
            captured["snapshot"] = {
                "state": state,
                "booz": booz,
                "objective": objective,
                "objective_value": objective_value,
                "objective_grad": objective_grad,
                "store_objective_grad": store_objective_grad,
            }
            return 1.0, np.zeros(2)

        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface="booz",
            JF=jf,
            bs="bs",
            objectives={"qs": "obj"},
            diagnostics={"iota": "diag"},
            log_path="/tmp/log.txt",
            benchmark_mode=True,
        )

        with patch.object(
            module,
            "snapshot_accepted_step_state",
            side_effect=fake_snapshot,
        ), patch.object(
            module,
            "accept_step",
            side_effect=AssertionError(
                "benchmark sync path should not call accept_step"
            ),
        ):
            adapter.sync_accepted_step(np.array([9.0, -10.0]))

        self.assertIsNone(jf.x)
        np.testing.assert_allclose(run_dict["x_prev"], np.zeros(2))
        self.assertIs(captured["snapshot"]["state"], run_dict)
        self.assertIsNone(captured["snapshot"]["objective_value"])
        self.assertIsNone(captured["snapshot"]["objective_grad"])
        self.assertFalse(captured["snapshot"]["store_objective_grad"])
        self.assertEqual(run_dict["lscount"], 4)

    def test_single_stage_adapter_benchmark_sync_reuses_reevaluated_objective(self):
        module = self.load_module()

        class _JF:
            def __init__(self):
                self._x = None

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = np.asarray(value)

        jf = _JF()
        run_dict = {"x_prev": np.zeros(2), "lscount": 2}
        captured = {}
        objective_value = 3.5
        objective_grad = np.array([1.0, -1.0])

        def fake_eval(
            x,
            state,
            booz,
            objective,
            objectives=None,
            diagnostics=None,
        ):
            captured["eval"] = {
                "x": np.asarray(x),
                "state": state,
                "objectives": objectives,
                "diagnostics": diagnostics,
            }
            return objective_value, objective_grad

        def fake_snapshot(
            state,
            booz,
            objective,
            *,
            objective_value=None,
            objective_grad=None,
            store_objective_grad=True,
        ):
            captured["snapshot"] = {
                "state": state,
                "booz": booz,
                "objective": objective,
                "objective_value": objective_value,
                "objective_grad": objective_grad,
                "store_objective_grad": store_objective_grad,
            }
            return objective_value, objective_grad

        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface="booz",
            JF=jf,
            bs="bs",
            objectives={"qs": "obj"},
            diagnostics={"iota": "diag"},
            log_path="/tmp/log.txt",
            benchmark_mode=True,
            reevaluate_before_accept=True,
        )

        adapter._refresh_accepted_step_runtime_state = lambda _x: False

        with patch.object(
            module,
            "_evaluate_candidate_impl",
            side_effect=fake_eval,
        ), patch.object(
            module,
            "snapshot_accepted_step_state",
            side_effect=fake_snapshot,
        ), patch.object(
            module,
            "accept_step",
            side_effect=AssertionError(
                "benchmark sync path should not call accept_step"
            ),
        ):
            adapter.sync_accepted_step(np.array([4.0, -5.0]))

        np.testing.assert_allclose(jf.x, np.array([4.0, -5.0]))
        np.testing.assert_allclose(run_dict["x_prev"], np.array([4.0, -5.0]))
        self.assertEqual(run_dict["lscount"], 2)
        self.assertIs(captured["snapshot"]["state"], run_dict)
        self.assertEqual(captured["snapshot"]["objective_value"], objective_value)
        np.testing.assert_allclose(
            captured["snapshot"]["objective_grad"],
            objective_grad,
        )
        self.assertTrue(captured["snapshot"]["store_objective_grad"])

    def test_single_stage_adapter_benchmark_sync_can_skip_objective_refresh(self):
        module = self.load_module()

        class _JF:
            def J(self):
                raise AssertionError("benchmark runtime-only sync should skip J()")

            def dJ(self):
                raise AssertionError("benchmark runtime-only sync should skip dJ()")

        jf = _JF()
        run_dict = {
            "x_prev": np.zeros(2),
            "lscount": 3,
            "J": 7.0,
            "dJ": np.array([1.0, 2.0]),
        }
        captured = {}

        def fake_snapshot(
            state,
            booz,
            objective,
            *,
            objective_value=None,
            objective_grad=None,
            store_objective_grad=True,
        ):
            captured["snapshot"] = {
                "state": state,
                "booz": booz,
                "objective": objective,
                "objective_value": objective_value,
                "objective_grad": objective_grad,
                "store_objective_grad": store_objective_grad,
            }
            return state["J"], state["dJ"]

        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface="booz",
            JF=jf,
            bs="bs",
            objectives={"qs": "obj"},
            diagnostics={"iota": "diag"},
            log_path="/tmp/log.txt",
            benchmark_mode=True,
            reevaluate_before_accept=True,
        )
        adapter._refresh_accepted_step_runtime_state = lambda _x: True

        def raise_if_called(_x):
            raise AssertionError(
                "runtime-only benchmark sync should not reevaluate objective"
            )

        adapter._reevaluate_accepted_step = raise_if_called

        with patch.object(
            module,
            "snapshot_accepted_step_state",
            side_effect=fake_snapshot,
        ), patch.object(
            module,
            "accept_step",
            side_effect=AssertionError(
                "benchmark sync path should not call accept_step"
            ),
        ):
            adapter.sync_accepted_step(np.array([4.0, -5.0]))

        np.testing.assert_allclose(run_dict["x_prev"], np.array([0.0, 0.0]))
        self.assertFalse(captured["snapshot"]["store_objective_grad"])
        self.assertIsNone(captured["snapshot"]["objective_value"])
        self.assertIsNone(captured["snapshot"]["objective_grad"])

    def test_snapshot_accepted_step_state_can_skip_objective_refresh(self):
        module = self.load_module()
        run_dict = {
            "lscount": 5,
            "J": 2.5,
            "dJ": np.array([3.0, -4.0]),
        }
        boozer_surface = types.SimpleNamespace(
            surface=types.SimpleNamespace(x=np.array([1.0, 2.0])),
            res={"iota": np.float64(0.125), "G": np.float64(1.75)},
        )

        class _JF:
            def J(self):
                raise AssertionError("store_objective_grad=False should skip J()")

            def dJ(self):
                raise AssertionError("store_objective_grad=False should skip dJ()")

        objective_value, objective_grad = module.snapshot_accepted_step_state(
            run_dict,
            boozer_surface,
            _JF(),
            store_objective_grad=False,
        )

        self.assertEqual(run_dict["lscount"], 0)
        np.testing.assert_allclose(run_dict["sdofs"], np.array([1.0, 2.0]))
        self.assertEqual(run_dict["iota"], 0.125)
        self.assertEqual(run_dict["G"], 1.75)
        self.assertEqual(objective_value, 2.5)
        np.testing.assert_allclose(objective_grad, np.array([3.0, -4.0]))

    def test_snapshot_accepted_step_state_from_values_can_refresh_objective_only(self):
        module = self.load_module()
        run_dict = {
            "lscount": 5,
            "J": 2.5,
            "dJ": np.array([3.0, -4.0]),
        }

        objective_value, objective_grad = (
            module.snapshot_accepted_step_state_from_values(
                run_dict,
                sdofs=np.array([1.0, 2.0]),
                iota=np.float64(0.125),
                G=np.float64(1.75),
                objective_value=7.5,
                store_objective_grad=False,
            )
        )

        self.assertEqual(run_dict["lscount"], 0)
        np.testing.assert_allclose(run_dict["sdofs"], np.array([1.0, 2.0]))
        self.assertEqual(run_dict["iota"], 0.125)
        self.assertEqual(run_dict["G"], 1.75)
        self.assertEqual(objective_value, 7.5)
        np.testing.assert_allclose(objective_grad, np.array([3.0, -4.0]))

    def test_evaluate_surface_self_intersection_skips_when_backend_unavailable(self):
        module = self.load_module()

        class SentinelSurface:
            def is_self_intersecting(self):
                raise AssertionError(
                    "surface.is_self_intersecting should not be called"
                )

        with self.patch_surface_self_intersection_backend_unavailable(module):
            self.assertEqual(
                module.evaluate_surface_self_intersection(SentinelSurface()),
                (False, False),
            )

    def test_initialize_boozer_surface_skips_optional_self_intersection_gate(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()

        with self.patch_surface_self_intersection_backend_unavailable(
            module
        ), patch.object(
            FakeSurfaceXYZTensorFourier,
            "is_self_intersecting",
            side_effect=AssertionError(
                "surface.is_self_intersecting should not be called"
            ),
        ):
            boozer_surface = self.initialize_boozer_surface(
                module, surf_prev, constraint_weight=1.0
            )

        self.assertIsInstance(boozer_surface, FakeBoozerSurface)

    def test_diagnostic_field_prefers_cpu_reference_when_present(self):
        module = self.load_module()
        cpu_field = object()
        jax_field = object()

        self.assertIs(module.diagnostic_field(jax_field, cpu_field), cpu_field)
        self.assertIs(module.diagnostic_field(jax_field, None), jax_field)

    def test_build_iota_objective_uses_supplied_wrapper_class(self):
        module = self.load_module()
        calls = []

        class FakeIotaWrapper:
            def __init__(self, boozer_surface):
                calls.append(boozer_surface)

        marker = object()
        wrapper = module.build_iota_objective(marker, FakeIotaWrapper)

        self.assertIsInstance(wrapper, FakeIotaWrapper)
        self.assertEqual(calls, [marker])

    def test_resolve_single_stage_iota_metric_uses_cached_boozer_iota_in_benchmark_mode(
        self,
    ):
        module = self.load_module()
        boozer_surface = types.SimpleNamespace(res={"iota": np.float64(0.1234)})

        with patch.object(
            module,
            "build_iota_objective",
            side_effect=AssertionError(
                "benchmark-mode iota metric should not rebuild the wrapper"
            ),
        ):
            resolved = module.resolve_single_stage_iota_metric(
                boozer_surface,
                object(),
                benchmark_mode=True,
            )

        self.assertEqual(resolved, 0.1234)

    def test_select_boozer_residual_class_routes_exact_stage_to_exact_wrapper(self):
        module = self.load_module()

        self.assertIs(
            module.select_boozer_residual_class(use_jax=True, boozer_kind="exact"),
            module.BoozerResidualExact,
        )
        self.assertIs(
            module.select_boozer_residual_class(use_jax=False, boozer_kind="exact"),
            module.BoozerResidualExact,
        )

    def test_select_boozer_residual_class_routes_least_squares_by_backend(self):
        module = self.load_module()

        class FakeBoozerResidualJAX:
            pass

        with patch.object(
            module,
            "get_jax_surface_objective_classes",
            return_value=(FakeBoozerResidualJAX, object(), object()),
        ):
            self.assertIs(
                module.select_boozer_residual_class(
                    use_jax=True, boozer_kind="least_squares"
                ),
                FakeBoozerResidualJAX,
            )

        self.assertIs(
            module.select_boozer_residual_class(
                use_jax=False, boozer_kind="least_squares"
            ),
            module.BoozerResidual,
        )

    def test_build_boozer_residual_objective_uses_supplied_wrapper_class(self):
        module = self.load_module()
        calls = []

        class FakeResidualWrapper:
            def __init__(self, boozer_surface, bs_obj):
                calls.append((boozer_surface, bs_obj))

        fake_boozer_surface = object()
        fake_bs = object()
        wrapper = module.build_boozer_residual_objective(
            fake_boozer_surface,
            fake_bs,
            FakeResidualWrapper,
        )

        self.assertIsInstance(wrapper, FakeResidualWrapper)
        self.assertEqual(calls, [(fake_boozer_surface, fake_bs)])

    def test_cpu_boozer_surface_zero_weight_contract_uses_explicit_none_check(self):
        source = (
            Path(__file__).resolve().parents[2]
            / "src"
            / "simsopt"
            / "geo"
            / "boozersurface.py"
        ).read_text()
        self.assertIn("constraint_weight is not None", source)

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

        class _BoozerSurface:
            surface = _Surface()
            res = {
                "success": False,
                "iota": TEST_IOTA,
                "G": TEST_G0,
                "PLU": None,
                "vjp": None,
            }

            def run_code(self, iota, G, *, sdofs=None):
                return self.res

        class _JF:
            def __init__(self):
                self.x = np.zeros(5)

            def J(self):
                raise AssertionError("JF.J must not be called on failure path")

            def dJ(self):
                raise AssertionError("JF.dJ must not be called on failure path")

        run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "sdofs": np.ones(3),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": last_J,
            "dJ": last_dJ.copy(),
        }
        booz = _BoozerSurface()
        jf = _JF()

        J_out, dJ_out = module.evaluate_candidate(np.ones(5), run_dict, booz, jf)

        self.assertGreater(J_out, last_J)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        self.assertIsNot(dJ_out, run_dict["dJ"])

    def test_fun_fallback_cpu_rolls_back_surface_state(self):
        """CPU failure path must restore surface.x and res from run_dict."""
        module = self.load_module()
        CpuBoozerSurface = module.BoozerSurface

        class _Surface:
            def __init__(self):
                self.x = np.array([9.0, 8.0, 7.0])

            def is_self_intersecting(self):
                return False

        class _CpuMock(CpuBoozerSurface):
            def __init__(self):
                self.surface = _Surface()
                self.res = {
                    "success": False,
                    "iota": -999.0,
                    "G": -999.0,
                }

            def run_code(self, iota, G=None):
                # Simulate a failed solve that leaves dirty state
                self.surface.x = np.array([0.0, 0.0, 0.0])
                self.res["iota"] = -999.0
                self.res["G"] = -999.0
                return self.res

        sdofs_warm = np.array([1.0, 2.0, 3.0])
        run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "sdofs": sdofs_warm.copy(),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 42.0,
            "dJ": np.array([1.0, -2.0, 3.0, -4.0, 5.0]),
        }
        booz = _CpuMock()
        jf = types.SimpleNamespace(x=np.zeros(5))

        J_out, _ = module.evaluate_candidate(np.ones(5), run_dict, booz, jf)

        # Failure: elevated J
        self.assertGreater(J_out, 42.0)
        # Rollback: surface.x and res restored from run_dict
        np.testing.assert_array_equal(booz.surface.x, sdofs_warm)
        self.assertEqual(booz.res["iota"], TEST_IOTA)
        self.assertEqual(booz.res["G"], TEST_G0)

    def test_evaluate_candidate_does_not_mutate_external_state(self):
        """evaluate_candidate must not directly mutate JF.x, surface.x, or res.

        This tests the direct-call contract: ``evaluate_candidate`` itself
        performs no mutations.  ``JF.x = x`` happens in
        ``SingleStageAdapter.__call__``, and the real ``run_code`` syncs
        ``self.surface`` internally — neither is tested here.
        """
        module = self.load_module()

        class _Surface:
            def __init__(self):
                self.x = np.array([9.0, 8.0, 7.0])

            def volume(self):
                return 1.0

        class _BoozerSurface:
            def __init__(self):
                self.surface = _Surface()
                self.res = {
                    "success": True,
                    "iter": 1,
                    "iota": -3.0,
                    "G": -4.0,
                }
                self.run_code_calls = []

            def run_code(self, iota, G, *, sdofs=None):
                self.run_code_calls.append((iota, G, sdofs))
                return self.res

        class _JF:
            def __init__(self):
                self.x = np.array([-5.0, -4.0, -3.0, -2.0, -1.0])

            def J(self):
                return 3.14

            def dJ(self):
                return np.arange(5.0)

        run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "sdofs": np.array([1.0, 2.0, 3.0]),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.zeros(5),
        }
        booz = _BoozerSurface()
        jf = _JF()
        surface_x_before = booz.surface.x.copy()
        res_before = booz.res.copy()
        jf_x_before = jf.x.copy()

        with patch.object(
            module, "update_self_intersection_status", return_value=False
        ):
            module.evaluate_candidate(np.ones(5), run_dict, booz, jf)

        # evaluate_candidate must not directly mutate these
        np.testing.assert_array_equal(jf.x, jf_x_before)
        np.testing.assert_array_equal(booz.surface.x, surface_x_before)
        self.assertEqual(booz.res, res_before)

        # evaluate_candidate must forward sdofs from run_dict to run_code
        self.assertEqual(len(booz.run_code_calls), 1)
        call_iota, call_G, call_sdofs = booz.run_code_calls[0]
        self.assertEqual(call_iota, TEST_IOTA)
        self.assertEqual(call_G, TEST_G0)
        np.testing.assert_array_equal(call_sdofs, run_dict["sdofs"])

    def test_accept_step_does_not_mutate_bs_points(self):
        """accept_step must restore bs evaluation points after BdotN diagnostic."""
        module = self.load_module()

        INITIAL_PTS = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        GAMMA_SHAPE = (3, 4, 3)

        class _BS:
            """Fake BiotSavart that tracks set_points calls."""

            def __init__(self):
                self._current_pts = INITIAL_PTS.copy()
                self.set_points_calls = []

            def get_points_cart_ref(self):
                return self._current_pts

            def set_points(self, pts):
                self.set_points_calls.append(pts.copy())
                self._current_pts = np.asarray(pts).copy()

            def B(self):
                n = self._current_pts.shape[0]
                return np.ones((n, 3)) * 0.01

        class _Surface:
            def __init__(self):
                self.x = np.zeros(5)

            def gamma(self):
                return np.ones(GAMMA_SHAPE)

            def unitnormal(self):
                n = np.zeros(GAMMA_SHAPE)
                n[..., 2] = 1.0
                return n

            def volume(self):
                return 1.0

        class _BoozerSurface:
            def __init__(self):
                self.surface = _Surface()
                self.res = {"success": True, "iter": 1, "iota": 0.15, "G": 1.0}

        class _JF:
            def J(self):
                return 1.0

            def dJ(self):
                return np.zeros(5)

        class _Objective:
            def J(self):
                return 0.5

            def dJ(self):
                return np.zeros(5)

            def shortest_distance(self):
                return 0.1

        class _IotaObj:
            def J(self):
                return 0.15

        class _Curve:
            def gamma(self):
                return np.zeros((10, 3))

            def kappa(self):
                return np.ones(10)

        class _CurveLength:
            def J(self):
                return 6.0

        bs = _BS()
        booz = _BoozerSurface()

        run_dict = {
            "lscount": 5,
            "sdofs": np.zeros(5),
            "iota": 0.15,
            "G": 1.0,
            "J": 1.0,
            "dJ": np.zeros(5),
            "it": 1,
            "intersecting": False,
            "self_intersection_check_available": False,
        }
        objectives = {
            "cc": _Objective(),
            "cs": _Objective(),
            "surf": _Objective(),
            "boozer": _Objective(),
        }
        diagnostics_refs = {
            "iota": _IotaObj(),
            "banana_curve": _Curve(),
            "curvelength": _CurveLength(),
        }
        module.CC_DIST = 0.05
        module.CS_DIST = 0.02
        module.SS_DIST = 0.04
        module.CURVATURE_THRESHOLD = 40.0
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            with patch.object(
                module, "update_self_intersection_status", return_value=False
            ), patch.object(module, "BiotSavart", _BS):
                module.accept_step(
                    run_dict, booz, _JF(), bs, objectives, diagnostics_refs, log_path
                )

        # bs must be restored to its original evaluation points
        np.testing.assert_array_equal(bs._current_pts, INITIAL_PTS)
        # set_points must have been called twice: once for gamma, once for restore
        self.assertEqual(len(bs.set_points_calls), 2)
        np.testing.assert_array_equal(bs.set_points_calls[1], INITIAL_PTS)

    def test_accept_step_explicitly_materializes_jax_diagnostics(self):
        module = self.load_module()
        host_calls = {"float": 0, "array": 0}
        original_host_float = module.host_float
        original_host_array = module.host_array

        def counted_host_float(value):
            host_calls["float"] += 1
            return original_host_float(value)

        def counted_host_array(value, *, dtype=np.float64):
            host_calls["array"] += 1
            return original_host_array(value, dtype=dtype)

        class _BS:
            def __init__(self):
                self._current_pts = np.array([[1.0, 2.0, 3.0]])

            def get_points_cart_ref(self):
                return self._current_pts

            def set_points(self, pts):
                self._current_pts = np.asarray(pts).copy()

            def B(self):
                n = self._current_pts.shape[0]
                return jnp.ones((n, 3), dtype=jnp.float64) * 0.01

        class _Surface:
            def __init__(self):
                self.x = np.zeros(5)

            def gamma(self):
                return np.ones((1, 2, 3))

            def unitnormal(self):
                normal = np.zeros((1, 2, 3))
                normal[..., 2] = 1.0
                return normal

            def volume(self):
                return jnp.asarray(1.0, dtype=jnp.float64)

        class _BoozerSurface:
            def __init__(self):
                self.surface = _Surface()
                self.res = {
                    "success": jnp.asarray(True),
                    "iter": jnp.asarray(1),
                    "iota": jnp.asarray(0.15, dtype=jnp.float64),
                    "G": jnp.asarray(1.0, dtype=jnp.float64),
                }

        class _JF:
            def J(self):
                return jnp.asarray(1.0, dtype=jnp.float64)

            def dJ(self):
                return jnp.arange(5, dtype=jnp.float64)

        class _Objective:
            def J(self):
                return jnp.asarray(0.5, dtype=jnp.float64)

            def dJ(self):
                return jnp.zeros(5, dtype=jnp.float64)

            def shortest_distance(self):
                return jnp.asarray(0.1, dtype=jnp.float64)

        class _IotaObj:
            def J(self):
                return jnp.asarray(0.15, dtype=jnp.float64)

        class _Curve:
            def gamma(self):
                return np.zeros((10, 3))

            def kappa(self):
                return np.ones(10)

        class _CurveLength:
            def J(self):
                return jnp.asarray(6.0, dtype=jnp.float64)

        run_dict = {
            "lscount": 5,
            "sdofs": np.zeros(5),
            "iota": 0.15,
            "G": 1.0,
            "J": 1.0,
            "dJ": np.zeros(5),
            "it": 1,
            "intersecting": False,
            "self_intersection_check_available": False,
        }
        objectives = {
            "cc": _Objective(),
            "cs": _Objective(),
            "surf": _Objective(),
            "boozer": _Objective(),
        }
        diagnostics_refs = {
            "iota": _IotaObj(),
            "banana_curve": _Curve(),
            "curvelength": _CurveLength(),
        }
        module.CC_DIST = 0.05
        module.CS_DIST = 0.02
        module.SS_DIST = 0.04
        module.CURVATURE_THRESHOLD = 40.0

        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = os.path.join(tmpdir, "test.log")
            with patch.object(
                module, "host_float", counted_host_float
            ), patch.object(
                module, "host_array", counted_host_array
            ), patch.object(
                module, "update_self_intersection_status", return_value=False
            ), patch.object(
                module, "BiotSavart", _BS
            ):
                module.accept_step(
                    run_dict,
                    _BoozerSurface(),
                    _JF(),
                    _BS(),
                    objectives,
                    diagnostics_refs,
                    log_path,
                )

        self.assertGreaterEqual(host_calls["float"], 6)
        self.assertGreaterEqual(host_calls["array"], 3)

    def test_evaluate_candidate_cpu_path_does_not_pass_sdofs(self):
        """CPU BoozerSurface.run_code() does not accept sdofs=.

        Regression test: evaluate_candidate must detect the legacy CPU
        warm-start contract from the run_code signature and use the old
        warm-start path (surface.x and res mutation) instead of passing
        sdofs= to run_code.
        """
        module = self.load_module()
        CpuBoozerSurface = module.BoozerSurface

        class _Surface:
            def __init__(self):
                self.x = np.array([9.0, 8.0, 7.0])

            def volume(self):
                return 1.0

            def is_self_intersecting(self):
                return False

        class _CpuMock(CpuBoozerSurface):
            """Mock that passes isinstance check and rejects sdofs=."""

            def __init__(self):
                self.surface = _Surface()
                self.res = {
                    "success": True,
                    "iter": 1,
                    "iota": TEST_IOTA,
                    "G": TEST_G0,
                }
                self.run_code_calls = []

            def run_code(self, iota, G=None):
                # CPU signature — no sdofs parameter
                self.run_code_calls.append((iota, G))
                return self.res

        class _JF:
            x = np.zeros(5)

            def J(self):
                return 3.14

            def dJ(self):
                return np.arange(5.0)

        sdofs_warm = np.array([1.0, 2.0, 3.0])
        run_dict = self._make_candidate_run_dict(sdofs_warm)
        booz = _CpuMock()
        jf = _JF()

        with patch.object(
            module, "update_self_intersection_status", return_value=False
        ):
            module.evaluate_candidate(np.ones(5), run_dict, booz, jf)

        # CPU path must call run_code(iota, G) without sdofs
        self.assertEqual(len(booz.run_code_calls), 1)
        self.assertEqual(booz.run_code_calls[0], (TEST_IOTA, TEST_G0))
        # CPU path must warm-start surface.x from run_dict before run_code
        np.testing.assert_array_equal(booz.surface.x, sdofs_warm)

    def test_evaluate_candidate_prefers_explicit_warm_start_capability(self):
        """Explicit warm-start capability must override BoozerSurface identity."""
        module = self.load_module()
        CpuBoozerSurface = module.BoozerSurface

        class _Surface:
            def __init__(self):
                self.x = np.array([9.0, 8.0, 7.0])

            def volume(self):
                return 1.0

            def is_self_intersecting(self):
                return False

        class _WarmStartCapableCpuSubclass(CpuBoozerSurface):
            supports_explicit_surface_warm_start = True

            def __init__(self):
                self.surface = _Surface()
                self.res = {
                    "success": True,
                    "iter": 1,
                    "iota": TEST_IOTA,
                    "G": TEST_G0,
                }
                self.run_code_calls = []

            def run_code(self, iota, G=None, *, sdofs=None):
                self.run_code_calls.append((iota, G, None if sdofs is None else sdofs.copy()))
                return self.res

        class _JF:
            x = np.zeros(5)

            def J(self):
                return 3.14

            def dJ(self):
                return np.arange(5.0)

        sdofs_warm = np.array([1.0, 2.0, 3.0])
        run_dict = self._make_candidate_run_dict(sdofs_warm)
        booz = _WarmStartCapableCpuSubclass()
        jf = _JF()

        with patch.object(
            module, "update_self_intersection_status", return_value=False
        ):
            module.evaluate_candidate(np.ones(5), run_dict, booz, jf)

        self.assertEqual(len(booz.run_code_calls), 1)
        call_iota, call_G, call_sdofs = booz.run_code_calls[0]
        self.assertEqual(call_iota, TEST_IOTA)
        self.assertEqual(call_G, TEST_G0)
        np.testing.assert_array_equal(call_sdofs, sdofs_warm)

    def test_evaluate_candidate_prefers_explicit_false_capability_over_signature(self):
        """Explicit false capability must keep CPU subclasses on the legacy path."""
        module = self.load_module()
        CpuBoozerSurface = module.BoozerSurface

        class _Surface:
            def __init__(self):
                self.x = np.array([9.0, 8.0, 7.0])

            def volume(self):
                return 1.0

            def is_self_intersecting(self):
                return False

        class _CpuMock(CpuBoozerSurface):
            def __init__(self):
                self.surface = _Surface()
                self.res = {
                    "success": True,
                    "iter": 1,
                    "iota": TEST_IOTA,
                    "G": TEST_G0,
                }
                self.run_code_calls = []

            def run_code(self, iota, G=None, *, sdofs=None):
                self.run_code_calls.append((iota, G, sdofs))
                return self.res

        class _JF:
            x = np.zeros(5)

            def J(self):
                return 3.14

            def dJ(self):
                return np.arange(5.0)

        sdofs_warm = np.array([1.0, 2.0, 3.0])
        run_dict = self._make_candidate_run_dict(sdofs_warm)
        booz = _CpuMock()
        jf = _JF()

        with patch.object(
            module, "update_self_intersection_status", return_value=False
        ):
            module.evaluate_candidate(np.ones(5), run_dict, booz, jf)

        self.assertEqual(len(booz.run_code_calls), 1)
        call_iota, call_G, call_sdofs = booz.run_code_calls[0]
        self.assertEqual(call_iota, TEST_IOTA)
        self.assertEqual(call_G, TEST_G0)
        self.assertIsNone(call_sdofs)
        np.testing.assert_array_equal(booz.surface.x, sdofs_warm)

    def test_evaluate_candidate_failure_restores_cpu_state_on_legacy_path(self):
        """Legacy CPU warm-start path must restore state after failed evaluation."""
        module = self.load_module()
        CpuBoozerSurface = module.BoozerSurface

        class _Surface:
            def __init__(self):
                self.x = np.array([9.0, 8.0, 7.0])

            def volume(self):
                return 1.0

            def is_self_intersecting(self):
                return False

        class _CpuMock(CpuBoozerSurface):
            supports_explicit_surface_warm_start = False

            def __init__(self):
                self.surface = _Surface()
                self.res = {
                    "success": True,
                    "iter": 1,
                    "iota": -1.0,
                    "G": -2.0,
                }
                self.run_code_calls = []

            def run_code(self, iota, G=None, *, sdofs=None):
                self.run_code_calls.append((iota, G, sdofs))
                self.surface.x = np.array([-9.0, -8.0, -7.0])
                self.res["success"] = False
                self.res["iota"] = 99.0
                self.res["G"] = 77.0
                return self.res

        class _JF:
            x = np.zeros(5)

            def J(self):
                raise AssertionError("JF.J must not be called on failed solve")

            def dJ(self):
                raise AssertionError("JF.dJ must not be called on failed solve")

        last_J = 12.0
        last_dJ = np.arange(5.0)
        expected_failure_value = last_J + max(abs(last_J), 1.0)
        sdofs_warm = np.array([1.0, 2.0, 3.0])
        run_dict = self._make_candidate_run_dict(sdofs_warm)
        run_dict["J"] = last_J
        run_dict["dJ"] = last_dJ.copy()
        booz = _CpuMock()
        jf = _JF()

        with patch.object(
            module, "update_self_intersection_status", return_value=False
        ):
            J_out, dJ_out = module.evaluate_candidate(np.ones(5), run_dict, booz, jf)

        self.assertEqual(len(booz.run_code_calls), 1)
        call_iota, call_G, call_sdofs = booz.run_code_calls[0]
        self.assertEqual(call_iota, TEST_IOTA)
        self.assertEqual(call_G, TEST_G0)
        self.assertIsNone(call_sdofs)
        self.assertEqual(J_out, expected_failure_value)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        np.testing.assert_array_equal(booz.surface.x, sdofs_warm)
        self.assertEqual(booz.res["iota"], TEST_IOTA)
        self.assertEqual(booz.res["G"], TEST_G0)
        self.assertFalse(booz.res["success"])

    def test_snapshot_restore_round_trip(self):
        """Wave 1.4: snapshot → restore → snapshot produces identical arrays."""
        module = self.load_module()

        class _Surface:
            def __init__(self):
                self._x = np.array([10.0, 20.0, 30.0])

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, val):
                self._x = np.asarray(val)

        class _BoozerSurface:
            def __init__(self):
                self.surface = _Surface()
                self.res = {"success": True, "iter": 1, "iota": 0.15, "G": 1.0}

        class _JF:
            def __init__(self):
                self._x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, val):
                self._x = np.asarray(val)

            def J(self):
                return 42.0

            def dJ(self):
                return np.array([0.1, 0.2, 0.3, 0.4, 0.5])

        class _Curve:
            def __init__(self, scale):
                self._scale = scale

            def gamma(self):
                return np.ones((10, 3)) * self._scale

            def gammadash(self):
                return np.zeros((10, 3)) + self._scale * 0.1

        class _Current:
            def __init__(self, val):
                self._val = val

            def get_value(self):
                return self._val

        class _Coil:
            def __init__(self, current_val, curve_scale):
                self.curve = _Curve(curve_scale)
                self.current = _Current(current_val)

        class _BS:
            def __init__(self):
                self.coils = [
                    _Coil(100.0, 1.0),
                    _Coil(200.0, 2.0),
                    _Coil(300.0, 3.0),
                ]

        jf = _JF()
        booz = _BoozerSurface()
        bs_obj = _BS()

        with patch.object(
            module, "surface_self_intersection_check_available", return_value=True
        ):
            dofs1, rd1, sc1 = module.snapshot_to_pytree(
                jf, booz, bs_obj, num_tf_coils=2
            )

        # Verify snapshot extracted the expected state
        np.testing.assert_array_equal(dofs1, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(rd1["sdofs"], [10, 20, 30])
        self.assertEqual(rd1["iota"], 0.15)
        self.assertEqual(rd1["G"], 1.0)
        self.assertEqual(rd1["J"], 42.0)
        self.assertEqual(rd1["initial_objective"], 42.0)
        self.assertEqual(rd1["it"], 1)
        self.assertEqual(rd1["lscount"], 0)
        self.assertTrue(rd1["self_intersection_check_available"])

        # Verify TF gamma frozen as static arrays
        self.assertEqual(sc1["num_tf_coils"], 2)
        self.assertEqual(len(sc1["tf_gamma"]), 2)
        np.testing.assert_array_equal(sc1["tf_gamma"][0], np.ones((10, 3)))
        np.testing.assert_array_equal(sc1["tf_gamma"][1], np.ones((10, 3)) * 2)
        self.assertAlmostEqual(sc1["tf_currents"][0], 100.0)
        self.assertAlmostEqual(sc1["tf_currents"][1], 200.0)

        # Mutate the Optimizable graph to simulate post-optimization state
        jf.x = np.array([99.0, 99.0, 99.0, 99.0, 99.0])
        booz.surface.x = np.array([99.0, 99.0, 99.0])
        booz.res["iota"] = 999.0
        booz.res["G"] = 999.0

        # Restore from the snapshot
        module.restore_from_pytree(jf, booz, rd1, coil_dofs=dofs1)

        # Verify restore wrote back correctly
        np.testing.assert_array_equal(jf.x, [1, 2, 3, 4, 5])
        np.testing.assert_array_equal(booz.surface.x, [10, 20, 30])
        self.assertEqual(booz.res["iota"], 0.15)
        self.assertEqual(booz.res["G"], 1.0)

        # Re-snapshot and verify round-trip identity
        with patch.object(
            module, "surface_self_intersection_check_available", return_value=True
        ):
            dofs2, rd2, sc2 = module.snapshot_to_pytree(
                jf, booz, bs_obj, num_tf_coils=2
            )

        np.testing.assert_array_equal(dofs1, dofs2)
        np.testing.assert_array_equal(rd1["sdofs"], rd2["sdofs"])
        self.assertEqual(rd1["iota"], rd2["iota"])
        self.assertEqual(rd1["G"], rd2["G"])
        self.assertEqual(rd1["J"], rd2["J"])
        self.assertEqual(rd1["initial_objective"], rd2["initial_objective"])
        np.testing.assert_array_equal(rd1["dJ"], rd2["dJ"])
        np.testing.assert_array_equal(rd1["x_prev"], rd2["x_prev"])
        for key in [
            "it",
            "lscount",
            "intersecting",
            "self_intersection_check_available",
        ]:
            self.assertEqual(rd1[key], rd2[key], msg=f"mismatch on {key}")
        for i in range(2):
            np.testing.assert_array_equal(sc1["tf_gamma"][i], sc2["tf_gamma"][i])
            np.testing.assert_array_equal(
                sc1["tf_gammadash"][i], sc2["tf_gammadash"][i]
            )
            self.assertEqual(sc1["tf_currents"][i], sc2["tf_currents"][i])

    def test_resolve_single_stage_outer_optimizer_initial_dofs_uses_target_lane_bs_space(
        self,
    ):
        module = self.load_module()

        class _JF:
            x = np.array([1.0, 2.0, 3.0])

        class _BS:
            x = np.array([9.0, 8.0])

        np.testing.assert_allclose(
            module.resolve_single_stage_outer_optimizer_initial_dofs(
                _JF(),
                _BS(),
                use_target_lane=False,
            ),
            np.array([1.0, 2.0, 3.0]),
        )
        target_lane_dofs = module.resolve_single_stage_outer_optimizer_initial_dofs(
            _JF(),
            _BS(),
            use_target_lane=True,
        )
        self.assertIsInstance(target_lane_dofs, jax.Array)
        np.testing.assert_allclose(
            module._single_stage_optimizer_dofs_array(target_lane_dofs),
            np.array([9.0, 8.0]),
        )

    def test_single_stage_optimizer_dofs_array_hostifies_target_lane_state_explicitly(
        self,
    ):
        module = self.load_module()
        host_calls = []
        original_host_array = module.host_array

        def counted_host_array(value, *, dtype=np.float64):
            host_calls.append((value, dtype))
            return original_host_array(value, dtype=dtype)

        target_lane_dofs = module.SingleStageOuterOptimizerState(
            coil_dofs=jax.device_put(np.array([1.0, -2.0], dtype=np.float64))
        )

        with patch.object(module, "host_array", counted_host_array):
            resolved = module._single_stage_optimizer_dofs_array(target_lane_dofs)

        self.assertEqual(len(host_calls), 1)
        self.assertIsInstance(host_calls[0][0], jax.Array)
        self.assertEqual(np.dtype(host_calls[0][1]), np.dtype(np.float64))
        np.testing.assert_allclose(resolved, np.array([1.0, -2.0]))

    def test_build_traceable_single_stage_outer_objective_config_hostifies_vessel_gamma(
        self,
    ):
        module = self.load_module()
        host_calls = []
        original_host_array = module.host_array
        banana_curve = object()
        vessel_gamma = jax.device_put(np.arange(12.0, dtype=np.float64).reshape(2, 2, 3))

        def counted_host_array(value, *, dtype=np.float64):
            host_calls.append((value, dtype))
            return original_host_array(value, dtype=dtype)

        boozer_surface = types.SimpleNamespace(surface=types.SimpleNamespace(nfp=2))
        bs = types.SimpleNamespace(coils=[types.SimpleNamespace(curve=banana_curve)])
        vessel_surface = types.SimpleNamespace(surface_spec=lambda: object())

        with patch.object(module, "host_array", counted_host_array), patch(
            "simsopt.jax_core.surface_rzfourier.surface_rz_fourier_gamma_from_spec",
            return_value=vessel_gamma,
        ):
            config = module.build_traceable_single_stage_outer_objective_config(
                boozer_surface,
                bs,
                banana_curve,
                vessel_surface,
                non_qs_weight=1.0,
                residual_weight=2.0,
                iota_weight=3.0,
                length_weight=4.0,
                length_target=5.0,
                curve_curve_weight=6.0,
                curve_curve_threshold=0.05,
                curve_surface_weight=7.0,
                curve_surface_threshold=0.02,
                surface_vessel_weight=8.0,
                surface_vessel_threshold=0.04,
                curvature_weight=9.0,
                curvature_threshold=40.0,
            )

        self.assertEqual(config["banana_curve_index"], 0)
        self.assertEqual(len(host_calls), 1)
        self.assertIsInstance(host_calls[0][0], jax.Array)
        self.assertEqual(np.dtype(host_calls[0][1]), np.dtype(np.float64))
        np.testing.assert_allclose(
            config["vessel_gamma"],
            np.arange(12.0, dtype=np.float64).reshape(4, 3),
        )

    def test_restore_from_pytree_uses_custom_coil_dof_setter_when_provided(self):
        module = self.load_module()

        class _JF:
            def __init__(self):
                self._x = np.array([1.0, 2.0])

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, value):
                self._x = np.asarray(value)

        class _Surface:
            def __init__(self):
                self.x = np.array([10.0, 20.0])

        class _Booz:
            def __init__(self):
                self.surface = _Surface()
                self.res = {"iota": 0.1, "G": 2.0}

        jf = _JF()
        booz = _Booz()
        run_dict = {"sdofs": np.array([3.0, 4.0]), "iota": 0.2, "G": 5.0}
        captured = {}

        def fake_setter(value):
            captured["x"] = np.asarray(value)

        module.restore_from_pytree(
            jf,
            booz,
            run_dict,
            coil_dofs=np.array([6.0, 7.0]),
            apply_coil_dofs=fake_setter,
        )

        np.testing.assert_allclose(captured["x"], np.array([6.0, 7.0]))
        np.testing.assert_allclose(jf.x, np.array([1.0, 2.0]))
        np.testing.assert_allclose(booz.surface.x, np.array([3.0, 4.0]))
        self.assertEqual(booz.res["iota"], 0.2)
        self.assertEqual(booz.res["G"], 5.0)

    def test_restore_without_coil_dofs_leaves_jf_unchanged(self):
        """restore_from_pytree with coil_dofs=None must not touch JF.x."""
        module = self.load_module()

        class _Surface:
            def __init__(self):
                self._x = np.zeros(2)

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, val):
                self._x = np.asarray(val)

        class _BoozerSurface:
            def __init__(self):
                self.surface = _Surface()
                self.res = {"iota": 0.0, "G": 0.0}

        class _JF:
            def __init__(self):
                self._x = np.array([7.0, 8.0])

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, val):
                self._x = np.asarray(val)

        jf = _JF()
        booz = _BoozerSurface()
        run_dict = {"sdofs": np.array([1.0, 2.0]), "iota": 0.15, "G": 1.0}

        module.restore_from_pytree(jf, booz, run_dict)

        np.testing.assert_array_equal(jf.x, [7.0, 8.0])
        np.testing.assert_array_equal(booz.surface.x, [1.0, 2.0])
        self.assertEqual(booz.res["iota"], 0.15)
        self.assertEqual(booz.res["G"], 1.0)

    def test_snapshot_records_unavailable_self_intersection_backend(self):
        """snapshot_to_pytree must propagate False when backend is absent."""
        module = self.load_module()

        class _Surface:
            def __init__(self):
                self._x = np.zeros(2)

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, val):
                self._x = np.asarray(val)

        class _BoozerSurface:
            def __init__(self):
                self.surface = _Surface()
                self.res = {"success": True, "iota": 0.15, "G": 1.0}

        class _JF:
            def __init__(self):
                self._x = np.zeros(3)

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, val):
                self._x = np.asarray(val)

            def J(self):
                return 0.0

            def dJ(self):
                return np.zeros(3)

        class _Coil:
            def __init__(self):
                self.curve = type(
                    "C",
                    (),
                    {
                        "gamma": lambda self: np.zeros((5, 3)),
                        "gammadash": lambda self: np.zeros((5, 3)),
                    },
                )()
                self.current = type("I", (), {"get_value": lambda self: 1.0})()

        class _BS:
            coils = [_Coil()]

        with patch.object(
            module, "surface_self_intersection_check_available", return_value=False
        ):
            _, rd, _ = module.snapshot_to_pytree(
                _JF(), _BoozerSurface(), _BS(), num_tf_coils=1
            )

        self.assertFalse(rd["self_intersection_check_available"])


class BoozerFallbackLBFGSBTests(unittest.TestCase):
    """Issue #2: elevated-J fallback must not flush L-BFGS-B Hessian memory."""

    def test_elevated_j_stale_gradient_preserves_bfgs_memory(self):
        from scipy.optimize import minimize

        def rosenbrock(x):
            f = sum(
                100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2
                for i in range(len(x) - 1)
            )
            g = np.zeros_like(x)
            for i in range(len(x) - 1):
                g[i] += -400 * x[i] * (x[i + 1] - x[i] ** 2) - 2 * (1 - x[i])
                g[i + 1] += 200 * (x[i + 1] - x[i] ** 2)
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

        res = minimize(
            fun_with_fallback,
            x0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 500, "maxcor": 10},
        )

        self.assertTrue(res.success, f"L-BFGS-B did not converge: {res.message}")
        self.assertGreater(res.hess_inv.n_corrs, 0)


def _compiled_segment_segment_distance():
    from simsopt.jax_core import segment_segment_distance_pure

    return jax.jit(segment_segment_distance_pure)


_SEGMENT_SEGMENT_DISTANCE = _compiled_segment_segment_distance()


def _segment_segment_distance(P1, P2, Q1, Q2):
    return float(
        np.asarray(
            _SEGMENT_SEGMENT_DISTANCE(
                np.asarray(P1, dtype=np.float64),
                np.asarray(P2, dtype=np.float64),
                np.asarray(Q1, dtype=np.float64),
                np.asarray(Q2, dtype=np.float64),
            )
        )
    )


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
            np.array(p1, dtype=float),
            np.array(p2, dtype=float),
            np.array(q1, dtype=float),
            np.array(q2, dtype=float),
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
            self.assertAlmostEqual(
                d_algo,
                d_brute,
                places=9,
                msg=f"Near-parallel mismatch: algo={d_algo}, brute={d_brute}",
            )
        self.assertGreater(
            n_parallel, 900, "Not enough pairs hit the near-parallel branch"
        )

    def test_random_brute_force(self):
        """Verify against exhaustive interior + edge search on 1000 random pairs."""
        rng = np.random.RandomState(12345)
        for _ in range(1000):
            P1, P2, Q1, Q2 = rng.randn(4, 3)
            d_algo = _segment_segment_distance(P1, P2, Q1, Q2)
            d_brute = _brute_force_segment_distance(P1, P2, Q1, Q2)
            self.assertAlmostEqual(
                d_algo,
                d_brute,
                places=9,
                msg=f"Mismatch: algo={d_algo}, brute={d_brute}",
            )


class HardwareConstraintTests(unittest.TestCase):
    def test_jax_curvature_threshold_respects_floor_and_ceiling(self):
        stage2_module = load_stage2_module()
        single_stage_module = load_single_stage_example_module()

        self.assertEqual(stage2_module.resolve_curvature_threshold(10.0), 20.0)
        self.assertEqual(stage2_module.resolve_curvature_threshold(20.0), 20.0)
        self.assertEqual(stage2_module.resolve_curvature_threshold(39.0), 39.0)
        self.assertEqual(stage2_module.resolve_curvature_threshold(41.0), 40.0)
        self.assertEqual(single_stage_module.resolve_curvature_threshold(10.0), 20.0)
        self.assertEqual(single_stage_module.resolve_curvature_threshold(20.0), 20.0)
        self.assertEqual(single_stage_module.resolve_curvature_threshold(39.0), 39.0)
        self.assertEqual(single_stage_module.resolve_curvature_threshold(41.0), 40.0)

    def test_stage2_hardware_constraints_pass_at_boundaries(self):
        module = load_stage2_module()

        status = module.evaluate_stage2_hardware_constraints(
            coil_length=1.75,
            length_target=1.75,
            curve_curve_min_dist=0.05,
            cc_threshold=0.05,
            max_curvature=40.0,
            curvature_threshold=40.0,
        )

        self.assertTrue(status["success"])
        self.assertEqual(status["violations"], [])

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

    def test_stage2_hardware_constraints_report_self_intersection_violation(self):
        module = load_stage2_module()

        status = module.evaluate_stage2_hardware_constraints(
            coil_length=1.75,
            length_target=1.75,
            curve_curve_min_dist=0.05,
            cc_threshold=0.05,
            max_curvature=40.0,
            curvature_threshold=40.0,
            self_intersecting=True,
        )

        self.assertFalse(status["success"])
        self.assertEqual(status["violations"], ["banana_curve is self-intersecting"])

    def test_stage2_hardware_constraints_report_isolated_violations(self):
        module = load_stage2_module()

        for expected_label, kwargs in (
            (
                "coil_length",
                dict(
                    coil_length=1.8,
                    length_target=1.75,
                    curve_curve_min_dist=0.05,
                    cc_threshold=0.05,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "coil_coil_min_dist",
                dict(
                    coil_length=1.75,
                    length_target=1.75,
                    curve_curve_min_dist=0.04,
                    cc_threshold=0.05,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "max_curvature",
                dict(
                    coil_length=1.75,
                    length_target=1.75,
                    curve_curve_min_dist=0.05,
                    cc_threshold=0.05,
                    max_curvature=41.0,
                    curvature_threshold=40.0,
                ),
            ),
        ):
            with self.subTest(metric=expected_label):
                status = module.evaluate_stage2_hardware_constraints(**kwargs)
                self.assertFalse(status["success"])
                self.assertEqual(len(status["violations"]), 1)
                self.assertIn(expected_label, status["violations"][0])

    def test_stage2_hardware_constraints_reject_non_finite_metrics(self):
        module = load_stage2_module()

        for metric_name, bad_value, kwargs in (
            (
                "coil_length",
                np.nan,
                dict(
                    coil_length=np.nan,
                    length_target=1.75,
                    curve_curve_min_dist=0.05,
                    cc_threshold=0.05,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "coil_coil_min_dist",
                np.nan,
                dict(
                    coil_length=1.75,
                    length_target=1.75,
                    curve_curve_min_dist=np.nan,
                    cc_threshold=0.05,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "max_curvature",
                np.nan,
                dict(
                    coil_length=1.75,
                    length_target=1.75,
                    curve_curve_min_dist=0.05,
                    cc_threshold=0.05,
                    max_curvature=np.nan,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "coil_coil_min_dist",
                np.inf,
                dict(
                    coil_length=1.75,
                    length_target=1.75,
                    curve_curve_min_dist=np.inf,
                    cc_threshold=0.05,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
        ):
            with self.subTest(metric=metric_name, bad_value=bad_value):
                status = module.evaluate_stage2_hardware_constraints(**kwargs)
                self.assertFalse(status["success"])
                self.assertEqual(len(status["violations"]), 1)
                self.assertIn(metric_name, status["violations"][0])
                self.assertIn("not finite", status["violations"][0])

    def test_single_stage_hardware_constraints_pass_at_boundaries(self):
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
        )

        self.assertTrue(status["success"])
        self.assertEqual(status["violations"], [])

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

    def test_single_stage_hardware_constraints_pure_matches_host_status(self):
        module = load_single_stage_example_module()

        pure_status = module.evaluate_single_stage_hardware_constraints_pure(
            curve_curve_min_dist=jnp.asarray(0.04, dtype=jnp.float64),
            cc_dist=jnp.asarray(0.05, dtype=jnp.float64),
            curve_surface_min_dist=jnp.asarray(0.01, dtype=jnp.float64),
            cs_dist=jnp.asarray(0.02, dtype=jnp.float64),
            surface_vessel_min_dist=jnp.asarray(0.03, dtype=jnp.float64),
            ss_dist=jnp.asarray(0.04, dtype=jnp.float64),
            max_curvature=jnp.asarray(41.0, dtype=jnp.float64),
            curvature_threshold=jnp.asarray(40.0, dtype=jnp.float64),
        )
        hostified_status = module._hostify_single_stage_hardware_constraints(
            pure_status
        )
        direct_status = module.evaluate_single_stage_hardware_constraints(
            curve_curve_min_dist=0.04,
            cc_dist=0.05,
            curve_surface_min_dist=0.01,
            cs_dist=0.02,
            surface_vessel_min_dist=0.03,
            ss_dist=0.04,
            max_curvature=41.0,
            curvature_threshold=40.0,
        )

        self.assertEqual(hostified_status, direct_status)

    def test_single_stage_hardware_constraints_report_isolated_violations(self):
        module = load_single_stage_example_module()

        for expected_label, kwargs in (
            (
                "coil_coil_min_dist",
                dict(
                    curve_curve_min_dist=0.04,
                    cc_dist=0.05,
                    curve_surface_min_dist=0.02,
                    cs_dist=0.02,
                    surface_vessel_min_dist=0.04,
                    ss_dist=0.04,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "coil_surface_min_dist",
                dict(
                    curve_curve_min_dist=0.05,
                    cc_dist=0.05,
                    curve_surface_min_dist=0.01,
                    cs_dist=0.02,
                    surface_vessel_min_dist=0.04,
                    ss_dist=0.04,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "surface_vessel_min_dist",
                dict(
                    curve_curve_min_dist=0.05,
                    cc_dist=0.05,
                    curve_surface_min_dist=0.02,
                    cs_dist=0.02,
                    surface_vessel_min_dist=0.03,
                    ss_dist=0.04,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "max_curvature",
                dict(
                    curve_curve_min_dist=0.05,
                    cc_dist=0.05,
                    curve_surface_min_dist=0.02,
                    cs_dist=0.02,
                    surface_vessel_min_dist=0.04,
                    ss_dist=0.04,
                    max_curvature=41.0,
                    curvature_threshold=40.0,
                ),
            ),
        ):
            with self.subTest(metric=expected_label):
                status = module.evaluate_single_stage_hardware_constraints(**kwargs)
                self.assertFalse(status["success"])
                self.assertEqual(len(status["violations"]), 1)
                self.assertIn(expected_label, status["violations"][0])

    def test_single_stage_hardware_constraints_reject_non_finite_metrics(self):
        module = load_single_stage_example_module()

        for metric_name, bad_value, kwargs in (
            (
                "coil_coil_min_dist",
                np.nan,
                dict(
                    curve_curve_min_dist=np.nan,
                    cc_dist=0.05,
                    curve_surface_min_dist=0.02,
                    cs_dist=0.02,
                    surface_vessel_min_dist=0.04,
                    ss_dist=0.04,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "coil_surface_min_dist",
                np.nan,
                dict(
                    curve_curve_min_dist=0.05,
                    cc_dist=0.05,
                    curve_surface_min_dist=np.nan,
                    cs_dist=0.02,
                    surface_vessel_min_dist=0.04,
                    ss_dist=0.04,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "surface_vessel_min_dist",
                np.nan,
                dict(
                    curve_curve_min_dist=0.05,
                    cc_dist=0.05,
                    curve_surface_min_dist=0.02,
                    cs_dist=0.02,
                    surface_vessel_min_dist=np.nan,
                    ss_dist=0.04,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "max_curvature",
                np.nan,
                dict(
                    curve_curve_min_dist=0.05,
                    cc_dist=0.05,
                    curve_surface_min_dist=0.02,
                    cs_dist=0.02,
                    surface_vessel_min_dist=0.04,
                    ss_dist=0.04,
                    max_curvature=np.nan,
                    curvature_threshold=40.0,
                ),
            ),
            (
                "surface_vessel_min_dist",
                np.inf,
                dict(
                    curve_curve_min_dist=0.05,
                    cc_dist=0.05,
                    curve_surface_min_dist=0.02,
                    cs_dist=0.02,
                    surface_vessel_min_dist=np.inf,
                    ss_dist=0.04,
                    max_curvature=40.0,
                    curvature_threshold=40.0,
                ),
            ),
        ):
            with self.subTest(metric=metric_name, bad_value=bad_value):
                status = module.evaluate_single_stage_hardware_constraints(**kwargs)
                self.assertFalse(status["success"])
                self.assertEqual(len(status["violations"]), 1)
                self.assertIn(metric_name, status["violations"][0])
                self.assertIn("not finite", status["violations"][0])

    def test_evaluate_candidate_rejects_on_hardware_constraint_failure(self):
        module = load_single_stage_example_module()
        last_J = 12.0
        last_dJ = np.array([1.0, -1.0, 2.0])

        class _Surface:
            def __init__(self):
                self.x = np.ones(2)

            def is_self_intersecting(self):
                return False

            def volume(self):
                return 1.0

        class _BoozerSurface:
            def __init__(self):
                self.surface = _Surface()
                self.res = {
                    "success": True,
                    "iota": TEST_IOTA,
                    "G": TEST_G0,
                }

            def run_code(self, iota, G, *, sdofs=None):
                return self.res

        class _JF:
            def __init__(self):
                self.x = np.zeros(3)

            def J(self):
                raise AssertionError("JF.J must not be called on hardware failure")

            def dJ(self):
                raise AssertionError("JF.dJ must not be called on hardware failure")

        class _DistanceObjective:
            def __init__(self, distance):
                self.distance = distance

            def shortest_distance(self):
                return self.distance

        class _Curve:
            def kappa(self):
                return np.array([41.0])

        run_dict = {
            "x_prev": np.zeros(3),
            "lscount": 0,
            "sdofs": np.ones(2),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": last_J,
            "dJ": last_dJ.copy(),
        }
        booz = _BoozerSurface()
        jf = _JF()
        objectives = {
            "cc": _DistanceObjective(0.04),
            "cs": _DistanceObjective(0.03),
            "surf": _DistanceObjective(0.05),
        }
        diagnostics = {"banana_curve": _Curve()}
        module.CC_DIST = 0.05
        module.CS_DIST = 0.02
        module.SS_DIST = 0.04
        module.CURVATURE_THRESHOLD = 40.0

        with patch.object(module, "update_self_intersection_status", return_value=False):
            J_out, dJ_out = module._evaluate_candidate_impl(
                np.ones(3),
                run_dict,
                booz,
                jf,
                objectives,
                diagnostics,
            )

        self.assertEqual(J_out, 24.0)
        np.testing.assert_array_equal(dJ_out, last_dJ)
        self.assertFalse(run_dict["hardware_constraint_status"]["success"])

    def test_surface_surface_distance_exposes_shortest_distance(self):
        class _Surface(Optimizable):
            def __init__(self, gamma):
                self._gamma = np.asarray(gamma, dtype=float)
                super().__init__()

            def gamma(self):
                return self._gamma

        surf1 = _Surface([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
        surf2 = _Surface([[[0.5, 0.0, 0.0], [4.0, 0.0, 0.0]]])

        objective = SurfaceSurfaceDistance(surf1, surf2, minimum_distance=0.4)

        self.assertAlmostEqual(float(objective.shortest_distance()), 0.5, places=12)

    def test_apply_hardware_constraint_verdict_preserves_init_only_success(self):
        for loader in (load_single_stage_example_module, load_stage2_module):
            with self.subTest(module_loader=loader.__name__):
                module = loader()

                success, termination = module.apply_hardware_constraint_verdict(
                    True,
                    "init_only",
                    {"success": False, "violations": ["too close"]},
                    init_only=True,
                )

                self.assertTrue(success)
                self.assertEqual(termination, "init_only")

    def test_apply_hardware_constraint_verdict_marks_real_failures(self):
        for loader in (load_single_stage_example_module, load_stage2_module):
            with self.subTest(module_loader=loader.__name__):
                module = loader()

                success, termination = module.apply_hardware_constraint_verdict(
                    True,
                    "converged",
                    {"success": False, "violations": ["too close"]},
                    init_only=False,
                )

                self.assertFalse(success)
                self.assertEqual(
                    termination,
                    "converged; hardware_constraints_failed",
                )

    def test_sanitize_json_payload_replaces_non_finite_numbers(self):
        module = load_single_stage_example_module()

        payload = {
            "finite": 1.25,
            "nan": float("nan"),
            "inf": float("inf"),
            "nested": [np.float64(2.0), np.float64(np.nan)],
        }

        sanitized = module.sanitize_json_payload(payload)

        self.assertEqual(sanitized["finite"], 1.25)
        self.assertIsNone(sanitized["nan"])
        self.assertIsNone(sanitized["inf"])
        self.assertEqual(sanitized["nested"][0], 2.0)
        self.assertIsNone(sanitized["nested"][1])

    def test_target_lane_success_filter_cache_signature_accepts_spec_dataclasses(
        self,
    ):
        module = load_single_stage_example_module()

        def build_extraction_spec(*, current_input_end):
            curve = CurveXYZFourierSpec(
                dofs=np.asarray([1.0, 2.0], dtype=np.float64),
                quadpoints=np.asarray([0.0, 0.5], dtype=np.float64),
                order=1,
            )
            curve_map = OptimizableDofMapSpec(
                template_full_dofs=np.asarray([1.0, 2.0, 3.0], dtype=np.float64),
                owner_segments=((0, 2, 0, 2),),
                input_mode="replace",
                input_start=0,
                input_end=2,
            )
            current_map = OptimizableDofMapSpec(
                template_full_dofs=np.asarray([4.0], dtype=np.float64),
                owner_segments=((0, 1, 0, 1),),
                input_mode="replace",
                input_start=0,
                input_end=current_input_end,
            )
            symmetry = CoilSymmetrySpec(
                rotmat=np.eye(3, dtype=np.float64),
                scale=1.0,
                has_rotation=False,
            )
            return CoilSetDofExtractionSpec(
                coils=(
                    CoilDofExtractionSpec(
                        curve=curve,
                        curve_map=curve_map,
                        current_map=current_map,
                        symmetry=symmetry,
                    ),
                )
            )

        payload = {
            "coil_dof_extraction_spec": build_extraction_spec(current_input_end=1),
        }
        matching_payload = {
            "coil_dof_extraction_spec": build_extraction_spec(current_input_end=1),
        }
        changed_payload = {
            "coil_dof_extraction_spec": build_extraction_spec(current_input_end=0),
        }

        signature = module._target_lane_success_filter_cache_signature(payload)
        matching_signature = module._target_lane_success_filter_cache_signature(
            matching_payload
        )
        changed_signature = module._target_lane_success_filter_cache_signature(
            changed_payload
        )

        self.assertEqual(len(signature), 64)
        self.assertEqual(signature, matching_signature)
        self.assertNotEqual(signature, changed_signature)

    def test_stage2_write_json_file_sanitizes_non_finite_payloads(self):
        module = load_stage2_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "payload.json")
            module.write_json_file(
                output_path,
                {"nan_value": float("nan"), "inf_value": float("inf")},
            )
            with open(output_path, "r", encoding="utf-8") as infile:
                payload = json.load(infile)

        self.assertIsNone(payload["nan_value"])
        self.assertIsNone(payload["inf_value"])


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
    """Issue #31: ftol/gtol must not be None for any mpol value."""

    def test_ftol_gtol_have_defaults_for_all_mpol(self):
        module = load_single_stage_example_module()
        ftol_by_mpol = module.ftol_by_mpol
        gtol_by_mpol = module.gtol_by_mpol
        for mpol in range(1, 30):
            ftol = ftol_by_mpol.get(mpol, 1e-5 if mpol < 8 else 1e-10)
            gtol = gtol_by_mpol.get(mpol, 1e-2 if mpol < 8 else 1e-7)
            self.assertIsNotNone(ftol, f"ftol is None for mpol={mpol}")
            self.assertIsNotNone(gtol, f"gtol is None for mpol={mpol}")
            self.assertIsInstance(ftol, float, f"ftol not float for mpol={mpol}")
            self.assertIsInstance(gtol, float, f"gtol not float for mpol={mpol}")
            self.assertGreater(ftol, 0, f"ftol not positive for mpol={mpol}")
            self.assertGreater(gtol, 0, f"gtol not positive for mpol={mpol}")

    def test_defaults_match_dictionary_endpoints(self):
        module = load_single_stage_example_module()
        ftol_by_mpol = module.ftol_by_mpol
        gtol_by_mpol = module.gtol_by_mpol
        self.assertEqual(ftol_by_mpol.get(7, 1e-5 if 7 < 8 else 1e-10), 1e-5)
        self.assertEqual(ftol_by_mpol.get(19, 1e-5 if 19 < 8 else 1e-10), 1e-10)
        self.assertEqual(gtol_by_mpol.get(7, 1e-2 if 7 < 8 else 1e-7), 1e-2)
        self.assertEqual(gtol_by_mpol.get(19, 1e-2 if 19 < 8 else 1e-7), 1e-7)

    def test_source_uses_default_argument(self):
        """The deployed .get() calls must include a default, not bare .get(mpol)."""
        source = EXAMPLE_MODULE_PATH.read_text()
        self.assertNotIn("ftol_by_mpol.get(mpol)", source)
        self.assertNotIn("gtol_by_mpol.get(mpol)", source)


class ResultsEnvelopeTests(unittest.TestCase):
    def test_stage2_results_envelope_captures_contract_and_artifacts(self):
        module = load_stage2_module()
        module.build_runtime_provenance = lambda **_: {"repo_sha": "deadbeef"}
        args = types.SimpleNamespace(
            backend="jax",
            optimizer_backend="ondevice",
            least_squares_algorithm="lm",
            constraint_method="penalty",
            init_only=False,
            skip_postprocess=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            envelope = module.build_stage2_results_envelope(
                output_root=tmpdir,
                plasma_surf_filename="wout_fixture.nc",
                file_loc="/tmp/wout_fixture.nc",
                nphi=31,
                ntheta=16,
                num_quadpoints=64,
                order=2,
                field_diagnostic_stride=7,
                R0=0.915,
                s=0.24,
                banana_surf_radius=0.22,
                theta_center=0.1,
                phi_center=0.2,
                theta_width=0.3,
                phi_width=0.4,
                LENGTH_WEIGHT=5e-4,
                CC_WEIGHT=100.0,
                CURVATURE_WEIGHT=1e-4,
                SQUARED_FLUX_WEIGHT=1.0,
                LENGTH_TARGET=2.5,
                CC_THRESHOLD=0.05,
                CURVATURE_THRESHOLD=40.0,
                args=args,
                MAXITER=25,
            )

        self.assertEqual(envelope["schema_version"], 1)
        self.assertEqual(
            envelope["problem_contract"]["runtime_contract"]["constraint_method"],
            "penalty",
        )
        self.assertEqual(envelope["provenance"]["repo_sha"], "deadbeef")
        self.assertEqual(
            envelope["problem_contract"]["equilibrium"]["filename"], "wout_fixture.nc"
        )
        self.assertEqual(
            envelope["problem_contract"]["runtime_contract"]["optimizer_backend"],
            "ondevice",
        )
        self.assertTrue(
            envelope["artifacts"]["required"]["results.json"]["exists"]
        )
        self.assertFalse(
            envelope["artifacts"]["required"]["biot_savart_opt.json"]["exists"]
        )

    def test_single_stage_results_envelope_captures_seed_and_artifact_policy(self):
        module = load_single_stage_example_module()
        module.build_runtime_provenance = lambda **_: {"repo_sha": "deadbeef"}
        args = types.SimpleNamespace(
            backend="jax",
            benchmark_mode=False,
            minimal_artifacts=True,
            init_only=False,
            profile_target_lane_only=False,
            profile_target_lane_batch_size=1,
            diagnose_target_lane_gradient=False,
            diagnose_target_lane_scaled_phase1=False,
            disable_target_lane_success_filter=False,
            maxcor=16,
            outer_maxls=8,
            target_lane_outer_initial_step_size=None,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
        )
        stage2_results = {"banana_surf_radius": 0.22}

        with tempfile.TemporaryDirectory() as tmpdir:
            envelope = module.build_single_stage_results_envelope(
                output_root=tmpdir,
                plasma_surf_filename="wout_fixture.nc",
                file_loc="/tmp/wout_fixture.nc",
                mpol=6,
                ntor=6,
                nphi=127,
                ntheta=48,
                vol_target=0.15,
                iota_target=0.21,
                stage2_bs_path="/tmp/stage2/biot_savart_opt.json",
                stage2_results_path="/tmp/stage2/results.json",
                stage2_source="derived",
                stage2_results=stage2_results,
                warm_start_run_dir="/tmp/warm-start",
                warm_start_state={
                    "surface_path": "/tmp/warm-start/surf_opt.json",
                    "results_path": "/tmp/warm-start/results.json",
                },
                R0=0.915,
                s=0.24,
                order=2,
                CONSTRAINT_WEIGHT=1.0,
                CC_DIST=0.1,
                CC_WEIGHT=1.0,
                CS_DIST=0.2,
                CS_WEIGHT=2.0,
                SS_DIST=0.3,
                SURF_DIST_WEIGHT=3.0,
                CURVATURE_WEIGHT=4.0,
                CURVATURE_THRESHOLD=20.0,
                LENGTH_WEIGHT=5.0,
                RES_WEIGHT=6.0,
                IOTAS_WEIGHT=7.0,
                optimizer_backend_record="ondevice",
                boozer_optimizer_backend_record="ondevice",
                boozer_least_squares_algorithm_record="lm",
                outer_optimizer_method="lbfgs-ondevice",
                target_lane_sync_record="final-only",
                requested_experimental_target_lane_vg=False,
                use_target_lane_vg=True,
                target_lane_boozer_bfgs_tol_record=1e-6,
                target_lane_boozer_bfgs_maxiter_record=48,
                target_lane_boozer_newton_tol_record=1e-10,
                target_lane_boozer_newton_maxiter_record=8,
                args=args,
                MAXITER=300,
                write_restart_artifacts=True,
                write_full_artifacts=False,
            )

        self.assertEqual(envelope["schema_version"], 1)

    def test_single_stage_results_envelope_records_scaled_phase1_diagnostic_artifact(
        self,
    ):
        module = load_single_stage_example_module()
        module.build_runtime_provenance = lambda **_: {"repo_sha": "deadbeef"}
        args = types.SimpleNamespace(
            backend="jax",
            benchmark_mode=False,
            minimal_artifacts=True,
            init_only=False,
            profile_target_lane_only=False,
            profile_target_lane_batch_size=1,
            diagnose_target_lane_gradient=False,
            diagnose_target_lane_scaled_phase1=True,
            disable_target_lane_success_filter=False,
            maxcor=16,
            outer_maxls=8,
            target_lane_outer_initial_step_size=None,
            initial_step_scale=0.25,
            initial_step_maxiter=4,
        )
        stage2_results = {"banana_surf_radius": 0.22}

        with tempfile.TemporaryDirectory() as tmpdir:
            envelope = module.build_single_stage_results_envelope(
                output_root=tmpdir,
                plasma_surf_filename="wout_fixture.nc",
                file_loc="/tmp/wout_fixture.nc",
                mpol=6,
                ntor=6,
                nphi=127,
                ntheta=48,
                vol_target=0.15,
                iota_target=0.21,
                stage2_bs_path="/tmp/stage2/biot_savart_opt.json",
                stage2_results_path="/tmp/stage2/results.json",
                stage2_source="derived",
                stage2_results=stage2_results,
                warm_start_run_dir=None,
                warm_start_state=None,
                R0=0.915,
                s=0.24,
                order=2,
                CONSTRAINT_WEIGHT=1.0,
                CC_DIST=0.1,
                CC_WEIGHT=1.0,
                CS_DIST=0.2,
                CS_WEIGHT=2.0,
                SS_DIST=0.3,
                SURF_DIST_WEIGHT=3.0,
                CURVATURE_WEIGHT=4.0,
                CURVATURE_THRESHOLD=20.0,
                LENGTH_WEIGHT=5.0,
                RES_WEIGHT=6.0,
                IOTAS_WEIGHT=7.0,
                optimizer_backend_record="ondevice",
                boozer_optimizer_backend_record="ondevice",
                boozer_least_squares_algorithm_record="lm",
                outer_optimizer_method="lbfgs-ondevice",
                target_lane_sync_record="final-only",
                requested_experimental_target_lane_vg=False,
                use_target_lane_vg=True,
                target_lane_boozer_bfgs_tol_record=1e-6,
                target_lane_boozer_bfgs_maxiter_record=48,
                target_lane_boozer_newton_tol_record=1e-10,
                target_lane_boozer_newton_maxiter_record=8,
                args=args,
                MAXITER=300,
                write_restart_artifacts=False,
                write_full_artifacts=False,
            )

        self.assertIn(
            "target_lane_scaled_phase1_diagnosis.json",
            envelope["artifacts"]["required"],
        )
        self.assertTrue(
            envelope["problem_contract"]["runtime_contract"][
                "diagnose_target_lane_scaled_phase1"
            ]
        )
        self.assertEqual(envelope["provenance"]["repo_sha"], "deadbeef")
        self.assertEqual(envelope["problem_contract"]["stage2_seed"]["order"], 2)
        self.assertEqual(
            envelope["problem_contract"]["runtime_contract"]["outer_optimizer_method"],
            "lbfgs-ondevice",
        )
        self.assertFalse(
            envelope["artifacts"]["policy"]["write_restart_artifacts"]
        )
        self.assertFalse(envelope["artifacts"]["policy"]["write_full_artifacts"])

    def test_single_stage_results_envelope_records_invalid_state_event_policy(self):
        module = load_single_stage_example_module()
        module.build_runtime_provenance = lambda **_: {"repo_sha": "deadbeef"}
        args = types.SimpleNamespace(
            backend="jax",
            benchmark_mode=False,
            minimal_artifacts=True,
            init_only=False,
            profile_target_lane_only=False,
            profile_target_lane_batch_size=1,
            diagnose_target_lane_gradient=False,
            diagnose_target_lane_scaled_phase1=False,
            record_target_lane_invalid_state_events=True,
            disable_target_lane_success_filter=False,
            maxcor=16,
            outer_maxls=8,
            target_lane_outer_initial_step_size=None,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
        )
        stage2_results = {"banana_surf_radius": 0.22}

        with tempfile.TemporaryDirectory() as tmpdir:
            envelope = module.build_single_stage_results_envelope(
                output_root=tmpdir,
                plasma_surf_filename="wout_fixture.nc",
                file_loc="/tmp/wout_fixture.nc",
                mpol=6,
                ntor=6,
                nphi=127,
                ntheta=48,
                vol_target=0.15,
                iota_target=0.21,
                stage2_bs_path="/tmp/stage2/biot_savart_opt.json",
                stage2_results_path="/tmp/stage2/results.json",
                stage2_source="derived",
                stage2_results=stage2_results,
                warm_start_run_dir=None,
                warm_start_state=None,
                R0=0.915,
                s=0.24,
                order=2,
                CONSTRAINT_WEIGHT=1.0,
                CC_DIST=0.1,
                CC_WEIGHT=1.0,
                CS_DIST=0.2,
                CS_WEIGHT=2.0,
                SS_DIST=0.3,
                SURF_DIST_WEIGHT=3.0,
                CURVATURE_WEIGHT=4.0,
                CURVATURE_THRESHOLD=20.0,
                LENGTH_WEIGHT=5.0,
                RES_WEIGHT=6.0,
                IOTAS_WEIGHT=7.0,
                optimizer_backend_record="ondevice",
                boozer_optimizer_backend_record="ondevice",
                boozer_least_squares_algorithm_record="lm",
                outer_optimizer_method="lbfgs-ondevice",
                target_lane_sync_record="final-only",
                requested_experimental_target_lane_vg=False,
                use_target_lane_vg=True,
                target_lane_boozer_bfgs_tol_record=1e-6,
                target_lane_boozer_bfgs_maxiter_record=48,
                target_lane_boozer_newton_tol_record=1e-10,
                target_lane_boozer_newton_maxiter_record=8,
                args=args,
                MAXITER=300,
                write_restart_artifacts=False,
                write_full_artifacts=False,
            )

        self.assertTrue(
            envelope["problem_contract"]["runtime_contract"][
                "record_target_lane_invalid_state_events"
            ]
        )


if __name__ == "__main__":
    unittest.main()
