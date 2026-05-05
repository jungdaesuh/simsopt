import copy
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
from simsopt.field.biotsavart_jax_backend import (
    SingleStageRuntimeSpecBiotSavartJAX,
)
from simsopt.jax_core.specs import (
    CoilDofExtractionSpec,
    CoilSetDofExtractionSpec,
    CoilSymmetrySpec,
    CurveXYZFourierSpec,
    OptimizableDofMapSpec,
)
from simsopt.objectives.utilities import forward_backward
from .surface_test_helpers import get_surface


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
    "optimizer_backend='ondevice', optimizer_backend='scipy-jax', or "
    "optimizer_backend='scipy-jax-fullgraph'"
)
_SINGLE_STAGE_CPU_ONLY_SCIPY = (
    "single-stage outer loop CPU/reference lane only supports optimizer_backend='scipy'"
)
_OPTIMIZER_BACKEND_INVALID = (
    "optimizer_backend must be one of: scipy, ondevice, scipy-jax, "
    "scipy-jax-fullgraph."
)


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
        self.stellsym = True
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
    def _jax_runtime_seed_spec_field_kwargs(module):
        return {
            "coil_dof_extraction_spec": module.jax_specs.make_coil_set_dof_extraction_spec(
                ()
            ),
            "coil_dofs": np.asarray([], dtype=np.float64),
            "num_tf_coils": 0,
            "banana_curve_index": 0,
            "tf_current_A": 0.0,
            "banana_current_A": 0.0,
            "stage2_seed": {
                "major_radius": 1.0,
                "toroidal_flux": 0.5,
                "order": 2,
                "banana_surf_radius": 0.22,
            },
        }

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
            "banana_current_A": 123.0,
            "field_error": 0.013,
            "final_volume": 6.5,
            "final_iota": 0.21,
            "curve_curve_min_dist": 0.8 if include_distance_metrics else None,
            "curve_surface_min_dist": 0.9 if include_distance_metrics else None,
            "surface_vessel_min_dist": 1.0 if include_distance_metrics else None,
        }

    @staticmethod
    def _make_reporting_runtime_builder(
        captured,
        runtime_summary,
        *,
        objective_value=1.25,
        objective_grad=None,
        forward_result=None,
    ):
        resolved_grad = (
            np.asarray(objective_grad, dtype=np.float64)
            if objective_grad is not None
            else np.array([0.3, -0.4], dtype=np.float64)
        )

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

            def _value_and_grad(coil_dofs):
                del coil_dofs
                captured["value_and_grad_called"] = True
                return (
                    jnp.asarray(objective_value, dtype=jnp.float64),
                    jnp.asarray(resolved_grad, dtype=jnp.float64),
                )

            def _objective(coil_dofs):
                del coil_dofs
                captured["objective_called"] = True
                return jnp.asarray(objective_value, dtype=jnp.float64)

            runtime_bundle = {
                "objective": _objective,
                "reporting_metrics": _reporting_metrics,
                "value_and_grad": _value_and_grad,
            }
            if forward_result is not None:
                runtime_bundle["forward_result"] = forward_result
            return runtime_bundle

        return _runtime_builder

    @staticmethod
    def _make_seeded_value_and_grad_builder(
        *,
        value_and_grad,
        optimizer_initial_value_and_grad,
        captured=None,
    ):
        def _seeded_builder(
            boozer_surface,
            bs,
            iota_target,
            *,
            outer_objective_config=None,
            success_filter=None,
        ):
            del boozer_surface, bs, iota_target
            if captured is not None:
                captured.append((outer_objective_config, success_filter))
            return types.SimpleNamespace(
                value_and_grad=value_and_grad,
                optimizer_initial_value_and_grad=optimizer_initial_value_and_grad,
            )

        return _seeded_builder

    @staticmethod
    def _make_fake_target_lane_accepted_step_summary():
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

    @staticmethod
    def _make_fake_target_lane_state_sync(
        captured,
        *,
        sdofs=None,
        iota=None,
        G=None,
    ):
        def fake_sync(state, x, *, benchmark_mode, update_run_state=True):
            captured["sync"] = {
                "state": state,
                "x": np.asarray(x),
                "benchmark_mode": benchmark_mode,
                "update_run_state": update_run_state,
            }
            if update_run_state and sdofs is not None:
                state["sdofs"] = np.asarray(sdofs)
            if update_run_state and iota is not None:
                state["iota"] = iota
            if update_run_state and G is not None:
                state["G"] = G
            return (
                SingleStageExampleTests._make_fake_target_lane_accepted_step_summary()
            )

        return fake_sync

    @staticmethod
    def _make_fake_accept_step_capture(captured):
        def fake_accept_step(
            state,
            booz,
            objective,
            bs,
            objectives,
            diagnostics,
            log_path,
            *,
            objective_value=None,
            objective_grad=None,
        ):
            captured["accept"] = {
                "state": state,
                "booz": booz,
                "objective": objective,
                "bs": bs,
                "objectives": objectives,
                "diagnostics": diagnostics,
                "log_path": log_path,
                "objective_value": objective_value,
                "objective_grad": objective_grad,
            }

        return fake_accept_step

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
        fake_jax_module.build_boozer_surface_runtime_state = lambda surface: {
            "mpol": surface.mpol,
            "ntor": surface.ntor,
            "nfp": surface.nfp,
            "stellsym": surface.stellsym,
            "quadpoints_phi": np.asarray(surface.quadpoints_phi).copy(),
            "quadpoints_theta": np.asarray(surface.quadpoints_theta).copy(),
        }
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
            if optimizer_backend not in {
                "scipy",
                "ondevice",
                "scipy-jax",
                "scipy-jax-fullgraph",
            }:
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
            if optimizer_backend not in {
                "scipy",
                "ondevice",
                "scipy-jax",
                "scipy-jax-fullgraph",
            }:
                raise ValueError(_OPTIMIZER_BACKEND_INVALID)
            if field_backend != "jax" or optimizer_backend not in {
                "ondevice",
                "scipy-jax",
                "scipy-jax-fullgraph",
            }:
                raise ValueError(f"the {_SINGLE_STAGE_JAX_ONLY_ONDEVICE}.")
            require_target_backend_x64(optimizer_backend)
            method = "lbfgs-ondevice"
            if optimizer_backend == "scipy-jax":
                method = "lbfgs-scipy-jax"
            if optimizer_backend == "scipy-jax-fullgraph":
                method = "lbfgs-scipy-jax-fullgraph"
            return TargetOptimizerContract(method)

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

        with patch.object(
            module,
            "evaluate_surface_self_intersection",
            return_value=(False, True),
        ), self.patch_initialize_boozer_surface_jax(module, fake_boozer_surface_jax):
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
        self.assertNotIn("materialize_dense_linearization", boozer_surface.options)
        self.assertEqual(
            type(boozer_surface.surface).__name__,
            "DeferredSurfaceXYZTensorFourier",
        )
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 0)
        self.assertEqual(boozer_surface.surface_runtime_state["mpol"], TEST_MPOL)
        self.assertEqual(boozer_surface.surface_runtime_state["ntor"], TEST_NTOR)
        np.testing.assert_allclose(
            boozer_surface.surface_runtime_state["quadpoints_phi"],
            surf_prev.quadpoints_phi,
        )

    def test_initialize_boozer_surface_prewarms_supported_jax_self_intersection(
        self,
    ):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        timings = {}
        fake_boozer_surface_jax = self.build_fake_boozer_surface_jax_class(
            record_run_calls=False
        )

        with patch.object(
            module,
            "_supported_surface_self_intersection_inputs",
            return_value={"supported": True},
        ), patch.object(
            module,
            "prewarm_supported_surface_self_intersection",
            return_value=False,
        ) as prewarm, patch.object(
            module,
            "evaluate_surface_self_intersection",
            return_value=(False, True),
        ), self.patch_initialize_boozer_surface_jax(module, fake_boozer_surface_jax):
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
                timings_out=timings,
            )

        prewarm.assert_called_once()
        self.assertIn("jax_compile_prewarm_self_intersection_s", timings)

    def test_initialize_boozer_surface_jax_backend_materializes_surface_only_on_host_use(
        self,
    ):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        fake_boozer_surface_jax = self.build_fake_boozer_surface_jax_class(
            record_run_calls=False
        )

        with patch.object(
            module,
            "evaluate_surface_self_intersection",
            return_value=(False, True),
        ), self.patch_initialize_boozer_surface_jax(module, fake_boozer_surface_jax):
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
            self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 0)
            boozer_surface.surface.set_dofs(
                np.zeros(
                    len(module.stellsym_scatter_indices(TEST_MPOL, TEST_NTOR)),
                    dtype=np.float64,
                )
            )
            materialized_surface = boozer_surface.surface._materialize_surface()
            self.assertIs(
                materialized_surface, FakeSurfaceXYZTensorFourier.instances[0]
            )
        np.testing.assert_allclose(
            materialized_surface.get_dofs(),
            np.asarray(boozer_surface.surface.get_dofs()),
        )

    def test_initialize_boozer_surface_jax_exact_backend_uses_deferred_surface(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        fake_boozer_surface_jax = self.build_fake_boozer_surface_jax_class(
            record_run_calls=True
        )

        with patch.object(
            module,
            "evaluate_surface_self_intersection",
            return_value=(False, True),
        ), self.patch_initialize_boozer_surface_jax(module, fake_boozer_surface_jax):
            boozer_surface = module.initialize_boozer_surface(
                surf_prev,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                bs=object(),
                vol_target=TEST_VOL_TARGET,
                constraint_weight=None,
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
        self.assertEqual(
            type(boozer_surface.surface).__name__,
            "DeferredSurfaceXYZTensorFourier",
        )
        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 0)
        self.assertEqual(boozer_surface.surface_runtime_state["mpol"], TEST_MPOL)
        self.assertEqual(boozer_surface.surface_runtime_state["ntor"], TEST_NTOR)
        np.testing.assert_allclose(
            boozer_surface.surface_runtime_state["quadpoints_theta"],
            np.linspace(0, 1, 2 * TEST_MPOL + 1, endpoint=False),
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

    def test_initialize_boozer_surface_cpu_sets_projected_dofs_via_set_dofs(self):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        projected_dofs = np.array([4.0, 5.0, 6.0])

        class RecordingSurface(FakeSurfaceXYZTensorFourier):
            def __init__(self, **kwargs):
                dofs = kwargs.get("dofs")
                self.received_dofs = (
                    None if dofs is None else np.asarray(dofs, dtype=np.float64)
                )
                super().__init__(**kwargs)

        with patch.object(
            module, "SurfaceXYZTensorFourier", RecordingSurface
        ), patch.object(module, "Volume", FakeVolume), patch.object(
            module, "BoozerSurface", RecordingCPUBoozerSurface
        ), patch.object(
            module,
            "project_surface_dofs_to_resolution",
            return_value=projected_dofs,
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
            )

        self.assertIsInstance(boozer_surface, RecordingCPUBoozerSurface)
        surface = boozer_surface.surface
        self.assertIsNone(surface.received_dofs)
        np.testing.assert_array_equal(surface.get_dofs(), projected_dofs)

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

    def test_initialize_boozer_surface_limited_memory_disables_dense_linearization(
        self,
    ):
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
                boozer_limited_memory=True,
            )

        options = fake_boozer_surface_jax.instances[0].options
        self.assertIs(options["materialize_dense_linearization"], False)
        self.assertIs(options["force_ondevice_limited_memory"], True)

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

    def test_initialize_boozer_surface_skips_self_intersection_after_failed_solve(
        self,
    ):
        module = self.load_module()
        surf_prev = FakeSurfPrev()
        base_boozer_surface_jax = self.build_fake_boozer_surface_jax_class(
            record_run_calls=False
        )

        class FailingBoozerSurfaceJAX(base_boozer_surface_jax):
            instances = []

            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.res = {
                    "success": False,
                    "iter": 0,
                    "iota": TEST_IOTA,
                    "G": TEST_G0,
                }

        with patch.object(
            module,
            "evaluate_surface_self_intersection",
            side_effect=AssertionError(
                "self-intersection check must not run after a failed Boozer solve"
            ),
        ), self.patch_initialize_boozer_surface_jax(module, FailingBoozerSurfaceJAX):
            with self.assertRaisesRegex(
                RuntimeError,
                "Something went wrong with the Boozer solve",
            ):
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
                )

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

    def test_resolve_target_lane_boozer_init_base_overrides_uses_target_lane_defaults(
        self,
    ):
        module = self.load_module()

        overrides = module.resolve_target_lane_boozer_init_base_overrides(
            field_backend="jax",
            optimizer_backend="ondevice",
            boozer_limited_memory=False,
            target_lane_boozer_bfgs_tol=1.0e-8,
            target_lane_boozer_bfgs_maxiter=None,
            target_lane_boozer_newton_tol=None,
            target_lane_boozer_newton_maxiter=12,
        )

        self.assertEqual(
            overrides,
            {
                "least_squares_algorithm_override": None,
                "bfgs_tol_override": 1.0e-8,
                "bfgs_maxiter_override": None,
                "newton_tol_override": 1.0e-8,
                "newton_maxiter_override": 12,
            },
        )

    def test_resolve_target_lane_boozer_init_base_overrides_floors_bfgs_tol(
        self,
    ):
        module = self.load_module()

        overrides = module.resolve_target_lane_boozer_init_base_overrides(
            field_backend="jax",
            optimizer_backend="ondevice",
            boozer_limited_memory=False,
            target_lane_boozer_bfgs_tol=3.0e-6,
            target_lane_boozer_bfgs_maxiter=None,
            target_lane_boozer_newton_tol=None,
            target_lane_boozer_newton_maxiter=None,
        )

        self.assertEqual(overrides["bfgs_tol_override"], 1.0e-8)
        self.assertEqual(overrides["newton_tol_override"], 1.0e-8)

    def test_resolve_target_lane_boozer_init_base_overrides_is_empty_off_target_lane(
        self,
    ):
        module = self.load_module()

        overrides = module.resolve_target_lane_boozer_init_base_overrides(
            field_backend="cpu",
            optimizer_backend="scipy",
            boozer_limited_memory=False,
            target_lane_boozer_bfgs_tol=1.0e-8,
            target_lane_boozer_bfgs_maxiter=48,
            target_lane_boozer_newton_tol=1.0e-10,
            target_lane_boozer_newton_maxiter=12,
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

    def test_resolve_target_lane_boozer_init_base_overrides_floors_bfgs_maxiter(
        self,
    ):
        module = self.load_module()

        overrides = module.resolve_target_lane_boozer_init_base_overrides(
            field_backend="jax",
            optimizer_backend="ondevice",
            boozer_limited_memory=False,
            target_lane_boozer_bfgs_tol=1.0e-8,
            target_lane_boozer_bfgs_maxiter=48,
            target_lane_boozer_newton_tol=None,
            target_lane_boozer_newton_maxiter=None,
        )

        self.assertEqual(overrides["bfgs_maxiter_override"], 128)
        self.assertEqual(overrides["newton_tol_override"], 1.0e-8)

    def test_resolve_target_lane_boozer_init_base_overrides_skips_full_memory_newton_floor_for_lbfgs(
        self,
    ):
        module = self.load_module()

        overrides = module.resolve_target_lane_boozer_init_base_overrides(
            field_backend="jax",
            optimizer_backend="ondevice",
            boozer_limited_memory=True,
            target_lane_boozer_bfgs_tol=None,
            target_lane_boozer_bfgs_maxiter=None,
            target_lane_boozer_newton_tol=None,
            target_lane_boozer_newton_maxiter=None,
        )

        self.assertIsNone(overrides["bfgs_tol_override"])
        self.assertIsNone(overrides["newton_tol_override"])

    def test_resolve_warm_start_boozer_init_overrides_keeps_explicit_surface_algorithm(
        self,
    ):
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

    def test_resolve_warm_start_boozer_init_overrides_uses_quasi_newton_for_legacy_path(
        self,
    ):
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

    def test_resolve_warm_start_boozer_init_overrides_preserves_explicit_algorithm(
        self,
    ):
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

        with patch.object(
            module, "jax_solver_stage_callback_supported", return_value=False
        ):
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
            with patch.object(
                module,
                "jax_solver_stage_callback_supported",
                wraps=module.jax_solver_stage_callback_supported,
            ):
                with self.patch_initialize_boozer_surface_jax(
                    module, FakeBoozerSurfaceJAX
                ):
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
                        on_stage=lambda label, **extra: stage_events.append(
                            (label, extra)
                        ),
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

    def test_build_event_progress_recorder_preserves_event_history(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, "outer_optimizer_progress.json")
            record_event = module.build_event_progress_recorder(progress_path)
            record_event("phase1_started", phase="phase1", maxiter=1)
            record_event("phase1_progress_iter_1", phase="phase1", iteration=1)

            with open(progress_path, encoding="utf-8") as infile:
                payload = json.load(infile)

        self.assertEqual(payload["current_event"], "phase1_progress_iter_1")
        self.assertEqual(payload["event_count"], 2)
        self.assertEqual(
            [event["label"] for event in payload["events"]],
            ["phase1_started", "phase1_progress_iter_1"],
        )
        self.assertEqual(
            [event["event_index"] for event in payload["events"]],
            [0, 1],
        )
        self.assertGreaterEqual(payload["events"][0]["event_elapsed_s"], 0.0)
        self.assertEqual(payload["events"][0]["phase"], "phase1")
        self.assertEqual(payload["events"][1]["iteration"], 1)

    def test_build_event_progress_recorder_serializes_scaled_phase_optimizer_state(self):
        module = self.load_module()
        phase1_state = module.build_target_lane_scaled_outer_phase_state(
            np.array([10.0, 20.0], dtype=np.float64),
            jax.device_put(np.array([1.5, -2.5], dtype=np.float64)),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            progress_path = os.path.join(tmpdir, "outer_optimizer_progress.json")
            record_event = module.build_event_progress_recorder(progress_path)
            record_event(
                "phase1_started",
                phase="phase1",
                optimizer_dofs=module._summarize_host_vector(phase1_state),
            )

            with open(progress_path, encoding="utf-8") as infile:
                payload = json.load(infile)

        self.assertEqual(payload["current_event"], "phase1_started")
        np.testing.assert_allclose(
            payload["events"][0]["optimizer_dofs"]["values"],
            np.array([1.5, -2.5], dtype=np.float64),
        )
        self.assertTrue(payload["events"][0]["optimizer_dofs"]["all_finite"])

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
        self.assertEqual(
            module.resolve_boozer_optimizer_backend("jax", "scipy-jax", None),
            "ondevice",
        )
        self.assertEqual(
            module.resolve_boozer_optimizer_backend(
                "jax",
                "scipy-jax-fullgraph",
                None,
            ),
            "scipy",
        )
        self.assertEqual(
            module.resolve_boozer_optimizer_backend(
                "jax", "scipy-jax", "ondevice"
            ),
            "ondevice",
        )
        self.assertEqual(
            module.resolve_boozer_optimizer_backend("jax", "ondevice", "scipy"),
            "scipy",
        )
        with self.assertRaisesRegex(
            ValueError, "requires boozer_optimizer_backend to be"
        ):
            module.resolve_boozer_optimizer_backend("jax", "hybrid", None)

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
            "quasi-newton",
        )
        self.assertEqual(
            module.resolve_single_stage_default_boozer_least_squares_algorithm(
                "jax",
                "scipy-jax",
            ),
            "quasi-newton",
        )
        self.assertEqual(
            module.resolve_single_stage_default_boozer_least_squares_algorithm(
                "jax",
                "scipy-jax-fullgraph",
            ),
            "quasi-newton",
        )
        with self.assertRaisesRegex(
            ValueError, "requires boozer_optimizer_backend to be"
        ):
            module.resolve_single_stage_default_boozer_least_squares_algorithm(
                "jax",
                "hybrid",
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

    def test_resolve_single_stage_boozer_limited_memory_defaults_to_full_memory(self):
        module = self.load_module()

        self.assertFalse(
            module.resolve_single_stage_boozer_limited_memory("cpu", "scipy")
        )
        self.assertFalse(
            module.resolve_single_stage_boozer_limited_memory("jax", "ondevice")
        )
        self.assertFalse(
            module.resolve_single_stage_boozer_limited_memory(
                "jax",
                "ondevice",
                None,
                False,
            )
        )
        with self.assertRaisesRegex(
            ValueError,
            "boozer_limited_memory=True is not supported",
        ):
            module.resolve_single_stage_boozer_limited_memory(
                "jax",
                "ondevice",
                None,
                True,
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
        self.assertEqual(args.boozer_least_squares_algorithm, "quasi-newton")
        self.assertFalse(args.boozer_least_squares_algorithm_explicit)
        self.assertIsNone(args.boozer_limited_memory)

    def test_parse_args_scipy_jax_outer_defaults_boozer_to_ondevice(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--optimizer-backend",
                "scipy-jax",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.optimizer_backend, "scipy-jax")
        self.assertIsNone(args.boozer_optimizer_backend)
        self.assertEqual(args.boozer_least_squares_algorithm, "quasi-newton")
        self.assertFalse(args.boozer_least_squares_algorithm_explicit)
        self.assertIsNone(args.boozer_limited_memory)

    def test_parse_args_scipy_jax_fullgraph_outer_defaults_boozer_to_ondevice(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--optimizer-backend",
                "scipy-jax-fullgraph",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.optimizer_backend, "scipy-jax-fullgraph")
        self.assertIsNone(args.boozer_optimizer_backend)
        self.assertEqual(args.boozer_least_squares_algorithm, "quasi-newton")
        self.assertFalse(args.boozer_least_squares_algorithm_explicit)
        self.assertIsNone(args.boozer_limited_memory)

    def test_parse_args_rejects_scipy_jax_fullstate_outer_backend(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--optimizer-backend",
                "scipy-jax-fullstate",
            ],
        ):
            with self.assertRaises(SystemExit):
                module.parse_args()

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
        self.assertIsNone(args.boozer_limited_memory)

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
        self.assertEqual(args.boozer_least_squares_algorithm, "quasi-newton")
        self.assertFalse(args.boozer_least_squares_algorithm_explicit)
        self.assertIsNone(args.boozer_limited_memory)

    def test_parse_args_accepts_boozer_limited_memory_override(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--boozer-limited-memory",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.boozer_limited_memory)

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
        self.assertFalse(args.profile_target_lane_memory_analysis)
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
        self.assertFalse(args.initial_step_scale_explicit)
        self.assertFalse(args.initial_step_maxiter_explicit)
        self.assertIsNone(args.target_lane_boozer_bfgs_tol)
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
        self.assertFalse(args.initial_step_scale_explicit)
        self.assertFalse(args.initial_step_maxiter_explicit)
        self.assertIsNone(args.target_lane_boozer_bfgs_tol)
        self.assertIsNone(args.target_lane_boozer_bfgs_maxiter)
        self.assertIsNone(args.target_lane_boozer_newton_tol)
        self.assertIsNone(args.target_lane_boozer_newton_maxiter)

    def test_parse_args_marks_explicit_initial_phase_defaults(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--initial-step-scale",
                "1.0",
                "--initial-step-maxiter",
                "0",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.initial_step_scale, 1.0)
        self.assertEqual(args.initial_step_maxiter, 0)
        self.assertTrue(args.initial_step_scale_explicit)
        self.assertTrue(args.initial_step_maxiter_explicit)

    def test_parse_args_marks_explicit_stage2_bs_path(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--stage2-bs-path",
                "/tmp/stage2/biot_savart_opt.json",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.stage2_bs_path, "/tmp/stage2/biot_savart_opt.json")
        self.assertTrue(args.stage2_bs_path_explicit)

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

    def test_build_scaled_outer_problem_scaled_state_target_lane_mixed_inputs_and_numpy_scale_are_transfer_safe(
        self,
    ):
        module = self.load_module()
        seen = {"callback": []}

        def base_fun(x):
            return jnp.sum(x * x), x + x

        def base_callback(x):
            seen["callback"].append(np.asarray(jax.device_get(x), dtype=np.float64))

        scaled_state = module.ScaledOuterPhaseOptimizerState(
            step_dofs=jax.device_put(np.array([2.0, -4.0], dtype=np.float64)),
            anchor_dofs=np.array([10.0, 20.0], dtype=np.float64),
        )
        scaled_fun, scaled_callback = module.build_scaled_outer_problem(
            base_fun,
            base_callback,
            np.zeros(2, dtype=np.float64),
            np.float64(0.25),
            anchor_in_state=True,
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

    def test_build_scaled_outer_scalar_problem_scaled_state_target_lane_mixed_inputs_and_numpy_scale_are_transfer_safe(
        self,
    ):
        module = self.load_module()
        seen = {"callback": []}

        def base_fun(x):
            return jnp.sum(x * x)

        def base_callback(x):
            seen["callback"].append(np.asarray(jax.device_get(x), dtype=np.float64))

        scaled_state = module.ScaledOuterPhaseOptimizerState(
            step_dofs=np.array([2.0, -4.0], dtype=np.float64),
            anchor_dofs=jax.device_put(np.array([10.0, 20.0], dtype=np.float64)),
        )
        scaled_fun, scaled_callback = module.build_scaled_outer_scalar_problem(
            base_fun,
            base_callback,
            np.zeros(2, dtype=np.float64),
            np.float64(0.25),
            anchor_in_state=True,
        )

        with jax.transfer_guard("disallow"):
            value = scaled_fun(scaled_state)
            scaled_callback(scaled_state)

        self.assertAlmostEqual(float(jax.device_get(value)), 10.5**2 + 19.0**2)
        np.testing.assert_allclose(seen["callback"][0], [10.5, 19.0])

    def test_resolve_single_stage_warm_start_paths_requires_artifacts(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(
                FileNotFoundError,
                "single-stage warm start run directory is missing required artifacts",
            ):
                module.resolve_single_stage_warm_start_paths(tmpdir)

    def test_load_single_stage_warm_start_state_rejects_live_surface_fallback(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            (run_dir / "surf_opt.json").write_text("{}", encoding="utf-8")
            (run_dir / "results.json").write_text(
                json.dumps({"FINAL_IOTA": 0.123, "FINAL_G": 4.5}),
                encoding="utf-8",
            )
            (run_dir / "biot_savart_opt.json").write_text("{}", encoding="utf-8")

            with patch.object(
                module,
                "load",
                side_effect=AssertionError("JAX warm-start loader must not call load()"),
            ), self.assertRaisesRegex(ValueError, "surface payload is not a SIMSON"):
                module.load_single_stage_warm_start_state(str(run_dir))

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
                side_effect=AssertionError(
                    "serialized warm start should not call load()"
                ),
            ):
                warm_start = module.load_single_stage_warm_start_state(str(run_dir))

        self.assertIsInstance(warm_start["surface"], module.SerializedSurfaceState)
        self.assertEqual(warm_start["surface"].surface_class, "SurfaceXYZTensorFourier")
        np.testing.assert_allclose(warm_start["surface"].dofs, surface_dofs)
        self.assertEqual(warm_start["iota"], 0.123)
        self.assertEqual(warm_start["G"], 4.5)
        self.assertIsNone(warm_start["biot_savart_path"])

    def test_single_stage_jax_runtime_seed_spec_round_trips_target_surface_dofs(self):
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
            path = module.write_single_stage_jax_runtime_seed_spec(
                tmpdir,
                surface=surface,
                surface_dofs=surface_dofs,
                iota=0.123,
                G=4.5,
                mpol=2,
                ntor=1,
                quadpoints_phi=surface.quadpoints_phi,
                quadpoints_theta=surface.quadpoints_theta,
                **self._jax_runtime_seed_spec_field_kwargs(module),
            )
            with patch.object(
                module,
                "project_surface_dofs_to_resolution",
                side_effect=AssertionError(
                    "JAX runtime spec load must not reproject warm-start surfaces"
                ),
            ):
                loaded = module.load_single_stage_jax_runtime_seed_spec(
                    tmpdir,
                    mpol=2,
                    ntor=1,
                    quadpoints_phi=surface.quadpoints_phi,
                    quadpoints_theta=surface.quadpoints_theta,
                )

        self.assertTrue(path.endswith(module._SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME))
        np.testing.assert_allclose(module.host_array(loaded["surface_dofs"]), surface_dofs)
        self.assertEqual(float(module.host_array(loaded["iota"])), 0.123)
        self.assertEqual(float(module.host_array(loaded["G"])), 4.5)
        self.assertEqual(loaded["stage2_seed"]["banana_surf_radius"], 0.22)

    def test_jax_runtime_seed_spec_lane_rejects_non_target_outer_contract(self):
        module = self.load_module()

        with self.assertRaisesRegex(ValueError, "target optimizer lane"):
            module.require_single_stage_jax_target_lane(
                use_jax=True,
                use_target_lane=False,
            )

        module.require_single_stage_jax_target_lane(
            use_jax=True,
            use_target_lane=True,
        )
        module.require_single_stage_jax_target_lane(
            use_jax=True,
            use_target_lane=False,
            optimizer_method=module._JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_METHOD,
        )

    def test_scipy_jax_contract_uses_target_scipy_control_lane(self):
        module = self.load_module()
        from simsopt.geo.optimizer_jax import (
            ReferenceOptimizerContract,
            TargetOptimizerContract,
        )

        self.assertTrue(
            module.single_stage_optimizer_contract_uses_array_native_target_lane(
                TargetOptimizerContract(method="lbfgs-ondevice"),
                constraint_method="penalty",
            )
        )
        self.assertTrue(
            module.single_stage_optimizer_contract_uses_array_native_target_lane(
                TargetOptimizerContract(method="lbfgs-scipy-jax"),
                constraint_method="penalty",
            )
        )
        self.assertFalse(
            module.single_stage_optimizer_contract_uses_array_native_target_lane(
                TargetOptimizerContract(
                    method=module._JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_METHOD,
                ),
                constraint_method="penalty",
            )
        )
        self.assertTrue(
            module.single_stage_optimizer_contract_uses_full_graph_jax_scipy(
                TargetOptimizerContract(
                    method=module._JAX_FULL_GRAPH_SCIPY_OUTER_OPTIMIZER_METHOD,
                )
            )
        )
        self.assertFalse(
            module.single_stage_optimizer_contract_uses_array_native_target_lane(
                ReferenceOptimizerContract(method="lbfgs"),
                constraint_method="penalty",
            )
        )

    def test_full_graph_jax_dof_map_reorders_native_surface_tail_to_cpu_order(self):
        module = self.load_module()

        class FakeLineage:
            def __init__(self, local_dof_size):
                self.local_dof_size = local_dof_size

        coil_block = FakeLineage(2)
        plasma_surface = FakeLineage(3)
        vessel_surface = FakeLineage(1)
        jf = types.SimpleNamespace(
            unique_dof_lineage=(coil_block, plasma_surface, vessel_surface),
        )
        boozer_surface = types.SimpleNamespace(surface=plasma_surface)

        dof_map = module.build_single_stage_full_graph_jax_cpu_order_dof_map(
            jf,
            boozer_surface,
            vessel_surface,
        )

        np.testing.assert_array_equal(
            dof_map.optimizer_to_native_indices,
            np.array([0, 1, 5, 2, 3, 4], dtype=np.int64),
        )
        native_dofs = np.array([10.0, 11.0, 20.0, 21.0, 22.0, 30.0])
        optimizer_dofs = np.array([10.0, 11.0, 30.0, 20.0, 21.0, 22.0])
        np.testing.assert_array_equal(
            dof_map.optimizer_from_native_dofs(native_dofs),
            optimizer_dofs,
        )
        np.testing.assert_array_equal(
            dof_map.native_from_optimizer_dofs(optimizer_dofs),
            native_dofs,
        )
        np.testing.assert_array_equal(
            dof_map.optimizer_from_native_gradient(native_dofs),
            optimizer_dofs,
        )

    def test_full_graph_jax_adapter_maps_native_full_gradient_once(self):
        module = self.load_module()
        applied = {}
        run_dict = {}
        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface=object(),
            JF=object(),
            bs=object(),
            objectives={},
            diagnostics={},
            log_path="unused.log",
            apply_coil_dofs=lambda x: applied.setdefault("x", np.asarray(x)),
            optimizer_gradient_transform=lambda grad: grad[[0, 2, 1]],
        )

        with patch.object(
            module,
            "evaluate_candidate",
            return_value=(1.5, np.array([10.0, 20.0, 30.0])),
        ):
            value, gradient = adapter(np.array([1.0, 2.0, 3.0]))

        self.assertEqual(value, 1.5)
        np.testing.assert_array_equal(applied["x"], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_array_equal(gradient, np.array([10.0, 30.0, 20.0]))

        run_dict["last_candidate_failure"] = {"reason": "nonfinite"}
        with patch.object(
            module,
            "evaluate_candidate",
            return_value=(2.5, np.array([40.0, 50.0, 60.0])),
        ):
            value, gradient = adapter(np.array([4.0, 5.0, 6.0]))

        self.assertEqual(value, 2.5)
        np.testing.assert_array_equal(gradient, np.array([40.0, 50.0, 60.0]))

    def test_single_stage_adapter_benchmark_sync_uses_cached_objective(self):
        module = self.load_module()
        run_dict = {
            "it": 7,
            "J": 9.0,
            "dJ": np.array([-1.0, -2.0]),
            "lscount": 3,
        }
        boozer_surface = types.SimpleNamespace(
            surface=types.SimpleNamespace(x=np.array([0.1, 0.2])),
            res={"success": True, "iota": 0.0034, "G": 2.0},
        )
        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface=boozer_surface,
            JF=object(),
            bs=object(),
            objectives={},
            diagnostics={},
            log_path="unused.log",
            apply_coil_dofs=lambda _x: None,
            benchmark_mode=True,
        )

        accepted_x = np.array([1.0, 2.0])
        with patch.object(
            module,
            "evaluate_candidate",
            return_value=(1.25, np.array([10.0, 20.0])),
        ):
            adapter(accepted_x)

        adapter.sync_accepted_step(accepted_x)

        self.assertEqual(run_dict["J"], 1.25)
        np.testing.assert_array_equal(run_dict["dJ"], np.array([10.0, 20.0]))
        np.testing.assert_array_equal(run_dict["sdofs"], np.array([0.1, 0.2]))
        self.assertEqual(run_dict["iota"], 0.0034)
        self.assertEqual(run_dict["G"], 2.0)
        self.assertEqual(run_dict["lscount"], 0)
        self.assertEqual(run_dict["it"], 8)

    def test_single_stage_adapter_records_objective_evaluation_trace(self):
        module = self.load_module()
        events = []
        run_dict = {
            "it": 5,
            "lscount": 12,
            "accepted_iterations": 4,
            "hardware_constraint_status": {
                "success": True,
                "max_curvature": 38.8,
            },
        }
        boozer_surface = types.SimpleNamespace(
            surface=types.SimpleNamespace(x=np.array([0.1, 0.2])),
            res={"success": True, "iota": 0.0034, "G": 2.0},
        )
        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface=boozer_surface,
            JF=object(),
            bs=object(),
            objectives={},
            diagnostics={},
            log_path="unused.log",
            apply_coil_dofs=lambda _x: None,
            optimizer_gradient_transform=lambda grad: grad[[0, 2, 1]],
            objective_evaluation_trace_callback=events.append,
        )

        with patch.object(
            module,
            "evaluate_candidate",
            return_value=(1.25, np.array([10.0, 20.0, 30.0])),
        ):
            value, gradient = adapter(np.array([1.0, 2.0, 3.0]))

        self.assertEqual(value, 1.25)
        np.testing.assert_array_equal(gradient, np.array([10.0, 30.0, 20.0]))
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["accepted_iteration_target"], 5)
        self.assertEqual(event["line_search_evaluation"], 12)
        self.assertEqual(event["accepted_iterations"], 4)
        self.assertEqual(event["candidate_optimizer_dofs"]["values"], [1.0, 2.0, 3.0])
        self.assertEqual(event["objective"]["value"], 1.25)
        self.assertEqual(event["native_gradient"]["inf_norm"], 30.0)
        self.assertEqual(event["optimizer_gradient"]["inf_norm"], 30.0)
        self.assertTrue(event["native_gradient_used"])
        self.assertTrue(event["solver_success"])
        self.assertEqual(event["boozer_iota"]["value"], 0.0034)
        self.assertEqual(event["boozer_surface_dofs"]["values"], [0.1, 0.2])
        self.assertTrue(event["hardware_status"]["success"])
        self.assertIsNone(event["candidate_failure"])

    def test_single_stage_objective_trace_records_unsolved_boozer_state(self):
        module = self.load_module()
        events = []
        run_dict = {
            "it": 3,
            "lscount": 9,
            "accepted_iterations": 0,
            "hardware_constraint_status": None,
        }
        boozer_surface = types.SimpleNamespace(
            res={"success": False, "iota": 0.005, "G": 2.0},
        )
        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface=boozer_surface,
            JF=object(),
            bs=object(),
            objectives={},
            diagnostics={},
            log_path="unused.log",
            apply_coil_dofs=lambda _x: None,
            objective_evaluation_trace_callback=events.append,
        )

        with patch.object(
            module,
            "evaluate_candidate",
            return_value=(2.5, np.array([10.0, 20.0])),
        ):
            adapter(np.array([1.0, 2.0]))

        self.assertEqual(len(events), 1)
        self.assertFalse(events[0]["solver_success"])
        self.assertIsNone(events[0]["boozer_surface_dofs"])

    def test_single_stage_runtime_stage2_seed_payload_requires_order(self):
        module = self.load_module()

        with self.assertRaises(KeyError):
            module.build_single_stage_runtime_stage2_seed_payload(
                {
                    "MAJOR_RADIUS": 1.0,
                    "TOROIDAL_FLUX": 0.5,
                },
                banana_surf_radius=0.22,
            )

    def test_resolve_single_stage_runtime_seed_g_uses_current_derived_value(self):
        module = self.load_module()

        class Current:
            def __init__(self, value):
                self.value = value

            def get_value(self):
                return self.value

        class Coil:
            def __init__(self, value):
                self.current = Current(value)

        self.assertEqual(
            module.resolve_single_stage_runtime_seed_G(4.5, [Coil(1.0)]),
            4.5,
        )
        self.assertAlmostEqual(
            module.resolve_single_stage_runtime_seed_G(None, [Coil(2.0), Coil(-3.0)]),
            4.0 * np.pi * 1e-7 * 5.0,
        )

    def test_load_single_stage_jax_warm_start_state_preserves_donor_surface_contract(
        self,
    ):
        module = self.load_module()
        surface = module.SurfaceXYZTensorFourier(
            mpol=2,
            ntor=1,
            nfp=5,
            stellsym=True,
            quadpoints_phi=np.linspace(0.0, 0.2, 4, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = module.write_single_stage_jax_runtime_seed_spec(
                tmpdir,
                surface=surface,
                surface_dofs=surface.get_dofs(),
                iota=0.123,
                G=4.5,
                mpol=2,
                ntor=1,
                quadpoints_phi=surface.quadpoints_phi,
                quadpoints_theta=surface.quadpoints_theta,
                **self._jax_runtime_seed_spec_field_kwargs(module),
            )
            surface.save(str(Path(tmpdir) / "surf_opt.json"))
            (Path(tmpdir) / "results.json").write_text("{}", encoding="utf-8")
            with patch.object(
                module,
                "load",
                side_effect=AssertionError("JAX warm start must not load live objects"),
            ):
                warm_start = module.load_single_stage_jax_warm_start_state(tmpdir)

        self.assertIsInstance(warm_start["surface"], module.SerializedSurfaceState)
        self.assertTrue(warm_start["surface_path"].endswith("surf_opt.json"))
        self.assertEqual(warm_start["jax_runtime_spec_path"], spec_path)
        self.assertIsNone(warm_start["biot_savart_path"])

    def test_load_single_stage_jax_warm_start_state_honors_explicit_seed_spec(self):
        module = self.load_module()
        surface = module.SurfaceXYZTensorFourier(
            mpol=2,
            ntor=1,
            nfp=5,
            stellsym=True,
            quadpoints_phi=np.linspace(0.0, 0.2, 4, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            donor_dir = Path(tmpdir) / "donor"
            spec_dir = Path(tmpdir) / "spec"
            donor_dir.mkdir()
            spec_dir.mkdir()
            spec_path = module.write_single_stage_jax_runtime_seed_spec(
                spec_dir,
                surface=surface,
                surface_dofs=surface.get_dofs(),
                iota=0.123,
                G=4.5,
                mpol=2,
                ntor=1,
                quadpoints_phi=surface.quadpoints_phi,
                quadpoints_theta=surface.quadpoints_theta,
                **self._jax_runtime_seed_spec_field_kwargs(module),
            )
            surface.save(str(donor_dir / "surf_opt.json"))
            (donor_dir / "results.json").write_text("{}", encoding="utf-8")
            with patch.object(
                module,
                "load",
                side_effect=AssertionError("JAX warm start must not load live objects"),
            ):
                warm_start = module.load_single_stage_jax_warm_start_state(
                    donor_dir,
                    runtime_spec_path=spec_path,
                )

        self.assertEqual(warm_start["jax_runtime_spec_path"], spec_path)
        self.assertIsInstance(warm_start["surface"], module.SerializedSurfaceState)
        self.assertEqual(warm_start["surface_path"], str(donor_dir / "surf_opt.json"))
        self.assertIsNone(warm_start["biot_savart_path"])

    def test_compile_requested_single_stage_jax_runtime_seed_spec_uses_explicit_command_contract(
        self,
    ):
        module = self.load_module()
        args = types.SimpleNamespace(
            warm_start_run_dir="/tmp/donor-run",
            mpol=2,
            ntor=1,
            nphi=7,
            ntheta=5,
            num_tf_coils=4,
            jax_runtime_seed_spec="/tmp/runtime-spec.json",
        )

        with patch.object(
            module,
            "compile_single_stage_jax_runtime_seed_spec",
            return_value="/tmp/runtime-spec.json",
        ) as compile_spec:
            path = module.compile_requested_single_stage_jax_runtime_seed_spec(args)

        self.assertEqual(path, "/tmp/runtime-spec.json")
        compile_spec.assert_called_once_with(
            "/tmp/donor-run",
            mpol=2,
            ntor=1,
            nphi=7,
            ntheta=5,
            num_tf_coils=4,
            output_path_or_run_dir="/tmp/runtime-spec.json",
        )

    def test_compile_requested_single_stage_jax_runtime_seed_spec_requires_donor_dir(
        self,
    ):
        module = self.load_module()
        args = types.SimpleNamespace(warm_start_run_dir=None)

        with self.assertRaisesRegex(ValueError, "--warm-start-run-dir"):
            module.compile_requested_single_stage_jax_runtime_seed_spec(args)

    def test_load_single_stage_jax_runtime_seed_startup_state_uses_spec_payload(
        self,
    ):
        module = self.load_module()
        args = types.SimpleNamespace(
            jax_runtime_seed_spec="/tmp/runtime-spec.json",
            warm_start_run_dir="/tmp/donor-run",
        )
        runtime_spec_state = {
            "path": "/tmp/runtime-spec.json",
            "stage2_seed": {
                "major_radius": 1.0,
                "toroidal_flux": 0.5,
                "order": 2,
                "banana_surf_radius": 0.22,
            },
        }

        with patch.object(
            module,
            "load_single_stage_jax_runtime_seed_spec",
            return_value=runtime_spec_state,
        ) as load_spec:
            startup_state = module.load_single_stage_jax_runtime_seed_startup_state(
                args,
                mpol=2,
                ntor=1,
                nphi=7,
                ntheta=5,
            )

        load_spec.assert_called_once_with(
            "/tmp/runtime-spec.json",
            mpol=2,
            ntor=1,
            nphi=7,
            ntheta=5,
        )
        self.assertEqual(startup_state["stage2_bs_path"], "/tmp/runtime-spec.json")
        self.assertEqual(startup_state["stage2_results_path"], "/tmp/runtime-spec.json")
        self.assertIs(startup_state["runtime_spec_state"], runtime_spec_state)
        self.assertEqual(startup_state["stage2_results"]["order"], 2)

    def test_runtime_spec_biotsavart_full_artifact_curves_follow_updated_dofs(self):
        module = self.load_module()
        from simsopt.field import (
            SingleStageRuntimeSpecBiotSavartJAX as PackageRuntimeSpecBiotSavartJAX,
        )

        self.assertIs(
            PackageRuntimeSpecBiotSavartJAX,
            SingleStageRuntimeSpecBiotSavartJAX,
        )
        curve_dofs = np.linspace(0.1, 0.9, 9, dtype=np.float64)
        initial_coil_dofs = np.concatenate(
            (curve_dofs, np.asarray([3.0], dtype=np.float64))
        )
        updated_coil_dofs = initial_coil_dofs + np.linspace(
            0.2, 1.1, initial_coil_dofs.size, dtype=np.float64
        )

        def make_map(template_full_dofs, owner_segments):
            return module.jax_specs.make_optimizable_dof_map_spec(
                template_full_dofs=template_full_dofs,
                owner_segments=owner_segments,
                input_mode="full",
                input_start=0,
                input_end=len(template_full_dofs),
            )

        curve_template = module.jax_specs.make_curve_xyzfourier_spec(
            dofs=np.zeros(9, dtype=np.float64),
            quadpoints=np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64),
            order=1,
        )
        extraction_spec = module.jax_specs.make_coil_set_dof_extraction_spec(
            (
                module.jax_specs.make_coil_dof_extraction_spec(
                    curve=curve_template,
                    curve_map=make_map(
                        np.zeros(9, dtype=np.float64),
                        ((0, 9, 0, 9),),
                    ),
                    current_map=make_map(
                        np.zeros(1, dtype=np.float64),
                        ((9, 10, 0, 1),),
                    ),
                ),
            )
        )
        surface_spec = module.make_surface_xyz_tensor_fourier_spec(
            dofs=np.array([1.0, 0.1, 0.0, 0.1], dtype=np.float64),
            quadpoints_phi=np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64),
            quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False, dtype=np.float64),
            nfp=1,
            stellsym=True,
            mpol=1,
            ntor=0,
        )
        runtime_spec = module.make_single_stage_runtime_spec(
            seed=module.make_single_stage_seed_spec(
                surface=surface_spec,
                coil_set=module.coil_set_spec_from_dof_extraction_spec(
                    extraction_spec,
                    initial_coil_dofs,
                ),
                coil_dof_extraction=extraction_spec,
                coil_dofs=initial_coil_dofs,
                boozer_iota=0.1,
                boozer_G=1.2,
                target_labels=(),
                hardware_constants=(),
                self_intersection_mode=module._SINGLE_STAGE_JAX_SELF_INTERSECTION_MODE,
                schema_version=module._SINGLE_STAGE_JAX_RUNTIME_SPEC_VERSION,
                num_tf_coils=0,
                banana_curve_index=0,
                tf_current_A=0.0,
                banana_current_A=float(initial_coil_dofs[-1]),
            ),
            mpol=1,
            ntor=0,
            nfp=1,
            nphi=4,
            ntheta=5,
        )

        bs = SingleStageRuntimeSpecBiotSavartJAX(runtime_spec)
        child = Optimizable(x0=np.asarray([], dtype=np.float64), depends_on=[bs])

        self.assertIsInstance(bs, Optimizable)
        self.assertEqual(bs.local_dof_size, initial_coil_dofs.size)
        self.assertIn(bs, child.parents)
        self.assertIn(child, [child_ref() for child_ref in bs._children])

        captured_curves = [coil.curve for coil in bs.coils]
        initial_gamma = captured_curves[0].gamma()
        bs.x = updated_coil_dofs

        np.testing.assert_allclose(
            module.host_array(bs.coils[0].curve.get_dofs()),
            updated_coil_dofs[:9],
        )
        self.assertEqual(bs.coils[0].current.get_value(), updated_coil_dofs[-1])
        self.assertFalse(np.allclose(bs.coils[0].curve.gamma(), initial_gamma))
        np.testing.assert_allclose(bs.local_x, updated_coil_dofs)
        self.assertIs(captured_curves[0], bs.coils[0].curve)
        np.testing.assert_allclose(
            module.host_array(captured_curves[0].get_dofs()),
            updated_coil_dofs[:9],
        )

        captured = {}

        class ExportSurface:
            mpol = 1
            ntor = 0
            quadpoints_phi = np.asarray([0.0], dtype=np.float64)
            quadpoints_theta = np.asarray([0.0], dtype=np.float64)

            def gamma(self):
                return np.zeros((1, 1, 3), dtype=np.float64)

            def unitnormal(self):
                unit_normal = np.zeros((1, 1, 3), dtype=np.float64)
                unit_normal[:, :, 2] = 1.0
                return unit_normal

            def to_vtk(self, *_args, **_kwargs):
                return None

        class ExportField:
            def __init__(self, coils):
                self.coils = coils

            def set_points(self, points):
                captured["points"] = np.asarray(points)

            def B(self):
                return np.ones((1, 3), dtype=np.float64)

        def capture_curves_to_vtk(curves, *_args, **_kwargs):
            captured["curves_to_vtk_dofs"] = module.host_array(
                curves[0].get_dofs()
            )

        def capture_cross_section(_surf_coils, _surface, banana_curve, *_args):
            captured["cross_section_banana_dofs"] = module.host_array(
                banana_curve.get_dofs()
            )

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            module,
            "curves_to_vtk",
            side_effect=capture_curves_to_vtk,
        ), patch.object(
            module,
            "normPlot",
            return_value=None,
        ), patch.object(
            module,
            "cross_section_plot",
            side_effect=capture_cross_section,
        ):
            module.export_requested_single_stage_artifacts(
                solved_surface_state={
                    "sdofs": module.host_array(surface_spec.dofs),
                    "iota": 0.1,
                    "G": 1.2,
                },
                coil_dofs=updated_coil_dofs,
                num_tf_coils=0,
                tf_current_A=0.0,
                banana_current_A=float(updated_coil_dofs[-1]),
                stage2_seed=self._jax_runtime_seed_spec_field_kwargs(module)[
                    "stage2_seed"
                ],
                output_dir=tmpdir,
                boozer_surface=types.SimpleNamespace(surface=ExportSurface()),
                bs_diag=ExportField(bs.coils),
                surf_coils=object(),
                hbt=object(),
                VV=object(),
                write_restart_artifacts=False,
                write_host_restart_artifacts=False,
                write_full_artifacts=True,
                timings={},
            )

        np.testing.assert_allclose(
            captured["curves_to_vtk_dofs"],
            updated_coil_dofs[:9],
        )
        np.testing.assert_allclose(
            captured["cross_section_banana_dofs"],
            updated_coil_dofs[:9],
        )

    def test_runtime_spec_biotsavart_projects_cotangents_to_owner_dofs(self):
        module = self.load_module()
        owner_dofs = np.linspace(0.1, 1.0, 10, dtype=np.float64)

        def make_map(template_full_dofs, owner_segments):
            return module.jax_specs.make_optimizable_dof_map_spec(
                template_full_dofs=template_full_dofs,
                owner_segments=owner_segments,
                input_mode="full",
                input_start=0,
                input_end=len(template_full_dofs),
            )

        quadpoints = np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64)
        curve_template = module.jax_specs.make_curve_xyzfourier_spec(
            dofs=np.zeros(9, dtype=np.float64),
            quadpoints=quadpoints,
            order=1,
        )
        extraction_spec = module.jax_specs.make_coil_set_dof_extraction_spec(
            (
                module.jax_specs.make_coil_dof_extraction_spec(
                    curve=curve_template,
                    curve_map=make_map(
                        np.zeros(9, dtype=np.float64),
                        ((0, 9, 0, 9),),
                    ),
                    current_map=make_map(
                        np.zeros(1, dtype=np.float64),
                        ((9, 10, 0, 1),),
                    ),
                ),
            )
        )
        surface_spec = module.make_surface_xyz_tensor_fourier_spec(
            dofs=np.array([1.0, 0.1, 0.0, 0.1], dtype=np.float64),
            quadpoints_phi=np.linspace(0.0, 1.0, 4, endpoint=False, dtype=np.float64),
            quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False, dtype=np.float64),
            nfp=1,
            stellsym=True,
            mpol=1,
            ntor=0,
        )
        runtime_spec = module.make_single_stage_runtime_spec(
            seed=module.make_single_stage_seed_spec(
                surface=surface_spec,
                coil_set=module.coil_set_spec_from_dof_extraction_spec(
                    extraction_spec,
                    owner_dofs,
                ),
                coil_dof_extraction=extraction_spec,
                coil_dofs=owner_dofs,
                boozer_iota=0.1,
                boozer_G=1.2,
                target_labels=(),
                hardware_constants=(),
                self_intersection_mode=module._SINGLE_STAGE_JAX_SELF_INTERSECTION_MODE,
                schema_version=module._SINGLE_STAGE_JAX_RUNTIME_SPEC_VERSION,
                num_tf_coils=0,
                banana_curve_index=0,
                tf_current_A=0.0,
                banana_current_A=float(owner_dofs[-1]),
            ),
            mpol=1,
            ntor=0,
            nfp=1,
            nphi=4,
            ntheta=5,
        )
        bs = SingleStageRuntimeSpecBiotSavartJAX(runtime_spec)
        d_gamma = jnp.reshape(
            jnp.linspace(0.2, 1.3, 12, dtype=jnp.float64),
            (4, 3),
        )
        d_gammadash = jnp.reshape(
            jnp.linspace(-0.7, 0.4, 12, dtype=jnp.float64),
            (4, 3),
        )
        d_current = jnp.asarray([0.25], dtype=jnp.float64)

        actual = bs.coil_cotangents_to_dofs_gradient(
            ((d_gamma[None, ...], d_gammadash[None, ...], d_current),),
            ((0,),),
            coil_dofs=owner_dofs,
        )

        def scalar(owner):
            group = bs.coil_set_spec_from_dofs(owner).groups[0]
            return (
                jnp.vdot(group.gammas[0], d_gamma)
                + jnp.vdot(group.gammadashs[0], d_gammadash)
                + group.currents[0] * d_current[0]
            )

        expected = jax.grad(scalar)(jnp.asarray(owner_dofs, dtype=jnp.float64))

        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

        curve = bs.coils[0].curve
        self.assertIn(bs, curve.parents)

        def gamma_scalar(owner):
            group = bs.coil_set_spec_from_dofs(owner).groups[0]
            return jnp.vdot(group.gammas[0], d_gamma)

        expected_gamma = jax.grad(gamma_scalar)(
            jnp.asarray(owner_dofs, dtype=jnp.float64)
        )
        actual_gamma = curve.dgamma_by_dcoeff_vjp(d_gamma)(curve)
        np.testing.assert_allclose(
            actual_gamma,
            expected_gamma,
            rtol=1e-12,
            atol=1e-12,
        )

        length_weights = jnp.linspace(0.1, 0.4, 4, dtype=jnp.float64)

        def length_scalar(owner):
            group = bs.coil_set_spec_from_dofs(owner).groups[0]
            return jnp.vdot(
                jnp.linalg.norm(group.gammadashs[0], axis=1),
                length_weights,
            )

        expected_length = jax.grad(length_scalar)(
            jnp.asarray(owner_dofs, dtype=jnp.float64)
        )
        actual_length = curve.dincremental_arclength_by_dcoeff_vjp(length_weights)(
            curve
        )
        np.testing.assert_allclose(
            actual_length,
            expected_length,
            rtol=1e-12,
            atol=1e-12,
        )

        from simsopt.jax_core import (
            coil_specs_from_dof_extraction_spec,
            curve_geometry_from_spec,
        )

        kappa_weights = jnp.linspace(-0.2, 0.3, 4, dtype=jnp.float64)

        def kappa_scalar(owner):
            coil_spec = coil_specs_from_dof_extraction_spec(extraction_spec, owner)[0]
            _gamma, gammadash, gammadashdash = curve_geometry_from_spec(
                coil_spec.curve
            )
            kappa = jnp.linalg.norm(
                jnp.cross(gammadash, gammadashdash),
                axis=1,
            ) / (jnp.linalg.norm(gammadash, axis=1) ** 3)
            return jnp.vdot(kappa, kappa_weights)

        expected_kappa = jax.grad(kappa_scalar)(
            jnp.asarray(owner_dofs, dtype=jnp.float64)
        )
        actual_kappa = curve.dkappa_by_dcoeff_vjp(kappa_weights)(curve)
        np.testing.assert_allclose(
            actual_kappa,
            expected_kappa,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_jax_warm_start_surface_dofs_require_seed_spec_artifact(self):
        module = self.load_module()

        with tempfile.TemporaryDirectory() as tmpdir, self.assertRaisesRegex(
            FileNotFoundError,
            "run seed conversion first",
        ):
            module.resolve_jax_warm_start_surface_dofs_from_spec(
                tmpdir,
                mpol=2,
                ntor=1,
                quadpoints_phi=np.linspace(0.0, 0.2, 4, endpoint=False),
                quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
            )

    def test_resolve_single_stage_startup_seed_contract_prefers_warm_start_donor_seed(
        self,
    ):
        module = self.load_module()
        args = types.SimpleNamespace(
            stage2_bs_path_explicit=False,
            stage2_source="database",
        )

        contract = module.resolve_single_stage_startup_seed_contract(
            args,
            warm_start_state={
                "biot_savart_path": "/tmp/warm-start/biot_savart_opt.json",
            },
        )

        self.assertEqual(
            contract["stage2_bs_path"],
            "/tmp/warm-start/biot_savart_opt.json",
        )
        self.assertEqual(contract["stage2_source"], "warm_start_donor")
        self.assertFalse(contract["tf_current_limit_enforced"])
        self.assertFalse(contract["seed_hardware_validation_enforced"])

    def test_resolve_single_stage_startup_seed_contract_respects_explicit_stage2_seed(
        self,
    ):
        module = self.load_module()
        args = types.SimpleNamespace(
            stage2_bs_path_explicit=True,
            stage2_source="database",
        )

        with patch.object(
            module,
            "build_stage2_bs_path",
            return_value="/tmp/stage2/biot_savart_opt.json",
        ):
            contract = module.resolve_single_stage_startup_seed_contract(
                args,
                warm_start_state={
                    "biot_savart_path": "/tmp/warm-start/biot_savart_opt.json",
                },
            )

        self.assertEqual(
            contract["stage2_bs_path"],
            "/tmp/stage2/biot_savart_opt.json",
        )
        self.assertEqual(contract["stage2_source"], "explicit_path")
        self.assertFalse(contract["tf_current_limit_enforced"])
        self.assertFalse(contract["seed_hardware_validation_enforced"])

    def test_resolve_single_stage_startup_seed_contract_keeps_derived_seed_guard(self):
        module = self.load_module()
        args = types.SimpleNamespace(
            stage2_bs_path_explicit=False,
            stage2_source="database",
        )

        with patch.object(
            module,
            "build_stage2_bs_path",
            return_value="/tmp/database/biot_savart_opt.json",
        ):
            contract = module.resolve_single_stage_startup_seed_contract(
                args,
                warm_start_state=None,
            )

        self.assertEqual(
            contract["stage2_bs_path"],
            "/tmp/database/biot_savart_opt.json",
        )
        self.assertEqual(contract["stage2_source"], "database")
        self.assertTrue(contract["tf_current_limit_enforced"])
        self.assertTrue(contract["seed_hardware_validation_enforced"])

    def test_resolve_single_stage_search_policy_preserves_serialized_surface_state(
        self,
    ):
        module = self.load_module()
        policy = module.resolve_single_stage_search_policy(
            {
                "surface": module.SerializedSurfaceState(
                    surface_class="SurfaceXYZTensorFourier",
                    dofs=np.ones(3),
                    mpol=2,
                    ntor=1,
                    nfp=5,
                    stellsym=True,
                    quadpoints_phi=np.linspace(0.0, 0.2, 4, endpoint=False),
                    quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
                )
            },
            explicit_surface_warm_start=True,
        )

        self.assertEqual(policy.donor_class, "serialized_surface_state")
        self.assertEqual(policy.search_policy, "preserve_first")
        self.assertEqual(policy.adaptive_failure_penalty_weight, 1.0)
        self.assertIsNone(policy.auto_initial_step_scale)
        self.assertEqual(policy.invalid_step_retry_budget, 2)
        self.assertEqual(policy.retry_step_shrink_factor, 0.35)

    def test_resolve_single_stage_search_policy_repairs_stage2_seed_only_start(self):
        module = self.load_module()
        policy = module.resolve_single_stage_search_policy(
            None,
            explicit_surface_warm_start=False,
        )

        self.assertEqual(policy.donor_class, "stage2_seed_only")
        self.assertEqual(policy.search_policy, "repair_first")
        self.assertEqual(policy.adaptive_failure_penalty_weight, 1.5)
        self.assertEqual(policy.auto_initial_step_scale, 0.25)
        self.assertEqual(policy.auto_initial_step_maxiter, 3)
        self.assertEqual(policy.invalid_step_retry_budget, 2)
        self.assertEqual(policy.retry_step_shrink_factor, 0.5)

    def test_resolve_single_stage_search_policy_retries_projected_supported_surface(
        self,
    ):
        module = self.load_module()
        policy = module.resolve_single_stage_search_policy(
            {"surface": object()},
            explicit_surface_warm_start=True,
        )

        self.assertEqual(policy.donor_class, "projected_supported_surface")
        self.assertEqual(policy.search_policy, "global_search")
        self.assertEqual(policy.invalid_step_retry_budget, 2)
        self.assertEqual(policy.retry_step_shrink_factor, 0.5)

    def test_resolve_single_stage_policy_initial_phase_settings_only_auto_enables_defaults(
        self,
    ):
        module = self.load_module()
        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            auto_initial_step_scale=0.25,
            auto_initial_step_maxiter=3,
        )

        auto_settings = module.resolve_single_stage_policy_initial_phase_settings(
            policy,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
        )
        explicit_settings = module.resolve_single_stage_policy_initial_phase_settings(
            policy,
            initial_step_scale=0.5,
            initial_step_maxiter=2,
        )

        self.assertEqual(auto_settings["initial_step_scale"], 0.25)
        self.assertEqual(auto_settings["initial_step_maxiter"], 3)
        self.assertTrue(auto_settings["auto_enabled"])
        self.assertEqual(explicit_settings["initial_step_scale"], 0.5)
        self.assertEqual(explicit_settings["initial_step_maxiter"], 2)
        self.assertFalse(explicit_settings["auto_enabled"])

    def test_resolve_single_stage_policy_initial_phase_settings_respects_explicit_default_disable(
        self,
    ):
        module = self.load_module()
        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            auto_initial_step_scale=0.25,
            auto_initial_step_maxiter=3,
        )

        explicit_default_settings = (
            module.resolve_single_stage_policy_initial_phase_settings(
                policy,
                initial_step_scale=1.0,
                initial_step_maxiter=0,
                initial_step_scale_explicit=True,
                initial_step_maxiter_explicit=True,
                field_backend="cpu",
                optimizer_backend="scipy",
            )
        )

        self.assertEqual(explicit_default_settings["initial_step_scale"], 1.0)
        self.assertEqual(explicit_default_settings["initial_step_maxiter"], 0)
        self.assertFalse(explicit_default_settings["auto_enabled"])

    def test_resolve_single_stage_policy_initial_phase_settings_skips_target_lane_auto_phase(
        self,
    ):
        module = self.load_module()
        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            auto_initial_step_scale=0.25,
            auto_initial_step_maxiter=3,
        )

        target_lane_settings = module.resolve_single_stage_policy_initial_phase_settings(
            policy,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            field_backend="jax",
            optimizer_backend="ondevice",
        )

        self.assertEqual(target_lane_settings["initial_step_scale"], 1.0)
        self.assertEqual(target_lane_settings["initial_step_maxiter"], 0)
        self.assertFalse(target_lane_settings["auto_enabled"])

    def test_record_single_stage_local_incumbent_tracks_latest_and_best(self):
        module = self.load_module()
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["hardware_constraint_status"] = {"success": True, "violations": []}
        run_dict["intersecting"] = False

        updated = module.record_single_stage_local_incumbent(
            run_dict,
            stage="initial",
        )

        self.assertTrue(updated)
        self.assertEqual(run_dict["latest_local_stage"], "initial")
        self.assertEqual(run_dict["best_local_stage"], "initial")
        np.testing.assert_allclose(
            run_dict["latest_local_incumbent"]["coil_dofs"],
            np.zeros(5),
        )

        run_dict["x_prev"] = np.array([9.0, 8.0, 7.0, 6.0, 5.0])
        run_dict["J"] = 2.0
        updated = module.record_single_stage_local_incumbent(
            run_dict,
            stage="retry",
        )

        self.assertFalse(updated)
        self.assertEqual(run_dict["latest_local_stage"], "retry")
        self.assertEqual(run_dict["best_local_stage"], "initial")
        np.testing.assert_allclose(
            run_dict["latest_local_incumbent"]["coil_dofs"],
            np.array([9.0, 8.0, 7.0, 6.0, 5.0]),
        )

        run_dict["hardware_constraint_status"] = {
            "success": False,
            "violations": ["still_repairing"],
        }
        run_dict["x_prev"] = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
        run_dict["J"] = 0.5
        updated = module.record_single_stage_local_incumbent(
            run_dict,
            stage="repairing",
        )

        self.assertTrue(updated)
        self.assertEqual(run_dict["latest_local_stage"], "repairing")
        self.assertEqual(run_dict["best_local_stage"], "repairing")
        np.testing.assert_allclose(
            run_dict["latest_local_incumbent"]["coil_dofs"],
            np.array([5.0, 6.0, 7.0, 8.0, 9.0]),
        )

    def test_resolve_single_stage_retry_anchor_respects_policy(self):
        module = self.load_module()
        run_dict = {
            "latest_local_incumbent": {"coil_dofs": np.array([3.0])},
            "latest_local_stage": "latest",
            "best_local_incumbent": {"coil_dofs": np.array([1.0])},
            "best_local_stage": "best",
        }
        preserve_policy = module.SingleStageSearchPolicy(
            donor_class="serialized_surface_state",
            search_policy="preserve_first",
            adaptive_failure_penalty_weight=1.0,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.35,
        )
        repair_policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )

        preserve_anchor, preserve_stage = module.resolve_single_stage_retry_anchor(
            run_dict,
            preserve_policy,
        )
        repair_anchor, repair_stage = module.resolve_single_stage_retry_anchor(
            run_dict,
            repair_policy,
        )

        np.testing.assert_allclose(preserve_anchor["coil_dofs"], np.array([1.0]))
        self.assertEqual(preserve_stage, "best")
        np.testing.assert_allclose(repair_anchor["coil_dofs"], np.array([3.0]))
        self.assertEqual(repair_stage, "latest")

    def test_resolve_single_stage_retry_initial_step_size_shrinks_failure_step(self):
        module = self.load_module()
        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )

        retry_step_size = module.resolve_single_stage_retry_initial_step_size(
            None,
            [
                {
                    "step_scale": {"value": 0.0, "finite": True},
                    "requested_initial_step": {"value": 0.2, "finite": True},
                }
            ],
            single_stage_search_policy=policy,
            retry_index=0,
        )

        self.assertEqual(retry_step_size, 0.1)

    def test_resolve_single_stage_retry_initial_step_size_prefers_requested_step(
        self,
    ):
        module = self.load_module()
        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )

        retry_step_size = module.resolve_single_stage_retry_initial_step_size(
            None,
            [
                {
                    "step_scale": {"value": 0.0, "finite": True},
                    "requested_initial_step": {"value": 0.5, "finite": True},
                    "first_tested_alpha": {"value": 0.5, "finite": True},
                    "best_finite_alpha": {"value": 0.25, "finite": True},
                }
            ],
            single_stage_search_policy=policy,
            retry_index=0,
        )

        self.assertEqual(retry_step_size, 0.25)

    def test_build_single_stage_scaled_phase_retry_state_anchors_zero_step(self):
        module = self.load_module()

        retry_state = module.build_single_stage_scaled_phase_retry_state(
            np.array([3.0, 4.0]),
        )

        self.assertIsInstance(retry_state, module.ScaledOuterPhaseOptimizerState)
        np.testing.assert_allclose(retry_state.anchor_dofs, np.array([3.0, 4.0]))
        np.testing.assert_allclose(retry_state.step_dofs, np.zeros(2))

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

    def test_project_surface_dofs_to_resolution_rejects_unsupported_hybrid_surface(
        self,
    ):
        module = self.load_module()

        class FakeSurface:
            nfp = 5
            stellsym = True

            def gamma(self):
                raise AssertionError("unsupported surfaces must not use gamma()")

            def cross_section(self, phi, thetas=None):
                raise AssertionError(
                    "unsupported surfaces must not use cross_section()"
                )

        quadpoints_phi = np.linspace(0.0, 0.2, 5, endpoint=False)
        quadpoints_theta = np.linspace(0.0, 1.0, 7, endpoint=False)

        with self.assertRaisesRegex(TypeError, "dehybridized path"):
            module.project_surface_dofs_to_resolution(
                FakeSurface(),
                mpol=8,
                ntor=6,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )

    def test_project_surface_dofs_to_resolution_returns_matching_xyz_dofs(self):
        module = self.load_module()
        source_surface = module.SurfaceXYZTensorFourier(
            mpol=2,
            ntor=2,
            nfp=5,
            stellsym=True,
            quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
            quadpoints_phi=np.linspace(0.0, 0.2, 5, endpoint=False),
        )
        source_dofs = source_surface.get_dofs().copy()
        source_dofs[:] = np.linspace(0.04, 0.04 * source_dofs.size, source_dofs.size)
        source_surface.set_dofs(source_dofs)

        with patch.object(
            module,
            "_fit_surface_xyz_tensor_dofs_to_gamma",
            side_effect=AssertionError("matching resolution must not refit"),
        ):
            projected_dofs = module.project_surface_dofs_to_resolution(
                source_surface,
                mpol=source_surface.mpol,
                ntor=source_surface.ntor,
                quadpoints_phi=source_surface.quadpoints_phi,
                quadpoints_theta=source_surface.quadpoints_theta,
            )

        np.testing.assert_allclose(projected_dofs, source_dofs)

    def test_project_surface_dofs_to_resolution_returns_matching_serialized_xyz_dofs(
        self,
    ):
        module = self.load_module()
        source_dofs = np.linspace(
            0.02,
            0.02 * len(module.stellsym_scatter_indices(TEST_MPOL, TEST_NTOR)),
            len(module.stellsym_scatter_indices(TEST_MPOL, TEST_NTOR)),
        )
        serialized_surface = module.SerializedSurfaceState(
            surface_class="SurfaceXYZTensorFourier",
            dofs=source_dofs,
            mpol=TEST_MPOL,
            ntor=TEST_NTOR,
            nfp=5,
            stellsym=True,
            quadpoints_phi=np.linspace(
                0.0, 1.0 / 5.0, 2 * TEST_NTOR + 1, endpoint=False
            ),
            quadpoints_theta=np.linspace(0.0, 1.0, 2 * TEST_MPOL + 1, endpoint=False),
        )

        with patch.object(
            module,
            "_fit_surface_xyz_tensor_dofs_to_gamma",
            side_effect=AssertionError("matching serialized surface must not refit"),
        ):
            projected_dofs = module.project_surface_dofs_to_resolution(
                serialized_surface,
                mpol=serialized_surface.mpol,
                ntor=serialized_surface.ntor,
                quadpoints_phi=serialized_surface.quadpoints_phi,
                quadpoints_theta=serialized_surface.quadpoints_theta,
            )

        np.testing.assert_allclose(projected_dofs, source_dofs)

    def test_project_surface_dofs_to_resolution_returns_matching_deferred_xyz_dofs(
        self,
    ):
        module = self.load_module()
        source_dofs = np.linspace(
            0.03,
            0.03 * len(module.stellsym_scatter_indices(TEST_MPOL, TEST_NTOR)),
            len(module.stellsym_scatter_indices(TEST_MPOL, TEST_NTOR)),
        )
        deferred_surface = module.DeferredSurfaceXYZTensorFourier(
            mpol=TEST_MPOL,
            ntor=TEST_NTOR,
            nfp=5,
            stellsym=True,
            quadpoints_phi=np.linspace(
                0.0, 1.0 / 5.0, 2 * TEST_NTOR + 1, endpoint=False
            ),
            quadpoints_theta=np.linspace(0.0, 1.0, 2 * TEST_MPOL + 1, endpoint=False),
            dofs=source_dofs,
        )

        with patch.object(
            module,
            "_fit_surface_xyz_tensor_dofs_to_gamma",
            side_effect=AssertionError("matching deferred surface must not refit"),
        ):
            projected_dofs = module.project_surface_dofs_to_resolution(
                deferred_surface,
                mpol=deferred_surface.mpol,
                ntor=deferred_surface.ntor,
                quadpoints_phi=deferred_surface.quadpoints_phi,
                quadpoints_theta=deferred_surface.quadpoints_theta,
            )

        np.testing.assert_allclose(projected_dofs, source_dofs)
        self.assertIsNone(deferred_surface._materialized_surface)

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

    def test_project_surface_dofs_to_resolution_serialized_xyz_avoids_eager_host_surface_construction(
        self,
    ):
        module = self.load_module()
        serialized_surface = module.SerializedSurfaceState(
            surface_class="SurfaceXYZTensorFourier",
            dofs=np.linspace(
                0.02,
                0.02 * len(module.stellsym_scatter_indices(TEST_MPOL, TEST_NTOR)),
                len(module.stellsym_scatter_indices(TEST_MPOL, TEST_NTOR)),
            ),
            mpol=TEST_MPOL,
            ntor=TEST_NTOR,
            nfp=5,
            stellsym=True,
            quadpoints_phi=np.linspace(
                0.0, 1.0 / 5.0, 2 * TEST_NTOR + 1, endpoint=False
            ),
            quadpoints_theta=np.linspace(0.0, 1.0, 2 * TEST_MPOL + 1, endpoint=False),
        )

        with patch.object(
            module,
            "SurfaceXYZTensorFourier",
            FakeSurfaceXYZTensorFourier,
        ):
            projected_dofs = module.project_surface_dofs_to_resolution(
                serialized_surface,
                mpol=TEST_MPOL,
                ntor=TEST_NTOR,
                quadpoints_phi=serialized_surface.quadpoints_phi,
                quadpoints_theta=serialized_surface.quadpoints_theta,
            )

        self.assertEqual(len(FakeSurfaceXYZTensorFourier.instances), 0)
        self.assertEqual(projected_dofs.shape, serialized_surface.dofs.shape)

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

    def test_project_surface_dofs_to_resolution_reproduces_target_gamma_for_rz_nonstellsym(
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

        projected_gamma = module.surface_gamma_from_dofs(
            jnp.asarray(projected_dofs, dtype=jnp.float64),
            jnp.asarray(quadpoints_phi, dtype=jnp.float64),
            jnp.asarray(quadpoints_theta, dtype=jnp.float64),
            3,
            2,
            source_surface.nfp,
            source_surface.stellsym,
            scatter_indices=None,
        )
        target_gamma = module.surface_rz_fourier_gamma_from_dofs(
            module.surface_rz_fourier_spec_from_dofs(
                jnp.asarray(source_surface.get_dofs(), dtype=jnp.float64),
                quadpoints_phi=jnp.asarray(quadpoints_phi, dtype=jnp.float64),
                quadpoints_theta=jnp.asarray(quadpoints_theta, dtype=jnp.float64),
                mpol=source_surface.mpol,
                ntor=source_surface.ntor,
                nfp=source_surface.nfp,
                stellsym=source_surface.stellsym,
            ),
            jnp.asarray(source_surface.get_dofs(), dtype=jnp.float64),
        )

        np.testing.assert_allclose(
            np.asarray(projected_gamma),
            np.asarray(target_gamma),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_project_surface_dofs_to_resolution_allows_strict_transfer_guard_for_stellsym_rz_surface(
        self,
    ):
        module = self.load_module()
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

        expected_dofs = module.project_surface_dofs_to_resolution(
            source_surface,
            mpol=4,
            ntor=3,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )

        with jax.transfer_guard("disallow"):
            projected_dofs = module.project_surface_dofs_to_resolution(
                source_surface,
                mpol=4,
                ntor=3,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )

        np.testing.assert_allclose(
            projected_dofs,
            expected_dofs,
            rtol=1e-10,
            atol=1e-12,
        )

    def test_project_surface_dofs_to_resolution_avoids_jax_lstsq_solver_path(
        self,
    ):
        module = self.load_module()
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

        original_host_lstsq = np.linalg.lstsq
        with patch.object(
            module.jnp.linalg,
            "lstsq",
            side_effect=AssertionError(
                "project_surface_dofs_to_resolution must not use jax.linalg.lstsq"
            ),
        ), patch.object(
            module.np.linalg,
            "lstsq",
            wraps=original_host_lstsq,
        ) as host_lstsq:
            projected_dofs = module.project_surface_dofs_to_resolution(
                source_surface,
                mpol=4,
                ntor=3,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )

        self.assertTrue(host_lstsq.called)
        self.assertEqual(
            projected_dofs.shape,
            (len(module.stellsym_scatter_indices(4, 3)),),
        )
        self.assertTrue(np.all(np.isfinite(projected_dofs)))

    def test_surface_gamma_from_dofs_allows_strict_transfer_guard_for_eager_stellsym_xyz(
        self,
    ):
        module = self.load_module()
        surface = module.SurfaceXYZTensorFourier(
            mpol=1,
            ntor=1,
            stellsym=True,
            nfp=1,
            quadpoints_phi=np.array([0.23, 0.41]),
            quadpoints_theta=np.array([0.37, 0.59]),
        )
        dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
        scatter_indices = module.stellsym_scatter_indices(surface.mpol, surface.ntor)

        with jax.transfer_guard("disallow"):
            gamma = module.surface_gamma_from_dofs(
                jax.device_put(dofs),
                jax.device_put(np.asarray(surface.quadpoints_phi, dtype=np.float64)),
                jax.device_put(np.asarray(surface.quadpoints_theta, dtype=np.float64)),
                surface.mpol,
                surface.ntor,
                surface.nfp,
                surface.stellsym,
                scatter_indices=scatter_indices,
            )

        self.assertEqual(gamma.shape, (2, 2, 3))
        self.assertTrue(np.all(np.isfinite(np.asarray(jax.device_get(gamma)))))

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

            def _unpack_decision_vector_jax(
                self, solved_x, optimize_G, coil_set_spec=None
            ):
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
            success_filter = (
                module.build_single_stage_target_lane_hardware_success_filter(
                    FakeBoozerSurface(),
                    FakeBS(),
                    banana_curve,
                    vessel_surface,
                    cc_dist=0.05,
                    cs_dist=0.05,
                    ss_dist=0.05,
                    curvature_threshold=40.0,
                )
            )

        def contains_jax_array(value):
            return any(
                isinstance(leaf, jax.Array) for leaf in jax.tree_util.tree_leaves(value)
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

            def _unpack_decision_vector_jax(
                self, solved_x, optimize_G, coil_set_spec=None
            ):
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
            success_filter_a = (
                module.build_single_stage_target_lane_hardware_success_filter(
                    FakeBoozerSurface(),
                    FakeBS(),
                    banana_curve,
                    vessel_surface,
                    cc_dist=0.05,
                    cs_dist=0.05,
                    ss_dist=0.05,
                    curvature_threshold=40.0,
                )
            )
            success_filter_b = (
                module.build_single_stage_target_lane_hardware_success_filter(
                    FakeBoozerSurface(),
                    FakeBS(),
                    banana_curve,
                    vessel_surface,
                    cc_dist=0.05,
                    cs_dist=0.05,
                    ss_dist=0.05,
                    curvature_threshold=40.0,
                )
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

    def test_profile_target_lane_memory_analysis_forces_profile_collection(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--profile-target-lane-memory-analysis",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.profile_target_lane_memory_analysis)
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

    def test_parse_args_accepts_diagnose_target_lane_first_line_search(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--diagnose-target-lane-first-line-search",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.diagnose_target_lane_first_line_search)
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

    def test_parse_args_accepts_diagnostic_callbacks(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            [
                "single_stage_banana_example.py",
                "--backend",
                "jax",
                "--diagnostic-callbacks",
            ],
        ):
            args = module.parse_args()

        self.assertTrue(args.diagnostic_callbacks)
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

    def test_parse_args_accepts_full_artifacts(self):
        module = self.load_module()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            sys,
            "argv",
            ["single_stage_banana_example.py", "--full-artifacts"],
        ):
            args = module.parse_args()

        self.assertTrue(args.full_artifacts)

    def test_outer_optimizer_progress_records_for_all_target_lane_runs(self):
        module = self.load_module()

        self.assertTrue(
            module.should_record_single_stage_outer_optimizer_progress(True)
        )
        self.assertFalse(
            module.should_record_single_stage_outer_optimizer_progress(False)
        )
        self.assertTrue(
            module.should_record_single_stage_outer_optimizer_progress(
                False,
                optimizer_method="lbfgs",
            )
        )
        self.assertTrue(
            module.should_record_single_stage_outer_optimizer_progress(
                False,
                optimizer_method="lbfgs-trace",
            )
        )

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

    def test_summarize_optimizer_result_for_progress_handles_scaled_phase_state(self):
        module = self.load_module()
        x_state = module.build_target_lane_scaled_outer_phase_state(
            np.array([4.0, 5.0], dtype=np.float64),
            np.array([1.0, -2.0], dtype=np.float64),
        )
        jac_state = module.build_target_lane_scaled_outer_phase_state(
            np.array([4.0, 5.0], dtype=np.float64),
            np.array([3.0, -6.0], dtype=np.float64),
        )
        result = types.SimpleNamespace(
            success=False,
            nit=3,
            status=7,
            nfev=11,
            njev=5,
            ls_status=2,
            message="phase1 completed",
            fun=1.25,
            jac=jac_state,
            x=x_state,
        )

        summary = module.summarize_optimizer_result_for_progress(result)

        self.assertEqual(summary["iterations"], 3)
        self.assertEqual(summary["status"], 7)
        self.assertEqual(summary["ls_status"], 2)
        self.assertEqual(summary["message"], "phase1 completed")
        self.assertEqual(summary["diagnostics"]["fun"], 1.25)
        self.assertTrue(summary["diagnostics"]["jac_finite"])
        self.assertEqual(summary["diagnostics"]["jac_inf_norm"], 6.0)
        self.assertTrue(summary["diagnostics"]["x_finite"])
        self.assertFalse(summary["diagnostics"]["invalid_state"])

    def test_summarize_optimizer_result_for_progress_includes_state_trace(self):
        module = self.load_module()
        result = types.SimpleNamespace(
            success=True,
            nit=1,
            status=4,
            nfev=3,
            njev=3,
            ls_status=0,
            message="ok",
            fun=0.5,
            jac=np.array([0.1, -0.2], dtype=np.float64),
            x=np.array([1.0, 2.0], dtype=np.float64),
            optimizer_state_trace=(
                {
                    "iteration": 1,
                    "x": np.array([1.0, 2.0], dtype=np.float64),
                    "fun": 1.0,
                    "jac": np.array([0.5, -0.25], dtype=np.float64),
                    "jac_inf_norm": 0.5,
                    "search_direction": np.array([-0.5, 0.25], dtype=np.float64),
                    "search_direction_dot_grad": -0.3125,
                    "step_scale": 0.25,
                    "step": np.array([-0.125, 0.0625], dtype=np.float64),
                    "trial_x": np.array([0.875, 2.0625], dtype=np.float64),
                    "trial_fun": 0.5,
                    "trial_jac": np.array([0.1, -0.2], dtype=np.float64),
                    "trial_jac_inf_norm": 0.2,
                    "nfev": 3,
                    "njev": 3,
                    "line_search_status": 0,
                    "valid_curvature": True,
                    "accepted": True,
                    "converged": False,
                },
            ),
        )

        summary = module.summarize_optimizer_result_for_progress(result)

        trace = summary["optimizer_state_trace"]
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["iteration"], 1)
        self.assertEqual(trace[0]["x"]["values"], [1.0, 2.0])
        self.assertEqual(trace[0]["trial_x"]["values"], [0.875, 2.0625])
        self.assertEqual(trace[0]["step_scale"]["value"], 0.25)
        self.assertEqual(trace[0]["line_search_status"], 0)
        self.assertTrue(trace[0]["accepted"])

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

    def test_target_lane_diagnostic_callbacks_enabled_accepts_new_flag(self):
        module = self.load_module()

        self.assertTrue(
            module.target_lane_diagnostic_callbacks_enabled(
                types.SimpleNamespace(diagnostic_callbacks=True)
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

    def test_extend_target_lane_invalid_state_events_from_result_uses_structured_log(
        self,
    ):
        module = self.load_module()
        events = []
        result = types.SimpleNamespace(
            invalid_step_log=[
                {
                    "iteration": 4,
                    "step_scale": 0.125,
                    "line_search_failed": False,
                    "nonfinite_step": True,
                    "stalled_step": False,
                    "valid_curvature": False,
                    "trial_converged": False,
                    "ls_status": 2,
                    "requested_initial_step": 0.125,
                    "first_tested_alpha": 0.125,
                    "best_finite_alpha": 0.0625,
                    "returned_alpha": 0.0,
                    "failure_reason": "nonfinite",
                    "armijo_margin": 1.25,
                    "curvature_margin": 2.5,
                }
            ]
        )

        module.extend_target_lane_invalid_state_events_from_result(
            events,
            result,
            phase="phase1",
        )

        self.assertEqual(
            events,
            [
                {
                    "phase": "phase1",
                    "iteration": 4,
                    "step_scale": {
                        "value": 0.125,
                        "finite": True,
                        "classification": None,
                    },
                    "line_search_failed": False,
                    "nonfinite_step": True,
                    "stalled_step": False,
                    "valid_curvature": False,
                    "trial_converged": False,
                    "ls_status": 2,
                    "requested_initial_step": {
                        "value": 0.125,
                        "finite": True,
                        "classification": None,
                    },
                    "first_tested_alpha": {
                        "value": 0.125,
                        "finite": True,
                        "classification": None,
                    },
                    "best_finite_alpha": {
                        "value": 0.0625,
                        "finite": True,
                        "classification": None,
                    },
                    "returned_alpha": {
                        "value": 0.0,
                        "finite": True,
                        "classification": None,
                    },
                    "armijo_margin": {
                        "value": 1.25,
                        "finite": True,
                        "classification": None,
                    },
                    "curvature_margin": {
                        "value": 2.5,
                        "finite": True,
                        "classification": None,
                    },
                    "failure_reason": "nonfinite",
                }
            ],
        )

    def test_single_stage_retry_trigger_ignores_invalid_curvature_without_rejection_cause(
        self,
    ):
        module = self.load_module()
        events = [
            {
                "line_search_failed": False,
                "nonfinite_step": False,
                "stalled_step": False,
                "valid_curvature": False,
                "step_scale": {"value": 0.25},
            }
        ]

        self.assertFalse(module.single_stage_retry_triggered_by_invalid_state(events))

    def test_single_stage_retry_trigger_keeps_rejected_step_causes(
        self,
    ):
        module = self.load_module()

        for cause in ("line_search_failed", "nonfinite_step", "stalled_step"):
            event = {
                "line_search_failed": False,
                "nonfinite_step": False,
                "stalled_step": False,
                "valid_curvature": True,
                "step_scale": {"value": 0.25},
            }
            event[cause] = True

            self.assertTrue(module.single_stage_retry_triggered_by_invalid_state([event]))

    def test_resolve_single_stage_outer_maxls_rejects_nonpositive_budget(self):
        module = self.load_module()

        with self.assertRaisesRegex(ValueError, "outer_maxls must be at least 1"):
            module.resolve_single_stage_outer_maxls("jax", "ondevice", 0)

    def test_resolve_single_stage_outer_maxls_uses_benchmark_budget_for_target_lane(
        self,
    ):
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

        self.assertTrue(module.should_write_single_stage_full_artifacts(False, False))
        self.assertFalse(module.should_write_single_stage_full_artifacts(False, True))
        self.assertFalse(module.should_write_single_stage_full_artifacts(True, False))
        self.assertFalse(module.should_write_single_stage_full_artifacts(True, True))
        self.assertFalse(
            module.should_write_single_stage_full_artifacts(
                False,
                False,
                backend="jax",
            )
        )
        self.assertTrue(
            module.should_write_single_stage_full_artifacts(
                False,
                False,
                backend="jax",
                full_artifacts=True,
            )
        )

    def test_should_write_single_stage_restart_artifacts_only_skips_benchmark(self):
        module = self.load_module()

        self.assertTrue(module.should_write_single_stage_restart_artifacts(False))
        self.assertFalse(module.should_write_single_stage_restart_artifacts(True))

    def test_temporary_boozer_surface_option_overrides_restores_original_values(self):
        module = self.load_module()
        boozer_surface = types.SimpleNamespace(
            options={"bfgs_tol": 1e-10, "verbose": True}
        )

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

    def test_use_experimental_target_lane_value_and_grad_only_on_jax_targets(self):
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
        self.assertTrue(
            module.use_experimental_target_lane_value_and_grad(
                backend="jax",
                optimizer_backend="scipy-jax",
                enabled=True,
            )
        )
        self.assertTrue(
            module.use_target_lane_value_and_grad(
                backend="jax",
                optimizer_backend="scipy-jax",
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
            "get_traceable_single_stage_seeded_value_and_grad_builder",
            return_value=self._make_seeded_value_and_grad_builder(
                value_and_grad=value_and_grad_marker,
                optimizer_initial_value_and_grad=(
                    jnp.asarray(1.5, dtype=jnp.float64),
                    jnp.asarray([0.2, -0.4], dtype=jnp.float64),
                ),
            ),
        ), patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=_runtime_builder,
        ):
            (
                scalar_fun,
                value_and_grad_fun,
                target_lane_profile,
                optimizer_initial_value_and_grad,
            ) = (
                module.build_target_lane_outer_objectives(
                    object(),
                    object(),
                    object(),
                    use_value_and_grad=True,
                    profile_target_lane=False,
                    outer_objective_config=None,
                )
            )

        self.assertIsNotNone(scalar_fun)
        self.assertIs(value_and_grad_fun, value_and_grad_marker)
        self.assertIsNone(target_lane_profile)
        self.assertIsNotNone(optimizer_initial_value_and_grad)
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
            (
                scalar_fun,
                value_and_grad_fun,
                target_lane_profile,
                optimizer_initial_value_and_grad,
            ) = (
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
        self.assertIsNone(optimizer_initial_value_and_grad)
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

    def test_resolve_single_stage_final_penalty_metrics_rejects_missing_target_lane_snapshot(
        self,
    ):
        module = self.load_module()

        class RejectingPenalty:
            def J(self):
                raise AssertionError("host-side penalty wrapper should not be used")

        class RejectingDistance(RejectingPenalty):
            def shortest_distance(self):
                raise AssertionError("host-side distance wrapper should not be used")

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            side_effect=AssertionError(
                "final results must not rebuild target-lane host runtime bundles"
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "Missing cached target-lane final reporting metrics",
            ):
                module.resolve_single_stage_final_penalty_metrics(
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
                    run_dict={},
                    init_only=False,
                    termination_message="ok",
                    optimizer_success=True,
                )

    def test_resolve_single_stage_final_penalty_metrics_prefers_cached_target_lane_summary(
        self,
    ):
        module = self.load_module()
        cached_metrics = self._make_reporting_runtime_summary(
            include_distance_metrics=True
        )
        run_dict = {
            "target_lane_reporting_metrics": dict(cached_metrics),
            "target_lane_reporting_coil_dofs": np.array([1.0, -2.0], dtype=np.float64),
            "target_lane_reporting_include_distance_metrics": True,
        }

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            side_effect=AssertionError(
                "cached accepted-step reporting should avoid rebuilding the runtime bundle"
            ),
        ):
            metrics = module.resolve_single_stage_final_penalty_metrics(
                use_target_lane=True,
                benchmark_mode=False,
                skip_outer_optimizer=False,
                boozer_surface=object(),
                bs=object(),
                iota_target=0.21,
                coil_dofs=np.array([1.0, -2.0], dtype=np.float64),
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
                run_dict=run_dict,
            )
            init_only_metrics = module.resolve_single_stage_final_penalty_metrics(
                use_target_lane=True,
                benchmark_mode=False,
                skip_outer_optimizer=True,
                boozer_surface=object(),
                bs=object(),
                iota_target=0.21,
                coil_dofs=np.array([1.0, -2.0], dtype=np.float64),
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
                run_dict=run_dict,
                init_only=True,
            )

        self.assertEqual(metrics, cached_metrics)
        self.assertEqual(init_only_metrics, cached_metrics)

    def test_restore_single_stage_local_incumbent_preserves_target_lane_reporting_cache(
        self,
    ):
        module = self.load_module()
        cached_metrics = self._make_reporting_runtime_summary(
            include_distance_metrics=True
        )
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict.update(
            {
                "target_lane_reporting_metrics": copy.deepcopy(cached_metrics),
                "target_lane_reporting_coil_dofs": np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    dtype=np.float64,
                ),
                "target_lane_reporting_include_distance_metrics": True,
            }
        )

        incumbent_state = module.snapshot_single_stage_local_incumbent_state(run_dict)
        run_dict["target_lane_reporting_metrics"]["final_G"] = -1.0
        run_dict["target_lane_reporting_coil_dofs"] = np.ones(5, dtype=np.float64)
        run_dict["target_lane_reporting_include_distance_metrics"] = False

        module.restore_single_stage_local_incumbent_state(run_dict, incumbent_state)

        self.assertEqual(run_dict["target_lane_reporting_metrics"], cached_metrics)
        np.testing.assert_allclose(
            run_dict["target_lane_reporting_coil_dofs"],
            np.zeros(5, dtype=np.float64),
        )
        self.assertTrue(run_dict["target_lane_reporting_include_distance_metrics"])

    def test_snapshot_single_stage_local_incumbent_rejects_partial_target_lane_reporting_cache(
        self,
    ):
        module = self.load_module()
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["target_lane_reporting_metrics"] = {
            "final_non_qs": 0.11,
        }

        with self.assertRaisesRegex(RuntimeError, "partially populated"):
            module.snapshot_single_stage_local_incumbent_state(run_dict)

    def test_zero_accepted_step_target_lane_failure_keeps_reportable_anchor(self):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        cached_metrics = self._make_reporting_runtime_summary(
            include_distance_metrics=True
        )
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict.update(
            {
                "target_lane_reporting_metrics": copy.deepcopy(cached_metrics),
                "target_lane_reporting_coil_dofs": np.zeros(5, dtype=np.float64),
                "target_lane_reporting_include_distance_metrics": True,
            }
        )
        module.record_single_stage_local_incumbent(run_dict, stage="initial")
        invalid_state_events = []

        def fake_run_single_stage_optimizer(
            fun,
            dofs,
            *,
            callback,
            contract,
            maxiter,
            ftol,
            gtol,
            maxcor,
            outer_maxls,
            scalar_fun,
            progress_callback=None,
            target_lane_initial_step_size,
            failure_callback,
        ):
            del (
                fun,
                dofs,
                callback,
                contract,
                maxiter,
                ftol,
                gtol,
                maxcor,
                outer_maxls,
                scalar_fun,
                progress_callback,
                target_lane_initial_step_size,
                failure_callback,
            )
            return self._build_target_lane_retry_result(
                x=np.ones(5, dtype=np.float64),
                nit=0,
                nfev=2,
                njev=2,
                success=False,
                message="line search failed",
                status=5,
                step_scale=0.0,
                line_search_failed=True,
                ls_status=1,
            )

        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=1,
            retry_step_shrink_factor=0.5,
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=fake_run_single_stage_optimizer,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
                    phase="phase1",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=None,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                )
            )

        self.assertFalse(result.success)
        self.assertTrue(retry_summary["restored_preserved_local_state"])
        np.testing.assert_allclose(result.x, np.zeros(5, dtype=np.float64))
        metrics = module.resolve_single_stage_final_penalty_metrics(
            use_target_lane=True,
            benchmark_mode=False,
            skip_outer_optimizer=False,
            boozer_surface=object(),
            bs=object(),
            iota_target=0.21,
            coil_dofs=result.x,
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
            run_dict=run_dict,
        )

        self.assertEqual(metrics, cached_metrics)

    def test_cache_single_stage_target_lane_init_reporting_snapshot_uses_runtime_sync(
        self,
    ):
        module = self.load_module()
        calls = []

        def fake_sync(run_dict, coil_dofs, *, benchmark_mode, update_run_state):
            calls.append((run_dict, coil_dofs, benchmark_mode, update_run_state))
            run_dict["target_lane_reporting_metrics"] = {"banana_current_A": 123.0}
            run_dict["target_lane_reporting_coil_dofs"] = np.asarray(coil_dofs)
            run_dict["target_lane_reporting_include_distance_metrics"] = False
            return {"reporting_metrics": {"banana_current_A": 123.0}}

        with patch.object(
            module,
            "build_single_stage_target_lane_hardware_success_filter",
            side_effect=AssertionError(
                "penalty target-lane reporting does not use a hard success filter"
            ),
        ) as success_filter_builder, patch.object(
            module,
            "build_target_lane_outer_objective_config",
            return_value="objective-config",
        ) as config_builder, patch.object(
            module,
            "build_single_stage_target_lane_accepted_step_sync",
            return_value=fake_sync,
        ) as sync_builder:
            run_dict = {}
            summary = module.cache_single_stage_target_lane_init_reporting_snapshot(
                boozer_surface="boozer",
                bs="field",
                banana_curve="banana",
                vessel_surface="vessel",
                iota_target=0.21,
                run_dict=run_dict,
                coil_dofs=np.array([1.0, -2.0], dtype=np.float64),
                benchmark_mode=True,
                disable_success_filter=False,
                length_target=1.7,
                cc_dist=0.05,
                cc_weight=100.0,
                cs_dist=0.02,
                cs_weight=1.0,
                ss_dist=0.04,
                surf_dist_weight=1000.0,
                residual_weight=1000.0,
                iota_weight=100.0,
                length_weight=1.0,
                curvature_threshold=40.0,
                curvature_weight=0.1,
            )

        success_filter_builder.assert_not_called()
        config_builder.assert_called_once()
        sync_builder.assert_called_once_with(
            "boozer",
            "field",
            0.21,
            outer_objective_config="objective-config",
            success_filter=None,
        )
        self.assertEqual(summary["reporting_metrics"]["banana_current_A"], 123.0)
        self.assertEqual(calls[0][0], run_dict)
        np.testing.assert_allclose(calls[0][1], np.array([1.0, -2.0]))
        self.assertTrue(calls[0][2])
        self.assertFalse(calls[0][3])
        self.assertEqual(
            run_dict["target_lane_reporting_metrics"]["banana_current_A"],
            123.0,
        )

    def test_target_restart_artifact_export_skips_host_field_diagnostics(
        self,
    ):
        module = self.load_module()
        saved_paths = []
        events = []

        class RestartSurface:
            x = np.array([0.0, 0.0], dtype=np.float64)
            mpol = 1
            ntor = 1
            nfp = 5
            stellsym = True
            quadpoints_phi = np.linspace(0.0, 0.2, 3, endpoint=False)
            quadpoints_theta = np.linspace(0.0, 1.0, 3, endpoint=False)

            def save(self, path):
                saved_paths.append(str(path))

            def gamma(self):
                raise AssertionError("restart-only target export should not sample gamma")

            def unitnormal(self):
                raise AssertionError(
                    "restart-only target export should not sample unit normals"
                )

            def volume(self):
                raise AssertionError("final JSON should use snapshot metrics")

        class RestartField:
            x = np.array([0.0, 0.0], dtype=np.float64)
            coils = []

            def save(self, path):
                saved_paths.append(str(path))

            def B(self):
                raise AssertionError("restart-only target export should not evaluate B")

        run_dict = {
            "sdofs": np.array([0.1, -0.2], dtype=np.float64),
            "iota": 0.21,
            "G": 1.75,
            "self_intersection_check_available": True,
        }
        snapshot = types.SimpleNamespace(
            solved_surface_state={
                "sdofs": run_dict["sdofs"],
                "iota": run_dict["iota"],
                "G": run_dict["G"],
            },
        )
        boozer_surface = types.SimpleNamespace(
            surface=RestartSurface(),
            res={"iota": 0.0, "G": 0.0},
        )

        with tempfile.TemporaryDirectory() as tmpdir, patch.object(
            module,
            "update_self_intersection_status",
            side_effect=lambda rd, _surface, **_kwargs: rd.__setitem__(
                "self_intersection_check_available", True
            )
            or False,
        ):
            module.restore_single_stage_host_state(
                use_target_lane=True,
                JF=types.SimpleNamespace(),
                boozer_surface=boozer_surface,
                run_dict=run_dict,
                coil_dofs=np.array([1.0, -2.0], dtype=np.float64),
                apply_coil_dofs=lambda _dofs: None,
                bs_diag=RestartField(),
                record_outer_optimizer_event=lambda name, **payload: events.append(
                    (name, payload)
                ),
            )
            module.export_requested_single_stage_artifacts(
                solved_surface_state=snapshot.solved_surface_state,
                coil_dofs=np.array([1.0, -2.0], dtype=np.float64),
                num_tf_coils=0,
                tf_current_A=0.0,
                banana_current_A=0.0,
                stage2_seed=self._jax_runtime_seed_spec_field_kwargs(module)[
                    "stage2_seed"
                ],
                output_dir=tmpdir,
                boozer_surface=boozer_surface,
                bs_diag=RestartField(),
                surf_coils=object(),
                hbt=object(),
                VV=object(),
                write_restart_artifacts=True,
                write_host_restart_artifacts=True,
                write_full_artifacts=False,
                timings={},
            )
            result = module.run_single_stage_host_diagnostics(
                boozer_surface=boozer_surface,
                run_dict=run_dict,
                timings={},
            )
            runtime_spec_written = os.path.exists(
                os.path.join(tmpdir, module._SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME)
            )

        self.assertFalse(result.self_intersecting)
        self.assertTrue(result.self_intersection_check_available)
        self.assertTrue(
            any(path.endswith("biot_savart_opt.json") for path in saved_paths)
        )
        self.assertTrue(any(path.endswith("surf_opt.json") for path in saved_paths))
        self.assertTrue(runtime_spec_written)
        self.assertEqual(events[0][0], "host_state_restore_started")
        np.testing.assert_allclose(boozer_surface.surface.x, run_dict["sdofs"])
        self.assertEqual(boozer_surface.res["iota"], run_dict["iota"])
        self.assertEqual(boozer_surface.res["G"], run_dict["G"])

    def test_target_restart_artifacts_do_not_require_host_postprocess_boundary(self):
        module = self.load_module()

        self.assertFalse(
            module.single_stage_host_postprocess_required(
                use_target_lane=True,
                write_full_artifacts=False,
            )
        )
        self.assertFalse(
            module.single_stage_host_postprocess_required(
                use_target_lane=True,
                write_full_artifacts=False,
            )
        )

    def test_restart_artifact_export_requires_host_graph_on_cpu_lane(self):
        module = self.load_module()

        self.assertTrue(
            module.single_stage_host_artifact_export_required(
                use_target_lane=False,
                write_restart_artifacts=True,
                write_full_artifacts=False,
            )
        )
        self.assertFalse(
            module.single_stage_host_artifact_export_required(
                use_target_lane=True,
                write_restart_artifacts=True,
                write_full_artifacts=False,
            )
        )
        self.assertFalse(
            module.single_stage_host_artifact_export_required(
                use_target_lane=False,
                runtime_seed_restart_artifacts=True,
                write_restart_artifacts=True,
                write_full_artifacts=False,
            )
        )
        self.assertTrue(
            module.single_stage_host_artifact_export_required(
                use_target_lane=True,
                write_restart_artifacts=True,
                write_full_artifacts=True,
            )
        )

    def test_init_only_cpu_restart_export_does_not_restore_optimizer_dofs(self):
        module = self.load_module()

        self.assertFalse(
            module.single_stage_final_host_restore_required(
                skip_outer_optimizer=True,
                use_target_lane=False,
                host_state_restored_for_final=False,
            )
        )
        self.assertTrue(
            module.single_stage_final_host_restore_required(
                skip_outer_optimizer=False,
                use_target_lane=False,
                host_state_restored_for_final=False,
            )
        )
        self.assertTrue(
            module.single_stage_final_host_restore_required(
                skip_outer_optimizer=True,
                use_target_lane=True,
                host_state_restored_for_final=False,
            )
        )
        self.assertFalse(
            module.single_stage_final_host_restore_required(
                skip_outer_optimizer=False,
                use_target_lane=False,
                host_state_restored_for_final=True,
            )
        )

    def test_build_single_stage_final_result_snapshot_copies_result_state(self):
        module = self.load_module()
        run_dict = {
            "sdofs": np.array([0.1, -0.2], dtype=np.float64),
            "iota": 0.21,
            "G": 1.75,
        }
        final_metrics = self._make_reporting_runtime_summary(
            include_distance_metrics=True
        )
        hardware_status = {"success": True, "violations": []}
        final_distances = {
            "curve_curve_min_dist": 0.11,
            "curve_surface_min_dist": 0.22,
            "surface_vessel_min_dist": 0.33,
        }
        optimizer_result = {
            "iterations": 7,
            "success": True,
            "termination_message": "done",
            "status": 0,
            "nfev": 8,
            "njev": 9,
            "ls_status": None,
        }
        optimizer_diagnostics = {"fun": 1.25, "invalid_state": False}
        timings = {"outer_optimizer_s": 2.0}

        snapshot = module.build_single_stage_final_result_snapshot(
            final_coil_dofs=np.array([1.0, -2.0], dtype=np.float64),
            run_dict=run_dict,
            final_metrics=final_metrics,
            final_distances=final_distances,
            hardware_status=hardware_status,
            optimizer_result=optimizer_result,
            optimizer_diagnostics=optimizer_diagnostics,
            timings=timings,
            write_restart_artifacts=True,
            write_full_artifacts=False,
            boozer_optimizer_method="BFGS",
            field_error=final_metrics["field_error"],
            self_intersecting=False,
            self_intersection_check_available=True,
        )

        final_metrics["final_iota"] = 99.0
        final_distances["curve_curve_min_dist"] = 99.0
        hardware_status["success"] = False
        optimizer_result["iterations"] = 99
        optimizer_diagnostics["fun"] = 99.0
        timings["outer_optimizer_s"] = 99.0
        run_dict["sdofs"][0] = 99.0

        self.assertEqual(snapshot.final_metrics["final_iota"], 0.21)
        self.assertEqual(snapshot.final_distances["curve_curve_min_dist"], 0.11)
        self.assertTrue(snapshot.hardware_status["success"])
        self.assertEqual(snapshot.optimizer_result["iterations"], 7)
        self.assertEqual(snapshot.optimizer_diagnostics["fun"], 1.25)
        self.assertEqual(snapshot.timings["outer_optimizer_s"], 2.0)
        self.assertEqual(snapshot.boozer_optimizer_method, "BFGS")
        self.assertEqual(snapshot.results_payload, {})
        np.testing.assert_allclose(
            snapshot.solved_surface_state["sdofs"],
            np.array([0.1, -0.2], dtype=np.float64),
        )

    def test_write_single_stage_results_json_uses_snapshot_payload(self):
        module = self.load_module()
        snapshot = module.SingleStageFinalResultSnapshot(
            final_coil_dofs=np.array([1.0], dtype=np.float64),
            solved_surface_state={"sdofs": np.array([0.1], dtype=np.float64)},
            final_metrics={},
            final_distances={},
            hardware_status={},
            optimizer_result={},
            optimizer_diagnostics={},
            timings={"script_total_s": 1.5},
            artifact_policy={
                "write_restart_artifacts": False,
                "write_full_artifacts": False,
            },
            boozer_optimizer_method="BFGS",
            results_payload={},
            field_error=0.01,
            self_intersecting=False,
            self_intersection_check_available=True,
        )
        mutable_results = {"TIMINGS": {"script_total_s": 99.0}, "FIELD_ERROR": 0.01}
        final_snapshot = module.with_single_stage_results_payload(
            snapshot,
            mutable_results,
        )
        mutable_results["FIELD_ERROR"] = 99.0
        mutable_results["TIMINGS"] = {"script_total_s": 99.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            module.write_single_stage_results_json(tmpdir, final_snapshot)
            payload = json.loads((Path(tmpdir) / "results.json").read_text())

        self.assertEqual(payload["FIELD_ERROR"], 0.01)
        self.assertEqual(payload["TIMINGS"], {"script_total_s": 1.5})

    def test_summarize_single_stage_final_optimizer_result_copies_status(self):
        module = self.load_module()
        result = types.SimpleNamespace(status=2, nfev=5, njev=3, ls_status=4)

        summary = module.summarize_single_stage_final_optimizer_result(
            result=result,
            ran_optimizer=True,
            iterations=11,
            optimizer_success=False,
            termination_message="line search failed",
        )

        self.assertEqual(
            summary,
            {
                "iterations": 11,
                "success": False,
                "termination_message": "line search failed",
                "status": 2,
                "nfev": 5,
                "njev": 3,
                "ls_status": 4,
            },
        )

        result.status = 99
        self.assertEqual(summary["status"], 2)

    def test_resolve_single_stage_final_banana_current_uses_target_snapshot(self):
        module = self.load_module()

        class RejectingCurrent:
            def get_value(self):
                raise AssertionError("target results must not read host current")

        current = module.resolve_single_stage_final_banana_current_A(
            use_target_lane=True,
            final_metrics={"banana_current_A": 456.0},
            banana_current=RejectingCurrent(),
        )

        self.assertEqual(current, 456.0)

    def test_resolve_single_stage_final_penalty_metrics_skips_target_lane_distances_in_benchmark_mode(
        self,
    ):
        module = self.load_module()
        runtime_summary = self._make_reporting_runtime_summary(
            include_distance_metrics=False
        )
        runtime_summary["hardware_status"] = {
            "success": None,
            "violations": ["skipped_in_benchmark_mode"],
        }
        run_dict = {
            "target_lane_reporting_metrics": dict(runtime_summary),
            "target_lane_reporting_coil_dofs": np.array([1.0, -2.0], dtype=np.float64),
            "target_lane_reporting_include_distance_metrics": False,
        }

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            side_effect=AssertionError(
                "cached accepted-step reporting should avoid rebuilding the runtime bundle"
            ),
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
                run_dict=run_dict,
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
        fake_bs = types.SimpleNamespace(
            coil_set_spec_from_dofs=lambda coil_dofs: coil_dofs
        )

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=self._make_reporting_runtime_builder(
                captured, runtime_summary
            ),
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
        self.assertTrue(captured["value_and_grad_called"])
        self.assertIn("objective_value", summary)
        self.assertEqual(
            summary["reporting_metrics"]["final_non_qs"],
            runtime_summary["final_non_qs"],
        )
        self.assertEqual(run_dict["J"], summary["objective_value"])
        np.testing.assert_allclose(
            run_dict["dJ"],
            np.array([0.3, -0.4], dtype=np.float64),
        )
        self.assertTrue(run_dict["hardware_constraint_status"]["success"])
        np.testing.assert_allclose(
            run_dict["target_lane_reporting_coil_dofs"],
            np.array([1.0, -2.0], dtype=np.float64),
        )
        self.assertEqual(
            run_dict["target_lane_reporting_metrics"],
            summary["reporting_metrics"],
        )
        self.assertTrue(run_dict["target_lane_reporting_include_distance_metrics"])
        incumbent = run_dict["latest_local_incumbent"]
        for key in module._TARGET_LANE_REPORTING_CACHE_KEYS:
            self.assertIn(key, incumbent)
        self.assertEqual(
            incumbent["target_lane_reporting_metrics"],
            summary["reporting_metrics"],
        )
        np.testing.assert_allclose(
            incumbent["target_lane_reporting_coil_dofs"],
            np.array([1.0, -2.0], dtype=np.float64),
        )
        self.assertTrue(
            incumbent["target_lane_reporting_include_distance_metrics"]
        )

    def test_build_single_stage_target_lane_accepted_step_sync_can_skip_state_commit(
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
        fake_bs = types.SimpleNamespace(
            coil_set_spec_from_dofs=lambda coil_dofs: coil_dofs
        )

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=self._make_reporting_runtime_builder(
                captured, runtime_summary
            ),
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
                "x_prev": np.array([8.0, -3.0], dtype=np.float64),
                "it": 4,
            }
            run_dict_before = copy.deepcopy(run_dict)
            summary = sync(
                run_dict,
                jax.device_put(np.array([1.0, -2.0], dtype=np.float64)),
                benchmark_mode=False,
                update_run_state=False,
            )

        self.assertIn("objective_value", summary)
        self.assertNotIn("value_and_grad_called", captured)
        self.assertNotIn("hardware_constraint_status", run_dict)
        self.assertEqual(
            run_dict["target_lane_reporting_metrics"],
            summary["reporting_metrics"],
        )
        np.testing.assert_allclose(
            run_dict["target_lane_reporting_coil_dofs"],
            np.array([1.0, -2.0], dtype=np.float64),
        )
        self.assertTrue(run_dict["target_lane_reporting_include_distance_metrics"])
        np.testing.assert_allclose(run_dict["sdofs"], run_dict_before["sdofs"])
        self.assertEqual(run_dict["iota"], run_dict_before["iota"])
        self.assertEqual(run_dict["G"], run_dict_before["G"])
        self.assertEqual(run_dict["J"], run_dict_before["J"])
        np.testing.assert_allclose(run_dict["dJ"], run_dict_before["dJ"])
        np.testing.assert_allclose(run_dict["x_prev"], run_dict_before["x_prev"])
        self.assertEqual(run_dict["it"], run_dict_before["it"])

    def test_build_single_stage_target_lane_accepted_step_sync_prefers_runtime_forward_result(
        self,
    ):
        module = self.load_module()
        captured = {}
        runtime_summary = self._make_reporting_runtime_summary(
            include_distance_metrics=True
        )
        forward_result = {
            "success": jnp.asarray(True, dtype=bool),
            "primal_success": jnp.asarray(True, dtype=bool),
            "sdofs": jnp.asarray([0.6, -0.3], dtype=jnp.float64),
            "iota": jnp.asarray(0.24, dtype=jnp.float64),
            "G": jnp.asarray(1.9, dtype=jnp.float64),
            "x": jnp.asarray([0.6, -0.3, 0.24, 1.9], dtype=jnp.float64),
        }
        fake_boozer_surface = types.SimpleNamespace(
            run_code_traceable=lambda *_args: (_ for _ in ()).throw(
                AssertionError("runtime forward_result should be used")
            )
        )
        fake_bs = types.SimpleNamespace(
            coil_set_spec_from_dofs=lambda coil_dofs: coil_dofs
        )

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=self._make_reporting_runtime_builder(
                captured,
                runtime_summary,
                forward_result=lambda _coil_dofs: forward_result,
            ),
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
            sync(
                run_dict,
                jax.device_put(np.array([1.0, -2.0], dtype=np.float64)),
                benchmark_mode=False,
            )

        np.testing.assert_allclose(
            run_dict["sdofs"],
            np.array([0.6, -0.3], dtype=np.float64),
        )
        self.assertAlmostEqual(run_dict["iota"], 0.24)
        self.assertAlmostEqual(run_dict["G"], 1.9)

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

        def _profile(profile_suite, coil_dofs, *, include_memory_analysis=False):
            self.assertFalse(include_memory_analysis)
            profiled_calls.append((profile_suite, coil_dofs))
            return {"ok": True}

        with patch.object(
            module,
            "get_traceable_single_stage_runtime_bundle_builder",
            return_value=_runtime_builder,
        ), patch.object(
            module,
            "get_traceable_single_stage_seeded_value_and_grad_builder",
            return_value=self._make_seeded_value_and_grad_builder(
                value_and_grad=lambda x: x,
                optimizer_initial_value_and_grad=(
                    jnp.asarray(2.0, dtype=jnp.float64),
                    jnp.asarray([0.1, -0.2], dtype=jnp.float64),
                ),
            ),
        ), patch.object(
            module,
            "profile_traceable_target_lane_objective",
            side_effect=_profile,
        ):
            _, _, target_lane_profile, optimizer_initial_value_and_grad = (
                module.build_target_lane_outer_objectives(
                    object(),
                    bs,
                    object(),
                    use_value_and_grad=True,
                    profile_target_lane=True,
                    profile_batch_size=1,
                    outer_objective_config=None,
                )
            )

        self.assertIsNotNone(optimizer_initial_value_and_grad)
        self.assertEqual(target_lane_profile["ok"], True)
        self.assertEqual(len(profiled_calls), 1)
        self.assertEqual(profiled_calls[0][0], "profile-suite-marker")
        self.assertIsInstance(profiled_calls[0][1], jax.Array)
        self.assertNotEqual(
            tuple(np.asarray(profiled_calls[0][1], dtype=np.float64)),
            tuple(bs.x),
        )
        self.assertEqual(
            target_lane_profile["profile_point_kind"], "baseline_perturbed"
        )

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
            any(not np.array_equal(row, np.array([0.0, -2.0])) for row in host_batch)
        )

    def test_profile_traceable_target_lane_objective_records_memory_analysis(self):
        module = self.load_module()
        coil_dofs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

        profile_suite = {
            "forward_result": jax.jit(
                lambda x: {
                    "value": jnp.sum(x**2),
                    "x": 2.0 * x,
                    "linear_solve_factors": x + 1.0,
                    "success": jnp.asarray(True),
                }
            ),
            "forward_value": jax.jit(lambda x: jnp.sum(x**2)),
            "warmstart_predict": jax.jit(
                lambda x: {
                    "x": x + 0.5,
                    "success": jnp.asarray(True),
                }
            ),
            "inner_solve": jax.jit(
                lambda x: {
                    "x": 2.0 * x,
                    "linear_solve_factors": x + 1.0,
                    "success": jnp.asarray(True),
                }
            ),
            "surface_geometry": jax.jit(lambda x: x + 1.0),
            "field_eval": jax.jit(lambda x, solved_x: x + solved_x),
            "solved_total_objective": jax.jit(
                lambda x, solved_x: jnp.sum(x * solved_x)
            ),
            "solved_total_gradient": jax.jit(
                lambda x, solved_x, factors: x + solved_x + factors
            ),
            "value_and_grad_pipeline": jax.jit(
                lambda x: (jnp.sum(x**2), 2.0 * x)
            ),
        }

        profile = module.profile_traceable_target_lane_objective(
            profile_suite,
            coil_dofs,
            include_memory_analysis=True,
        )

        memory_analysis = profile["memory_analysis"]
        self.assertEqual(
            set(memory_analysis),
            {
                "forward_result",
                "forward_value",
                "warmstart_predict",
                "inner_solve",
                "surface_geometry",
                "field_eval",
                "solved_total_objective",
                "solved_total_gradient",
                "value_and_grad_pipeline",
            },
        )
        self.assertGreaterEqual(
            memory_analysis["value_and_grad_pipeline"]["total_size_in_bytes"],
            0,
        )
        self.assertEqual(profile["solve_success"], True)

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

        def _profile(_profile_suite, _coil_dofs, *, include_memory_analysis=False):
            self.assertFalse(include_memory_analysis)
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
            "get_traceable_single_stage_seeded_value_and_grad_builder",
            return_value=self._make_seeded_value_and_grad_builder(
                value_and_grad=lambda x: x,
                optimizer_initial_value_and_grad=(
                    jnp.asarray(2.0, dtype=jnp.float64),
                    jnp.asarray([0.1, -0.2], dtype=jnp.float64),
                ),
            ),
        ), patch.object(
            module,
            "profile_traceable_target_lane_objective",
            side_effect=_profile,
        ), patch.object(
            module,
            "profile_traceable_target_lane_seed_batch",
            side_effect=_profile_batch,
        ):
            _, _, target_lane_profile, optimizer_initial_value_and_grad = (
                module.build_target_lane_outer_objectives(
                    object(),
                    bs,
                    object(),
                    use_value_and_grad=True,
                    profile_target_lane=True,
                    profile_batch_size=3,
                    outer_objective_config=None,
                )
            )

        self.assertIsNotNone(optimizer_initial_value_and_grad)
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
            return_value=(objective_marker, None, profile_marker, None),
        ) as build_objectives:
            (
                scalar_fun,
                value_and_grad_fun,
                optimizer_initial_value_and_grad,
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
        self.assertIsNone(optimizer_initial_value_and_grad)
        self.assertIs(target_lane_profile, profile_marker)
        self.assertIsNone(success_filter)
        build_objective_config.assert_called_once()
        build_objectives.assert_called_once_with(
            unittest.mock.ANY,
            unittest.mock.ANY,
            unittest.mock.ANY,
            use_value_and_grad=False,
            profile_target_lane=True,
            profile_target_lane_memory_analysis=False,
            profile_batch_size=3,
            outer_objective_config="config-marker",
            success_filter=None,
        )

    def test_prepare_target_lane_outer_objectives_uses_smooth_penalties_without_hard_filter(
        self,
    ):
        module = self.load_module()
        recorded_states = []

        def value_and_grad_marker(state):
            recorded_states.append(state)
            return ("objective-value", "objective-grad")

        with patch.object(
            module,
            "build_single_stage_target_lane_hardware_success_filter",
            side_effect=AssertionError(
                "penalty target lane must stay differentiable through "
                "hardware-constraint penalty regions"
            ),
        ) as build_success_filter, patch.object(
            module,
            "build_target_lane_outer_objective_config",
            return_value="config-marker",
        ) as build_objective_config, patch.object(
            module,
            "build_target_lane_outer_objectives",
            return_value=(
                None,
                value_and_grad_marker,
                None,
                ("seed-value", "seed-grad"),
            ),
        ) as build_objectives:
            (
                scalar_fun,
                value_and_grad_fun,
                optimizer_initial_value_and_grad,
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

        self.assertIsNotNone(scalar_fun)
        self.assertIsNot(scalar_fun, value_and_grad_marker)
        self.assertIs(value_and_grad_fun, value_and_grad_marker)
        self.assertEqual(
            optimizer_initial_value_and_grad,
            ("seed-value", "seed-grad"),
        )
        self.assertIsNone(target_lane_profile)
        self.assertIsNone(success_filter)
        self.assertEqual(scalar_fun("trial-state"), "objective-value")
        self.assertEqual(recorded_states, ["trial-state"])
        build_success_filter.assert_not_called()
        build_objective_config.assert_called_once()
        build_objectives.assert_called_once_with(
            unittest.mock.ANY,
            unittest.mock.ANY,
            unittest.mock.ANY,
            use_value_and_grad=True,
            profile_target_lane=False,
            profile_target_lane_memory_analysis=False,
            profile_batch_size=1,
            outer_objective_config="config-marker",
            success_filter=None,
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
            diagnostic_builder_calls.append((outer_objective_config, success_filter))
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

    def test_build_target_lane_first_line_search_diagnosis_traces_first_trial(self):
        module = self.load_module()

        def value_and_grad(x):
            return 0.5 * jnp.dot(x, x), x

        diagnosis = module.build_target_lane_first_line_search_diagnosis(
            value_and_grad,
            np.asarray([1.0], dtype=np.float64),
            initial_value_and_grad=(
                jnp.asarray(0.5, dtype=jnp.float64),
                jnp.asarray([1.0], dtype=jnp.float64),
            ),
            initial_step_size=0.125,
            maxls=4,
            gtol=1.0e-12,
        )

        self.assertEqual(diagnosis["initial"]["directional_derivative"]["value"], -1.0)
        self.assertTrue(diagnosis["optimizer_step"]["would_accept"])
        self.assertFalse(diagnosis["optimizer_step"]["would_reject"])
        self.assertEqual(diagnosis["line_search"]["nfev"], 1)
        self.assertEqual(
            diagnosis["line_search"]["requested_initial_step"]["value"],
            0.125,
        )
        self.assertEqual(
            diagnosis["line_search"]["first_tested_alpha"]["value"],
            0.125,
        )
        self.assertEqual(
            diagnosis["line_search"]["best_finite_alpha"]["value"],
            0.125,
        )
        self.assertEqual(
            diagnosis["line_search"]["returned_alpha"]["value"],
            0.125,
        )
        self.assertEqual(diagnosis["line_search"]["failure_reason"], "accepted")
        self.assertLessEqual(
            diagnosis["line_search"]["armijo_margin"]["value"],
            0.0,
        )
        self.assertLessEqual(
            diagnosis["line_search"]["curvature_margin"]["value"],
            0.0,
        )
        self.assertEqual(len(diagnosis["line_search"]["trace"]), 1)
        self.assertEqual(
            diagnosis["line_search"]["trace"][0]["alpha"]["value"],
            0.125,
        )
        self.assertLessEqual(
            diagnosis["line_search"]["trace"][0]["armijo_margin"]["value"],
            0.0,
        )
        self.assertLessEqual(
            diagnosis["line_search"]["trace"][0]["curvature_margin"]["value"],
            0.0,
        )
        self.assertTrue(diagnosis["line_search"]["trace"][0]["armijo_satisfied"])
        self.assertTrue(diagnosis["line_search"]["trace"][0]["curvature_satisfied"])

    def test_build_target_lane_first_line_search_diagnosis_uses_lbfgs_seed_step(self):
        module = self.load_module()

        def value_and_grad(x):
            return 0.5 * jnp.dot(x, x), x

        diagnosis = module.build_target_lane_first_line_search_diagnosis(
            value_and_grad,
            np.asarray([100.0], dtype=np.float64),
            initial_value_and_grad=(
                jnp.asarray(5000.0, dtype=jnp.float64),
                jnp.asarray([100.0], dtype=jnp.float64),
            ),
            initial_step_size=None,
            maxls=4,
            gtol=1.0e-12,
        )

        expected_old_old_fval = 5050.0
        expected_alpha = 0.0101
        self.assertAlmostEqual(
            diagnosis["initial"]["old_old_fval"]["value"],
            expected_old_old_fval,
        )
        self.assertAlmostEqual(
            diagnosis["line_search"]["requested_initial_step"]["value"],
            expected_alpha,
        )
        self.assertAlmostEqual(
            diagnosis["line_search"]["first_tested_alpha"]["value"],
            expected_alpha,
        )
        self.assertAlmostEqual(
            diagnosis["line_search"]["trace"][0]["alpha"]["value"],
            diagnosis["line_search"]["first_tested_alpha"]["value"],
        )

    def test_build_target_lane_scaled_phase1_diagnosis_threads_runtime_and_optimizer(
        self,
    ):
        module = self.load_module()
        success_filter_marker = object()
        runtime_builder_calls = []
        seeded_builder_calls = []
        optimizer_calls = []
        optimizer_seed = (
            jnp.asarray(3.5, dtype=jnp.float64),
            jnp.asarray([4.0, -6.0], dtype=jnp.float64),
        )

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

        def _seeded_builder(
            boozer_surface,
            bs,
            iota_target,
            *,
            outer_objective_config=None,
            success_filter=None,
        ):
            del boozer_surface, bs, iota_target
            seeded_builder_calls.append((outer_objective_config, success_filter))
            return types.SimpleNamespace(
                value_and_grad=lambda x: ("seeded-value", x),
                optimizer_initial_value_and_grad=optimizer_seed,
            )

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
            optimizer_initial_value_and_grad,
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
                    "optimizer_initial_value_and_grad": (
                        optimizer_initial_value_and_grad
                    ),
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
            "get_traceable_single_stage_seeded_value_and_grad_builder",
            return_value=_seeded_builder,
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
        self.assertEqual(
            seeded_builder_calls,
            [("config-marker", success_filter_marker)],
        )
        self.assertEqual(len(optimizer_calls), 1)
        self.assertEqual(optimizer_calls[0]["fun"], "scaled-fun")
        self.assertEqual(optimizer_calls[0]["contract_method"], "lbfgs-ondevice")
        self.assertEqual(optimizer_calls[0]["maxiter"], 4)
        self.assertEqual(optimizer_calls[0]["callback"], "scaled-callback")
        self.assertIsNone(optimizer_calls[0]["scalar_fun"])
        optimizer_seed_value, optimizer_seed_grad = optimizer_calls[0][
            "optimizer_initial_value_and_grad"
        ]
        self.assertEqual(float(optimizer_seed_value), 3.5)
        self.assertIsInstance(optimizer_seed_grad, module.ScaledOuterPhaseOptimizerState)
        np.testing.assert_allclose(
            np.asarray(optimizer_seed_grad.step_dofs, dtype=np.float64),
            np.array([1.0, -1.5], dtype=np.float64),
        )
        np.testing.assert_allclose(
            np.asarray(optimizer_seed_grad.anchor_dofs, dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        )
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
        seeded_value_and_grad_calls = []
        optimizer_seed = (
            jax.device_put(np.asarray(5.0, dtype=np.float64)),
            jax.device_put(np.asarray([4.0, -6.0], dtype=np.float64)),
        )

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

        def _seeded_builder(
            boozer_surface,
            bs,
            iota_target,
            *,
            outer_objective_config=None,
            success_filter=None,
        ):
            del (
                boozer_surface,
                bs,
                iota_target,
                outer_objective_config,
                success_filter,
            )

            def _seeded_value_and_grad(x):
                seeded_value_and_grad_calls.append(x)
                self.assertIsInstance(x, jax.Array)
                x_host = np.asarray(jax.device_get(x), dtype=np.float64)
                return np.dot(x_host, x_host), 2.0 * x_host

            return types.SimpleNamespace(
                value_and_grad=_seeded_value_and_grad,
                optimizer_initial_value_and_grad=optimizer_seed,
            )

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
            optimizer_initial_value_and_grad,
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
                optimizer_initial_value_and_grad,
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
            "get_traceable_single_stage_seeded_value_and_grad_builder",
            return_value=_seeded_builder,
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
        self.assertEqual(seeded_value_and_grad_calls, [])

    def test_build_target_lane_scaled_phase1_diagnosis_writes_incremental_checkpoints(
        self,
    ):
        module = self.load_module()
        checkpoint_payloads = []
        optimizer_seed = (
            jnp.asarray(4.0, dtype=jnp.float64),
            jnp.asarray([2.0, -8.0], dtype=jnp.float64),
        )

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

        def _seeded_builder(
            boozer_surface,
            bs,
            iota_target,
            *,
            outer_objective_config=None,
            success_filter=None,
        ):
            del boozer_surface, bs, iota_target, outer_objective_config, success_filter
            return types.SimpleNamespace(
                value_and_grad=lambda x: ("seeded-value", x),
                optimizer_initial_value_and_grad=optimizer_seed,
            )

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
            optimizer_initial_value_and_grad,
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
            seed_value, seed_grad = optimizer_initial_value_and_grad
            self.assertEqual(float(seed_value), 4.0)
            self.assertIsInstance(seed_grad, module.ScaledOuterPhaseOptimizerState)
            np.testing.assert_allclose(
                np.asarray(seed_grad.step_dofs, dtype=np.float64),
                np.array([0.5, -2.0], dtype=np.float64),
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
            "get_traceable_single_stage_seeded_value_and_grad_builder",
            return_value=_seeded_builder,
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
                optimizer_backend="scipy-jax",
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

    def test_resolve_target_lane_accepted_step_callback_uses_observe_only_callback_for_per_accept(
        self,
    ):
        module = self.load_module()
        callback = object()
        observe_accepted_step = object()
        adapter = types.SimpleNamespace(
            callback=callback,
            observe_accepted_step=observe_accepted_step,
        )

        resolved_callback = module.resolve_target_lane_accepted_step_callback(
            adapter,
            use_target_lane=True,
            sync_policy="per-accept",
        )

        self.assertIs(resolved_callback, observe_accepted_step)

    def test_configure_target_lane_accepted_step_sync_installs_array_native_sync(
        self,
    ):
        module = self.load_module()
        adapter = types.SimpleNamespace(
            accepted_step_state_sync=None,
            reevaluate_before_accept=True,
        )
        configured_sync = object()

        with patch.object(
            module,
            "build_single_stage_target_lane_accepted_step_sync",
            return_value=configured_sync,
        ) as build_sync:
            module.configure_single_stage_target_lane_accepted_step_sync(
                adapter,
                "booz",
                "bs",
                0.2,
                use_target_lane=True,
                outer_objective_config={"kind": "outer"},
                success_filter="filter",
            )

        self.assertIs(adapter.accepted_step_state_sync, configured_sync)
        self.assertFalse(adapter.reevaluate_before_accept)
        build_sync.assert_called_once_with(
            "booz",
            "bs",
            0.2,
            outer_objective_config={"kind": "outer"},
            success_filter="filter",
        )

    def test_configure_target_lane_accepted_step_sync_skips_non_target_lane(self):
        module = self.load_module()
        adapter = types.SimpleNamespace(
            accepted_step_state_sync=None,
            reevaluate_before_accept=True,
        )

        with patch.object(
            module,
            "build_single_stage_target_lane_accepted_step_sync",
            side_effect=AssertionError("non-target lane should not build sync"),
        ):
            module.configure_single_stage_target_lane_accepted_step_sync(
                adapter,
                "booz",
                "bs",
                0.2,
                use_target_lane=False,
                outer_objective_config={"kind": "outer"},
                success_filter="filter",
            )

        self.assertIsNone(adapter.accepted_step_state_sync)
        self.assertTrue(adapter.reevaluate_before_accept)

    def test_resolve_target_lane_post_run_state_sync_uses_explicit_state_sync_when_callback_active(
        self,
    ):
        module = self.load_module()
        recorded = []
        adapter = types.SimpleNamespace(
            sync_accepted_step=lambda x: (_ for _ in ()).throw(
                AssertionError("callback-active sync should use explicit state sync")
            ),
            sync_accepted_step_state=lambda x: recorded.append(x),
        )

        sync = module.resolve_target_lane_post_run_state_sync(
            adapter,
            use_target_lane=True,
            accepted_step_callback=object(),
        )

        self.assertTrue(sync.simsopt_skip_failed_attempt_sync)
        sync("accepted-state")
        self.assertEqual(recorded, ["accepted-state"])

    def test_resolve_target_lane_post_run_state_sync_maps_scaled_phase_state(self):
        module = self.load_module()
        synced_states = []
        adapter = types.SimpleNamespace(
            sync_accepted_step=lambda x: synced_states.append(
                np.asarray(x, dtype=float)
            )
        )

        sync = module.resolve_target_lane_post_run_state_sync(
            adapter,
            use_target_lane=True,
            accepted_step_callback=None,
            scaled_phase_step_scale=0.5,
        )
        state = module.ScaledOuterPhaseOptimizerState(
            step_dofs=np.array([2.0, 4.0]),
            anchor_dofs=np.array([10.0, 20.0]),
        )

        sync(state)

        self.assertEqual(len(synced_states), 1)
        np.testing.assert_allclose(synced_states[0], np.array([11.0, 22.0]))

    def test_should_force_strict_target_lane_final_sync(self):
        module = self.load_module()

        self.assertFalse(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=False,
                res_nit=3,
                optimizer_status=0,
                accepted_step_callback=None,
                trial_boozer_override_active=True,
            )
        )
        self.assertFalse(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=True,
                res_nit=0,
                optimizer_status=0,
                accepted_step_callback=None,
                trial_boozer_override_active=True,
            )
        )
        self.assertTrue(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=True,
                res_nit=2,
                optimizer_status=5,
                accepted_step_callback=None,
                trial_boozer_override_active=False,
            )
        )
        self.assertFalse(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=True,
                res_nit=2,
                optimizer_status=6,
                accepted_step_callback=None,
                trial_boozer_override_active=False,
            )
        )
        self.assertTrue(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=True,
                res_nit=2,
                optimizer_status=1,
                accepted_step_callback=object(),
                trial_boozer_override_active=False,
            )
        )
        self.assertTrue(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=True,
                res_nit=2,
                optimizer_status=0,
                accepted_step_callback=None,
                trial_boozer_override_active=False,
            )
        )
        self.assertTrue(
            module.should_force_strict_target_lane_final_sync(
                use_target_lane=True,
                res_nit=2,
                optimizer_status=None,
                accepted_step_callback=object(),
                trial_boozer_override_active=True,
            )
        )

    def test_target_lane_result_syncability_helpers(self):
        module = self.load_module()

        self.assertTrue(module.target_lane_result_status_allows_state_sync(None))
        self.assertTrue(module.target_lane_result_status_allows_state_sync(0))
        self.assertTrue(module.target_lane_result_status_allows_state_sync(1))
        self.assertTrue(module.target_lane_result_status_allows_state_sync(5))
        self.assertFalse(module.target_lane_result_status_allows_state_sync(6))

        self.assertFalse(
            module.target_lane_result_has_syncable_state(
                types.SimpleNamespace(nit=0, status=0)
            )
        )
        self.assertTrue(
            module.target_lane_result_has_syncable_state(
                types.SimpleNamespace(nit=1, status=1)
            )
        )
        self.assertTrue(
            module.target_lane_result_has_syncable_state(
                types.SimpleNamespace(nit=1, status=5)
            )
        )
        self.assertFalse(
            module.target_lane_result_has_syncable_state(
                types.SimpleNamespace(nit=1, status=6)
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
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad,
            callback,
            progress_callback=None,
            failure_callback=None,
        ):
            del progress_callback, failure_callback
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
            progress_callback=None,
            failure_callback=None,
        ):
            del (
                fun,
                x0,
                method,
                tol,
                maxiter,
                options,
                value_and_grad,
                callback,
                progress_callback,
            )
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

    def test_run_single_stage_optimizer_scipy_jax_omits_private_diagnostics(self):
        module = self.load_module()
        captured = {}
        explicit_fun = lambda x: (
            jnp.asarray(jnp.dot(x, x), dtype=jnp.float64),
            jnp.asarray(2.0 * x, dtype=jnp.float64),
        )

        def fake_require_target_backend_x64(optimizer_backend):
            captured["x64_backend"] = optimizer_backend

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
            progress_callback=None,
            failure_callback=None,
            initial_value_and_grad=None,
        ):
            del (
                fun,
                x0,
                tol,
                maxiter,
                options,
                value_and_grad,
                callback,
                progress_callback,
            )
            captured["method"] = method
            captured["failure_callback"] = failure_callback
            captured["initial_value_and_grad"] = initial_value_and_grad
            return types.SimpleNamespace(x=np.zeros(2), nit=0, message="ok")

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=fake_require_target_backend_x64,
            jax_minimize=fake_jax_minimize,
        ):
            failure_callback = object()
            optimizer_seed = object()
            contract = module.resolve_single_stage_optimizer_contract(
                "jax", "scipy-jax"
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
                failure_callback=failure_callback,
                optimizer_initial_value_and_grad=optimizer_seed,
            )

        self.assertEqual(captured["x64_backend"], "scipy-jax")
        self.assertEqual(captured["method"], "lbfgs-scipy-jax")
        self.assertIsNone(captured["failure_callback"])
        self.assertIsNone(captured["initial_value_and_grad"])
        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_fullgraph_uses_full_optimizer_vector(self):
        module = self.load_module()
        captured = {}
        explicit_fun = lambda x: (
            jnp.asarray(jnp.dot(x, x), dtype=jnp.float64),
            jnp.asarray(2.0 * x, dtype=jnp.float64),
        )

        def fake_require_target_backend_x64(optimizer_backend):
            captured["x64_backend"] = optimizer_backend

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
            progress_callback=None,
            failure_callback=None,
            initial_value_and_grad=None,
        ):
            del fun, tol, maxiter, options, value_and_grad, callback
            del progress_callback, failure_callback, initial_value_and_grad
            captured["method"] = method
            captured["x0"] = np.asarray(x0)
            return types.SimpleNamespace(x=np.asarray(x0), nit=0, message="ok")

        dofs = np.arange(6, dtype=np.float64)
        with self.patch_optimizer_jax_module(
            require_target_backend_x64=fake_require_target_backend_x64,
            jax_minimize=fake_jax_minimize,
        ):
            contract = module.resolve_single_stage_optimizer_contract(
                "jax",
                "scipy-jax-fullgraph",
            )
            result = module.run_single_stage_optimizer(
                explicit_fun,
                dofs,
                contract=contract,
                maxiter=1,
                ftol=0.0,
                gtol=1e-6,
                maxcor=5,
                outer_maxls=6,
                callback=None,
                scalar_fun=None,
            )

        self.assertEqual(captured["x64_backend"], "scipy-jax-fullgraph")
        self.assertEqual(captured["method"], "lbfgs-scipy-jax-fullgraph")
        np.testing.assert_array_equal(captured["x0"], dofs)
        np.testing.assert_array_equal(result.x, dofs)

    def test_run_single_stage_optimizer_threads_target_lane_progress_callback(self):
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
            progress_callback=None,
            failure_callback=None,
        ):
            del (
                fun,
                x0,
                method,
                tol,
                maxiter,
                options,
                value_and_grad,
                callback,
                failure_callback,
            )
            captured["progress_callback"] = progress_callback
            return types.SimpleNamespace(x=np.zeros(2), nit=0, message="ok")

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=fake_require_target_backend_x64,
            jax_minimize=fake_jax_minimize,
        ):
            progress_callback = object()
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
                progress_callback=progress_callback,
                scalar_fun=None,
            )

        self.assertIs(captured["progress_callback"], progress_callback)
        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_rejects_failure_callback_on_reference_lane(
        self,
    ):
        module = self.load_module()

        def fake_reference_minimize(*args, **kwargs):
            raise AssertionError("reference_minimize should not run")

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=lambda _optimizer_backend: None,
            jax_minimize=fake_reference_minimize,
        ):
            contract = module.resolve_single_stage_optimizer_contract("cpu", "scipy")
            with self.assertRaisesRegex(
                ValueError,
                "only supports failure_callback for method='lbfgs-trace'",
            ):
                module.run_single_stage_optimizer(
                    lambda x: (
                        jnp.asarray(jnp.dot(x, x), dtype=jnp.float64),
                        jnp.asarray(2.0 * x, dtype=jnp.float64),
                    ),
                    np.array([0.0, 0.0]),
                    contract=contract,
                    maxiter=1,
                    ftol=0.0,
                    gtol=1e-6,
                    maxcor=5,
                    outer_maxls=6,
                    callback=None,
                    scalar_fun=None,
                    failure_callback=lambda *args: None,
                )

    def test_run_single_stage_optimizer_threads_reference_trace_contract(self):
        module = self.load_module()
        captured = {}
        failure_callback = object()
        progress_callback = object()
        initial_value_and_grad = (1.0, np.array([1.0, -1.0], dtype=np.float64))

        def fake_reference_minimize(
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad,
            callback,
            progress_callback=None,
            failure_callback=None,
            initial_value_and_grad=None,
        ):
            captured.update(
                {
                    "fun": fun,
                    "x0": np.asarray(x0),
                    "method": method,
                    "tol": tol,
                    "maxiter": maxiter,
                    "options": dict(options),
                    "value_and_grad": value_and_grad,
                    "callback": callback,
                    "progress_callback": progress_callback,
                    "failure_callback": failure_callback,
                    "initial_value_and_grad": initial_value_and_grad,
                }
            )
            return types.SimpleNamespace(x=np.zeros(2), nit=0, message="ok")

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=lambda _optimizer_backend: None,
            jax_minimize=fake_reference_minimize,
        ):
            contract = module.resolve_single_stage_optimizer_contract(
                "cpu",
                "scipy",
                "lbfgs-trace",
            )
            result = module.run_single_stage_optimizer(
                lambda x: (
                    jnp.asarray(jnp.dot(x, x), dtype=jnp.float64),
                    jnp.asarray(2.0 * x, dtype=jnp.float64),
                ),
                np.array([0.0, 0.0]),
                contract=contract,
                maxiter=7,
                ftol=0.0,
                gtol=1e-6,
                maxcor=5,
                outer_maxls=6,
                callback=None,
                progress_callback=progress_callback,
                scalar_fun=None,
                failure_callback=failure_callback,
                optimizer_initial_value_and_grad=initial_value_and_grad,
            )

        self.assertEqual(captured["method"], "lbfgs-trace")
        self.assertEqual(captured["tol"], 1e-6)
        self.assertEqual(captured["maxiter"], 7)
        self.assertEqual(captured["options"], {"maxcor": 5, "ftol": 0.0, "maxls": 6})
        self.assertTrue(captured["value_and_grad"])
        self.assertIs(captured["progress_callback"], progress_callback)
        self.assertIs(captured["failure_callback"], failure_callback)
        self.assertIs(captured["initial_value_and_grad"], initial_value_and_grad)
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
            progress_callback=None,
            failure_callback=None,
        ):
            del (
                fun,
                x0,
                method,
                tol,
                maxiter,
                value_and_grad,
                callback,
                progress_callback,
                failure_callback,
            )
            captured["options"] = dict(options)
            return types.SimpleNamespace(x=np.zeros(2), nit=0, message="ok")

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
                target_lane_initial_step_size=1.0e-4,
            )

        self.assertEqual(captured["options"]["initial_step_size"], 1.0e-4)
        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_threads_target_lane_initial_value_and_grad(
        self,
    ):
        module = self.load_module()
        captured = {}
        explicit_fun = lambda x: (
            jnp.asarray(jnp.dot(x, x), dtype=jnp.float64),
            jnp.asarray(2.0 * x, dtype=jnp.float64),
        )
        optimizer_seed = (
            jnp.asarray(4.0, dtype=jnp.float64),
            jnp.asarray([1.0, -2.0], dtype=jnp.float64),
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
            progress_callback=None,
            failure_callback=None,
            initial_value_and_grad=None,
        ):
            del (
                fun,
                x0,
                method,
                tol,
                maxiter,
                options,
                value_and_grad,
                callback,
                progress_callback,
                failure_callback,
            )
            captured["initial_value_and_grad"] = initial_value_and_grad
            return types.SimpleNamespace(x=np.zeros(2), nit=0, message="ok")

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
                optimizer_initial_value_and_grad=optimizer_seed,
            )

        self.assertIs(captured["initial_value_and_grad"], optimizer_seed)
        self.assertEqual(result.message, "ok")

    def _build_target_lane_retry_result(
        self,
        *,
        x,
        nit,
        success,
        message,
        status,
        step_scale=None,
        line_search_failed=True,
        nonfinite_step=False,
        stalled_step=False,
        valid_curvature=True,
        trial_converged=False,
        ls_status=0,
        nfev=None,
        njev=None,
    ):
        result_x = (
            x
            if hasattr(x, "step_dofs") and hasattr(x, "anchor_dofs")
            else np.asarray(x, dtype=float)
        )
        result = types.SimpleNamespace(
            x=result_x,
            nit=nit,
            success=success,
            message=message,
            status=status,
            invalid_step_log=[],
        )
        if step_scale is not None:
            result.invalid_step_log = [
                {
                    "iteration": 1,
                    "step_scale": float(step_scale),
                    "line_search_failed": bool(line_search_failed),
                    "nonfinite_step": bool(nonfinite_step),
                    "stalled_step": bool(stalled_step),
                    "valid_curvature": bool(valid_curvature),
                    "trial_converged": bool(trial_converged),
                    "ls_status": int(ls_status),
                    "requested_initial_step": float(step_scale),
                    "first_tested_alpha": float(step_scale),
                    "best_finite_alpha": float(step_scale),
                    "returned_alpha": float(step_scale),
                    "failure_reason": "test-invalid-step",
                    "armijo_margin": 0.0,
                    "curvature_margin": 0.0,
                }
            ]
        if nfev is not None:
            result.nfev = int(nfev)
        if njev is not None:
            result.njev = int(njev)
        return result

    def _build_best_syncable_retry_results(self):
        result_specs = [
            (np.array([9.0, 9.0]), 2, 4, 4, 0.2, 0.9),
            (np.array([4.0, 5.0]), 3, 6, 6, 0.1, 0.4),
            (np.array([1.0, 2.0]), 0, 5, 5, 0.05, 2.0),
        ]
        results = []
        for x, nit, nfev, njev, step_scale, metric in result_specs:
            result = self._build_target_lane_retry_result(
                x=x,
                nit=nit,
                nfev=nfev,
                njev=njev,
                success=False,
                message="failed",
                status=5,
                step_scale=step_scale,
            )
            result.fun = metric
            result.jac = np.full(2, metric)
            results.append(result)
        return results

    @staticmethod
    def _retry_policy(module):
        return module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )

    def test_run_single_stage_target_lane_optimizer_with_retries_retries_from_anchor(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["latest_local_incumbent"] = {
            "coil_dofs": np.array([1.0, 2.0]),
            "sdofs": np.array([3.0, 4.0]),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "intersecting": False,
            "self_intersection_check_available": True,
            "hardware_constraint_status": {"success": True, "violations": []},
        }
        run_dict["latest_local_stage"] = "initial"
        run_dict["best_local_incumbent"] = copy.deepcopy(
            run_dict["latest_local_incumbent"]
        )
        run_dict["best_local_stage"] = "initial"
        invalid_state_events = []
        optimizer_calls = []
        progress_events = []

        def fake_run_single_stage_optimizer(
            fun,
            dofs,
            *,
            callback,
            contract,
            maxiter,
            ftol,
            gtol,
            maxcor,
            outer_maxls,
            scalar_fun,
            progress_callback=None,
            target_lane_initial_step_size,
            failure_callback,
        ):
            del (
                fun,
                contract,
                maxiter,
                ftol,
                gtol,
                maxcor,
                outer_maxls,
                scalar_fun,
                progress_callback,
            )
            optimizer_calls.append(
                {
                    "dofs": np.asarray(dofs, dtype=float).copy(),
                    "callback": callback,
                    "initial_step_size": target_lane_initial_step_size,
                }
            )
            if len(optimizer_calls) == 1:
                return self._build_target_lane_retry_result(
                    x=np.array([9.0, 9.0]),
                    nit=0,
                    success=False,
                    message="failed",
                    status=5,
                    step_scale=0.2,
                )
            return self._build_target_lane_retry_result(
                x=np.array([1.0, 2.0]),
                nit=1,
                success=True,
                message="ok",
                status=0,
            )

        retry_callback = object()
        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=fake_run_single_stage_optimizer,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0]),
                    phase="phase2",
                    callback=None,
                    retry_callback=retry_callback,
                    result_state_sync=None,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                    progress_event_callback=(
                        lambda label, **extra: progress_events.append(
                            (label, extra)
                        )
                    ),
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(retry_summary["attempt_count"], 1)
        self.assertEqual(len(optimizer_calls), 2)
        np.testing.assert_allclose(optimizer_calls[1]["dofs"], np.array([1.0, 2.0]))
        self.assertIs(optimizer_calls[1]["callback"], retry_callback)
        self.assertEqual(optimizer_calls[1]["initial_step_size"], 0.1)
        self.assertEqual(
            [label for label, _ in progress_events],
            [
                "phase2_attempt_0_started",
                "phase2_attempt_0_returned",
                "phase2_retry_1_started",
                "phase2_retry_1_returned",
            ],
        )
        self.assertEqual(progress_events[0][1]["phase"], "phase2")
        self.assertEqual(progress_events[0][1]["attempt_index"], 0)
        self.assertEqual(progress_events[2][1]["anchor_stage"], "initial")
        self.assertEqual(
            progress_events[3][1]["result"]["message"],
            "ok",
        )

    def test_run_single_stage_target_lane_optimizer_with_retries_uses_seed_only_on_first_attempt(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        optimizer_seed = ("seed-value", "seed-grad")
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["latest_local_incumbent"] = {
            "coil_dofs": np.array([1.0, 2.0]),
            "sdofs": np.array([3.0, 4.0]),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "intersecting": False,
            "self_intersection_check_available": True,
            "hardware_constraint_status": {"success": True, "violations": []},
        }
        run_dict["latest_local_stage"] = "initial"
        run_dict["best_local_incumbent"] = copy.deepcopy(
            run_dict["latest_local_incumbent"]
        )
        run_dict["best_local_stage"] = "initial"
        invalid_state_events = []
        observed_seeds = []

        def fake_run_single_stage_optimizer(
            fun,
            dofs,
            *,
            callback,
            contract,
            maxiter,
            ftol,
            gtol,
            maxcor,
            outer_maxls,
            scalar_fun,
            progress_callback=None,
            target_lane_initial_step_size,
            failure_callback,
            optimizer_initial_value_and_grad=None,
        ):
            del (
                fun,
                dofs,
                callback,
                contract,
                maxiter,
                ftol,
                gtol,
                maxcor,
                outer_maxls,
                scalar_fun,
                progress_callback,
                target_lane_initial_step_size,
                failure_callback,
            )
            observed_seeds.append(optimizer_initial_value_and_grad)
            if len(observed_seeds) == 1:
                return self._build_target_lane_retry_result(
                    x=np.array([9.0, 9.0]),
                    nit=0,
                    success=False,
                    message="failed",
                    status=5,
                    step_scale=0.2,
                )
            return self._build_target_lane_retry_result(
                x=np.array([1.0, 2.0]),
                nit=1,
                success=True,
                message="ok",
                status=0,
            )

        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=fake_run_single_stage_optimizer,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0]),
                    phase="phase2",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=None,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    optimizer_initial_value_and_grad=optimizer_seed,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(retry_summary["attempt_count"], 1)
        self.assertEqual(observed_seeds, [optimizer_seed, None])

    def test_run_single_stage_target_lane_optimizer_with_retries_uses_explicit_post_run_sync_for_retry_anchor(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        anchor_state = {
            "coil_dofs": np.array([1.0, 2.0]),
            "sdofs": np.array([3.0, 4.0]),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.zeros(5),
            "intersecting": False,
            "self_intersection_check_available": True,
            "hardware_constraint_status": {"success": True, "violations": []},
        }
        run_dict["latest_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["latest_local_stage"] = "initial"
        run_dict["best_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["best_local_stage"] = "initial"
        invalid_state_events = []
        optimizer_calls = []
        synced_states = []

        def fake_run_single_stage_optimizer(
            fun,
            dofs,
            *,
            callback,
            contract,
            maxiter,
            ftol,
            gtol,
            maxcor,
            outer_maxls,
            scalar_fun,
            progress_callback=None,
            target_lane_initial_step_size,
            failure_callback,
        ):
            del (
                fun,
                contract,
                maxiter,
                ftol,
                gtol,
                maxcor,
                outer_maxls,
                scalar_fun,
                progress_callback,
                target_lane_initial_step_size,
                failure_callback,
            )
            optimizer_calls.append(
                {
                    "dofs": np.asarray(dofs, dtype=float).copy(),
                    "callback": callback,
                }
            )
            if len(optimizer_calls) == 1:
                return self._build_target_lane_retry_result(
                    x=np.array([9.0, 9.0]),
                    nit=1,
                    success=False,
                    message="failed",
                    status=5,
                    step_scale=0.2,
                )
            return self._build_target_lane_retry_result(
                x=np.array([9.0, 9.0]),
                nit=1,
                success=True,
                message="ok",
                status=0,
            )

        def result_state_sync(result_x):
            synced_x = np.asarray(result_x, dtype=float).copy()
            synced_states.append(synced_x)
            synced_anchor_state = copy.deepcopy(anchor_state)
            synced_anchor_state["coil_dofs"] = synced_x
            synced_anchor_state["J"] = 0.5
            run_dict["latest_local_incumbent"] = copy.deepcopy(synced_anchor_state)
            run_dict["latest_local_stage"] = "synced"
            run_dict["best_local_incumbent"] = copy.deepcopy(synced_anchor_state)
            run_dict["best_local_stage"] = "synced"

        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=fake_run_single_stage_optimizer,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0]),
                    phase="phase2",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=result_state_sync,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(retry_summary["attempt_count"], 1)
        self.assertEqual(len(synced_states), 1)
        np.testing.assert_allclose(optimizer_calls[1]["dofs"], np.array([9.0, 9.0]))
        self.assertIsNone(optimizer_calls[1]["callback"])

    def test_run_single_stage_target_lane_optimizer_with_retries_restores_anchor_on_exhaustion(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        anchor_state = {
            "coil_dofs": np.array([4.0, 5.0]),
            "sdofs": np.array([6.0, 7.0]),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.zeros(5),
            "intersecting": False,
            "self_intersection_check_available": True,
            "hardware_constraint_status": {"success": True, "violations": []},
        }
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["latest_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["latest_local_stage"] = "latest"
        run_dict["best_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["best_local_stage"] = "best"
        invalid_state_events = [
            {
                "line_search_failed": True,
                "nonfinite_step": False,
                "stalled_step": False,
                "valid_curvature": True,
                "step_scale": {"value": 0.2},
            }
        ]

        def always_fail(*args, **kwargs):
            del args, kwargs
            return types.SimpleNamespace(
                x=np.array([9.0, 9.0]),
                nit=0,
                success=False,
                message="failed",
                status=5,
            )

        policy = module.SingleStageSearchPolicy(
            donor_class="serialized_surface_state",
            search_policy="preserve_first",
            adaptive_failure_penalty_weight=1.0,
            invalid_step_retry_budget=0,
            retry_step_shrink_factor=0.35,
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=always_fail,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0]),
                    phase="phase2",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=None,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                )
            )

        self.assertFalse(result.success)
        self.assertTrue(retry_summary["restored_preserved_local_state"])
        self.assertEqual(retry_summary["restored_preserved_local_stage"], "best")
        np.testing.assert_allclose(result.x, np.array([4.0, 5.0]))
        np.testing.assert_allclose(run_dict["x_prev"], np.array([4.0, 5.0]))

    def test_run_single_stage_target_lane_optimizer_with_retries_keeps_syncable_unsuccessful_result(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        anchor_state = {
            "coil_dofs": np.array([4.0, 5.0]),
            "sdofs": np.array([6.0, 7.0]),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.zeros(5),
            "intersecting": False,
            "self_intersection_check_available": True,
            "hardware_constraint_status": {"success": True, "violations": []},
        }
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["latest_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["latest_local_stage"] = "latest"
        run_dict["best_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["best_local_stage"] = "best"
        candidate_x = np.array([9.0, 8.0])

        def maxiter_result(*args, **kwargs):
            del args, kwargs
            return types.SimpleNamespace(
                x=candidate_x.copy(),
                nit=1,
                success=False,
                message="maxiter",
                status=1,
            )

        policy = module.SingleStageSearchPolicy(
            donor_class="serialized_surface_state",
            search_policy="preserve_first",
            adaptive_failure_penalty_weight=1.0,
            invalid_step_retry_budget=0,
            retry_step_shrink_factor=0.35,
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=maxiter_result,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0]),
                    phase="phase2",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=None,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=[],
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                )
            )

        self.assertFalse(result.success)
        self.assertFalse(retry_summary["restored_preserved_local_state"])
        self.assertIsNone(retry_summary["restored_preserved_local_stage"])
        np.testing.assert_allclose(result.x, candidate_x)
        np.testing.assert_allclose(run_dict["x_prev"], np.zeros(5))

    def test_run_single_stage_target_lane_optimizer_with_retries_skips_flagged_failed_attempt_sync(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        invalid_state_events = [
            {
                "line_search_failed": True,
                "nonfinite_step": False,
                "stalled_step": False,
                "valid_curvature": True,
                "step_scale": {"value": 0.2},
            }
        ]
        sync_calls = []

        def flagged_result_state_sync(x):
            sync_calls.append(np.asarray(x, dtype=float))

        flagged_result_state_sync.simsopt_skip_failed_attempt_sync = True

        def failed_result(*args, **kwargs):
            del args, kwargs
            return types.SimpleNamespace(
                x=np.array([9.0, 8.0]),
                nit=1,
                success=False,
                message="failed",
                status=5,
            )

        policy = module.SingleStageSearchPolicy(
            donor_class="serialized_surface_state",
            search_policy="preserve_first",
            adaptive_failure_penalty_weight=1.0,
            invalid_step_retry_budget=0,
            retry_step_shrink_factor=0.35,
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=failed_result,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0]),
                    phase="phase2",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=flagged_result_state_sync,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                )
            )

        self.assertFalse(result.success)
        self.assertEqual(sync_calls, [])
        self.assertFalse(retry_summary["restored_preserved_local_state"])

    def test_run_single_stage_target_lane_optimizer_with_retries_tracks_total_iterations(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        anchor_state = {
            "coil_dofs": np.array([1.0, 2.0]),
            "sdofs": np.array([3.0, 4.0]),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.zeros(5),
            "intersecting": False,
            "self_intersection_check_available": True,
            "hardware_constraint_status": {"success": True, "violations": []},
        }
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["latest_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["latest_local_stage"] = "latest"
        run_dict["best_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["best_local_stage"] = "best"
        invalid_state_events = []
        observed_maxiters = []

        def fake_run_single_stage_optimizer(
            fun,
            dofs,
            *,
            callback,
            contract,
            maxiter,
            ftol,
            gtol,
            maxcor,
            outer_maxls,
            scalar_fun,
            progress_callback=None,
            target_lane_initial_step_size,
            failure_callback,
        ):
            del (
                fun,
                dofs,
                callback,
                contract,
                ftol,
                gtol,
                maxcor,
                outer_maxls,
                scalar_fun,
                progress_callback,
                target_lane_initial_step_size,
                failure_callback,
            )
            observed_maxiters.append(maxiter)
            if len(observed_maxiters) == 1:
                return self._build_target_lane_retry_result(
                    x=np.array([9.0, 9.0]),
                    nit=2,
                    nfev=4,
                    njev=4,
                    success=False,
                    message="failed",
                    status=5,
                    step_scale=0.2,
                )
            return self._build_target_lane_retry_result(
                x=np.array([1.0, 2.0]),
                nit=3,
                nfev=6,
                njev=6,
                success=True,
                message="ok",
                status=0,
            )

        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=fake_run_single_stage_optimizer,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0]),
                    phase="phase2",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=None,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(retry_summary["attempt_count"], 1)
        self.assertEqual(observed_maxiters, [5, 3])
        self.assertEqual(result.nit, 5)
        self.assertEqual(result.nfev, 10)
        self.assertEqual(result.njev, 10)

    def test_run_single_stage_target_lane_optimizer_with_retries_preserves_best_syncable_retry(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["hardware_constraint_status"] = {"success": True, "violations": []}
        invalid_state_events = []
        sync_metrics = [0.9, 0.4]

        def sync_accepted_state(x):
            metric = sync_metrics.pop(0)
            run_dict["x_prev"] = np.asarray(x, dtype=float)
            run_dict["sdofs"] = np.asarray([3.0, 4.0], dtype=float)
            run_dict["iota"] = TEST_IOTA
            run_dict["G"] = TEST_G0
            run_dict["J"] = metric
            run_dict["dJ"] = np.full(5, metric)
            module.record_single_stage_local_incumbent(
                run_dict,
                stage=f"sync_{metric}",
            )

        results = self._build_best_syncable_retry_results()
        policy = self._retry_policy(module)

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=results,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0]),
                    phase="phase2",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=sync_accepted_state,
                    contract=contract,
                    maxiter=10,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                )
            )

        self.assertFalse(result.success)
        self.assertEqual(result.fun, 0.4)
        self.assertEqual(result.nit, 5)
        self.assertEqual(result.nfev, 15)
        self.assertEqual(result.njev, 15)
        self.assertTrue(retry_summary["restored_preserved_local_state"])
        self.assertEqual(
            retry_summary["restored_preserved_local_stage"],
            "sync_0.4",
        )
        np.testing.assert_allclose(result.x, np.array([4.0, 5.0]))
        np.testing.assert_allclose(run_dict["x_prev"], np.array([4.0, 5.0]))
        self.assertEqual(run_dict["J"], 0.4)

    def test_run_single_stage_target_lane_optimizer_with_retries_selects_best_result_without_state_anchor(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["latest_local_incumbent"] = (
            module.snapshot_single_stage_local_incumbent_state(run_dict)
        )
        run_dict["latest_local_stage"] = "initial"
        invalid_state_events = []

        results = self._build_best_syncable_retry_results()
        policy = self._retry_policy(module)

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=results,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0]),
                    phase="phase2",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=None,
                    contract=contract,
                    maxiter=10,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                )
            )

        self.assertFalse(result.success)
        self.assertEqual(result.fun, 0.4)
        self.assertEqual(result.nit, 5)
        self.assertFalse(retry_summary["restored_preserved_local_state"])
        np.testing.assert_allclose(result.x, np.array([4.0, 5.0]))
        np.testing.assert_allclose(run_dict["x_prev"], np.zeros(5))

    def test_run_single_stage_target_lane_optimizer_with_retries_rebuilds_events_from_result_without_callback(
        self,
    ):
        """Retry must rebuild invalid_state_events from result.invalid_step_log
        alone when failure_callback is None, so strict-transfer-guard runs that
        cannot emit host callbacks still trigger anchor-restored retries.
        """

        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        anchor_state = {
            "coil_dofs": np.array([1.0, 2.0]),
            "sdofs": np.array([3.0, 4.0]),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.zeros(5),
            "intersecting": False,
            "self_intersection_check_available": True,
            "hardware_constraint_status": {"success": True, "violations": []},
        }
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["latest_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["latest_local_stage"] = "initial"
        run_dict["best_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["best_local_stage"] = "initial"
        invalid_state_events = []
        observed_failure_callbacks = []

        def fake_run_single_stage_optimizer(
            fun,
            dofs,
            *,
            callback,
            contract,
            maxiter,
            ftol,
            gtol,
            maxcor,
            outer_maxls,
            scalar_fun,
            progress_callback=None,
            target_lane_initial_step_size,
            failure_callback,
        ):
            del (
                fun,
                dofs,
                callback,
                contract,
                maxiter,
                ftol,
                gtol,
                maxcor,
                outer_maxls,
                scalar_fun,
                progress_callback,
                target_lane_initial_step_size,
            )
            observed_failure_callbacks.append(failure_callback)
            if len(observed_failure_callbacks) == 1:
                return self._build_target_lane_retry_result(
                    x=np.array([9.0, 9.0]),
                    nit=1,
                    success=False,
                    message="failed",
                    status=5,
                    step_scale=0.2,
                    nonfinite_step=True,
                )
            return self._build_target_lane_retry_result(
                x=np.array([1.0, 2.0]),
                nit=1,
                success=True,
                message="ok",
                status=0,
            )

        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=fake_run_single_stage_optimizer,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    np.array([0.0, 0.0]),
                    phase="phase1",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=None,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(retry_summary["attempt_count"], 1)
        self.assertEqual(
            observed_failure_callbacks, [None] * len(observed_failure_callbacks)
        )
        self.assertEqual(len(invalid_state_events), 1)
        rebuilt_event = invalid_state_events[0]
        self.assertEqual(rebuilt_event["phase"], "phase1")
        self.assertEqual(rebuilt_event["iteration"], 1)
        self.assertTrue(rebuilt_event["nonfinite_step"])
        self.assertTrue(rebuilt_event["line_search_failed"])
        self.assertEqual(rebuilt_event["step_scale"]["value"], 0.2)
        self.assertTrue(
            module.single_stage_retry_triggered_by_invalid_state(invalid_state_events)
        )

    def test_run_single_stage_target_lane_optimizer_with_retries_supports_scaled_phase_retry_state(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["latest_local_incumbent"] = {
            "coil_dofs": np.array([1.0, 2.0]),
            "sdofs": np.array([3.0, 4.0]),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            "intersecting": False,
            "self_intersection_check_available": True,
            "hardware_constraint_status": {"success": True, "violations": []},
        }
        run_dict["latest_local_stage"] = "initial"
        run_dict["best_local_incumbent"] = copy.deepcopy(
            run_dict["latest_local_incumbent"]
        )
        run_dict["best_local_stage"] = "initial"
        invalid_state_events = []
        optimizer_calls = []

        def fake_run_single_stage_optimizer(
            fun,
            dofs,
            *,
            callback,
            contract,
            maxiter,
            ftol,
            gtol,
            maxcor,
            outer_maxls,
            scalar_fun,
            progress_callback=None,
            target_lane_initial_step_size,
            failure_callback,
        ):
            del (
                fun,
                contract,
                maxiter,
                ftol,
                gtol,
                maxcor,
                outer_maxls,
                scalar_fun,
                progress_callback,
            )
            optimizer_calls.append(
                {
                    "dofs": dofs,
                    "callback": callback,
                    "initial_step_size": target_lane_initial_step_size,
                }
            )
            if len(optimizer_calls) == 1:
                return self._build_target_lane_retry_result(
                    x=dofs,
                    nit=0,
                    success=False,
                    message="failed",
                    status=5,
                    step_scale=0.2,
                )
            return self._build_target_lane_retry_result(
                x=dofs,
                nit=1,
                success=True,
                message="ok",
                status=0,
            )

        retry_callback = object()
        policy = module.SingleStageSearchPolicy(
            donor_class="stage2_seed_only",
            search_policy="repair_first",
            adaptive_failure_penalty_weight=1.5,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )
        initial_state = module.build_single_stage_scaled_phase_retry_state(
            np.array([9.0, 8.0]),
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=fake_run_single_stage_optimizer,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    initial_state,
                    phase="phase1",
                    callback=None,
                    retry_callback=retry_callback,
                    result_state_sync=None,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                    retry_dofs_factory=lambda anchor_state: (
                        module.build_single_stage_scaled_phase_retry_state(
                            anchor_state["coil_dofs"]
                        )
                    ),
                    restored_result_x_factory=lambda anchor_state: (
                        module.build_single_stage_scaled_phase_retry_state(
                            anchor_state["coil_dofs"]
                        )
                    ),
                )
            )

        self.assertTrue(result.success)
        self.assertEqual(retry_summary["attempt_count"], 1)
        self.assertEqual(len(optimizer_calls), 2)
        self.assertIs(optimizer_calls[1]["callback"], retry_callback)
        self.assertEqual(optimizer_calls[1]["initial_step_size"], 0.1)
        self.assertIsInstance(
            optimizer_calls[1]["dofs"],
            module.ScaledOuterPhaseOptimizerState,
        )
        np.testing.assert_allclose(
            optimizer_calls[1]["dofs"].anchor_dofs,
            np.array([1.0, 2.0]),
        )
        np.testing.assert_allclose(
            optimizer_calls[1]["dofs"].step_dofs,
            np.zeros(2),
        )

    def test_run_single_stage_target_lane_optimizer_with_retries_restores_scaled_phase_anchor_on_exhaustion(
        self,
    ):
        module = self.load_module()
        contract = module.resolve_single_stage_optimizer_contract("jax", "ondevice")
        anchor_state = {
            "coil_dofs": np.array([4.0, 5.0]),
            "sdofs": np.array([6.0, 7.0]),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.zeros(5),
            "intersecting": False,
            "self_intersection_check_available": True,
            "hardware_constraint_status": {"success": True, "violations": []},
        }
        run_dict = self._make_candidate_run_dict([1.0, 2.0])
        run_dict["latest_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["latest_local_stage"] = "latest"
        run_dict["best_local_incumbent"] = copy.deepcopy(anchor_state)
        run_dict["best_local_stage"] = "best"
        invalid_state_events = [
            {
                "line_search_failed": True,
                "nonfinite_step": False,
                "stalled_step": False,
                "valid_curvature": True,
                "step_scale": {"value": 0.2},
            }
        ]

        def always_fail(*args, **kwargs):
            del args, kwargs
            return types.SimpleNamespace(
                x=np.array([9.0, 9.0]),
                nit=0,
                success=False,
                message="failed",
                status=5,
            )

        policy = module.SingleStageSearchPolicy(
            donor_class="serialized_surface_state",
            search_policy="preserve_first",
            adaptive_failure_penalty_weight=1.0,
            invalid_step_retry_budget=0,
            retry_step_shrink_factor=0.35,
        )

        with patch.object(
            module,
            "run_single_stage_optimizer",
            side_effect=always_fail,
        ):
            result, retry_summary = (
                module.run_single_stage_target_lane_optimizer_with_retries(
                    lambda x: x,
                    module.build_single_stage_scaled_phase_retry_state(
                        np.array([8.0, 9.0]),
                    ),
                    phase="phase1",
                    callback=None,
                    retry_callback=None,
                    result_state_sync=None,
                    contract=contract,
                    maxiter=5,
                    ftol=0.0,
                    gtol=1.0e-6,
                    maxcor=5,
                    outer_maxls=6,
                    scalar_fun=None,
                    target_lane_initial_step_size=None,
                    failure_callback=None,
                    invalid_state_events=invalid_state_events,
                    run_dict=run_dict,
                    single_stage_search_policy=policy,
                    retry_dofs_factory=lambda candidate_anchor_state: (
                        module.build_single_stage_scaled_phase_retry_state(
                            candidate_anchor_state["coil_dofs"]
                        )
                    ),
                    restored_result_x_factory=lambda candidate_anchor_state: (
                        module.build_single_stage_scaled_phase_retry_state(
                            candidate_anchor_state["coil_dofs"]
                        )
                    ),
                )
            )

        self.assertFalse(result.success)
        self.assertTrue(retry_summary["restored_preserved_local_state"])
        self.assertEqual(retry_summary["restored_preserved_local_stage"], "best")
        self.assertIsInstance(result.x, module.ScaledOuterPhaseOptimizerState)
        np.testing.assert_allclose(result.x.anchor_dofs, np.array([4.0, 5.0]))
        np.testing.assert_allclose(result.x.step_dofs, np.zeros(2))

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
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad,
            callback,
            progress_callback=None,
            failure_callback=None,
        ):
            del progress_callback, failure_callback
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
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad,
            callback,
            progress_callback=None,
            failure_callback=None,
        ):
            del method, tol, maxiter, options, callback, progress_callback, failure_callback
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
            fun,
            x0,
            *,
            method,
            tol,
            maxiter,
            options,
            value_and_grad,
            callback,
            progress_callback=None,
            failure_callback=None,
        ):
            del (
                fun,
                x0,
                method,
                tol,
                maxiter,
                options,
                value_and_grad,
                callback,
                progress_callback,
                failure_callback,
            )
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
        ), patch.object(
            module,
            "accept_step",
            side_effect=self._make_fake_accept_step_capture(captured),
        ):
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
        self.assertEqual(captured["accept"]["objective_value"], 1.0)
        np.testing.assert_allclose(captured["accept"]["objective_grad"], np.zeros(2))
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
        ), patch.object(
            module,
            "accept_step",
            side_effect=self._make_fake_accept_step_capture(captured),
        ):
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
        self.assertEqual(captured["accept"]["objective_value"], 1.0)
        np.testing.assert_allclose(captured["accept"]["objective_grad"], np.zeros(2))
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
            accepted_step_state_sync=self._make_fake_target_lane_state_sync(
                captured,
                sdofs=np.array([9.0, -2.0]),
                iota=0.21,
                G=1.8,
            ),
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
        self.assertTrue(captured["sync"]["update_run_state"])
        log_summary.assert_called_once()
        np.testing.assert_allclose(run_dict["x_prev"], np.array([3.0, -4.0]))
        self.assertEqual(captured["setter"], [])
        np.testing.assert_allclose(jf.x, np.array([1.0, 2.0]))

    def test_single_stage_adapter_observe_accepted_step_skips_state_commit(self):
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
        run_dict = {"x_prev": np.array([8.0, -3.0]), "lscount": 4, "it": 3}
        captured = {}

        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface="booz",
            JF=jf,
            bs="bs",
            objectives={"qs": "obj"},
            diagnostics={"iota": "diag"},
            log_path="/tmp/log.txt",
            reevaluate_before_accept=True,
            accepted_step_state_sync=self._make_fake_target_lane_state_sync(
                captured,
                sdofs=np.array([9.0, -2.0]),
            ),
        )

        with patch.object(
            module,
            "log_single_stage_target_lane_accepted_step",
            return_value=None,
        ) as log_summary:
            adapter.observe_accepted_step(np.array([3.0, -4.0]))

        self.assertIs(captured["sync"]["state"], run_dict)
        np.testing.assert_allclose(captured["sync"]["x"], np.array([3.0, -4.0]))
        self.assertFalse(captured["sync"]["benchmark_mode"])
        self.assertFalse(captured["sync"]["update_run_state"])
        log_summary.assert_called_once()
        np.testing.assert_allclose(run_dict["x_prev"], np.array([8.0, -3.0]))
        self.assertNotIn("sdofs", run_dict)
        np.testing.assert_allclose(jf.x, np.array([1.0, 2.0]))

    def test_single_stage_adapter_sync_accepted_step_state_commits_without_logging(
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
        captured = {}

        adapter = module.SingleStageAdapter(
            run_dict=run_dict,
            boozer_surface="booz",
            JF=jf,
            bs="bs",
            objectives={"qs": "obj"},
            diagnostics={"iota": "diag"},
            log_path="/tmp/log.txt",
            reevaluate_before_accept=True,
            accepted_step_state_sync=self._make_fake_target_lane_state_sync(
                captured,
                sdofs=np.array([9.0, -2.0]),
            ),
        )

        with patch.object(
            module,
            "log_single_stage_target_lane_accepted_step",
            return_value=None,
        ) as log_summary:
            adapter.sync_accepted_step_state(np.array([3.0, -4.0]))

        self.assertIs(captured["sync"]["state"], run_dict)
        np.testing.assert_allclose(captured["sync"]["x"], np.array([3.0, -4.0]))
        self.assertFalse(captured["sync"]["benchmark_mode"])
        self.assertTrue(captured["sync"]["update_run_state"])
        log_summary.assert_not_called()
        np.testing.assert_allclose(run_dict["x_prev"], np.array([3.0, -4.0]))
        np.testing.assert_allclose(run_dict["sdofs"], np.array([9.0, -2.0]))
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
        run_dict = {"x_prev": np.zeros(2), "lscount": 4, "it": 0}
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
        run_dict = {"x_prev": np.zeros(2), "lscount": 2, "it": 0}
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
            "it": 0,
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

    def test_evaluate_surface_self_intersection_jax_requires_supported_surface(self):
        module = self.load_module()

        class SentinelSurface:
            def is_self_intersecting(self):
                raise AssertionError(
                    "JAX production self-intersection must not use host fallback"
                )

        with self.assertRaisesRegex(TypeError, "run seed conversion first"):
            module.evaluate_surface_self_intersection(
                SentinelSurface(),
                require_supported_surface=True,
            )

    def test_prewarm_supported_surface_self_intersection_requires_supported_surface(
        self,
    ):
        module = self.load_module()

        class SentinelSurface:
            pass

        with self.assertRaisesRegex(TypeError, "run seed conversion first"):
            module.prewarm_supported_surface_self_intersection(SentinelSurface())

    def test_surface_self_intersection_check_available_accepts_supported_surface_without_backend(
        self,
    ):
        module = self.load_module()
        surface = get_surface(
            "SurfaceRZFourier",
            True,
            full=True,
            nphi=200,
            ntheta=200,
            mpol=2,
            ntor=2,
        )

        with self.patch_surface_self_intersection_backend_unavailable(module):
            self.assertFalse(module.surface_self_intersection_check_available())
            self.assertTrue(module.surface_self_intersection_check_available(surface))

    def test_evaluate_surface_self_intersection_uses_supported_rzfourier_path_when_backend_unavailable(
        self,
    ):
        module = self.load_module()
        surface = get_surface(
            "SurfaceRZFourier",
            True,
            full=True,
            nphi=200,
            ntheta=200,
            mpol=2,
            ntor=2,
        )
        surface.x = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
            ],
            dtype=np.float64,
        )

        with self.patch_surface_self_intersection_backend_unavailable(module):
            self.assertEqual(
                module.evaluate_surface_self_intersection(surface),
                (True, True),
            )

    def test_evaluate_surface_self_intersection_reports_non_crossing_supported_rzfourier_without_backend(
        self,
    ):
        module = self.load_module()
        surface = get_surface(
            "SurfaceRZFourier",
            True,
            full=True,
            nphi=200,
            ntheta=200,
            mpol=2,
            ntor=2,
        )

        with self.patch_surface_self_intersection_backend_unavailable(module):
            self.assertEqual(
                module.evaluate_surface_self_intersection(surface),
                (False, True),
            )

    def test_evaluate_surface_self_intersection_uses_supported_xyztensor_path_when_backend_unavailable(
        self,
    ):
        module = self.load_module()
        crossing_surface = get_surface(
            "SurfaceRZFourier",
            True,
            full=True,
            nphi=200,
            ntheta=200,
            mpol=2,
            ntor=2,
            nfp=1,
        )
        crossing_surface.x = np.array(
            [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.1,
            ],
            dtype=np.float64,
        )
        surface = module.SurfaceXYZTensorFourier(
            mpol=2,
            ntor=2,
            nfp=1,
            stellsym=True,
            quadpoints_phi=crossing_surface.quadpoints_phi,
            quadpoints_theta=crossing_surface.quadpoints_theta,
        )
        surface.least_squares_fit(crossing_surface.gamma())

        with self.patch_surface_self_intersection_backend_unavailable(module):
            self.assertEqual(
                module.evaluate_surface_self_intersection(surface),
                (True, True),
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
            with patch.object(module, "host_float", counted_host_float), patch.object(
                module, "host_array", counted_host_array
            ), patch.object(
                module, "update_self_intersection_status", return_value=False
            ), patch.object(module, "BiotSavart", _BS):
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
                self.run_code_calls.append(
                    (iota, G, None if sdofs is None else sdofs.copy())
                )
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
        sdofs_warm = np.array([1.0, 2.0, 3.0])
        run_dict = self._make_candidate_run_dict(sdofs_warm)
        run_dict["J"] = last_J
        run_dict["dJ"] = last_dJ.copy()
        run_dict["donor_class"] = "stage2_seed_only"
        run_dict["search_policy"] = "repair_first"
        run_dict["adaptive_failure_penalty_weight"] = 1.5
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
        self.assertGreater(J_out, last_J + max(abs(last_J), 1.0))
        np.testing.assert_array_equal(dJ_out, last_dJ)
        np.testing.assert_array_equal(booz.surface.x, sdofs_warm)
        self.assertEqual(booz.res["iota"], TEST_IOTA)
        self.assertEqual(booz.res["G"], TEST_G0)
        self.assertFalse(booz.res["success"])
        self.assertEqual(run_dict["failure_count"], 1)
        failure_summary = run_dict["last_candidate_failure"]
        self.assertEqual(failure_summary["donor_class"], "stage2_seed_only")
        self.assertEqual(failure_summary["search_policy"], "repair_first")
        self.assertAlmostEqual(failure_summary["penalty"], J_out - last_J)
        self.assertEqual(failure_summary["step_norm"], 0.0)
        self.assertEqual(
            failure_summary["penalty_multiplier"],
            run_dict["adaptive_failure_penalty_weight"],
        )

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
        self.assertFalse(rd1["initial_objective_pending"])
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

    def test_snapshot_to_pytree_can_defer_legacy_initial_objective_for_target_lane(
        self,
    ):
        module = self.load_module()

        class _JF:
            x = np.array([99.0, 99.0])

            def J(self):
                raise AssertionError("legacy objective should not be evaluated")

            def dJ(self):
                raise AssertionError("legacy gradient should not be evaluated")

        class _Surface:
            x = np.array([10.0, 20.0, 30.0])

        class _BoozerSurface:
            def __init__(self):
                self.surface = _Surface()
                self.res = {
                    "success": True,
                    "iota": 0.15,
                    "G": 1.0,
                    "sdofs": np.array([10.0, 20.0, 30.0]),
                }

        class _Curve:
            def gamma(self):
                return np.ones((4, 3))

            def gammadash(self):
                return np.ones((4, 3)) * 2.0

        class _Current:
            def get_value(self):
                return 100.0

        class _Coil:
            curve = _Curve()
            current = _Current()

        class _BS:
            coils = [_Coil()]

        optimizer_dofs = np.array([1.0, 2.0, 3.0])
        with patch.object(
            module, "surface_self_intersection_check_available", return_value=True
        ):
            dofs, run_dict, _ = module.snapshot_to_pytree(
                _JF(),
                _BoozerSurface(),
                _BS(),
                num_tf_coils=1,
                coil_dofs_override=optimizer_dofs,
                evaluate_initial_objective=False,
            )

        np.testing.assert_allclose(dofs, optimizer_dofs)
        np.testing.assert_allclose(run_dict["x_prev"], optimizer_dofs)
        np.testing.assert_allclose(run_dict["dJ"], np.zeros_like(optimizer_dofs))
        self.assertTrue(np.isnan(run_dict["J"]))
        self.assertTrue(np.isnan(run_dict["initial_objective"]))
        self.assertTrue(run_dict["initial_objective_pending"])

        target_grad = np.array([4.0, 5.0, 6.0])
        module.seed_single_stage_initial_objective_from_values(
            run_dict,
            objective_value=12.5,
            objective_grad=target_grad,
        )
        self.assertEqual(run_dict["J"], 12.5)
        self.assertEqual(run_dict["initial_objective"], 12.5)
        np.testing.assert_allclose(run_dict["dJ"], target_grad)
        self.assertFalse(run_dict["initial_objective_pending"])

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
                use_coil_optimizer_dofs=False,
            ),
            np.array([1.0, 2.0, 3.0]),
        )
        target_lane_dofs = module.resolve_single_stage_outer_optimizer_initial_dofs(
            _JF(),
            _BS(),
            use_target_lane=True,
            use_coil_optimizer_dofs=True,
        )
        self.assertIsInstance(target_lane_dofs, jax.Array)
        np.testing.assert_allclose(
            module._single_stage_optimizer_dofs_array(target_lane_dofs),
            np.array([9.0, 8.0]),
        )

        bs = _BS()
        set_dofs = module.resolve_single_stage_outer_dof_setter(
            _JF(),
            bs,
            use_target_lane=True,
            use_coil_optimizer_dofs=True,
        )
        set_dofs(np.array([7.0, 6.0]))
        np.testing.assert_allclose(
            bs.x,
            np.array([7.0, 6.0]),
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

    def test_summarize_host_gradient_supports_scaled_outer_phase_state(self):
        module = self.load_module()
        phase1_grad = module.build_target_lane_scaled_outer_phase_state(
            np.array([10.0, 20.0], dtype=np.float64),
            jax.device_put(np.array([3.0, -4.0], dtype=np.float64)),
        )

        summary = module._summarize_host_gradient(phase1_grad)

        self.assertTrue(summary["all_finite"])
        self.assertEqual(summary["inf_norm"], 4.0)
        self.assertEqual(summary["size"], 2)
        self.assertEqual(summary["nonfinite_count"], 0)

    def test_build_traceable_single_stage_outer_objective_config_hostifies_vessel_gamma(
        self,
    ):
        module = self.load_module()
        host_calls = []
        original_host_array = module.host_array
        banana_curve = object()
        vessel_gamma = jax.device_put(
            np.arange(12.0, dtype=np.float64).reshape(2, 2, 3)
        )

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

    def test_restore_from_pytree_updates_diagnostic_field_when_provided(self):
        module = self.load_module()

        jf = types.SimpleNamespace(x=np.array([1.0, 2.0]))
        booz = types.SimpleNamespace(
            surface=types.SimpleNamespace(x=np.array([10.0, 20.0])),
            res={"iota": 0.1, "G": 2.0},
        )
        diagnostic_bs = types.SimpleNamespace(x=np.array([9.0, 9.0]))
        run_dict = {"sdofs": np.array([3.0, 4.0]), "iota": 0.2, "G": 5.0}

        module.restore_from_pytree(
            jf,
            booz,
            run_dict,
            coil_dofs=np.array([6.0, 7.0]),
            diagnostic_bs=diagnostic_bs,
        )

        np.testing.assert_allclose(jf.x, np.array([6.0, 7.0]))
        np.testing.assert_allclose(diagnostic_bs.x, np.array([6.0, 7.0]))
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

    def test_snapshot_to_pytree_prefers_solved_runtime_contract_over_live_surface(self):
        module = self.load_module()

        class _Surface:
            def __init__(self):
                self._x = np.array([99.0, 98.0, 97.0])

            @property
            def x(self):
                return self._x

            @x.setter
            def x(self, val):
                self._x = np.asarray(val)

        class _BoozerSurface:
            def __init__(self):
                self.surface = _Surface()
                self.res = {"success": True, "iter": 1, "iota": 9.9, "G": 8.8}

            def get_solved_runtime_state(self):
                return types.SimpleNamespace(
                    sdofs=jnp.asarray([10.0, 20.0, 30.0], dtype=jnp.float64),
                    iota=jnp.asarray(0.15, dtype=jnp.float64),
                    G=jnp.asarray(1.0, dtype=jnp.float64),
                )

        class _JF:
            def __init__(self):
                self._x = np.array([1.0, 2.0, 3.0])

            @property
            def x(self):
                return self._x

            def J(self):
                return 42.0

            def dJ(self):
                return np.array([0.1, 0.2, 0.3])

        class _Curve:
            def gamma(self):
                return np.ones((4, 3))

            def gammadash(self):
                return np.ones((4, 3)) * 0.1

        class _Current:
            def get_value(self):
                return 1.0

        class _Coil:
            def __init__(self):
                self.curve = _Curve()
                self.current = _Current()

        class _BS:
            def __init__(self):
                self.coils = [_Coil()]

        jf = _JF()
        booz = _BoozerSurface()
        bs_obj = _BS()

        with patch.object(
            module, "surface_self_intersection_check_available", return_value=True
        ):
            _, run_dict, _ = module.snapshot_to_pytree(jf, booz, bs_obj, num_tf_coils=1)

        np.testing.assert_allclose(run_dict["sdofs"], np.array([10.0, 20.0, 30.0]))
        self.assertEqual(run_dict["iota"], 0.15)
        self.assertEqual(run_dict["G"], 1.0)

    def test_boozer_residual_exact_compute_supports_cpu_vjp_contract(
        self,
    ):
        module = self.load_module()
        recorded = {}

        def fake_vjp(adjoint, boozer_surface, iota, G):
            recorded["adjoint"] = np.asarray(adjoint)
            recorded["boozer_surface"] = boozer_surface
            recorded["iota"] = iota
            recorded["G"] = G
            return module.Derivative({})

        in_surface = module.SurfaceXYZTensorFourier(
            mpol=1,
            ntor=1,
            stellsym=True,
            nfp=1,
            quadpoints_phi=np.linspace(0.0, 1.0, 4, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
        )
        in_surface.set_dofs(np.zeros_like(in_surface.get_dofs()))
        decision_size = in_surface.get_dofs().size + 2

        class _Label:
            def J(self):
                return 0.0

            def dJ_by_dsurfacecoefficients(self):
                return np.zeros(in_surface.get_dofs().size)

        class _BS:
            def set_points(self, _points):
                return None

            def B_vjp(self, _dJ_by_dB):
                return module.Derivative({})

        class _BoozerSurface(module.Optimizable):
            def __init__(self):
                module.Optimizable.__init__(self)
                self.surface = in_surface
                self.need_to_run_code = False
                self.label = _Label()
                self.targetlabel = 0.0
                self.res = {
                    "iota": 0.15,
                    "G": 1.0,
                    "PLU": tuple(np.eye(decision_size) for _ in range(3)),
                    "vjp": fake_vjp,
                }

        boozer_surface = _BoozerSurface()

        with patch.object(
            module,
            "boozer_surface_residual",
            return_value=(np.zeros(1), np.zeros((1, decision_size))),
        ), patch.object(
            module.BoozerResidualExact,
            "dJ_by_dB",
            lambda self: np.zeros((1, 3)),
        ):
            residual = module.BoozerResidualExact(boozer_surface, _BS())
            residual.compute()

        np.testing.assert_allclose(recorded["adjoint"], np.zeros(decision_size))
        self.assertIs(recorded["boozer_surface"], boozer_surface)
        self.assertEqual(recorded["iota"], 0.15)
        self.assertEqual(recorded["G"], 1.0)

    def test_snapshot_to_pytree_accepts_deferred_surface_parent_graph(self):
        module = self.load_module()
        from simsopt.geo.surfacerzfourier import SurfaceRZFourier

        quadpoints_phi = np.linspace(0.0, 1.0, 4, endpoint=False)
        quadpoints_theta = np.linspace(0.0, 1.0, 5, endpoint=False)
        template_surface = module.SurfaceXYZTensorFourier(
            mpol=1,
            ntor=1,
            nfp=1,
            stellsym=False,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        deferred_surface = module.DeferredSurfaceXYZTensorFourier(
            mpol=1,
            ntor=1,
            nfp=1,
            stellsym=False,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
            dofs=np.zeros_like(template_surface.x),
        )
        vessel_surface = SurfaceRZFourier(
            nfp=1,
            stellsym=False,
            mpol=1,
            ntor=0,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
        vessel_surface.set_rc(0, 0, 1.0)
        vessel_surface.set_rc(1, 0, 0.1)
        vessel_surface.set_zs(1, 0, 0.1)
        vessel_surface.fix_all()
        objective = SurfaceSurfaceDistance(
            deferred_surface,
            vessel_surface,
            minimum_distance=0.05,
        )

        class _JF:
            def __init__(self, composite_objective):
                self._objective = composite_objective

            @property
            def x(self):
                return self._objective.x

            def J(self):
                return jnp.asarray(0.25, dtype=jnp.float64)

            def dJ(self):
                return jnp.linspace(
                    0.0,
                    1.0,
                    self._objective.x.size,
                    dtype=jnp.float64,
                )

        class _BoozerSurface:
            def __init__(self, surface):
                self.surface = surface
                self.res = {"success": True, "iter": 1, "iota": 9.9, "G": 8.8}

            def get_solved_runtime_state(self):
                return types.SimpleNamespace(
                    sdofs=jnp.asarray(np.arange(template_surface.x.size), dtype=jnp.float64),
                    iota=jnp.asarray(0.15, dtype=jnp.float64),
                    G=jnp.asarray(1.0, dtype=jnp.float64),
                )

        class _Curve:
            def gamma(self):
                return np.ones((4, 3))

            def gammadash(self):
                return np.ones((4, 3)) * 0.1

        class _Current:
            def get_value(self):
                return 1.0

        class _Coil:
            def __init__(self):
                self.curve = _Curve()
                self.current = _Current()

        bs_obj = types.SimpleNamespace(coils=[_Coil()])

        with patch.object(
            module, "surface_self_intersection_check_available", return_value=True
        ):
            dofs, run_dict, _ = module.snapshot_to_pytree(
                _JF(objective),
                _BoozerSurface(deferred_surface),
                bs_obj,
                num_tf_coils=1,
            )

        np.testing.assert_allclose(dofs, objective.x)
        np.testing.assert_allclose(
            run_dict["sdofs"],
            np.arange(template_surface.x.size, dtype=np.float64),
        )
        self.assertEqual(run_dict["iota"], 0.15)
        self.assertEqual(run_dict["G"], 1.0)

    def test_snapshot_to_pytree_hostifies_tf_currents_under_strict_transfer_guard(self):
        module = self.load_module()
        host_calls = {"float": 0}
        original_host_float = module.host_float
        objective_value = jax.device_put(np.asarray(2.5, dtype=np.float64))
        objective_grad = jax.device_put(np.asarray([0.1, 0.2], dtype=np.float64))
        solved_sdofs = jax.device_put(np.asarray([3.0, 4.0], dtype=np.float64))
        solved_iota = jax.device_put(np.asarray(0.15, dtype=np.float64))
        solved_G = jax.device_put(np.asarray(1.0, dtype=np.float64))

        def counted_host_float(value):
            host_calls["float"] += 1
            return original_host_float(value)

        class _JF:
            x = np.array([1.0, 2.0], dtype=np.float64)

            def J(self):
                return objective_value

            def dJ(self):
                return objective_grad

        class _BoozerSurface:
            def __init__(self):
                self.surface = types.SimpleNamespace()
                self.res = {"success": True}

            def get_solved_runtime_state(self):
                return types.SimpleNamespace(
                    sdofs=solved_sdofs,
                    iota=solved_iota,
                    G=solved_G,
                )

        class _Curve:
            def gamma(self):
                return np.ones((4, 3))

            def gammadash(self):
                return np.ones((4, 3)) * 0.1

        class _Current:
            def __init__(self, value):
                self._value = jax.device_put(np.asarray(value, dtype=np.float64))

            def get_value(self):
                return self._value

        class _Coil:
            def __init__(self, current_value):
                self.curve = _Curve()
                self.current = _Current(current_value)

        bs_obj = types.SimpleNamespace(coils=[_Coil(100.0), _Coil(200.0)])

        with patch.object(module, "host_float", counted_host_float), patch.object(
            module, "surface_self_intersection_check_available", return_value=True
        ):
            with jax.transfer_guard("disallow"):
                _, _, static_config = module.snapshot_to_pytree(
                    _JF(),
                    _BoozerSurface(),
                    bs_obj,
                    num_tf_coils=2,
                )

        self.assertEqual(static_config["tf_currents"], [100.0, 200.0])
        self.assertGreaterEqual(host_calls["float"], 5)

    def test_host_curve_max_curvature_allows_strict_transfer_guard(self):
        module = self.load_module()
        host_calls = {"array": 0}
        original_host_array = module.host_array

        def counted_host_array(value, *, dtype=np.float64):
            host_calls["array"] += 1
            return original_host_array(value, dtype=dtype)

        class _Curve:
            def kappa(self):
                return jax.device_put(np.asarray([4.0, 6.0, 5.0], dtype=np.float64))

        with patch.object(module, "host_array", counted_host_array):
            with jax.transfer_guard("disallow"):
                max_curvature = module._host_curve_max_curvature(_Curve())

        self.assertEqual(max_curvature, 6.0)
        self.assertGreaterEqual(host_calls["array"], 1)

    def test_evaluate_single_stage_artifact_hardware_snapshot_hostifies_scalars(self):
        module = self.load_module()
        host_calls = {"float": 0}
        original_host_float = module.host_float

        def counted_host_float(value):
            host_calls["float"] += 1
            return original_host_float(value)

        with patch.object(module, "host_float", counted_host_float):
            with jax.transfer_guard("disallow"):
                snapshot = module.evaluate_single_stage_artifact_hardware_snapshot(
                    curve_curve_min_dist=jax.device_put(
                        np.asarray(0.20, dtype=np.float64)
                    ),
                    cc_dist=jax.device_put(np.asarray(0.05, dtype=np.float64)),
                    curve_surface_min_dist=jax.device_put(
                        np.asarray(0.30, dtype=np.float64)
                    ),
                    cs_dist=jax.device_put(np.asarray(0.04, dtype=np.float64)),
                    surface_vessel_min_dist=jax.device_put(
                        np.asarray(0.35, dtype=np.float64)
                    ),
                    ss_dist=jax.device_put(np.asarray(0.04, dtype=np.float64)),
                    max_curvature=jax.device_put(np.asarray(6.0, dtype=np.float64)),
                    curvature_threshold=jax.device_put(
                        np.asarray(10.0, dtype=np.float64)
                    ),
                    coil_length=jax.device_put(np.asarray(1.5, dtype=np.float64)),
                    length_target=jax.device_put(np.asarray(2.0, dtype=np.float64)),
                    banana_current_A=jax.device_put(
                        np.asarray(123.0, dtype=np.float64)
                    ),
                    banana_current_max_A=jax.device_put(
                        np.asarray(500.0, dtype=np.float64)
                    ),
                    tf_current_A=jax.device_put(
                        np.asarray(80000.0, dtype=np.float64)
                    ),
                    tf_current_limit_A=jax.device_put(
                        np.asarray(90000.0, dtype=np.float64)
                    ),
                )

        self.assertTrue(snapshot["artifact_hardware_status"]["success"])
        self.assertEqual(snapshot["max_curvature"], 6.0)
        self.assertEqual(snapshot["banana_current_A"], 123.0)
        self.assertEqual(snapshot["tf_current_A"], 80000.0)
        self.assertGreaterEqual(host_calls["float"], 14)


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
    def test_jax_curvature_threshold_matches_stage_specific_contracts(self):
        stage2_module = load_stage2_module()
        single_stage_module = load_single_stage_example_module()

        self.assertEqual(stage2_module.resolve_curvature_threshold(10.0), 10.0)
        self.assertEqual(stage2_module.resolve_curvature_threshold(20.0), 20.0)
        self.assertEqual(stage2_module.resolve_curvature_threshold(39.0), 39.0)
        self.assertEqual(stage2_module.resolve_curvature_threshold(80.0), 80.0)
        self.assertEqual(stage2_module.resolve_curvature_threshold(120.0), 100.0)
        self.assertEqual(single_stage_module.resolve_curvature_threshold(10.0), 20.0)
        self.assertEqual(single_stage_module.resolve_curvature_threshold(20.0), 20.0)
        self.assertEqual(single_stage_module.resolve_curvature_threshold(39.0), 39.0)
        self.assertEqual(single_stage_module.resolve_curvature_threshold(80.0), 80.0)
        self.assertEqual(single_stage_module.resolve_curvature_threshold(120.0), 100.0)

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
        self.assertEqual(
            set(status["violations"]),
            {
                "coil_length 1.800000 exceeds threshold 1.750000",
                "coil_coil_spacing 0.040000 below threshold 0.050000",
                "max_curvature 41.000000 exceeds threshold 40.000000",
            },
        )

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
                "coil_coil_spacing",
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
                "coil_coil_spacing",
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
                "coil_coil_spacing",
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

        with patch.object(
            module, "update_self_intersection_status", return_value=False
        ):
            J_out, dJ_out = module._evaluate_candidate_impl(
                np.ones(3),
                run_dict,
                booz,
                jf,
                objectives,
                diagnostics,
            )

        self.assertGreater(J_out, last_J + max(abs(last_J), 1.0))
        np.testing.assert_array_equal(dJ_out, last_dJ)
        self.assertFalse(run_dict["hardware_constraint_status"]["success"])
        self.assertEqual(run_dict["failure_count"], 1)
        failure_summary = run_dict["last_candidate_failure"]
        self.assertGreater(failure_summary["hardware_score"], 0.0)
        self.assertTrue(failure_summary["solver_success"])
        self.assertAlmostEqual(failure_summary["penalty"], J_out - last_J)

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

    def test_norm_field_summary_keeps_jax_field_on_host_reporting_boundary(self):
        spec = importlib.util.spec_from_file_location(
            f"plotting_utils_{uuid.uuid4().hex}",
            self.PLOTTING_UTILS_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)

        class Surface:
            quadpoints_phi = np.array([0.0, 0.5])
            quadpoints_theta = np.array([0.0, 0.5])

            def normal(self):
                normal = np.zeros((2, 2, 3), dtype=np.float64)
                normal[:, :, 2] = 1.0
                return normal

            def gamma(self):
                return np.zeros((2, 2, 3), dtype=np.float64)

        class Field:
            def __init__(self):
                self.points = None
                self.field = jax.device_put(np.ones((4, 3), dtype=np.float64))

            def set_points(self, points):
                self.points = points

            def B(self):
                return self.field

        with jax.transfer_guard("disallow"):
            field_error, *_ = module.norm_field_summary(Surface(), Field())

        self.assertTrue(np.isfinite(field_error))


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

    def test_parse_args_accepts_outer_ftol_override(self):
        module = load_single_stage_example_module()

        with patch.object(sys, "argv", ["single_stage_banana_example.py", "--outer-ftol", "0.0"]):
            args = module.parse_args()

        self.assertEqual(args.outer_ftol, 0.0)

    def test_source_uses_default_argument(self):
        """The deployed .get() calls must include a default, not bare .get(mpol)."""
        source = EXAMPLE_MODULE_PATH.read_text()
        self.assertNotIn("ftol_by_mpol.get(mpol)", source)
        self.assertNotIn("gtol_by_mpol.get(mpol)", source)


class ResultsEnvelopeTests(unittest.TestCase):
    def test_stage2_select_better_exact_hardware_pass_prefers_lower_field_objective(
        self,
    ):
        module = load_stage2_module()
        current = module.Stage2ExactHardwarePass(
            dofs=np.asarray([1.0, 2.0], dtype=float),
            field_objective=2.0,
            source="initial_state",
        )
        candidate = module.Stage2ExactHardwarePass(
            dofs=np.asarray([3.0, 4.0], dtype=float),
            field_objective=1.0,
            source="accepted_iterate_3",
        )

        chosen = module.select_better_stage2_exact_hardware_pass(current, candidate)

        self.assertIs(chosen, candidate)

    def test_stage2_restore_exact_hardware_pass_only_for_final_strict_miss(self):
        module = load_stage2_module()
        best = module.Stage2ExactHardwarePass(
            dofs=np.asarray([0.9, 0.8], dtype=float),
            field_objective=0.4,
            source="accepted_iterate_2",
        )

        restored_dofs, optimizer_success, termination_message = (
            module.restore_stage2_exact_hardware_pass_for_artifact_output(
                best,
                {"hardware_status": {"success": False, "violations": ["coil_length"]}},
                optimizer_success=True,
                termination_message="alm_ok",
            )
        )

        np.testing.assert_allclose(restored_dofs, np.asarray([0.9, 0.8], dtype=float))
        self.assertFalse(optimizer_success)
        self.assertEqual(
            termination_message,
            "alm_ok; restored_best_exact_hardware_pass",
        )

        restored_dofs, optimizer_success, termination_message = (
            module.restore_stage2_exact_hardware_pass_for_artifact_output(
                best,
                {"hardware_status": {"success": True, "violations": []}},
                optimizer_success=True,
                termination_message="alm_ok",
            )
        )

        self.assertIsNone(restored_dofs)
        self.assertTrue(optimizer_success)
        self.assertEqual(termination_message, "alm_ok")

    def test_stage2_select_better_feasible_partial_prefers_lower_objective(self):
        module = load_stage2_module()
        current = module.Stage2FeasiblePartial(
            dofs=np.asarray([1.0, 2.0], dtype=float),
            objective=2.0,
            curve_length=0.95,
            coil_coil_distance=0.07,
            max_curvature=9.0,
            accepted_index=1,
        )
        candidate = module.Stage2FeasiblePartial(
            dofs=np.asarray([3.0, 4.0], dtype=float),
            objective=1.0,
            curve_length=0.96,
            coil_coil_distance=0.08,
            max_curvature=10.0,
            accepted_index=2,
        )

        chosen = module.select_better_stage2_feasible_partial(current, candidate)

        self.assertIs(chosen, candidate)

    def test_stage2_should_restore_feasible_partial_only_for_non_success(self):
        module = load_stage2_module()
        best = module.Stage2FeasiblePartial(
            dofs=np.asarray([1.0], dtype=float),
            objective=1.0,
            curve_length=0.9,
            coil_coil_distance=0.08,
            max_curvature=8.0,
            accepted_index=2,
        )
        worse_final = module.Stage2FeasiblePartial(
            dofs=np.asarray([2.0], dtype=float),
            objective=2.0,
            curve_length=0.91,
            coil_coil_distance=0.07,
            max_curvature=8.5,
            accepted_index=4,
        )

        self.assertTrue(
            module.should_restore_stage2_feasible_partial(
                best,
                worse_final,
                optimizer_success=False,
            )
        )
        self.assertFalse(
            module.should_restore_stage2_feasible_partial(
                best,
                worse_final,
                optimizer_success=True,
            )
        )
        self.assertTrue(
            module.should_restore_stage2_feasible_partial(
                best,
                None,
                optimizer_success=False,
            )
        )

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
        problem_contract = envelope["problem_contract"]
        runtime_contract = problem_contract["runtime_contract"]
        hardware_thresholds = problem_contract["hardware_thresholds"]
        self.assertEqual(
            runtime_contract["constraint_method"],
            "penalty",
        )
        self.assertEqual(envelope["provenance"]["repo_sha"], "deadbeef")
        self.assertEqual(
            problem_contract["equilibrium"]["filename"],
            "wout_fixture.nc",
        )
        self.assertEqual(
            runtime_contract["optimizer_backend"],
            "ondevice",
        )
        self.assertEqual(hardware_thresholds["coil_plasma_distance"], 0.015)
        self.assertEqual(hardware_thresholds["coil_vessel_clearance"], 0.002)
        self.assertEqual(hardware_thresholds["plasma_vessel_distance"], 0.04)
        self.assertTrue(envelope["artifacts"]["required"]["results.json"]["exists"])
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
            outer_ftol=0.0,
            target_lane_outer_initial_step_size=None,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            profile_target_lane_memory_analysis=False,
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
                banana_surf_radius=0.219,
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
        problem_contract = envelope["problem_contract"]
        runtime_contract = problem_contract["runtime_contract"]
        hardware_thresholds = problem_contract["hardware_thresholds"]
        stage2_seed = problem_contract["stage2_seed"]
        self.assertEqual(hardware_thresholds["coil_vessel_clearance"], 0.002)
        self.assertEqual(runtime_contract["effective_banana_surface_radius"], 0.219)
        self.assertEqual(
            stage2_seed["banana_surface_radius"],
            0.22,
        )
        self.assertIsNone(stage2_seed["biot_savart_path"])
        self.assertEqual(
            stage2_seed["jax_runtime_spec_path"],
            "/tmp/stage2/results.json",
        )
        self.assertIn(
            module._SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME,
            envelope["artifacts"]["required"],
        )
        self.assertNotIn("biot_savart_opt.json", envelope["artifacts"]["required"])
        self.assertNotIn("surf_opt.json", envelope["artifacts"]["required"])

    def test_single_stage_restart_artifact_filenames_are_backend_specific(self):
        module = load_single_stage_example_module()

        jax_files = module.single_stage_restart_artifact_filenames(
            types.SimpleNamespace(backend="jax")
        )
        cpu_files = module.single_stage_restart_artifact_filenames(
            types.SimpleNamespace(backend="cpu")
        )

        self.assertEqual(jax_files, (module._SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME,))
        self.assertEqual(
            cpu_files,
            (
                "biot_savart_opt.json",
                "surf_opt.json",
                module._SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME,
            ),
        )

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
            outer_ftol=0.0,
            target_lane_outer_initial_step_size=None,
            initial_step_scale=0.25,
            initial_step_maxiter=4,
            profile_target_lane_memory_analysis=False,
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
                banana_surf_radius=0.22,
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
        self.assertFalse(envelope["artifacts"]["policy"]["write_restart_artifacts"])
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
            outer_ftol=0.0,
            target_lane_outer_initial_step_size=None,
            initial_step_scale=1.0,
            initial_step_maxiter=0,
            profile_target_lane_memory_analysis=False,
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
                banana_surf_radius=0.22,
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
