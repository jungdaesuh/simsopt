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

import jax.numpy as jnp
import numpy as np

from simsopt._core.optimizable import Optimizable
from simsopt.geo.surfaceobjectives import (
    SurfaceSurfaceDistance,
    boozer_surface_residual,
    boozer_surface_residual_dB,
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
_SINGLE_STAGE_HYBRID_UNSUPPORTED = (
    "optimizer_backend='hybrid' is transitional and not supported for "
    "the single-stage outer loop"
)
_SINGLE_STAGE_CPU_ONLY_SCIPY = (
    "single-stage outer loop CPU/reference lane only supports optimizer_backend='scipy'"
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

    def load_module(self):
        return load_single_stage_example_module()

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
            ):
                self.bs = bs
                self.surface = surface
                self.label = label
                self.targetlabel = targetlabel
                self.constraint_weight = constraint_weight
                self.options = options or {}
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
        with patch.object(
            module, "SurfaceXYZTensorFourier", FakeSurfaceXYZTensorFourier
        ), patch.object(module, "Volume", FakeVolume), patch.object(
            module, "BoozerSurface", FailingCPUBoozerSurface
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
        resolve_continuous_optimizer_contract=None,
        scipy_minimize_side_effect=None,
    ):
        if resolve_continuous_optimizer_contract is None:

            def resolve_continuous_optimizer_contract(
                field_backend,
                optimizer_backend,
                *,
                limited_memory,
                allow_hybrid,
                component_label,
            ):
                del component_label
                if field_backend != "jax" and optimizer_backend != "scipy":
                    raise ValueError(f"the {_SINGLE_STAGE_CPU_ONLY_SCIPY}.")
                if optimizer_backend == "hybrid":
                    if not allow_hybrid:
                        raise ValueError(_SINGLE_STAGE_HYBRID_UNSUPPORTED)
                    require_target_backend_x64(optimizer_backend)
                    return types.SimpleNamespace(
                        method="bfgs-hybrid",
                        use_scalar_objective=False,
                    )
                if optimizer_backend == "ondevice":
                    require_target_backend_x64(optimizer_backend)
                    return types.SimpleNamespace(
                        method="lbfgs-ondevice",
                        use_scalar_objective=(field_backend == "jax"),
                    )
                if optimizer_backend == "scipy":
                    return types.SimpleNamespace(
                        method="lbfgs" if limited_memory else "bfgs",
                        use_scalar_objective=False,
                    )
                raise ValueError(
                    "optimizer_backend must be one of: scipy, hybrid, ondevice."
                )

        from functools import partial

        resolve_outer_loop_optimizer_contract = partial(
            resolve_continuous_optimizer_contract,
            limited_memory=True,
            allow_hybrid=False,
        )

        fake_optimizer_module = types.ModuleType("simsopt.geo.optimizer_jax")
        fake_optimizer_module.jax_minimize = jax_minimize
        fake_optimizer_module.require_target_backend_x64 = require_target_backend_x64
        fake_optimizer_module.resolve_continuous_optimizer_contract = (
            resolve_continuous_optimizer_contract
        )
        fake_optimizer_module.resolve_outer_loop_optimizer_contract = (
            resolve_outer_loop_optimizer_contract
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
        self.assertEqual(boozer_surface.options["optimizer_backend"], "scipy")
        self.assertIs(boozer_surface.surface, FakeSurfaceXYZTensorFourier.instances[0])

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
            module.resolve_boozer_optimizer_backend("jax", "ondevice", "scipy"),
            "scipy",
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

        def _runtime_builder(*args, include_profile_suite=False, success_filter=None):
            runtime_calls.append((include_profile_suite, success_filter))
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
                )
            )

        self.assertIsNone(scalar_fun)
        self.assertIs(value_and_grad_fun, value_and_grad_marker)
        self.assertIsNone(target_lane_profile)
        self.assertEqual(runtime_calls, [(False, None)])

    def test_build_target_lane_outer_objectives_threads_success_filter_to_runtime_bundle(
        self,
    ):
        module = self.load_module()
        objective_marker = object()
        success_filter_marker = object()
        runtime_calls = []

        def _runtime_builder(*args, include_profile_suite=False, success_filter=None):
            runtime_calls.append((include_profile_suite, success_filter))
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
                    success_filter=success_filter_marker,
                )
            )

        self.assertIs(scalar_fun, objective_marker)
        self.assertIsNone(value_and_grad_fun)
        self.assertIsNone(target_lane_profile)
        self.assertEqual(runtime_calls, [(False, success_filter_marker)])

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

    def test_run_single_stage_optimizer_prefers_fused_target_lane_contract(self):
        module = self.load_module()
        captured = {}
        explicit_fun = lambda x: (float(np.dot(x, x)), np.asarray(2.0 * x, dtype=float))
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
                callback=callback,
                scalar_fun=scalar_fun,
            )

        self.assertEqual(captured["x64_backend"], "ondevice")
        self.assertIsInstance(captured["x0"], module.SingleStageOuterOptimizerState)
        self.assertEqual(captured["method"], "lbfgs-ondevice")
        np.testing.assert_allclose(
            module._single_stage_optimizer_dofs_array(captured["x0"]),
            np.array([1.0, -2.0]),
        )
        self.assertEqual(captured["tol"], 1e-6)
        self.assertEqual(captured["maxiter"], 7)
        self.assertEqual(captured["options"], {"maxcor": 9, "ftol": 1e-8})
        self.assertTrue(captured["value_and_grad"])
        self.assertIs(captured["callback"], callback)
        value, grad = captured["fun"](captured["x0"])
        self.assertEqual(value, 5.0)
        self.assertIsInstance(grad, module.SingleStageOuterOptimizerState)
        np.testing.assert_allclose(
            module._single_stage_optimizer_dofs_array(grad),
            np.array([2.0, -4.0]),
        )
        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_target_lane_requires_objective_contract(self):
        module = self.load_module()
        target_contract = types.SimpleNamespace(
            method="lbfgs-ondevice", use_scalar_objective=True
        )

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=lambda _optimizer_backend: None,
            jax_minimize=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("jax_minimize should not run without an objective")
            ),
        ):
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
                    callback=None,
                )

    def test_run_single_stage_optimizer_ondevice_does_not_enter_scipy_minimize(self):
        module = self.load_module()
        explicit_fun = lambda x: (float(np.dot(x, x)), np.asarray(2.0 * x, dtype=float))

        def fake_require_target_backend_x64(_optimizer_backend):
            return None

        def fake_jax_minimize(
            fun, x0, *, method, tol, maxiter, options, value_and_grad, callback
        ):
            value, grad = fun(x0)
            self.assertEqual(value, 0.0)
            self.assertIsInstance(grad, module.SingleStageOuterOptimizerState)
            np.testing.assert_allclose(
                module._single_stage_optimizer_dofs_array(grad),
                np.zeros(2),
            )
            self.assertIsInstance(x0, module.SingleStageOuterOptimizerState)
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
                callback=lambda _x: None,
                scalar_fun=None,
            )

        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_allows_explicit_experimental_target_lane(self):
        module = self.load_module()
        captured = {}
        explicit_fun = lambda x: (float(np.dot(x, x)), np.asarray(2.0 * x, dtype=float))

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
                callback=None,
                scalar_fun=None,
            )

        self.assertIsInstance(captured["x0"], module.SingleStageOuterOptimizerState)
        self.assertTrue(captured["value_and_grad"])
        value, grad = captured["fun"](captured["x0"])
        self.assertEqual(value, 0.0)
        self.assertIsInstance(grad, module.SingleStageOuterOptimizerState)
        self.assertEqual(result.message, "ok")

    def test_run_single_stage_optimizer_rejects_hybrid_outer_lane(self):
        module = self.load_module()

        def fake_require_target_backend_x64(optimizer_backend):
            raise AssertionError(
                f"x64 check should not run for unsupported hybrid lane: {optimizer_backend}"
            )

        def fake_jax_minimize(
            fun, x0, *, method, tol, maxiter, options, value_and_grad, callback
        ):
            raise AssertionError(
                "Unsupported single-stage hybrid lane must fail before jax_minimize."
            )

        with self.patch_optimizer_jax_module(
            require_target_backend_x64=fake_require_target_backend_x64,
            jax_minimize=fake_jax_minimize,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "optimizer_backend='hybrid' is transitional and not supported "
                "for the single-stage outer loop",
            ):
                module.resolve_single_stage_optimizer_contract("jax", "hybrid")

    def test_resolve_single_stage_outer_optimizer_method_rejects_cpu_ondevice(self):
        module = self.load_module()

        with self.assertRaisesRegex(
            ValueError,
            _SINGLE_STAGE_CPU_ONLY_SCIPY,
        ):
            module.resolve_single_stage_outer_optimizer_method("cpu", "ondevice")

    def test_resolve_single_stage_outer_optimizer_method_rejects_hybrid(self):
        module = self.load_module()

        with self.assertRaisesRegex(
            ValueError,
            _SINGLE_STAGE_HYBRID_UNSUPPORTED,
        ):
            module.resolve_single_stage_outer_optimizer_method("jax", "hybrid")

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

        def fake_snapshot(state, booz, objective):
            captured["snapshot"] = {
                "state": state,
                "booz": booz,
                "objective": objective,
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
        self.assertEqual(run_dict["lscount"], 4)

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

        Regression test: evaluate_candidate must detect the CPU backend
        via isinstance and use the old warm-start path (surface.x and
        res mutation) instead of passing sdofs= to run_code.
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
        run_dict = {
            "x_prev": np.zeros(5),
            "lscount": 0,
            "sdofs": sdofs_warm.copy(),
            "iota": TEST_IOTA,
            "G": TEST_G0,
            "J": 1.0,
            "dJ": np.zeros(5),
        }
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
        self.assertIsInstance(target_lane_dofs, module.SingleStageOuterOptimizerState)
        np.testing.assert_allclose(
            module._single_stage_optimizer_dofs_array(target_lane_dofs),
            np.array([9.0, 8.0]),
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


def _load_segment_distance_from_source():
    """Extract the deployed segment_segment_distance from banana_coil_solver.py via AST.

    Parses the source file, extracts just the _clamp01 and segment_segment_distance
    function definitions (stripping @njit decorators), and compiles them in an
    isolated namespace. This executes the REAL deployed algorithm without requiring
    numba or triggering module-level code (arg parsing, VMEC file loading, etc.).
    """
    import ast

    source = STAGE2_MODULE_PATH.read_text()
    tree = ast.parse(source)
    func_nodes = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_clamp01",
            "segment_segment_distance",
        ):
            node.decorator_list = []
            func_nodes.append(node)
    extracted = ast.Module(body=func_nodes, type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace = {"np": np}
    exec(compile(extracted, str(STAGE2_MODULE_PATH), "exec"), namespace)
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


if __name__ == "__main__":
    unittest.main()
