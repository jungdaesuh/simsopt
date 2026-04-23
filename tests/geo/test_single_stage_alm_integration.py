import ast
import importlib
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import uuid
import hashlib
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


SINGLE_STAGE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)
HARDWARE_CONSTRAINT_SCHEMA_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "hardware_constraint_schema.py"
)
SINGLE_STAGE_THRESHOLDED_PHYSICS_RERUN_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "run_single_stage_thresholded_physics_alm.py"
)
DEFAULT_ALM_WRAPPER_SURFACE = "wout_nfp10ginsburg_desc_s024match_iota20.nc"
LEGACY_STAGE2_RESULTS_PAYLOAD = {
    "PLASMA_SURF_FILENAME": DEFAULT_ALM_WRAPPER_SURFACE,
    "BANANA_CURRENT_A": 12000.0,
    "TF_CURRENT_A": 5000.0,
    "NUM_TF_COILS": 16,
    "init_only": False,
}
LEGACY_STAGE2_UPGRADED_FIELDS = {
    "BANANA_INIT_CURRENT_A": 1.0e4,
    "BANANA_CURRENT_MAX_A": 1.6e4,
    "TF_CURRENT_SUM_ABS_A": 80000.0,
    "LENGTH_TARGET": 1.7,
    "COIL_PLASMA_MIN_DIST_M": 0.015,
    "COIL_VESSEL_MIN_DIST_M": 0.002,
    "PLASMA_VESSEL_MIN_DIST_M": 0.04,
}


def extract_functions(module_path: Path, function_names: list[str], global_bindings: dict):
    tree = ast.parse(module_path.read_text(), filename=str(module_path))
    selected_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in set(function_names)
    ]
    module = ast.Module(body=selected_nodes, type_ignores=[])
    namespace = dict(global_bindings)
    exec(compile(module, str(module_path), "exec"), namespace)
    return {name: namespace[name] for name in function_names}


def single_stage_example_package_root() -> str:
    package_root = str(HARDWARE_CONSTRAINT_SCHEMA_MODULE_PATH.parents[1])
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    return package_root


def load_hardware_constraint_schema_module():
    single_stage_example_package_root()
    return importlib.import_module("banana_opt.hardware_constraint_schema")


def load_hardware_contracts_module():
    single_stage_example_package_root()
    return importlib.import_module("banana_opt.hardware_contracts")


def load_stage2_artifact_contracts_module():
    single_stage_example_package_root()
    return importlib.import_module("banana_opt.artifact_contracts")


def load_alm_utils_module():
    single_stage_example_package_root()
    return importlib.import_module("alm_utils")


def write_stage2_artifact_bundle(
    tmpdir_path: Path,
    *,
    results_payload: dict[str, object],
) -> Path:
    stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
    stage2_results_path = tmpdir_path / "results.json"
    stage2_bs_path.write_text("{}", encoding="utf-8")
    stage2_results_path.write_text(
        json.dumps(results_payload),
        encoding="utf-8",
    )
    return stage2_bs_path


def assert_legacy_stage2_fields_upgraded(
    testcase: unittest.TestCase,
    stage2_results: dict[str, object],
) -> None:
    for key, expected in LEGACY_STAGE2_UPGRADED_FIELDS.items():
        testcase.assertEqual(stage2_results[key], expected)


def load_single_stage_thresholded_physics_rerun_module():
    spec = importlib.util.spec_from_file_location(
        f"run_single_stage_thresholded_physics_alm_{uuid.uuid4().hex}",
        SINGLE_STAGE_THRESHOLDED_PHYSICS_RERUN_MODULE_PATH,
        )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_single_stage_thresholded_physics_rerun_args(**overrides):
    defaults = {
        "python_executable": "python",
        "dry_run": False,
        "plasma_surf_filename": DEFAULT_ALM_WRAPPER_SURFACE,
        "stage2_bs_path": "relative/seed.json",
        "equilibria_dir": None,
        "output_root": "outputs",
        "summary_json": None,
        "allow_init_only_stage2_seed": False,
        "single_stage_timeout_seconds": 0.0,
        "backend": "jax",
        "optimizer_backend": None,
        "boozer_optimizer_backend": None,
        "boozer_least_squares_algorithm": None,
        "minimal_artifacts": False,
        "benchmark_mode": False,
        "nphi": 91,
        "ntheta": 32,
        "mpol": 8,
        "ntor": 6,
        "maxiter": 300,
        "iota_target": 0.2,
        "vol_target": 0.1,
        "cc_dist": 0.05,
        "cs_dist": 0.015,
        "ss_dist": 0.04,
        "curvature_threshold": 100.0,
        "length_target": 1.7,
        "banana_current_max_A": 1.6e4,
        "alm_max_outer_iters": 20,
        "alm_max_subproblem_continuations": 20,
        "alm_penalty_init": 1.0,
        "alm_penalty_scale": 10.0,
        "alm_penalty_max": 1.0e8,
        "alm_feas_tol": 1.0e-6,
        "alm_stationarity_tol": 1.0e-6,
        "alm_trust_radius_init": 0.05,
        "alm_trust_radius_min": 1.0e-4,
        "alm_trust_radius_shrink": 0.5,
        "alm_trust_radius_grow": 1.5,
        "alm_max_inner_attempts": 4,
        "alm_distance_smoothing": 0.005,
        "alm_curvature_smoothing": 0.05,
        "alm_qs_threshold": 3e-3,
        "alm_boozer_threshold": 1e-2,
        "alm_iota_penalty_threshold": 1e-4,
        "alm_length_penalty_threshold": 0.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class SingleStageAlmIntegrationTests(unittest.TestCase):
    def test_validate_banana_winding_surface_radius_enforces_coil_vessel_clearance(self):
        contracts_module = load_hardware_contracts_module()

        self.assertEqual(
            contracts_module.validate_banana_winding_surface_radius(0.220),
            0.220,
        )
        with self.assertRaisesRegex(ValueError, "coil-to-vessel clearance contract"):
            contracts_module.validate_banana_winding_surface_radius(0.2201)

    def test_single_stage_alm_inner_optimizer_contract_selects_target_only_for_alm(self):
        from simsopt.geo.optimizer_jax import (
            ReferenceOptimizerContract,
            TargetOptimizerContract,
        )

        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["resolve_single_stage_alm_inner_optimizer_contract"],
            {},
        )
        resolve_contract = functions["resolve_single_stage_alm_inner_optimizer_contract"]
        target_contract = TargetOptimizerContract(method="lbfgs-ondevice")

        self.assertIs(
            resolve_contract("alm", target_contract),
            target_contract,
        )
        self.assertIsNone(resolve_contract("penalty", target_contract))
        self.assertIsNone(
            resolve_contract(
                "alm",
                ReferenceOptimizerContract(method="scipy-lbfgsb"),
            )
        )

    def test_single_stage_alm_constraint_names_follow_shared_schema(self):
        schema_module = load_hardware_constraint_schema_module()
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["single_stage_alm_constraint_names"],
            {
                "hardware_constraint_alm_names": schema_module.hardware_constraint_alm_names,
                "SINGLE_STAGE_THRESHOLDED_PHYSICS_CONSTRAINT_NAMES": (
                    "qs_error",
                    "boozer_residual",
                    "iota_penalty",
                    "length_penalty",
                ),
            },
        )

        weighted_sum_names = functions["single_stage_alm_constraint_names"](
            alm_formulation="weighted_sum",
            include_surface_surface=True,
        )
        thresholded_names = functions["single_stage_alm_constraint_names"](
            alm_formulation="thresholded_physics",
            include_surface_surface=False,
        )

        self.assertEqual(
            weighted_sum_names,
            [
                "coil_coil_spacing",
                "coil_surface_spacing",
                "surface_vessel_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "banana_current_upper_bound",
            ],
        )
        self.assertEqual(
            thresholded_names,
            [
                "coil_coil_spacing",
                "coil_surface_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "banana_current_upper_bound",
                "qs_error",
                "boozer_residual",
                "iota_penalty",
                "length_penalty",
            ],
        )

    def test_weighted_sum_alm_runtime_config_keeps_thresholds_optional(self):
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["build_traceable_single_stage_alm_runtime_config"],
            {},
        )

        config = functions["build_traceable_single_stage_alm_runtime_config"](
            constraint_names=("coil_coil_spacing",),
            alm_formulation="weighted_sum",
            distance_smoothing=0.005,
            curvature_smoothing=0.05,
            qs_threshold=None,
            boozer_threshold=None,
            iota_penalty_threshold=None,
            length_penalty_threshold=None,
            banana_current_threshold=1.6e4,
        )

        self.assertIsNone(config["qs_threshold"])
        self.assertIsNone(config["boozer_threshold"])
        self.assertIsNone(config["iota_penalty_threshold"])
        self.assertIsNone(config["length_penalty_threshold"])

    def test_resolve_single_stage_banana_surface_radius_uses_shared_validator(self):
        contracts_module = load_hardware_contracts_module()
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["resolve_single_stage_banana_surface_radius"],
            {
                "validate_banana_winding_surface_radius": (
                    contracts_module.validate_banana_winding_surface_radius
                ),
            },
        )

        args = SimpleNamespace(banana_surf_radius=None)
        stage2_results = {"banana_surf_radius": 0.220}

        self.assertEqual(
            functions["resolve_single_stage_banana_surface_radius"](args, stage2_results),
            0.220,
        )

        args = SimpleNamespace(banana_surf_radius=0.2201)
        with self.assertRaisesRegex(ValueError, "coil-to-vessel clearance contract"):
            functions["resolve_single_stage_banana_surface_radius"](args, stage2_results)

    def test_single_stage_load_stage2_results_upgrades_legacy_artifact_metadata(self):
        artifact_contracts_module = load_stage2_artifact_contracts_module()
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["load_stage2_results"],
            {
                "json": json,
                "os": os,
                "upgrade_legacy_stage2_artifact_results": (
                    artifact_contracts_module.upgrade_legacy_stage2_artifact_results
                ),
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = write_stage2_artifact_bundle(
                tmpdir_path,
                results_payload=LEGACY_STAGE2_RESULTS_PAYLOAD,
            )

            _, stage2_results = functions["load_stage2_results"](str(stage2_bs_path))

        assert_legacy_stage2_fields_upgraded(self, stage2_results)

    def test_single_stage_target_lane_self_intersection_success_filter_rejects_crossing_surface(
        self,
    ):
        from simsopt._core.jax_host_boundary import host_array
        from simsopt.jax_core._math_utils import (
            as_jax_float64 as _as_jax_float64,
            as_jax_int32 as _as_jax_int32,
            as_runtime_float64 as _as_runtime_float64,
        )
        from simsopt.geo.curve import surfrz_gamma_lin, surfxyztensor_gamma_lin
        from simsopt.jax_core.curve_geometry import (
            closed_curve_self_intersection_summary,
        )

        helpers_dir = str(Path(__file__).resolve().parent)
        if helpers_dir not in sys.path:
            sys.path.insert(0, helpers_dir)
        from surface_test_helpers import get_surface

        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            [
                "_hostify_target_lane_constant_tree",
                "_surface_rzfourier_dof_count",
                "_supported_surface_self_intersection_inputs",
                "_surface_phi0_cross_section_from_supported_dofs",
                "_supported_surface_self_intersection_flag_from_dofs",
                "build_single_stage_target_lane_self_intersection_success_filter",
            ],
            {
                "hashlib": hashlib,
                "host_array": host_array,
                "jax": jax,
                "jnp": jnp,
                "json": json,
                "np": np,
                "_target_lane_success_filter_cache_signature": lambda payload: "test",
                "_SURFACE_SELF_INTERSECTION_BISECTION_STEPS": 48,
                "_SURFACE_SELF_INTERSECTION_TOLERANCE_FACTOR": 1.0e-9,
                "_as_jax_float64": _as_jax_float64,
                "_as_jax_int32": _as_jax_int32,
                "_as_runtime_float64": _as_runtime_float64,
                "closed_curve_self_intersection_summary": (
                    closed_curve_self_intersection_summary
                ),
                "surfrz_gamma_lin": surfrz_gamma_lin,
                "surfxyztensor_gamma_lin": surfxyztensor_gamma_lin,
            },
        )

        build_filter = functions[
            "build_single_stage_target_lane_self_intersection_success_filter"
        ]
        crossing_surface_dofs = np.array(
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
        solve_tail = jnp.asarray([0.15, 1.0], dtype=jnp.float64)
        empty_coil_dofs = jnp.zeros((0,), dtype=jnp.float64)

        def make_surface(dofs=None):
            surface = get_surface(
                "SurfaceRZFourier",
                True,
                full=True,
                nphi=200,
                ntheta=200,
                mpol=2,
                ntor=2,
            )
            if dofs is not None:
                surface.x = np.asarray(dofs, dtype=np.float64)
            return surface

        crossing_surface = make_surface(crossing_surface_dofs)
        smooth_surface = make_surface()
        surface_dof_count = crossing_surface.get_dofs().size

        class FakeBoozerSurface:
            def __init__(self, surface):
                self.surface = surface
                self.res = {"G": 1.0}

            def _unpack_decision_vector_jax(self, x, optimize_G, coil_set_spec=None):
                del coil_set_spec
                x_arr = jnp.asarray(x, dtype=jnp.float64).reshape((-1,))
                return (
                    x_arr[:surface_dof_count],
                    x_arr[surface_dof_count],
                    x_arr[surface_dof_count + 1],
                )

        success_filter = build_filter(FakeBoozerSurface(crossing_surface), object())

        def success_filter_result(surface):
            decision_vector = jnp.concatenate(
                (
                    jnp.asarray(surface.get_dofs(), dtype=jnp.float64),
                    solve_tail,
                )
            )
            return bool(np.asarray(success_filter(empty_coil_dofs, decision_vector)))

        self.assertFalse(success_filter_result(crossing_surface))
        self.assertTrue(success_filter_result(smooth_surface))

    def test_single_stage_partial_alm_state_payload_serializes_numpy_fields(self):
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["_jsonable_value", "build_single_stage_alm_partial_state"],
            {"np": np},
        )
        build_single_stage_alm_partial_state = functions[
            "build_single_stage_alm_partial_state"
        ]

        payload = build_single_stage_alm_partial_state(
            {
                "accepted_iterations": 3,
                "it": 4,
                "J": 0.25,
                "accepted_boozer_stage": "initial",
                "accepted_hardware_status": {
                    "success": np.bool_(False),
                    "violations": ["cc"],
                },
                "trial_hardware_status": {
                    "success": np.bool_(False),
                    "violations": ["cs"],
                },
                "topology_gate_status": {
                    "success": np.bool_(True),
                    "survived_lines": np.int64(6),
                },
            },
            ["curve_curve", "curve_surface"],
            [{"outer_iteration": 1, "action": "penalty_increase"}],
            {
                "outer_iteration": 1,
                "constraint_values": np.array([0.1, 0.2]),
                "solver_constraint_values": np.array([0.3, 0.4]),
                "action": "penalty_increase",
            },
            np.array([0.5, 0.25]),
            10.0,
            outer_iteration=2,
            termination_message="still running",
            optimizer_success=None,
            termination_reason="subproblem_continue",
            inner_optimizer_success=False,
            inner_optimizer_message="iteration limit",
            converged_to_tolerances=False,
            restored_best_feasible=True,
            restored_best_feasible_reason="final_iterate_infeasible",
            final_max_feasibility_violation=0.2,
            final_stationarity_norm=0.15,
        )

        self.assertEqual(payload["outer_iteration"], 2)
        self.assertEqual(payload["constraint_names"], ["curve_curve", "curve_surface"])
        self.assertEqual(payload["multipliers"], [0.5, 0.25])
        self.assertEqual(payload["history_length"], 1)
        self.assertEqual(payload["latest_history_entry"]["constraint_values"], [0.1, 0.2])
        self.assertEqual(
            payload["latest_history_entry"]["solver_constraint_values"],
            [0.3, 0.4],
        )
        self.assertEqual(payload["accepted_hardware_status"]["violations"], ["cc"])
        self.assertIs(payload["accepted_hardware_status"]["success"], False)
        self.assertEqual(payload["trial_hardware_status"]["violations"], ["cs"])
        self.assertIs(payload["trial_hardware_status"]["success"], False)
        self.assertIs(payload["topology_gate_status"]["success"], True)
        self.assertEqual(payload["topology_gate_status"]["survived_lines"], 6)
        self.assertEqual(payload["termination_message"], "still running")
        self.assertEqual(payload["termination_reason"], "subproblem_continue")
        self.assertFalse(payload["inner_optimizer_success"])
        self.assertEqual(payload["inner_optimizer_message"], "iteration limit")
        self.assertFalse(payload["converged_to_tolerances"])
        self.assertTrue(payload["restored_best_feasible"])
        self.assertEqual(
            payload["restored_best_feasible_reason"],
            "final_iterate_infeasible",
        )
        self.assertEqual(payload["final_max_feasibility_violation"], 0.2)
        self.assertEqual(payload["final_stationarity_norm"], 0.15)

    def test_native_alm_target_lane_does_not_call_scipy_minimize(self):
        from simsopt.geo.optimizer_jax import (
            PRIVATE_OPTIMIZER_JAX_VERSION,
            TargetOptimizerContract,
            private_optimizer_runtime_is_supported,
        )

        if not private_optimizer_runtime_is_supported(PRIVATE_OPTIMIZER_JAX_VERSION):
            self.skipTest("Private JAX optimizer runtime is unavailable in this environment.")

        alm_utils = load_alm_utils_module()
        original_minimize = alm_utils.minimize

        def fail_if_called(*args, **kwargs):
            raise AssertionError("SciPy minimize should not be called on the native ALM lane.")

        alm_utils.minimize = fail_if_called
        try:
            settings = alm_utils.ALMSettings(
                max_outer_iterations=1,
                max_subproblem_continuations=0,
                penalty_init=1.0,
                penalty_scale=10.0,
                feasibility_tol=1e-6,
                stationarity_tol=1e-6,
                max_inner_attempts=1,
            )
            inner_options = {
                "maxiter": 25,
                "maxcor": 5,
                "ftol": 1e-12,
                "gtol": 1e-12,
                "maxls": 20,
            }

            def evaluate_problem(x, multipliers, penalty):
                x_arr = np.asarray(x, dtype=float).reshape((-1,))
                multiplier = float(np.asarray(multipliers, dtype=float).reshape((-1,))[0])
                penalty_value = float(penalty)
                base_total = float((x_arr[0] - 1.0) ** 2)
                base_grad = np.asarray([2.0 * (x_arr[0] - 1.0)], dtype=float)
                constraint_value = np.asarray([x_arr[0]], dtype=float)
                positive_shift = max(0.0, multiplier + penalty_value * float(constraint_value[0]))
                total = base_total + 0.5 / penalty_value * (
                    positive_shift**2 - multiplier**2
                )
                grad = base_grad + np.asarray([positive_shift], dtype=float)
                feasibility = np.maximum(constraint_value, 0.0)
                return {
                    "total": float(total),
                    "grad": grad,
                    "constraint_values": constraint_value,
                    "dual_update_values": constraint_value.copy(),
                    "feasibility_values": feasibility,
                    "max_feasibility_violation": float(feasibility[0]),
                }

            def target_inner_value_and_grad(x, multipliers, penalty):
                import jax.numpy as jnp

                x_arr = jnp.asarray(x, dtype=jnp.float64).reshape((-1,))
                multiplier = jnp.asarray(multipliers, dtype=jnp.float64).reshape((-1,))[0]
                penalty_value = jnp.asarray(penalty, dtype=jnp.float64)
                base_total = jnp.square(x_arr[0] - 1.0)
                constraint_value = x_arr[0]
                positive_shift = jnp.maximum(
                    0.0,
                    multiplier + penalty_value * constraint_value,
                )
                total = base_total + 0.5 / penalty_value * (
                    jnp.square(positive_shift) - jnp.square(multiplier)
                )
                grad = jnp.asarray([2.0 * (x_arr[0] - 1.0) + positive_shift], dtype=jnp.float64)
                return total, grad

            result = alm_utils.minimize_alm(
                np.asarray([1.5], dtype=float),
                ["x_upper_bound"],
                evaluate_problem,
                settings,
                inner_options,
                inner_optimizer_contract=TargetOptimizerContract(method="lbfgs-ondevice"),
                target_inner_value_and_grad=target_inner_value_and_grad,
            )
        finally:
            alm_utils.minimize = original_minimize

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "inner_result"))
        self.assertLess(float(np.asarray(result.x, dtype=float)[0]), 1.5)
        self.assertEqual(len(result.multipliers), 1)
        self.assertEqual(len(result.constraint_values), 1)
        self.assertTrue(np.all(np.isfinite(np.asarray(result.multipliers, dtype=float))))
        self.assertTrue(
            np.all(np.isfinite(np.asarray(result.constraint_values, dtype=float)))
        )

    def test_target_alm_requires_native_value_and_grad(self):
        from simsopt.geo.optimizer_jax import TargetOptimizerContract

        alm_utils = load_alm_utils_module()
        settings = alm_utils.ALMSettings(
            max_outer_iterations=1,
            max_subproblem_continuations=0,
            penalty_init=1.0,
            penalty_scale=10.0,
            feasibility_tol=1e-6,
            stationarity_tol=1e-6,
            max_inner_attempts=1,
        )
        inner_options = {
            "maxiter": 10,
            "maxcor": 5,
            "ftol": 1e-12,
            "gtol": 1e-12,
            "maxls": 20,
        }

        def evaluate_problem(x, multipliers, penalty):
            x_arr = np.asarray(x, dtype=float).reshape((-1,))
            signed_constraint = np.asarray([x_arr[0] - 1.0], dtype=float)
            base_value = float(np.dot(x_arr, x_arr))
            base_grad = 2.0 * x_arr
            evaluation = alm_utils.augmented_inequality_objective(
                base_value,
                base_grad,
                signed_constraint,
                [np.asarray([1.0], dtype=float)],
                multipliers,
                penalty,
            )
            evaluation.update(
                {
                    "constraint_names": ["x_upper_bound"],
                    "dual_update_values": signed_constraint.copy(),
                    "constraint_grads": [np.asarray([1.0], dtype=float)],
                    "constraint_activity_tolerances": np.asarray([1.0e-6], dtype=float),
                    "feasibility_values": np.maximum(signed_constraint, 0.0),
                    "hard_signed_constraint_values": signed_constraint.copy(),
                    "hard_violation_values": np.maximum(signed_constraint, 0.0),
                    "surrogate_signed_constraint_values": signed_constraint.copy(),
                    "hard_dual_update_values": signed_constraint.copy(),
                    "max_feasibility_violation": float(
                        np.maximum(signed_constraint, 0.0)[0]
                    ),
                    "metric_grad": base_grad.copy(),
                    "metric_stationarity_norm": float(np.linalg.norm(base_grad)),
                }
            )
            return evaluation

        with self.assertRaisesRegex(
            ValueError,
            "target_inner_value_and_grad",
        ):
            alm_utils.minimize_alm(
                np.asarray([1.5], dtype=float),
                ["x_upper_bound"],
                evaluate_problem,
                settings,
                inner_options,
                inner_optimizer_contract=TargetOptimizerContract(method="lbfgs-ondevice"),
            )

    def test_single_stage_results_surface_keeps_surrogate_alm_aliases(self):
        source = SINGLE_STAGE_MODULE_PATH.read_text()

        self.assertIn('"ALM_PARTIAL_STATE_FILENAME": "alm_state.partial.json"', source)
        self.assertIn('"ALM_CONVERGED": getattr(', source)
        self.assertIn('"ALM_FINAL_MULTIPLIERS": list(alm_result.multipliers)', source)
        self.assertIn(
            '"ALM_FINAL_CONSTRAINT_VALUES": list(alm_result.constraint_values)',
            source,
        )

    def test_single_stage_thresholded_physics_wrapper_pins_live_jax_alm(self):
        module = load_single_stage_thresholded_physics_rerun_module()
        args = make_single_stage_thresholded_physics_rerun_args(
            equilibria_dir="eqdir",
            optimizer_backend="ondevice",
            minimal_artifacts=True,
        )
        command = module.build_single_stage_thresholded_physics_command(args)

        self.assertEqual(command[0], "python")
        self.assertEqual(command[command.index("--backend") + 1], "jax")
        self.assertEqual(
            command[command.index("--stage2-bs-path") + 1],
            str(Path("relative/seed.json").resolve()),
        )
        self.assertEqual(
            command[command.index("--output-root") + 1],
            str(Path("outputs").resolve()),
        )
        self.assertIn("--equilibria-dir", command)
        self.assertEqual(
            command[command.index("--equilibria-dir") + 1],
            str(Path("eqdir").resolve()),
        )
        self.assertIn("--constraint-method", command)
        self.assertEqual(command[command.index("--constraint-method") + 1], "alm")
        self.assertIn("--alm-formulation", command)
        self.assertEqual(
            command[command.index("--alm-formulation") + 1],
            "thresholded_physics",
        )
        self.assertEqual(
            command[command.index("--optimizer-backend") + 1],
            "ondevice",
        )
        self.assertIn("--banana-current-max-A", command)
        self.assertEqual(
            command[command.index("--banana-current-max-A") + 1],
            "16000.0",
        )
        self.assertIn("--length-target", command)
        self.assertEqual(command[command.index("--length-target") + 1], "1.7")
        self.assertIn("--minimal-artifacts", command)

    def test_single_stage_thresholded_physics_wrapper_rejects_init_only_stage2_seed(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
            stage2_results_path = tmpdir_path / "results.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            stage2_results_path.write_text(
                json.dumps(
                    {
                        "PLASMA_SURF_FILENAME": DEFAULT_ALM_WRAPPER_SURFACE,
                        "init_only": True,
                    }
                ),
                encoding="utf-8",
            )

            args = make_single_stage_thresholded_physics_rerun_args(
                stage2_bs_path=str(stage2_bs_path),
                plasma_surf_filename=DEFAULT_ALM_WRAPPER_SURFACE,
            )

            with self.assertRaisesRegex(ValueError, "requires a non-init-only Stage 2 artifact"):
                module.load_validated_stage2_seed_metadata(args)

    def test_single_stage_thresholded_physics_wrapper_upgrades_legacy_stage2_seed_metadata(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = write_stage2_artifact_bundle(
                tmpdir_path,
                results_payload=LEGACY_STAGE2_RESULTS_PAYLOAD,
            )

            args = make_single_stage_thresholded_physics_rerun_args(
                stage2_bs_path=str(stage2_bs_path),
                plasma_surf_filename=DEFAULT_ALM_WRAPPER_SURFACE,
            )
            _, _, stage2_results = module.load_validated_stage2_seed_metadata(args)

        assert_legacy_stage2_fields_upgraded(self, stage2_results)

    def test_single_stage_thresholded_physics_wrapper_dry_run_writes_marker_and_summary(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"

            argv = [
                "run_single_stage_thresholded_physics_alm.py",
                "--dry-run",
                "--plasma-surf-filename",
                DEFAULT_ALM_WRAPPER_SURFACE,
                "--stage2-bs-path",
                str(tmpdir_path / "missing_seed.json"),
                "--output-root",
                str(output_root),
                "--summary-json",
                str(summary_path),
            ]
            old_argv = sys.argv
            try:
                sys.argv = argv
                self.assertEqual(module.main(), 0)
            finally:
                sys.argv = old_argv

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            marker_path = (output_root / "DRY_RUN_ONLY.txt").resolve()
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["backend"], "jax")
            self.assertEqual(summary["output_contract"], "dry_run_summary_only")
            self.assertFalse(summary["contains_solver_outputs"])
            self.assertEqual(summary["dry_run_marker_path"], str(marker_path))
            self.assertTrue(marker_path.exists())
            self.assertIn("dry run only", marker_path.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
