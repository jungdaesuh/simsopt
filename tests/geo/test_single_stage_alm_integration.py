import ast
import hashlib
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from geo.test_basin_hopping import EXPECTED_BASIN_TELEMETRY_FIELDS


SINGLE_STAGE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)
SINGLE_STAGE_CONSTRAINTS_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "single_stage_constraints.py"
)
SINGLE_STAGE_OBJECTIVES_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "single_stage_objectives.py"
)
SINGLE_STAGE_BANANA_CURRENT_MODE_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "single_stage_banana_current_mode.py"
)
STAGE2_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "banana_coil_solver.py"
)
STAGE2_OBJECTIVES_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "stage2_objectives.py"
)
HARDWARE_CONSTRAINT_SCHEMA_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "banana_opt"
    / "hardware_constraint_schema.py"
)
STAGE2_ALM_WRAPPER_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "run_stage2_alm.py"
)
SINGLE_STAGE_THRESHOLDED_PHYSICS_RERUN_MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "single_stage_optimization"
    / "run_single_stage_thresholded_physics_alm.py"
)
DEFAULT_ALM_WRAPPER_SURFACE = "wout_nfp10ginsburg_desc_s024match_iota20.nc"


def write_stage2_results_with_digest(stage2_bs_path: Path, stage2_results: dict) -> Path:
    results_path = stage2_bs_path.with_name("results.json")
    payload = dict(stage2_results)
    payload["STAGE2_BS_SHA256"] = hashlib.sha256(
        stage2_bs_path.read_bytes()
    ).hexdigest()
    results_path.write_text(json.dumps(payload), encoding="utf-8")
    return results_path


def load_alm_utils_module():
    alm_utils_path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "single_stage_optimization"
        / "alm_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"alm_utils_{uuid.uuid4().hex}",
        alm_utils_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_stage2_alm_wrapper_module():
    spec = importlib.util.spec_from_file_location(
        f"run_stage2_alm_{uuid.uuid4().hex}",
        STAGE2_ALM_WRAPPER_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_single_stage_thresholded_physics_rerun_module():
    spec = importlib.util.spec_from_file_location(
        f"run_single_stage_thresholded_physics_alm_{uuid.uuid4().hex}",
        SINGLE_STAGE_THRESHOLDED_PHYSICS_RERUN_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_hardware_constraint_schema_module():
    package_root = str(HARDWARE_CONSTRAINT_SCHEMA_MODULE_PATH.parents[1])
    if package_root not in sys.path:
        sys.path.insert(0, package_root)
    return importlib.import_module("banana_opt.hardware_constraint_schema")


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


def find_assigned_dict(module_path: Path, variable_name: str) -> ast.Dict:
    tree = ast.parse(module_path.read_text(), filename=str(module_path))

    class DictAssignmentVisitor(ast.NodeVisitor):
        def __init__(self):
            self.dict_node = None

        def visit_Assign(self, node):
            if self.dict_node is not None:
                return
            if not isinstance(node.value, ast.Dict):
                return
            if any(
                isinstance(target, ast.Name) and target.id == variable_name
                for target in node.targets
            ):
                self.dict_node = node.value

    visitor = DictAssignmentVisitor()
    visitor.visit(tree)
    if visitor.dict_node is None:
        raise AssertionError(f"Could not find dict assignment for {variable_name}")
    return visitor.dict_node


def find_function_return_dict(module_path: Path, function_name: str) -> ast.Dict:
    tree = ast.parse(module_path.read_text(), filename=str(module_path))

    class ReturnDictVisitor(ast.NodeVisitor):
        def __init__(self):
            self.dict_node = None

        def visit_FunctionDef(self, node):
            if node.name != function_name or self.dict_node is not None:
                return
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and isinstance(child.value, ast.Dict):
                    self.dict_node = child.value
                    return

    visitor = ReturnDictVisitor()
    visitor.visit(tree)
    if visitor.dict_node is None:
        raise AssertionError(f"Could not find return dict for {function_name}")
    return visitor.dict_node


def _make_alm_args(**overrides):
    defaults = {
        "alm_max_outer_iters": 7,
        "alm_max_subproblem_continuations": 9,
        "alm_penalty_init": 2.0,
        "alm_penalty_scale": 3.0,
        "alm_penalty_max": 25.0,
        "alm_feas_tol": 1e-4,
        "alm_stationarity_tol": 2e-4,
        "alm_trust_radius_init": 0.15,
        "alm_trust_radius_min": 1e-3,
        "alm_trust_radius_shrink": 0.4,
        "alm_trust_radius_grow": 1.8,
        "alm_max_inner_attempts": 5,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_stage2_alm_wrapper_args(**overrides):
    defaults = {
        "python_executable": "python",
        "dry_run": False,
        "plasma_surf_filename": DEFAULT_ALM_WRAPPER_SURFACE,
        "profile": "standard_80ka",
        "stage2_spec_json": None,
        "equilibria_dir": None,
        "output_root": "outputs",
        "summary_json": None,
        "stage2_timeout_seconds": 0.0,
        "toroidal_flux": None,
        "cc_threshold": None,
        "curvature_threshold": None,
        "order": None,
        "tf_current_A": None,
        "banana_current_max_A": None,
        "length_target": None,
        "target_lcfs_max_major_radius_m": None,
        "target_lcfs_max_minor_radius_m": None,
        "banana_surf_radius": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_complete_stage2_spec_payload(**overrides):
    payload = {
        "major_radius": 0.976,
        "toroidal_flux": 0.30,
        "length_weight": 0.001,
        "cc_weight": 50.0,
        "cc_threshold": 0.06,
        "curvature_weight": 0.0002,
        "curvature_threshold": 45.0,
        "banana_surf_radius": 0.21,
        "tf_current_A": -8.0e4,
        "order": 3,
        "banana_init_current_A": 1.2e4,
        "banana_current_max_A": 1.5e4,
        "alm_max_outer_iters": 15,
        "alm_penalty_init": 2.0,
        "alm_penalty_scale": 5.0,
        "alm_penalty_max": 5.0e5,
        "basin_hops": 0,
        "basin_stepsize": 0.01,
        "basin_temperature": 1.0,
        "basin_niter_success": 0,
        "basin_seed": None,
        "init_only": False,
    }
    payload.update(overrides)
    return payload


def make_single_stage_thresholded_physics_rerun_args(**overrides):
    defaults = {
        "python_executable": "python",
        "dry_run": False,
        "plasma_surf_filename": DEFAULT_ALM_WRAPPER_SURFACE,
        "stage2_bs_path": "relative/seed.json",
        "output_root": "outputs",
        "equilibria_dir": None,
        "equilibrium_path": None,
        "stage2_seed_surf_path": None,
        "seed_order_upgrade": None,
        "summary_json": None,
        "allow_init_only_stage2_seed": False,
        "single_stage_timeout_seconds": 0.0,
        "nphi": 91,
        "ntheta": 32,
        "mpol": 8,
        "ntor": 6,
        "constraint_weight": 1.0,
        "boozer_I": None,
        "plasma_current_A": None,
        "single_stage_banana_current_mode": "shared",
        "num_tf_coils": 20,
        "banana_surf_radius": None,
        "stage2_seed_tf_current_A": None,
        "maxiter": 300,
        "iota_target": 0.2,
        "vol_target": 0.1,
        "cc_dist": 0.05,
        "cs_dist": 0.02,
        "curvature_threshold": 40.0,
        "hardware_search_mode": "warn",
        "alm_max_outer_iters": 20,
        "alm_max_subproblem_continuations": 4,
        "alm_penalty_init": 1.0,
        "alm_penalty_scale": 10.0,
        "alm_penalty_max": 1.0e8,
        "alm_feas_tol": 1e-4,
        "alm_stationarity_tol": 1e-4,
        "alm_trust_radius_init": 0.05,
        "alm_trust_radius_min": 1e-4,
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
    def test_single_stage_parse_args_exposes_alm_trust_region_controls(self):
        source = SINGLE_STAGE_MODULE_PATH.read_text()

        self.assertIn('"--alm-penalty-max"', source)
        self.assertIn('"ALM_PENALTY_MAX", "1e8"', source)
        self.assertIn('"--alm-trust-radius-init"', source)
        self.assertIn('"ALM_TRUST_RADIUS_INIT", "0.05"', source)
        self.assertIn('"--alm-trust-radius-min"', source)
        self.assertIn('"ALM_TRUST_RADIUS_MIN", "1e-4"', source)
        self.assertIn('"--alm-trust-radius-shrink"', source)
        self.assertIn('"ALM_TRUST_RADIUS_SHRINK", "0.5"', source)
        self.assertIn('"--alm-trust-radius-grow"', source)
        self.assertIn('"ALM_TRUST_RADIUS_GROW", "1.5"', source)
        self.assertIn('"--alm-max-inner-attempts"', source)
        self.assertIn('"ALM_MAX_INNER_ATTEMPTS", "4"', source)
        self.assertIn('"--alm-max-subproblem-continuations"', source)
        self.assertIn('"ALM_MAX_SUBPROBLEM_CONTINUATIONS", "20"', source)
        self.assertIn('"--alm-distance-smoothing"', source)
        self.assertIn('"ALM_DISTANCE_SMOOTHING", "0.005"', source)
        self.assertIn('"--alm-curvature-smoothing"', source)
        self.assertIn('"ALM_CURVATURE_SMOOTHING", "0.05"', source)

    def test_single_stage_parse_args_behavioral_alm_flag_wiring(self):
        """Behaviorally verify ALM argparse flags parse to correct dest/type/value.

        Complements the source-text assertIn checks above by actually calling
        parse_args and inspecting the resulting namespace.
        """
        import argparse as _argparse
        import os as _os

        globals_for_extract = {
            "argparse": _argparse,
            "os": _os,
            "COIL_COIL_MIN_DIST_M": 0.05,
            "COIL_LENGTH_HARD_LIMIT_M": 2.0,
            "COIL_LENGTH_TARGET_M": 1.9,
            "COIL_PLASMA_MIN_DIST_M": 0.015,
            "DEFAULT_EQUILIBRIA_DIR": "/tmp/fake_eq",
            "DEFAULT_SINGLE_STAGE_OUTPUT_ROOT": "/tmp/fake_out",
            "DEFAULT_DATABASE_STAGE2_ROOT": "/tmp/fake_db",
            "DEFAULT_LOCAL_STAGE2_ROOT": "/tmp/fake_local",
            "DEFAULT_HARDWARE_SEARCH_MODE": "hard",
            "DEFAULT_HARDWARE_SEARCH_SOFT_ITERATIONS": 0,
            "DEFAULT_CURVATURE_TRAVERSAL_BAND": 0.0,
            "DEFAULT_CURVATURE_TRAVERSAL_EVAL_BUDGET": 0,
            "DEFAULT_LBFGSB_MAXCOR": 40,
            "DEFAULT_INNER_SURFACE_RATIO": 0.8,
            "DEFAULT_STAGE2_SEEDS_BY_PLASMA": {},
            "FRONTIER_SCALARIZATION_TYPE_WEIGHT_SCHEDULE": "weight_schedule_v1",
            "FRONTIER_SCALARIZATION_TYPE_REFERENCE_POINT": "reference_point_v1",
            "FRONTIER_SCALARIZATION_TYPE_ACHIEVEMENT": "achievement_v1",
            "FRONTIER_SCALARIZATION_TYPE_EPSILON": "epsilon_constraint_v1",
            "SINGLE_SURFACE": "single_surface",
            "EXPERIMENTAL_MULTISURFACE": "experimental_multisurface",
            "PUBLISHED_MULTISURFACE": "published_multisurface",
            "SURFACE_MODE_CHOICES": (
                "single_surface",
                "experimental_multisurface",
                "published_multisurface",
            ),
            "BANANA_CURRENT_HARD_LIMIT_A": 1.6e4,
            "BANANA_CURRENT_MODE_SHARED": "shared",
            "BANANA_CURRENT_MODE_INDEPENDENT": "independent",
            "BANANA_CURRENT_COORDINATE_SCALING_NONE": "none",
            "BANANA_CURRENT_COORDINATE_SCALING_SEED_RELATIVE": "seed-relative",
            "MAX_CURVATURE_INV_M": 100.0,
            "PLASMA_VESSEL_MIN_DIST_M": 0.04,
            "ACCEPT_OFFSPEC_R0_SEED_ENV": "ACCEPT_OFFSPEC_R0_SEED",
            "ACCEPT_OFFSPEC_R0_SEED_HELP": "test-fixture-offspec-help",
            "env_flag": lambda name: False,
            "_DEFAULT_SINGLE_STAGE_SEED_REGIME": "auto",
            "_SINGLE_STAGE_SEED_REGIME_AUTO": "auto",
            "_SINGLE_STAGE_SEED_REGIME_PRESERVE_FIRST": "preserve_first",
            "_SINGLE_STAGE_SEED_REGIME_REPAIR_FIRST": "repair_first",
            "_SINGLE_STAGE_SEED_REGIME_BRIDGE_ONLY": "bridge_only",
            "_SINGLE_STAGE_SEED_REGIME_GLOBAL_SEARCH": "global_search",
        }
        fns = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["add_confinement_surrogate_args", "parse_args"],
            globals_for_extract,
        )
        test_argv = [
            "prog",
            "--alm-penalty-max", "42.0",
            "--alm-trust-radius-init", "0.1",
            "--alm-trust-radius-min", "1e-5",
            "--alm-trust-radius-shrink", "0.3",
            "--alm-trust-radius-grow", "2.0",
            "--alm-max-inner-attempts", "7",
            "--alm-max-subproblem-continuations", "15",
            "--alm-distance-smoothing", "0.01",
            "--alm-curvature-smoothing", "0.1",
        ]
        with patch.object(sys, "argv", test_argv):
            args = fns["parse_args"]()
        self.assertAlmostEqual(args.alm_penalty_max, 42.0)
        self.assertAlmostEqual(args.alm_trust_radius_init, 0.1)
        self.assertAlmostEqual(args.alm_trust_radius_min, 1e-5)
        self.assertAlmostEqual(args.alm_trust_radius_shrink, 0.3)
        self.assertAlmostEqual(args.alm_trust_radius_grow, 2.0)
        self.assertEqual(args.alm_max_inner_attempts, 7)
        self.assertEqual(args.alm_max_subproblem_continuations, 15)
        self.assertAlmostEqual(args.alm_distance_smoothing, 0.01)
        self.assertAlmostEqual(args.alm_curvature_smoothing, 0.1)

    def test_single_stage_parse_args_exposes_thresholded_physics_formulation_controls(self):
        source = SINGLE_STAGE_MODULE_PATH.read_text()

        self.assertIn('"--alm-formulation"', source)
        self.assertIn('"ALM_FORMULATION", "weighted_sum"', source)
        self.assertIn('"--alm-qs-threshold"', source)
        self.assertIn('"ALM_QS_THRESHOLD"', source)
        self.assertIn('"--alm-boozer-threshold"', source)
        self.assertIn('"ALM_BOOZER_THRESHOLD"', source)
        self.assertIn('"--alm-iota-penalty-threshold"', source)
        self.assertIn('"ALM_IOTA_PENALTY_THRESHOLD"', source)
        self.assertIn('"--alm-length-penalty-threshold"', source)
        self.assertIn('"ALM_LENGTH_PENALTY_THRESHOLD"', source)

    def test_single_stage_validate_thresholded_physics_formulation_requires_explicit_thresholds(self):
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["validate_single_stage_alm_formulation_args"],
            {},
        )
        validate_args = functions["validate_single_stage_alm_formulation_args"]
        args = SimpleNamespace(
            alm_formulation="thresholded_physics",
            single_stage_goal_mode="target",
            constraint_method="alm",
            alm_qs_threshold=None,
            alm_boozer_threshold=1e-6,
            alm_iota_penalty_threshold=1e-5,
            alm_length_penalty_threshold=1e-4,
        )

        with self.assertRaisesRegex(ValueError, "--alm-qs-threshold"):
            validate_args(args)

    def test_single_stage_validate_thresholded_physics_formulation_requires_length_threshold(self):
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["validate_single_stage_alm_formulation_args"],
            {},
        )
        validate_args = functions["validate_single_stage_alm_formulation_args"]
        args = SimpleNamespace(
            alm_formulation="thresholded_physics",
            single_stage_goal_mode="target",
            constraint_method="alm",
            alm_qs_threshold=1e-6,
            alm_boozer_threshold=1e-6,
            alm_iota_penalty_threshold=1e-5,
            alm_length_penalty_threshold=None,
        )

        with self.assertRaisesRegex(ValueError, "--alm-length-penalty-threshold"):
            validate_args(args)

    def test_single_stage_validate_thresholded_physics_formulation_rejects_penalty_mode(self):
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["validate_single_stage_alm_formulation_args"],
            {},
        )
        validate_args = functions["validate_single_stage_alm_formulation_args"]
        args = SimpleNamespace(
            alm_formulation="thresholded_physics",
            single_stage_goal_mode="target",
            constraint_method="penalty",
            alm_qs_threshold=1e-6,
            alm_boozer_threshold=1e-6,
            alm_iota_penalty_threshold=1e-5,
            alm_length_penalty_threshold=1e-4,
        )

        with self.assertRaisesRegex(ValueError, "--constraint-method=alm"):
            validate_args(args)

    def test_single_stage_builds_bounded_alm_settings(self):
        alm_utils = load_alm_utils_module()
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["build_single_stage_alm_settings"],
            {"ALMSettings": alm_utils.ALMSettings},
        )
        build_single_stage_alm_settings = functions["build_single_stage_alm_settings"]
        settings = build_single_stage_alm_settings(_make_alm_args())

        self.assertEqual(settings.max_outer_iterations, 7)
        self.assertEqual(settings.max_subproblem_continuations, 9)
        self.assertEqual(settings.penalty_init, 2.0)
        self.assertEqual(settings.penalty_scale, 3.0)
        self.assertEqual(settings.penalty_max, 25.0)
        self.assertEqual(settings.feasibility_tol, 1e-4)
        self.assertEqual(settings.stationarity_tol, 2e-4)
        self.assertEqual(settings.trust_radius_init, 0.15)
        self.assertEqual(settings.trust_radius_min, 1e-3)
        self.assertEqual(settings.trust_radius_shrink, 0.4)
        self.assertEqual(settings.trust_radius_grow, 1.8)
        self.assertEqual(settings.max_inner_attempts, 5)

    def test_single_stage_zero_trust_radius_disables_bounds_in_settings(self):
        alm_utils = load_alm_utils_module()
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["build_single_stage_alm_settings"],
            {"ALMSettings": alm_utils.ALMSettings},
        )
        build_single_stage_alm_settings = functions["build_single_stage_alm_settings"]
        settings = build_single_stage_alm_settings(
            _make_alm_args(alm_trust_radius_init=0.0)
        )

        self.assertIsNone(settings.trust_radius_init)

    def test_single_stage_adaptive_smoothing_counts_normalized_hard_surrogate_gaps(self):
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            [
                "adapt_alm_smoothing_from_history",
                "normalized_hard_surrogate_gap_counts",
                "shrink_alm_smoothing_for_gap_count",
            ],
            {
                "np": np,
                "_ALM_DISTANCE_GAP_CONSTRAINT_NAMES": frozenset(
                    (
                        "coil_coil_spacing",
                        "coil_surface_spacing",
                        "surface_vessel_spacing",
                        "surface_surface_spacing",
                    )
                ),
                "_ALM_CURVATURE_GAP_CONSTRAINT_NAMES": frozenset(
                    ("max_curvature", "poloidal_extent")
                ),
            },
        )
        history_entry = {
            "constraint_names": [
                "surface_surface_spacing",
                "max_curvature",
                "banana_current_upper_bound",
            ],
            "surrogate_minus_hard_normalized_gap": [2.0e-3, 4.0e-3, 1.0],
            "surrogate_hard_sign_mismatch_by_constraint": [False, True, True],
            "effective_feasibility_tolerance": 1.0e-3,
        }

        counts = functions["normalized_hard_surrogate_gap_counts"](history_entry)
        self.assertEqual(counts, {"distance": 1, "curvature": 1})
        adapted = functions["adapt_alm_smoothing_from_history"](
            0.004,
            0.04,
            history_entry,
            distance_smoothing_min=0.001,
            curvature_smoothing_min=0.01,
        )
        self.assertEqual(adapted["gap_counts"], counts)
        self.assertAlmostEqual(adapted["distance_smoothing"], 0.0032)
        self.assertAlmostEqual(adapted["curvature_smoothing"], 0.032)

    def test_single_stage_alm_surface_stack_gate_relaxes_spacing_only_for_solver(self):
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            [
                "single_stage_surface_stack_alm_enabled",
                "surface_stack_search_gate_for_solver",
            ],
            {},
        )
        search_gate = {
            "surface_gap_threshold": 0.02,
            "vessel_gap_threshold": 0.04,
            "enforce_nesting": True,
        }

        solver_gate = functions["surface_stack_search_gate_for_solver"](
            search_gate,
            constraint_method="alm",
            surface_count=2,
        )

        self.assertTrue(
            functions["single_stage_surface_stack_alm_enabled"](2, 0.02)
        )
        self.assertEqual(solver_gate["surface_gap_threshold"], 0.0)
        self.assertEqual(solver_gate["vessel_gap_threshold"], 0.04)
        self.assertTrue(solver_gate["enforce_nesting"])
        self.assertEqual(search_gate["surface_gap_threshold"], 0.02)

    def test_single_stage_source_uses_projected_inequality_alm_and_outer_accept_callback(self):
        source = SINGLE_STAGE_MODULE_PATH.read_text()
        objectives_source = SINGLE_STAGE_OBJECTIVES_MODULE_PATH.read_text()

        self.assertIn("_evaluate_alm_objective_impl(", source)
        self.assertIn(
            "augmented_inequality_objective_fn=augmented_inequality_objective",
            objectives_source,
        )
        self.assertNotIn("augmented_objective(", objectives_source)
        self.assertIn("accepted_callback=callback", source)
        self.assertNotIn("inner_callback=callback", source)
        self.assertIn("history_callback=history_callback", source)
        self.assertIn("single_stage_alm_constraint_names(", source)
        self.assertIn("constraint_blocks=alm_constraint_blocks", source)
        self.assertIn("hard_surrogate_diagnostics=True", source)
        self.assertIn("surface_stack_constraint_fn=_smooth_min_surface_stack_signed_constraint", source)
        self.assertIn("alm_formulation=args.alm_formulation", source)

    def test_hardware_constraint_schema_declares_expected_targets(self):
        schema_module = load_hardware_constraint_schema_module()
        specs = {
            spec.name: spec
            for spec in schema_module.hardware_constraint_schema()
        }

        self.assertEqual(
            set(specs),
            {
                "coil_coil_spacing",
                "coil_surface_spacing",
                "surface_vessel_spacing",
                "surface_surface_spacing",
                "max_curvature",
                "coil_length",
                "poloidal_extent",
                "banana_current",
                "tf_current",
            },
        )
        self.assertEqual(specs["coil_length"].applies_to, frozenset({"alm", "artifact"}))
        self.assertEqual(
            specs["poloidal_extent"].applies_to,
            frozenset({"penalty", "alm", "artifact"}),
        )
        self.assertEqual(specs["surface_surface_spacing"].applies_to, frozenset({"alm"}))
        self.assertEqual(specs["surface_surface_spacing"].alm_block, "surface")
        self.assertEqual(specs["poloidal_extent"].kind, "upper_bound")
        self.assertEqual(
            specs["banana_current"].applies_to,
            frozenset({"penalty", "alm", "artifact"}),
        )
        self.assertEqual(specs["banana_current"].traversal_policy, "forbidden")
        self.assertEqual(specs["tf_current"].applies_to, frozenset({"artifact"}))

    def test_penalty_box_bound_names_follow_forbidden_traversal_policy(self):
        schema_module = load_hardware_constraint_schema_module()

        self.assertEqual(
            schema_module.hardware_constraint_penalty_box_bound_names(
                traversal_policy="forbidden",
            ),
            ("banana_current",),
        )
        self.assertEqual(
            schema_module.resolve_penalty_box_bound_threshold(
                "banana_current",
                requested_threshold=2.0e4,
            ),
            1.6e4,
        )

    def test_traversal_policy_runtime_subset_is_intentionally_penalty_only(self):
        schema_module = load_hardware_constraint_schema_module()

        self.assertEqual(
            schema_module.hardware_constraint_alm_names(
                names=("coil_length", "banana_current"),
            ),
            ("coil_length_upper_bound", "banana_current_upper_bound"),
        )
        self.assertEqual(
            schema_module.hardware_constraint_specs(
                applies_to="artifact",
                traversal_policy="forbidden",
            ),
            (
                schema_module.get_hardware_constraint_spec("banana_current"),
                schema_module.get_hardware_constraint_spec("tf_current"),
            ),
        )

    def test_hardware_constraint_alm_metadata_uses_active_threshold_without_current_cap(self):
        schema_module = load_hardware_constraint_schema_module()

        metadata = schema_module.hardware_constraint_alm_metadata(
            "banana_current_upper_bound",
            threshold_overrides={"banana_current": 2.0e4},
            activity_tolerance=1.0e-3,
            objective_value_kind="hard",
            gradient_value_kind="hard",
            dual_update_value_kind="hard",
            feasibility_value_kind="hard",
        )

        self.assertEqual(metadata.block, "current")
        self.assertEqual(metadata.scale, 2.0e4)
        self.assertEqual(metadata.raw_threshold, 2.0e4)
        self.assertEqual(metadata.activity_tolerance, 1.0e-3)
        self.assertEqual(metadata.objective_value_kind, "hard")
        self.assertEqual(metadata.certification_value_kind, "hard")

    def test_alm_constraint_metadata_payload_preserves_ordered_sidecars(self):
        schema_module = load_hardware_constraint_schema_module()
        metadata_by_name = {
            "coil_coil_spacing": schema_module.hardware_constraint_alm_metadata(
                "coil_coil_spacing",
                threshold_overrides={"coil_coil_spacing": 0.05},
            ),
            "banana_current_upper_bound": schema_module.hardware_constraint_alm_metadata(
                "banana_current_upper_bound",
                threshold_overrides={"banana_current": 16000.0},
                objective_value_kind="hard",
                gradient_value_kind="hard",
                dual_update_value_kind="hard",
                feasibility_value_kind="hard",
            ),
        }

        payload = schema_module.alm_constraint_metadata_payload(
            ("banana_current_upper_bound", "coil_coil_spacing"),
            metadata_by_name,
        )

        self.assertEqual(payload["constraint_blocks"], ["current", "geometry"])
        self.assertEqual(payload["constraint_scales"], [16000.0, 0.05])
        self.assertEqual(payload["objective_value_kinds"], ["hard", "surrogate"])

    def test_hardware_constraint_status_splits_allowed_and_forbidden_traversal(self):
        schema_module = load_hardware_constraint_schema_module()

        status = schema_module.build_hardware_constraint_status(
            {
                "coil_coil_spacing": 0.05,
                "coil_surface_spacing": 0.02,
                "surface_vessel_spacing": 0.04,
                "max_curvature": 40.0,
                "coil_length": 2.1,
                "poloidal_extent": 0.7,
                "banana_current": 1.7e4,
                "tf_current": 9.0e4,
            },
            applies_to="artifact",
            threshold_overrides={
                "coil_length": 2.0,
                "banana_current": 1.6e4,
                "tf_current": 8.0e4,
            },
        )
        allowed_status = status["allowed_traversal_status"]
        forbidden_status = status["forbidden_traversal_status"]

        self.assertFalse(status["success"])
        self.assertEqual(
            list(allowed_status["constraints"]),
            [
                "coil_coil_spacing",
                "coil_surface_spacing",
                "surface_vessel_spacing",
                "max_curvature",
                "coil_length",
                "poloidal_extent",
            ],
        )
        self.assertEqual(
            list(forbidden_status["constraints"]),
            ["banana_current", "tf_current"],
        )
        self.assertEqual(
            allowed_status["violations"],
            ["coil_length 2.100000 exceeds threshold 2.000000"],
        )
        self.assertEqual(
            forbidden_status["violations"],
            [
                "|banana_current| 17000.000000 exceeds threshold 16000.000000",
                "|tf_current| 90000.000000 exceeds threshold 80000.000000",
            ],
        )

    def test_penalty_traversal_policy_execution_is_centralized_in_shared_helper(self):
        single_stage_source = SINGLE_STAGE_MODULE_PATH.read_text()
        single_stage_helper_source = (
            SINGLE_STAGE_BANANA_CURRENT_MODE_MODULE_PATH.read_text()
        )
        stage2_source = STAGE2_MODULE_PATH.read_text()

        self.assertIn(
            "apply_single_stage_penalty_banana_current_bounds(",
            single_stage_source,
        )
        self.assertIn(
            "apply_penalty_traversal_forbidden_box_bounds(",
            stage2_source,
        )
        self.assertNotIn("apply_banana_current_upper_bound(", single_stage_source)
        self.assertNotIn("banana_current_exceeds_limit(", single_stage_source)
        self.assertNotIn("apply_banana_current_upper_bound(", stage2_source)
        self.assertNotIn("banana_current_exceeds_limit(", stage2_source)
        self.assertIn(
            "apply_banana_current_upper_bound(",
            single_stage_helper_source,
        )
        self.assertIn(
            "banana_current_exceeds_limit(",
            single_stage_helper_source,
        )

    def test_single_stage_alm_constraint_names_follow_shared_schema(self):
        schema_module = load_hardware_constraint_schema_module()
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["single_stage_alm_constraint_blocks", "single_stage_alm_constraint_names"],
            {
                "hardware_constraint_alm_metadata": schema_module.hardware_constraint_alm_metadata,
                "hardware_constraint_alm_names": schema_module.hardware_constraint_alm_names,
                "SINGLE_STAGE_THRESHOLDED_PHYSICS_CONSTRAINT_NAMES": (
                    "qs_error",
                    "boozer_residual",
                    "iota_penalty",
                    "length_penalty",
                ),
            },
        )

        constraint_names = functions["single_stage_alm_constraint_names"](
            alm_formulation="weighted_sum",
            include_surface_surface=True,
        )
        self.assertEqual(
            constraint_names,
            [
                "coil_coil_spacing",
                "coil_surface_spacing",
                "surface_vessel_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "poloidal_extent",
                "banana_current_upper_bound",
            ],
        )
        stacked_constraint_names = functions["single_stage_alm_constraint_names"](
            alm_formulation="thresholded_physics",
            include_surface_surface=False,
            include_surface_stack=True,
        )
        self.assertIn("surface_surface_spacing", stacked_constraint_names)
        self.assertEqual(
            functions["single_stage_alm_constraint_blocks"](
                (
                    "surface_surface_spacing",
                    "banana_current_0_upper_bound",
                    "qs_error",
                )
            ),
            ("surface", "current", "physics"),
        )

    def test_stage2_alm_constraint_names_follow_shared_schema(self):
        schema_module = load_hardware_constraint_schema_module()
        functions = extract_functions(
            STAGE2_MODULE_PATH,
            ["stage2_alm_constraint_names"],
            {
                "hardware_constraint_alm_names": schema_module.hardware_constraint_alm_names,
            },
        )

        constraint_names = functions["stage2_alm_constraint_names"](
            include_coil_surface=True,
        )
        self.assertEqual(
            constraint_names,
            (
                "coil_coil_spacing",
                "coil_surface_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "banana_current_upper_bound",
            ),
        )

    def test_single_stage_partial_alm_state_payload_serializes_numpy_fields(self):
        from pathlib import PurePath

        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            [
                "_jsonable_value",
                "build_single_stage_alm_partial_state",
            ],
            {"np": np, "PurePath": PurePath},
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

    def test_stage2_alm_wrapper_requires_profile_or_spec_json(self):
        module = load_stage2_alm_wrapper_module()

        with patch.object(
            sys,
            "argv",
            [
                "run_stage2_alm.py",
                "--plasma-surf-filename",
                DEFAULT_ALM_WRAPPER_SURFACE,
            ],
        ):
            with self.assertRaises(SystemExit) as excinfo:
                module.parse_args()

        self.assertEqual(excinfo.exception.code, 2)

    def test_stage2_alm_wrapper_rejects_soft_iota_mode_at_parse_time(self):
        module = load_stage2_alm_wrapper_module()

        with self.assertRaises(SystemExit) as excinfo:
            module.parse_args(
                [
                    "--plasma-surf-filename",
                    DEFAULT_ALM_WRAPPER_SURFACE,
                    "--profile",
                    "standard_80ka",
                    "--stage2-iota-mode",
                    "soft",
                ]
            )

        self.assertEqual(excinfo.exception.code, 2)

    def test_stage2_alm_wrapper_does_not_expose_soft_iota_weight_flag(self):
        module = load_stage2_alm_wrapper_module()

        with self.assertRaises(SystemExit) as excinfo:
            module.parse_args(
                [
                    "--plasma-surf-filename",
                    DEFAULT_ALM_WRAPPER_SURFACE,
                    "--profile",
                    "standard_80ka",
                    "--stage2-iota-weight",
                    "3.0",
                ]
            )

        self.assertEqual(excinfo.exception.code, 2)

    def test_stage2_alm_wrapper_pins_alm_and_resolves_cli_paths(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args(
            equilibria_dir="eqdir",
            tf_current_A=-7.5e4,
            toroidal_flux=0.37,
            cc_threshold=0.07,
        )
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)
        command = module.build_stage2_command(config, python_executable=args.python_executable)

        self.assertEqual(resolved_spec_source, "profile:standard_80ka")
        self.assertEqual(config.constraint_method, "alm")
        self.assertEqual(config.tf_current_A, -7.5e4)
        self.assertEqual(config.toroidal_flux, 0.37)
        self.assertEqual(config.cc_threshold, 0.07)
        self.assertEqual(config.output_root, Path("outputs").resolve())
        self.assertEqual(config.equilibria_dir, str(Path("eqdir").resolve()))
        self.assertEqual(config.finite_current_mode, "wataru_proxy_field")
        self.assertIn("--constraint-method", command)
        self.assertEqual(command[command.index("--constraint-method") + 1], "alm")
        self.assertIn("--finite-current-mode", command)
        self.assertEqual(
            command[command.index("--finite-current-mode") + 1],
            "wataru_proxy_field",
        )
        self.assertEqual(
            command[command.index("--output-root") + 1],
            str(Path("outputs").resolve()),
        )
        self.assertEqual(
            command[command.index("--equilibria-dir") + 1],
            str(Path("eqdir").resolve()),
        )
        self.assertIn("--alm-max-outer-iters", command)
        self.assertEqual(command[command.index("--alm-max-outer-iters") + 1], "10")
        self.assertIn("--alm-penalty-init", command)
        self.assertEqual(command[command.index("--alm-penalty-init") + 1], "1.0")
        self.assertIn("--alm-penalty-scale", command)
        self.assertEqual(command[command.index("--alm-penalty-scale") + 1], "10.0")
        self.assertIn("--alm-penalty-max", command)
        self.assertEqual(command[command.index("--alm-penalty-max") + 1], "100000000.0")
        self.assertIn("--banana-current-max-A", command)
        self.assertEqual(command[command.index("--banana-current-max-A") + 1], "16000.0")
        self.assertEqual(command[command.index("--toroidal-flux") + 1], "0.37")

    def test_stage2_alm_wrapper_command_threads_constraint_metadata_flags(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args(cc_threshold=0.06)
        resolved_spec, _ = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)

        command = module.build_stage2_command(
            config,
            constraint_profile_label="profile:standard_80ka",
            constraint_override_reason="cli:cc_threshold",
            python_executable=args.python_executable,
        )

        self.assertIn("--constraint-profile-label", command)
        self.assertEqual(
            command[command.index("--constraint-profile-label") + 1],
            "profile:standard_80ka",
        )
        self.assertIn("--constraint-override-reason", command)
        self.assertEqual(
            command[command.index("--constraint-override-reason") + 1],
            "cli:cc_threshold",
        )

    def test_stage2_alm_wrapper_standard_profile_matches_hardware_baseline(self):
        module = load_stage2_alm_wrapper_module()

        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(
            make_stage2_alm_wrapper_args()
        )

        self.assertEqual(resolved_spec_source, "profile:standard_80ka")
        self.assertEqual(resolved_spec["tf_current_A"], -8.0e4)
        self.assertEqual(resolved_spec["cc_threshold"], 0.05)
        self.assertEqual(resolved_spec["curvature_threshold"], 100.0)
        self.assertEqual(resolved_spec["banana_surf_radius"], 0.21)
        self.assertEqual(resolved_spec["finite_current_mode"], "wataru_proxy_field")

    def test_stage2_alm_wrapper_rejects_tf_current_above_hard_limit(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args(tf_current_A=-8.5e4)
        resolved_spec, _ = module.resolve_stage2_spec_payload(args)

        with self.assertRaisesRegex(ValueError, "TF coil current must be negative"):
            module.build_stage2_alm_config(args, resolved_spec=resolved_spec)

    def test_stage2_alm_wrapper_cli_overrides_banana_current_and_surface_radius(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args(
            banana_current_max_A=1.5e4,
            banana_surf_radius=0.22,
        )
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)
        metadata = module.build_stage2_constraint_artifacts(
            args=args,
            config=config,
            source_label=resolved_spec_source,
        )
        command = module.build_stage2_command(
            config,
            python_executable=args.python_executable,
        )

        self.assertEqual(resolved_spec["banana_current_max_A"], 1.5e4)
        self.assertEqual(resolved_spec["banana_surf_radius"], 0.22)
        self.assertEqual(config.banana_current_max_A, 1.5e4)
        self.assertEqual(config.banana_surf_radius, 0.22)
        self.assertEqual(metadata["EFFECTIVE_VALUES"]["BANANA_CURRENT_MAX_A"], 1.5e4)
        self.assertEqual(metadata["EFFECTIVE_VALUES"]["banana_surf_radius"], 0.22)
        self.assertEqual(
            metadata["OVERRIDE_REASON"],
            "cli:banana_current_max_A,banana_surf_radius",
        )
        self.assertIn("--banana-current-max-A", command)
        self.assertEqual(
            command[command.index("--banana-current-max-A") + 1],
            "15000.0",
        )
        self.assertIn("--banana-surf-radius", command)
        self.assertEqual(
            command[command.index("--banana-surf-radius") + 1],
            "0.22",
        )

    def test_stage2_alm_wrapper_command_adds_offspec_engineering_flag(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args()
        args.allow_offspec_engineering_constraints = True
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(
            args,
            resolved_spec={**resolved_spec, "length_target": 3.0},
        )
        metadata = module.build_stage2_constraint_artifacts(
            args=args,
            config=config,
            source_label=resolved_spec_source,
        )
        command = module.build_stage2_command(
            config,
            python_executable=args.python_executable,
        )

        self.assertIn("--allow-offspec-engineering-constraints", command)
        self.assertEqual(config.length_target, 3.0)
        self.assertEqual(metadata["EFFECTIVE_VALUES"]["COIL_LENGTH_TARGET_M"], 3.0)
        self.assertEqual(
            metadata["OVERRIDE_REASON"],
            "allow_offspec_engineering_constraints",
        )

    def test_stage2_alm_wrapper_metadata_keeps_cli_override_and_offspec_tag(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args(cc_threshold=0.06)
        args.allow_offspec_engineering_constraints = True
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(
            args,
            resolved_spec={**resolved_spec, "length_target": 3.0},
        )

        metadata = module.build_stage2_constraint_artifacts(
            args=args,
            config=config,
            source_label=resolved_spec_source,
        )

        self.assertEqual(
            metadata["OVERRIDE_REASON"],
            "cli:cc_threshold;allow_offspec_engineering_constraints",
        )

    def test_stage2_alm_wrapper_spec_json_must_be_complete(self):
        module = load_stage2_alm_wrapper_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "stage2_spec.json"
            spec_path.write_text(json.dumps({"major_radius": 0.976}), encoding="utf-8")
            args = make_stage2_alm_wrapper_args(
                profile=None,
                stage2_spec_json=str(spec_path),
            )

            with self.assertRaisesRegex(ValueError, "must define all required keys"):
                module.resolve_stage2_spec_payload(args)

    def test_stage2_alm_wrapper_spec_json_valid_complete_spec_resolves(self):
        module = load_stage2_alm_wrapper_module()
        complete_spec = make_complete_stage2_spec_payload(
            major_radius=1.0,
            banana_surf_radius=0.25,
            tf_current_A=-7.5e4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "stage2_spec.json"
            spec_path.write_text(json.dumps(complete_spec), encoding="utf-8")
            args = make_stage2_alm_wrapper_args(
                profile=None,
                stage2_spec_json=str(spec_path),
            )

            resolved_spec, source_label = module.resolve_stage2_spec_payload(args)

        self.assertTrue(source_label.startswith("json:"))
        self.assertEqual(resolved_spec["major_radius"], 1.0)
        self.assertEqual(resolved_spec["tf_current_A"], -7.5e4)
        self.assertEqual(resolved_spec["order"], 3)
        self.assertEqual(resolved_spec["banana_current_max_A"], 1.5e4)

    def test_stage2_alm_wrapper_spec_json_accepts_length_target_and_lcfs_ceiling_fields(self):
        module = load_stage2_alm_wrapper_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "stage2_spec.json"
            spec_path.write_text(
                json.dumps(
                    make_complete_stage2_spec_payload(
                        length_target=1.6,
                        target_lcfs_max_major_radius_m=0.91,
                        target_lcfs_max_minor_radius_m=0.14,
                    )
                ),
                encoding="utf-8",
            )
            args = make_stage2_alm_wrapper_args(
                profile=None,
                stage2_spec_json=str(spec_path),
            )

            resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
            config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)
            metadata = module.build_stage2_constraint_artifacts(
                args=args,
                config=config,
                source_label=resolved_spec_source,
            )
            command = module.build_stage2_command(
                config,
                python_executable=args.python_executable,
            )

        self.assertEqual(config.length_target, 1.6)
        self.assertEqual(config.target_lcfs_max_major_radius_m, 0.91)
        self.assertEqual(config.target_lcfs_max_minor_radius_m, 0.14)
        self.assertEqual(metadata["EFFECTIVE_VALUES"]["COIL_LENGTH_TARGET_M"], 1.6)
        self.assertEqual(metadata["EFFECTIVE_VALUES"]["TARGET_LCFS_MAX_MAJOR_RADIUS_M"], 0.91)
        self.assertEqual(metadata["EFFECTIVE_VALUES"]["TARGET_LCFS_MAX_MINOR_RADIUS_M"], 0.14)
        self.assertIn("--length-target", command)
        self.assertEqual(command[command.index("--length-target") + 1], "1.6")
        self.assertIn("--target-lcfs-max-major-radius-m", command)
        self.assertEqual(
            command[command.index("--target-lcfs-max-major-radius-m") + 1],
            "0.91",
        )
        self.assertIn("--target-lcfs-max-minor-radius-m", command)
        self.assertEqual(
            command[command.index("--target-lcfs-max-minor-radius-m") + 1],
            "0.14",
        )

    def test_stage2_alm_wrapper_summary_includes_resolved_config(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args()
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)
        command = module.build_stage2_command(config, python_executable=args.python_executable)

        summary = module.build_summary(
            args,
            config=config,
            resolved_spec_source=resolved_spec_source,
            command=command,
            artifact_path=Path("/tmp/stage2/biot_savart_opt.json"),
            artifact_reused=False,
        )

        self.assertEqual(summary["resolved_spec_source"], "profile:standard_80ka")
        self.assertIn("resolved_stage2_config", summary)
        self.assertEqual(summary["resolved_stage2_config"]["constraint_method"], "alm")
        self.assertEqual(summary["resolved_stage2_config"]["alm_penalty_max"], 1.0e8)
        self.assertEqual(
            summary["resolved_stage2_config"]["alm_max_subproblem_continuations"],
            20,
        )
        self.assertEqual(summary["resolved_stage2_config"]["alm_distance_smoothing"], 0.005)
        self.assertEqual(summary["resolved_stage2_config"]["alm_curvature_smoothing"], 0.25)
        self.assertEqual(summary["resolved_stage2_config"]["curvature_threshold"], 100.0)
        self.assertEqual(summary["resolved_stage2_config"]["banana_surf_radius"], 0.21)
        self.assertEqual(
            summary["resolved_stage2_config"]["finite_current_mode"],
            "wataru_proxy_field",
        )
        self.assertEqual(summary["resolved_stage2_config"]["output_root"], str(Path("outputs").resolve()))
        self.assertEqual(
            summary["fixed_stage2_hardware_contract"],
            {
                "COIL_PLASMA_MIN_DIST_M": 0.015,
                "PLASMA_VESSEL_MIN_DIST_M": 0.04,
            },
        )
        self.assertEqual(summary["output_contract"], "materialized_stage2_artifact")
        self.assertFalse(summary["contains_solver_outputs"])

    def test_stage2_alm_wrapper_constraint_metadata_uses_effective_clamped_values(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args(
            cc_threshold=0.01,
            curvature_threshold=50.0,
        )
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)

        metadata = module.build_stage2_constraint_artifacts(
            args=args,
            config=config,
            source_label=resolved_spec_source,
        )

        self.assertEqual(metadata["EFFECTIVE_VALUES"]["CC_THRESHOLD"], 0.05)
        self.assertEqual(metadata["EFFECTIVE_VALUES"]["CURVATURE_THRESHOLD"], 50.0)
        self.assertEqual(
            metadata["EFFECTIVE_VALUES"]["VACUUM_VESSEL_MAJOR_RADIUS_M"],
            0.976,
        )

    def test_stage2_alm_wrapper_dry_run_writes_explicit_marker(self):
        module = load_stage2_alm_wrapper_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"

            with patch.object(
                sys,
                "argv",
                [
                    "run_stage2_alm.py",
                    "--dry-run",
                    "--plasma-surf-filename",
                    DEFAULT_ALM_WRAPPER_SURFACE,
                    "--profile",
                    "standard_80ka",
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                ],
            ):
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            marker_path = (output_root / "DRY_RUN_ONLY.txt").resolve()
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["output_contract"], "dry_run_summary_only")
            self.assertFalse(summary["contains_solver_outputs"])
            self.assertEqual(summary["dry_run_marker_path"], str(marker_path))
            self.assertTrue(marker_path.exists())
            self.assertIn("dry run only", marker_path.read_text(encoding="utf-8").lower())

    def test_stage2_alm_wrapper_expected_metadata_includes_basin_identity(self):
        module = load_stage2_alm_wrapper_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "stage2_spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "major_radius": 0.976,
                        "toroidal_flux": 0.24,
                        "length_weight": 5.0e-4,
                        "cc_weight": 100.0,
                        "cc_threshold": 0.05,
                        "curvature_weight": 1.0e-4,
                        "curvature_threshold": 40.0,
                        "banana_surf_radius": 0.22,
                        "tf_current_A": -8.0e4,
                        "order": 2,
                        "banana_init_current_A": 1.0e4,
                        "banana_current_max_A": 1.6e4,
                        "alm_max_outer_iters": 10,
                        "alm_penalty_init": 1.0,
                        "alm_penalty_scale": 10.0,
                        "alm_penalty_max": 1.0e8,
                        "basin_hops": 3,
                        "basin_stepsize": 0.01,
                        "basin_temperature": 2.5,
                        "basin_niter_success": 8,
                        "basin_seed": 11,
                        "init_only": False,
                    }
                ),
                encoding="utf-8",
            )
            args = make_stage2_alm_wrapper_args(
                profile=None,
                stage2_spec_json=str(spec_path),
            )
            resolved_spec, _ = module.resolve_stage2_spec_payload(args)
            config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)

        metadata = module._expected_stage2_artifact_metadata(config)

        self.assertEqual(metadata["basin_hops"], 3)
        self.assertEqual(metadata["basin_stepsize"], 0.01)
        self.assertEqual(metadata["basin_temperature"], 2.5)
        self.assertEqual(metadata["basin_niter_success"], 8)
        self.assertEqual(metadata["basin_seed"], 11)
        self.assertEqual(metadata["COIL_PLASMA_MIN_DIST_M"], 0.015)
        self.assertEqual(metadata["PLASMA_VESSEL_MIN_DIST_M"], 0.04)
        self.assertEqual(metadata["LENGTH_TARGET"], 1.9)

    def test_stage2_alm_wrapper_load_validated_artifact_rejects_legacy_contract_metadata(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args()
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)
        constraint_metadata = module.build_stage2_constraint_artifacts(
            args=args,
            config=config,
            source_label=resolved_spec_source,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "biot_savart_opt.json"
            artifact_path.write_text("{}", encoding="utf-8")

            legacy_results = module._expected_stage2_artifact_metadata(config)
            legacy_results.pop("COIL_PLASMA_MIN_DIST_M")
            legacy_results.pop("PLASMA_VESSEL_MIN_DIST_M")
            legacy_results.pop("LENGTH_TARGET")
            legacy_results.pop("ALM_DISTANCE_SMOOTHING")
            legacy_results.pop("ALM_CURVATURE_SMOOTHING")
            write_stage2_results_with_digest(artifact_path, legacy_results)

            with patch.object(module, "resolve_stage2_artifact_path", return_value=artifact_path):
                with self.assertRaisesRegex(ValueError, "legacy constraint contract schema"):
                    module.load_validated_stage2_artifact(
                        config,
                        constraint_metadata=constraint_metadata,
                    )

    def test_stage2_alm_wrapper_load_validated_artifact_rejects_mismatched_constraint_metadata(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args(cc_threshold=0.06)
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)
        constraint_metadata = module.build_stage2_constraint_artifacts(
            args=args,
            config=config,
            source_label=resolved_spec_source,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "biot_savart_opt.json"
            artifact_path.write_text("{}", encoding="utf-8")
            stage2_results = module._expected_stage2_artifact_metadata(config)
            stage2_results.update(constraint_metadata)
            stage2_results["CONTRACT_HASH"] = "mismatched"
            write_stage2_results_with_digest(artifact_path, stage2_results)

            with patch.object(module, "resolve_stage2_artifact_path", return_value=artifact_path):
                with self.assertRaisesRegex(ValueError, "CONTRACT_HASH"):
                    module.load_validated_stage2_artifact(
                        config,
                        constraint_metadata=constraint_metadata,
                    )

    def test_stage2_alm_wrapper_load_validated_artifact_allows_equivalent_spec_json_source_labels(self):
        module = load_stage2_alm_wrapper_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_payload = make_complete_stage2_spec_payload(
                length_target=1.6,
                target_lcfs_max_major_radius_m=0.91,
                target_lcfs_max_minor_radius_m=0.14,
            )
            spec_path_a = Path(tmpdir) / "a.json"
            spec_path_b = Path(tmpdir) / "b.json"
            spec_path_a.write_text(json.dumps(spec_payload), encoding="utf-8")
            spec_path_b.write_text(json.dumps(spec_payload), encoding="utf-8")

            args_a = make_stage2_alm_wrapper_args(
                profile=None,
                stage2_spec_json=str(spec_path_a),
            )
            args_b = make_stage2_alm_wrapper_args(
                profile=None,
                stage2_spec_json=str(spec_path_b),
            )
            resolved_spec_a, resolved_spec_source_a = module.resolve_stage2_spec_payload(args_a)
            resolved_spec_b, resolved_spec_source_b = module.resolve_stage2_spec_payload(args_b)
            config = module.build_stage2_alm_config(args_a, resolved_spec=resolved_spec_a)
            constraint_metadata_a = module.build_stage2_constraint_artifacts(
                args=args_a,
                config=config,
                source_label=resolved_spec_source_a,
            )
            constraint_metadata_b = module.build_stage2_constraint_artifacts(
                args=args_b,
                config=config,
                source_label=resolved_spec_source_b,
            )

            artifact_path = Path(tmpdir) / "biot_savart_opt.json"
            results_path = artifact_path.with_name("results.json")
            artifact_path.write_text("{}", encoding="utf-8")
            stage2_results = module._expected_stage2_artifact_metadata(config)
            stage2_results.update(constraint_metadata_a)
            write_stage2_results_with_digest(artifact_path, stage2_results)

            with patch.object(module, "resolve_stage2_artifact_path", return_value=artifact_path):
                loaded_results_path, loaded_results = module.load_validated_stage2_artifact(
                    config,
                    constraint_metadata=constraint_metadata_b,
                )

        self.assertEqual(loaded_results_path, results_path)
        self.assertEqual(loaded_results["CONTRACT_HASH"], constraint_metadata_a["CONTRACT_HASH"])

    def test_stage2_alm_wrapper_load_validated_artifact_allows_no_op_cli_override_reason_drift(self):
        module = load_stage2_alm_wrapper_module()
        args_default = make_stage2_alm_wrapper_args()
        args_no_op = make_stage2_alm_wrapper_args(cc_threshold=0.05)
        resolved_spec_default, resolved_spec_source_default = module.resolve_stage2_spec_payload(
            args_default
        )
        config = module.build_stage2_alm_config(
            args_default,
            resolved_spec=resolved_spec_default,
        )
        constraint_metadata_default = module.build_stage2_constraint_artifacts(
            args=args_default,
            config=config,
            source_label=resolved_spec_source_default,
        )
        constraint_metadata_no_op = module.build_stage2_constraint_artifacts(
            args=args_no_op,
            config=config,
            source_label=resolved_spec_source_default,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "biot_savart_opt.json"
            results_path = artifact_path.with_name("results.json")
            artifact_path.write_text("{}", encoding="utf-8")
            stage2_results = module._expected_stage2_artifact_metadata(config)
            stage2_results.update(constraint_metadata_default)
            write_stage2_results_with_digest(artifact_path, stage2_results)

            with patch.object(module, "resolve_stage2_artifact_path", return_value=artifact_path):
                loaded_results_path, loaded_results = module.load_validated_stage2_artifact(
                    config,
                    constraint_metadata=constraint_metadata_no_op,
                )

        self.assertEqual(loaded_results_path, results_path)
        self.assertIsNone(loaded_results["OVERRIDE_REASON"])

    def test_stage2_alm_wrapper_spec_json_backfills_optional_alm_solver_keys(self):
        module = load_stage2_alm_wrapper_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            spec_path = Path(tmpdir) / "stage2_spec.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "major_radius": 0.976,
                        "toroidal_flux": 0.24,
                        "length_weight": 5.0e-4,
                        "cc_weight": 100.0,
                        "cc_threshold": 0.05,
                        "curvature_weight": 1.0e-4,
                        "curvature_threshold": 40.0,
                        "banana_surf_radius": 0.22,
                        "tf_current_A": -8.0e4,
                        "order": 2,
                        "banana_init_current_A": 1.0e4,
                        "banana_current_max_A": 1.6e4,
                        "alm_max_outer_iters": 10,
                        "alm_penalty_init": 1.0,
                        "alm_penalty_scale": 10.0,
                        "alm_penalty_max": 1.0e8,
                        "basin_hops": 0,
                        "basin_stepsize": 0.01,
                        "basin_temperature": 1.0,
                        "basin_niter_success": 0,
                        "basin_seed": None,
                        "init_only": False,
                    }
                ),
                encoding="utf-8",
            )
            args = make_stage2_alm_wrapper_args(
                profile=None,
                stage2_spec_json=str(spec_path),
            )

            resolved_spec, _ = module.resolve_stage2_spec_payload(args)

        self.assertEqual(resolved_spec["alm_max_subproblem_continuations"], 20)
        self.assertEqual(resolved_spec["alm_feas_tol"], 1.0e-6)
        self.assertEqual(resolved_spec["alm_stationarity_tol"], 1.0e-6)
        self.assertEqual(resolved_spec["alm_distance_smoothing"], 0.005)
        self.assertEqual(resolved_spec["alm_curvature_smoothing"], 0.25)

    def test_stage2_alm_wrapper_summary_includes_explicit_clearance_results(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args()
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)
        command = module.build_stage2_command(config, python_executable=args.python_executable)

        summary = module.build_summary(
            args,
            config=config,
            resolved_spec_source=resolved_spec_source,
            command=command,
            artifact_path=Path("/tmp/stage2/biot_savart_opt.json"),
            artifact_reused=True,
            stage2_results_path=Path("/tmp/stage2/results.json"),
            stage2_results={
                "TERMINATION_MESSAGE": "done",
                "OPTIMIZER_SUCCESS": True,
                "ALM_OUTER_ITERATIONS": 4,
                "ALM_FINAL_PENALTY": 25.0,
                "CURVE_CURVE_MIN_DIST": 0.07,
                "MAX_CURVATURE": 91.0,
                "COIL_LENGTH": 1.69,
                "FIELD_ERROR": 2.0e-4,
                "HARDWARE_CONSTRAINTS_OK": True,
                "CURVE_SURFACE_MIN_DIST": 0.017,
                "COIL_PLASMA_MIN_DIST_M": 0.015,
                "SURFACE_VESSEL_MIN_DIST": 0.041,
                "PLASMA_VESSEL_MIN_DIST_M": 0.04,
            },
        )

        self.assertEqual(summary["coil_plasma_min_dist"], 0.017)
        self.assertEqual(summary["coil_plasma_threshold"], 0.015)
        self.assertEqual(summary["plasma_vessel_min_dist"], 0.041)
        self.assertEqual(summary["plasma_vessel_threshold"], 0.04)

    def test_stage2_alm_wrapper_summary_prefers_loaded_constraint_metadata(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args(cc_threshold=0.06)
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)
        command = module.build_stage2_command(config, python_executable=args.python_executable)
        constraint_metadata = module.build_stage2_constraint_artifacts(
            args=args,
            config=config,
            source_label=resolved_spec_source,
        )
        loaded_constraint_metadata = {
            "CONSTRAINT_PROFILE": "artifact:reused",
            "EFFECTIVE_VALUES": {"TF_CURRENT_A": -75000.0},
            "OVERRIDE_REASON": "artifact:reused",
            "CONTRACT_HASH": "artifact-hash",
            "CONTRACT_SCHEMA_VERSION": 1,
        }

        summary = module.build_summary(
            args,
            config=config,
            resolved_spec_source=resolved_spec_source,
            command=command,
            artifact_path=Path("/tmp/stage2/biot_savart_opt.json"),
            artifact_reused=True,
            stage2_results_path=Path("/tmp/stage2/results.json"),
            stage2_results={
                "TERMINATION_MESSAGE": "done",
                "OPTIMIZER_SUCCESS": True,
                **loaded_constraint_metadata,
            },
            constraint_metadata=constraint_metadata,
        )

        self.assertEqual(summary["CONSTRAINT_PROFILE"], "artifact:reused")
        self.assertEqual(summary["OVERRIDE_REASON"], "artifact:reused")
        self.assertEqual(summary["CONTRACT_HASH"], "artifact-hash")
        self.assertEqual(summary["CONTRACT_SCHEMA_VERSION"], 1)
        self.assertEqual(summary["EFFECTIVE_VALUES"], {"TF_CURRENT_A": -75000.0})

    def test_stage2_alm_wrapper_summary_clears_wrapper_hash_for_legacy_constraint_metadata(self):
        module = load_stage2_alm_wrapper_module()
        args = make_stage2_alm_wrapper_args(cc_threshold=0.06)
        resolved_spec, resolved_spec_source = module.resolve_stage2_spec_payload(args)
        config = module.build_stage2_alm_config(args, resolved_spec=resolved_spec)
        command = module.build_stage2_command(config, python_executable=args.python_executable)
        constraint_metadata = module.build_stage2_constraint_artifacts(
            args=args,
            config=config,
            source_label=resolved_spec_source,
        )

        summary = module.build_summary(
            args,
            config=config,
            resolved_spec_source=resolved_spec_source,
            command=command,
            artifact_path=Path("/tmp/stage2/biot_savart_opt.json"),
            artifact_reused=True,
            stage2_results_path=Path("/tmp/stage2/results.json"),
            stage2_results={
                "TERMINATION_MESSAGE": "done",
                "OPTIMIZER_SUCCESS": True,
                "CONTRACT_SCHEMA_VERSION": 0,
                "CONSTRAINT_PROFILE": None,
                "EFFECTIVE_VALUES": None,
                "OVERRIDE_REASON": None,
            },
            constraint_metadata=constraint_metadata,
        )

        self.assertIsNone(summary["CONTRACT_HASH"])
        self.assertEqual(summary["CONTRACT_SCHEMA_VERSION"], 0)
        self.assertIsNone(summary["CONSTRAINT_PROFILE"])
        self.assertIsNone(summary["EFFECTIVE_VALUES"])
        self.assertIsNone(summary["OVERRIDE_REASON"])

    def test_stage2_alm_wrapper_normalizes_basin_seed_against_basin_hops(self):
        module = load_stage2_alm_wrapper_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            disabled_spec_path = Path(tmpdir) / "stage2_spec_disabled.json"
            disabled_spec_path.write_text(
                json.dumps(
                    {
                        "major_radius": 0.976,
                        "toroidal_flux": 0.24,
                        "length_weight": 5.0e-4,
                        "cc_weight": 100.0,
                        "cc_threshold": 0.05,
                        "curvature_weight": 1.0e-4,
                        "curvature_threshold": 40.0,
                        "banana_surf_radius": 0.22,
                        "tf_current_A": -8.0e4,
                        "order": 2,
                        "banana_init_current_A": 1.0e4,
                        "banana_current_max_A": 1.6e4,
                        "alm_max_outer_iters": 10,
                        "alm_penalty_init": 1.0,
                        "alm_penalty_scale": 10.0,
                        "alm_penalty_max": 1.0e8,
                        "basin_hops": 0,
                        "basin_stepsize": 0.01,
                        "basin_temperature": 2.5,
                        "basin_niter_success": 8,
                        "basin_seed": 11,
                        "init_only": False,
                    }
                ),
                encoding="utf-8",
            )
            disabled_args = make_stage2_alm_wrapper_args(
                profile=None,
                stage2_spec_json=str(disabled_spec_path),
            )
            disabled_spec, _ = module.resolve_stage2_spec_payload(disabled_args)
            disabled_config = module.build_stage2_alm_config(
                disabled_args,
                resolved_spec=disabled_spec,
            )

            enabled_spec_path = Path(tmpdir) / "stage2_spec_enabled.json"
            enabled_spec_path.write_text(
                json.dumps(
                    {
                        "major_radius": 0.976,
                        "toroidal_flux": 0.24,
                        "length_weight": 5.0e-4,
                        "cc_weight": 100.0,
                        "cc_threshold": 0.05,
                        "curvature_weight": 1.0e-4,
                        "curvature_threshold": 40.0,
                        "banana_surf_radius": 0.22,
                        "tf_current_A": -8.0e4,
                        "order": 2,
                        "banana_init_current_A": 1.0e4,
                        "banana_current_max_A": 1.6e4,
                        "alm_max_outer_iters": 10,
                        "alm_penalty_init": 1.0,
                        "alm_penalty_scale": 10.0,
                        "alm_penalty_max": 1.0e8,
                        "basin_hops": 3,
                        "basin_stepsize": 0.01,
                        "basin_temperature": 2.5,
                        "basin_niter_success": 8,
                        "basin_seed": -1,
                        "init_only": False,
                    }
                ),
                encoding="utf-8",
            )
            enabled_args = make_stage2_alm_wrapper_args(
                profile=None,
                stage2_spec_json=str(enabled_spec_path),
            )
            enabled_spec, _ = module.resolve_stage2_spec_payload(enabled_args)
            with patch.object(module.os, "urandom", return_value=b"\x00\x00\x00*"):
                enabled_config = module.build_stage2_alm_config(
                    enabled_args,
                    resolved_spec=enabled_spec,
                )

        self.assertIsNone(disabled_config.basin_seed)
        self.assertIsNone(module._expected_stage2_artifact_metadata(disabled_config)["basin_seed"])
        self.assertEqual(enabled_config.basin_seed, 42)
        self.assertEqual(module._expected_stage2_artifact_metadata(enabled_config)["basin_seed"], 42)

    def test_single_stage_thresholded_physics_rerun_wrapper_pins_thresholded_physics_thresholds_and_warn_mode(self):
        source = SINGLE_STAGE_THRESHOLDED_PHYSICS_RERUN_MODULE_PATH.read_text()

        self.assertIn('"--constraint-method"', source)
        self.assertIn('"alm"', source)
        self.assertIn('"--alm-formulation"', source)
        self.assertIn('"thresholded_physics"', source)
        self.assertIn('"--hardware-search-mode"', source)
        self.assertIn('"warn"', source)
        self.assertNotIn('"adaptive"', source)
        self.assertIn('"--alm-qs-threshold"', source)
        self.assertIn('"--alm-boozer-threshold"', source)
        self.assertIn('"--alm-iota-penalty-threshold"', source)
        self.assertIn('"--alm-length-penalty-threshold"', source)

    def test_single_stage_thresholded_physics_rerun_wrapper_resolves_cli_paths(self):
        module = load_single_stage_thresholded_physics_rerun_module()
        args = make_single_stage_thresholded_physics_rerun_args(equilibria_dir="eqdir")

        command = module.build_single_stage_thresholded_physics_command(args)

        self.assertEqual(
            command[command.index("--stage2-bs-path") + 1],
            str(Path("relative/seed.json").resolve()),
        )
        self.assertEqual(
            command[command.index("--output-root") + 1],
            str(Path("outputs").resolve()),
        )
        self.assertEqual(
            command[command.index("--equilibria-dir") + 1],
            str(Path("eqdir").resolve()),
        )

    def test_single_stage_thresholded_physics_rerun_wrapper_parse_args_accepts_seed_order_upgrade(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with patch.object(
            sys,
            "argv",
            [
                "run_single_stage_thresholded_physics_alm.py",
                "--plasma-surf-filename",
                DEFAULT_ALM_WRAPPER_SURFACE,
                "--stage2-bs-path",
                "seed.json",
                "--seed-order-upgrade",
                "4",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.seed_order_upgrade, 4)

    def test_single_stage_thresholded_physics_rerun_wrapper_parse_args_accepts_warm_start_surface_stem(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with patch.object(
            sys,
            "argv",
            [
                "run_single_stage_thresholded_physics_alm.py",
                "--plasma-surf-filename",
                DEFAULT_ALM_WRAPPER_SURFACE,
                "--stage2-bs-path",
                "seed.json",
                "--warm-start-surface-stem",
                "recovery/surf_best_feasible",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.warm_start_surface_stem, "recovery/surf_best_feasible")

    def test_single_stage_thresholded_physics_rerun_wrapper_parse_args_accepts_stage2_seed_surf_path(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with patch.object(
            sys,
            "argv",
            [
                "run_single_stage_thresholded_physics_alm.py",
                "--plasma-surf-filename",
                DEFAULT_ALM_WRAPPER_SURFACE,
                "--stage2-bs-path",
                "seed.json",
                "--stage2-seed-surf-path",
                "stage2/surf_opt_boozer_surface.json",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(
            args.stage2_seed_surf_path,
            "stage2/surf_opt_boozer_surface.json",
        )

    def test_single_stage_thresholded_physics_rerun_wrapper_parse_args_accepts_independent_banana_current_mode(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with patch.object(
            sys,
            "argv",
            [
                "run_single_stage_thresholded_physics_alm.py",
                "--plasma-surf-filename",
                DEFAULT_ALM_WRAPPER_SURFACE,
                "--stage2-bs-path",
                "seed.json",
                "--single-stage-banana-current-mode",
                "independent",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.single_stage_banana_current_mode, "independent")

    def test_single_stage_thresholded_physics_rerun_wrapper_forwards_stage2_handoff_flags(self):
        module = load_single_stage_thresholded_physics_rerun_module()
        args = make_single_stage_thresholded_physics_rerun_args(
            allow_init_only_stage2_seed=True,
            equilibrium_path="eq/demo.nc",
            stage2_seed_surf_path="stage2/surf_opt_boozer_surface.json",
            warm_start_surface_stem="recovery/surf_best_feasible",
            seed_order_upgrade=4,
            constraint_weight=-1.0,
            boozer_I=0.031,
            plasma_current_A=9000.0,
            num_tf_coils=18,
            banana_surf_radius=0.23,
            stage2_seed_tf_current_A=12345.0,
        )

        command = module.build_single_stage_thresholded_physics_command(args)

        self.assertIn("--allow-init-only-stage2-seed", command)
        self.assertEqual(
            command[command.index("--equilibrium-path") + 1],
            str(Path("eq/demo.nc").resolve()),
        )
        self.assertEqual(
            command[command.index("--stage2-seed-surf-path") + 1],
            str(Path("stage2/surf_opt_boozer_surface.json").resolve()),
        )
        self.assertEqual(
            command[command.index("--warm-start-surface-stem") + 1],
            str(Path("recovery/surf_best_feasible").resolve()),
        )
        self.assertEqual(command[command.index("--seed-order-upgrade") + 1], "4")
        self.assertEqual(command[command.index("--constraint-weight") + 1], "-1.0")
        self.assertEqual(command[command.index("--boozer-I") + 1], "0.031")
        self.assertEqual(
            command[command.index("--plasma-current-A") + 1],
            "9000.0",
        )
        self.assertEqual(command[command.index("--num-tf-coils") + 1], "18")
        self.assertEqual(
            command[command.index("--banana-surf-radius") + 1],
            "0.23",
        )
        self.assertEqual(
            command[command.index("--stage2-seed-tf-current-A") + 1],
            "12345.0",
        )
        self.assertEqual(
            command[command.index("--single-stage-banana-current-mode") + 1],
            "shared",
        )

    def test_single_stage_thresholded_physics_rerun_wrapper_adds_offspec_flag(self):
        module = load_single_stage_thresholded_physics_rerun_module()
        args = make_single_stage_thresholded_physics_rerun_args(
            curvature_threshold=150.0,
        )

        command = module.build_single_stage_thresholded_physics_command(args)

        self.assertIn("--allow-offspec-engineering-constraints", command)

    def test_single_stage_thresholded_physics_rerun_wrapper_parse_args_rejects_adaptive_mode(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with patch.object(
            sys,
            "argv",
            [
                "run_single_stage_thresholded_physics_alm.py",
                "--plasma-surf-filename",
                DEFAULT_ALM_WRAPPER_SURFACE,
                "--stage2-bs-path",
                "seed.json",
                "--hardware-search-mode",
                "adaptive",
            ],
        ):
            with self.assertRaises(SystemExit) as excinfo:
                module.parse_args()

        self.assertEqual(excinfo.exception.code, 2)

    def test_single_stage_thresholded_physics_rerun_wrapper_defaults_match_single_stage_entrypoint(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with patch.object(
            sys,
            "argv",
            [
                "run_single_stage_thresholded_physics_alm.py",
                "--plasma-surf-filename",
                DEFAULT_ALM_WRAPPER_SURFACE,
                "--stage2-bs-path",
                "seed.json",
            ],
        ):
            args = module.parse_args()

        self.assertEqual(args.cs_dist, 0.015)
        self.assertEqual(args.curvature_threshold, 100.0)
        self.assertEqual(args.single_stage_banana_current_mode, "shared")

    def test_single_stage_thresholded_physics_rerun_wrapper_rejects_stage2_surface_mismatch(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            write_stage2_results_with_digest(
                stage2_bs_path,
                {"PLASMA_SURF_FILENAME": "other_surface.nc"},
            )
            args = make_single_stage_thresholded_physics_rerun_args(
                plasma_surf_filename=DEFAULT_ALM_WRAPPER_SURFACE,
                stage2_bs_path=str(stage2_bs_path),
            )

            with self.assertRaisesRegex(ValueError, "Stage 2 artifact surface mismatch"):
                module.load_validated_stage2_seed_metadata(args)

    def test_single_stage_thresholded_physics_rerun_wrapper_rejects_init_only_stage2_seed(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            write_stage2_results_with_digest(
                stage2_bs_path,
                {
                    "PLASMA_SURF_FILENAME": DEFAULT_ALM_WRAPPER_SURFACE,
                    "init_only": True,
                },
            )
            args = make_single_stage_thresholded_physics_rerun_args(
                plasma_surf_filename=DEFAULT_ALM_WRAPPER_SURFACE,
                stage2_bs_path=str(stage2_bs_path),
            )

            with self.assertRaisesRegex(ValueError, "non-init-only Stage 2 artifact"):
                module.load_validated_stage2_seed_metadata(args)

    def test_single_stage_thresholded_physics_rerun_wrapper_allows_init_only_stage2_seed_with_override(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            stage2_bs_path = tmpdir_path / "biot_savart_opt.json"
            stage2_bs_path.write_text("{}", encoding="utf-8")
            results_path = write_stage2_results_with_digest(
                stage2_bs_path,
                {
                    "PLASMA_SURF_FILENAME": DEFAULT_ALM_WRAPPER_SURFACE,
                    "init_only": True,
                },
            )
            args = make_single_stage_thresholded_physics_rerun_args(
                plasma_surf_filename=DEFAULT_ALM_WRAPPER_SURFACE,
                stage2_bs_path=str(stage2_bs_path),
                allow_init_only_stage2_seed=True,
            )

            _, loaded_results_path, loaded_results = (
                module.load_validated_stage2_seed_metadata(args)
            )

        self.assertEqual(loaded_results_path, results_path.resolve())
        self.assertTrue(loaded_results["init_only"])

    def test_single_stage_thresholded_physics_rerun_wrapper_dry_run_does_not_require_existing_stage2_artifact(self):
        module = load_single_stage_thresholded_physics_rerun_module()

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            output_root = tmpdir_path / "outputs"
            summary_path = tmpdir_path / "summary.json"
            missing_stage2_bs_path = tmpdir_path / "missing" / "biot_savart_opt.json"

            with patch.object(
                sys,
                "argv",
                [
                    "run_single_stage_thresholded_physics_alm.py",
                    "--dry-run",
                    "--plasma-surf-filename",
                    DEFAULT_ALM_WRAPPER_SURFACE,
                    "--stage2-bs-path",
                    str(missing_stage2_bs_path),
                    "--output-root",
                    str(output_root),
                    "--summary-json",
                    str(summary_path),
                ],
            ):
                self.assertEqual(module.main(), 0)

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            marker_path = (output_root / "DRY_RUN_ONLY.txt").resolve()
            self.assertEqual(summary["stage2_bs_path"], str(missing_stage2_bs_path.resolve()))
            self.assertTrue(summary["dry_run"])
            self.assertEqual(summary["output_contract"], "dry_run_summary_only")
            self.assertFalse(summary["contains_solver_outputs"])
            self.assertEqual(summary["dry_run_marker_path"], str(marker_path))
            self.assertNotIn("stage2_results_path", summary)
            self.assertNotIn("stage2_artifact_plasma_surf_filename", summary)
            self.assertTrue(marker_path.exists())
            self.assertIn("dry run only", marker_path.read_text(encoding="utf-8").lower())

    def test_single_stage_basin_hopping_uses_shared_helper_and_records_telemetry(self):
        source = SINGLE_STAGE_MODULE_PATH.read_text()
        results_dict = find_assigned_dict(SINGLE_STAGE_MODULE_PATH, "results")

        self.assertIn("from banana_opt.basin_hopping import run_basin_hopping", source)
        self.assertIn("run_basin_hopping(", source)
        self.assertIn("basin_temperature=args.basin_temperature", source)
        self.assertIn("basin_niter_success=basin_niter_success", source)

        entries = {
            key.value: value
            for key, value in zip(results_dict.keys, results_dict.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        for field_name in EXPECTED_BASIN_TELEMETRY_FIELDS:
            self.assertIn(field_name, entries)
        self.assertIn("basin_temperature", entries)
        self.assertIn("basin_niter_success", entries)

    def test_stage2_constraint_activity_tolerances_track_smoothing_windows(self):
        source = STAGE2_MODULE_PATH.read_text()
        self.assertIn("stage2_constraint_activity_tolerances", source)

        functions = extract_functions(
            STAGE2_OBJECTIVES_MODULE_PATH,
            ["stage2_constraint_activity_tolerances"],
            {"_SMOOTHING_EPS": np.finfo(float).eps},
        )
        stage2_constraint_activity_tolerances = functions[
            "stage2_constraint_activity_tolerances"
        ]

        tolerances = stage2_constraint_activity_tolerances(0.005, 0.05)

        self.assertEqual(tolerances, [1e-3, 0.02, 0.2, 1e-3])

    def test_stage2_builds_bounded_alm_settings(self):
        alm_utils = load_alm_utils_module()
        functions = extract_functions(
            STAGE2_OBJECTIVES_MODULE_PATH,
            ["build_stage2_alm_settings"],
            {"ALMSettings": alm_utils.ALMSettings},
        )
        build_stage2_alm_settings = functions["build_stage2_alm_settings"]
        settings = build_stage2_alm_settings(_make_alm_args())

        self.assertEqual(settings.max_outer_iterations, 7)
        self.assertEqual(settings.max_subproblem_continuations, 9)
        self.assertEqual(settings.penalty_init, 2.0)
        self.assertEqual(settings.trust_radius_init, 0.15)
        self.assertEqual(settings.trust_radius_min, 1e-3)
        self.assertEqual(settings.max_inner_attempts, 5)

    def test_stage2_zero_trust_radius_disables_bounds_in_settings(self):
        alm_utils = load_alm_utils_module()
        functions = extract_functions(
            STAGE2_OBJECTIVES_MODULE_PATH,
            ["build_stage2_alm_settings"],
            {"ALMSettings": alm_utils.ALMSettings},
        )
        build_stage2_alm_settings = functions["build_stage2_alm_settings"]
        settings = build_stage2_alm_settings(
            _make_alm_args(alm_trust_radius_init=0.0)
        )

        self.assertIsNone(settings.trust_radius_init)

    def test_stage2_parse_args_exposes_restart_seed_flag(self):
        source = STAGE2_MODULE_PATH.read_text()

        self.assertIn('"--stage2-bs-path"', source)
        self.assertIn('"STAGE2_BS_PATH"', source)

    def test_stage2_results_contract_records_hardware_status_fields(self):
        source = STAGE2_MODULE_PATH.read_text()
        self.assertIn("_build_stage2_results_impl(", source)
        objectives_source = STAGE2_OBJECTIVES_MODULE_PATH.read_text()
        self.assertIn("fixed_stage2_clearance_contract()", objectives_source)
        self.assertIn(
            "build_hardware_constraint_artifact_payload_fields(",
            objectives_source,
        )
        self.assertIn(
            "_build_stage2_artifact_hardware_snapshot(",
            objectives_source,
        )

    def test_stage2_results_contract_records_basin_hopping_telemetry(self):
        results_dict = find_function_return_dict(
            STAGE2_OBJECTIVES_MODULE_PATH,
            "build_stage2_results",
        )

        entries = {
            key.value: value
            for key, value in zip(results_dict.keys, results_dict.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

        for field_name in EXPECTED_BASIN_TELEMETRY_FIELDS:
            self.assertIn(field_name, entries)
            value_node = entries[field_name]
            self.assertIsInstance(value_node, ast.Name)
            self.assertEqual(value_node.id, field_name)

    def test_stage2_seed_loader_reuses_saved_biot_savart_configuration(self):
        functions = extract_functions(
            STAGE2_MODULE_PATH,
            ["load_stage2_seed_configuration"],
            {
                "np": np,
                "load": None,
                "curves_to_vtk": None,
                "partition_loaded_stage2_coils": None,
            },
        )
        load_stage2_seed_configuration = functions["load_stage2_seed_configuration"]

        class FakeCurrent:
            def __init__(self, value):
                self._value = value

            def get_value(self):
                return self._value

        class FakeCurve:
            pass

        class FakeCoil:
            def __init__(self, curve, current):
                self.curve = curve
                self.current = current

        class FakeBiotSavart:
            def __init__(self, coils):
                self.coils = coils
                self.points = None

            def set_points(self, points):
                self.points = points

            def B(self):
                return np.zeros_like(self.points)

        class FakeSurface:
            def __init__(self):
                self.saved_path = None
                self.extra_data = None

            def gamma(self):
                return np.zeros((2, 2, 3))

            def unitnormal(self):
                return np.ones((2, 2, 3))

            def to_vtk(self, path, extra_data):
                self.saved_path = path
                self.extra_data = extra_data

        coils = [
            FakeCoil(FakeCurve(), FakeCurrent(100000.0)),
            FakeCoil(FakeCurve(), FakeCurrent(100000.0)),
            FakeCoil(FakeCurve(), FakeCurrent(9500.0)),
            FakeCoil(FakeCurve(), FakeCurrent(-9500.0)),
        ]
        fake_bs = FakeBiotSavart(coils)
        vtk_calls = {}

        load_stage2_seed_configuration.__globals__["load"] = lambda path: fake_bs
        load_stage2_seed_configuration.__globals__["curves_to_vtk"] = (
            lambda curves, path, close=True: vtk_calls.update(
                {"curves": curves, "path": path, "close": close}
            )
        )
        load_stage2_seed_configuration.__globals__["partition_loaded_stage2_coils"] = (
            lambda loaded_coils, *, stage2_results, requested_num_tf_coils: SimpleNamespace(
                tf_coils=loaded_coils[:requested_num_tf_coils],
                banana_coils=loaded_coils[requested_num_tf_coils:],
                proxy_coils=[],
                vf_coils=[],
            )
        )

        surf = FakeSurface()
        result = load_stage2_seed_configuration(
            "/tmp/seed.json",
            surf,
            2,
            "/tmp/out/",
            stage2_results={"COIL_GROUPS": "unused-by-test"},
        )

        self.assertIs(result[0], fake_bs)
        self.assertEqual(result[1], [coil.curve for coil in coils])
        self.assertIs(result[2], coils[2].curve)
        self.assertEqual(result[3], coils[2:])
        self.assertEqual(result[4], coils[:2])
        self.assertEqual(fake_bs.points.shape, (4, 3))
        self.assertEqual(vtk_calls["path"], "/tmp/out/curves_init")
        self.assertTrue(vtk_calls["close"])
        self.assertEqual(surf.saved_path, "/tmp/out/surf_init")
        self.assertEqual(surf.extra_data["B_N"].shape, (2, 2, 1))

    def test_single_stage_constraint_activity_tolerances_match_selection_windows(self):
        source = SINGLE_STAGE_CONSTRAINTS_MODULE_PATH.read_text()
        self.assertIn("single_stage_constraint_activity_tolerances", source)

        functions = extract_functions(
            SINGLE_STAGE_CONSTRAINTS_MODULE_PATH,
            ["single_stage_constraint_activity_tolerances"],
            {"np": np, "_SMOOTHING_EPS": np.finfo(float).eps},
        )
        single_stage_constraint_activity_tolerances = functions[
            "single_stage_constraint_activity_tolerances"
        ]

        tolerances = single_stage_constraint_activity_tolerances(
            0.005,
            0.05,
            include_surface_surface=True,
        )

        self.assertEqual(tolerances.tolist(), [0.02, 0.02, 0.2, 0.02])


if __name__ == "__main__":
    unittest.main()
