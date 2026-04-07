import ast
import importlib.util
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

import numpy as np


SINGLE_STAGE_MODULE_PATH = (
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


def make_single_stage_alm_args(**overrides):
    defaults = {
        "alm_max_outer_iters": 7,
        "alm_max_subproblem_continuations": 9,
        "alm_penalty_init": 2.0,
        "alm_penalty_scale": 3.0,
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


class SingleStageAlmIntegrationTests(unittest.TestCase):
    def test_single_stage_parse_args_exposes_alm_trust_region_controls(self):
        source = SINGLE_STAGE_MODULE_PATH.read_text()

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

    def test_single_stage_builds_bounded_alm_settings(self):
        alm_utils = load_alm_utils_module()
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
            ["build_single_stage_alm_settings"],
            {"ALMSettings": alm_utils.ALMSettings},
        )
        build_single_stage_alm_settings = functions["build_single_stage_alm_settings"]
        settings = build_single_stage_alm_settings(make_single_stage_alm_args())

        self.assertEqual(settings.max_outer_iterations, 7)
        self.assertEqual(settings.max_subproblem_continuations, 9)
        self.assertEqual(settings.penalty_init, 2.0)
        self.assertEqual(settings.penalty_scale, 3.0)
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
            make_single_stage_alm_args(alm_trust_radius_init=0.0)
        )

        self.assertIsNone(settings.trust_radius_init)

    def test_single_stage_source_uses_projected_inequality_alm_and_outer_accept_callback(self):
        source = SINGLE_STAGE_MODULE_PATH.read_text()

        self.assertIn("augmented_inequality_objective(", source)
        self.assertNotIn("augmented_objective(", source)
        self.assertIn("accepted_callback=callback", source)
        self.assertNotIn("inner_callback=callback", source)

    def test_stage2_constraint_activity_tolerances_track_smoothing_windows(self):
        functions = extract_functions(
            STAGE2_MODULE_PATH,
            ["stage2_constraint_activity_tolerances"],
            {"_SMOOTHING_EPS": np.finfo(float).eps},
        )
        stage2_constraint_activity_tolerances = functions[
            "stage2_constraint_activity_tolerances"
        ]

        tolerances = stage2_constraint_activity_tolerances(0.005, 0.05)

        self.assertEqual(tolerances, [1e-3, 0.02, 0.2])

    def test_single_stage_constraint_activity_tolerances_match_selection_windows(self):
        functions = extract_functions(
            SINGLE_STAGE_MODULE_PATH,
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
