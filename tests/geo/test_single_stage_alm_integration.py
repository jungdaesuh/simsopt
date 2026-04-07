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


def make_stage2_alm_args(**overrides):
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
        objectives_source = SINGLE_STAGE_OBJECTIVES_MODULE_PATH.read_text()

        self.assertIn("_evaluate_alm_objective_impl(", source)
        self.assertIn(
            "augmented_inequality_objective_fn=augmented_inequality_objective",
            objectives_source,
        )
        self.assertNotIn("augmented_objective(", objectives_source)
        self.assertIn("accepted_callback=callback", source)
        self.assertNotIn("inner_callback=callback", source)

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

        self.assertEqual(tolerances, [1e-3, 0.02, 0.2])

    def test_stage2_builds_bounded_alm_settings(self):
        alm_utils = load_alm_utils_module()
        functions = extract_functions(
            STAGE2_OBJECTIVES_MODULE_PATH,
            ["build_stage2_alm_settings"],
            {"ALMSettings": alm_utils.ALMSettings},
        )
        build_stage2_alm_settings = functions["build_stage2_alm_settings"]
        settings = build_stage2_alm_settings(make_stage2_alm_args())

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
            make_stage2_alm_args(alm_trust_radius_init=0.0)
        )

        self.assertIsNone(settings.trust_radius_init)

    def test_stage2_parse_args_exposes_restart_seed_flag(self):
        source = STAGE2_MODULE_PATH.read_text()

        self.assertIn('"--stage2-bs-path"', source)
        self.assertIn('"STAGE2_BS_PATH"', source)

    def test_stage2_results_contract_records_hardware_status_fields(self):
        source = STAGE2_MODULE_PATH.read_text()
        self.assertIn("_build_stage2_results_impl(", source)

        results_dict = find_function_return_dict(
            STAGE2_OBJECTIVES_MODULE_PATH,
            "build_stage2_results",
        )

        entries = {
            key.value: value
            for key, value in zip(results_dict.keys, results_dict.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

        self.assertIn("HARDWARE_CONSTRAINTS_OK", entries)
        self.assertIn("HARDWARE_CONSTRAINT_VIOLATIONS", entries)

        for field_name, expected_status_key in (
            ("HARDWARE_CONSTRAINTS_OK", "success"),
            ("HARDWARE_CONSTRAINT_VIOLATIONS", "violations"),
        ):
            value_node = entries[field_name]
            self.assertIsInstance(value_node, ast.Subscript)
            self.assertIsInstance(value_node.value, ast.Name)
            self.assertEqual(value_node.value.id, "hardware_status")
            self.assertIsInstance(value_node.slice, ast.Constant)
            self.assertEqual(value_node.slice.value, expected_status_key)

    def test_stage2_seed_loader_reuses_saved_biot_savart_configuration(self):
        functions = extract_functions(
            STAGE2_MODULE_PATH,
            ["load_stage2_seed_configuration"],
            {"np": np, "load": None, "curves_to_vtk": None},
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

        surf = FakeSurface()
        result = load_stage2_seed_configuration("/tmp/seed.json", surf, 2, "/tmp/out/")

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
