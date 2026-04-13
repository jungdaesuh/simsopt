import ast
import importlib.util
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
SINGLE_STAGE_PATH = EXAMPLES_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py"
SINGLE_STAGE_GEOMETRY_PATH = EXAMPLES_ROOT / "banana_opt" / "single_stage_geometry.py"
STAGE2_OBJECTIVES_PATH = EXAMPLES_ROOT / "banana_opt" / "stage2_objectives.py"
SMOOTHING_PATH = EXAMPLES_ROOT / "banana_opt" / "smoothing.py"
ALM_UTILS_PATH = EXAMPLES_ROOT / "alm_utils.py"

SINGLE_STAGE_SNAPSHOT = (
    "721073a85ab87cc5017705e77aa0593f568387dc",
    "examples/single_stage_optimization/SINGLE_STAGE/single_stage_banana_example.py",
)
STAGE2_SNAPSHOT = (
    "ed3924100c1ef4cab08191e74d4a5bcd306c27e2",
    "examples/single_stage_optimization/STAGE_2/banana_coil_solver.py",
)
SMOOTHING_EPS = float(np.finfo(float).eps)


def _load_module(module_path: Path, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _git_show(commit: str, rel_path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{rel_path}"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise unittest.SkipTest(
            f"snapshot source unavailable for {commit}:{rel_path}"
        ) from exc
    return result.stdout


def _extract_snapshot_functions(
    commit: str,
    rel_path: str,
    function_names: tuple[str, ...],
    extra_globals: dict | None = None,
):
    source = _git_show(commit, rel_path)
    tree = ast.parse(source)
    selected = [
        node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    extracted = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(extracted)
    namespace = {"np": np}
    if extra_globals:
        namespace.update(extra_globals)
    exec(compile(extracted, f"{commit}:{rel_path}", "exec"), namespace)
    return {name: namespace[name] for name in function_names}


def _load_alm_utils_module():
    return _load_module(ALM_UTILS_PATH, "alm_utils")


class _FakeSurface:
    def __init__(self, label, major_radius, volume, dofs):
        self.label = label
        self._major_radius = float(major_radius)
        self._volume = float(volume)
        self._dofs = np.asarray(dofs, dtype=float)

    def major_radius(self):
        return self._major_radius

    def get_dofs(self):
        return self._dofs.copy()

    def set_dofs(self, dofs):
        self._dofs = np.asarray(dofs, dtype=float)

    def volume(self):
        return self._volume


class _FakeSurfaceFactory:
    surfaces = {
        0.42: _FakeSurface("outer", major_radius=4.0, volume=9.0, dofs=[4.0, 8.0]),
        0.21: _FakeSurface("inner", major_radius=2.0, volume=3.0, dofs=[2.0, 6.0]),
    }

    @classmethod
    def from_wout(cls, file_loc, range, nphi, ntheta, s):
        del file_loc, range, nphi, ntheta
        surface = cls.surfaces[round(float(s), 2)]
        return _FakeSurface(
            surface.label,
            major_radius=surface.major_radius(),
            volume=surface.volume(),
            dofs=surface.get_dofs(),
        )


class _FakeDistanceObjective:
    def __init__(self, distance):
        self._distance = float(distance)

    def shortest_distance(self):
        return self._distance


class _FakeCurve:
    def __init__(self, kappa_values):
        self._kappa_values = np.asarray(kappa_values, dtype=float)

    def kappa(self):
        return self._kappa_values


class _FakeLengthObjective:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)

    def J(self):
        return self._value

    def dJ(self, partials=False):
        if not partials:
            raise AssertionError("Expected partial derivative request")
        return lambda objective: self._grad.copy()


class _FakeBaseObjective:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)
        self.x = None

    def J(self):
        return self._value

    def dJ(self):
        return self._grad.copy()


class _FakeCurveCurveDistance:
    def __init__(self, minimum_distance, shortest_distance):
        self.minimum_distance = float(minimum_distance)
        self._shortest_distance = float(shortest_distance)
        self.curves = ["curve_a", "curve_b"]

    def shortest_distance(self):
        return self._shortest_distance


class _FakeCurvatureObjective:
    def __init__(self, threshold, kappa_values):
        self.threshold = float(threshold)
        self.curve = _FakeCurve(kappa_values)


class _FakeFluxObjective:
    def __init__(self, value):
        self._value = float(value)

    def J(self):
        return self._value


class _FakeBiotSavart:
    def __init__(self, field_shape):
        self._field = np.zeros(field_shape, dtype=float)

    def B(self):
        return self._field.copy()


class _FakeSurfaceNormals:
    def __init__(self, shape):
        self._unitnormal = np.zeros(shape, dtype=float)

    def unitnormal(self):
        return self._unitnormal.copy()


class SnapshotParityTests(unittest.TestCase):
    def setUp(self):
        self.current_single_stage = _load_module(SINGLE_STAGE_PATH, "single_stage_example")
        self.current_smoothing = _load_module(SMOOTHING_PATH, "banana_smoothing")
        self.current_single_stage_geometry = _load_module(
            SINGLE_STAGE_GEOMETRY_PATH,
            "banana_single_stage_geometry",
        )
        self.current_stage2_objectives = _load_module(
            STAGE2_OBJECTIVES_PATH,
            "banana_stage2_objectives",
        )
        self.alm_utils = _load_alm_utils_module()

    def test_single_stage_smoothing_matches_snapshot(self):
        snapshot = _extract_snapshot_functions(
            *SINGLE_STAGE_SNAPSHOT,
            function_names=("_stable_softmax", "_smoothmax_selected", "_smoothmin_selected"),
            extra_globals={"_SMOOTHING_EPS": SMOOTHING_EPS},
        )
        values = np.array([3.0, -2.0, 5.5, 0.25], dtype=float)

        np.testing.assert_allclose(
            snapshot["_stable_softmax"](values),
            self.current_smoothing.stable_softmax(values, SMOOTHING_EPS),
        )

        for temperature in (1e-12, 0.25, 1.5):
            expected_max = snapshot["_smoothmax_selected"](values, temperature)
            actual_max = self.current_smoothing.smoothmax_selected(
                values,
                temperature,
                SMOOTHING_EPS,
            )
            np.testing.assert_allclose(actual_max[0], expected_max[0])
            np.testing.assert_allclose(actual_max[1], expected_max[1])

            expected_min = snapshot["_smoothmin_selected"](values, temperature)
            actual_min = self.current_smoothing.smoothmin_selected(
                values,
                temperature,
                SMOOTHING_EPS,
            )
            np.testing.assert_allclose(actual_min[0], expected_min[0])
            np.testing.assert_allclose(actual_min[1], expected_min[1])

    def test_stage2_smoothing_matches_snapshot(self):
        snapshot = _extract_snapshot_functions(
            *STAGE2_SNAPSHOT,
            function_names=("_stable_softmax", "_smoothmax_selected", "_smoothmin_selected"),
            extra_globals={"_SMOOTHING_EPS": SMOOTHING_EPS},
        )
        values = np.array([7.0, 7.5, -0.5, 1.0], dtype=float)

        np.testing.assert_allclose(
            snapshot["_stable_softmax"](values),
            self.current_smoothing.stable_softmax(values, SMOOTHING_EPS),
        )

        for temperature in (1e-9, 0.02, 2.0):
            expected_max = snapshot["_smoothmax_selected"](values, temperature)
            actual_max = self.current_smoothing.smoothmax_selected(
                values,
                temperature,
                SMOOTHING_EPS,
            )
            np.testing.assert_allclose(actual_max[0], expected_max[0])
            np.testing.assert_allclose(actual_max[1], expected_max[1])

            expected_min = snapshot["_smoothmin_selected"](values, temperature)
            actual_min = self.current_smoothing.smoothmin_selected(
                values,
                temperature,
                SMOOTHING_EPS,
            )
            np.testing.assert_allclose(actual_min[0], expected_min[0])
            np.testing.assert_allclose(actual_min[1], expected_min[1])

    def test_stage2_hardware_constraints_matches_snapshot(self):
        snapshot = _extract_snapshot_functions(
            *STAGE2_SNAPSHOT,
            function_names=("evaluate_stage2_hardware_constraints",),
        )["evaluate_stage2_hardware_constraints"]
        cases = [
            (1.7, 1.75, 0.08, 0.05, 39.0, 40.0),
            (1.8, 1.75, 0.04, 0.05, 41.0, 40.0),
        ]

        for case in cases:
            expected = snapshot(*case)
            actual = self.current_stage2_objectives.evaluate_stage2_hardware_constraints(*case)
            self.assertEqual(actual, expected)

    def test_single_stage_hardware_constraints_matches_snapshot(self):
        snapshot = _extract_snapshot_functions(
            *SINGLE_STAGE_SNAPSHOT,
            function_names=("evaluate_single_stage_hardware_constraints",),
        )["evaluate_single_stage_hardware_constraints"]
        cases = [
            (0.06, 0.05, 0.03, 0.02, 0.05, 0.04, 38.0, 40.0),
            (0.04, 0.05, 0.01, 0.02, 0.03, 0.04, 41.5, 40.0),
        ]

        for case in cases:
            expected = snapshot(*case)
            actual = self.current_single_stage_geometry.evaluate_single_stage_hardware_constraints(
                *case
            )
            self.assertEqual(actual, expected)

    def test_build_surface_configs_matches_single_stage_snapshot(self):
        snapshot = _extract_snapshot_functions(
            *SINGLE_STAGE_SNAPSHOT,
            function_names=("scale_surface_to_major_radius", "build_surface_configs"),
            extra_globals={"SurfaceRZFourier": _FakeSurfaceFactory},
        )["build_surface_configs"]

        expected = snapshot(
            "dummy.nc",
            32,
            64,
            0.42,
            6.0,
            10.0,
            2,
            0.5,
        )
        actual = self.current_single_stage_geometry.build_surface_configs(
            "dummy.nc",
            32,
            64,
            0.42,
            6.0,
            10.0,
            2,
            0.5,
            surface_factory=_FakeSurfaceFactory,
        )

        self.assertEqual(
            [(entry["name"], entry["seed_label"], entry["target_volume"]) for entry in actual],
            [(entry["name"], entry["seed_label"], entry["target_volume"]) for entry in expected],
        )
        for actual_entry, expected_entry in zip(actual, expected):
            np.testing.assert_allclose(
                actual_entry["initial_surface"].get_dofs(),
                expected_entry["initial_surface"].get_dofs(),
            )

    def test_single_stage_reference_surface_wrapper_order(self):
        surfaces = SimpleNamespace(vessel="VV", hbt="HBT", coil_winding_surface="CWS")

        with mock.patch.object(
            self.current_single_stage,
            "build_banana_reference_surfaces",
            return_value=surfaces,
        ):
            self.assertEqual(
                self.current_single_stage.build_hbt_reference_surfaces(5, 0.2),
                ("VV", "HBT", "CWS"),
            )

    def test_stage2_reference_surface_wrapper_order(self):
        stage2_module = _load_module(
            EXAMPLES_ROOT / "STAGE_2" / "banana_coil_solver.py",
            "banana_stage2_solver",
        )
        surfaces = SimpleNamespace(vessel="VV", hbt="HBT", coil_winding_surface="CWS")

        with mock.patch.object(
            stage2_module,
            "build_banana_reference_surfaces",
            return_value=surfaces,
        ):
            self.assertEqual(
                stage2_module.build_hbt_reference_surfaces(5, 0.2),
                ("HBT", "CWS", "VV"),
            )

    def test_compute_surface_vessel_min_dist_matches_snapshot(self):
        snapshot = _extract_snapshot_functions(
            *SINGLE_STAGE_SNAPSHOT,
            function_names=("compute_single_stage_surface_vessel_min_dist",),
            extra_globals={"cdist": self.current_single_stage_geometry.cdist},
        )["compute_single_stage_surface_vessel_min_dist"]

        direct_obj = _FakeDistanceObjective(0.17)
        direct_status = {"outer_vessel_gap": 0.05}
        self.assertEqual(
            self.current_single_stage_geometry.compute_single_stage_surface_vessel_min_dist(
                direct_obj,
                direct_status,
            ),
            snapshot(direct_obj, direct_status),
        )

        cached_status = {"outer_vessel_gap": 0.11}
        self.assertEqual(
            self.current_single_stage_geometry.compute_single_stage_surface_vessel_min_dist(
                None,
                cached_status,
            ),
            snapshot(None, cached_status),
        )

        outer_surface = SimpleNamespace(
            gamma=lambda: np.array([[[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]]], dtype=float)
        )
        vessel_surface = SimpleNamespace(
            gamma=lambda: np.array([[[1.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]]], dtype=float)
        )
        actual = self.current_single_stage_geometry.compute_single_stage_surface_vessel_min_dist(
            None,
            {"outer_vessel_gap": None},
            outer_surface,
            vessel_surface,
        )
        expected = snapshot(None, {"outer_vessel_gap": None}, outer_surface, vessel_surface)
        self.assertEqual(actual, expected)

    def test_topology_gate_deficit_matches_snapshot(self):
        snapshot = _extract_snapshot_functions(
            *SINGLE_STAGE_SNAPSHOT,
            function_names=("topology_gate_deficit",),
        )["topology_gate_deficit"]
        statuses = [
            {"enabled": False, "survival_threshold": 0.25, "survival_fraction": 1.0},
            {"enabled": True, "survival_threshold": 0.8, "survival_fraction": 0.5},
            {"enabled": True, "survival_threshold": 0.6, "survival_fraction": 0.9},
        ]

        for status in statuses:
            self.assertEqual(
                self.current_single_stage_geometry.topology_gate_deficit(status),
                snapshot(status),
            )

    def test_stage2_alm_problem_matches_snapshot(self):
        base_objective = _FakeBaseObjective(3.5, [1.2, -0.5])
        new_surf = _FakeSurfaceNormals((2, 2, 3))
        new_bs = _FakeBiotSavart((4, 3))
        Jf = _FakeFluxObjective(0.25)
        Jls = _FakeLengthObjective(1.8, [0.3, 0.4])
        Jccdist = _FakeCurveCurveDistance(0.05, 0.04)
        Jc = _FakeCurvatureObjective(40.0, [35.0, 42.0, 38.0])
        banana_current = SimpleNamespace(
            get_value=lambda: 9500.0,
            vjp=lambda _value: (lambda _objective: np.array([0.7, -0.4])),
        )

        def smooth_min_distance_signed_constraint(curves, minimum_distance, temperature, objective):
            del curves, minimum_distance, temperature, objective
            return -0.008, np.array([0.6, 0.2])

        def smooth_max_curvature_signed_constraint(curve, threshold, temperature, objective):
            del curve, threshold, temperature, objective
            return 0.75, np.array([0.9, -0.1])

        def stage2_constraint_activity_tolerances(distance_smoothing, curvature_smoothing):
            return [1e-3, distance_smoothing * 4.0, curvature_smoothing * 4.0, 1e-3]

        snapshot = _extract_snapshot_functions(
            *STAGE2_SNAPSHOT,
            function_names=("evaluate_stage2_alm_problem",),
            extra_globals={
                "augmented_inequality_objective": self.alm_utils.augmented_inequality_objective,
                "lower_bound_residual": self.alm_utils.lower_bound_residual,
                "upper_bound_residual": self.alm_utils.upper_bound_residual,
                "stage2_constraint_activity_tolerances": stage2_constraint_activity_tolerances,
                "smooth_min_distance_signed_constraint": smooth_min_distance_signed_constraint,
                "smooth_max_curvature_signed_constraint": smooth_max_curvature_signed_constraint,
            },
        )["evaluate_stage2_alm_problem"]

        common_args = dict(
            dofs=np.array([0.25, -0.4]),
            base_objective=base_objective,
            new_bs=new_bs,
            new_surf=new_surf,
            Jf=Jf,
            Jls=Jls,
            length_target=1.75,
            Jccdist=Jccdist,
            Jc=Jc,
            banana_current=banana_current,
            banana_current_max_A=16000.0,
            distance_smoothing=0.005,
            curvature_smoothing=0.02,
            multipliers=np.array([0.1, 0.2, 0.3, 0.4]),
            penalty=12.0,
        )

        expected = snapshot(
            dofs=common_args["dofs"],
            base_objective=common_args["base_objective"],
            new_bs=common_args["new_bs"],
            new_surf=common_args["new_surf"],
            Jf=common_args["Jf"],
            Jls=common_args["Jls"],
            length_target=common_args["length_target"],
            Jccdist=common_args["Jccdist"],
            Jc=common_args["Jc"],
            distance_smoothing=common_args["distance_smoothing"],
            curvature_smoothing=common_args["curvature_smoothing"],
            multipliers=np.array([0.1, 0.2, 0.3]),
            penalty=common_args["penalty"],
        )
        actual = self.current_stage2_objectives.evaluate_stage2_alm_problem(
            **common_args,
            stage2_constraint_activity_tolerances=stage2_constraint_activity_tolerances,
            smooth_min_distance_signed_constraint=smooth_min_distance_signed_constraint,
            smooth_max_curvature_signed_constraint=smooth_max_curvature_signed_constraint,
        )

        self.assertEqual(actual["constraint_names"][:3], expected["constraint_names"])
        self.assertEqual(actual["constraint_names"][3], "banana_current_upper_bound")
        self.assertEqual(actual["constraint_activity_tolerances"][:3], [1e-3, 0.02, 0.08])
        self.assertEqual(actual["constraint_activity_tolerances"][3], 1e-3)
        np.testing.assert_allclose(actual["grad"], expected["grad"])
        np.testing.assert_allclose(actual["constraint_grads"][:3], expected["constraint_grads"])
        np.testing.assert_allclose(actual["constraint_grads"][3], [0.7, -0.4])
        np.testing.assert_allclose(actual["dual_update_values"][:3], expected["dual_update_values"])
        np.testing.assert_allclose(actual["dual_update_values"][3], -6500.0)
        np.testing.assert_allclose(
            actual["hard_signed_constraint_values"],
            [0.05, 0.01, 2.0, -6500.0],
        )
        np.testing.assert_allclose(
            actual["hard_violation_values"],
            [0.05, 0.01, 2.0, 0.0],
        )
        np.testing.assert_allclose(
            actual["surrogate_signed_constraint_values"][:3],
            expected["dual_update_values"],
        )
        np.testing.assert_allclose(
            actual["surrogate_signed_constraint_values"][3],
            -6500.0,
        )
        np.testing.assert_allclose(
            actual["hard_dual_update_values"],
            [0.05, 0.01, 2.0, -6500.0],
        )
        np.testing.assert_allclose(actual["feasibility_values"][:3], expected["feasibility_values"])
        np.testing.assert_allclose(actual["feasibility_values"][3], 0.0)
        self.assertAlmostEqual(actual["base_value"], expected["base_value"])
        self.assertAlmostEqual(
            actual["max_feasibility_violation"],
            expected["max_feasibility_violation"],
        )

    def test_build_stage2_bs_path_prefers_init_current_aware_contract(self):
        base_args = dict(
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
            stage2_seed_banana_init_current_A=1.0e4,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            seed_spec = self.current_single_stage.Stage2SeedSpec(
                plasma_surf_filename="demo.nc",
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
                banana_init_current_A=1.0e4,
            )

            local_dir = self.current_single_stage.format_local_stage2_seed_dir(seed_spec)
            local_candidate = (
                tmp_path / "local" / "outputs-demo.nc" / f"{local_dir}-CM=penalty" / "biot_savart_opt.json"
            )
            local_candidate.parent.mkdir(parents=True)
            local_candidate.write_text("{}", encoding="utf-8")
            local_args = SimpleNamespace(
                **base_args,
                stage2_source="local",
                local_stage2_root=str(tmp_path / "local"),
                database_stage2_root=str(tmp_path / "database"),
            )
            self.assertEqual(
                self.current_single_stage.build_stage2_bs_path(local_args),
                str(local_candidate),
            )

            database_dir = self.current_single_stage.format_database_stage2_seed_dir(seed_spec)
            database_candidate = (
                tmp_path / "database" / "outputs-demo.nc" / database_dir / "biot_savart_opt.json"
            )
            database_candidate.parent.mkdir(parents=True)
            database_candidate.write_text("{}", encoding="utf-8")
            database_args = SimpleNamespace(
                **base_args,
                stage2_source="database",
                local_stage2_root=str(tmp_path / "local"),
                database_stage2_root=str(tmp_path / "database"),
            )
            self.assertEqual(
                self.current_single_stage.build_stage2_bs_path(database_args),
                str(database_candidate),
            )


if __name__ == "__main__":
    unittest.main()
