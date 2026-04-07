import importlib.util
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
STAGE2_OBJECTIVES_PATH = EXAMPLES_ROOT / "banana_opt" / "stage2_objectives.py"
SINGLE_STAGE_GEOMETRY_PATH = EXAMPLES_ROOT / "banana_opt" / "single_stage_geometry.py"
SINGLE_STAGE_CONSTRAINTS_PATH = EXAMPLES_ROOT / "banana_opt" / "single_stage_constraints.py"


def _load_module(module_path: Path, prefix: str):
    module_name = f"{prefix}_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    original_sys_path = sys.path.copy()
    sys.path.insert(0, str(EXAMPLES_ROOT))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_sys_path
    return module


class _FakeScalarObjective:
    def __init__(self, value):
        self._value = float(value)

    def J(self):
        return self._value


class _FakeLengthObjective:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)

    def J(self):
        return self._value

    def dJ(self, partials=False):
        if not partials:
            raise AssertionError("Expected partial derivative request")
        return lambda _objective: self._grad.copy()


class _FakeBaseObjective:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)
        self.x = None

    def J(self):
        return self._value

    def dJ(self):
        return self._grad.copy()


class _FakeCurveDistance:
    def __init__(self, minimum_distance, shortest_distance):
        self.minimum_distance = float(minimum_distance)
        self._shortest_distance = float(shortest_distance)
        self.curves = ["curve_a", "curve_b"]

    def shortest_distance(self):
        return self._shortest_distance


class _FakeCurvatureObjective:
    def __init__(self, threshold, kappa_values, objective_value):
        self.threshold = float(threshold)
        self.curve = SimpleNamespace(kappa=lambda: np.asarray(kappa_values, dtype=float))
        self._objective_value = float(objective_value)

    def J(self):
        return self._objective_value


class _FakeCurve:
    def __init__(self, gamma_points, kappa_values=None):
        self._gamma = np.asarray(gamma_points, dtype=float)
        self._kappa = np.asarray(kappa_values if kappa_values is not None else [], dtype=float)

    def gamma(self):
        return self._gamma.copy()

    def kappa(self):
        return self._kappa.copy()

    def dkappa_by_dcoeff_vjp(self, weights):
        weighted_sum = float(np.sum(weights))
        return lambda _objective: np.array([weighted_sum, -weighted_sum], dtype=float)

    def dgamma_by_dcoeff_vjp(self, point_gradient):
        gradient_sum = np.sum(point_gradient, axis=0)
        return _FakeDerivative(np.array([gradient_sum[0], gradient_sum[1]], dtype=float))


class _FakeSurfaceWithGradient:
    def __init__(self, gamma_points):
        self._gamma = np.asarray(gamma_points, dtype=float)

    def gamma(self):
        return self._gamma.copy()

    def dgamma_by_dcoeff_vjp(self, point_gradient):
        gradient_sum = np.sum(point_gradient.reshape((-1, 3)), axis=0)
        return _FakeDerivative(np.array([gradient_sum[0], gradient_sum[2]], dtype=float))


class _FakeDerivative:
    def __init__(self, gradient=None):
        if isinstance(gradient, dict) or gradient is None:
            self._gradient = np.zeros(2, dtype=float)
        else:
            self._gradient = np.asarray(gradient, dtype=float)

    def __call__(self, _objective):
        return self._gradient.copy()

    def __add__(self, other):
        return _FakeDerivative(self._gradient + other._gradient)

    def __iadd__(self, other):
        self._gradient = self._gradient + other._gradient
        return self

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)


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


class _FakeSurfaceState:
    def __init__(self, x):
        self.x = np.asarray(x, dtype=float)


class _FakeBoozerSurface:
    def __init__(self, x, iota, G):
        self.surface = _FakeSurfaceState(x)
        self.res = {"iota": iota, "G": G}
        self.calls = []

    def run_code(self, iota, G):
        self.calls.append((float(iota), float(G)))
        self.res["iota"] = iota
        self.res["G"] = G


def _surface_entry(x, iota, G):
    return {"boozer_surface": _FakeBoozerSurface(x, iota, G)}


class _ModuleTestCase(unittest.TestCase):
    MODULE_PATH = None
    MODULE_PREFIX = None

    def setUp(self):
        self.module = _load_module(self.MODULE_PATH, self.MODULE_PREFIX)


class Stage2ObjectiveModuleTests(_ModuleTestCase):
    MODULE_PATH = STAGE2_OBJECTIVES_PATH
    MODULE_PREFIX = "banana_stage2_objectives"

    def test_make_stage2_fun_returns_value_grad_and_logs_metrics(self):
        class _JF:
            def __init__(self):
                self.x = None

            def J(self):
                return 1.23

            def dJ(self):
                return np.array([1.0, -2.0])

        fun = self.module.make_stage2_fun(
            _JF(),
            _FakeBiotSavart((2, 3)),
            _FakeSurfaceNormals((1, 2, 3)),
            _FakeScalarObjective(0.12),
            _FakeScalarObjective(1.75),
            SimpleNamespace(shortest_distance=lambda: 0.055),
            _FakeScalarObjective(39.5),
        )

        with mock.patch("builtins.print") as print_mock:
            value, grad = fun(np.array([0.2, -0.1]))

        self.assertAlmostEqual(value, 1.23)
        np.testing.assert_allclose(grad, [1.0, -2.0])
        log_line = print_mock.call_args[0][0]
        self.assertIn("J=1.2e+00", log_line)
        self.assertIn("Jf=1.2e-01", log_line)
        self.assertIn("Len=1.8m", log_line)
        self.assertIn("C-C-Sep=0.06m", log_line)
        self.assertIn("Curvature=39.50", log_line)

    def test_evaluate_stage2_alm_problem_exposes_constraint_payload(self):
        base_objective = _FakeBaseObjective(3.5, [1.2, -0.5])
        new_surf = _FakeSurfaceNormals((2, 2, 3))
        new_bs = _FakeBiotSavart((4, 3))
        Jf = _FakeScalarObjective(0.25)
        Jls = _FakeLengthObjective(2.2, [0.3, 0.4])
        Jccdist = _FakeCurveDistance(0.05, 0.04)
        Jc = _FakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5)

        def fake_augmented(base_value, base_grad, signed_values, grads, multipliers, penalty):
            self.assertAlmostEqual(base_value, 3.5)
            np.testing.assert_allclose(base_grad, [1.2, -0.5])
            np.testing.assert_allclose(signed_values, [0.2, -0.008, 0.75])
            np.testing.assert_allclose(grads[0], [0.3, 0.4])
            np.testing.assert_allclose(grads[1], [0.6, 0.2])
            np.testing.assert_allclose(grads[2], [0.9, -0.1])
            np.testing.assert_allclose(multipliers, [0.1, 0.2, 0.3])
            self.assertAlmostEqual(penalty, 12.0)
            return {
                "total": 9.0,
                "grad": np.array([7.0, -3.0]),
                "stationarity_norm": 0.5,
            }

        with mock.patch.object(
            self.module,
            "augmented_inequality_objective",
            side_effect=fake_augmented,
        ), mock.patch("builtins.print"):
            result = self.module.evaluate_stage2_alm_problem(
                dofs=np.array([0.25, -0.4]),
                base_objective=base_objective,
                new_bs=new_bs,
                new_surf=new_surf,
                Jf=Jf,
                Jls=Jls,
                length_target=2.0,
                Jccdist=Jccdist,
                Jc=Jc,
                distance_smoothing=0.005,
                curvature_smoothing=0.02,
                multipliers=np.array([0.1, 0.2, 0.3]),
                penalty=12.0,
                stage2_constraint_activity_tolerances=lambda ds, cs: [ds * 4.0, cs * 4.0],
                smooth_min_distance_signed_constraint=lambda *_args: (-0.008, np.array([0.6, 0.2])),
                smooth_max_curvature_signed_constraint=lambda *_args: (0.75, np.array([0.9, -0.1])),
            )

        np.testing.assert_allclose(base_objective.x, [0.25, -0.4])
        self.assertEqual(
            result["constraint_names"],
            ["coil_length_upper_bound", "coil_coil_spacing", "max_curvature"],
        )
        np.testing.assert_allclose(result["dual_update_values"], [0.2, -0.008, 0.75])
        np.testing.assert_allclose(result["feasibility_values"], [0.2, 0.01, 1.0])
        self.assertEqual(result["constraint_activity_tolerances"], [0.02, 0.08])
        self.assertAlmostEqual(result["max_feasibility_violation"], 1.0)
        self.assertAlmostEqual(result["total"], 9.0)
        np.testing.assert_allclose(result["grad"], [7.0, -3.0])

    def test_stage2_constraint_activity_tolerances_track_smoothing_windows(self):
        tolerances = self.module.stage2_constraint_activity_tolerances(0.005, 0.05)
        self.assertEqual(tolerances, [1e-3, 0.02, 0.2])

    def test_smooth_max_curvature_signed_constraint_uses_active_window(self):
        curve = _FakeCurve(
            gamma_points=[[0.0, 0.0, 0.0]],
            kappa_values=[3.0, 5.0, 4.4],
        )

        signed_value, grad = self.module.smooth_max_curvature_signed_constraint(
            curve,
            threshold=4.0,
            temperature=0.2,
            base_objective_optimizable=SimpleNamespace(),
        )

        self.assertGreater(signed_value, 1.0)
        np.testing.assert_allclose(grad, [1.0, -1.0])

    def test_smooth_min_distance_signed_constraint_returns_zero_grad_without_pairs(self):
        objective = SimpleNamespace(x=np.array([2.0, -3.0]))
        curve = _FakeCurve(gamma_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        signed_value, grad = self.module.smooth_min_distance_signed_constraint(
            [curve],
            minimum_distance=0.05,
            temperature=0.01,
            base_objective_optimizable=objective,
        )

        self.assertAlmostEqual(signed_value, 0.05)
        np.testing.assert_allclose(grad, [0.0, 0.0])


class SingleStageGeometryModuleTests(_ModuleTestCase):
    MODULE_PATH = SINGLE_STAGE_GEOMETRY_PATH
    MODULE_PREFIX = "banana_single_stage_geometry"

    def test_snapshot_and_restore_surface_states_round_trip(self):
        surface_data = [
            _surface_entry([1.0, 2.0], 0.31, 5.0),
            _surface_entry([3.0, 4.0], 0.47, 7.0),
        ]

        state = self.module.snapshot_surface_states(surface_data)
        surface_data[0]["boozer_surface"].surface.x[:] = 99.0
        surface_data[0]["boozer_surface"].res["iota"] = -1.0
        surface_data[0]["boozer_surface"].res["G"] = -2.0

        self.module.restore_surface_states(surface_data, state)

        np.testing.assert_allclose(surface_data[0]["boozer_surface"].surface.x, [1.0, 2.0])
        self.assertAlmostEqual(surface_data[0]["boozer_surface"].res["iota"], 0.31)
        self.assertAlmostEqual(surface_data[0]["boozer_surface"].res["G"], 5.0)
        np.testing.assert_allclose(state["sdofs"][0], [1.0, 2.0])

    def test_solve_surface_stack_at_dofs_restores_state_and_runs_surfaces(self):
        surface_data = [
            _surface_entry([9.0, 9.0], 1.0, 2.0),
            _surface_entry([8.0, 8.0], 3.0, 4.0),
        ]
        state = {
            "sdofs": [np.array([1.0, 2.0]), np.array([3.0, 4.0])],
            "iota": [0.11, 0.22],
            "G": [5.5, 6.6],
        }
        objective = SimpleNamespace(x=None)

        with mock.patch.object(
            self.module,
            "evaluate_surface_stack",
            return_value={"success": True},
        ) as evaluate_mock:
            result = self.module.solve_surface_stack_at_dofs(
                x=np.array([7.0, -2.0]),
                objective=objective,
                surface_data=surface_data,
                state=state,
                vessel_surface="VV",
                surface_gap_threshold=0.05,
                vessel_gap_threshold=0.04,
                enforce_nesting=False,
            )

        np.testing.assert_allclose(objective.x, [7.0, -2.0])
        np.testing.assert_allclose(surface_data[0]["boozer_surface"].surface.x, [1.0, 2.0])
        np.testing.assert_allclose(surface_data[1]["boozer_surface"].surface.x, [3.0, 4.0])
        self.assertEqual(surface_data[0]["boozer_surface"].calls, [(0.11, 5.5)])
        self.assertEqual(surface_data[1]["boozer_surface"].calls, [(0.22, 6.6)])
        evaluate_mock.assert_called_once_with(
            surface_data,
            vessel_surface="VV",
            surface_gap_threshold=0.05,
            vessel_gap_threshold=0.04,
            enforce_nesting=False,
        )
        self.assertEqual(result, {"success": True})

    def test_continuation_inner_surface_weight_validates_and_ramps(self):
        self.assertEqual(
            self.module.continuation_inner_surface_weight(1, 0, 5, 0.2),
            1.0,
        )
        self.assertAlmostEqual(
            self.module.continuation_inner_surface_weight(2, 2, 5, 0.1),
            0.46,
        )
        self.assertEqual(
            self.module.continuation_inner_surface_weight(2, 5, 0, 0.1),
            1.0,
        )
        with self.assertRaises(ValueError):
            self.module.continuation_inner_surface_weight(2, 0, 5, 1.5)

    def test_evaluate_single_stage_hardware_snapshot_shapes_scalar_status(self):
        result = self.module.evaluate_single_stage_hardware_snapshot(
            curve_curve_distance_obj=SimpleNamespace(shortest_distance=lambda: 0.04),
            cc_dist=0.05,
            curve_surface_distance_obj=SimpleNamespace(shortest_distance=lambda: 0.03),
            cs_dist=0.02,
            surface_vessel_distance_obj=SimpleNamespace(shortest_distance=lambda: 0.01),
            surface_status={"outer_vessel_gap": 0.5},
            ss_dist=0.04,
            banana_curve=SimpleNamespace(kappa=lambda: np.array([39.0, 41.0])),
            curvature_threshold=40.0,
        )

        self.assertAlmostEqual(result["curve_curve_min_dist"], 0.04)
        self.assertAlmostEqual(result["curve_surface_min_dist"], 0.03)
        self.assertAlmostEqual(result["surface_vessel_min_dist"], 0.01)
        self.assertAlmostEqual(result["max_curvature"], 41.0)
        self.assertFalse(result["status"]["success"])
        self.assertEqual(len(result["status"]["violations"]), 3)


class SingleStageConstraintModuleTests(_ModuleTestCase):
    MODULE_PATH = SINGLE_STAGE_CONSTRAINTS_PATH
    MODULE_PREFIX = "banana_single_stage_constraints"

    def test_single_stage_constraint_activity_tolerances_match_selection_windows(self):
        tolerances = self.module.single_stage_constraint_activity_tolerances(
            0.005,
            0.05,
            include_surface_surface=True,
        )
        np.testing.assert_allclose(tolerances, [0.02, 0.02, 0.2, 0.02])

    def test_smooth_max_curvature_signed_constraint_uses_active_window(self):
        curve = _FakeCurve(
            gamma_points=[[0.0, 0.0, 0.0]],
            kappa_values=[3.0, 5.0, 4.4],
        )

        signed_value, grad, violation = self.module.smooth_max_curvature_signed_constraint(
            curve,
            threshold=4.0,
            temperature=0.2,
            objective_optimizable=SimpleNamespace(),
        )

        self.assertGreater(signed_value, 1.0)
        self.assertEqual(violation, signed_value)
        np.testing.assert_allclose(grad, [1.0, -1.0])

    def test_smooth_min_curve_curve_signed_constraint_returns_zero_grad_without_pairs(self):
        objective = SimpleNamespace(x=np.array([2.0, -3.0]))
        curve = _FakeCurve(gamma_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        signed_value, grad, violation = self.module.smooth_min_curve_curve_signed_constraint(
            [curve],
            minimum_distance=0.05,
            temperature=0.01,
            objective_optimizable=objective,
        )

        self.assertAlmostEqual(signed_value, 0.05)
        self.assertEqual(violation, 0.0)
        np.testing.assert_allclose(grad, [0.0, 0.0])

    def test_smooth_min_surface_surface_signed_constraint_reports_positive_violation(self):
        surface_1 = _FakeSurfaceWithGradient(
            gamma_points=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        )
        surface_2 = _FakeSurfaceWithGradient(
            gamma_points=[[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]]
        )

        with mock.patch.object(self.module, "Derivative", _FakeDerivative):
            signed_value, grad, violation = self.module.smooth_min_surface_surface_signed_constraint(
                surface_1,
                surface_2,
                minimum_distance=0.5,
                temperature=0.01,
                objective_optimizable=SimpleNamespace(),
            )

        self.assertGreater(violation, 0.0)
        self.assertAlmostEqual(violation, signed_value)
        self.assertEqual(grad.shape, (2,))
