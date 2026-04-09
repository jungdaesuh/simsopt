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
SINGLE_STAGE_OBJECTIVES_PATH = EXAMPLES_ROOT / "banana_opt" / "single_stage_objectives.py"
SINGLE_STAGE_SEARCH_POLICY_PATH = EXAMPLES_ROOT / "banana_opt" / "single_stage_search_policy.py"
SINGLE_STAGE_INCUMBENTS_PATH = EXAMPLES_ROOT / "banana_opt" / "incumbents.py"


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


class _FakeAlgebraicObjective:
    def __init__(self, value, gradient, projected_gradient=None):
        self._value = float(value)
        self._gradient = np.asarray(gradient, dtype=float)
        projected = gradient if projected_gradient is None else projected_gradient
        self._projected_gradient = np.asarray(projected, dtype=float)

    def J(self):
        return self._value

    def dJ(self, partials=False):
        if partials:
            return lambda _objective: self._projected_gradient.copy()
        return self._gradient.copy()

    def __add__(self, other):
        if other == 0:
            return self
        return _FakeAlgebraicObjective(
            self._value + other._value,
            self._gradient + other._gradient,
            self._projected_gradient + other._projected_gradient,
        )

    __radd__ = __add__

    def __mul__(self, scalar):
        return _FakeAlgebraicObjective(
            self._value * scalar,
            self._gradient * scalar,
            self._projected_gradient * scalar,
        )

    __rmul__ = __mul__


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


class _FakeSurfaceWithArrayGradient(_FakeSurfaceWithGradient):
    def dgamma_by_dcoeff_vjp(self, point_gradient):
        return np.sum(point_gradient.reshape((-1, 3)), axis=0)


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

    def test_build_stage2_alm_settings_converts_zero_trust_radius_to_none(self):
        settings = self.module.build_stage2_alm_settings(
            SimpleNamespace(
                alm_max_outer_iters=7,
                alm_max_subproblem_continuations=9,
                alm_penalty_init=2.0,
                alm_penalty_scale=3.0,
                alm_feas_tol=1e-4,
                alm_stationarity_tol=2e-4,
                alm_trust_radius_init=0.0,
                alm_trust_radius_min=1e-3,
                alm_trust_radius_shrink=0.4,
                alm_trust_radius_grow=1.8,
                alm_max_inner_attempts=5,
            )
        )

        self.assertEqual(settings.max_outer_iterations, 7)
        self.assertEqual(settings.max_subproblem_continuations, 9)
        self.assertEqual(settings.penalty_init, 2.0)
        self.assertIsNone(settings.trust_radius_init)
        self.assertEqual(settings.trust_radius_min, 1e-3)
        self.assertEqual(settings.max_inner_attempts, 5)

    def test_build_stage2_results_maps_hardware_and_alm_fields(self):
        args = SimpleNamespace(
            init_only=False,
            basin_hops=2,
            basin_stepsize=0.01,
            basin_temperature=2.5,
            basin_niter_success=6,
            alm_max_outer_iters=7,
            alm_max_subproblem_continuations=9,
            alm_penalty_init=2.0,
            alm_penalty_scale=3.0,
            alm_feas_tol=1e-4,
            alm_stationarity_tol=2e-4,
            alm_trust_radius_init=0.15,
            alm_trust_radius_min=1e-3,
            alm_trust_radius_shrink=0.4,
            alm_trust_radius_grow=1.8,
            alm_max_inner_attempts=5,
            alm_distance_smoothing=0.005,
            alm_curvature_smoothing=0.05,
            alm_taylor_test=True,
            alm_taylor_test_seed=123,
        )
        alm_result = SimpleNamespace(
            outer_iterations=4,
            penalty=8.0,
            multipliers=np.array([0.1, 0.2, 0.3]),
            constraint_values=np.array([0.0, 0.01, 0.0]),
            solver_constraint_values=np.array([0.0, 0.2, 0.0]),
            trust_radius=0.125,
            history=[{"outer_iteration": 1}],
        )
        hardware_status = {"success": False, "violations": ["too_curved"]}

        result = self.module.build_stage2_results(
            args=args,
            plasma_surf_filename="demo.nc",
            file_loc="/tmp/demo.nc",
            stage2_bs_path="/tmp/seed.json",
            tf_current_A=8.0e4,
            tf_current_sum_abs_A=1.6e5,
            num_tf_coils=2,
            banana_current_A=9.5e3,
            banana_to_tf_current_ratio=0.11875,
            cc_threshold=0.05,
            cc_weight=100.0,
            curvature_weight=1.0e-4,
            curvature_threshold=40.0,
            length_weight=5.0e-4,
            constraint_method="alm",
            theta_center=np.pi,
            phi_center=np.pi / 4.0,
            theta_width=np.pi / 6.0,
            phi_width=np.pi / 8.0,
            length_target=1.75,
            major_radius=0.915,
            toroidal_flux=0.24,
            nfp=22,
            banana_surf_radius=0.22,
            order=2,
            max_iterations=300,
            iterations=17,
            termination_message="hardware_constraints_failed",
            optimizer_success=False,
            basin_seed=7,
            basin_iterations=3,
            basin_minimization_failures=1,
            basin_accepted_hops=2,
            basin_rejected_hops=1,
            basin_best_objective=0.42,
            basin_accept_test_rejections=1,
            basin_accept_test_triggered=True,
            alm_result=alm_result,
            alm_taylor_result={"passed": True},
            final_volume=0.12,
            field_error=0.03,
            intersecting=True,
            final_max_curvature=41.0,
            final_coil_length=1.8,
            final_curve_curve_min_dist=0.04,
            hardware_status=hardware_status,
        )

        self.assertFalse(result["HARDWARE_CONSTRAINTS_OK"])
        self.assertEqual(result["HARDWARE_CONSTRAINT_VIOLATIONS"], ["too_curved"])
        self.assertEqual(result["ALM_MAX_OUTER_ITERS"], 7)
        self.assertEqual(result["ALM_OUTER_ITERATIONS"], 4)
        self.assertEqual(result["ALM_FINAL_TRUST_RADIUS"], 0.125)
        self.assertEqual(result["basin_seed"], 7)
        self.assertEqual(result["basin_temperature"], 2.5)
        self.assertEqual(result["basin_niter_success"], 6)
        self.assertEqual(result["basin_accepted_hops"], 2)
        self.assertEqual(result["basin_rejected_hops"], 1)
        self.assertEqual(result["basin_best_objective"], 0.42)
        self.assertEqual(result["basin_accept_test_rejections"], 1)
        self.assertTrue(result["basin_accept_test_triggered"])
        self.assertAlmostEqual(result["BANANA_TO_TF_CURRENT_RATIO"], 0.11875)

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


class SingleStageObjectiveModuleTests(_ModuleTestCase):
    MODULE_PATH = SINGLE_STAGE_OBJECTIVES_PATH
    MODULE_PREFIX = "banana_single_stage_objectives"

    @staticmethod
    def _make_projected_base_terms():
        return (
            SimpleNamespace(name="full"),
            [_FakeAlgebraicObjective(2.0, [2.0, 0.0], [2.0, 0.0, 0.0, 0.0])],
            [_FakeAlgebraicObjective(3.0, [0.5, 0.5], [0.5, 0.5, 0.0, 0.0])],
            _FakeAlgebraicObjective(4.0, [0.2, 0.1], [0.2, 0.1, 0.0, 0.0]),
            _FakeAlgebraicObjective(5.0, [1.0, 1.5], [1.0, 1.5, 0.0, 0.0]),
        )

    def test_average_surface_objectives_uses_weighted_mean(self):
        single = _FakeAlgebraicObjective(2.0, [2.0, -1.0])
        single_avg = self.module.average_surface_objectives([single])
        self.assertAlmostEqual(single_avg.J(), 2.0)
        np.testing.assert_allclose(single_avg.dJ(), [2.0, -1.0])

        left = _FakeAlgebraicObjective(2.0, [2.0, 0.0])
        right = _FakeAlgebraicObjective(6.0, [4.0, 2.0])
        weighted_avg = self.module.average_surface_objectives(
            [left, right],
            weights=np.array([0.5, 1.0]),
        )
        self.assertAlmostEqual(weighted_avg.J(), (0.5 * 2.0 + 6.0) / 1.5)
        np.testing.assert_allclose(weighted_avg.dJ(), [10.0 / 3.0, 4.0 / 3.0])

    def test_evaluate_total_objective_preserves_component_breakdown(self):
        nonqs = [
            _FakeAlgebraicObjective(2.0, [2.0, 0.0]),
            _FakeAlgebraicObjective(6.0, [4.0, 0.0]),
        ]
        brs = [
            _FakeAlgebraicObjective(10.0, [1.0, 1.0]),
            _FakeAlgebraicObjective(20.0, [3.0, 3.0]),
        ]
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        surface_term = _FakeAlgebraicObjective(1.5, [0.1, 0.2])

        result = self.module.evaluate_total_objective(
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
            JSurfSurf=surface_term,
            SURF_DIST_WEIGHT=8.0,
        )

        self.assertAlmostEqual(result["J_QS"], (0.5 * 2.0 + 6.0) / 1.5)
        self.assertAlmostEqual(result["J_Boozer"], (0.5 * 10.0 + 20.0) / 1.5)
        self.assertAlmostEqual(result["J_surf"], 1.5)
        np.testing.assert_allclose(result["dJ_surf"], [0.1, 0.2])
        self.assertAlmostEqual(result["total"], 50.0)
        np.testing.assert_allclose(result["grad"], [8.8, 6.266666666666667])

    def test_evaluate_alm_objective_builds_constraint_payload(self):
        nonqs = [_FakeAlgebraicObjective(2.0, [2.0, 0.0])]
        brs = [_FakeAlgebraicObjective(3.0, [0.5, 0.5])]
        jiota = _FakeAlgebraicObjective(4.0, [0.2, 0.1])
        jlength = _FakeAlgebraicObjective(5.0, [1.0, 1.5])
        jcc = _FakeAlgebraicObjective(0.6, [0.3, 0.4])
        jcs = _FakeAlgebraicObjective(0.7, [0.5, 0.6])
        jcurv = _FakeAlgebraicObjective(0.8, [0.7, 0.8])
        jsurf = _FakeAlgebraicObjective(0.9, [0.9, 1.0])

        def fake_augmented(base_value, base_grad, constraint_values, constraint_grads, multipliers, penalty):
            self.assertAlmostEqual(base_value, 25.0)
            np.testing.assert_allclose(base_grad, [4.6, 2.8])
            np.testing.assert_allclose(constraint_values, [-0.1, 0.2, 0.3, -0.4])
            np.testing.assert_allclose(constraint_grads[0], [1.0, 0.0])
            np.testing.assert_allclose(constraint_grads[1], [0.0, 1.0])
            np.testing.assert_allclose(constraint_grads[2], [1.0, -1.0])
            np.testing.assert_allclose(constraint_grads[3], [0.5, 0.5])
            np.testing.assert_allclose(multipliers, [0.1, 0.2, 0.3, 0.4])
            self.assertAlmostEqual(penalty, 9.0)
            return {
                "total": 25.0,
                "grad": np.array([8.0, -3.0]),
                "stationarity_norm": 0.125,
            }

        result = self.module.evaluate_alm_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=jiota,
            IOTAS_WEIGHT=3.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=1.0,
            JCurveCurve=jcc,
            JCurveSurface=jcs,
            JCurvature=jcurv,
            multipliers=np.array([0.1, 0.2, 0.3, 0.4]),
            penalty=9.0,
            objective_optimizable=SimpleNamespace(),
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface="outer",
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=(
                "coil_coil_spacing",
                "coil_surface_spacing",
                "max_curvature",
                "surface_vessel_spacing",
            ),
            curve_curve_constraint_fn=lambda *_args: (-0.1, np.array([1.0, 0.0]), 0.0),
            curve_surface_constraint_fn=lambda *_args: (0.2, np.array([0.0, 1.0]), 0.2),
            curvature_constraint_fn=lambda *_args: (0.3, np.array([1.0, -1.0]), 0.3),
            JSurfSurf=jsurf,
            vessel_surface="vessel",
            surface_surface_min_distance=0.04,
            surface_surface_constraint_fn=lambda *_args: (-0.4, np.array([0.5, 0.5]), 0.0),
            augmented_inequality_objective_fn=fake_augmented,
            activity_tolerances_fn=lambda ds, cs, include_surface_surface: np.array(
                [ds * 4.0, ds * 4.0, cs * 4.0, ds * 4.0] if include_surface_surface else [ds * 4.0, ds * 4.0, cs * 4.0],
                dtype=float,
            ),
        )

        self.assertEqual(
            result["constraint_names"],
            [
                "coil_coil_spacing",
                "coil_surface_spacing",
                "max_curvature",
                "surface_vessel_spacing",
            ],
        )
        np.testing.assert_allclose(result["dual_update_values"], [-0.1, 0.2, 0.3, -0.4])
        np.testing.assert_allclose(result["feasibility_values"], [0.0, 0.2, 0.3, 0.0])
        np.testing.assert_allclose(result["constraint_activity_tolerances"], [0.04, 0.04, 0.2, 0.04])
        self.assertAlmostEqual(result["base_total"], 25.0)
        self.assertAlmostEqual(result["max_feasibility_violation"], 0.3)
        self.assertAlmostEqual(result["J_cc"], 0.6)
        self.assertAlmostEqual(result["J_cs"], 0.7)
        self.assertAlmostEqual(result["J_surf"], 0.9)
        self.assertAlmostEqual(result["J_curvature"], 0.8)
        np.testing.assert_allclose(result["grad"], [8.0, -3.0])

    def test_evaluate_base_objective_projects_total_gradient_when_requested(self):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()

        result = self.module.evaluate_base_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=jiota,
            IOTAS_WEIGHT=3.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=1.0,
            objective_optimizable=objective,
        )

        np.testing.assert_allclose(result["grad"], [4.6, 2.8, 0.0, 0.0])

    def test_evaluate_base_objective_gil_formulation_zeros_base_value_and_grad(self):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()

        result = self.module.evaluate_base_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=jiota,
            IOTAS_WEIGHT=3.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=1.0,
            objective_optimizable=objective,
            alm_formulation="gil",
        )

        self.assertAlmostEqual(result["total"], 0.0)
        self.assertAlmostEqual(result["physics_total"], 25.0)
        np.testing.assert_allclose(result["grad"], [0.0, 0.0, 0.0, 0.0])

    def test_evaluate_alm_objective_projects_base_gradient_into_constraint_space(self):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()

        def fake_augmented(base_value, base_grad, constraint_values, constraint_grads, multipliers, penalty):
            self.assertAlmostEqual(base_value, 25.0)
            np.testing.assert_allclose(base_grad, [4.6, 2.8, 0.0, 0.0])
            np.testing.assert_allclose(constraint_values, [-0.1, 0.2, 0.3])
            np.testing.assert_allclose(constraint_grads[0], [1.0, 0.0, 0.0, 0.0])
            np.testing.assert_allclose(constraint_grads[1], [0.0, 1.0, 0.0, 0.0])
            np.testing.assert_allclose(constraint_grads[2], [1.0, -1.0, 0.0, 0.0])
            np.testing.assert_allclose(multipliers, [0.1, 0.2, 0.3])
            self.assertAlmostEqual(penalty, 9.0)
            return {
                "total": 26.0,
                "grad": np.array([9.0, -2.0, 0.0, 0.0]),
                "stationarity_norm": 0.25,
            }

        result = self.module.evaluate_alm_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=jiota,
            IOTAS_WEIGHT=3.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=1.0,
            JCurveCurve=_FakeAlgebraicObjective(0.6, [0.3, 0.4]),
            JCurveSurface=_FakeAlgebraicObjective(0.7, [0.5, 0.6]),
            JCurvature=_FakeAlgebraicObjective(0.8, [0.7, 0.8]),
            multipliers=np.array([0.1, 0.2, 0.3]),
            penalty=9.0,
            objective_optimizable=objective,
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface="outer",
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=(
                "coil_coil_spacing",
                "coil_surface_spacing",
                "max_curvature",
                "surface_vessel_spacing",
            ),
            curve_curve_constraint_fn=lambda *_args: (-0.1, np.array([1.0, 0.0, 0.0, 0.0]), 0.0),
            curve_surface_constraint_fn=lambda *_args: (0.2, np.array([0.0, 1.0, 0.0, 0.0]), 0.2),
            curvature_constraint_fn=lambda *_args: (0.3, np.array([1.0, -1.0, 0.0, 0.0]), 0.3),
            augmented_inequality_objective_fn=fake_augmented,
            activity_tolerances_fn=lambda ds, cs, include_surface_surface: np.array(
                [ds * 4.0, ds * 4.0, cs * 4.0],
                dtype=float,
            ),
        )

        np.testing.assert_allclose(result["grad"], [9.0, -2.0, 0.0, 0.0])

    def test_evaluate_alm_objective_gil_formulation_promotes_physics_terms_to_constraints(self):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()

        def fake_augmented(base_value, base_grad, constraint_values, constraint_grads, multipliers, penalty):
            self.assertAlmostEqual(base_value, 0.0)
            np.testing.assert_allclose(base_grad, [0.0, 0.0, 0.0, 0.0])
            np.testing.assert_allclose(
                constraint_values,
                [-0.1, 0.2, 0.3, 1.0, 2.0, 3.5, 5.0],
            )
            np.testing.assert_allclose(constraint_grads[3], [2.0, 0.0, 0.0, 0.0])
            np.testing.assert_allclose(constraint_grads[4], [0.5, 0.5, 0.0, 0.0])
            np.testing.assert_allclose(constraint_grads[5], [0.2, 0.1, 0.0, 0.0])
            np.testing.assert_allclose(constraint_grads[6], [1.0, 1.5, 0.0, 0.0])
            np.testing.assert_allclose(multipliers, np.arange(7, dtype=float))
            self.assertAlmostEqual(penalty, 4.0)
            return {
                "total": 11.0,
                "grad": np.array([7.0, -4.0, 0.0, 0.0]),
                "stationarity_norm": 0.5,
            }

        result = self.module.evaluate_alm_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=jiota,
            IOTAS_WEIGHT=3.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=1.0,
            JCurveCurve=_FakeAlgebraicObjective(0.6, [0.3, 0.4]),
            JCurveSurface=_FakeAlgebraicObjective(0.7, [0.5, 0.6]),
            JCurvature=_FakeAlgebraicObjective(0.8, [0.7, 0.8]),
            multipliers=np.arange(7, dtype=float),
            penalty=4.0,
            objective_optimizable=objective,
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface="outer",
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=(
                "coil_coil_spacing",
                "coil_surface_spacing",
                "max_curvature",
                "qs_error",
                "boozer_residual",
                "iota_penalty",
                "length_penalty",
            ),
            curve_curve_constraint_fn=lambda *_args: (-0.1, np.array([1.0, 0.0, 0.0, 0.0]), 0.0),
            curve_surface_constraint_fn=lambda *_args: (0.2, np.array([0.0, 1.0, 0.0, 0.0]), 0.2),
            curvature_constraint_fn=lambda *_args: (0.3, np.array([1.0, -1.0, 0.0, 0.0]), 0.3),
            augmented_inequality_objective_fn=fake_augmented,
            activity_tolerances_fn=lambda ds, cs, include_surface_surface: np.array(
                [ds * 4.0, ds * 4.0, cs * 4.0],
                dtype=float,
            ),
            alm_formulation="gil",
            qs_threshold=1.0,
            boozer_threshold=1.0,
            iota_penalty_threshold=0.5,
            length_penalty_threshold=0.0,
        )

        self.assertEqual(
            result["constraint_names"],
            [
                "coil_coil_spacing",
                "coil_surface_spacing",
                "max_curvature",
                "qs_error",
                "boozer_residual",
                "iota_penalty",
                "length_penalty",
            ],
        )
        self.assertAlmostEqual(result["physics_total"], 25.0)
        self.assertAlmostEqual(result["base_total"], 25.0)
        np.testing.assert_allclose(result["constraint_activity_tolerances"], [0.04, 0.04, 0.2, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result["grad"], [7.0, -4.0, 0.0, 0.0])


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


class SingleStageIncumbentsModuleTests(_ModuleTestCase):
    MODULE_PATH = SINGLE_STAGE_INCUMBENTS_PATH
    MODULE_PREFIX = "banana_single_stage_incumbents"

    def test_snapshot_and_restore_single_stage_incumbent_state_round_trip(self):
        run_dict = {
            "accepted_x": np.array([1.0, -2.0]),
            "surface_state": {
                "sdofs": [np.array([1.0, 2.0])],
                "iota": [0.3],
                "G": [4.0],
            },
            "J": 3.5,
            "dJ": np.array([0.25, -0.5]),
            "search_eval": {"total": 3.5, "grad": np.array([0.25, -0.5])},
            "surface_status": {"success": True, "values": [1.0]},
            "search_surface_status": {"success": False, "bad_phis": [2]},
            "accepted_hardware_status": {"success": True, "violations": []},
            "topology_gate_status": {"success": False, "reason": "ridge"},
            "last_successful_eval": {"total": 9.0},
            "last_successful_eval_weights": np.array([1.0]),
        }

        incumbent = self.module.snapshot_single_stage_incumbent_state(run_dict)
        run_dict["accepted_x"][:] = 99.0
        run_dict["surface_state"]["sdofs"][0][:] = -1.0
        run_dict["dJ"][:] = 7.0
        run_dict["search_eval"]["grad"][:] = 8.0
        run_dict["surface_status"]["success"] = False
        run_dict["accepted_hardware_status"]["success"] = False

        self.module.restore_single_stage_incumbent_state(run_dict, incumbent)

        np.testing.assert_allclose(run_dict["accepted_x"], [1.0, -2.0])
        np.testing.assert_allclose(run_dict["surface_state"]["sdofs"][0], [1.0, 2.0])
        np.testing.assert_allclose(run_dict["dJ"], [0.25, -0.5])
        np.testing.assert_allclose(run_dict["search_eval"]["grad"], [0.25, -0.5])
        self.assertTrue(run_dict["surface_status"]["success"])
        self.assertTrue(run_dict["accepted_hardware_status"]["success"])
        self.assertFalse(run_dict["topology_gate_status"]["success"])
        self.assertNotIn("last_successful_eval", run_dict)
        self.assertNotIn("last_successful_eval_weights", run_dict)


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
        surface_1 = _FakeSurfaceWithArrayGradient(
            gamma_points=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        )
        surface_2 = _FakeSurfaceWithArrayGradient(
            gamma_points=[[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]]
        )

        with mock.patch.object(
            self.module,
            "_new_derivative",
            side_effect=lambda: _FakeDerivative({}),
        ), mock.patch.object(
            self.module,
            "_surface_dgamma_by_dcoeff_derivative",
            side_effect=lambda _surface, point_gradient: _FakeDerivative(
                np.array(
                    [
                        np.sum(point_gradient.reshape((-1, 3)), axis=0)[0],
                        np.sum(point_gradient.reshape((-1, 3)), axis=0)[2],
                    ],
                    dtype=float,
                )
            ),
        ):
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

    def test_surface_vjp_helper_wraps_raw_surface_array_output_as_derivative(self):
        derivative = self.module._new_derivative()
        surface = _FakeSurfaceWithArrayGradient(
            gamma_points=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        )

        derivative += self.module._surface_dgamma_by_dcoeff_derivative(
            surface,
            np.array([[[1.0, 0.0, 2.0], [0.5, 0.0, 3.0]]], dtype=float),
        )

        self.assertTrue(hasattr(derivative, "data"))
        np.testing.assert_allclose(
            derivative.data[surface],
            np.array([1.5, 0.0, 5.0]),
        )


class SingleStageSearchPolicyModuleTests(_ModuleTestCase):
    MODULE_PATH = SINGLE_STAGE_SEARCH_POLICY_PATH
    MODULE_PREFIX = "banana_single_stage_search_policy"

    def test_hard_mode_rejects_hardware_violation(self):
        decision = self.module.decide_hardware_search_action(
            self.module.HardwareSearchPolicy("hard", 0),
            {"success": False},
            self.module.SearchContext(
                accepted_iterations=0,
                gate_scale=1.0,
                previous_objective=12.0,
            ),
        )

        self.assertTrue(decision.reject)
        self.assertFalse(decision.warning_only)
        self.assertEqual(decision.rejection_increment, 12.0)
        self.assertEqual(decision.reason, "hard_reject")

    def test_warn_mode_keeps_hardware_violation_warning_only(self):
        decision = self.module.decide_hardware_search_action(
            self.module.HardwareSearchPolicy("warn", 0),
            {"success": False},
            self.module.SearchContext(
                accepted_iterations=9,
                gate_scale=1.0,
                previous_objective=3.5,
            ),
        )

        self.assertFalse(decision.reject)
        self.assertTrue(decision.warning_only)
        self.assertIsNone(decision.rejection_increment)
        self.assertEqual(decision.reason, "warn_mode")

    def test_adaptive_mode_warns_only_while_gate_scale_is_relaxed(self):
        decision = self.module.decide_hardware_search_action(
            self.module.HardwareSearchPolicy("adaptive", 2),
            {"success": False},
            self.module.SearchContext(
                accepted_iterations=1,
                gate_scale=0.4,
                previous_objective=5.0,
            ),
        )

        self.assertFalse(decision.reject)
        self.assertTrue(decision.warning_only)
        self.assertEqual(decision.reason, "adaptive_soft_phase")

    def test_adaptive_mode_rejects_when_gate_scale_is_not_relaxed(self):
        decision = self.module.decide_hardware_search_action(
            self.module.HardwareSearchPolicy("adaptive", 2),
            {"success": False},
            self.module.SearchContext(
                accepted_iterations=1,
                gate_scale=1.0,
                previous_objective=5.0,
            ),
        )

        self.assertTrue(decision.reject)
        self.assertFalse(decision.warning_only)
        self.assertEqual(decision.reason, "hard_reject")

    def test_adaptive_mode_rejects_after_relaxed_gate_budget_exhausts(self):
        decision = self.module.decide_hardware_search_action(
            self.module.HardwareSearchPolicy("adaptive", 1),
            {"success": False},
            self.module.SearchContext(
                accepted_iterations=2,
                gate_scale=0.4,
                previous_objective=-7.0,
            ),
        )

        self.assertTrue(decision.reject)
        self.assertFalse(decision.warning_only)
        self.assertEqual(decision.rejection_increment, 7.0)
        self.assertEqual(decision.reason, "hard_reject")
