import importlib.util
import sys
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from simsopt.field.coil import Current, ScaledCurrent


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
ALM_UTILS_PATH = EXAMPLES_ROOT / "alm_utils.py"
STAGE2_OBJECTIVES_PATH = EXAMPLES_ROOT / "banana_opt" / "stage2_objectives.py"
SINGLE_STAGE_GEOMETRY_PATH = EXAMPLES_ROOT / "banana_opt" / "single_stage_geometry.py"
SINGLE_STAGE_CONSTRAINTS_PATH = (
    EXAMPLES_ROOT / "banana_opt" / "single_stage_constraints.py"
)
SMOOTH_DISTANCE_SELECTION_PATH = (
    EXAMPLES_ROOT / "banana_opt" / "smooth_distance_selection.py"
)
SINGLE_STAGE_OBJECTIVES_PATH = (
    EXAMPLES_ROOT / "banana_opt" / "single_stage_objectives.py"
)
SINGLE_STAGE_SEARCH_POLICY_PATH = (
    EXAMPLES_ROOT / "banana_opt" / "single_stage_search_policy.py"
)
SINGLE_STAGE_INCUMBENTS_PATH = EXAMPLES_ROOT / "banana_opt" / "incumbents.py"
POLOIDAL_EXTENT_PATH = EXAMPLES_ROOT / "banana_opt" / "poloidal_extent.py"
TAYLOR_TEST_EPSILONS = (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4)


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


class _XAwareQuadraticObjective:
    def __init__(self, owner, constant, linear, quadratic=0.0):
        self.owner = owner
        self.constant = float(constant)
        self.linear = np.asarray(linear, dtype=float)
        self.quadratic = float(quadratic)
        self.x = np.zeros_like(self.linear)

    def _x(self):
        source = self if self.owner is None else self.owner
        return np.asarray(source.x, dtype=float)

    def J(self):
        x = self._x()
        return float(
            self.constant
            + np.dot(self.linear, x)
            + 0.5 * self.quadratic * np.dot(x, x)
        )

    def gradient(self):
        return self.linear + self.quadratic * self._x()

    def dJ(self, partials=False):
        if partials:
            return lambda _objective=None: self.gradient()
        return self.gradient()

    def __add__(self, other):
        if other == 0:
            return self
        return _XAwareQuadraticObjective(
            self.owner,
            self.constant + other.constant,
            self.linear + other.linear,
            self.quadratic + other.quadratic,
        )

    __radd__ = __add__

    def __mul__(self, scalar):
        scalar = float(scalar)
        return _XAwareQuadraticObjective(
            self.owner,
            scalar * self.constant,
            scalar * self.linear,
            scalar * self.quadratic,
        )

    __rmul__ = __mul__


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


class _UnexpectedCurveDistance(_FakeCurveDistance):
    def shortest_distance(self):
        raise AssertionError("exact sampled distance should not be evaluated")


class _FakeCurvatureObjective:
    def __init__(self, threshold, kappa_values, objective_value):
        self.threshold = float(threshold)
        self.curve = SimpleNamespace(
            kappa=lambda: np.asarray(kappa_values, dtype=float)
        )
        self._objective_value = float(objective_value)

    def J(self):
        return self._objective_value


class _FakeCurve:
    def __init__(self, gamma_points, kappa_values=None):
        self._gamma = np.asarray(gamma_points, dtype=float)
        self._kappa = np.asarray(
            kappa_values if kappa_values is not None else [], dtype=float
        )

    def gamma(self):
        return self._gamma.copy()

    def kappa(self):
        return self._kappa.copy()

    def dkappa_by_dcoeff_vjp(self, weights):
        weighted_sum = float(np.sum(weights))
        return lambda _objective: np.array([weighted_sum, -weighted_sum], dtype=float)

    def dgamma_by_dcoeff_vjp(self, point_gradient):
        gradient_sum = np.sum(point_gradient, axis=0)
        return _FakeDerivative(
            np.array([gradient_sum[0], gradient_sum[1]], dtype=float)
        )


class _FakeSurfaceWithGradient:
    def __init__(self, gamma_points):
        self._gamma = np.asarray(gamma_points, dtype=float)

    def gamma(self):
        return self._gamma.copy()

    def dgamma_by_dcoeff_vjp(self, point_gradient):
        gradient_sum = np.sum(point_gradient.reshape((-1, 3)), axis=0)
        return _FakeDerivative(
            np.array([gradient_sum[0], gradient_sum[2]], dtype=float)
        )


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


class _FakeCurrentObjective:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)

    def get_value(self):
        return self._value

    def vjp(self, value):
        cotangent = float(np.asarray(value, dtype=float).reshape(-1)[0])
        return _FakeDerivative(cotangent * self._grad)


class _XAwareCurrentObjective:
    def __init__(self, owner, constant, linear):
        self.owner = owner
        self.constant = float(constant)
        self.linear = np.asarray(linear, dtype=float)

    def get_value(self):
        x = np.asarray(self.owner.x, dtype=float)
        return float(self.constant + np.dot(self.linear, x))

    def vjp(self, value):
        cotangent = float(np.asarray(value, dtype=float).reshape(-1)[0])
        return _FakeDerivative(cotangent * self.linear)


def _affine_signed_constraint(owner, offset, linear, *, include_violation):
    linear = np.asarray(linear, dtype=float)

    def constraint(*_args):
        signed_value = float(offset + np.dot(linear, owner.x))
        if include_violation:
            return signed_value, linear.copy(), max(0.0, signed_value)
        return signed_value, linear.copy()

    return constraint


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
    def __init__(self, owner, x):
        self._owner = owner
        self._x = np.asarray(x, dtype=float)
        self._self_intersecting = False
        self._self_intersecting_raises: Exception | None = None
        self.self_intersection_calls = 0

    @property
    def x(self):
        return self._x.copy()

    @x.setter
    def x(self, value):
        self._x = np.asarray(value, dtype=float)
        self._owner.need_to_run_code = True

    def set_self_intersecting(self, flag: bool):
        self._self_intersecting = bool(flag)
        self._self_intersecting_raises = None

    def set_self_intersecting_raises(self, error: Exception):
        self._self_intersecting_raises = error

    def is_self_intersecting(self, angle=0.0, thetas=None):
        self.self_intersection_calls += 1
        if self._self_intersecting_raises is not None:
            raise self._self_intersecting_raises
        return self._self_intersecting


class _FakeBoozerSurface:
    def __init__(self, x, iota, G):
        self.need_to_run_code = False
        self.surface = _FakeSurfaceState(self, x)
        self.res = {"iota": iota, "G": G, "success": True}
        self.calls = []
        self._queued_results = []

    def queue_result(
        self, *, surface_x=None, iota=None, G=None, success=True, raises=None
    ):
        self._queued_results.append(
            {
                "surface_x": None
                if surface_x is None
                else np.asarray(surface_x, dtype=float),
                "iota": iota,
                "G": G,
                "success": bool(success),
                "raises": raises,
            }
        )

    def run_code(self, iota, G):
        self.calls.append((float(iota), None if G is None else float(G)))
        if self._queued_results:
            queued_result = self._queued_results.pop(0)
            if queued_result["surface_x"] is not None:
                self.surface.x = queued_result["surface_x"].copy()
            self.res["iota"] = (
                float(iota)
                if queued_result["iota"] is None
                else float(queued_result["iota"])
            )
            self.res["G"] = (
                G if queued_result["G"] is None else float(queued_result["G"])
            )
            self.res["success"] = queued_result["success"]
            self.need_to_run_code = False
            if queued_result["raises"] is not None:
                raise queued_result["raises"]
            return {"success": queued_result["success"]}

        self.res["iota"] = float(iota)
        self.res["G"] = G
        self.res["success"] = True
        self.need_to_run_code = False
        return {"success": True}


def _surface_entry(x, iota, G):
    return {"boozer_surface": _FakeBoozerSurface(x, iota, G)}


class _ModuleTestCase(unittest.TestCase):
    MODULE_PATH = None
    MODULE_PREFIX = None

    def setUp(self):
        self.module = _load_module(self.MODULE_PATH, self.MODULE_PREFIX)


class PoloidalExtentModuleTests(_ModuleTestCase):
    MODULE_PATH = POLOIDAL_EXTENT_PATH
    MODULE_PREFIX = "banana_poloidal_extent"

    def test_inboard_poloidal_angles_use_inboard_midplane_zero(self):
        angles = self.module.inboard_poloidal_angles(
            np.array(
                [
                    [0.876, 0.0, 0.0],
                    [0.976, 0.0, 0.1],
                    [0.976, 0.0, -0.1],
                    [1.076, 0.0, 0.0],
                ],
                dtype=float,
            ),
            R_winding=0.976,
        )

        np.testing.assert_allclose(
            angles,
            [0.0, np.pi / 2.0, -np.pi / 2.0, np.pi],
            atol=1.0e-12,
        )

    def test_max_poloidal_extent_rad_uses_curve_gamma(self):
        curve = _FakeCurve(
            [
                [0.876, 0.0, 0.0],
                [0.976, 0.0, 0.1],
            ]
        )

        self.assertAlmostEqual(
            self.module.max_poloidal_extent_rad(curve, R_winding=0.976),
            np.pi / 2.0,
        )

    def test_smooth_constraint_returns_signed_violation_and_curve_gradient(self):
        curve = _FakeCurve([[0.976, 0.0, 0.2]])

        signed_value, grad_value, violation = (
            self.module.smooth_max_poloidal_extent_signed_constraint(
                curve,
                R_winding=0.976,
                theta_threshold=np.pi / 4.0,
                temperature=1.0e-3,
                objective_optimizable=object(),
            )
        )

        self.assertAlmostEqual(signed_value, np.pi / 4.0)
        self.assertAlmostEqual(violation, np.pi / 4.0)
        np.testing.assert_allclose(grad_value, [5.0, 0.0], atol=1.0e-12)


class Stage2ObjectiveModuleTests(_ModuleTestCase):
    MODULE_PATH = STAGE2_OBJECTIVES_PATH
    MODULE_PREFIX = "banana_stage2_objectives"

    def _assert_restored_fake_boozer_state(self, fake_boozer_surface):
        np.testing.assert_allclose(fake_boozer_surface.surface.x, [0.0, 0.0])
        self.assertAlmostEqual(fake_boozer_surface.res["iota"], 0.21)
        self.assertAlmostEqual(fake_boozer_surface.res["G"], 0.35)
        self.assertTrue(fake_boozer_surface.res["success"])

    def _build_fake_stage2_iota_runtime(self, fake_boozer_surface):
        class _FakeIotaTerm:
            def __init__(self, boozer_surface):
                self.boozer_surface = boozer_surface

            def J(self):
                if getattr(self.boozer_surface, "need_to_run_code", False):
                    res = self.boozer_surface.res
                    self.boozer_surface.run_code(res["iota"], G=res["G"])
                    self.boozer_surface.need_to_run_code = False
                return float(self.boozer_surface.res["iota"])

        class _FakeQuadraticPenalty:
            def __init__(self, term, target):
                self.term = term
                self.target = float(target)

            def J(self):
                delta = self.term.J() - self.target
                return 0.5 * delta * delta

            def dJ(self):
                return np.array([0.2, -0.1], dtype=float)

        return self.module.build_stage2_iota_runtime(
            equilibrium_file="demo.nc",
            bs=SimpleNamespace(),
            tf_coils=[object(), object()],
            major_radius=0.976,
            toroidal_flux=0.24,
            nphi=91,
            ntheta=32,
            mpol=8,
            ntor=6,
            vol_target=0.12,
            iota_target=0.2,
            iota_tolerance=5.0e-3,
            constraint_weight=None,
            num_tf_coils=2,
            mode="soft",
            weight=3.0,
            build_surface_configs_fn=lambda *_args, **_kwargs: [
                {
                    "initial_surface": SimpleNamespace(nfp=5),
                    "target_volume": 0.12,
                }
            ],
            attempt_initialize_boozer_surface_fn=lambda *_args, **_kwargs: (
                SimpleNamespace(
                    success=True,
                    boozer_surface=fake_boozer_surface,
                    solve_success=True,
                    self_intersecting=False,
                    solved_iota=0.21,
                    error_type=None,
                    error_message=None,
                )
            ),
            compute_tf_G0_fn=lambda _tf_coils: 0.35,
            iotas_cls=_FakeIotaTerm,
            quadratic_penalty_cls=_FakeQuadraticPenalty,
        )

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
            emit_diagnostics=True,
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

    def test_make_stage2_fun_fast_path_skips_diagnostics(self):
        class _JF:
            def __init__(self):
                self.x = None

            def J(self):
                return 1.23

            def dJ(self):
                return np.array([1.0, -2.0])

        class _UnexpectedDiagnostic:
            def J(self):
                raise AssertionError("diagnostic objective should not be evaluated")

            def shortest_distance(self):
                raise AssertionError("diagnostic distance should not be evaluated")

        class _UnexpectedBiotSavart:
            def B(self):
                raise AssertionError("diagnostic field should not be evaluated")

        class _UnexpectedSurface:
            def unitnormal(self):
                raise AssertionError("diagnostic normal should not be evaluated")

        fun = self.module.make_stage2_fun(
            _JF(),
            _UnexpectedBiotSavart(),
            _UnexpectedSurface(),
            _UnexpectedDiagnostic(),
            _UnexpectedDiagnostic(),
            _UnexpectedDiagnostic(),
            _UnexpectedDiagnostic(),
        )

        with mock.patch("builtins.print") as print_mock:
            value, grad = fun(np.array([0.2, -0.1]))

        self.assertAlmostEqual(value, 1.23)
        np.testing.assert_allclose(grad, [1.0, -2.0])
        print_mock.assert_not_called()

    def test_make_stage2_fun_soft_mode_computes_and_freezes_effective_weight(self):
        class _JF:
            def __init__(self):
                self.x = None

            def J(self):
                return 1.23

            def dJ(self):
                return np.array([1.0, -2.0])

        stage2_iota_runtime = SimpleNamespace(
            mode="soft",
            weight=3.0,
            effective_weight=None,
            penalty_threshold=2.0e-2,
            penalty_objective=SimpleNamespace(
                dJ=mock.Mock(
                    side_effect=AssertionError(
                        "soft penalty gradient should come from guarded evaluation"
                    )
                )
            ),
        )
        first_state = self.module.Stage2IotaState(
            iota=0.2,
            penalty=1.0e-12,
            abs_error=1.0e-6,
            feasible=True,
            solve_failed=False,
        )
        second_state = self.module.Stage2IotaState(
            iota=0.24,
            penalty=0.6,
            abs_error=0.04,
            feasible=True,
            solve_failed=False,
        )
        first_evaluation = self.module.Stage2IotaEvaluation(
            state=first_state,
            penalty_grad=np.array([0.2, -0.1]),
        )
        second_evaluation = self.module.Stage2IotaEvaluation(
            state=second_state,
            penalty_grad=np.array([0.1, -0.2]),
        )
        fun = self.module.make_stage2_fun(
            _JF(),
            _FakeBiotSavart((2, 3)),
            _FakeSurfaceNormals((1, 2, 3)),
            _FakeScalarObjective(0.12),
            _FakeScalarObjective(1.75),
            SimpleNamespace(shortest_distance=lambda: 0.055),
            _FakeScalarObjective(39.5),
            stage2_iota_runtime=stage2_iota_runtime,
            emit_diagnostics=True,
        )

        with (
            mock.patch.object(
                self.module,
                "evaluate_stage2_iota",
                side_effect=[first_evaluation, second_evaluation],
            ),
            mock.patch("builtins.print"),
        ):
            first_value, first_grad = fun(np.array([0.2, -0.1]))
            second_value, second_grad = fun(np.array([0.2, -0.1]))

        expected_effective_weight = 3.0 * 1.23 / 2.0e-2
        self.assertAlmostEqual(
            stage2_iota_runtime.effective_weight,
            expected_effective_weight,
        )
        self.assertAlmostEqual(
            first_value,
            1.23 + expected_effective_weight * first_state.penalty,
        )
        np.testing.assert_allclose(
            first_grad,
            [1.0, -2.0] + expected_effective_weight * np.array([0.2, -0.1]),
        )
        self.assertAlmostEqual(
            second_value,
            1.23 + expected_effective_weight * second_state.penalty,
        )
        np.testing.assert_allclose(
            second_grad,
            [1.0, -2.0] + expected_effective_weight * np.array([0.1, -0.2]),
        )

    def test_make_stage2_fun_soft_mode_rejects_failed_iota_solve(self):
        class _JF:
            def __init__(self):
                self.x = None

            def J(self):
                return 1.23

            def dJ(self):
                return np.array([1.0, -2.0])

        stage2_iota_runtime = SimpleNamespace(
            mode="soft",
            weight=3.0,
            effective_weight=None,
            penalty_threshold=2.0e-2,
            penalty_objective=SimpleNamespace(
                dJ=mock.Mock(
                    side_effect=AssertionError("soft penalty gradient should not run")
                )
            ),
        )
        soft_state = self.module.Stage2IotaState(
            iota=0.24,
            penalty=0.4,
            abs_error=0.04,
            feasible=True,
            solve_failed=False,
        )
        failed_state = self.module.Stage2IotaState(
            iota=0.24,
            penalty=0.4,
            abs_error=0.04,
            feasible=False,
            solve_failed=True,
        )
        soft_evaluation = self.module.Stage2IotaEvaluation(
            state=soft_state,
            penalty_grad=np.array([0.2, -0.1]),
        )
        failed_evaluation = self.module.Stage2IotaEvaluation(
            state=failed_state,
            penalty_grad=None,
        )
        jf = _JF()
        fun = self.module.make_stage2_fun(
            jf,
            _FakeBiotSavart((2, 3)),
            _FakeSurfaceNormals((1, 2, 3)),
            _FakeScalarObjective(0.12),
            _FakeScalarObjective(1.75),
            SimpleNamespace(shortest_distance=lambda: 0.055),
            _FakeScalarObjective(39.5),
            stage2_iota_runtime=stage2_iota_runtime,
            emit_diagnostics=True,
        )

        with (
            mock.patch.object(
                self.module,
                "evaluate_stage2_iota",
                side_effect=[soft_evaluation, failed_evaluation],
            ),
            mock.patch("builtins.print") as print_mock,
        ):
            value, grad = fun(np.array([0.3, -0.2]))
            failed_value, failed_grad = fun(np.array([0.4, -0.3]))

        expected_effective_weight = 3.0 * 1.23 / 0.4
        self.assertAlmostEqual(value, 1.23 + expected_effective_weight * 0.4)
        np.testing.assert_allclose(
            grad,
            [1.0, -2.0] + expected_effective_weight * np.array([0.2, -0.1]),
        )
        self.assertAlmostEqual(failed_value, 2.46)
        np.testing.assert_allclose(failed_grad, [2.0, -4.0])
        np.testing.assert_allclose(jf.x, [0.4, -0.3])
        self.assertIn("IotaSolveFailed=1", print_mock.call_args[0][0])

    def test_make_stage2_fun_soft_mode_first_failure_adds_constant_reject_offset(self):
        class _JF:
            def __init__(self):
                self.x = None

            def J(self):
                return 0.4

            def dJ(self):
                return np.array([0.5, -0.25])

        stage2_iota_runtime = SimpleNamespace(
            mode="soft",
            weight=3.0,
            effective_weight=None,
            penalty_threshold=2.0e-2,
            penalty_objective=SimpleNamespace(
                dJ=mock.Mock(
                    side_effect=AssertionError("soft penalty gradient should not run")
                )
            ),
        )
        failed_state = self.module.Stage2IotaState(
            iota=0.24,
            penalty=0.4,
            abs_error=0.04,
            feasible=False,
            solve_failed=True,
        )
        failed_evaluation = self.module.Stage2IotaEvaluation(
            state=failed_state,
            penalty_grad=None,
        )
        jf = _JF()
        fun = self.module.make_stage2_fun(
            jf,
            _FakeBiotSavart((2, 3)),
            _FakeSurfaceNormals((1, 2, 3)),
            _FakeScalarObjective(0.12),
            _FakeScalarObjective(1.75),
            SimpleNamespace(shortest_distance=lambda: 0.055),
            _FakeScalarObjective(39.5),
            stage2_iota_runtime=stage2_iota_runtime,
        )

        with (
            mock.patch.object(
                self.module,
                "evaluate_stage2_iota",
                return_value=failed_evaluation,
            ),
            mock.patch("builtins.print"),
        ):
            value, grad = fun(np.array([0.3, -0.2]))

        self.assertAlmostEqual(value, 1.4)
        np.testing.assert_allclose(grad, [0.5, -0.25])
        np.testing.assert_allclose(jf.x, [0.3, -0.2])

    def test_evaluate_stage2_alm_problem_exposes_constraint_payload(self):
        base_objective = _FakeBaseObjective(3.5, [1.2, -0.5])
        new_surf = _FakeSurfaceNormals((2, 2, 3))
        new_bs = _FakeBiotSavart((4, 3))
        Jf = _FakeScalarObjective(0.25)
        Jls = _FakeLengthObjective(2.2, [0.3, 0.4])
        Jccdist = _UnexpectedCurveDistance(0.05, 0.04)
        Jc = _FakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5)
        banana_current = _FakeCurrentObjective(9500.0, [0.7, -0.4])

        def fake_augmented(
            base_value, base_grad, signed_values, grads, multipliers, penalty
        ):
            self.assertAlmostEqual(base_value, 3.5)
            np.testing.assert_allclose(base_grad, [1.2, -0.5])
            np.testing.assert_allclose(
                signed_values,
                [-0.16, 0.01875, 0.1, -0.40625],
            )
            np.testing.assert_allclose(grads[0], [12.0, 4.0])
            np.testing.assert_allclose(grads[1], [0.0225, -0.0025])
            np.testing.assert_allclose(grads[2], [0.15, 0.2])
            np.testing.assert_allclose(grads[3], [4.375e-5, -2.5e-5])
            np.testing.assert_allclose(multipliers, [0.1, 0.2, 0.3, 0.4])
            self.assertAlmostEqual(penalty, 12.0)
            return {
                "total": 9.0,
                "grad": np.array([7.0, -3.0]),
                "stationarity_norm": 0.5,
            }

        with (
            mock.patch.object(
                self.module,
                "augmented_inequality_objective",
                side_effect=fake_augmented,
            ),
            mock.patch("builtins.print"),
        ):
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
                banana_current=banana_current,
                banana_current_max_A=16000.0,
                distance_smoothing=0.005,
                curvature_smoothing=0.02,
                multipliers=np.array([0.1, 0.2, 0.3, 0.4]),
                penalty=12.0,
                stage2_constraint_activity_tolerances=lambda ds, cs: [
                    1e-3,
                    ds * 4.0,
                    cs * 4.0,
                    1e-3,
                ],
                smooth_min_distance_signed_constraint=lambda *_args: (
                    -0.008,
                    np.array([0.6, 0.2]),
                ),
                smooth_max_curvature_signed_constraint=lambda *_args: (
                    0.75,
                    np.array([0.9, -0.1]),
                ),
            )

        np.testing.assert_allclose(base_objective.x, [0.25, -0.4])
        self.assertEqual(
            result["constraint_names"],
            [
                "coil_coil_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "banana_current_upper_bound",
            ],
        )
        np.testing.assert_allclose(
            result["dual_update_values"],
            [-0.16, 0.01875, 0.1, -0.40625],
        )
        np.testing.assert_allclose(
            result["hard_signed_constraint_values"],
            [-0.16, 0.025, 0.1, -0.40625],
        )
        np.testing.assert_allclose(
            result["hard_violation_values"],
            [0.0, 0.025, 0.1, 0.0],
        )
        np.testing.assert_allclose(
            result["surrogate_signed_constraint_values"],
            [-0.16, 0.01875, 0.1, -0.40625],
        )
        np.testing.assert_allclose(
            result["hard_dual_update_values"],
            [-0.16, 0.025, 0.1, -0.40625],
        )
        np.testing.assert_allclose(
            result["feasibility_values"],
            [0.0, 0.025, 0.1, 0.0],
        )
        np.testing.assert_allclose(
            result["constraint_activity_tolerances"],
            [0.4, 0.002, 5.0e-4, 6.25e-8],
        )
        np.testing.assert_allclose(
            result["constraint_scales"],
            [0.05, 40.0, 2.0, 16000.0],
        )
        self.assertEqual(
            result["constraint_blocks"],
            ["geometry", "geometry", "geometry", "current"],
        )
        self.assertEqual(
            result["objective_value_kinds"],
            ["surrogate", "surrogate", "hard", "hard"],
        )
        self.assertEqual(
            result["dual_update_value_kinds"],
            ["hard", "hard", "hard", "hard"],
        )
        self.assertEqual(
            result["feasibility_value_kinds"],
            ["hard", "hard", "hard", "hard"],
        )
        np.testing.assert_allclose(
            result["raw_dual_update_values"],
            [-0.008, 0.75, 0.2, -6500.0],
        )
        np.testing.assert_allclose(
            result["raw_feasibility_values"],
            [0.0, 1.0, 0.2, 0.0],
        )
        np.testing.assert_allclose(
            result["raw_constraint_activity_tolerances"],
            [0.02, 0.08, 1e-3, 1e-3],
        )
        self.assertAlmostEqual(result["max_feasibility_violation"], 0.1)
        self.assertAlmostEqual(result["total"], 9.0)
        np.testing.assert_allclose(result["grad"], [7.0, -3.0])

    def test_stage2_normalized_alm_constraints_pass_directional_taylor_test(self):
        alm_utils = _load_module(ALM_UTILS_PATH, "banana_alm_utils")
        base_objective = _XAwareQuadraticObjective(
            None,
            constant=1.5,
            linear=[0.4, -0.2],
            quadratic=0.3,
        )
        Jls = _XAwareQuadraticObjective(
            base_objective,
            constant=2.15,
            linear=[0.2, -0.1],
        )
        banana_current = _XAwareCurrentObjective(
            base_objective,
            constant=16500.0,
            linear=[120.0, -80.0],
        )
        Jccdist = _UnexpectedCurveDistance(0.05, 0.04)
        Jc = _FakeCurvatureObjective(40.0, [41.0, 39.5], 7.5)
        distance_constraint = _affine_signed_constraint(
            base_objective,
            0.012,
            [0.004, -0.002],
            include_violation=False,
        )
        curvature_constraint = _affine_signed_constraint(
            base_objective,
            0.8,
            [0.05, 0.03],
            include_violation=False,
        )

        def evaluate_problem(x, multipliers, penalty):
            return self.module.evaluate_stage2_alm_problem(
                dofs=np.asarray(x, dtype=float),
                base_objective=base_objective,
                new_bs=_FakeBiotSavart((4, 3)),
                new_surf=_FakeSurfaceNormals((2, 2, 3)),
                Jf=_FakeScalarObjective(0.0),
                Jls=Jls,
                length_target=2.0,
                Jccdist=Jccdist,
                Jc=Jc,
                banana_current=banana_current,
                banana_current_max_A=16000.0,
                distance_smoothing=0.005,
                curvature_smoothing=0.02,
                multipliers=multipliers,
                penalty=penalty,
                stage2_constraint_activity_tolerances=lambda ds, cs: [
                    1.0e-3,
                    ds * 4.0,
                    cs * 4.0,
                    1.0e-3,
                ],
                smooth_min_distance_signed_constraint=distance_constraint,
                smooth_max_curvature_signed_constraint=curvature_constraint,
            )

        result = alm_utils.run_directional_taylor_test(
            evaluate_problem,
            np.array([0.2, -0.3]),
            np.array([0.05, 0.1, 0.15, 0.2]),
            7.0,
            direction=np.array([0.6, -0.4]),
            epsilons=TAYLOR_TEST_EPSILONS,
        )

        self.assertTrue(result["passed"], result)

    def test_evaluate_stage2_alm_problem_fast_path_skips_report_diagnostics(self):
        class _UnexpectedBiotSavart:
            def B(self):
                raise AssertionError("diagnostic field should not be evaluated")

        class _UnexpectedSurfaceNormals:
            def unitnormal(self):
                raise AssertionError("diagnostic normals should not be evaluated")

        class _UnexpectedFluxObjective:
            def J(self):
                raise AssertionError("diagnostic flux should not be evaluated")

        result = self.module.evaluate_stage2_alm_problem(
            dofs=np.array([0.25, -0.4]),
            base_objective=_FakeBaseObjective(3.5, [1.2, -0.5]),
            new_bs=_UnexpectedBiotSavart(),
            new_surf=_UnexpectedSurfaceNormals(),
            Jf=_UnexpectedFluxObjective(),
            Jls=_FakeLengthObjective(2.2, [0.3, 0.4]),
            length_target=2.0,
            Jccdist=_UnexpectedCurveDistance(0.05, 0.04),
            Jc=_FakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5),
            banana_current=_FakeCurrentObjective(9500.0, [0.7, -0.4]),
            banana_current_max_A=16000.0,
            distance_smoothing=0.005,
            curvature_smoothing=0.02,
            multipliers=np.array([0.1, 0.2, 0.3, 0.4]),
            penalty=12.0,
            stage2_constraint_activity_tolerances=lambda ds, cs: [
                1e-3,
                ds * 4.0,
                cs * 4.0,
                1e-3,
            ],
            smooth_min_distance_signed_constraint=lambda *_args: (
                -0.008,
                np.array([0.6, 0.2]),
            ),
            smooth_max_curvature_signed_constraint=lambda *_args: (
                0.75,
                np.array([0.9, -0.1]),
            ),
        )

        self.assertNotIn("diagnostics_included", result)
        self.assertAlmostEqual(result["base_value"], 3.5)
        self.assertEqual(
            result["constraint_names"],
            [
                "coil_coil_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "banana_current_upper_bound",
            ],
        )
        expected_alm_state = {
            "hard_signed_constraint_values": [-0.16, 0.025, 0.1, -0.40625],
            "surrogate_signed_constraint_values": [-0.16, 0.01875, 0.1, -0.40625],
            "hard_violation_values": [0.0, 0.025, 0.1, 0.0],
            "constraint_activity_tolerances": [0.4, 0.002, 0.0005, 6.25e-8],
            "max_feasibility_violation": 0.1,
            "total": 3.5887760416666665,
            "grad": [1.4345625, -0.2010625],
            "stationarity_norm": 1.4485840311533535,
        }
        for key, expected in expected_alm_state.items():
            np.testing.assert_allclose(
                result[key],
                expected,
                rtol=1e-12,
                atol=0.0,
            )

    def test_evaluate_stage2_alm_problem_sanitizes_nonfinite_inputs(self):
        base_objective = _FakeBaseObjective(np.nan, [np.inf, np.nan])
        new_surf = _FakeSurfaceNormals((2, 2, 3))
        new_bs = _FakeBiotSavart((4, 3))
        Jf = _FakeScalarObjective(0.25)
        Jls = _FakeLengthObjective(2.2, [0.3, 0.4])
        Jccdist = _FakeCurveDistance(0.05, 0.04)
        Jc = _FakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5)
        banana_current = _FakeCurrentObjective(9500.0, [0.7, -0.4])

        def fake_augmented(
            base_value, base_grad, signed_values, grads, multipliers, penalty
        ):
            self.assertAlmostEqual(base_value, 1.0)
            np.testing.assert_allclose(base_grad, [0.0, 0.0])
            np.testing.assert_allclose(signed_values, [20.0, 0.01875, 0.1, -0.40625])
            np.testing.assert_allclose(grads[0], [0.0, 0.0])
            np.testing.assert_allclose(grads[1], [0.0225, -0.0025])
            np.testing.assert_allclose(grads[2], [0.15, 0.2])
            np.testing.assert_allclose(grads[3], [4.375e-5, -2.5e-5])
            np.testing.assert_allclose(multipliers, [0.1, 0.2, 0.3, 0.4])
            self.assertAlmostEqual(penalty, 12.0)
            return {
                "total": 9.0,
                "grad": np.array([7.0, -3.0]),
                "base_grad": np.array([0.0, 0.0]),
                "stationarity_norm": 0.5,
            }

        with (
            mock.patch.object(
                self.module,
                "augmented_inequality_objective",
                side_effect=fake_augmented,
            ),
            mock.patch("builtins.print"),
        ):
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
                banana_current=banana_current,
                banana_current_max_A=16000.0,
                distance_smoothing=0.005,
                curvature_smoothing=0.02,
                multipliers=np.array([0.1, 0.2, 0.3, 0.4]),
                penalty=12.0,
                stage2_constraint_activity_tolerances=lambda ds, cs: [
                    1e-3,
                    ds * 4.0,
                    cs * 4.0,
                    1e-3,
                ],
                smooth_min_distance_signed_constraint=lambda *_args: (
                    np.nan,
                    np.array([np.nan, np.nan]),
                ),
                smooth_max_curvature_signed_constraint=lambda *_args: (
                    0.75,
                    np.array([0.9, -0.1]),
                ),
            )

        self.assertTrue(result["nonfinite_inputs_sanitized"])
        self.assertEqual(
            result["nonfinite_input_fields"],
            [
                "base_grad",
                "base_value",
                "constraint_values[0]",
                "constraint_grads[0]",
                "hard_signed_constraint_values[0]",
                "hard_violation_values[0]",
            ],
        )
        self.assertTrue(result["nonfinite_evaluation"])
        self.assertEqual(
            result["nonfinite_fields"],
            [
                "base_grad",
                "base_value",
                "constraint_values[0]",
                "constraint_grads[0]",
                "hard_signed_constraint_values[0]",
                "hard_violation_values[0]",
            ],
        )
        self.assertTrue(np.isnan(result["total"]))
        np.testing.assert_allclose(
            result["dual_update_values"], [20.0, 0.01875, 0.1, -0.40625]
        )
        np.testing.assert_allclose(
            result["raw_dual_update_values"], [1.0, 0.75, 0.2, -6500.0]
        )
        np.testing.assert_allclose(
            result["hard_signed_constraint_values"],
            [20.0, 0.025, 0.1, -0.40625],
        )
        np.testing.assert_allclose(
            result["raw_hard_signed_constraint_values"],
            [1.0, 1.0, 0.2, -6500.0],
        )
        np.testing.assert_allclose(
            result["hard_violation_values"],
            [20.0, 0.025, 0.1, 0.0],
        )
        np.testing.assert_allclose(
            result["raw_hard_violation_values"],
            [1.0, 1.0, 0.2, 0.0],
        )
        np.testing.assert_allclose(
            result["surrogate_signed_constraint_values"],
            [20.0, 0.01875, 0.1, -0.40625],
        )
        np.testing.assert_allclose(
            result["raw_surrogate_signed_constraint_values"],
            [1.0, 0.75, 0.2, -6500.0],
        )
        np.testing.assert_allclose(result["constraint_grads"][0], [0.0, 0.0])

    def test_stage2_constraint_activity_tolerances_track_smoothing_windows(self):
        tolerances = self.module.stage2_constraint_activity_tolerances(0.005, 0.05)
        self.assertEqual(tolerances, [1e-3, 0.02, 0.2, 1e-3])

    def test_stage2_constraint_activity_tolerances_accept_explicit_endcaps(self):
        tolerances = self.module.stage2_constraint_activity_tolerances(
            0.005,
            0.05,
            length_tolerance=2e-3,
            banana_current_tolerance=3e-3,
        )
        self.assertEqual(tolerances, [2e-3, 0.02, 0.2, 3e-3])

    def test_evaluate_stage2_alm_problem_caps_banana_current_by_magnitude(self):
        base_objective = _FakeAlgebraicObjective(
            3.5, [1.2, -0.5], projected_gradient=[0.25, -0.4]
        )
        new_bs = _FakeBiotSavart((1, 1, 3))
        new_surf = _FakeSurfaceNormals((1, 1, 3))
        Jf = _FakeAlgebraicObjective(3.5, [1.2, -0.5])
        Jls = _FakeLengthObjective(2.2, [0.3, 0.4])
        Jccdist = _FakeCurveDistance(0.05, 0.04)
        Jc = _FakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5)
        banana_current = _FakeCurrentObjective(-17000.0, [0.7, -0.4])

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
            banana_current=banana_current,
            banana_current_max_A=16000.0,
            distance_smoothing=0.005,
            curvature_smoothing=0.02,
            multipliers=np.array([0.1, 0.2, 0.3, 0.4]),
            penalty=12.0,
            stage2_constraint_activity_tolerances=lambda ds, cs: [
                1e-3,
                ds * 4.0,
                cs * 4.0,
                1e-3,
            ],
            smooth_min_distance_signed_constraint=lambda *_args: (
                -0.008,
                np.array([0.6, 0.2]),
            ),
            smooth_max_curvature_signed_constraint=lambda *_args: (
                0.75,
                np.array([0.9, -0.1]),
            ),
        )

        self.assertEqual(result["constraint_names"][3], "banana_current_upper_bound")
        self.assertAlmostEqual(result["dual_update_values"][3], 0.0625)
        self.assertAlmostEqual(result["feasibility_values"][3], 0.0625)
        self.assertAlmostEqual(result["raw_dual_update_values"][3], 1000.0)
        np.testing.assert_allclose(result["constraint_grads"][3], [-4.375e-5, 2.5e-5])

    def test_evaluate_stage2_alm_problem_uses_activity_tolerance_helper(self):
        base_objective = _FakeAlgebraicObjective(
            3.5, [1.2, -0.5], projected_gradient=[0.25, -0.4]
        )
        new_bs = _FakeBiotSavart((1, 1, 3))
        new_surf = _FakeSurfaceNormals((1, 1, 3))
        Jf = _FakeAlgebraicObjective(3.5, [1.2, -0.5])
        Jls = _FakeLengthObjective(2.2, [0.3, 0.4])
        Jccdist = _FakeCurveDistance(0.05, 0.04)
        Jc = _FakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5)
        banana_current = _FakeCurrentObjective(9500.0, [0.7, -0.4])

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
            banana_current=banana_current,
            banana_current_max_A=16000.0,
            distance_smoothing=0.005,
            curvature_smoothing=0.02,
            multipliers=np.array([0.1, 0.2, 0.3, 0.4]),
            penalty=12.0,
            stage2_constraint_activity_tolerances=lambda ds, cs: [
                2e-3,
                ds * 5.0,
                cs * 6.0,
                7e-3,
            ],
            smooth_min_distance_signed_constraint=lambda *_args: (
                -0.008,
                np.array([0.6, 0.2]),
            ),
            smooth_max_curvature_signed_constraint=lambda *_args: (
                0.75,
                np.array([0.9, -0.1]),
            ),
        )

        np.testing.assert_allclose(
            result["constraint_activity_tolerances"],
            [0.5, 0.003, 1.0e-3, 4.375e-7],
        )
        np.testing.assert_allclose(
            result["raw_constraint_activity_tolerances"],
            [0.025, 0.12, 2e-3, 7e-3],
        )

    def test_evaluate_stage2_alm_problem_includes_iota_penalty_constraint(self):
        base_objective = _FakeBaseObjective(3.5, [1.2, -0.5])
        new_surf = _FakeSurfaceNormals((2, 2, 3))
        new_bs = _FakeBiotSavart((4, 3))
        Jf = _FakeScalarObjective(0.25)
        Jls = _FakeLengthObjective(2.2, [0.3, 0.4])
        Jccdist = _FakeCurveDistance(0.05, 0.04)
        Jc = _FakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5)
        banana_current = _FakeCurrentObjective(9500.0, [0.7, -0.4])
        stage2_iota_runtime = SimpleNamespace(
            mode="alm",
            target=0.2,
            tolerance=0.05,
            penalty_threshold=0.5,
            iota_term=SimpleNamespace(J=lambda: 0.18),
            penalty_objective=SimpleNamespace(
                dJ=mock.Mock(
                    side_effect=AssertionError(
                        "ALM iota gradient should come from guarded evaluation"
                    )
                )
            ),
        )
        iota_state = self.module.Stage2IotaState(
            iota=0.18,
            penalty=0.6,
            abs_error=0.02,
            feasible=False,
            solve_failed=False,
        )
        iota_evaluation = self.module.Stage2IotaEvaluation(
            state=iota_state,
            penalty_grad=np.array([0.2, 0.1]),
        )

        def fake_augmented(
            base_value, base_grad, signed_values, grads, multipliers, penalty
        ):
            self.assertAlmostEqual(base_value, 3.5)
            np.testing.assert_allclose(base_grad, [1.2, -0.5])
            np.testing.assert_allclose(
                signed_values,
                [-0.16, 0.01875, 0.1, -0.40625, 0.2],
            )
            np.testing.assert_allclose(grads[4], [0.4, 0.2])
            np.testing.assert_allclose(multipliers, [0.1, 0.2, 0.3, 0.4, 0.5])
            self.assertAlmostEqual(penalty, 12.0)
            return {
                "total": 9.5,
                "grad": np.array([7.0, -3.0]),
                "stationarity_norm": 0.5,
            }

        with (
            mock.patch.object(
                self.module,
                "evaluate_stage2_iota",
                return_value=iota_evaluation,
            ),
            mock.patch.object(
                self.module,
                "augmented_inequality_objective",
                side_effect=fake_augmented,
            ),
            mock.patch("builtins.print"),
        ):
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
                banana_current=banana_current,
                banana_current_max_A=16000.0,
                distance_smoothing=0.005,
                curvature_smoothing=0.02,
                multipliers=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                penalty=12.0,
                stage2_constraint_activity_tolerances=lambda ds, cs: [
                    1e-3,
                    ds * 4.0,
                    cs * 4.0,
                    1e-3,
                    0.5,
                ],
                smooth_min_distance_signed_constraint=lambda *_args: (
                    -0.008,
                    np.array([0.6, 0.2]),
                ),
                smooth_max_curvature_signed_constraint=lambda *_args: (
                    0.75,
                    np.array([0.9, -0.1]),
                ),
                stage2_iota_runtime=stage2_iota_runtime,
            )

        self.assertEqual(
            result["constraint_names"],
            [
                "coil_coil_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "banana_current_upper_bound",
                "iota_penalty",
            ],
        )
        np.testing.assert_allclose(
            result["dual_update_values"],
            [-0.16, 0.01875, 0.1, -0.40625, 0.2],
        )
        np.testing.assert_allclose(
            result["raw_dual_update_values"],
            [-0.008, 0.75, 0.2, -6500.0, 0.1],
        )
        np.testing.assert_allclose(
            result["hard_violation_values"],
            [0.0, 0.025, 0.1, 0.0, 0.2],
        )
        np.testing.assert_allclose(
            result["raw_hard_violation_values"],
            [0.0, 1.0, 0.2, 0.0, 0.1],
        )
        np.testing.assert_allclose(
            result["constraint_activity_tolerances"],
            [0.4, 0.002, 0.0005, 6.25e-8, 1.0],
        )
        np.testing.assert_allclose(
            result["raw_constraint_activity_tolerances"],
            [0.02, 0.08, 1e-3, 1e-3, 0.5],
        )

    def test_evaluate_stage2_alm_problem_rejects_failed_iota_solves_without_penalty_gradient(
        self,
    ):
        base_objective = _FakeBaseObjective(3.5, [1.2, -0.5])
        new_surf = _FakeSurfaceNormals((2, 2, 3))
        new_bs = _FakeBiotSavart((4, 3))
        Jf = _FakeScalarObjective(0.25)
        Jls = _FakeLengthObjective(2.2, [0.3, 0.4])
        Jccdist = _FakeCurveDistance(0.05, 0.04)
        Jc = _FakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5)
        banana_current = _FakeCurrentObjective(9500.0, [0.7, -0.4])
        stage2_iota_runtime = SimpleNamespace(
            mode="alm",
            target=0.2,
            tolerance=0.05,
            penalty_threshold=0.5,
            iota_term=SimpleNamespace(J=lambda: 0.18),
            penalty_objective=SimpleNamespace(
                dJ=mock.Mock(
                    side_effect=AssertionError("penalty gradient should not run")
                )
            ),
        )

        def fake_augmented(
            base_value, base_grad, signed_values, grads, multipliers, penalty
        ):
            self.assertAlmostEqual(base_value, 3.5)
            np.testing.assert_allclose(base_grad, [1.2, -0.5])
            np.testing.assert_allclose(
                signed_values,
                [-0.16, 0.01875, 0.1, -0.40625, 2.0],
            )
            np.testing.assert_allclose(grads[4], [0.0, 0.0])
            np.testing.assert_allclose(multipliers, [0.1, 0.2, 0.3, 0.4, 0.5])
            self.assertAlmostEqual(penalty, 12.0)
            return {
                "total": 9.5,
                "grad": np.array([7.0, -3.0]),
                "stationarity_norm": 0.5,
            }

        failed_state = self.module.Stage2IotaState(
            iota=0.18,
            penalty=0.01,
            abs_error=0.02,
            feasible=False,
            solve_failed=True,
        )
        failed_evaluation = self.module.Stage2IotaEvaluation(
            state=failed_state,
            penalty_grad=None,
        )
        with (
            mock.patch.object(
                self.module,
                "evaluate_stage2_iota",
                return_value=failed_evaluation,
            ),
            mock.patch.object(
                self.module,
                "augmented_inequality_objective",
                side_effect=fake_augmented,
            ),
            mock.patch("builtins.print"),
        ):
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
                banana_current=banana_current,
                banana_current_max_A=16000.0,
                distance_smoothing=0.005,
                curvature_smoothing=0.02,
                multipliers=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
                penalty=12.0,
                stage2_constraint_activity_tolerances=lambda ds, cs: [
                    1e-3,
                    ds * 4.0,
                    cs * 4.0,
                    1e-3,
                    0.5,
                ],
                smooth_min_distance_signed_constraint=lambda *_args: (
                    -0.008,
                    np.array([0.6, 0.2]),
                ),
                smooth_max_curvature_signed_constraint=lambda *_args: (
                    0.75,
                    np.array([0.9, -0.1]),
                ),
                stage2_iota_runtime=stage2_iota_runtime,
            )

        np.testing.assert_allclose(
            result["constraint_grads"][4],
            [0.0, 0.0],
        )
        np.testing.assert_allclose(
            result["dual_update_values"],
            [-0.16, 0.01875, 0.1, -0.40625, 2.0],
        )
        np.testing.assert_allclose(
            result["raw_dual_update_values"],
            [-0.008, 0.75, 0.2, -6500.0, 1.0],
        )
        np.testing.assert_allclose(
            result["hard_violation_values"],
            [0.0, 0.025, 0.1, 0.0, 2.0],
        )
        np.testing.assert_allclose(
            result["raw_hard_violation_values"],
            [0.0, 1.0, 0.2, 0.0, 1.0],
        )

    def test_build_stage2_iota_runtime_instruments_boozer_hot_loop(self):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(fake_boozer_surface)

        self.assertAlmostEqual(runtime.initial_state.iota, 0.21)
        self.assertAlmostEqual(runtime.initial_state.penalty, 5.0e-5)
        self.assertFalse(runtime.initial_state.solve_failed)
        self.assertEqual(runtime.stats.runtime_calls, 0)
        self.assertIsNotNone(runtime.guarded_boozer_evaluator)

        fake_boozer_surface.need_to_run_code = True
        state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertAlmostEqual(state.iota, 0.21)
        self.assertFalse(state.solve_failed)
        self.assertEqual(runtime.stats.runtime_calls, 1)
        self.assertGreaterEqual(runtime.stats.runtime_seconds, 0.0)

    def test_evaluate_stage2_iota_state_guarded_path_does_not_require_penalty_gradient(
        self,
    ):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(fake_boozer_surface)
        runtime.penalty_objective = SimpleNamespace(
            J=lambda: 5.0e-5,
            dJ=mock.Mock(
                side_effect=AssertionError("state-only path should not evaluate dJ")
            ),
        )
        fake_boozer_surface.need_to_run_code = True

        state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertAlmostEqual(state.iota, 0.21)
        self.assertFalse(state.solve_failed)
        self.assertEqual(runtime.stats.runtime_calls, 1)

    def test_build_stage2_iota_runtime_restores_last_successful_state_on_failed_solve(
        self,
    ):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(fake_boozer_surface)
        fake_boozer_surface.queue_result(
            surface_x=[9.0, -4.0],
            iota=0.41,
            G=0.72,
            success=False,
        )
        fake_boozer_surface.need_to_run_code = True

        state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertAlmostEqual(state.iota, 0.21)
        self.assertAlmostEqual(state.penalty, 5.0e-5)
        self.assertFalse(state.feasible)
        self.assertTrue(state.solve_failed)
        self.assertEqual(runtime.stats.runtime_calls, 1)
        self.assertTrue(runtime.guarded_boozer_evaluator.last_solve_failed)
        self._assert_restored_fake_boozer_state(fake_boozer_surface)
        self.assertEqual(fake_boozer_surface.calls, [(0.21, 0.35)])

        second_state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertFalse(second_state.feasible)
        self.assertTrue(second_state.solve_failed)
        self.assertEqual(runtime.stats.runtime_calls, 1)

    def test_build_stage2_iota_runtime_restores_last_successful_state_on_boozer_exception(
        self,
    ):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(fake_boozer_surface)
        fake_boozer_surface.queue_result(
            surface_x=[9.0, -4.0],
            iota=0.41,
            G=0.72,
            success=False,
            raises=RuntimeError("boom"),
        )
        fake_boozer_surface.need_to_run_code = True

        state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertAlmostEqual(state.iota, 0.21)
        self.assertAlmostEqual(state.penalty, 5.0e-5)
        self.assertFalse(state.feasible)
        self.assertTrue(state.solve_failed)
        self.assertEqual(runtime.stats.runtime_calls, 1)
        self._assert_restored_fake_boozer_state(fake_boozer_surface)

        second_state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertFalse(second_state.feasible)
        self.assertTrue(second_state.solve_failed)
        self.assertEqual(runtime.stats.runtime_calls, 1)

    def test_build_stage2_iota_runtime_keeps_last_successful_snapshot_across_failures(
        self,
    ):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(fake_boozer_surface)
        fake_boozer_surface.queue_result(
            surface_x=[1.5, -2.5],
            iota=0.24,
            G=0.38,
            success=True,
        )
        fake_boozer_surface.need_to_run_code = True

        success_state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertAlmostEqual(success_state.iota, 0.24)
        self.assertFalse(success_state.solve_failed)
        np.testing.assert_allclose(fake_boozer_surface.surface.x, [1.5, -2.5])
        self.assertAlmostEqual(fake_boozer_surface.res["iota"], 0.24)
        self.assertAlmostEqual(fake_boozer_surface.res["G"], 0.38)

        fake_boozer_surface.queue_result(
            surface_x=[7.0, 8.0],
            iota=0.44,
            G=0.91,
            success=False,
        )
        fake_boozer_surface.need_to_run_code = True

        failure_state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertAlmostEqual(failure_state.iota, 0.24)
        self.assertFalse(failure_state.feasible)
        self.assertTrue(failure_state.solve_failed)
        self.assertEqual(runtime.stats.runtime_calls, 2)
        np.testing.assert_allclose(fake_boozer_surface.surface.x, [1.5, -2.5])
        self.assertAlmostEqual(fake_boozer_surface.res["iota"], 0.24)
        self.assertAlmostEqual(fake_boozer_surface.res["G"], 0.38)
        np.testing.assert_allclose(
            runtime.guarded_boozer_evaluator.last_successful_state.surface_dofs,
            [1.5, -2.5],
        )
        self.assertAlmostEqual(
            runtime.guarded_boozer_evaluator.last_successful_state.iota,
            0.24,
        )
        self.assertAlmostEqual(
            runtime.guarded_boozer_evaluator.last_successful_state.G,
            0.38,
        )

    def test_build_stage2_iota_runtime_sets_no_failure_reason_on_successful_solve(self):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(fake_boozer_surface)
        fake_boozer_surface.need_to_run_code = True

        state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertFalse(state.solve_failed)
        self.assertIsNone(runtime.guarded_boozer_evaluator.last_failure_reason)

    def test_build_stage2_iota_runtime_restores_state_on_self_intersecting_solve(self):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(fake_boozer_surface)
        fake_boozer_surface.queue_result(
            surface_x=[3.5, -1.5],
            iota=0.27,
            G=0.42,
            success=True,
        )
        fake_boozer_surface.surface.set_self_intersecting(True)
        fake_boozer_surface.need_to_run_code = True

        state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertAlmostEqual(state.iota, 0.21)
        self.assertAlmostEqual(state.penalty, 5.0e-5)
        self.assertFalse(state.feasible)
        self.assertTrue(state.solve_failed)
        self.assertEqual(
            runtime.guarded_boozer_evaluator.last_failure_reason,
            "self_intersecting",
        )
        self._assert_restored_fake_boozer_state(fake_boozer_surface)
        self.assertEqual(fake_boozer_surface.surface.self_intersection_calls, 1)

    def test_build_stage2_iota_runtime_treats_self_intersection_check_exception_as_failure(
        self,
    ):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(fake_boozer_surface)
        fake_boozer_surface.queue_result(
            surface_x=[6.0, 2.0],
            iota=0.33,
            G=0.55,
            success=True,
        )
        fake_boozer_surface.surface.set_self_intersecting_raises(
            RuntimeError("ground missing")
        )
        fake_boozer_surface.need_to_run_code = True

        state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertAlmostEqual(state.iota, 0.21)
        self.assertTrue(state.solve_failed)
        self.assertEqual(
            runtime.guarded_boozer_evaluator.last_failure_reason,
            "self_intersecting",
        )
        self._assert_restored_fake_boozer_state(fake_boozer_surface)

    def test_build_stage2_iota_runtime_does_not_check_self_intersection_on_solve_failure(
        self,
    ):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(fake_boozer_surface)
        fake_boozer_surface.queue_result(
            surface_x=[9.0, -4.0],
            iota=0.41,
            G=0.72,
            success=False,
        )
        fake_boozer_surface.need_to_run_code = True

        state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertTrue(state.solve_failed)
        self.assertEqual(
            runtime.guarded_boozer_evaluator.last_failure_reason,
            "solve_failed",
        )
        self.assertEqual(fake_boozer_surface.surface.self_intersection_calls, 0)

    def test_build_stage2_iota_runtime_recovers_after_self_intersection_failure(self):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(fake_boozer_surface)

        fake_boozer_surface.queue_result(
            surface_x=[4.5, -2.5],
            iota=0.27,
            G=0.42,
            success=True,
        )
        fake_boozer_surface.surface.set_self_intersecting(True)
        fake_boozer_surface.need_to_run_code = True
        failure_state = self.module.evaluate_stage2_iota_state(runtime)
        self.assertTrue(failure_state.solve_failed)
        self.assertEqual(
            runtime.guarded_boozer_evaluator.last_failure_reason,
            "self_intersecting",
        )

        fake_boozer_surface.surface.set_self_intersecting(False)
        fake_boozer_surface.queue_result(
            surface_x=[1.1, -0.9],
            iota=0.23,
            G=0.36,
            success=True,
        )
        fake_boozer_surface.need_to_run_code = True
        success_state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertFalse(success_state.solve_failed)
        self.assertIsNone(runtime.guarded_boozer_evaluator.last_failure_reason)
        self.assertAlmostEqual(success_state.iota, 0.23)
        np.testing.assert_allclose(fake_boozer_surface.surface.x, [1.1, -0.9])
        np.testing.assert_allclose(
            runtime.guarded_boozer_evaluator.last_successful_state.surface_dofs,
            [1.1, -0.9],
        )

    def test_evaluate_banana_current_upper_bound_accepts_scaled_current_vjp(self):
        leaf_current = Current(17000.0)
        banana_current = ScaledCurrent(leaf_current, -1.0)

        (
            banana_current_abs_A,
            banana_current_violation,
            banana_current_signed_value,
            banana_current_grad,
        ) = self.module.evaluate_banana_current_upper_bound(
            banana_current=banana_current,
            banana_current_max_A=16000.0,
            base_objective_optimizable=banana_current,
        )

        self.assertAlmostEqual(banana_current_abs_A, 17000.0)
        self.assertAlmostEqual(banana_current_violation, 1000.0)
        self.assertAlmostEqual(banana_current_signed_value, 1000.0)
        np.testing.assert_allclose(banana_current_grad, [1.0])

    def test_build_stage2_alm_settings_converts_zero_trust_radius_to_none(self):
        settings = self.module.build_stage2_alm_settings(
            SimpleNamespace(
                alm_max_outer_iters=7,
                alm_max_subproblem_continuations=9,
                alm_penalty_init=2.0,
                alm_penalty_scale=3.0,
                alm_penalty_max=50.0,
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
            banana_init_current_A=1.0e4,
            banana_current_max_A=1.6e4,
            basin_hops=2,
            basin_stepsize=0.01,
            basin_temperature=2.5,
            basin_niter_success=6,
            alm_max_outer_iters=7,
            alm_max_subproblem_continuations=9,
            alm_penalty_init=2.0,
            alm_penalty_scale=3.0,
            alm_penalty_max=50.0,
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
            normalized_constraint_values=np.array([0.0, 0.01, 0.0]),
            raw_constraint_values=np.array([0.0, 1.0, 0.0]),
            solver_constraint_values=np.array([0.0, 0.2, 0.0]),
            normalized_solver_constraint_values=np.array([0.0, 0.2, 0.0]),
            raw_solver_constraint_values=np.array([0.0, 8.0, 0.0]),
            hard_signed_constraint_values=np.array([0.0, 0.02, 0.0]),
            raw_hard_signed_constraint_values=np.array([0.0, 2.0, 0.0]),
            hard_violation_values=np.array([0.0, 0.01, 0.0]),
            raw_hard_violation_values=np.array([0.0, 1.0, 0.0]),
            surrogate_signed_constraint_values=np.array([0.0, 0.2, 0.0]),
            raw_surrogate_signed_constraint_values=np.array([0.0, 8.0, 0.0]),
            constraint_scales=[1.0, 100.0, 1.0],
            constraint_blocks=["geometry", "current", "physics"],
            constraint_scale_sources=["one", "limit", "threshold"],
            raw_dual_estimates=[0.1, 0.002, 0.3],
            alm_schema_version="alm_normalized_constraints_v1",
            trust_radius=0.125,
            multiplier_cap_binding=True,
            multiplier_cap_binding_indices=[1],
            final_hard_max_violation=0.01,
            final_surrogate_max_value=0.2,
            hard_positive_shift_zero=True,
            signal_mismatch_active=False,
            final_penalty_gradient_norm=0.25,
            history=[{"outer_iteration": 1}],
        )
        hardware_status = {"success": False, "violations": ["too_curved"]}

        result = self.module.build_stage2_results(
            args=args,
            plasma_surf_filename="demo.nc",
            file_loc="/tmp/demo.nc",
            stage2_bs_path="/tmp/seed.json",
            tf_current_A=-8.0e4,
            tf_current_sum_abs_A=1.6e5,
            num_tf_coils=2,
            num_banana_coils=4,
            num_proxy_coils=0,
            num_vf_coils=0,
            initial_banana_current_A=1.2e4,
            banana_current_A=9.5e3,
            banana_to_tf_current_ratio=0.11875,
            finite_current_mode="boozer_surrogate",
            boozer_current_convention="mu0",
            proxy_plasma_current_A=0.0,
            vf_current_A=0.0,
            vf_template_path=None,
            total_coils=6,
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
            major_radius=0.976,
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
            basin_nonfinite_rejections=0,
            basin_normalized_step_rejections=1,
            basin_completed_hops=3,
            basin_initial_objective=0.51,
            basin_best_hop_objective=0.42,
            basin_best_hop_index=2,
            basin_best_result_source="hop",
            basin_objective_improvement=0.09,
            alm_result=alm_result,
            alm_taylor_result={"passed": True},
            final_volume=0.12,
            final_plasma_major_radius_m=0.92,
            final_plasma_minor_radius_m=0.15,
            field_error=0.03,
            intersecting=True,
            final_max_curvature=41.0,
            final_coil_length=1.8,
            final_curve_curve_min_dist=0.04,
            final_curve_surface_min_dist=0.017,
            plasma_vessel_min_dist=0.041,
            hardware_status=hardware_status,
        )

        self.assertFalse(result["HARDWARE_CONSTRAINTS_OK"])
        self.assertEqual(result["HARDWARE_CONSTRAINT_VIOLATIONS"], ["too_curved"])
        self.assertEqual(result["ALM_MAX_OUTER_ITERS"], 7)
        self.assertEqual(result["ALM_OUTER_ITERATIONS"], 4)
        self.assertEqual(result["ALM_FINAL_TRUST_RADIUS"], 0.125)
        self.assertEqual(result["ALM_SCHEMA_VERSION"], "alm_normalized_constraints_v1")
        self.assertTrue(result["ALM_MULTIPLIER_CAP_BINDING"])
        self.assertEqual(result["ALM_MULTIPLIER_CAP_BINDING_INDICES"], [1])
        np.testing.assert_allclose(result["ALM_FINAL_CONSTRAINT_VALUES"], [0.0, 1.0, 0.0])
        np.testing.assert_allclose(
            result["ALM_FINAL_NORMALIZED_CONSTRAINT_VALUES"],
            [0.0, 0.01, 0.0],
        )
        np.testing.assert_allclose(
            result["ALM_FINAL_SOLVER_CONSTRAINT_VALUES"],
            [0.0, 8.0, 0.0],
        )
        np.testing.assert_allclose(
            result["ALM_FINAL_NORMALIZED_SOLVER_CONSTRAINT_VALUES"],
            [0.0, 0.2, 0.0],
        )
        np.testing.assert_allclose(
            result["ALM_FINAL_HARD_SIGNED_CONSTRAINT_VALUES"],
            [0.0, 2.0, 0.0],
        )
        np.testing.assert_allclose(
            result["ALM_FINAL_NORMALIZED_HARD_SIGNED_CONSTRAINT_VALUES"],
            [0.0, 0.02, 0.0],
        )
        np.testing.assert_allclose(
            result["ALM_FINAL_HARD_VIOLATION_VALUES"],
            [0.0, 1.0, 0.0],
        )
        np.testing.assert_allclose(
            result["ALM_FINAL_NORMALIZED_HARD_VIOLATION_VALUES"],
            [0.0, 0.01, 0.0],
        )
        np.testing.assert_allclose(
            result["ALM_FINAL_SURROGATE_SIGNED_CONSTRAINT_VALUES"],
            [0.0, 8.0, 0.0],
        )
        np.testing.assert_allclose(
            result["ALM_FINAL_NORMALIZED_SURROGATE_SIGNED_CONSTRAINT_VALUES"],
            [0.0, 0.2, 0.0],
        )
        np.testing.assert_allclose(result["ALM_CONSTRAINT_SCALES"], [1.0, 100.0, 1.0])
        self.assertEqual(
            result["ALM_CONSTRAINT_BLOCKS"],
            ["geometry", "current", "physics"],
        )
        self.assertEqual(
            result["ALM_CONSTRAINT_SCALE_SOURCES"],
            ["one", "limit", "threshold"],
        )
        np.testing.assert_allclose(result["ALM_FINAL_RAW_DUAL_ESTIMATES"], [0.1, 0.002, 0.3])
        self.assertEqual(result["ALM_FINAL_HARD_MAX_VIOLATION"], 0.01)
        self.assertEqual(result["ALM_FINAL_SURROGATE_MAX_VALUE"], 0.2)
        self.assertTrue(result["ALM_FINAL_HARD_POSITIVE_SHIFT_ZERO"])
        self.assertFalse(result["ALM_FINAL_SIGNAL_MISMATCH_ACTIVE"])
        self.assertEqual(result["ALM_FINAL_PENALTY_GRADIENT_NORM"], 0.25)
        self.assertEqual(result["basin_seed"], 7)
        self.assertEqual(result["basin_temperature"], 2.5)
        self.assertEqual(result["basin_niter_success"], 6)
        self.assertEqual(result["basin_accepted_hops"], 2)
        self.assertEqual(result["basin_rejected_hops"], 1)
        self.assertEqual(result["basin_best_objective"], 0.42)
        self.assertEqual(result["basin_accept_test_rejections"], 1)
        self.assertTrue(result["basin_accept_test_triggered"])
        self.assertEqual(result["basin_nonfinite_rejections"], 0)
        self.assertEqual(result["basin_normalized_step_rejections"], 1)
        self.assertEqual(result["basin_completed_hops"], 3)
        self.assertEqual(result["basin_initial_objective"], 0.51)
        self.assertEqual(result["basin_best_hop_objective"], 0.42)
        self.assertEqual(result["basin_best_hop_index"], 2)
        self.assertEqual(result["basin_best_result_source"], "hop")
        self.assertEqual(result["basin_objective_improvement"], 0.09)
        self.assertEqual(result["BANANA_INIT_CURRENT_A"], 1.2e4)
        self.assertEqual(result["BANANA_CURRENT_MAX_A"], 1.6e4)
        self.assertEqual(result["TF_CURRENT_LIMIT_A"], 8.0e4)
        self.assertAlmostEqual(result["BANANA_TO_TF_CURRENT_RATIO"], 0.11875)
        self.assertEqual(result["COIL_LENGTH"], 1.8)
        self.assertEqual(result["MAX_CURVATURE"], 41.0)
        self.assertEqual(result["CURVE_CURVE_MIN_DIST"], 0.04)
        self.assertEqual(result["CURVE_SURFACE_MIN_DIST"], 0.017)
        self.assertEqual(result["SURFACE_VESSEL_MIN_DIST"], 0.041)

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

    def test_smooth_min_distance_signed_constraint_returns_zero_grad_without_pairs(
        self,
    ):
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

    def test_smooth_min_curve_surface_signed_constraint_includes_surface_vjp(self):
        curve = _FakeCurve(gamma_points=[[0.0, 0.0, 0.0]])
        surface = _FakeSurfaceWithGradient(gamma_points=[[[0.02, 0.0, 0.04]]])

        with (
            mock.patch.object(
                self.module,
                "_new_derivative",
                side_effect=lambda: _FakeDerivative({}),
            ),
            mock.patch.object(
                self.module,
                "surface_dgamma_by_dcoeff_derivative",
                side_effect=lambda _surface, point_gradient: _FakeDerivative(
                    np.array(
                        [
                            np.sum(point_gradient.reshape((-1, 3)), axis=0)[0],
                            np.sum(point_gradient.reshape((-1, 3)), axis=0)[2],
                        ],
                        dtype=float,
                    )
                ),
            ),
        ):
            signed_value, grad = self.module.smooth_min_curve_surface_signed_constraint(
                [curve],
                surface,
                minimum_distance=0.05,
                temperature=0.01,
                base_objective_optimizable=SimpleNamespace(),
            )

        self.assertAlmostEqual(signed_value, 0.05 - np.sqrt(0.02**2 + 0.04**2))
        np.testing.assert_allclose(grad, [0.0, -0.04 / np.sqrt(0.02**2 + 0.04**2)])


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
            JVolume=None,
            VOLUME_WEIGHT=0.0,
        )

        self.assertAlmostEqual(result["J_QS"], (0.5 * 2.0 + 6.0) / 1.5)
        self.assertAlmostEqual(result["J_Boozer"], (0.5 * 10.0 + 20.0) / 1.5)
        self.assertAlmostEqual(result["J_surf"], 1.5)
        self.assertAlmostEqual(result["J_volume"], 0.0)
        np.testing.assert_allclose(result["dJ_surf"], [0.1, 0.2])
        self.assertAlmostEqual(result["total"], 50.0)
        np.testing.assert_allclose(result["grad"], [8.8, 6.266666666666667])

    def test_evaluate_total_objective_fast_path_skips_component_breakdown(self):
        nonqs = [_FakeAlgebraicObjective(2.0, [2.0, 0.0])]
        brs = [_FakeAlgebraicObjective(3.0, [0.5, 0.5])]
        jiota = _FakeAlgebraicObjective(4.0, [0.2, 0.1])
        jlength = _FakeAlgebraicObjective(5.0, [1.0, 1.5])
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])

        result = self.module.evaluate_total_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=jiota,
            IOTAS_WEIGHT=3.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=1.0,
            JCurveCurve=zero,
            CC_WEIGHT=5.0,
            JCurveSurface=zero,
            CS_WEIGHT=6.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=7.0,
            include_diagnostics=False,
        )

        self.assertFalse(result["diagnostics_included"])
        self.assertEqual(
            set(result),
            {
                "total",
                "grad",
                "surface_weights",
                "diagnostics_included",
                "constraint_names",
                "dual_update_values",
                "feasibility_values",
                "search_hardware_constraint_payload_kind",
                "finite_eval_ok",
                "nonfinite_fields",
            },
        )
        self.assertEqual(
            result["search_hardware_constraint_payload_kind"],
            "penalty_objective",
        )
        self.assertEqual(
            result["constraint_names"],
            ["coil_coil_spacing", "coil_surface_spacing", "max_curvature"],
        )
        np.testing.assert_allclose(result["dual_update_values"], [0.0, 0.0, 0.0])
        np.testing.assert_allclose(result["feasibility_values"], [0.0, 0.0, 0.0])
        self.assertAlmostEqual(result["total"], 25.0)
        np.testing.assert_allclose(result["grad"], [4.6, 2.8])

    def test_evaluate_total_objective_fast_path_emits_penalty_constraint_payload(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])

        result = self.module.evaluate_total_objective(
            np.array([1.0]),
            [zero],
            [zero],
            RES_WEIGHT=0.0,
            Jiota=zero,
            IOTAS_WEIGHT=0.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=0.0,
            JCurveCurve=_FakeAlgebraicObjective(0.6, [0.1, 0.0]),
            CC_WEIGHT=1.0,
            JCurveSurface=_FakeAlgebraicObjective(0.7, [0.0, 0.1]),
            CS_WEIGHT=1.0,
            JCurvature=_FakeAlgebraicObjective(0.8, [0.2, 0.3]),
            CURVATURE_WEIGHT=1.0,
            JSurfSurf=_FakeAlgebraicObjective(0.9, [0.4, 0.5]),
            SURF_DIST_WEIGHT=1.0,
            include_diagnostics=False,
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
        np.testing.assert_allclose(
            result["dual_update_values"],
            [0.6, 0.7, 0.8, 0.9],
        )
        self.assertEqual(
            result["search_hardware_constraint_payload_kind"],
            "penalty_objective",
        )

    def test_evaluate_total_objective_supports_frontier_specific_objective_terms(self):
        nonqs = [_FakeAlgebraicObjective(2.0, [2.0, 0.0])]
        brs = [_FakeAlgebraicObjective(3.0, [0.5, 0.5])]
        jiota = _FakeAlgebraicObjective(-0.4, [-0.2, -0.1])
        jvolume = _FakeAlgebraicObjective(-0.3, [-0.3, -0.2])
        jlength = _FakeAlgebraicObjective(5.0, [1.0, 1.5])
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        normalized_nonqs = _FakeAlgebraicObjective(10.0, [10.0, 0.0])
        normalized_boozer = _FakeAlgebraicObjective(20.0, [1.0, 1.0])

        result = self.module.evaluate_total_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=0.5,
            Jiota=jiota,
            IOTAS_WEIGHT=2.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=1.0,
            JCurveCurve=zero,
            CC_WEIGHT=1.0,
            JCurveSurface=zero,
            CS_WEIGHT=1.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=1.0,
            JNonQSObjective=normalized_nonqs,
            JBoozerObjective=normalized_boozer,
            JVolume=jvolume,
            VOLUME_WEIGHT=4.0,
        )

        self.assertAlmostEqual(result["J_QS"], 2.0)
        self.assertAlmostEqual(result["J_QS_objective"], 10.0)
        self.assertAlmostEqual(result["J_Boozer"], 3.0)
        self.assertAlmostEqual(result["J_Boozer_objective"], 20.0)
        self.assertAlmostEqual(result["J_volume"], -0.3)
        self.assertAlmostEqual(result["total"], 23.0)
        np.testing.assert_allclose(result["grad"], [9.9, 1.0])

    def test_evaluate_alm_objective_builds_constraint_payload(self):
        nonqs = [_FakeAlgebraicObjective(2.0, [2.0, 0.0])]
        brs = [_FakeAlgebraicObjective(3.0, [0.5, 0.5])]
        jiota = _FakeAlgebraicObjective(4.0, [0.2, 0.1])
        jlength = _FakeAlgebraicObjective(5.0, [1.0, 1.5])
        jcc = _FakeAlgebraicObjective(0.6, [0.3, 0.4])
        jcs = _FakeAlgebraicObjective(0.7, [0.5, 0.6])
        jcurv = _FakeAlgebraicObjective(0.8, [0.7, 0.8])
        jsurf = _FakeAlgebraicObjective(0.9, [0.9, 1.0])

        def fake_augmented(
            base_value,
            base_grad,
            constraint_values,
            constraint_grads,
            multipliers,
            penalty,
        ):
            self.assertAlmostEqual(base_value, 25.0)
            np.testing.assert_allclose(base_grad, [4.6, 2.8])
            np.testing.assert_allclose(constraint_values, [-2.0, 10.0, 0.0075, -10.0])
            np.testing.assert_allclose(constraint_grads[0], [20.0, 0.0])
            np.testing.assert_allclose(constraint_grads[1], [0.0, 50.0])
            np.testing.assert_allclose(constraint_grads[2], [0.025, -0.025])
            np.testing.assert_allclose(constraint_grads[3], [12.5, 12.5])
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
            JVolume=None,
            VOLUME_WEIGHT=0.0,
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
            surface_surface_constraint_fn=lambda *_args: (
                -0.4,
                np.array([0.5, 0.5]),
                0.0,
            ),
            augmented_inequality_objective_fn=fake_augmented,
            activity_tolerances_fn=lambda ds, cs, include_surface_surface, include_surface_stack=False: np.array(
                [ds * 4.0, ds * 4.0, cs * 4.0, ds * 4.0]
                if include_surface_surface
                else [ds * 4.0, ds * 4.0, cs * 4.0],
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
        np.testing.assert_allclose(result["dual_update_values"], [-2.0, 10.0, 0.0075, -10.0])
        np.testing.assert_allclose(result["feasibility_values"], [0.0, 10.0, 0.0075, 0.0])
        np.testing.assert_allclose(
            result["constraint_activity_tolerances"], [0.8, 2.0, 0.005, 1.0]
        )
        np.testing.assert_allclose(
            result["constraint_scales"],
            [0.05, 0.02, 40.0, 0.04],
        )
        self.assertEqual(
            result["constraint_blocks"],
            ["geometry", "geometry", "geometry", "surface"],
        )
        self.assertEqual(
            result["objective_value_kinds"],
            ["surrogate", "surrogate", "surrogate", "surrogate"],
        )
        np.testing.assert_allclose(
            result["raw_dual_update_values"],
            [-0.1, 0.2, 0.3, -0.4],
        )
        np.testing.assert_allclose(
            result["raw_feasibility_values"],
            [0.0, 0.2, 0.3, 0.0],
        )
        self.assertAlmostEqual(result["base_total"], 25.0)
        self.assertAlmostEqual(result["max_feasibility_violation"], 10.0)
        self.assertAlmostEqual(result["J_cc"], 0.6)
        self.assertAlmostEqual(result["J_cs"], 0.7)
        self.assertAlmostEqual(result["J_surf"], 0.9)
        self.assertAlmostEqual(result["J_curvature"], 0.8)
        np.testing.assert_allclose(result["grad"], [8.0, -3.0])

    def test_evaluate_alm_objective_uses_hard_surface_stack_for_dual_signal(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        curves = (
            _FakeCurve(gamma_points=[[0.0, 0.0, 0.0]]),
            _FakeCurve(gamma_points=[[1.0, 0.0, 0.0]]),
        )
        outer_surface = _FakeSurfaceWithGradient(
            gamma_points=[[[0.5, 0.0, 0.0]]]
        )
        surface_a = _FakeSurfaceWithGradient(
            gamma_points=[[[0.0, 0.0, 0.0]]]
        )
        surface_b = _FakeSurfaceWithGradient(
            gamma_points=[[[0.04, 0.0, 0.0]]]
        )
        banana_curve = _FakeCurve(
            gamma_points=[[0.0, 0.0, 0.0]],
            kappa_values=[5.0],
        )

        def fake_augmented(
            base_value,
            base_grad,
            constraint_values,
            constraint_grads,
            multipliers,
            penalty,
        ):
            self.assertAlmostEqual(base_value, 0.0)
            np.testing.assert_allclose(base_grad, [0.0, 0.0])
            np.testing.assert_allclose(constraint_values, [0.4])
            np.testing.assert_allclose(constraint_grads[0], [8.0, 12.0])
            np.testing.assert_allclose(multipliers, [0.2])
            self.assertAlmostEqual(penalty, 3.0)
            return {
                "total": 1.0,
                "grad": np.array([1.0, 2.0]),
                "stationarity_norm": 0.1,
            }

        result = self.module.evaluate_alm_objective(
            np.array([1.0]),
            [zero],
            [zero],
            RES_WEIGHT=0.0,
            Jiota=zero,
            IOTAS_WEIGHT=0.0,
            JVolume=None,
            VOLUME_WEIGHT=0.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=0.0,
            JCurveCurve=zero,
            JCurveSurface=zero,
            JCurvature=zero,
            multipliers=np.array([0.2]),
            penalty=3.0,
            objective_optimizable=SimpleNamespace(),
            curves=curves,
            curve_curve_min_distance=0.05,
            outer_surface=outer_surface,
            curve_surface_min_distance=0.02,
            banana_curve=banana_curve,
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=("surface_surface_spacing",),
            curve_curve_constraint_fn=lambda *_args: (
                -0.1,
                np.array([0.0, 0.0]),
                0.0,
            ),
            curve_surface_constraint_fn=lambda *_args: (
                -0.2,
                np.array([0.0, 0.0]),
                0.0,
            ),
            curvature_constraint_fn=lambda *_args: (
                -0.3,
                np.array([0.0, 0.0]),
                0.0,
            ),
            surface_stack_surfaces=(surface_a, surface_b),
            surface_stack_min_distance=0.05,
            surface_stack_constraint_fn=lambda *_args: (
                0.02,
                np.array([0.4, 0.6]),
                0.02,
            ),
            hard_surrogate_diagnostics=True,
            augmented_inequality_objective_fn=fake_augmented,
        )

        self.assertEqual(result["constraint_blocks"], ["surface"])
        self.assertEqual(result["dual_update_value_kinds"], ["hard"])
        self.assertEqual(result["feasibility_value_kinds"], ["hard"])
        np.testing.assert_allclose(result["dual_update_values"], [0.2])
        np.testing.assert_allclose(result["feasibility_values"], [0.2])
        np.testing.assert_allclose(result["hard_signed_constraint_values"], [0.2])
        np.testing.assert_allclose(result["surrogate_signed_constraint_values"], [0.4])
        np.testing.assert_allclose(result["raw_dual_update_values"], [0.01])
        np.testing.assert_allclose(result["raw_hard_signed_constraint_values"], [0.01])
        np.testing.assert_allclose(
            result["raw_surrogate_signed_constraint_values"],
            [0.02],
        )

    def test_single_stage_normalized_alm_constraints_pass_directional_taylor_test(self):
        alm_utils = _load_module(ALM_UTILS_PATH, "banana_alm_utils")
        objective = SimpleNamespace(x=np.zeros(2, dtype=float))
        nonqs = [
            _XAwareQuadraticObjective(
                objective,
                constant=1.0,
                linear=[0.3, -0.1],
                quadratic=0.2,
            )
        ]
        brs = [
            _XAwareQuadraticObjective(
                objective,
                constant=0.5,
                linear=[-0.2, 0.4],
                quadratic=0.1,
            )
        ]
        jiota = _XAwareQuadraticObjective(objective, 0.2, [0.05, -0.03])
        jlength = _XAwareQuadraticObjective(objective, 0.4, [0.02, 0.01])
        jcc = _XAwareQuadraticObjective(objective, 0.0, [0.0, 0.0])
        jcs = _XAwareQuadraticObjective(objective, 0.0, [0.0, 0.0])
        jcurv = _XAwareQuadraticObjective(objective, 0.0, [0.0, 0.0])
        curve_curve_constraint = _affine_signed_constraint(
            objective,
            0.02,
            [0.003, -0.001],
            include_violation=True,
        )
        curve_surface_constraint = _affine_signed_constraint(
            objective,
            0.01,
            [-0.002, 0.004],
            include_violation=True,
        )
        curvature_constraint = _affine_signed_constraint(
            objective,
            0.8,
            [0.05, 0.03],
            include_violation=True,
        )

        def evaluate_problem(x, multipliers, penalty):
            objective.x = np.asarray(x, dtype=float)
            return self.module.evaluate_alm_objective(
                np.array([1.0]),
                nonqs,
                brs,
                RES_WEIGHT=1.5,
                Jiota=jiota,
                IOTAS_WEIGHT=0.7,
                JVolume=None,
                VOLUME_WEIGHT=0.0,
                JCurveLength=jlength,
                LENGTH_WEIGHT=0.9,
                JCurveCurve=jcc,
                JCurveSurface=jcs,
                JCurvature=jcurv,
                multipliers=multipliers,
                penalty=penalty,
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
                ),
                curve_curve_constraint_fn=curve_curve_constraint,
                curve_surface_constraint_fn=curve_surface_constraint,
                curvature_constraint_fn=curvature_constraint,
            )

        result = alm_utils.run_directional_taylor_test(
            evaluate_problem,
            np.array([0.15, -0.25]),
            np.array([0.03, 0.04, 0.05]),
            6.0,
            direction=np.array([-0.5, 0.7]),
            epsilons=TAYLOR_TEST_EPSILONS,
        )

        self.assertTrue(result["passed"], result)

    def test_evaluate_alm_objective_supports_independent_banana_current_constraints(self):
        nonqs = [_FakeAlgebraicObjective(2.0, [0.2, 0.0])]
        brs = [_FakeAlgebraicObjective(3.0, [0.0, 0.3])]
        jiota = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        jlength = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        current_a = _FakeCurrentObjective(17000.0, [2.0, -1.0])
        current_b = _FakeCurrentObjective(-19000.0, [0.5, 1.5])

        def fake_augmented(
            base_value,
            base_grad,
            constraint_values,
            constraint_grads,
            multipliers,
            penalty,
        ):
            self.assertAlmostEqual(base_value, 5.0)
            np.testing.assert_allclose(base_grad, [0.2, 0.3])
            np.testing.assert_allclose(constraint_values, [0.0625, 0.1875])
            np.testing.assert_allclose(
                constraint_grads,
                [
                    [0.000125, -0.0000625],
                    [-0.00003125, -0.00009375],
                ],
            )
            np.testing.assert_allclose(multipliers, [0.1, 0.2])
            self.assertAlmostEqual(penalty, 8.0)
            return {
                "total": 5.5,
                "grad": np.array([0.4, 0.6]),
                "stationarity_norm": 0.25,
            }

        result = self.module.evaluate_alm_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=1.0,
            Jiota=jiota,
            IOTAS_WEIGHT=0.0,
            JVolume=None,
            VOLUME_WEIGHT=0.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=0.0,
            JCurveCurve=zero,
            JCurveSurface=zero,
            JCurvature=zero,
            multipliers=np.array([0.1, 0.2]),
            penalty=8.0,
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
                self.module.independent_banana_current_alm_constraint_name(0),
                self.module.independent_banana_current_alm_constraint_name(1),
            ),
            curve_curve_constraint_fn=lambda *_args: (-0.1, np.array([0.0, 0.0]), 0.0),
            curve_surface_constraint_fn=lambda *_args: (-0.2, np.array([0.0, 0.0]), 0.0),
            curvature_constraint_fn=lambda *_args: (-0.3, np.array([0.0, 0.0]), 0.0),
            banana_currents=(current_a, current_b),
            banana_current_threshold=16000.0,
            augmented_inequality_objective_fn=fake_augmented,
            include_diagnostics=False,
        )

        self.assertEqual(
            result["constraint_names"],
            ["banana_current_0_upper_bound", "banana_current_1_upper_bound"],
        )
        self.assertEqual(result["constraint_blocks"], ["current", "current"])
        np.testing.assert_allclose(result["constraint_scales"], [16000.0, 16000.0])
        np.testing.assert_allclose(result["raw_dual_update_values"], [1000.0, 3000.0])
        np.testing.assert_allclose(result["raw_feasibility_values"], [1000.0, 3000.0])
        np.testing.assert_allclose(result["dual_update_values"], [0.0625, 0.1875])
        np.testing.assert_allclose(
            result["constraint_activity_tolerances"],
            [6.25e-8, 6.25e-8],
        )
        self.assertEqual(result["objective_value_kinds"], ["hard", "hard"])
        self.assertEqual(result["dual_update_value_kinds"], ["hard", "hard"])

    def test_evaluate_alm_objective_reports_active_banana_current_threshold(self):
        nonqs = [_FakeAlgebraicObjective(2.0, [0.2, 0.0])]
        brs = [_FakeAlgebraicObjective(3.0, [0.0, 0.3])]
        jiota = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        jlength = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        banana_current = _FakeCurrentObjective(17000.0, [2.0, -1.0])

        result = self.module.evaluate_alm_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=1.0,
            Jiota=jiota,
            IOTAS_WEIGHT=0.0,
            JVolume=None,
            VOLUME_WEIGHT=0.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=0.0,
            JCurveCurve=zero,
            JCurveSurface=zero,
            JCurvature=zero,
            multipliers=np.array([0.1]),
            penalty=8.0,
            objective_optimizable=SimpleNamespace(),
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface="outer",
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=("banana_current_upper_bound",),
            curve_curve_constraint_fn=lambda *_args: (-0.1, np.array([0.0, 0.0]), 0.0),
            curve_surface_constraint_fn=lambda *_args: (-0.2, np.array([0.0, 0.0]), 0.0),
            curvature_constraint_fn=lambda *_args: (-0.3, np.array([0.0, 0.0]), 0.0),
            banana_current=banana_current,
            banana_current_threshold=20000.0,
            include_diagnostics=True,
        )

        self.assertEqual(result["banana_current_upper_bound_threshold"], 20000.0)
        np.testing.assert_allclose(result["constraint_scales"], [20000.0])

    def test_evaluate_alm_objective_fast_path_keeps_constraint_payload_only(self):
        nonqs = [_FakeAlgebraicObjective(2.0, [2.0, 0.0])]
        brs = [_FakeAlgebraicObjective(3.0, [0.5, 0.5])]
        jiota = _FakeAlgebraicObjective(4.0, [0.2, 0.1])
        jlength = _FakeAlgebraicObjective(5.0, [1.0, 1.5])
        jcc = _FakeAlgebraicObjective(0.6, [0.3, 0.4])
        jcs = _FakeAlgebraicObjective(0.7, [0.5, 0.6])
        jcurv = _FakeAlgebraicObjective(0.8, [0.7, 0.8])

        result = self.module.evaluate_alm_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=jiota,
            IOTAS_WEIGHT=3.0,
            JVolume=None,
            VOLUME_WEIGHT=0.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=1.0,
            JCurveCurve=jcc,
            JCurveSurface=jcs,
            JCurvature=jcurv,
            multipliers=np.array([0.1, 0.2, 0.3]),
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
            ),
            curve_curve_constraint_fn=lambda *_args: (
                -0.1,
                np.array([1.0, 0.0]),
                0.0,
            ),
            curve_surface_constraint_fn=lambda *_args: (
                0.2,
                np.array([0.0, 1.0]),
                0.2,
            ),
            curvature_constraint_fn=lambda *_args: (
                0.3,
                np.array([1.0, -1.0]),
                0.3,
            ),
            include_diagnostics=False,
        )

        self.assertFalse(result["diagnostics_included"])
        for diagnostic_key in (
            "J_QS",
            "dJ_QS",
            "J_Boozer",
            "dJ_Boozer",
            "J_cc",
            "dJ_cc",
            "J_cs",
            "dJ_cs",
            "J_curvature",
            "dJ_curvature",
        ):
            self.assertNotIn(diagnostic_key, result)
        self.assertEqual(
            result["constraint_names"],
            ["coil_coil_spacing", "coil_surface_spacing", "max_curvature"],
        )
        np.testing.assert_allclose(result["dual_update_values"], [-2.0, 10.0, 0.0075])
        np.testing.assert_allclose(result["raw_dual_update_values"], [-0.1, 0.2, 0.3])
        self.assertAlmostEqual(result["max_feasibility_violation"], 10.0)

    def test_evaluate_base_objective_projects_total_gradient_when_requested(self):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()

        result = self.module.evaluate_base_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=jiota,
            IOTAS_WEIGHT=3.0,
            JVolume=None,
            VOLUME_WEIGHT=0.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=1.0,
            objective_optimizable=objective,
        )

        np.testing.assert_allclose(result["grad"], [4.6, 2.8, 0.0, 0.0])

    def test_evaluate_base_objective_thresholded_physics_formulation_zeros_base_value_and_grad(
        self,
    ):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()

        result = self.module.evaluate_base_objective(
            np.array([1.0]),
            nonqs,
            brs,
            RES_WEIGHT=2.0,
            Jiota=jiota,
            IOTAS_WEIGHT=3.0,
            JVolume=None,
            VOLUME_WEIGHT=0.0,
            JCurveLength=jlength,
            LENGTH_WEIGHT=1.0,
            objective_optimizable=objective,
            alm_formulation="thresholded_physics",
        )

        self.assertAlmostEqual(result["total"], 0.0)
        self.assertAlmostEqual(result["physics_total"], 25.0)
        np.testing.assert_allclose(result["grad"], [0.0, 0.0, 0.0, 0.0])

    def test_evaluate_base_objective_rejects_unknown_alm_formulation(self):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()

        with self.assertRaisesRegex(ValueError, "Unsupported ALM formulation"):
            self.module.evaluate_base_objective(
                np.array([1.0]),
                nonqs,
                brs,
                RES_WEIGHT=2.0,
                Jiota=jiota,
                IOTAS_WEIGHT=3.0,
                JVolume=None,
                VOLUME_WEIGHT=0.0,
                JCurveLength=jlength,
                LENGTH_WEIGHT=1.0,
                objective_optimizable=objective,
                alm_formulation="typo",
            )

    def test_evaluate_alm_objective_projects_base_gradient_into_constraint_space(self):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()

        def fake_augmented(
            base_value,
            base_grad,
            constraint_values,
            constraint_grads,
            multipliers,
            penalty,
        ):
            self.assertAlmostEqual(base_value, 25.0)
            np.testing.assert_allclose(base_grad, [4.6, 2.8, 0.0, 0.0])
            np.testing.assert_allclose(constraint_values, [-2.0, 10.0, 0.0075])
            np.testing.assert_allclose(constraint_grads[0], [20.0, 0.0, 0.0, 0.0])
            np.testing.assert_allclose(constraint_grads[1], [0.0, 50.0, 0.0, 0.0])
            np.testing.assert_allclose(
                constraint_grads[2],
                [0.025, -0.025, 0.0, 0.0],
            )
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
            JVolume=None,
            VOLUME_WEIGHT=0.0,
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
            ),
            curve_curve_constraint_fn=lambda *_args: (
                -0.1,
                np.array([1.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            curve_surface_constraint_fn=lambda *_args: (
                0.2,
                np.array([0.0, 1.0, 0.0, 0.0]),
                0.2,
            ),
            curvature_constraint_fn=lambda *_args: (
                0.3,
                np.array([1.0, -1.0, 0.0, 0.0]),
                0.3,
            ),
            augmented_inequality_objective_fn=fake_augmented,
            activity_tolerances_fn=lambda ds, cs, include_surface_surface, include_surface_stack=False: np.array(
                [ds * 4.0, ds * 4.0, cs * 4.0],
                dtype=float,
            ),
        )

        np.testing.assert_allclose(result["grad"], [9.0, -2.0, 0.0, 0.0])

    def test_evaluate_alm_objective_thresholded_physics_formulation_promotes_physics_terms_to_constraints(
        self,
    ):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()

        def fake_augmented(
            base_value,
            base_grad,
            constraint_values,
            constraint_grads,
            multipliers,
            penalty,
        ):
            self.assertAlmostEqual(base_value, 0.0)
            np.testing.assert_allclose(base_grad, [0.0, 0.0, 0.0, 0.0])
            np.testing.assert_allclose(
                constraint_values,
                [-2.0, 10.0, 0.0075, 1.0, 2.0, 7.0, 5.0e12],
            )
            np.testing.assert_allclose(constraint_grads[3], [2.0, 0.0, 0.0, 0.0])
            np.testing.assert_allclose(constraint_grads[4], [0.5, 0.5, 0.0, 0.0])
            np.testing.assert_allclose(constraint_grads[5], [0.4, 0.2, 0.0, 0.0])
            np.testing.assert_allclose(
                constraint_grads[6],
                [1.0e12, 1.5e12, 0.0, 0.0],
            )
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
            JVolume=None,
            VOLUME_WEIGHT=0.0,
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
            curve_curve_constraint_fn=lambda *_args: (
                -0.1,
                np.array([1.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            curve_surface_constraint_fn=lambda *_args: (
                0.2,
                np.array([0.0, 1.0, 0.0, 0.0]),
                0.2,
            ),
            curvature_constraint_fn=lambda *_args: (
                0.3,
                np.array([1.0, -1.0, 0.0, 0.0]),
                0.3,
            ),
            augmented_inequality_objective_fn=fake_augmented,
            activity_tolerances_fn=lambda ds, cs, include_surface_surface, include_surface_stack=False: np.array(
                [ds * 4.0, ds * 4.0, cs * 4.0],
                dtype=float,
            ),
            alm_formulation="thresholded_physics",
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
        np.testing.assert_allclose(
            result["constraint_activity_tolerances"],
            [0.8, 2.0, 0.005, 0.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            result["raw_constraint_activity_tolerances"],
            [0.04, 0.04, 0.2, 0.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result["grad"], [7.0, -4.0, 0.0, 0.0])

    def test_evaluate_alm_objective_thresholded_physics_formulation_requires_explicit_thresholds(
        self,
    ):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()

        with self.assertRaisesRegex(
            ValueError,
            "thresholded_physics ALM formulation requires explicit objective thresholds",
        ):
            self.module.evaluate_alm_objective(
                np.array([1.0]),
                nonqs,
                brs,
                RES_WEIGHT=2.0,
                Jiota=jiota,
                IOTAS_WEIGHT=3.0,
                JVolume=None,
                VOLUME_WEIGHT=0.0,
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
                curve_curve_constraint_fn=lambda *_args: (
                    -0.1,
                    np.array([1.0, 0.0, 0.0, 0.0]),
                    0.0,
                ),
                curve_surface_constraint_fn=lambda *_args: (
                    0.2,
                    np.array([0.0, 1.0, 0.0, 0.0]),
                    0.2,
                ),
                curvature_constraint_fn=lambda *_args: (
                    0.3,
                    np.array([1.0, -1.0, 0.0, 0.0]),
                    0.3,
                ),
                alm_formulation="thresholded_physics",
                qs_threshold=1.0,
                boozer_threshold=1.0,
                iota_penalty_threshold=None,
                length_penalty_threshold=0.0,
            )


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

        np.testing.assert_allclose(
            surface_data[0]["boozer_surface"].surface.x, [1.0, 2.0]
        )
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
        np.testing.assert_allclose(
            surface_data[0]["boozer_surface"].surface.x, [1.0, 2.0]
        )
        np.testing.assert_allclose(
            surface_data[1]["boozer_surface"].surface.x, [3.0, 4.0]
        )
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
        self.assertIn("search_hardware_status", result)
        self.assertIn("artifact_hardware_status", result)
        self.assertNotIn("status", result)
        self.assertNotIn("artifact_status", result)
        self.assertFalse(result["search_hardware_status"]["success"])
        self.assertEqual(len(result["search_hardware_status"]["violations"]), 3)

    def test_evaluate_single_stage_hardware_snapshot_keeps_top_level_constraints_in_search_role(
        self,
    ):
        result = self.module.evaluate_single_stage_hardware_snapshot(
            curve_curve_distance_obj=SimpleNamespace(shortest_distance=lambda: 0.05),
            cc_dist=0.05,
            curve_surface_distance_obj=SimpleNamespace(shortest_distance=lambda: 0.02),
            cs_dist=0.02,
            surface_vessel_distance_obj=SimpleNamespace(shortest_distance=lambda: 0.04),
            surface_status={"outer_vessel_gap": 0.5},
            ss_dist=0.04,
            banana_curve=SimpleNamespace(kappa=lambda: np.array([40.0, 40.0])),
            curvature_threshold=40.0,
            banana_current_A=1.7e4,
            banana_current_max_A=1.6e4,
        )

        self.assertIn("banana_current", result["constraints"])
        self.assertEqual(
            result["constraints"]["banana_current"]["threshold"],
            1.6e4,
        )
        self.assertEqual(
            result["constraints"],
            result["search_hardware_status"]["constraints"],
        )

    def test_evaluate_single_stage_search_hardware_snapshot_uses_surrogate_constraints(
        self,
    ):
        result = self.module.evaluate_single_stage_search_hardware_snapshot(
            {
                "constraint_names": [
                    "coil_coil_spacing",
                    "coil_surface_spacing",
                    "surface_vessel_spacing",
                    "max_curvature",
                    "banana_current_upper_bound",
                ],
                "dual_update_values": np.array([0.01, -0.003, 0.004, 0.5, 1000.0]),
                "search_hardware_constraint_payload_kind": "signed_residual",
            },
            cc_dist=0.05,
            cs_dist=0.02,
            ss_dist=0.04,
            curvature_threshold=40.0,
            banana_current_max_A=1.6e4,
        )

        self.assertAlmostEqual(result["curve_curve_min_dist"], 0.04)
        self.assertAlmostEqual(result["curve_surface_min_dist"], 0.023)
        self.assertAlmostEqual(result["surface_vessel_min_dist"], 0.036)
        self.assertAlmostEqual(result["max_curvature"], 40.5)
        self.assertAlmostEqual(result["banana_current_A"], 1.7e4)
        self.assertFalse(result["search_hardware_status"]["success"])
        self.assertIn(
            "coil_coil_spacing",
            result["search_hardware_status"]["constraints"],
        )
        self.assertIn(
            "max_curvature",
            result["search_hardware_status"]["constraints"],
        )
        self.assertIn(
            "banana_current",
            result["search_hardware_status"]["constraints"],
        )

    def test_evaluate_single_stage_search_hardware_snapshot_uses_penalty_objective_payload(
        self,
    ):
        result = self.module.evaluate_single_stage_search_hardware_snapshot(
            {
                "constraint_names": [
                    "coil_coil_spacing",
                    "coil_surface_spacing",
                    "max_curvature",
                ],
                "dual_update_values": np.array([0.6, 0.0, 0.8]),
                "search_hardware_constraint_payload_kind": "penalty_objective",
            },
            cc_dist=0.05,
            cs_dist=0.02,
            ss_dist=0.04,
            curvature_threshold=40.0,
            banana_current_A=1.7e4,
            banana_current_max_A=1.6e4,
        )

        self.assertIsNone(result["curve_curve_min_dist"])
        self.assertIsNone(result["max_curvature"])
        self.assertFalse(result["search_hardware_status"]["success"])
        self.assertAlmostEqual(
            result["search_hardware_status"]["banana_current_A"],
            1.7e4,
        )
        self.assertEqual(
            result["search_hardware_status"]["violation_ratios"],
            {
                "coil_coil_spacing_penalty": 0.6,
                "coil_surface_spacing_penalty": 0.0,
                "max_curvature_penalty": 0.8,
            },
        )

    def test_evaluate_single_stage_search_hardware_snapshot_requires_payload_kind(self):
        with self.assertRaises(KeyError):
            self.module.evaluate_single_stage_search_hardware_snapshot(
                {
                    "constraint_names": [
                        "coil_coil_spacing",
                        "coil_surface_spacing",
                        "max_curvature",
                    ],
                    "dual_update_values": np.array([0.0, 0.0, 0.0]),
                },
                cc_dist=0.05,
                cs_dist=0.02,
                ss_dist=0.04,
                curvature_threshold=40.0,
            )


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


class SmoothDistanceSelectionModuleTests(_ModuleTestCase):
    MODULE_PATH = SMOOTH_DISTANCE_SELECTION_PATH
    MODULE_PREFIX = "banana_smooth_distance_selection"

    def test_kdtree_pairwise_selection_matches_bruteforce_threshold(self):
        left = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
            ],
            dtype=float,
        )
        right = np.array(
            [
                [0.1, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [3.4, 0.0, 0.0],
            ],
            dtype=float,
        )

        self.assertAlmostEqual(self.module.pairwise_block_min(left, right), 0.1)
        rows, cols, diffs, distances = self.module.select_pairwise_near_min(
            left,
            right,
            threshold=0.45,
        )

        self.assertEqual(set(zip(rows.tolist(), cols.tolist())), {(0, 0), (2, 2)})
        np.testing.assert_allclose(
            np.linalg.norm(diffs, axis=1),
            distances,
        )


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

        signed_value, grad, violation = (
            self.module.smooth_max_curvature_signed_constraint(
                curve,
                threshold=4.0,
                temperature=0.2,
                objective_optimizable=SimpleNamespace(),
            )
        )

        self.assertGreater(signed_value, 1.0)
        self.assertEqual(violation, signed_value)
        np.testing.assert_allclose(grad, [1.0, -1.0])

    def test_smooth_min_curve_curve_signed_constraint_returns_zero_grad_without_pairs(
        self,
    ):
        objective = SimpleNamespace(x=np.array([2.0, -3.0]))
        curve = _FakeCurve(gamma_points=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

        signed_value, grad, violation = (
            self.module.smooth_min_curve_curve_signed_constraint(
                [curve],
                minimum_distance=0.05,
                temperature=0.01,
                objective_optimizable=objective,
            )
        )

        self.assertAlmostEqual(signed_value, 0.05)
        self.assertEqual(violation, 0.0)
        np.testing.assert_allclose(grad, [0.0, 0.0])

    def test_smooth_min_surface_surface_signed_constraint_reports_positive_violation(
        self,
    ):
        surface_1 = _FakeSurfaceWithArrayGradient(
            gamma_points=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        )
        surface_2 = _FakeSurfaceWithArrayGradient(
            gamma_points=[[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]]
        )

        with (
            mock.patch.object(
                self.module,
                "_new_derivative",
                side_effect=lambda: _FakeDerivative({}),
            ),
            mock.patch.object(
                self.module,
                "surface_dgamma_by_dcoeff_derivative",
                side_effect=lambda _surface, point_gradient: _FakeDerivative(
                    np.array(
                        [
                            np.sum(point_gradient.reshape((-1, 3)), axis=0)[0],
                            np.sum(point_gradient.reshape((-1, 3)), axis=0)[2],
                        ],
                        dtype=float,
                    )
                ),
            ),
        ):
            signed_value, grad, violation = (
                self.module.smooth_min_surface_surface_signed_constraint(
                    surface_1,
                    surface_2,
                    minimum_distance=0.5,
                    temperature=0.01,
                    objective_optimizable=SimpleNamespace(),
                )
            )

        self.assertGreater(violation, 0.0)
        self.assertAlmostEqual(violation, signed_value)
        self.assertEqual(grad.shape, (2,))

    def test_smooth_min_surface_stack_signed_constraint_uses_adjacent_pairs(self):
        surfaces = (
            _FakeSurfaceWithArrayGradient(
                gamma_points=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
            ),
            _FakeSurfaceWithArrayGradient(
                gamma_points=[[[0.1, 0.0, 0.0], [1.1, 0.0, 0.0]]]
            ),
            _FakeSurfaceWithArrayGradient(
                gamma_points=[[[0.7, 0.0, 0.0], [1.7, 0.0, 0.0]]]
            ),
        )

        with (
            mock.patch.object(
                self.module,
                "_new_derivative",
                side_effect=lambda: _FakeDerivative({}),
            ),
            mock.patch.object(
                self.module,
                "surface_dgamma_by_dcoeff_derivative",
                side_effect=lambda _surface, point_gradient: _FakeDerivative(
                    np.array(
                        [
                            np.sum(point_gradient.reshape((-1, 3)), axis=0)[0],
                            np.sum(point_gradient.reshape((-1, 3)), axis=0)[2],
                        ],
                        dtype=float,
                    )
                ),
            ),
        ):
            signed_value, grad, violation = (
                self.module.smooth_min_surface_stack_signed_constraint(
                    surfaces,
                    minimum_distance=0.5,
                    temperature=0.01,
                    objective_optimizable=SimpleNamespace(),
                )
            )

        self.assertGreater(violation, 0.0)
        self.assertAlmostEqual(violation, signed_value)
        self.assertEqual(grad.shape, (2,))

    def test_surface_vjp_helper_wraps_raw_surface_array_output_as_derivative(self):
        derivative = self.module._new_derivative()
        surface = _FakeSurfaceWithArrayGradient(
            gamma_points=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]]
        )

        derivative += self.module.surface_dgamma_by_dcoeff_derivative(
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

    def test_curvature_traversal_allows_inside_threshold(self):
        decision = self.module.decide_curvature_traversal(
            max_curvature=99.0,
            curvature_threshold=100.0,
            policy=self.module.CurvatureTraversalPolicy(0.05, 0),
            used_budget=0,
        )

        self.assertTrue(decision.allow_boozer_eval)
        self.assertFalse(decision.over_threshold)
        self.assertEqual(decision.reason, "within_threshold")
        self.assertEqual(decision.far_invalid_limit, 105.0)

    def test_curvature_traversal_allows_overcap_inside_budgeted_band(self):
        decision = self.module.decide_curvature_traversal(
            max_curvature=104.0,
            curvature_threshold=100.0,
            policy=self.module.CurvatureTraversalPolicy(0.05, 2),
            used_budget=1,
        )

        self.assertTrue(decision.allow_boozer_eval)
        self.assertTrue(decision.over_threshold)
        self.assertEqual(decision.reason, "within_traversal_band")

    def test_curvature_traversal_rejects_far_invalid_curvature(self):
        decision = self.module.decide_curvature_traversal(
            max_curvature=106.0,
            curvature_threshold=100.0,
            policy=self.module.CurvatureTraversalPolicy(0.05, 2),
            used_budget=0,
        )

        self.assertFalse(decision.allow_boozer_eval)
        self.assertTrue(decision.over_threshold)
        self.assertEqual(decision.reason, "far_invalid_curvature")

    def test_curvature_traversal_rejects_after_budget_exhausts(self):
        decision = self.module.decide_curvature_traversal(
            max_curvature=104.0,
            curvature_threshold=100.0,
            policy=self.module.CurvatureTraversalPolicy(0.05, 2),
            used_budget=2,
        )

        self.assertFalse(decision.allow_boozer_eval)
        self.assertTrue(decision.over_threshold)
        self.assertEqual(decision.reason, "curvature_traversal_budget_exhausted")
