import importlib.util
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from scipy.io import netcdf_file
from simsopt._core import Optimizable
from simsopt._core.derivative import Derivative, derivative_dec
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
HARDWARE_CONSTRAINT_SCHEMA_PATH = (
    EXAMPLES_ROOT / "banana_opt" / "hardware_constraint_schema.py"
)
HARDWARE_CONTRACTS_PATH = EXAMPLES_ROOT / "banana_opt" / "hardware_contracts.py"
WOUT_CONVENTION_PATH = EXAMPLES_ROOT / "banana_opt" / "wout_convention.py"
SINGLE_STAGE_SEARCH_POLICY_PATH = (
    EXAMPLES_ROOT / "banana_opt" / "single_stage_search_policy.py"
)
SINGLE_STAGE_INCUMBENTS_PATH = EXAMPLES_ROOT / "banana_opt" / "incumbents.py"
POLOIDAL_EXTENT_PATH = EXAMPLES_ROOT / "banana_opt" / "poloidal_extent.py"
ELLIPSE_WIDTH_PATH = EXAMPLES_ROOT / "banana_opt" / "ellipse_width.py"
SELF_INTERSECT_PATH = EXAMPLES_ROOT / "banana_opt" / "self_intersect.py"
TAYLOR_TEST_EPSILONS = (1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4)
MANUFACTURABILITY_ALM_CONSTRAINT_NAMES = (
    "width_min",
    "width_max",
    "self_intersect",
)

sys.path.insert(0, str(EXAMPLES_ROOT))
from alm_utils import ALM_SCHEMA_VERSION  # noqa: E402

del sys.path[0]


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


def _in_bounds_lcfs_major_radius_m():
    hardware_contracts = _load_module(
        HARDWARE_CONTRACTS_PATH,
        "banana_hw_contracts",
    )
    return hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M - 0.01


def _in_bounds_lcfs_minor_radius_m():
    hardware_contracts = _load_module(
        HARDWARE_CONTRACTS_PATH,
        "banana_hw_contracts",
    )
    return hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M - 0.01


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


class _FakeWidthObjective:
    def __init__(self, value, grad):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)

    def J(self):
        return self._value

    def dJ(self, partials=False):
        if not partials:
            raise AssertionError("Expected partial derivative request")
        return lambda _objective: self._grad.copy()


class _FakeSelfIntersectObjective:
    def __init__(self, value, grad, shortest_self_distance=0.5):
        self._value = float(value)
        self._grad = np.asarray(grad, dtype=float)
        self._shortest_self_distance = float(shortest_self_distance)

    def J(self):
        return self._value

    def dJ(self, partials=False):
        if not partials:
            raise AssertionError("Expected partial derivative request")
        return lambda _objective: self._grad.copy()

    def shortest_self_distance(self):
        return self._shortest_self_distance


def _default_geometric_parity_kwargs():
    return dict(
        Jw=_FakeWidthObjective(0.10, [0.0, 0.0]),
        width_min_threshold=0.05,
        width_max_threshold=0.17,
        Jself=_FakeSelfIntersectObjective(0.0, [0.0, 0.0], shortest_self_distance=0.5),
        self_intersect_threshold=0.0,
        length_min_target=0.95,
    )


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
            self.constant + np.dot(self.linear, x) + 0.5 * self.quadratic * np.dot(x, x)
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


class _FakeResidueObjective(_FakeAlgebraicObjective):
    def to_json_dict(self):
        return {
            "schema_version": "test_residue_objective_v1",
            "enabled": True,
            "target_manifest_id": "test-targets",
            "validation_id": "test-validation",
            "objective_weight": 1.0,
            "residue_scale": 1.0,
            "value": self.J(),
            "branches": [],
        }


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
        self.x = self._gamma.reshape(-1).copy()

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


class _FakeRadiusSurface(Optimizable):
    def __init__(self, major_radius, minor_radius, major_grad, minor_grad):
        self._major_radius = float(major_radius)
        self._minor_radius = float(minor_radius)
        self._major_grad = np.asarray(major_grad, dtype=float)
        self._minor_grad = np.asarray(minor_grad, dtype=float)
        super().__init__(x0=np.zeros(self._major_grad.size))

    def major_radius(self):
        return self._major_radius

    def minor_radius(self):
        return self._minor_radius

    def dmajor_radius_by_dcoeff(self):
        return self._major_grad.copy()

    def dminor_radius_by_dcoeff(self):
        return self._minor_grad.copy()


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


class _FakeOptimizableCurve(Optimizable):
    def __init__(self, gamma_points, gammadash_points, *, order=3):
        self._gamma = np.asarray(gamma_points, dtype=float)
        self._gammadash = np.asarray(gammadash_points, dtype=float)
        self.quadpoints = np.linspace(0.0, 1.0, len(self._gamma), endpoint=False)
        self.order = int(order)
        super().__init__()

    def gamma(self):
        return self._gamma.copy()

    def gammadash(self):
        return self._gammadash.copy()

    def dgamma_by_dcoeff_vjp(self, point_gradient):
        return _FakeDerivative(
            [
                float(np.sum(point_gradient[:, 0])),
                float(np.sum(point_gradient[:, 2])),
            ]
        )

    def dgammadash_by_dcoeff_vjp(self, tangent_gradient):
        return _FakeDerivative(
            [
                float(np.sum(tangent_gradient[:, 0])),
                float(np.sum(tangent_gradient[:, 2])),
            ]
        )


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


def _constant_constraint_result(signed_value, grad, violation):
    grad = np.asarray(grad, dtype=float)

    def constraint(*_args, **_kwargs):
        return signed_value, grad.copy(), violation

    return constraint


def _constant_constraint_result_with_hard_signal(
    signed_value,
    grad,
    violation,
    hard_signed_value,
    hard_violation,
):
    grad = np.asarray(grad, dtype=float)

    def constraint(*_args, **_kwargs):
        return (
            signed_value,
            grad.copy(),
            violation,
            hard_signed_value,
            hard_violation,
        )

    return constraint


def _zero_constraint_result_2d(*_args, **_kwargs):
    return 0.0, np.zeros(2), 0.0


class _FakeBiotSavart:
    def __init__(self, field_shape):
        self._field = np.zeros(field_shape, dtype=float)
        self.points = None

    def B(self):
        return self._field.copy()

    def set_points(self, points):
        self.points = np.asarray(points, dtype=float).copy()

    def clear_cached_properties(self):
        pass


class _FakeSurfaceNormals:
    def __init__(self, shape):
        self._unitnormal = np.zeros(shape, dtype=float)

    def unitnormal(self):
        return self._unitnormal.copy()

    def gamma(self):
        return np.zeros(self._unitnormal.shape, dtype=float)


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

    @staticmethod
    def _inboard_poloidal_points(theta, count=1, radius=0.2, R_winding=0.976):
        R = R_winding - radius * np.cos(theta)
        Z = radius * np.sin(theta)
        return np.tile(np.array([[R, 0.0, Z]], dtype=float), (count, 1))

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

    def test_hard_constraint_reports_true_poloidal_violation(self):
        theta = 0.60
        threshold = np.pi / 4.0
        curve = _FakeCurve(self._inboard_poloidal_points(theta, count=256))

        signed_value, violation = self.module.poloidal_extent_signed_constraint(
            curve,
            R_winding=0.976,
            theta_threshold=threshold,
        )

        self.assertAlmostEqual(signed_value, theta - threshold)
        self.assertEqual(violation, 0.0)

    def test_smooth_constraint_can_return_separate_surrogate_and_hard_signals(self):
        theta = 0.60
        threshold = np.pi / 4.0
        curve = _FakeCurve(self._inboard_poloidal_points(theta, count=256))

        (
            surrogate_signed_value,
            _grad_value,
            surrogate_violation,
            hard_signed_value,
            hard_violation,
        ) = self.module.smooth_max_poloidal_extent_signed_constraint(
            curve,
            R_winding=0.976,
            theta_threshold=threshold,
            temperature=0.05,
            objective_optimizable=object(),
            include_hard_signal=True,
        )

        self.assertGreater(surrogate_signed_value, 0.0)
        self.assertAlmostEqual(surrogate_violation, surrogate_signed_value)
        self.assertAlmostEqual(hard_signed_value, theta - threshold)
        self.assertEqual(hard_violation, 0.0)


def _manufacturability_test_curve():
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    gamma = np.column_stack(
        [
            0.976 + 0.08 * np.cos(theta),
            0.08 * np.sin(theta),
            0.02 * np.sin(2.0 * theta),
        ]
    )
    gammadash = np.column_stack(
        [
            -0.08 * 2.0 * np.pi * np.sin(theta),
            0.08 * 2.0 * np.pi * np.cos(theta),
            0.04 * 2.0 * np.pi * np.cos(2.0 * theta),
        ]
    )
    return _FakeOptimizableCurve(gamma, gammadash, order=3)


class EllipseWidthModuleTests(_ModuleTestCase):
    MODULE_PATH = ELLIPSE_WIDTH_PATH
    MODULE_PREFIX = "banana_ellipse_width"

    def test_projected_ellipse_width_instantiates_and_returns_finite_width(self):
        objective = self.module.ProjectedEllipseWidth(
            _manufacturability_test_curve(),
            0.976,
            0.210,
        )

        width = objective.J()

        self.assertGreater(width, 0.0)
        self.assertTrue(np.isfinite(width))


class CurveSelfIntersectModuleTests(_ModuleTestCase):
    MODULE_PATH = SELF_INTERSECT_PATH
    MODULE_PREFIX = "banana_self_intersect"

    def test_curve_self_intersect_instantiates_and_returns_finite_penalty(self):
        objective = self.module.CurveSelfIntersect(
            _manufacturability_test_curve(),
            0.01,
            neighbor_skip=2,
        )

        penalty = objective.J()

        self.assertGreaterEqual(penalty, 0.0)
        self.assertTrue(np.isfinite(penalty))


class Stage2ObjectiveModuleTests(_ModuleTestCase):
    MODULE_PATH = STAGE2_OBJECTIVES_PATH
    MODULE_PREFIX = "banana_stage2_objectives"

    def _assert_stage2_alm_signal_contract(self, result):
        for index, dual_kind in enumerate(result["dual_update_value_kinds"]):
            hard_dual_signal = dual_kind == "hard"
            normalized_dual_key = (
                "hard_dual_update_values"
                if hard_dual_signal
                else "surrogate_signed_constraint_values"
            )
            raw_dual_key = (
                "raw_hard_dual_update_values"
                if hard_dual_signal
                else "raw_surrogate_signed_constraint_values"
            )
            self.assertAlmostEqual(
                result["dual_update_values"][index],
                result[normalized_dual_key][index],
            )
            self.assertAlmostEqual(
                result["raw_dual_update_values"][index],
                result[raw_dual_key][index],
            )

        for index, feasibility_kind in enumerate(result["feasibility_value_kinds"]):
            if feasibility_kind == "hard":
                expected_feasibility = result["hard_violation_values"][index]
                expected_raw_feasibility = result["raw_hard_violation_values"][index]
            else:
                expected_feasibility = max(
                    result["surrogate_signed_constraint_values"][index],
                    0.0,
                )
                expected_raw_feasibility = max(
                    result["raw_surrogate_signed_constraint_values"][index],
                    0.0,
                )
            self.assertAlmostEqual(
                result["feasibility_values"][index],
                expected_feasibility,
            )
            self.assertAlmostEqual(
                result["raw_feasibility_values"][index],
                expected_raw_feasibility,
            )
        np.testing.assert_allclose(
            result["raw_constraint_values"],
            result["raw_surrogate_signed_constraint_values"],
        )
        np.testing.assert_allclose(
            result["raw_solver_constraint_values"],
            result["raw_surrogate_signed_constraint_values"],
        )

    def _assert_restored_fake_boozer_state(self, fake_boozer_surface):
        np.testing.assert_allclose(fake_boozer_surface.surface.x, [0.0, 0.0])
        self.assertAlmostEqual(fake_boozer_surface.res["iota"], 0.21)
        self.assertAlmostEqual(fake_boozer_surface.res["G"], 0.35)
        self.assertTrue(fake_boozer_surface.res["success"])

    def _build_fake_stage2_iota_runtime(
        self,
        fake_boozer_surface,
        *,
        mode="soft",
        iota_target=0.2,
        iota_tolerance=5.0e-3,
    ):
        class _FakeIotaTerm:
            def __init__(self, boozer_surface):
                self.boozer_surface = boozer_surface

            def J(self):
                if getattr(self.boozer_surface, "need_to_run_code", False):
                    res = self.boozer_surface.res
                    self.boozer_surface.run_code(res["iota"], G=res["G"])
                    self.boozer_surface.need_to_run_code = False
                return float(self.boozer_surface.res["iota"])

            def dJ(self):
                return np.array([0.2, -0.1], dtype=float)

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
            iota_target=iota_target,
            iota_tolerance=iota_tolerance,
            constraint_weight=None,
            num_tf_coils=2,
            mode=mode,
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
            derive_signed_G_fn=lambda _bs, *, tf_coils: 0.35,
            iotas_cls=_FakeIotaTerm,
            quadratic_penalty_cls=_FakeQuadraticPenalty,
        )

    def test_stage2_alm_constraint_metadata_requires_explicit_iota_threshold(self):
        with self.assertRaisesRegex(ValueError, "iota_penalty.*explicit threshold"):
            self.module._stage2_alm_constraint_metadata(
                ("iota_penalty",),
                threshold_overrides={},
                activity_tolerance_by_name={"iota_penalty": 0.0},
                iota_penalty_threshold=None,
            )

    def test_stage2_alm_constraint_metadata_rejects_explicit_zero_iota_threshold(self):
        # Zero is invalid as a normalization scale base — the silent-floor
        # behavior would inflate the signal by 1e12. Validation now fails fast.
        with self.assertRaisesRegex(
            ValueError,
            r"ALM threshold 'stage2:iota_penalty' must be a finite positive value",
        ):
            self.module._stage2_alm_constraint_metadata(
                ("iota_penalty",),
                threshold_overrides={},
                activity_tolerance_by_name={"iota_penalty": 0.0},
                iota_penalty_threshold=0.0,
            )

    def test_stage2_alm_constraint_metadata_rejects_explicit_negative_iota_threshold(
        self,
    ):
        with self.assertRaisesRegex(
            ValueError,
            r"ALM threshold 'stage2:iota_penalty' must be a finite positive value",
        ):
            self.module._stage2_alm_constraint_metadata(
                ("iota_penalty",),
                threshold_overrides={},
                activity_tolerance_by_name={"iota_penalty": 0.0},
                iota_penalty_threshold=-1.0,
            )

    def test_stage2_alm_constraint_metadata_accepts_subnormal_positive_iota_threshold(
        self,
    ):
        # Subnormal positive threshold passes validation; floor applies as
        # defense-in-depth so scale never goes below ALM_OBJECTIVE_SCALE_FLOOR.
        metadata_by_name = self.module._stage2_alm_constraint_metadata(
            ("iota_penalty",),
            threshold_overrides={},
            activity_tolerance_by_name={"iota_penalty": 0.0},
            iota_penalty_threshold=1.0e-15,
        )
        metadata = metadata_by_name["iota_penalty"]
        self.assertEqual(metadata.raw_threshold, 1.0e-15)
        self.assertEqual(metadata.scale, self.module.ALM_OBJECTIVE_SCALE_FLOOR)

    def test_stage2_iota_alm_metadata_records_scale_floor_when_threshold_below_floor(
        self,
    ):
        # M7 provenance: a tiny-but-positive threshold floors the scale and the
        # metadata records both the bool flag and the ":floored" source suffix.
        metadata_by_name = self.module._stage2_alm_constraint_metadata(
            ("iota_penalty",),
            threshold_overrides={},
            activity_tolerance_by_name={"iota_penalty": 0.0},
            iota_penalty_threshold=5.0e-13,
        )
        metadata = metadata_by_name["iota_penalty"]
        self.assertEqual(metadata.scale, self.module.ALM_OBJECTIVE_SCALE_FLOOR)
        self.assertTrue(metadata.scale_floor_applied)
        self.assertTrue(metadata.source.endswith(":floored"))
        self.assertEqual(metadata.source, "stage2_iota_penalty_threshold:floored")

    def test_stage2_iota_alm_metadata_does_not_record_floor_when_threshold_above_floor(
        self,
    ):
        # M7 provenance: a healthy positive threshold leaves scale == raw and
        # neither the flag nor a ":floored" suffix is recorded.
        metadata_by_name = self.module._stage2_alm_constraint_metadata(
            ("iota_penalty",),
            threshold_overrides={},
            activity_tolerance_by_name={"iota_penalty": 0.0},
            iota_penalty_threshold=1.0,
        )
        metadata = metadata_by_name["iota_penalty"]
        self.assertEqual(metadata.scale, 1.0)
        self.assertFalse(metadata.scale_floor_applied)
        self.assertFalse(metadata.source.endswith(":floored"))
        self.assertEqual(metadata.source, "stage2_iota_penalty_threshold")

    def test_stage2_alm_constraint_metadata_requires_explicit_length_threshold(self):
        with self.assertRaisesRegex(
            ValueError,
            "coil_length_upper_bound.*explicit threshold",
        ):
            self.module._stage2_alm_constraint_metadata(
                ("coil_length_upper_bound",),
                threshold_overrides={},
                activity_tolerance_by_name={"coil_length_upper_bound": 0.0},
            )

    def test_make_stage2_fun_returns_value_grad_and_logs_metrics(self):
        class _JF:
            def __init__(self):
                self.x = None

            def J(self):
                return 1.23

            def dJ(self):
                return np.array([1.0, -2.0])

        new_bs = _FakeBiotSavart((2, 3))
        new_surf = _FakeSurfaceNormals((1, 2, 3))
        fun = self.module.make_stage2_fun(
            _JF(),
            new_bs,
            new_surf,
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
        new_bs = _FakeBiotSavart((2, 3))
        new_surf = _FakeSurfaceNormals((1, 2, 3))
        fun = self.module.make_stage2_fun(
            _JF(),
            new_bs,
            new_surf,
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
        np.testing.assert_allclose(new_bs.points, new_surf.gamma().reshape((-1, 3)))
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
        Jw = _FakeWidthObjective(0.10, [0.5, -0.25])
        Jself = _FakeSelfIntersectObjective(
            0.0, [0.1, -0.2], shortest_self_distance=0.5
        )

        def fake_augmented(
            base_value, base_grad, signed_values, grads, multipliers, penalty
        ):
            self.assertAlmostEqual(base_value, 3.5)
            np.testing.assert_allclose(base_grad, [1.2, -0.5])
            # Order: coil_coil_spacing, max_curvature, coil_length_upper_bound,
            # coil_length_min, width_min, width_max, self_intersect,
            # banana_current_upper_bound.
            np.testing.assert_allclose(
                signed_values,
                [
                    -0.16,
                    0.01875,
                    0.1,
                    -1.25 / 0.95,
                    -0.05 / 0.05,
                    -0.07 / 0.17,
                    0.0,
                    -0.40625,
                ],
            )
            np.testing.assert_allclose(grads[0], [12.0, 4.0])
            np.testing.assert_allclose(grads[1], [0.0225, -0.0025])
            np.testing.assert_allclose(grads[2], [0.15, 0.2])
            np.testing.assert_allclose(grads[3], [-0.3 / 0.95, -0.4 / 0.95])
            np.testing.assert_allclose(grads[4], [-10.0, 5.0])
            np.testing.assert_allclose(grads[5], [0.5 / 0.17, -0.25 / 0.17])
            np.testing.assert_allclose(grads[6], [0.1, -0.2])
            np.testing.assert_allclose(grads[7], [4.375e-5, -2.5e-5])
            np.testing.assert_allclose(
                multipliers, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            )
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
                multipliers=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
                penalty=12.0,
                stage2_constraint_activity_tolerances=lambda ds, cs: [
                    ds * 4.0,
                    cs * 4.0,
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-6,
                    1e-3,
                ],
                smooth_min_distance_signed_constraint=lambda *_args: (
                    -0.008,
                    np.array([0.6, 0.2]),
                    -0.008,
                ),
                smooth_max_curvature_signed_constraint=lambda *_args: (
                    0.75,
                    np.array([0.9, -0.1]),
                ),
                Jw=Jw,
                width_min_threshold=0.05,
                width_max_threshold=0.17,
                Jself=Jself,
                self_intersect_threshold=0.0,
                length_min_target=0.95,
            )

        np.testing.assert_allclose(base_objective.x, [0.25, -0.4])
        self.assertEqual(
            result["constraint_names"],
            [
                "coil_coil_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "coil_length_min",
                "width_min",
                "width_max",
                "self_intersect",
                "banana_current_upper_bound",
            ],
        )
        normalized_signs = [
            -0.16,
            0.025,
            0.1,
            -1.25 / 0.95,
            -0.05 / 0.05,
            -0.07 / 0.17,
            0.0,
            -0.40625,
        ]
        normalized_surrogate_signs = [
            -0.16,
            0.01875,
            0.1,
            -1.25 / 0.95,
            -0.05 / 0.05,
            -0.07 / 0.17,
            0.0,
            -0.40625,
        ]
        normalized_violations = [0.0, 0.025, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(
            result["dual_update_values"],
            normalized_signs,
        )
        np.testing.assert_allclose(
            result["hard_signed_constraint_values"],
            normalized_signs,
        )
        np.testing.assert_allclose(
            result["hard_violation_values"],
            normalized_violations,
        )
        np.testing.assert_allclose(
            result["surrogate_signed_constraint_values"],
            normalized_surrogate_signs,
        )
        np.testing.assert_allclose(
            result["hard_dual_update_values"],
            normalized_signs,
        )
        np.testing.assert_allclose(
            result["feasibility_values"],
            normalized_violations,
        )
        np.testing.assert_allclose(
            result["constraint_activity_tolerances"],
            [
                0.4,
                0.002,
                5.0e-4,
                1.0526315789473685e-3,
                0.02,
                5.882352941176471e-3,
                1.0e-6,
                6.25e-8,
            ],
        )
        np.testing.assert_allclose(
            result["constraint_scales"],
            [0.05, 40.0, 2.0, 0.95, 0.05, 0.17, 1.0, 16000.0],
        )
        self.assertEqual(
            result["constraint_blocks"],
            [
                "geometry",
                "geometry",
                "geometry",
                "geometry",
                "geometry",
                "geometry",
                "geometry",
                "current",
            ],
        )
        self.assertEqual(
            result["objective_value_kinds"],
            [
                "surrogate",
                "surrogate",
                "hard",
                "hard",
                "hard",
                "hard",
                "hard",
                "hard",
            ],
        )
        self.assertEqual(
            result["dual_update_value_kinds"],
            ["hard", "hard", "hard", "hard", "hard", "hard", "hard", "hard"],
        )
        self.assertEqual(
            result["feasibility_value_kinds"],
            ["hard", "hard", "hard", "hard", "hard", "hard", "hard", "hard"],
        )
        np.testing.assert_allclose(
            result["raw_dual_update_values"],
            [-0.008, 1.0, 0.2, -1.25, -0.05, -0.07, 0.0, -6500.0],
        )
        np.testing.assert_allclose(
            result["raw_feasibility_values"],
            [0.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            result["raw_constraint_activity_tolerances"],
            [0.02, 0.08, 1e-3, 1e-3, 1e-3, 1e-3, 1e-6, 1e-3],
        )
        self.assertAlmostEqual(result["max_feasibility_violation"], 0.1)
        self.assertAlmostEqual(result["total"], 9.0)
        np.testing.assert_allclose(result["grad"], [7.0, -3.0])
        self._assert_stage2_alm_signal_contract(result)

    def test_evaluate_stage2_alm_problem_uses_hard_poloidal_feasibility(self):
        base_objective = _FakeBaseObjective(3.5, [1.2, -0.5])
        new_surf = _FakeSurfaceNormals((2, 2, 3))
        new_bs = _FakeBiotSavart((4, 3))
        Jf = _FakeScalarObjective(0.25)
        Jls = _FakeLengthObjective(2.2, [0.3, 0.4])
        Jccdist = _UnexpectedCurveDistance(0.05, 0.04)
        Jc = _FakeCurvatureObjective(40.0, [35.0, 41.0, 38.0], 7.5)
        banana_current = _FakeCurrentObjective(9500.0, [0.7, -0.4])
        Jpoloidal = SimpleNamespace(
            curve=_FakeCurve([[0.876, 0.0, 0.0]]),
            R_winding=0.976,
            Z_winding=0.0,
        )

        def fake_augmented(
            base_value, base_grad, signed_values, grads, multipliers, penalty
        ):
            self.assertAlmostEqual(base_value, 3.5)
            np.testing.assert_allclose(base_grad, [1.2, -0.5])
            # Order with poloidal extent included:
            # coil_coil_spacing, max_curvature, coil_length_upper_bound,
            # coil_length_min, poloidal_extent, width_min, width_max,
            # self_intersect, banana_current_upper_bound.
            self.assertAlmostEqual(signed_values[4], 0.2)
            self.assertAlmostEqual(penalty, 12.0)
            np.testing.assert_allclose(grads[4], [0.4, 0.6])
            return {
                "total": 9.0,
                "grad": np.array([7.0, -3.0]),
                "stationarity_norm": 0.5,
            }

        def fake_poloidal_constraint(*_args, include_hard_signal=False, **_kwargs):
            self.assertTrue(include_hard_signal)
            return 0.2, np.array([0.4, 0.6]), 0.2, -0.1, 0.0

        def fake_stage2_constraint_activity_tolerances(
            distance_smoothing,
            curvature_smoothing,
            include_poloidal_extent=False,
        ):
            tolerances = [
                distance_smoothing * 4.0,
                curvature_smoothing * 4.0,
                1e-3,
                1e-3,
            ]
            if include_poloidal_extent:
                tolerances.append(curvature_smoothing)
            tolerances.extend([1e-3, 1e-3, 1e-6, 1e-3])
            return tolerances

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
                multipliers=np.zeros(9),
                penalty=12.0,
                stage2_constraint_activity_tolerances=(
                    fake_stage2_constraint_activity_tolerances
                ),
                smooth_min_distance_signed_constraint=lambda *_args: (
                    -0.008,
                    np.array([0.6, 0.2]),
                    -0.008,
                ),
                smooth_max_curvature_signed_constraint=lambda *_args: (
                    0.75,
                    np.array([0.9, -0.1]),
                ),
                Jpoloidal=Jpoloidal,
                poloidal_extent_threshold_rad=1.0,
                poloidal_extent_smoothing=0.05,
                smooth_poloidal_extent_signed_constraint=fake_poloidal_constraint,
                **_default_geometric_parity_kwargs(),
            )

        self.assertEqual(
            result["constraint_names"],
            [
                "coil_coil_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "coil_length_min",
                "poloidal_extent",
                "width_min",
                "width_max",
                "self_intersect",
                "banana_current_upper_bound",
            ],
        )
        self.assertAlmostEqual(result["raw_feasibility_values"][4], 0.0)
        self.assertAlmostEqual(
            result["raw_surrogate_signed_constraint_values"][4],
            0.2,
        )

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
        distance_constraint_base = _affine_signed_constraint(
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

        def distance_constraint(*args):
            signed_value, grad = distance_constraint_base(*args)
            return signed_value, grad, signed_value

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
                    ds * 4.0,
                    cs * 4.0,
                    1.0e-3,
                    1.0e-3,
                    1.0e-3,
                    1.0e-3,
                    1.0e-6,
                    1.0e-3,
                ],
                smooth_min_distance_signed_constraint=distance_constraint,
                smooth_max_curvature_signed_constraint=curvature_constraint,
                **_default_geometric_parity_kwargs(),
            )

        result = alm_utils.run_directional_taylor_test(
            evaluate_problem,
            np.array([0.2, -0.3]),
            np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]),
            7.0,
            epsilons=TAYLOR_TEST_EPSILONS,
        )

        self.assertTrue(result["passed"], result)
        self.assertGreaterEqual(result["direction_count"], 4)

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
            multipliers=np.zeros(8),
            penalty=12.0,
            stage2_constraint_activity_tolerances=lambda ds, cs: [
                ds * 4.0,
                cs * 4.0,
                1e-3,
                1e-3,
                1e-3,
                1e-3,
                1e-6,
                1e-3,
            ],
            smooth_min_distance_signed_constraint=lambda *_args: (
                -0.008,
                np.array([0.6, 0.2]),
                0.01,
            ),
            smooth_max_curvature_signed_constraint=lambda *_args: (
                0.75,
                np.array([0.9, -0.1]),
            ),
            **_default_geometric_parity_kwargs(),
        )

        self.assertNotIn("diagnostics_included", result)
        self.assertAlmostEqual(result["base_value"], 3.5)
        self.assertEqual(
            result["constraint_names"],
            [
                "coil_coil_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "coil_length_min",
                "width_min",
                "width_max",
                "self_intersect",
                "banana_current_upper_bound",
            ],
        )
        self.assertAlmostEqual(result["max_feasibility_violation"], 0.2)
        self._assert_stage2_alm_signal_contract(result)

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
            np.testing.assert_allclose(
                signed_values,
                [
                    20.0,
                    0.01875,
                    0.1,
                    -1.25 / 0.95,
                    -0.05 / 0.05,
                    -0.07 / 0.17,
                    0.0,
                    -0.40625,
                ],
            )
            np.testing.assert_allclose(grads[0], [0.0, 0.0])
            np.testing.assert_allclose(grads[1], [0.0225, -0.0025])
            np.testing.assert_allclose(grads[2], [0.15, 0.2])
            np.testing.assert_allclose(grads[3], [-0.3 / 0.95, -0.4 / 0.95])
            np.testing.assert_allclose(grads[7], [4.375e-5, -2.5e-5])
            np.testing.assert_allclose(
                multipliers, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
            )
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
                multipliers=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
                penalty=12.0,
                stage2_constraint_activity_tolerances=lambda ds, cs: [
                    ds * 4.0,
                    cs * 4.0,
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-6,
                    1e-3,
                ],
                smooth_min_distance_signed_constraint=lambda *_args: (
                    np.nan,
                    np.array([np.nan, np.nan]),
                    np.nan,
                ),
                smooth_max_curvature_signed_constraint=lambda *_args: (
                    0.75,
                    np.array([0.9, -0.1]),
                ),
                **_default_geometric_parity_kwargs(),
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
            result["raw_dual_update_values"],
            [1.0, 1.0, 0.2, -1.25, -0.05, -0.07, 0.0, -6500.0],
        )
        np.testing.assert_allclose(
            result["raw_hard_signed_constraint_values"],
            [1.0, 1.0, 0.2, -1.25, -0.05, -0.07, 0.0, -6500.0],
        )
        np.testing.assert_allclose(
            result["raw_hard_violation_values"],
            [1.0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(
            result["raw_surrogate_signed_constraint_values"],
            [1.0, 0.75, 0.2, -1.25, -0.05, -0.07, 0.0, -6500.0],
        )
        np.testing.assert_allclose(result["constraint_grads"][0], [0.0, 0.0])
        self._assert_stage2_alm_signal_contract(result)

    def test_stage2_constraint_activity_tolerances_track_smoothing_windows(self):
        tolerances = self.module.stage2_constraint_activity_tolerances(0.005, 0.05)
        # Order: coil_coil_spacing, max_curvature, coil_length_upper_bound,
        # coil_length_min, width_min, width_max, self_intersect,
        # banana_current_upper_bound.
        self.assertEqual(
            tolerances,
            [0.02, 0.2, 1e-3, 1e-3, 1e-3, 1e-3, 1e-6, 1e-3],
        )

    def test_stage2_constraint_activity_tolerances_accept_explicit_endcaps(self):
        tolerances = self.module.stage2_constraint_activity_tolerances(
            0.005,
            0.05,
            length_tolerance=2e-3,
            banana_current_tolerance=3e-3,
        )
        self.assertEqual(
            tolerances,
            [0.02, 0.2, 2e-3, 2e-3, 1e-3, 1e-3, 1e-6, 3e-3],
        )

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
            multipliers=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            penalty=12.0,
            stage2_constraint_activity_tolerances=lambda ds, cs: [
                ds * 4.0,
                cs * 4.0,
                1e-3,
                1e-3,
                1e-3,
                1e-3,
                1e-6,
                1e-3,
            ],
            smooth_min_distance_signed_constraint=lambda *_args: (
                -0.008,
                np.array([0.6, 0.2]),
                -0.008,
            ),
            smooth_max_curvature_signed_constraint=lambda *_args: (
                0.75,
                np.array([0.9, -0.1]),
            ),
            **_default_geometric_parity_kwargs(),
        )

        self.assertEqual(result["constraint_names"][-1], "banana_current_upper_bound")
        self.assertAlmostEqual(result["dual_update_values"][-1], 0.0625)
        self.assertAlmostEqual(result["feasibility_values"][-1], 0.0625)
        self.assertAlmostEqual(result["raw_dual_update_values"][-1], 1000.0)
        np.testing.assert_allclose(result["constraint_grads"][-1], [-4.375e-5, 2.5e-5])

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
            multipliers=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
            penalty=12.0,
            stage2_constraint_activity_tolerances=lambda ds, cs: [
                ds * 5.0,
                cs * 6.0,
                2e-3,
                4e-3,
                5e-3,
                6e-3,
                3e-6,
                7e-3,
            ],
            smooth_min_distance_signed_constraint=lambda *_args: (
                -0.008,
                np.array([0.6, 0.2]),
                -0.008,
            ),
            smooth_max_curvature_signed_constraint=lambda *_args: (
                0.75,
                np.array([0.9, -0.1]),
            ),
            **_default_geometric_parity_kwargs(),
        )

        np.testing.assert_allclose(
            result["constraint_activity_tolerances"],
            [
                0.5,
                0.003,
                1.0e-3,
                4e-3 / 0.95,
                5e-3 / 0.05,
                6e-3 / 0.17,
                3e-6,
                4.375e-7,
            ],
        )
        np.testing.assert_allclose(
            result["raw_constraint_activity_tolerances"],
            [0.025, 0.12, 2e-3, 4e-3, 5e-3, 6e-3, 3e-6, 7e-3],
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
            # Iota constraint is appended last after the geometric parity set.
            self.assertAlmostEqual(signed_values[-1], 0.2)
            np.testing.assert_allclose(grads[-1], [0.4, 0.2])
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
                multipliers=np.zeros(9),
                penalty=12.0,
                stage2_constraint_activity_tolerances=lambda ds, cs: [
                    ds * 4.0,
                    cs * 4.0,
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-6,
                    1e-3,
                    0.5,
                ],
                smooth_min_distance_signed_constraint=lambda *_args: (
                    -0.008,
                    np.array([0.6, 0.2]),
                    -0.008,
                ),
                smooth_max_curvature_signed_constraint=lambda *_args: (
                    0.75,
                    np.array([0.9, -0.1]),
                ),
                stage2_iota_runtime=stage2_iota_runtime,
                **_default_geometric_parity_kwargs(),
            )

        self.assertEqual(
            result["constraint_names"],
            [
                "coil_coil_spacing",
                "max_curvature",
                "coil_length_upper_bound",
                "coil_length_min",
                "width_min",
                "width_max",
                "self_intersect",
                "banana_current_upper_bound",
                "iota_penalty",
            ],
        )
        self.assertAlmostEqual(result["raw_dual_update_values"][-1], 0.1)
        self.assertAlmostEqual(result["raw_hard_violation_values"][-1], 0.1)
        self.assertAlmostEqual(result["raw_constraint_activity_tolerances"][-1], 0.5)
        self._assert_stage2_alm_signal_contract(result)

    def test_evaluate_stage2_alm_problem_rejects_missing_iota_threshold(self):
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
            penalty_threshold=None,
        )

        with (
            mock.patch.object(
                self.module,
                "evaluate_stage2_iota",
                side_effect=AssertionError("iota evaluation should not run"),
            ),
            self.assertRaisesRegex(ValueError, "iota_penalty.*explicit threshold"),
        ):
            self.module.evaluate_stage2_alm_problem(
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
                multipliers=np.zeros(9),
                penalty=12.0,
                stage2_constraint_activity_tolerances=lambda ds, cs: [
                    ds * 4.0,
                    cs * 4.0,
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-6,
                    1e-3,
                    0.5,
                ],
                smooth_min_distance_signed_constraint=lambda *_args: (
                    -0.008,
                    np.array([0.6, 0.2]),
                    -0.008,
                ),
                smooth_max_curvature_signed_constraint=lambda *_args: (
                    0.75,
                    np.array([0.9, -0.1]),
                ),
                stage2_iota_runtime=stage2_iota_runtime,
                **_default_geometric_parity_kwargs(),
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
            self.assertAlmostEqual(signed_values[-1], 2.0)
            np.testing.assert_allclose(grads[-1], [0.0, 0.0])
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
                multipliers=np.zeros(9),
                penalty=12.0,
                stage2_constraint_activity_tolerances=lambda ds, cs: [
                    ds * 4.0,
                    cs * 4.0,
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-3,
                    1e-6,
                    1e-3,
                    0.5,
                ],
                smooth_min_distance_signed_constraint=lambda *_args: (
                    -0.008,
                    np.array([0.6, 0.2]),
                    -0.008,
                ),
                smooth_max_curvature_signed_constraint=lambda *_args: (
                    0.75,
                    np.array([0.9, -0.1]),
                ),
                stage2_iota_runtime=stage2_iota_runtime,
                **_default_geometric_parity_kwargs(),
            )

        np.testing.assert_allclose(
            result["constraint_grads"][-1],
            [0.0, 0.0],
        )
        self.assertAlmostEqual(result["raw_dual_update_values"][-1], 1.0)
        self.assertAlmostEqual(result["raw_hard_violation_values"][-1], 1.0)
        self._assert_stage2_alm_signal_contract(result)

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

    def test_build_stage2_iota_runtime_uses_warm_start_surface_seed(self):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, 0.35)
        warm_start_surface = SimpleNamespace(nfp=7)
        calls = {}

        def fail_build_surface_configs(*_args, **_kwargs):
            raise AssertionError("warm-start path must not rebuild VMEC surface config")

        def fake_warm_start_loader(path):
            calls["warm_start_path"] = path
            return SimpleNamespace(
                surface=warm_start_surface,
                iota=0.17654321,
                G=0.35,
                has_solved_state=True,
            )

        def fake_attempt_initialize(
            surf_prev,
            _mpol,
            _ntor,
            _bs,
            vol_target,
            _constraint_weight,
            iota,
            G0,
            *,
            boozer_I,
            initial_surface_guess,
            nfp,
        ):
            calls["surf_prev"] = surf_prev
            calls["target_volume"] = vol_target
            calls["iota"] = iota
            calls["G0"] = G0
            calls["boozer_I"] = boozer_I
            calls["initial_surface_guess"] = initial_surface_guess
            calls["nfp"] = nfp
            return SimpleNamespace(
                success=True,
                boozer_surface=fake_boozer_surface,
                solve_success=True,
                self_intersecting=False,
                solved_iota=0.21,
                error_type=None,
                error_message=None,
            )

        runtime = self.module.build_stage2_iota_runtime(
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
            mode="report",
            boozer_I=0.0125,
            stage2_seed_surf_path="/tmp/warm_surface.json",
            build_surface_configs_fn=fail_build_surface_configs,
            attempt_initialize_boozer_surface_fn=fake_attempt_initialize,
            derive_signed_G_fn=lambda _bs, *, tf_coils: -0.35,
            warm_start_loader=fake_warm_start_loader,
            iotas_cls=lambda _surface: SimpleNamespace(J=lambda: 0.21),
            quadratic_penalty_cls=lambda term, target: SimpleNamespace(
                J=lambda: 0.5 * (term.J() - target) ** 2,
                dJ=lambda: np.array([0.2, -0.1], dtype=float),
            ),
        )

        self.assertAlmostEqual(runtime.initial_state.iota, 0.21)
        self.assertEqual(calls["warm_start_path"], "/tmp/warm_surface.json")
        self.assertIs(calls["surf_prev"], warm_start_surface)
        self.assertIs(calls["initial_surface_guess"], warm_start_surface)
        self.assertEqual(calls["nfp"], 7)
        self.assertAlmostEqual(calls["target_volume"], 0.12)
        self.assertAlmostEqual(calls["iota"], 0.17654321)
        self.assertAlmostEqual(calls["G0"], 0.35)
        self.assertAlmostEqual(calls["boozer_I"], 0.0125)

    def test_build_stage2_iota_runtime_rebuilds_when_warm_start_has_no_solved_state(
        self,
    ):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.21, -0.35)
        warm_start_surface = SimpleNamespace(nfp=7)
        cold_surface = SimpleNamespace(nfp=5)
        calls = {}

        def fake_build_surface_configs(*_args):
            calls["built_cold_surface"] = True
            return [{"initial_surface": cold_surface, "target_volume": 0.12}]

        def fake_warm_start_loader(path):
            calls["warm_start_path"] = path
            return SimpleNamespace(
                surface=warm_start_surface,
                iota=0.17654321,
                G=None,
                has_solved_state=False,
            )

        def fake_attempt_initialize(
            surf_prev,
            _mpol,
            _ntor,
            _bs,
            vol_target,
            _constraint_weight,
            iota,
            G0,
            *,
            boozer_I,
            initial_surface_guess,
            nfp,
        ):
            calls["surf_prev"] = surf_prev
            calls["target_volume"] = vol_target
            calls["iota"] = iota
            calls["G0"] = G0
            calls["boozer_I"] = boozer_I
            calls["initial_surface_guess"] = initial_surface_guess
            calls["nfp"] = nfp
            return SimpleNamespace(
                success=True,
                boozer_surface=fake_boozer_surface,
                solve_success=True,
                self_intersecting=False,
                solved_iota=0.21,
                error_type=None,
                error_message=None,
            )

        runtime = self.module.build_stage2_iota_runtime(
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
            mode="report",
            stage2_seed_surf_path="/tmp/surface_only.json",
            build_surface_configs_fn=fake_build_surface_configs,
            attempt_initialize_boozer_surface_fn=fake_attempt_initialize,
            derive_signed_G_fn=lambda _bs, *, tf_coils: -0.35,
            warm_start_loader=fake_warm_start_loader,
            iotas_cls=lambda _surface: SimpleNamespace(J=lambda: 0.21),
            quadratic_penalty_cls=lambda term, target: SimpleNamespace(
                J=lambda: 0.5 * (term.J() - target) ** 2,
                dJ=lambda: np.array([0.2, -0.1], dtype=float),
            ),
        )

        self.assertAlmostEqual(runtime.initial_state.iota, 0.21)
        self.assertEqual(calls["warm_start_path"], "/tmp/surface_only.json")
        self.assertTrue(calls["built_cold_surface"])
        self.assertIs(calls["surf_prev"], cold_surface)
        self.assertIsNone(calls["initial_surface_guess"])
        self.assertEqual(calls["nfp"], 5)
        self.assertAlmostEqual(calls["target_volume"], 0.12)
        self.assertAlmostEqual(calls["iota"], 0.2)
        self.assertAlmostEqual(calls["G0"], -0.35)

    def test_stage2_iota_alm_floor_does_not_penalize_iota_above_floor(self):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.23, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(
            fake_boozer_surface,
            mode="alm-floor",
            iota_target=0.2,
        )

        self.assertAlmostEqual(runtime.initial_state.iota, 0.23)
        self.assertAlmostEqual(runtime.initial_state.penalty, 0.0)
        self.assertAlmostEqual(runtime.initial_state.abs_error, 0.03)
        self.assertTrue(runtime.initial_state.feasible)
        np.testing.assert_allclose(runtime.penalty_objective.dJ(), [0.0, 0.0])

    def test_stage2_iota_alm_floor_penalizes_only_shortfall(self):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 0.17, 0.35)
        runtime = self._build_fake_stage2_iota_runtime(
            fake_boozer_surface,
            mode="alm-floor",
            iota_target=0.2,
            iota_tolerance=0.005,
        )

        self.assertAlmostEqual(runtime.initial_state.iota, 0.17)
        self.assertAlmostEqual(runtime.initial_state.penalty, 0.5 * 0.03 * 0.03)
        self.assertAlmostEqual(runtime.initial_state.abs_error, 0.03)
        self.assertFalse(runtime.initial_state.feasible)
        np.testing.assert_allclose(
            runtime.penalty_objective.dJ(),
            [-0.006, 0.003],
        )

    def test_build_stage2_iota_runtime_keeps_failed_bootstrap_as_runtime_state(self):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 3908.0, 0.35)
        fake_boozer_surface.res["success"] = False

        runtime = self.module.build_stage2_iota_runtime(
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
                    success=False,
                    boozer_surface=fake_boozer_surface,
                    solve_success=False,
                    self_intersecting=True,
                    solved_iota=3908.0,
                    solved_G=0.35,
                    error_type=None,
                    error_message=None,
                )
            ),
            derive_signed_G_fn=lambda _bs, *, tf_coils: 0.35,
            iotas_cls=lambda _surface: SimpleNamespace(J=lambda: 3908.0),
            quadratic_penalty_cls=lambda term, target: SimpleNamespace(
                J=lambda: 0.5 * (term.J() - target) ** 2,
                dJ=lambda: np.array([0.2, -0.1], dtype=float),
            ),
        )

        self.assertTrue(runtime.initial_state.solve_failed)
        self.assertFalse(runtime.initial_state.feasible)
        self.assertAlmostEqual(runtime.initial_state.iota, 3908.0)
        self.assertTrue(runtime.guarded_boozer_evaluator.last_solve_failed)
        self.assertEqual(
            runtime.guarded_boozer_evaluator.last_failure_reason,
            "self_intersecting",
        )

        state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertTrue(state.solve_failed)
        self.assertAlmostEqual(state.iota, 3908.0)
        self.assertEqual(runtime.stats.runtime_calls, 0)

    def test_build_stage2_iota_runtime_activates_after_failed_bootstrap_recovers(self):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 3908.0, 0.35)
        fake_boozer_surface.res["success"] = False
        runtime = self.module.build_stage2_iota_runtime(
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
            iota_tolerance=1.0e-2,
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
                    success=False,
                    boozer_surface=fake_boozer_surface,
                    solve_success=False,
                    self_intersecting=False,
                    solved_iota=3908.0,
                    solved_G=0.35,
                    error_type=None,
                    error_message=None,
                )
            ),
            derive_signed_G_fn=lambda _bs, *, tf_coils: 0.35,
            iotas_cls=lambda surface: SimpleNamespace(
                J=lambda: float(surface.res["iota"])
            ),
            quadratic_penalty_cls=lambda term, target: SimpleNamespace(
                J=lambda: 0.5 * (term.J() - target) ** 2,
                dJ=lambda: np.array([0.2, -0.1], dtype=float),
            ),
        )
        fake_boozer_surface.queue_result(
            surface_x=[1.0, -1.0],
            iota=0.205,
            G=0.36,
            success=True,
        )
        fake_boozer_surface.need_to_run_code = True

        state = self.module.evaluate_stage2_iota_state(runtime)

        self.assertFalse(state.solve_failed)
        self.assertTrue(state.feasible)
        self.assertAlmostEqual(state.iota, 0.205)
        self.assertFalse(runtime.guarded_boozer_evaluator.last_solve_failed)
        self.assertIsNone(runtime.guarded_boozer_evaluator.last_failure_reason)
        self.assertEqual(runtime.stats.runtime_calls, 1)
        np.testing.assert_allclose(
            runtime.guarded_boozer_evaluator.last_successful_state.surface_dofs,
            [1.0, -1.0],
        )

    def test_failed_bootstrap_runtime_preserves_self_intersection_check_error(self):
        fake_boozer_surface = _FakeBoozerSurface([0.0, 0.0], 3908.0, 0.35)
        fake_boozer_surface.res["success"] = False
        runtime = self.module.build_stage2_iota_runtime(
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
            iota_tolerance=1.0e-2,
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
                    success=False,
                    boozer_surface=fake_boozer_surface,
                    solve_success=False,
                    self_intersecting=False,
                    solved_iota=3908.0,
                    solved_G=0.35,
                    error_type=None,
                    error_message=None,
                )
            ),
            derive_signed_G_fn=lambda _bs, *, tf_coils: 0.35,
            iotas_cls=lambda surface: SimpleNamespace(
                J=lambda: float(surface.res["iota"])
            ),
            quadratic_penalty_cls=lambda term, target: SimpleNamespace(
                J=lambda: 0.5 * (term.J() - target) ** 2,
                dJ=lambda: np.array([0.2, -0.1], dtype=float),
            ),
        )
        fake_boozer_surface.queue_result(iota=0.205, G=0.36, success=True)
        fake_boozer_surface.surface.set_self_intersecting_raises(
            RuntimeError("ground missing")
        )
        fake_boozer_surface.need_to_run_code = True

        with self.assertRaisesRegex(RuntimeError, "ground missing"):
            self.module.evaluate_stage2_iota_state(runtime)

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

    def test_build_stage2_iota_runtime_propagates_self_intersection_check_exception(
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

        with self.assertRaisesRegex(RuntimeError, "ground missing"):
            self.module.evaluate_stage2_iota_state(runtime)
        self.assertIsNone(runtime.guarded_boozer_evaluator.last_failure_reason)
        self.assertEqual(fake_boozer_surface.surface.self_intersection_calls, 1)
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
                alm_fix_signal_mismatch_guard=False,
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
            banana_init_current_A=-1.0e4,
            banana_current_max_A=1.6e4,
            vf_current_max_A=1.6e4,
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
            alm_fix_signal_mismatch_guard=True,
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
            alm_schema_version=ALM_SCHEMA_VERSION,
            exit_class="feasible_stationarity_unmet",
            hard_constraints_feasible=True,
            stationarity_satisfied=False,
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
            wout_convention="signed_cw",
            wout_off_spec=False,
            num_tf_coils=2,
            num_banana_coils=4,
            num_proxy_coils=0,
            num_vf_coils=0,
            initial_banana_current_A=-1.2e4,
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
        self.assertTrue(result["ALM_FIX_SIGNAL_MISMATCH_GUARD"])
        self.assertEqual(result["ALM_OUTER_ITERATIONS"], 4)
        self.assertEqual(result["ALM_FINAL_TRUST_RADIUS"], 0.125)
        self.assertEqual(result["ALM_SCHEMA_VERSION"], ALM_SCHEMA_VERSION)
        self.assertEqual(result["ALM_EXIT_CLASS"], "feasible_stationarity_unmet")
        self.assertTrue(result["ALM_HARD_CONSTRAINTS_FEASIBLE"])
        self.assertFalse(result["ALM_STATIONARITY_SATISFIED"])
        self.assertEqual(result["CURVE_CURVE_DISTANCE_METRIC_KIND"], "banana_coils")
        self.assertTrue(result["ALM_MULTIPLIER_CAP_BINDING"])
        self.assertEqual(result["ALM_MULTIPLIER_CAP_BINDING_INDICES"], [1])
        np.testing.assert_allclose(
            result["ALM_FINAL_CONSTRAINT_VALUES"], [0.0, 1.0, 0.0]
        )
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
        np.testing.assert_allclose(
            result["ALM_FINAL_RAW_DUAL_ESTIMATES"], [0.1, 0.002, 0.3]
        )
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
        self.assertEqual(result["BANANA_INIT_CURRENT_A"], -1.2e4)
        self.assertEqual(result["BANANA_CURRENT_MAX_A"], 1.6e4)
        self.assertEqual(result["TF_CURRENT_LIMIT_A"], 8.0e4)
        self.assertEqual(result["WOUT_CONVENTION"], "signed_cw")
        self.assertIs(result["WOUT_OFF_SPEC"], False)
        self.assertAlmostEqual(result["BANANA_TO_TF_CURRENT_RATIO"], 0.11875)
        self.assertEqual(result["COIL_LENGTH"], 1.8)
        self.assertEqual(result["MAX_CURVATURE"], 41.0)
        self.assertEqual(result["CURVE_CURVE_MIN_DIST"], 0.04)
        self.assertEqual(result["CURVE_SURFACE_MIN_DIST"], 0.017)

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

        signed_value, grad, hard_signed_value = (
            self.module.smooth_min_distance_signed_constraint(
                [curve],
                minimum_distance=0.05,
                temperature=0.01,
                base_objective_optimizable=objective,
            )
        )

        self.assertAlmostEqual(signed_value, 0.05)
        self.assertAlmostEqual(hard_signed_value, 0.05)
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
            signed_value, grad, hard_signed_value = (
                self.module.smooth_min_curve_surface_signed_constraint(
                    [curve],
                    surface,
                    minimum_distance=0.05,
                    temperature=0.01,
                    base_objective_optimizable=SimpleNamespace(),
                )
            )

        self.assertAlmostEqual(signed_value, 0.05 - np.sqrt(0.02**2 + 0.04**2))
        self.assertAlmostEqual(hard_signed_value, 0.05 - np.sqrt(0.02**2 + 0.04**2))
        np.testing.assert_allclose(grad, [0.0, -0.04 / np.sqrt(0.02**2 + 0.04**2)])


class WoutConventionModuleTests(_ModuleTestCase):
    MODULE_PATH = WOUT_CONVENTION_PATH
    MODULE_PREFIX = "banana_wout_convention"

    @staticmethod
    def _write_wout(path: Path, *, rbtor: float, phi_edge: float):
        with netcdf_file(str(path), "w", mmap=False) as handle:
            handle.createDimension("scalar", 1)
            handle.createDimension("surf", 2)
            rbtor_var = handle.createVariable("rbtor", "f8", ("scalar",))
            rbtor_var[:] = [rbtor]
            phi_var = handle.createVariable("phi", "f8", ("surf",))
            phi_var[:] = [0.0, phi_edge]

    @staticmethod
    def _write_malformed_wout(path: Path):
        with netcdf_file(str(path), "w", mmap=False) as handle:
            handle.createDimension("scalar", 1)
            handle.createDimension("row", 2)
            handle.createDimension("col", 1)
            rbtor_var = handle.createVariable("rbtor", "f8", ("scalar",))
            rbtor_var[:] = [-0.32]
            phi_var = handle.createVariable("phi", "f8", ("row", "col"))
            phi_var[:] = [[0.0], [-0.005]]

    def test_artifact_fields_match_signed_tf_lane(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wout_path = Path(tmpdir) / "wout_signed.nc"
            self._write_wout(wout_path, rbtor=-0.32, phi_edge=-0.005)

            fields = self.module.wout_convention_artifact_fields(
                wout_path=wout_path,
                tf_current_A=-8.0e4,
            )

        self.assertEqual(fields["WOUT_CONVENTION"], "signed_cw")
        self.assertIs(fields["WOUT_OFF_SPEC"], False)

    def test_artifact_fields_reject_malformed_phi_shape(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            wout_path = Path(tmpdir) / "wout_malformed.nc"
            self._write_malformed_wout(wout_path)

            with self.assertRaisesRegex(ValueError, "phi is not 1-D"):
                self.module.wout_convention_artifact_fields(
                    wout_path=wout_path,
                    tf_current_A=-8.0e4,
                )

    def test_artifact_validator_rejects_stale_wout_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            wout_path = root / "wout_signed.nc"
            results_path = root / "results.json"
            self._write_wout(wout_path, rbtor=-0.32, phi_edge=-0.005)

            with self.assertRaisesRegex(ValueError, "WOUT_CONVENTION"):
                self.module.validate_wout_convention_artifact_fields(
                    stage2_results_path=results_path,
                    stage2_artifact_results={
                        "PLASMA_SURF_PATH": str(wout_path),
                        "TF_CURRENT_A": -8.0e4,
                        "WOUT_CONVENTION": "positive_ccw",
                        "WOUT_OFF_SPEC": False,
                    },
                )


class SingleStageObjectiveModuleTests(_ModuleTestCase):
    MODULE_PATH = SINGLE_STAGE_OBJECTIVES_PATH
    MODULE_PREFIX = "banana_single_stage_objectives"

    def test_physics_alm_metadata_records_scale_floor_when_threshold_below_floor(self):
        # M7 provenance: a tiny-but-positive threshold (5e-13 < 1e-12 floor)
        # floors the scale and stamps `scale_floor_applied=True` plus a
        # `:floored` source suffix so post-run audits are unambiguous.
        metadata = self.module._physics_alm_metadata(
            "qs_error",
            threshold=5.0e-13,
            activity_tolerance=0.0,
        )
        self.assertEqual(metadata.scale, self.module.ALM_OBJECTIVE_SCALE_FLOOR)
        self.assertTrue(metadata.scale_floor_applied)
        self.assertTrue(metadata.source.endswith(":floored"))
        self.assertEqual(metadata.source, "threshold:qs_error:floored")
        self.assertEqual(metadata.raw_threshold, 5.0e-13)

    def test_physics_alm_metadata_does_not_record_floor_when_threshold_above_floor(
        self,
    ):
        # M7 provenance: a healthy threshold leaves scale == raw and the
        # provenance flag stays `False` with no `:floored` source suffix.
        metadata = self.module._physics_alm_metadata(
            "qs_error",
            threshold=1.0,
            activity_tolerance=0.0,
        )
        self.assertEqual(metadata.scale, 1.0)
        self.assertFalse(metadata.scale_floor_applied)
        self.assertFalse(metadata.source.endswith(":floored"))
        self.assertEqual(metadata.source, "threshold:qs_error")
        self.assertEqual(metadata.raw_threshold, 1.0)

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

    def test_evaluate_total_objective_includes_length_floor_penalty(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        length_max = _FakeAlgebraicObjective(1.0, [0.1, 0.2])
        length_min = _FakeAlgebraicObjective(2.0, [0.3, -0.4])

        result = self.module.evaluate_total_objective(
            np.array([1.0]),
            [zero],
            [zero],
            RES_WEIGHT=0.0,
            Jiota=zero,
            IOTAS_WEIGHT=0.0,
            JCurveLength=length_max,
            LENGTH_WEIGHT=5.0,
            JCurveCurve=zero,
            CC_WEIGHT=0.0,
            JCurveSurface=zero,
            CS_WEIGHT=0.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=0.0,
            JCurveLengthMin=length_min,
        )

        self.assertAlmostEqual(result["total"], 15.0)
        self.assertAlmostEqual(result["J_len"], 1.0)
        self.assertAlmostEqual(result["J_len_min"], 2.0)
        np.testing.assert_allclose(result["grad"], [2.0, -1.0])

    def test_evaluate_total_objective_includes_self_intersect_term(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        self_intersect = _FakeAlgebraicObjective(2.0, [0.5, -0.25])

        result = self.module.evaluate_total_objective(
            np.array([1.0]),
            [zero],
            [zero],
            RES_WEIGHT=0.0,
            Jiota=zero,
            IOTAS_WEIGHT=0.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=0.0,
            JCurveCurve=zero,
            CC_WEIGHT=0.0,
            JCurveSurface=zero,
            CS_WEIGHT=0.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=0.0,
            JCurveSelfIntersect=self_intersect,
            SELFINT_WEIGHT=3.0,
        )

        self.assertAlmostEqual(result["total"], 6.0)
        self.assertAlmostEqual(result["J_self_intersect"], 2.0)
        np.testing.assert_allclose(result["dJ_self_intersect"], [0.5, -0.25])
        self.assertAlmostEqual(
            result["self_intersect_min_distance"],
            self.module.BANANA_SELF_INTERSECT_MIN_DISTANCE_M,
        )
        np.testing.assert_allclose(result["grad"], [1.5, -0.75])

    def test_evaluate_total_objective_skips_geometric_parity_when_objectives_missing(
        self,
    ):
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
            JCurveCurve=zero,
            CC_WEIGHT=0.0,
            JCurveSurface=zero,
            CS_WEIGHT=0.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=0.0,
        )

        self.assertAlmostEqual(result["J_coil_width"], 0.0)
        self.assertAlmostEqual(result["J_self_intersect"], 0.0)
        np.testing.assert_allclose(result["dJ_coil_width"], [0.0, 0.0])
        np.testing.assert_allclose(result["dJ_self_intersect"], [0.0, 0.0])
        self.assertIsNone(result["coil_width_min_threshold"])
        self.assertIsNone(result["coil_width_max_threshold"])
        self.assertIsNone(result["self_intersect_min_distance"])

    def test_evaluate_total_objective_includes_coil_width_term_via_quadratic_penalty(
        self,
    ):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        coil_width = _FakeAlgebraicObjective(0.02, [1.0, 0.5])

        def fake_quadratic_penalty(obj, cons, kind):
            value = float(obj.J())
            grad = obj.dJ()
            if kind == "min":
                hinge = min(value - float(cons), 0.0)
            elif kind == "max":
                hinge = max(value - float(cons), 0.0)
            else:
                raise ValueError(f"Unsupported quadratic-penalty kind {kind!r}")
            return _FakeAlgebraicObjective(
                0.5 * hinge * hinge,
                hinge * grad,
            )

        with mock.patch.object(
            self.module,
            "QuadraticPenalty",
            fake_quadratic_penalty,
        ):
            result = self.module.evaluate_total_objective(
                np.array([1.0]),
                [zero],
                [zero],
                RES_WEIGHT=0.0,
                Jiota=zero,
                IOTAS_WEIGHT=0.0,
                JCurveLength=zero,
                LENGTH_WEIGHT=0.0,
                JCurveCurve=zero,
                CC_WEIGHT=0.0,
                JCurveSurface=zero,
                CS_WEIGHT=0.0,
                JCurvature=zero,
                CURVATURE_WEIGHT=0.0,
                JCoilWidth=coil_width,
                WIDTH_WEIGHT=2.0,
            )

        # width=0.02 < min=0.1 -> min hinge = -0.08 -> 0.5*(-0.08)^2 = 0.0032
        # width=0.02 < max=0.17 -> max hinge = 0
        # weighted contribution: 2.0 * 0.0032 = 0.0064
        # gradient of min hinge: hinge * grad = -0.08 * [1.0, 0.5] = [-0.08, -0.04]
        # weighted: 2.0 * [-0.08, -0.04] = [-0.16, -0.08]
        self.assertAlmostEqual(result["total"], 0.0064, places=8)
        self.assertAlmostEqual(result["J_coil_width"], 0.02)
        np.testing.assert_allclose(result["dJ_coil_width"], [1.0, 0.5])
        self.assertAlmostEqual(
            result["coil_width_min_threshold"],
            self.module.BANANA_WIDTH_MIN_M,
        )
        self.assertAlmostEqual(
            result["coil_width_max_threshold"],
            self.module.BANANA_WIDTH_MAX_M,
        )
        np.testing.assert_allclose(result["grad"], [-0.16, -0.08])

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

    def test_evaluate_total_objective_includes_residue_objective(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        residue = _FakeResidueObjective(0.7, [0.4, -0.2])

        result = self.module.evaluate_total_objective(
            np.array([1.0]),
            [zero],
            [zero],
            RES_WEIGHT=0.0,
            Jiota=zero,
            IOTAS_WEIGHT=0.0,
            JCurveLength=zero,
            LENGTH_WEIGHT=0.0,
            JCurveCurve=zero,
            CC_WEIGHT=0.0,
            JCurveSurface=zero,
            CS_WEIGHT=0.0,
            JCurvature=zero,
            CURVATURE_WEIGHT=0.0,
            JResidueObjective=residue,
        )

        self.assertAlmostEqual(result["total"], 0.7)
        self.assertAlmostEqual(result["J_residue_objective"], 0.7)
        self.assertTrue(result["residue_objective_enabled"])
        self.assertEqual(
            result["residue_objective_payload"]["target_manifest_id"],
            "test-targets",
        )
        np.testing.assert_allclose(result["grad"], [0.4, -0.2])
        np.testing.assert_allclose(result["dJ_residue_objective"], [0.4, -0.2])

    def test_evaluate_alm_objective_includes_width_and_self_intersect_constraints(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        coil_width = _FakeAlgebraicObjective(0.12, [0.2, 0.0])
        self_intersect = _FakeAlgebraicObjective(0.003, [0.0, 0.4])

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
            np.testing.assert_allclose(
                constraint_values,
                [-1.4, -0.05 / 0.17, 0.003],
            )
            np.testing.assert_allclose(constraint_grads[0], [-4.0, 0.0])
            np.testing.assert_allclose(
                constraint_grads[1],
                [0.2 / 0.17, 0.0],
            )
            np.testing.assert_allclose(constraint_grads[2], [0.0, 0.4])
            np.testing.assert_allclose(multipliers, [0.1, 0.2, 0.3])
            self.assertAlmostEqual(penalty, 7.0)
            return {
                "total": 1.25,
                "grad": np.array([3.0, -2.0]),
                "stationarity_norm": 0.5,
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
            multipliers=np.array([0.1, 0.2, 0.3]),
            penalty=7.0,
            objective_optimizable=SimpleNamespace(),
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface="outer",
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=MANUFACTURABILITY_ALM_CONSTRAINT_NAMES,
            curve_curve_constraint_fn=lambda *_args: (-0.1, np.array([0.0, 0.0]), 0.0),
            curve_surface_constraint_fn=lambda *_args: (
                -0.1,
                np.array([0.0, 0.0]),
                0.0,
            ),
            curvature_constraint_fn=lambda *_args: (-0.1, np.array([0.0, 0.0]), 0.0),
            augmented_inequality_objective_fn=fake_augmented,
            activity_tolerances_fn=lambda *_args, **_kwargs: np.array(
                [0.0, 0.0, 0.0],
                dtype=float,
            ),
            JCoilWidth=coil_width,
            width_min_threshold=0.05,
            width_max_threshold=0.17,
            JCurveSelfIntersect=self_intersect,
        )

        self.assertEqual(
            result["constraint_names"],
            list(MANUFACTURABILITY_ALM_CONSTRAINT_NAMES),
        )
        np.testing.assert_allclose(result["constraint_scales"], [0.05, 0.17, 1.0])
        np.testing.assert_allclose(result["raw_thresholds"], [0.05, 0.17, 0.0])
        np.testing.assert_allclose(
            result["raw_constraint_values"], [-0.07, -0.05, 0.003]
        )
        np.testing.assert_allclose(result["feasibility_values"], [0.0, 0.0, 0.003])
        self.assertEqual(
            result["constraint_blocks"], ["geometry", "geometry", "geometry"]
        )
        self.assertEqual(result["objective_value_kinds"], ["hard", "hard", "hard"])
        self.assertEqual(result["gradient_value_kinds"], ["hard", "hard", "hard"])
        self.assertEqual(result["dual_update_value_kinds"], ["hard", "hard", "hard"])
        self.assertAlmostEqual(result["J_coil_width"], 0.12)
        np.testing.assert_allclose(result["dJ_coil_width"], [0.2, 0.0])
        self.assertAlmostEqual(result["J_self_intersect"], 0.003)
        np.testing.assert_allclose(result["dJ_self_intersect"], [0.0, 0.4])

    def test_evaluate_alm_objective_accepts_real_width_and_self_intersect_objectives(
        self,
    ):
        ellipse_module = _load_module(ELLIPSE_WIDTH_PATH, "banana_ellipse_width_real")
        self_intersect_module = _load_module(
            SELF_INTERSECT_PATH,
            "banana_self_intersect_real",
        )
        curve = _manufacturability_test_curve()
        coil_width = ellipse_module.ProjectedEllipseWidth(
            curve,
            0.976,
            0.210,
        )
        self_intersect = self_intersect_module.CurveSelfIntersect(
            curve,
            0.20,
            neighbor_skip=1,
        )
        width_value = coil_width.J()
        self_intersect_value = self_intersect.J()
        self.assertGreater(width_value, 0.0)
        self.assertGreater(self_intersect_value, 0.0)
        width_violation = 0.01
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])

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
            multipliers=np.array([0.0, 0.0, 0.0]),
            penalty=1.0,
            objective_optimizable=curve,
            curves=[curve],
            curve_curve_min_distance=0.05,
            outer_surface=object(),
            curve_surface_min_distance=0.02,
            banana_curve=curve,
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=MANUFACTURABILITY_ALM_CONSTRAINT_NAMES,
            curve_curve_constraint_fn=_zero_constraint_result_2d,
            curve_surface_constraint_fn=_zero_constraint_result_2d,
            curvature_constraint_fn=_zero_constraint_result_2d,
            activity_tolerances_fn=lambda *_args, **_kwargs: np.zeros(3),
            JCoilWidth=coil_width,
            width_min_threshold=width_value + width_violation,
            width_max_threshold=width_value - width_violation,
            JCurveSelfIntersect=self_intersect,
        )

        self.assertEqual(
            result["constraint_names"],
            list(MANUFACTURABILITY_ALM_CONSTRAINT_NAMES),
        )
        self.assertGreater(result["raw_constraint_values"][0], 0.0)
        self.assertGreater(result["raw_constraint_values"][1], 0.0)
        self.assertGreater(result["raw_constraint_values"][2], 0.0)
        np.testing.assert_allclose(
            result["raw_constraint_values"],
            [width_violation, width_violation, self_intersect_value],
            rtol=1.0e-10,
            atol=1.0e-12,
        )
        for gradient in result["raw_constraint_grads"]:
            self.assertEqual(gradient.shape, (2,))
            self.assertTrue(np.all(np.isfinite(gradient)))
        self.assertGreater(np.linalg.norm(result["raw_constraint_grads"][2]), 0.0)

    def test_evaluate_alm_objective_includes_lcfs_radius_constraints(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        lcfs_surface = _FakeRadiusSurface(
            major_radius=0.923,
            minor_radius=0.151,
            major_grad=[2.0, -1.0],
            minor_grad=[0.5, 0.25],
        )

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
            multipliers=np.array([0.0, 0.0]),
            penalty=1.0,
            objective_optimizable=lcfs_surface,
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface=lcfs_surface,
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=("lcfs_major_radius", "lcfs_minor_radius"),
            curve_curve_constraint_fn=lambda *_args: (-0.1, np.array([0.0, 0.0]), 0.0),
            curve_surface_constraint_fn=lambda *_args: (
                -0.1,
                np.array([0.0, 0.0]),
                0.0,
            ),
            curvature_constraint_fn=lambda *_args: (-0.1, np.array([0.0, 0.0]), 0.0),
            activity_tolerances_fn=lambda *_args, **_kwargs: np.array(
                [0.0, 0.0, 0.0],
                dtype=float,
            ),
            lcfs_surface=lcfs_surface,
            lcfs_major_radius_threshold=0.920,
            lcfs_minor_radius_threshold=0.150,
        )

        self.assertEqual(
            result["constraint_names"],
            ["lcfs_major_radius", "lcfs_minor_radius"],
        )
        np.testing.assert_allclose(result["raw_constraint_values"], [0.003, 0.001])
        np.testing.assert_allclose(
            result["raw_constraint_grads"],
            [[2.0, -1.0], [0.5, 0.25]],
        )
        self.assertEqual(result["constraint_blocks"], ["surface", "surface"])
        self.assertEqual(result["objective_value_kinds"], ["hard", "hard"])
        self.assertEqual(result["gradient_value_kinds"], ["hard", "hard"])
        self.assertEqual(result["dual_update_value_kinds"], ["hard", "hard"])

    def test_evaluate_alm_objective_includes_lcfs_edge_envelope_constraints(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        lcfs_surface = _FakeRadiusSurface(
            major_radius=0.910,
            minor_radius=0.140,
            major_grad=[2.0, -1.0],
            minor_grad=[0.5, 0.25],
        )

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
            multipliers=np.array([0.0, 0.0, 0.0]),
            penalty=1.0,
            objective_optimizable=lcfs_surface,
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface=lcfs_surface,
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=(
                "lcfs_outboard_edge",
                "lcfs_inboard_edge",
                "lcfs_minor_radius",
            ),
            curve_curve_constraint_fn=lambda *_args: (-0.1, np.array([0.0, 0.0]), 0.0),
            curve_surface_constraint_fn=lambda *_args: (
                -0.1,
                np.array([0.0, 0.0]),
                0.0,
            ),
            curvature_constraint_fn=lambda *_args: (-0.1, np.array([0.0, 0.0]), 0.0),
            activity_tolerances_fn=lambda *_args, **_kwargs: np.array(
                [0.0, 0.0, 0.0],
                dtype=float,
            ),
            lcfs_surface=lcfs_surface,
            lcfs_constraint_mode="edge_envelope",
            lcfs_outboard_edge_threshold=1.035,
            lcfs_inboard_edge_threshold=0.771,
            lcfs_minor_radius_threshold=0.132,
        )

        self.assertEqual(
            result["constraint_names"],
            ["lcfs_outboard_edge", "lcfs_inboard_edge", "lcfs_minor_radius"],
        )
        np.testing.assert_allclose(
            result["raw_constraint_values"],
            [0.015, 0.001, 0.008],
            atol=1e-14,
        )
        np.testing.assert_allclose(
            result["raw_constraint_grads"],
            [[2.5, -0.75], [-1.5, 1.25], [0.5, 0.25]],
        )
        self.assertEqual(result["constraint_blocks"], ["surface", "surface", "surface"])
        self.assertEqual(result["objective_value_kinds"], ["hard", "hard", "hard"])
        self.assertEqual(result["gradient_value_kinds"], ["hard", "hard", "hard"])
        self.assertEqual(result["dual_update_value_kinds"], ["hard", "hard", "hard"])

    def test_evaluate_alm_objective_uses_hard_surface_stack_for_dual_signal(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        curves = (
            _FakeCurve(gamma_points=[[0.0, 0.0, 0.0]]),
            _FakeCurve(gamma_points=[[1.0, 0.0, 0.0]]),
        )
        outer_surface = _FakeSurfaceWithGradient(gamma_points=[[[0.5, 0.0, 0.0]]])
        surface_a = _FakeSurfaceWithGradient(gamma_points=[[[0.0, 0.0, 0.0]]])
        surface_b = _FakeSurfaceWithGradient(gamma_points=[[[0.04, 0.0, 0.0]]])
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
        np.testing.assert_allclose(result["raw_constraint_values"], [0.02])
        np.testing.assert_allclose(result["raw_solver_constraint_values"], [0.02])
        np.testing.assert_allclose(result["raw_dual_update_values"], [0.01])
        np.testing.assert_allclose(result["raw_hard_signed_constraint_values"], [0.01])
        np.testing.assert_allclose(
            result["raw_surrogate_signed_constraint_values"],
            [0.02],
        )

    def test_evaluate_alm_objective_uses_hard_poloidal_feasibility_signal(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        curve = _FakeCurve(
            gamma_points=[[0.876, 0.0, 0.0]],
            kappa_values=[5.0],
        )
        JPoloidalExtent = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        JPoloidalExtent.curve = curve
        JPoloidalExtent.R_winding = 0.976
        JPoloidalExtent.Z_winding = 0.0

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
            np.testing.assert_allclose(constraint_values, [0.2])
            np.testing.assert_allclose(constraint_grads[0], [0.4, 0.6])
            np.testing.assert_allclose(multipliers, [0.0])
            self.assertAlmostEqual(penalty, 1.0)
            return {
                "total": 0.2,
                "grad": np.array([0.4, 0.6]),
                "stationarity_norm": 0.0,
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
            multipliers=np.array([0.0]),
            penalty=1.0,
            objective_optimizable=SimpleNamespace(),
            curves=(curve,),
            curve_curve_min_distance=0.05,
            outer_surface=_FakeSurfaceWithGradient(gamma_points=[[[0.0, 0.0, 0.0]]]),
            curve_surface_min_distance=0.02,
            banana_curve=curve,
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=("poloidal_extent",),
            curve_curve_constraint_fn=_constant_constraint_result(
                -0.1,
                [0.0, 0.0],
                0.0,
            ),
            curve_curve_constraint_with_hard_signal_fn=(
                _constant_constraint_result_with_hard_signal(
                    -0.1,
                    [0.0, 0.0],
                    0.0,
                    -0.1,
                    0.0,
                )
            ),
            curve_surface_constraint_fn=_constant_constraint_result(
                -0.2,
                [0.0, 0.0],
                0.0,
            ),
            curve_surface_constraint_with_hard_signal_fn=(
                _constant_constraint_result_with_hard_signal(
                    -0.2,
                    [0.0, 0.0],
                    0.0,
                    -0.2,
                    0.0,
                )
            ),
            curvature_constraint_fn=_constant_constraint_result(
                -0.3,
                [0.0, 0.0],
                0.0,
            ),
            hard_surrogate_diagnostics=True,
            augmented_inequality_objective_fn=fake_augmented,
            activity_tolerances_fn=lambda *_args, **_kwargs: (
                1.0e-3,
                1.0e-3,
                1.0e-3,
                1.0e-3,
            ),
            JPoloidalExtent=JPoloidalExtent,
            poloidal_extent_threshold=1.0,
            poloidal_extent_smoothing=0.05,
            poloidal_extent_constraint_fn=_constant_constraint_result(
                0.2,
                [0.4, 0.6],
                0.2,
            ),
            poloidal_extent_constraint_with_hard_signal_fn=(
                _constant_constraint_result_with_hard_signal(
                    0.2,
                    [0.4, 0.6],
                    0.2,
                    -0.1,
                    0.0,
                )
            ),
        )

        self.assertEqual(result["constraint_blocks"], ["geometry"])
        self.assertEqual(result["dual_update_value_kinds"], ["hard"])
        self.assertEqual(result["feasibility_value_kinds"], ["hard"])
        np.testing.assert_allclose(result["surrogate_signed_constraint_values"], [0.2])
        np.testing.assert_allclose(result["hard_signed_constraint_values"], [-0.1])
        np.testing.assert_allclose(result["hard_violation_values"], [0.0])
        np.testing.assert_allclose(result["dual_update_values"], [-0.1])
        np.testing.assert_allclose(result["feasibility_values"], [0.0])

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
            epsilons=TAYLOR_TEST_EPSILONS,
        )

        self.assertTrue(result["passed"], result)
        self.assertGreaterEqual(result["direction_count"], 4)

    def test_evaluate_alm_objective_supports_independent_banana_current_constraints(
        self,
    ):
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
            curve_surface_constraint_fn=lambda *_args: (
                -0.2,
                np.array([0.0, 0.0]),
                0.0,
            ),
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
            [1.0e-3, 1.0e-3],
        )
        self.assertEqual(result["objective_value_kinds"], ["hard", "hard"])
        self.assertEqual(result["dual_update_value_kinds"], ["hard", "hard"])

    def test_independent_banana_current_alm_payload_feeds_hardware_snapshot(self):
        geometry_module = _load_module(
            SINGLE_STAGE_GEOMETRY_PATH,
            "banana_single_stage_geometry",
        )
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        current_a = _FakeCurrentObjective(17000.0, [2.0, -1.0])
        current_b = _FakeCurrentObjective(-15500.0, [0.5, 1.5])

        objective_eval = self.module.evaluate_alm_objective(
            np.array([1.0]),
            [_FakeAlgebraicObjective(2.0, [0.2, 0.0])],
            [_FakeAlgebraicObjective(0.0, [0.0, 0.3])],
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
            multipliers=np.array([0.1, 0.2, 0.3]),
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
                "qs_error",
            ),
            curve_curve_constraint_fn=lambda *_args: (-0.1, np.array([0.0, 0.0]), 0.0),
            curve_surface_constraint_fn=lambda *_args: (
                -0.2,
                np.array([0.0, 0.0]),
                0.0,
            ),
            curvature_constraint_fn=lambda *_args: (-0.3, np.array([0.0, 0.0]), 0.0),
            banana_currents=(current_a, current_b),
            banana_current_threshold=16000.0,
            alm_formulation="thresholded_physics",
            qs_threshold=1.0,
            boozer_threshold=1.0,
            iota_penalty_threshold=1.0,
            length_penalty_threshold=1.0,
            augmented_inequality_objective_fn=lambda *_args: {
                "total": 5.5,
                "grad": np.array([0.4, 0.6]),
                "stationarity_norm": 0.25,
            },
            include_diagnostics=False,
        )

        snapshot = geometry_module.evaluate_single_stage_search_hardware_snapshot(
            objective_eval,
            cc_dist=0.05,
            cs_dist=0.02,
            ss_dist=0.04,
            curvature_threshold=40.0,
            banana_current_max_A=16000.0,
        )

        self.assertEqual(
            objective_eval["constraint_names"],
            [
                "banana_current_0_upper_bound",
                "banana_current_1_upper_bound",
                "qs_error",
            ],
        )
        self.assertEqual(
            objective_eval["constraint_blocks"],
            ["current", "current", "physics"],
        )
        np.testing.assert_allclose(
            objective_eval["raw_dual_update_values"],
            [1000.0, -500.0, 1.0],
        )
        self.assertAlmostEqual(snapshot["banana_current_A"], 17000.0)
        self.assertEqual(
            set(snapshot["search_hardware_status"]["constraints"]),
            {"banana_current"},
        )
        self.assertNotIn("qs_error", snapshot["search_hardware_status"]["constraints"])

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
            curve_surface_constraint_fn=lambda *_args: (
                -0.2,
                np.array([0.0, 0.0]),
                0.0,
            ),
            curvature_constraint_fn=lambda *_args: (-0.3, np.array([0.0, 0.0]), 0.0),
            banana_current=banana_current,
            banana_current_threshold=20000.0,
            include_diagnostics=True,
        )

        self.assertEqual(result["banana_current_upper_bound_threshold"], 20000.0)
        np.testing.assert_allclose(result["constraint_scales"], [20000.0])
        np.testing.assert_allclose(result["constraint_activity_tolerances"], [1.0e-3])
        np.testing.assert_allclose(result["raw_constraint_activity_tolerances"], [20.0])

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
        self.assertAlmostEqual(result["physics_terms_total"], 25.0)
        np.testing.assert_allclose(result["grad"], [0.0, 0.0, 0.0, 0.0])

    def test_evaluate_base_objective_weighted_sum_includes_residue_objective(self):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()
        residue = _FakeResidueObjective(
            0.7,
            [0.4, -0.2],
            [0.4, -0.2, 0.0, 0.0],
        )

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
            JResidueObjective=residue,
        )

        self.assertAlmostEqual(result["total"], 25.7)
        self.assertAlmostEqual(result["physics_total"], 25.7)
        self.assertAlmostEqual(result["physics_terms_total"], 25.0)
        self.assertAlmostEqual(result["J_residue_objective"], 0.7)
        np.testing.assert_allclose(result["grad"], [5.0, 2.6, 0.0, 0.0])

    def test_evaluate_base_objective_thresholded_physics_optimizes_residue_objective(
        self,
    ):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()
        residue = _FakeResidueObjective(
            0.7,
            [0.4, -0.2],
            [0.4, -0.2, 0.0, 0.0],
        )

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
            JResidueObjective=residue,
        )

        self.assertAlmostEqual(result["total"], 0.7)
        self.assertAlmostEqual(result["physics_total"], 0.7)
        self.assertAlmostEqual(result["physics_terms_total"], 25.0)
        np.testing.assert_allclose(result["grad"], [0.4, -0.2, 0.0, 0.0])

    def test_evaluate_alm_objective_base_total_includes_residue_objective(self):
        objective, nonqs, brs, jiota, jlength = self._make_projected_base_terms()
        residue = _FakeResidueObjective(
            0.7,
            [0.4, -0.2],
            [0.4, -0.2, 0.0, 0.0],
        )
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])

        def fake_augmented(
            base_value,
            base_grad,
            constraint_values,
            constraint_grads,
            multipliers,
            penalty,
        ):
            self.assertAlmostEqual(base_value, 25.7)
            np.testing.assert_allclose(base_grad, [5.0, 2.6, 0.0, 0.0])
            np.testing.assert_allclose(constraint_values, [-2.0])
            np.testing.assert_allclose(constraint_grads, [[20.0, 0.0, 0.0, 0.0]])
            np.testing.assert_allclose(multipliers, [0.1])
            self.assertAlmostEqual(penalty, 4.0)
            return {
                "total": 26.0,
                "grad": np.array([5.1, 2.7, 0.0, 0.0]),
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
            JCurveCurve=zero,
            JCurveSurface=zero,
            JCurvature=zero,
            multipliers=np.array([0.1]),
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
            constraint_names=("coil_coil_spacing",),
            curve_curve_constraint_fn=lambda *_args: (
                -0.1,
                np.array([1.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            curve_surface_constraint_fn=lambda *_args: (
                0.0,
                np.array([0.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            curvature_constraint_fn=lambda *_args: (
                0.0,
                np.array([0.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            augmented_inequality_objective_fn=fake_augmented,
            activity_tolerances_fn=lambda ds, cs, include_surface_stack=False: np.array(
                [ds, ds, cs],
                dtype=float,
            ),
            JResidueObjective=residue,
        )

        self.assertAlmostEqual(result["physics_total"], 25.7)
        self.assertAlmostEqual(result["physics_terms_total"], 25.0)
        self.assertAlmostEqual(result["base_total"], 25.7)
        self.assertAlmostEqual(result["total"], 26.0)

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
            JCurveCurve=_FakeAlgebraicObjective(
                0.6,
                [0.3, 0.4],
                [0.3, 0.4, 0.0, 0.0],
            ),
            JCurveSurface=_FakeAlgebraicObjective(
                0.7,
                [0.5, 0.6],
                [0.5, 0.6, 0.0, 0.0],
            ),
            JCurvature=_FakeAlgebraicObjective(
                0.8,
                [0.7, 0.8],
                [0.7, 0.8, 0.0, 0.0],
            ),
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
            activity_tolerances_fn=lambda ds, cs, include_surface_stack=False: np.array(
                [ds * 4.0, ds * 4.0, cs * 4.0],
                dtype=float,
            ),
        )

        np.testing.assert_allclose(result["grad"], [9.0, -2.0, 0.0, 0.0])
        np.testing.assert_allclose(result["dJ_cc"], [0.3, 0.4, 0.0, 0.0])
        np.testing.assert_allclose(result["dJ_cs"], [0.5, 0.6, 0.0, 0.0])
        np.testing.assert_allclose(result["dJ_curvature"], [0.7, 0.8, 0.0, 0.0])

    def test_evaluate_alm_objective_enforces_coil_length_upper_bound_constraint(self):
        objective = SimpleNamespace(name="full")
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        raw_length = _FakeAlgebraicObjective(
            0.8,
            [0.2, -0.1],
            [0.2, -0.1, 0.0, 0.0],
        )

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
            multipliers=np.array([0.0]),
            penalty=1.0,
            objective_optimizable=objective,
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface="outer",
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=("coil_length_upper_bound",),
            curve_curve_constraint_fn=lambda *_args: (
                0.0,
                np.array([0.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            curve_surface_constraint_fn=lambda *_args: (
                0.0,
                np.array([0.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            curvature_constraint_fn=lambda *_args: (
                0.0,
                np.array([0.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            activity_tolerances_fn=lambda ds, cs, include_surface_stack=False: np.array(
                [ds, ds, cs],
                dtype=float,
            ),
            coil_length_objective=raw_length,
            coil_length_threshold=0.75,
        )

        self.assertEqual(result["constraint_names"], ["coil_length_upper_bound"])
        self.assertAlmostEqual(result["coil_length_upper_bound_threshold"], 0.75)
        np.testing.assert_allclose(result["raw_constraint_values"], [0.05])
        np.testing.assert_allclose(result["raw_feasibility_values"], [0.05])
        np.testing.assert_allclose(
            result["raw_constraint_grads"][0],
            [0.2, -0.1, 0.0, 0.0],
        )

    def test_evaluate_alm_objective_enforces_coil_length_lower_bound_constraint(self):
        objective = SimpleNamespace(name="full")
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
        raw_length = _FakeAlgebraicObjective(
            0.8,
            [0.2, -0.1],
            [0.2, -0.1, 0.0, 0.0],
        )

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
            multipliers=np.array([0.0]),
            penalty=1.0,
            objective_optimizable=objective,
            curves=["curve_a"],
            curve_curve_min_distance=0.05,
            outer_surface="outer",
            curve_surface_min_distance=0.02,
            banana_curve="banana",
            curvature_threshold=40.0,
            distance_smoothing=0.01,
            curvature_smoothing=0.05,
            constraint_names=("coil_length_min",),
            curve_curve_constraint_fn=lambda *_args: (
                0.0,
                np.array([0.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            curve_surface_constraint_fn=lambda *_args: (
                0.0,
                np.array([0.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            curvature_constraint_fn=lambda *_args: (
                0.0,
                np.array([0.0, 0.0, 0.0, 0.0]),
                0.0,
            ),
            activity_tolerances_fn=lambda ds, cs, include_surface_stack=False: np.array(
                [ds, ds, cs],
                dtype=float,
            ),
            coil_length_objective=raw_length,
            coil_length_min_threshold=0.95,
        )

        self.assertEqual(result["constraint_names"], ["coil_length_min"])
        self.assertAlmostEqual(result["coil_length_min_threshold"], 0.95)
        np.testing.assert_allclose(result["raw_constraint_values"], [0.15])
        np.testing.assert_allclose(result["raw_feasibility_values"], [0.15])
        np.testing.assert_allclose(
            result["raw_constraint_grads"][0],
            [-0.2, 0.1, -0.0, -0.0],
        )

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
            activity_tolerances_fn=lambda ds, cs, include_surface_stack=False: np.array(
                [ds * 4.0, ds * 4.0, cs * 4.0],
                dtype=float,
            ),
            alm_formulation="thresholded_physics",
            qs_threshold=1.0,
            boozer_threshold=1.0,
            iota_penalty_threshold=0.5,
            length_penalty_threshold=1.0e-12,
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
        # The warm-start iotas are threaded through to evaluate_surface_stack as the iota-collapse
        # reference (defense-in-depth reject). The fake surfaces' run_code(iota, G) does not accept
        # the early-exit kwargs, so the dispatch guard correctly omits them (calls unchanged above).
        evaluate_mock.assert_called_once_with(
            surface_data,
            vessel_surface="VV",
            surface_gap_threshold=0.05,
            enforce_nesting=False,
            reference_iotas=[0.11, 0.22],
        )
        self.assertEqual(result, {"success": True})

    def test_evaluate_surface_stack_checks_adjacent_nesting_pairs(self):
        class _FakeSurface:
            def __init__(self, volume):
                self._volume = float(volume)

            def volume(self):
                return self._volume

            def is_self_intersecting(self):
                return False

            def cross_section(self, *_args, **_kwargs):
                return np.zeros((4, 3))

        def _entry(name, volume, iota):
            return {
                "name": name,
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(volume),
                    res={"iota": iota, "success": True},
                ),
            }

        surface_data = [
            _entry("inner0", 1.0, 0.1),
            _entry("inner1", 2.0, 0.2),
            _entry("outer", 3.0, 0.3),
        ]

        with (
            mock.patch.object(
                self.module,
                "surface_pointcloud_gap",
                return_value=0.1,
            ),
            mock.patch.object(
                self.module,
                "cross_sections_are_nested",
                side_effect=[(True, []), (False, [0.25])],
            ) as nested_mock,
        ):
            result = self.module.evaluate_surface_stack(
                surface_data,
                surface_gap_threshold=0.0,
                enforce_nesting=True,
            )

        self.assertFalse(result["success"])
        self.assertFalse(result["nesting_ok"])
        self.assertEqual(result["bad_nesting_phis"], [0.25])
        self.assertEqual(
            result["bad_nesting_pairs"],
            [
                {
                    "inner_index": 1,
                    "outer_index": 2,
                    "inner_name": "inner1",
                    "outer_name": "outer",
                    "bad_phis": [0.25],
                }
            ],
        )
        self.assertEqual(nested_mock.call_count, 2)

    def test_evaluate_surface_stack_marks_goes_back_nesting_pair_invalid(self):
        class _FakeSurface:
            def __init__(self, volume):
                self._volume = float(volume)

            def volume(self):
                return self._volume

            def is_self_intersecting(self):
                return False

            def cross_section(self, *_args, **_kwargs):
                return np.zeros((4, 3))

        def _entry(name, volume, iota):
            return {
                "name": name,
                "boozer_surface": SimpleNamespace(
                    surface=_FakeSurface(volume),
                    res={"iota": iota, "success": True},
                ),
            }

        surface_data = [
            _entry("inner", 1.0, 0.1),
            _entry("outer", 2.0, 0.2),
        ]

        with (
            mock.patch.object(
                self.module,
                "surface_pointcloud_gap",
                return_value=0.1,
            ),
            mock.patch.object(
                self.module,
                "cross_sections_are_nested",
                side_effect=RuntimeError("surface 'goes back' on itself"),
            ),
        ):
            result = self.module.evaluate_surface_stack(
                surface_data,
                surface_gap_threshold=0.0,
                enforce_nesting=True,
            )

        self.assertFalse(result["success"])
        self.assertFalse(result["nesting_ok"])
        self.assertEqual(result["bad_nesting_phis"], [])
        self.assertEqual(
            result["bad_nesting_pairs"],
            [
                {
                    "inner_index": 0,
                    "outer_index": 1,
                    "inner_name": "inner",
                    "outer_name": "outer",
                    "bad_phis": [],
                }
            ],
        )

    def test_evaluate_surface_stack_reraises_unexpected_nesting_error(self):
        surface_data = [
            {
                "name": "inner",
                "boozer_surface": SimpleNamespace(
                    surface=SimpleNamespace(
                        volume=lambda: 1.0,
                        is_self_intersecting=lambda: False,
                        cross_section=lambda *_args, **_kwargs: np.zeros((4, 3)),
                    ),
                    res={"iota": 0.1, "success": True},
                ),
            },
            {
                "name": "outer",
                "boozer_surface": SimpleNamespace(
                    surface=SimpleNamespace(
                        volume=lambda: 2.0,
                        is_self_intersecting=lambda: False,
                        cross_section=lambda *_args, **_kwargs: np.zeros((4, 3)),
                    ),
                    res={"iota": 0.2, "success": True},
                ),
            },
        ]

        with (
            mock.patch.object(
                self.module,
                "surface_pointcloud_gap",
                return_value=0.1,
            ),
            mock.patch.object(
                self.module,
                "cross_sections_are_nested",
                side_effect=RuntimeError("unexpected cross-section bug"),
            ),
            self.assertRaisesRegex(RuntimeError, "unexpected cross-section bug"),
        ):
            self.module.evaluate_surface_stack(
                surface_data,
                surface_gap_threshold=0.0,
                enforce_nesting=True,
            )

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
            surface_status={"outer_vessel_gap": 0.01},
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
        self.assertEqual(len(result["search_hardware_status"]["violations"]), 2)
        self.assertFalse(result["artifact_hardware_status"]["success"])
        self.assertIn(
            "missing required hardware constraint metric width_min",
            result["artifact_hardware_status"]["violations"],
        )
        self.assertIn(
            "missing required hardware constraint metric width_max",
            result["artifact_hardware_status"]["violations"],
        )
        self.assertIn(
            "missing required hardware constraint metric self_intersect",
            result["artifact_hardware_status"]["violations"],
        )

    def test_evaluate_single_stage_hardware_snapshot_records_artifact_width_and_self_intersect(
        self,
    ):
        result = self.module.evaluate_single_stage_hardware_snapshot(
            curve_curve_distance_obj=SimpleNamespace(shortest_distance=lambda: 0.05),
            cc_dist=0.05,
            curve_surface_distance_obj=SimpleNamespace(shortest_distance=lambda: 0.02),
            cs_dist=0.02,
            surface_status={"outer_vessel_gap": 0.04},
            ss_dist=0.04,
            banana_curve=SimpleNamespace(kappa=lambda: np.array([39.0, 40.0])),
            curvature_threshold=40.0,
            outer_surface=SimpleNamespace(
                major_radius=_in_bounds_lcfs_major_radius_m,
                minor_radius=_in_bounds_lcfs_minor_radius_m,
            ),
            coil_length=1.9,
            length_target=1.9,
            poloidal_extent_rad=0.7,
            poloidal_extent_threshold_rad=0.8,
            tf_current_A=-8.0e4,
            tf_current_limit_A=8.0e4,
            banana_current_A=-1.6e4,
            banana_current_max_A=1.6e4,
            coil_width=0.12,
            width_min_threshold=0.05,
            width_max_threshold=0.17,
            self_intersect_penalty=0.0,
            self_intersect_threshold=0.0,
        )

        artifact_status = result["artifact_hardware_status"]

        self.assertTrue(artifact_status["success"])
        self.assertEqual(artifact_status["violations"], [])
        self.assertEqual(
            artifact_status["constraints"]["width_min"]["value"],
            0.12,
        )
        self.assertEqual(
            artifact_status["constraints"]["width_max"]["value"],
            0.12,
        )
        self.assertEqual(
            artifact_status["constraints"]["self_intersect"]["value"],
            0.0,
        )
        self.assertEqual(result["coil_width"], 0.12)
        self.assertEqual(result["self_intersect_penalty"], 0.0)

    def test_evaluate_single_stage_hardware_snapshot_keeps_top_level_constraints_in_search_role(
        self,
    ):
        result = self.module.evaluate_single_stage_hardware_snapshot(
            curve_curve_distance_obj=SimpleNamespace(shortest_distance=lambda: 0.05),
            cc_dist=0.05,
            curve_surface_distance_obj=SimpleNamespace(shortest_distance=lambda: 0.02),
            cs_dist=0.02,
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
                    "max_curvature",
                    "banana_current_upper_bound",
                ],
                "dual_update_values": np.array([0.01, -0.003, 0.5, 1000.0]),
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

    def test_evaluate_single_stage_search_hardware_snapshot_uses_raw_alm_residuals(
        self,
    ):
        result = self.module.evaluate_single_stage_search_hardware_snapshot(
            {
                "constraint_names": [
                    "coil_coil_spacing",
                    "coil_surface_spacing",
                    "max_curvature",
                    "banana_current_upper_bound",
                ],
                "dual_update_values": np.array(
                    [1.024514, 21.270667, -0.00018, -0.03125]
                ),
                "raw_dual_update_values": np.array([-0.002, -0.074, -0.018, -500.0]),
                "search_hardware_constraint_payload_kind": "signed_residual",
            },
            cc_dist=0.05,
            cs_dist=0.015,
            ss_dist=0.04,
            curvature_threshold=100.0,
            banana_current_max_A=1.6e4,
        )

        self.assertAlmostEqual(result["curve_curve_min_dist"], 0.052)
        self.assertAlmostEqual(result["curve_surface_min_dist"], 0.089)
        self.assertAlmostEqual(result["max_curvature"], 99.982)
        self.assertAlmostEqual(result["banana_current_A"], 1.55e4)
        self.assertTrue(result["search_hardware_status"]["success"])

    def test_evaluate_single_stage_search_hardware_snapshot_uses_independent_banana_current_residuals(
        self,
    ):
        result = self.module.evaluate_single_stage_search_hardware_snapshot(
            {
                "constraint_names": [
                    "banana_current_0_upper_bound",
                    "banana_current_1_upper_bound",
                    "qs_error",
                ],
                "dual_update_values": np.array([500.0, 2000.0, 0.0]),
                "raw_dual_update_values": np.array([1000.0, -500.0, 0.0]),
                "constraint_blocks": ["current", "current", "physics"],
                "search_hardware_constraint_payload_kind": "signed_residual",
            },
            cc_dist=0.05,
            cs_dist=0.02,
            ss_dist=0.04,
            curvature_threshold=40.0,
            banana_current_max_A=1.6e4,
        )

        self.assertAlmostEqual(result["banana_current_A"], 1.7e4)
        self.assertFalse(result["search_hardware_status"]["success"])
        self.assertEqual(
            set(result["search_hardware_status"]["constraints"]),
            {"banana_current"},
        )
        self.assertNotIn("qs_error", result["search_hardware_status"]["constraints"])

    def test_evaluate_single_stage_search_hardware_snapshot_rejects_unknown_nonphysics_alm_name(
        self,
    ):
        with self.assertRaisesRegex(KeyError, "banana_current_o_upper_bound"):
            self.module.evaluate_single_stage_search_hardware_snapshot(
                {
                    "constraint_names": ["banana_current_o_upper_bound"],
                    "dual_update_values": np.array([0.0]),
                    "raw_dual_update_values": np.array([0.0]),
                    "constraint_blocks": ["current"],
                    "search_hardware_constraint_payload_kind": "signed_residual",
                },
                cc_dist=0.05,
                cs_dist=0.02,
                ss_dist=0.04,
                curvature_threshold=40.0,
                banana_current_max_A=1.6e4,
            )

    def test_evaluate_single_stage_search_hardware_snapshot_uses_lcfs_alm_residuals(
        self,
    ):
        result = self.module.evaluate_single_stage_search_hardware_snapshot(
            {
                "constraint_names": [
                    "coil_coil_spacing",
                    "coil_surface_spacing",
                    "max_curvature",
                    "lcfs_major_radius",
                    "lcfs_minor_radius",
                ],
                "dual_update_values": np.array([0.0, 0.0, 0.0, 1.0e-4, 2.0e-4]),
                "raw_dual_update_values": np.array(
                    [-0.002, -0.074, -0.018, 0.006, 0.003]
                ),
                "search_hardware_constraint_payload_kind": "signed_residual",
            },
            cc_dist=0.05,
            cs_dist=0.015,
            ss_dist=0.04,
            curvature_threshold=100.0,
        )

        self.assertFalse(result["search_hardware_status"]["success"])
        hardware_contracts = _load_module(
            HARDWARE_CONTRACTS_PATH,
            "banana_hw_contracts",
        )
        self.assertAlmostEqual(
            result["lcfs_major_radius_m"],
            hardware_contracts.TARGET_LCFS_MAX_MAJOR_RADIUS_M + 0.006,
        )
        self.assertAlmostEqual(
            result["lcfs_minor_radius_m"],
            hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M + 0.003,
        )
        self.assertIn(
            "lcfs_major_radius",
            result["search_hardware_status"]["constraints"],
        )
        self.assertIn(
            "lcfs_minor_radius",
            result["search_hardware_status"]["constraints"],
        )
        self.assertTrue(
            any("lcfs_major_radius" in violation for violation in result["violations"])
        )
        self.assertTrue(
            any("lcfs_minor_radius" in violation for violation in result["violations"])
        )

    def test_evaluate_single_stage_search_hardware_snapshot_uses_lcfs_edge_residuals(
        self,
    ):
        hardware_contracts = _load_module(
            HARDWARE_CONTRACTS_PATH,
            "banana_hw_contracts",
        )
        result = self.module.evaluate_single_stage_search_hardware_snapshot(
            {
                "constraint_names": [
                    "coil_coil_spacing",
                    "coil_surface_spacing",
                    "max_curvature",
                    "lcfs_outboard_edge",
                    "lcfs_inboard_edge",
                    "lcfs_minor_radius",
                ],
                "dual_update_values": np.array(
                    [0.0, 0.0, 0.0, 1.0e-4, 2.0e-4, 3.0e-4]
                ),
                "raw_dual_update_values": np.array(
                    [-0.002, -0.074, -0.018, 0.002, 0.003, 0.004]
                ),
                "search_hardware_constraint_payload_kind": "signed_residual",
            },
            cc_dist=0.05,
            cs_dist=0.015,
            ss_dist=0.04,
            curvature_threshold=100.0,
        )

        self.assertFalse(result["search_hardware_status"]["success"])
        self.assertAlmostEqual(
            result["lcfs_outboard_edge_m"],
            hardware_contracts.LCFS_OUTBOARD_RADIUS_MAX_M + 0.002,
        )
        self.assertAlmostEqual(
            result["lcfs_inboard_edge_m"],
            hardware_contracts.LCFS_INBOARD_RADIUS_MIN_M - 0.003,
        )
        self.assertAlmostEqual(
            result["lcfs_minor_radius_m"],
            hardware_contracts.TARGET_LCFS_MAX_MINOR_RADIUS_M + 0.004,
        )
        self.assertIn(
            "lcfs_outboard_edge",
            result["search_hardware_status"]["constraints"],
        )
        self.assertIn(
            "lcfs_inboard_edge",
            result["search_hardware_status"]["constraints"],
        )
        self.assertTrue(
            any("lcfs_outboard_edge" in violation for violation in result["violations"])
        )
        self.assertTrue(
            any("lcfs_inboard_edge" in violation for violation in result["violations"])
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

    def test_kdtree_pairwise_selection_returns_empty_arrays_when_no_pairs_match(self):
        left = np.array([[0.0, 0.0, 0.0]], dtype=float)
        right = np.array([[2.0, 0.0, 0.0]], dtype=float)

        rows, cols, diffs, distances = self.module.select_pairwise_near_min(
            left,
            right,
            threshold=0.5,
        )

        self.assertEqual(rows.tolist(), [])
        self.assertEqual(cols.tolist(), [])
        self.assertEqual(diffs.shape, (0, 3))
        self.assertEqual(distances.tolist(), [])

    def test_kdtree_pairwise_selection_reuses_supplied_trees(self):
        left = np.array([[0.0, 0.0, 0.0]], dtype=float)
        right = np.array([[0.25, 0.0, 0.0]], dtype=float)
        left_tree = self.module.point_tree(left)
        right_tree = self.module.point_tree(right)

        with mock.patch.object(
            self.module,
            "point_tree",
            side_effect=AssertionError("selection should reuse supplied trees"),
        ):
            rows, cols, diffs, distances = self.module.select_pairwise_near_min(
                left,
                right,
                threshold=0.5,
                left_tree=left_tree,
                right_tree=right_tree,
            )

        self.assertEqual(rows.tolist(), [0])
        self.assertEqual(cols.tolist(), [0])
        np.testing.assert_allclose(diffs, [[-0.25, 0.0, 0.0]])
        np.testing.assert_allclose(distances, [0.25])

    def test_surface_tree_cache_reuses_until_surface_dofs_change(self):
        class Surface:
            def __init__(self):
                self.x = np.array([0.0], dtype=float)
                self.gamma_calls = 0

            def gamma(self):
                self.gamma_calls += 1
                offset = float(self.x[0])
                return np.array(
                    [[[offset, 0.0, 0.0], [offset + 1.0, 0.0, 0.0]]],
                    dtype=float,
                )

        surface = Surface()
        points, tree, shape = self.module.surface_points_tree_shape(surface)
        same_points, same_tree, same_shape = self.module.surface_points_tree_shape(
            surface
        )

        self.assertEqual(surface.gamma_calls, 1)
        self.assertIs(same_points, points)
        self.assertIs(same_tree, tree)
        self.assertEqual(same_shape, shape)

        surface.x = np.array([2.0], dtype=float)
        updated_points, updated_tree, updated_shape = (
            self.module.surface_points_tree_shape(surface)
        )

        self.assertEqual(surface.gamma_calls, 2)
        self.assertIsNot(updated_points, points)
        self.assertIsNot(updated_tree, tree)
        self.assertEqual(updated_shape, shape)
        np.testing.assert_allclose(updated_points[:, 0], [2.0, 3.0])


class SingleStageConstraintModuleTests(_ModuleTestCase):
    MODULE_PATH = SINGLE_STAGE_CONSTRAINTS_PATH
    MODULE_PREFIX = "banana_single_stage_constraints"

    def test_single_stage_constraint_activity_tolerances_match_selection_windows(self):
        tolerances = self.module.single_stage_constraint_activity_tolerances(
            0.005,
            0.05,
        )
        np.testing.assert_allclose(tolerances, [0.02, 0.02, 0.2])

    def test_softmin_selection_window_uses_shared_truncation_factor(self):
        self.assertAlmostEqual(self.module.softmin_selection_window(0.25), 1.0)

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

        signed_value, grad, violation, hard_signed_value, hard_violation = (
            self.module.smooth_min_curve_curve_signed_constraint(
                [curve],
                minimum_distance=0.05,
                temperature=0.01,
                objective_optimizable=objective,
                include_hard_signal=True,
            )
        )

        self.assertAlmostEqual(signed_value, -0.05)
        self.assertEqual(violation, 0.0)
        self.assertAlmostEqual(hard_signed_value, -0.05)
        self.assertEqual(hard_violation, 0.0)
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


class HardwareConstraintSchemaModuleTests(unittest.TestCase):
    # Direct import (not dynamic _load_module) because hardware_constraint_schema
    # itself defines `ALMConstraintMetadata` as a dataclass at module scope; the
    # dataclass decorator needs `cls.__module__` registered in `sys.modules`,
    # which the test-only loader does not provide. The transitive imports made
    # by sibling tests succeed because they load files that *consume* the
    # dataclass under its real `banana_opt.hardware_constraint_schema` name.
    def setUp(self):
        sys.path.insert(0, str(EXAMPLES_ROOT))
        try:
            import banana_opt.hardware_constraint_schema as schema_module
        finally:
            sys.path[:] = [p for p in sys.path if p != str(EXAMPLES_ROOT)]
        self.module = schema_module

    def test_resolve_alm_scale_with_provenance_floors_when_below_floor(self):
        # M7 SSOT helper: floor activates → scale clamps and source gains the
        # `:floored` suffix; flag is True.
        scale, floor_applied, source = self.module.resolve_alm_scale_with_provenance(
            5.0e-13, 1.0e-12, "threshold:demo"
        )
        self.assertEqual(scale, 1.0e-12)
        self.assertTrue(floor_applied)
        self.assertEqual(source, "threshold:demo:floored")

    def test_resolve_alm_scale_with_provenance_passes_through_when_above_floor(self):
        # M7 SSOT helper: when raw >= floor the scale stays equal to raw and
        # the source string is unchanged.
        scale, floor_applied, source = self.module.resolve_alm_scale_with_provenance(
            2.5, 1.0e-12, "threshold:demo"
        )
        self.assertEqual(scale, 2.5)
        self.assertFalse(floor_applied)
        self.assertEqual(source, "threshold:demo")

    def test_lcfs_artifact_upper_bound_tolerates_roundoff(self):
        spec = self.module.get_hardware_constraint_spec("lcfs_major_radius")
        roundoff_major_radius = self.module.TARGET_LCFS_MAX_MAJOR_RADIUS_M + 3.9e-15

        self.assertEqual(
            self.module.hardware_constraint_violation(spec, roundoff_major_radius),
            0.0,
        )

    def test_lcfs_edge_constraint_specs_use_envelope_bounds(self):
        outboard_spec = self.module.get_hardware_constraint_spec("lcfs_outboard_edge")
        inboard_spec = self.module.get_hardware_constraint_spec("lcfs_inboard_edge")

        self.assertEqual(outboard_spec.kind, "upper_bound")
        self.assertEqual(inboard_spec.kind, "lower_bound")
        self.assertEqual(outboard_spec.threshold, self.module.LCFS_OUTBOARD_RADIUS_MAX_M)
        self.assertEqual(inboard_spec.threshold, self.module.LCFS_INBOARD_RADIUS_MIN_M)
        self.assertEqual(self.module.hardware_constraint_alm_block("lcfs_outboard_edge"), "surface")
        self.assertEqual(self.module.hardware_constraint_alm_block("lcfs_inboard_edge"), "surface")

    def test_lcfs_artifact_payload_writes_ok_for_roundoff(self):
        roundoff_major_radius = self.module.TARGET_LCFS_MAX_MAJOR_RADIUS_M + 3.9e-15
        lcfs_constraint_names = {"lcfs_major_radius", "lcfs_minor_radius"}
        status = self.module.build_hardware_constraint_status(
            {
                "lcfs_major_radius": roundoff_major_radius,
                "lcfs_minor_radius": self.module.TARGET_LCFS_MAX_MINOR_RADIUS_M,
            },
            applies_to="artifact",
            names=lcfs_constraint_names,
            require_values=True,
        )
        payload = self.module.build_hardware_constraint_artifact_payload_fields(
            {
                "lcfs_major_radius_m": roundoff_major_radius,
                "lcfs_minor_radius_m": self.module.TARGET_LCFS_MAX_MINOR_RADIUS_M,
                "artifact_hardware_status": status,
            },
            names=lcfs_constraint_names,
        )

        self.assertTrue(status["success"])
        self.assertEqual(status["violations"], [])
        self.assertIs(payload["HARDWARE_CONSTRAINTS_OK"], True)
        self.assertEqual(payload["HARDWARE_CONSTRAINT_VIOLATIONS"], [])

    def test_lcfs_artifact_upper_bound_rejects_real_excess(self):
        spec = self.module.get_hardware_constraint_spec("lcfs_major_radius")
        real_excess_major_radius = self.module.TARGET_LCFS_MAX_MAJOR_RADIUS_M + 1.0e-11

        self.assertGreater(
            self.module.hardware_constraint_violation(spec, real_excess_major_radius),
            0.0,
        )

    def test_hardware_constraint_alm_metadata_records_floor_for_subnormal_threshold(
        self,
    ):
        # M7 hardware-schema mirror: the threshold-driven path records floor
        # provenance when the override falls below the physical scale floor.
        metadata = self.module.hardware_constraint_alm_metadata(
            "coil_coil_spacing",
            threshold_overrides={"coil_coil_spacing": 1.0e-300},
        )
        self.assertEqual(metadata.scale, self.module.ALM_PHYSICAL_SCALE_FLOOR)
        self.assertTrue(metadata.scale_floor_applied)
        self.assertEqual(metadata.source, "threshold:coil_coil_spacing:floored")

    def test_hardware_constraint_alm_metadata_does_not_record_floor_for_healthy_threshold(
        self,
    ):
        # M7 hardware-schema mirror: the healthy-threshold path leaves
        # `scale_floor_applied=False` and the source string unsuffixed.
        metadata = self.module.hardware_constraint_alm_metadata(
            "coil_coil_spacing",
            threshold_overrides={"coil_coil_spacing": 0.05},
        )
        self.assertEqual(metadata.scale, 0.05)
        self.assertFalse(metadata.scale_floor_applied)
        self.assertEqual(metadata.source, "threshold:coil_coil_spacing")

    def test_hardware_constraint_alm_metadata_records_floor_when_spec_alm_scale_subnormal(
        self,
    ):
        # M7 hardware-schema mirror: the alm_scale-driven path also records
        # floor provenance and stamps the `schema:` base source.
        spec = self.module.HardwareConstraintSpec(
            name="coil_coil_spacing",
            kind="lower_bound",
            threshold=0.05,
            applies_to=frozenset({"alm"}),
            traversal_policy="allowed",
            alm_scale=1.0e-300,
            alm_block="geometry",
        )
        scale, floor_applied, source = self.module._resolved_alm_scale_with_provenance(
            spec, raw_threshold=0.05
        )
        self.assertEqual(scale, self.module.ALM_PHYSICAL_SCALE_FLOOR)
        self.assertTrue(floor_applied)
        self.assertEqual(source, "schema:coil_coil_spacing.alm_scale:floored")

    def test_hardware_constraint_alm_metadata_does_not_record_floor_when_spec_alm_scale_healthy(
        self,
    ):
        # M7 hardware-schema mirror: healthy spec.alm_scale path keeps
        # `scale_floor_applied=False` and the unsuffixed `schema:` source.
        spec = self.module.HardwareConstraintSpec(
            name="coil_coil_spacing",
            kind="lower_bound",
            threshold=0.05,
            applies_to=frozenset({"alm"}),
            traversal_policy="allowed",
            alm_scale=0.1,
            alm_block="geometry",
        )
        scale, floor_applied, source = self.module._resolved_alm_scale_with_provenance(
            spec, raw_threshold=0.05
        )
        self.assertEqual(scale, 0.1)
        self.assertFalse(floor_applied)
        self.assertEqual(source, "schema:coil_coil_spacing.alm_scale")

    def test_alm_constraint_metadata_default_scale_floor_applied_is_false(self):
        # Backward-compat: existing constructors that omit `scale_floor_applied`
        # default to False so legacy call sites and pickles still load.
        metadata = self.module.ALMConstraintMetadata(
            scale=1.0,
            block="physics",
            activity_tolerance=0.0,
            raw_threshold=1.0,
            source="threshold:legacy_demo",
            objective_value_kind="raw_physics",
            gradient_value_kind="raw_physics",
            dual_update_value_kind="hard",
            feasibility_value_kind="hard",
            certification_value_kind="hard",
        )
        self.assertFalse(metadata.scale_floor_applied)


class _LinearIotaLeaf(Optimizable):
    """Real ``Optimizable`` stub mimicking an ``Iotas`` Boozer-surface term.

    ``J`` is an affine function of its own dofs (``base + slope . x``) so the
    coil-routed Boozer adjoint is replaced by an exact, hand-checkable
    ``Derivative``. This lets the shear term's value, sign, and gradient be
    validated without solving a Boozer surface.
    """

    def __init__(self, base, slope):
        self._base = float(base)
        self._slope = np.asarray(slope, dtype=float)
        Optimizable.__init__(self, x0=np.zeros(self._slope.size))

    def J(self):
        return self._base + float(self._slope @ self.local_full_x)

    @derivative_dec
    def dJ(self):
        return Derivative({self: self._slope.copy()})

    return_fn_map = {"J": J, "dJ": dJ}


class IotaShearShortfallTests(_ModuleTestCase):
    MODULE_PATH = SINGLE_STAGE_OBJECTIVES_PATH
    MODULE_PREFIX = "banana_single_stage_objectives_shear"

    # CW lineage: axis-side iota (0.345) exceeds edge iota (0.139), so
    # spread = edge - axis is NEGATIVE and |spread| = 0.206 is the shear.
    AXIS_IOTA = 0.345
    EDGE_IOTA = 0.139
    SHEAR_TARGET = 0.30  # above |spread| so the one-sided term is active

    def _axis_edge_terms(self):
        axis = _LinearIotaLeaf(self.AXIS_IOTA, [0.0, 0.0])
        edge = _LinearIotaLeaf(self.EDGE_IOTA, [1.0, -2.0])
        return axis, edge

    def test_shear_shortfall_value_matches_one_sided_penalty(self):
        axis, edge = self._axis_edge_terms()
        term = self.module.IotaShearShortfall(axis, edge, self.SHEAR_TARGET)

        spread = self.EDGE_IOTA - self.AXIS_IOTA
        shortfall = min(abs(spread) - self.SHEAR_TARGET, 0.0)
        self.assertAlmostEqual(term.J(), 0.5 * shortfall**2)
        self.assertGreater(term.J(), 0.0)  # below the shear floor -> penalized

    def test_shear_shortfall_is_zero_at_or_above_target(self):
        # |spread| = 0.206 >= target 0.10 -> no shortfall, exactly zero.
        axis, edge = self._axis_edge_terms()
        term = self.module.IotaShearShortfall(axis, edge, 0.10)
        self.assertEqual(term.J(), 0.0)

    def test_shear_shortfall_gradient_matches_finite_difference(self):
        axis, edge = self._axis_edge_terms()
        term = self.module.IotaShearShortfall(axis, edge, self.SHEAR_TARGET)

        analytic = term.dJ()
        x0 = term.x.copy()
        h = 1.0e-6
        fd = np.zeros_like(x0)
        for index in range(x0.size):
            step = np.zeros_like(x0)
            step[index] = h
            term.x = x0 + step
            j_plus = term.J()
            term.x = x0 - step
            j_minus = term.J()
            fd[index] = (j_plus - j_minus) / (2.0 * h)
        term.x = x0
        np.testing.assert_allclose(analytic, fd, atol=1.0e-7)

    def test_shear_shortfall_gradient_rewards_more_shear(self):
        # With spread < 0 (axis > edge), increasing |spread| (more shear) must
        # LOWER the penalty. Give each surface a single +iota dof so the sign of
        # d(penalty)/d(iota) is read directly: raising axis-iota grows the gap
        # (dJ < 0) and raising edge-iota shrinks it (dJ > 0). The concatenated
        # free-dof vector orders axis dofs first, then edge dofs.
        axis = _LinearIotaLeaf(self.AXIS_IOTA, [1.0])
        edge = _LinearIotaLeaf(self.EDGE_IOTA, [1.0])
        term = self.module.IotaShearShortfall(axis, edge, self.SHEAR_TARGET)
        grad = term.dJ()
        self.assertLess(grad[0], 0.0, msg=f"d/d(axis iota) not negative: {grad}")
        self.assertGreater(grad[1], 0.0, msg=f"d/d(edge iota) not positive: {grad}")

    def test_builder_returns_none_for_single_surface(self):
        # Single-surface mode has no axis->edge profile: clean no-op, no crash.
        single = [_LinearIotaLeaf(self.EDGE_IOTA, [1.0])]
        self.assertIsNone(
            self.module.build_single_stage_shear_objective(single, self.SHEAR_TARGET)
        )
        self.assertIsNone(
            self.module.build_single_stage_shear_objective([], self.SHEAR_TARGET)
        )

    def test_builder_uses_innermost_and_outermost_surfaces(self):
        # The builder must pick surface_iota_terms[0] (axis) and [-1] (edge),
        # ignoring any intermediate surfaces, and reproduce the direct term.
        axis, edge = self._axis_edge_terms()
        middle = _LinearIotaLeaf(0.25, [0.5, 0.5])
        built = self.module.build_single_stage_shear_objective(
            [axis, middle, edge], self.SHEAR_TARGET
        )
        direct = self.module.IotaShearShortfall(axis, edge, self.SHEAR_TARGET)
        self.assertIsNotNone(built)
        self.assertAlmostEqual(built.J(), direct.J())
        np.testing.assert_allclose(built.dJ(), direct.dJ())

    @staticmethod
    def _base_total_objective_args():
        # The 15 positional build_total_objective args (JnonQSRatio, RES_WEIGHT,
        # JBoozerResidual, IOTAS_WEIGHT, Jiota, VOLUME_WEIGHT, JVolume,
        # LENGTH_WEIGHT, JCurveLength, CC_WEIGHT, JCurveCurve, CS_WEIGHT,
        # JCurveSurface, CURVATURE_WEIGHT, JCurvature). JVolume is None so the
        # volume term is skipped, matching the other assembly tests.
        return [
            _FakeAlgebraicObjective(1.0, [1.0, 0.0]),
            2.0,
            _FakeAlgebraicObjective(3.0, [0.0, 2.0]),
            4.0,
            _FakeAlgebraicObjective(5.0, [1.0, 1.0]),
            6.0,
            None,
            8.0,
            _FakeAlgebraicObjective(9.0, [0.0, 3.0]),
            10.0,
            _FakeAlgebraicObjective(11.0, [1.0, -1.0]),
            12.0,
            _FakeAlgebraicObjective(13.0, [0.5, 0.5]),
            14.0,
            _FakeAlgebraicObjective(15.0, [2.0, -2.0]),
        ]

    def test_build_total_objective_shear_term_default_off_is_identical(self):
        # Weight 0 (and a None term) must leave the assembled objective
        # byte-identical to the call that omits the shear kwargs entirely.
        baseline = self.module.build_total_objective(*self._base_total_objective_args())
        default_off = self.module.build_total_objective(
            *self._base_total_objective_args(), JShear=None, SHEAR_WEIGHT=0.0
        )
        self.assertEqual(baseline.J(), default_off.J())
        np.testing.assert_array_equal(baseline.dJ(), default_off.dJ())

        # A nonzero weight with a None term stays inert (the None guard wins).
        weighted_none = self.module.build_total_objective(
            *self._base_total_objective_args(), JShear=None, SHEAR_WEIGHT=123.0
        )
        self.assertEqual(baseline.J(), weighted_none.J())
        np.testing.assert_array_equal(baseline.dJ(), weighted_none.dJ())

    def test_build_total_objective_adds_weighted_shear_term(self):
        # The assembly contract: a non-None JShear enters as + SHEAR_WEIGHT*JShear.
        # build_total_objective composes every term via +/*, so the shear term
        # must share the algebra family of the other terms; use the algebraic
        # fake here (the IotaShearShortfall value/sign/gradient are covered by the
        # dedicated term tests above).
        baseline = self.module.build_total_objective(*self._base_total_objective_args())
        shear = _FakeAlgebraicObjective(0.5, [0.25, -0.75])
        with_shear = self.module.build_total_objective(
            *self._base_total_objective_args(), JShear=shear, SHEAR_WEIGHT=10.0
        )
        self.assertAlmostEqual(with_shear.J() - baseline.J(), 10.0 * shear.J())
        np.testing.assert_allclose(with_shear.dJ() - baseline.dJ(), 10.0 * shear.dJ())

    def test_evaluate_base_objective_adds_weighted_shear_term(self):
        zero = _FakeAlgebraicObjective(0.0, [0.0, 0.0])
        shear = _FakeAlgebraicObjective(0.5, [0.25, -0.75])
        baseline = self.module.evaluate_base_objective(
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
            include_diagnostics=False,
        )
        with_shear = self.module.evaluate_base_objective(
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
            JShear=shear,
            SHEAR_WEIGHT=10.0,
            include_diagnostics=False,
        )

        self.assertAlmostEqual(with_shear["total"] - baseline["total"], 5.0)
        np.testing.assert_allclose(
            with_shear["grad"] - baseline["grad"],
            [2.5, -7.5],
        )

        thresholded = self.module.evaluate_base_objective(
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
            alm_formulation="thresholded_physics",
            JShear=shear,
            SHEAR_WEIGHT=10.0,
            include_diagnostics=False,
        )

        self.assertAlmostEqual(thresholded["total"], 5.0)
        np.testing.assert_allclose(thresholded["grad"], [2.5, -7.5])

        diagnostics = self.module.evaluate_base_objective(
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
            JShear=shear,
            SHEAR_WEIGHT=10.0,
        )
        self.assertTrue(diagnostics["shear_objective_enabled"])
        self.assertAlmostEqual(diagnostics["shear_weight"], 10.0)
        self.assertAlmostEqual(diagnostics["J_shear"], 0.5)
        np.testing.assert_allclose(diagnostics["dJ_shear"], [0.25, -0.75])

        off_diagnostics = self.module.evaluate_base_objective(
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
            JShear=shear,
            SHEAR_WEIGHT=0.0,
        )
        self.assertFalse(off_diagnostics["shear_objective_enabled"])
        self.assertAlmostEqual(off_diagnostics["J_shear"], 0.5)
        np.testing.assert_allclose(off_diagnostics["dJ_shear"], [0.25, -0.75])
