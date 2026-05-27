from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from simsopt.field.biotsavart import BiotSavart

from .fieldline_map import (
    DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    DifferentiableMagneticFieldLike,
    FieldlineIntegratorOptions,
    FieldlineTangentReturnResult,
    integrate_tangent_target_return_map,
)
from .greene_residue import (
    GreeneResidueDiagnostic,
    greene_residue_diagnostic_from_matrix,
)
from .periodic_orbit import (
    BRANCH_STATUS_CONVERGED,
    DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    PeriodicOrbitResult,
    PeriodicOrbitSolverOptions,
    solve_periodic_orbit,
)
from .poincare_chart import PoincareChart
from .rational_target import RationalTarget


FROZEN_ORBIT_FD_MODE = "frozen_orbit"
BRANCH_RESOLVED_FD_MODE = "branch_resolved"
BIOT_SAVART_BRANCH_RESOLVED_FD_MODE = "biot_savart_branch_resolved"


SectionState = tuple[float, float]
MonodromyMatrix = tuple[tuple[float, float], tuple[float, float]]
DofDirection = tuple[float, ...]


class ScalarFieldFactory(Protocol):
    def __call__(self, parameter_value: float) -> DifferentiableMagneticFieldLike: ...


@dataclass(frozen=True, slots=True)
class FrozenOrbitResidueEvaluation:
    parameter_value: float
    fixed_state: tuple[float, float]
    tangent_result: FieldlineTangentReturnResult
    residue_diagnostic: GreeneResidueDiagnostic

    @property
    def residue(self) -> float:
        return self.residue_diagnostic.residue


@dataclass(frozen=True, slots=True)
class ResidueCentralDifferenceDiagnostic:
    mode: str
    parameter_name: str
    parameter_value: float
    step: float
    base_residue: float
    plus_residue: float
    minus_residue: float
    derivative: float
    base_status: str | None
    plus_status: str | None
    minus_status: str | None
    base_state: tuple[float, float]
    plus_state: tuple[float, float]
    minus_state: tuple[float, float]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["base_state"] = list(self.base_state)
        payload["plus_state"] = list(self.plus_state)
        payload["minus_state"] = list(self.minus_state)
        return payload


@dataclass(frozen=True, slots=True)
class BiotSavartBranchResidueCentralDifferenceDiagnostic:
    mode: str
    step: float
    direction_norm: float
    direction: DofDirection
    base_residue: float
    plus_residue: float
    minus_residue: float
    derivative: float
    base_status: str
    plus_status: str
    minus_status: str
    base_state: SectionState
    plus_state: SectionState
    minus_state: SectionState
    branch_state_derivative: SectionState
    base_final_state: SectionState
    plus_final_state: SectionState
    minus_final_state: SectionState
    final_state_derivative: SectionState
    base_closure_residual: SectionState
    plus_closure_residual: SectionState
    minus_closure_residual: SectionState
    closure_residual_derivative: SectionState
    base_monodromy: MonodromyMatrix
    plus_monodromy: MonodromyMatrix
    minus_monodromy: MonodromyMatrix
    monodromy_derivative: MonodromyMatrix
    base_det_m: float
    plus_det_m: float
    minus_det_m: float
    base_winding: float
    plus_winding: float
    minus_winding: float

    @property
    def residue_derivative(self) -> float:
        return self.derivative

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["residue_derivative"] = self.derivative
        payload["direction"] = list(self.direction)
        payload["base_state"] = list(self.base_state)
        payload["plus_state"] = list(self.plus_state)
        payload["minus_state"] = list(self.minus_state)
        payload["branch_state_derivative"] = list(self.branch_state_derivative)
        payload["base_final_state"] = list(self.base_final_state)
        payload["plus_final_state"] = list(self.plus_final_state)
        payload["minus_final_state"] = list(self.minus_final_state)
        payload["final_state_derivative"] = list(self.final_state_derivative)
        payload["base_closure_residual"] = list(self.base_closure_residual)
        payload["plus_closure_residual"] = list(self.plus_closure_residual)
        payload["minus_closure_residual"] = list(self.minus_closure_residual)
        payload["closure_residual_derivative"] = list(self.closure_residual_derivative)
        payload["base_monodromy"] = _matrix_to_jsonable(self.base_monodromy)
        payload["plus_monodromy"] = _matrix_to_jsonable(self.plus_monodromy)
        payload["minus_monodromy"] = _matrix_to_jsonable(self.minus_monodromy)
        payload["monodromy_derivative"] = _matrix_to_jsonable(self.monodromy_derivative)
        return payload


@dataclass(frozen=True, slots=True)
class BiotSavartVjpDotDiagnostic:
    step: float
    vjp_dot: float
    central_difference: float
    absolute_error: float
    relative_error: float
    base_value: float
    plus_value: float
    minus_value: float

    def to_json_dict(self) -> dict[str, object]:
        return asdict(self)


def frozen_orbit_residue_evaluation(
    field: DifferentiableMagneticFieldLike,
    fixed_state: Sequence[float],
    *,
    parameter_value: float,
    target: RationalTarget,
    chart: PoincareChart,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
) -> FrozenOrbitResidueEvaluation:
    state = _normalize_state(fixed_state)
    tangent_result = integrate_tangent_target_return_map(
        field,
        state,
        target=target,
        chart=chart,
        options=integrator_options,
    )
    return FrozenOrbitResidueEvaluation(
        parameter_value=float(parameter_value),
        fixed_state=state,
        tangent_result=tangent_result,
        residue_diagnostic=greene_residue_diagnostic_from_matrix(
            tangent_result.monodromy
        ),
    )


def frozen_orbit_residue_central_difference(
    field_factory: ScalarFieldFactory,
    *,
    parameter_name: str,
    parameter_value: float,
    step: float,
    fixed_state: Sequence[float],
    target: RationalTarget,
    chart: PoincareChart,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
) -> ResidueCentralDifferenceDiagnostic:
    step_value = _positive_step(step)
    base = frozen_orbit_residue_evaluation(
        field_factory(float(parameter_value)),
        fixed_state,
        parameter_value=float(parameter_value),
        target=target,
        chart=chart,
        integrator_options=integrator_options,
    )
    plus = frozen_orbit_residue_evaluation(
        field_factory(float(parameter_value) + step_value),
        fixed_state,
        parameter_value=float(parameter_value) + step_value,
        target=target,
        chart=chart,
        integrator_options=integrator_options,
    )
    minus = frozen_orbit_residue_evaluation(
        field_factory(float(parameter_value) - step_value),
        fixed_state,
        parameter_value=float(parameter_value) - step_value,
        target=target,
        chart=chart,
        integrator_options=integrator_options,
    )
    return ResidueCentralDifferenceDiagnostic(
        mode=FROZEN_ORBIT_FD_MODE,
        parameter_name=str(parameter_name),
        parameter_value=float(parameter_value),
        step=step_value,
        base_residue=float(base.residue),
        plus_residue=float(plus.residue),
        minus_residue=float(minus.residue),
        derivative=float((plus.residue - minus.residue) / (2.0 * step_value)),
        base_status=None,
        plus_status=None,
        minus_status=None,
        base_state=base.fixed_state,
        plus_state=plus.fixed_state,
        minus_state=minus.fixed_state,
    )


def branch_resolved_residue_central_difference(
    field_factory: ScalarFieldFactory,
    initial_state: Sequence[float],
    *,
    parameter_name: str,
    parameter_value: float,
    step: float,
    target: RationalTarget,
    chart: PoincareChart,
    branch: str,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
) -> ResidueCentralDifferenceDiagnostic:
    step_value = _positive_step(step)
    base = solve_periodic_orbit(
        field_factory(float(parameter_value)),
        initial_state,
        target=target,
        chart=chart,
        branch=branch,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    plus = solve_periodic_orbit(
        field_factory(float(parameter_value) + step_value),
        base.state,
        target=target,
        chart=chart,
        branch=branch,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    minus = solve_periodic_orbit(
        field_factory(float(parameter_value) - step_value),
        base.state,
        target=target,
        chart=chart,
        branch=branch,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    _require_converged(base, label="base")
    _require_converged(plus, label="plus")
    _require_converged(minus, label="minus")
    return ResidueCentralDifferenceDiagnostic(
        mode=BRANCH_RESOLVED_FD_MODE,
        parameter_name=str(parameter_name),
        parameter_value=float(parameter_value),
        step=step_value,
        base_residue=float(base.residue_diagnostic.residue),
        plus_residue=float(plus.residue_diagnostic.residue),
        minus_residue=float(minus.residue_diagnostic.residue),
        derivative=float(
            (plus.residue_diagnostic.residue - minus.residue_diagnostic.residue)
            / (2.0 * step_value)
        ),
        base_status=base.status,
        plus_status=plus.status,
        minus_status=minus.status,
        base_state=base.state,
        plus_state=plus.state,
        minus_state=minus.state,
    )


def branch_resolved_biot_savart_residue_central_difference(
    biot_savart: BiotSavart,
    initial_state: Sequence[float],
    *,
    direction: Sequence[float],
    step: float,
    target: RationalTarget,
    chart: PoincareChart,
    branch: str,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
) -> BiotSavartBranchResidueCentralDifferenceDiagnostic:
    field = _require_direct_biot_savart(biot_savart)
    step_value = _positive_step(step)
    original_x = np.asarray(field.x, dtype=float).copy()
    direction_array, direction_norm = _normalized_direction(
        direction,
        expected_shape=original_x.shape,
    )
    try:
        field.x = original_x
        base = solve_periodic_orbit(
            field,
            initial_state,
            target=target,
            chart=chart,
            branch=branch,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
        _require_converged(base, label="base")

        field.x = original_x + step_value * direction_array
        plus = solve_periodic_orbit(
            field,
            base.state,
            target=target,
            chart=chart,
            branch=branch,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
        _require_converged(plus, label="plus")

        field.x = original_x - step_value * direction_array
        minus = solve_periodic_orbit(
            field,
            base.state,
            target=target,
            chart=chart,
            branch=branch,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
        _require_converged(minus, label="minus")
    finally:
        field.x = original_x

    plus_residue = float(plus.residue_diagnostic.residue)
    minus_residue = float(minus.residue_diagnostic.residue)
    base_final_state = _normalize_state(base.tangent_result.return_map.final_state)
    plus_final_state = _normalize_state(plus.tangent_result.return_map.final_state)
    minus_final_state = _normalize_state(minus.tangent_result.return_map.final_state)
    base_monodromy = _monodromy_matrix(base.tangent_result.monodromy)
    plus_monodromy = _monodromy_matrix(plus.tangent_result.monodromy)
    minus_monodromy = _monodromy_matrix(minus.tangent_result.monodromy)
    return BiotSavartBranchResidueCentralDifferenceDiagnostic(
        mode=BIOT_SAVART_BRANCH_RESOLVED_FD_MODE,
        step=step_value,
        direction_norm=direction_norm,
        direction=tuple(float(component) for component in direction_array),
        base_residue=float(base.residue_diagnostic.residue),
        plus_residue=plus_residue,
        minus_residue=minus_residue,
        derivative=float((plus_residue - minus_residue) / (2.0 * step_value)),
        base_status=base.status,
        plus_status=plus.status,
        minus_status=minus.status,
        base_state=base.state,
        plus_state=plus.state,
        minus_state=minus.state,
        branch_state_derivative=_central_difference_state(
            plus.state,
            minus.state,
            step_value,
        ),
        base_final_state=base_final_state,
        plus_final_state=plus_final_state,
        minus_final_state=minus_final_state,
        final_state_derivative=_central_difference_state(
            plus_final_state,
            minus_final_state,
            step_value,
        ),
        base_closure_residual=base.closure_residual,
        plus_closure_residual=plus.closure_residual,
        minus_closure_residual=minus.closure_residual,
        closure_residual_derivative=_central_difference_state(
            plus.closure_residual,
            minus.closure_residual,
            step_value,
        ),
        base_monodromy=base_monodromy,
        plus_monodromy=plus_monodromy,
        minus_monodromy=minus_monodromy,
        monodromy_derivative=_central_difference_monodromy(
            plus_monodromy,
            minus_monodromy,
            step_value,
        ),
        base_det_m=float(base.tangent_result.det_m),
        plus_det_m=float(plus.tangent_result.det_m),
        minus_det_m=float(minus.tangent_result.det_m),
        base_winding=float(base.winding),
        plus_winding=float(plus.winding),
        minus_winding=float(minus.winding),
    )


def biot_savart_b_and_dB_vjp_dot_test(
    biot_savart: BiotSavart,
    *,
    points: Sequence[Sequence[float]],
    b_cotangent: Sequence[Sequence[float]],
    grad_b_cotangent: Sequence[Sequence[Sequence[float]]],
    direction: Sequence[float],
    step: float,
) -> BiotSavartVjpDotDiagnostic:
    field = _require_direct_biot_savart(biot_savart)
    step_value = _positive_step(step)
    original_x = np.asarray(field.x, dtype=float).copy()
    direction_array, _direction_norm = _normalized_direction(
        direction,
        expected_shape=original_x.shape,
    )
    points_array = _normalized_points(points)
    b_cotangent_array = _normalized_cotangent(
        b_cotangent,
        expected_shape=(points_array.shape[0], 3),
        label="B cotangent",
    )
    grad_b_cotangent_array = _normalized_cotangent(
        grad_b_cotangent,
        expected_shape=(points_array.shape[0], 3, 3),
        label="grad-B cotangent",
    )
    try:
        field.x = original_x
        field.set_points(points_array)
        dB_part, dgradB_part = field.B_and_dB_vjp(
            b_cotangent_array,
            grad_b_cotangent_array,
        )
        gradient = np.asarray((dB_part + dgradB_part)(field), dtype=float)
        if gradient.shape != original_x.shape:
            raise ValueError(
                "BiotSavart B_and_dB_vjp gradient shape does not match free DOFs"
            )
        vjp_dot = float(np.dot(gradient, direction_array))
        base_value = _biot_savart_linear_observable(
            field,
            points_array,
            b_cotangent_array,
            grad_b_cotangent_array,
        )

        field.x = original_x + step_value * direction_array
        plus_value = _biot_savart_linear_observable(
            field,
            points_array,
            b_cotangent_array,
            grad_b_cotangent_array,
        )

        field.x = original_x - step_value * direction_array
        minus_value = _biot_savart_linear_observable(
            field,
            points_array,
            b_cotangent_array,
            grad_b_cotangent_array,
        )
    finally:
        field.x = original_x
        field.set_points(points_array)

    central_difference = float((plus_value - minus_value) / (2.0 * step_value))
    absolute_error = abs(vjp_dot - central_difference)
    relative_error = absolute_error / max(1.0, abs(vjp_dot), abs(central_difference))
    return BiotSavartVjpDotDiagnostic(
        step=step_value,
        vjp_dot=vjp_dot,
        central_difference=central_difference,
        absolute_error=float(absolute_error),
        relative_error=float(relative_error),
        base_value=base_value,
        plus_value=plus_value,
        minus_value=minus_value,
    )


def _normalize_state(state: Sequence[float]) -> SectionState:
    if len(state) != 2:
        raise ValueError("Residue sensitivity state must be [R, Z]")
    return (float(state[0]), float(state[1]))


def _monodromy_matrix(matrix: object) -> MonodromyMatrix:
    monodromy = np.asarray(matrix, dtype=float)
    if monodromy.shape != (2, 2):
        raise ValueError(
            f"Greene residue monodromy must have shape (2, 2), got {monodromy.shape}"
        )
    return (
        (float(monodromy[0, 0]), float(monodromy[0, 1])),
        (float(monodromy[1, 0]), float(monodromy[1, 1])),
    )


def _central_difference_state(
    plus_state: Sequence[float],
    minus_state: Sequence[float],
    step: float,
) -> SectionState:
    plus = np.asarray(_normalize_state(plus_state), dtype=float)
    minus = np.asarray(_normalize_state(minus_state), dtype=float)
    derivative = (plus - minus) / (2.0 * float(step))
    return (float(derivative[0]), float(derivative[1]))


def _central_difference_monodromy(
    plus_monodromy: MonodromyMatrix,
    minus_monodromy: MonodromyMatrix,
    step: float,
) -> MonodromyMatrix:
    plus = np.asarray(plus_monodromy, dtype=float)
    minus = np.asarray(minus_monodromy, dtype=float)
    return _monodromy_matrix((plus - minus) / (2.0 * float(step)))


def _matrix_to_jsonable(matrix: MonodromyMatrix) -> list[list[float]]:
    return [list(row) for row in matrix]


def _positive_step(step: float) -> float:
    step_value = float(step)
    if step_value <= 0.0:
        raise ValueError("Residue sensitivity finite-difference step must be positive")
    return step_value


def _normalized_direction(
    direction: Sequence[float],
    *,
    expected_shape: tuple[int, ...],
) -> tuple[np.ndarray, float]:
    direction_array = np.asarray(direction, dtype=float)
    if direction_array.shape != expected_shape:
        raise ValueError(
            "BiotSavart residue direction must match the field free-DOF shape "
            f"{expected_shape}, got {direction_array.shape}"
        )
    if not np.all(np.isfinite(direction_array)):
        raise ValueError("BiotSavart residue direction must be finite")
    direction_norm = float(np.linalg.norm(direction_array))
    if direction_norm <= 0.0:
        raise ValueError("BiotSavart residue direction must be nonzero")
    return direction_array, direction_norm


def _normalized_points(points: Sequence[Sequence[float]]) -> np.ndarray:
    points_array = np.asarray(points, dtype=float)
    if points_array.ndim != 2 or points_array.shape[1] != 3:
        raise ValueError("BiotSavart VJP dot test points must have shape (npoints, 3)")
    if points_array.shape[0] == 0:
        raise ValueError("BiotSavart VJP dot test requires at least one point")
    if not np.all(np.isfinite(points_array)):
        raise ValueError("BiotSavart VJP dot test points must be finite")
    return points_array


def _normalized_cotangent(
    values: object,
    *,
    expected_shape: tuple[int, ...],
    label: str,
) -> np.ndarray:
    cotangent = np.asarray(values, dtype=float)
    if cotangent.shape != expected_shape:
        raise ValueError(
            f"BiotSavart VJP dot test {label} must have shape "
            f"{expected_shape}, got {cotangent.shape}"
        )
    if not np.all(np.isfinite(cotangent)):
        raise ValueError(f"BiotSavart VJP dot test {label} must be finite")
    return cotangent


def _require_direct_biot_savart(field: BiotSavart) -> BiotSavart:
    if not isinstance(field, BiotSavart):
        raise TypeError("Residue coil-DOF diagnostics require direct BiotSavart")
    return field


def _biot_savart_linear_observable(
    field: BiotSavart,
    points: np.ndarray,
    b_cotangent: np.ndarray,
    grad_b_cotangent: np.ndarray,
) -> float:
    field.set_points(points)
    return float(
        np.sum(np.asarray(field.B(), dtype=float) * b_cotangent)
        + np.sum(np.asarray(field.dB_by_dX(), dtype=float) * grad_b_cotangent)
    )


def _require_converged(result: PeriodicOrbitResult, *, label: str) -> None:
    if result.status != BRANCH_STATUS_CONVERGED:
        raise ValueError(
            f"Branch-resolved residue finite difference requires {label} branch "
            f"status {BRANCH_STATUS_CONVERGED}, got {result.status}"
        )
