from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np

from simsopt._core.derivative import Derivative
from simsopt.field.biotsavart import BiotSavart

from .fieldline_map import (
    DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    DifferentiableMagneticFieldLike,
    FieldlineIntegratorOptions,
    FieldlineReturnResult,
    FieldlineTangentReturnResult,
    cartesian_from_cylindrical,
    cylindrical_basis,
    full_torus_return_section_unwrapped_thetas,
    integrate_tangent_target_return_map,
    target_winding_residual,
)
from .greene_residue import (
    GreeneResidueDiagnostic,
    greene_residue_diagnostic_from_matrix,
)
from .periodic_orbit import (
    BRANCH_STATUS_BAD_DETERMINANT,
    BRANCH_STATUS_BRANCH_MISMATCH,
    BRANCH_STATUS_CONVERGED,
    BRANCH_STATUS_MAX_ITERATIONS,
    BRANCH_STATUS_NEWTON_STALLED,
    BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW,
    BRANCH_STATUS_WRONG_WINDING,
    PeriodicOrbitResult,
    DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    PeriodicOrbitSolverOptions,
    radial_label_in_target_window,
    residue_classification_matches_branch,
    solve_periodic_orbit,
    tangent_map_determinant_within_tolerance,
)
from .poincare_chart import PoincareChart
from .rational_target import RationalTarget


FROZEN_ORBIT_FD_MODE = "frozen_orbit"
BRANCH_RESOLVED_FD_MODE = "branch_resolved"
BIOT_SAVART_BRANCH_RESOLVED_FD_MODE = "biot_savart_branch_resolved"
BIOT_SAVART_BRANCH_RESOLVED_TAYLOR_MODE = "biot_savart_branch_resolved_taylor"
BIOT_SAVART_BRANCH_RESOLVED_VJP_MODE = "biot_savart_branch_resolved_vjp"
BIOT_SAVART_BRANCH_RESOLVED_VJP_TAYLOR_MODE = "biot_savart_branch_resolved_vjp_taylor"
BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_ACTIVE = "active"
BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_SATISFIED_FROZEN = "satisfied_frozen"
DEFAULT_RESIDUE_SATISFIED_THRESHOLD = 1.0e-4


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


@dataclass(frozen=True, slots=True)
class BiotSavartResidueTaylorSample:
    step: float
    residue: float
    first_order_prediction: float
    residual: float
    absolute_residual: float
    status: str
    state: SectionState

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = list(self.state)
        return payload


@dataclass(frozen=True, slots=True)
class BiotSavartBranchResidueTaylorDiagnostic:
    mode: str
    derivative_step: float
    direction_norm: float
    direction: DofDirection
    base_residue: float
    directional_derivative: float
    base_status: str
    base_state: SectionState
    samples: tuple[BiotSavartResidueTaylorSample, ...]
    observed_orders: tuple[float, ...]

    def to_json_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["direction"] = list(self.direction)
        payload["base_state"] = list(self.base_state)
        payload["samples"] = [sample.to_json_dict() for sample in self.samples]
        payload["observed_orders"] = list(self.observed_orders)
        return payload


@dataclass(frozen=True, slots=True)
class BiotSavartBranchResidueVjpDiagnostic:
    mode: str
    gradient_status: str
    residue: float
    base_status: str
    state: SectionState
    final_state: SectionState
    closure_residual: SectionState
    monodromy: MonodromyMatrix
    trace_m: float
    det_m: float
    dresidue_dstate: SectionState
    implicit_adjoint: SectionState
    cotangent_point_count: int
    b_cotangent_norm: float
    grad_b_cotangent_norm: float
    gradient_norm: float
    gradient: DofDirection
    derivative: Derivative

    @property
    def residue_derivative(self) -> Derivative:
        return self.derivative

    def to_json_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "gradient_status": self.gradient_status,
            "residue": self.residue,
            "base_status": self.base_status,
            "state": list(self.state),
            "final_state": list(self.final_state),
            "closure_residual": list(self.closure_residual),
            "monodromy": _matrix_to_jsonable(self.monodromy),
            "trace_m": self.trace_m,
            "det_m": self.det_m,
            "dresidue_dstate": list(self.dresidue_dstate),
            "implicit_adjoint": list(self.implicit_adjoint),
            "cotangent_point_count": self.cotangent_point_count,
            "b_cotangent_norm": self.b_cotangent_norm,
            "grad_b_cotangent_norm": self.grad_b_cotangent_norm,
            "gradient_norm": self.gradient_norm,
            "gradient": list(self.gradient),
        }


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


def branch_resolved_biot_savart_residue_taylor_diagnostic(
    biot_savart: BiotSavart,
    initial_state: Sequence[float],
    *,
    direction: Sequence[float],
    derivative_step: float,
    probe_steps: Sequence[float],
    target: RationalTarget,
    chart: PoincareChart,
    branch: str,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
) -> BiotSavartBranchResidueTaylorDiagnostic:
    field = _require_direct_biot_savart(biot_savart)
    steps = _normalized_probe_steps(probe_steps)
    derivative = branch_resolved_biot_savart_residue_central_difference(
        field,
        initial_state,
        direction=direction,
        step=derivative_step,
        target=target,
        chart=chart,
        branch=branch,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    original_x = np.asarray(field.x, dtype=float).copy()
    direction_array, direction_norm = _normalized_direction(
        direction,
        expected_shape=original_x.shape,
    )
    samples: list[BiotSavartResidueTaylorSample] = []
    try:
        for step_value in steps:
            field.x = original_x + step_value * direction_array
            result = solve_periodic_orbit(
                field,
                derivative.base_state,
                target=target,
                chart=chart,
                branch=branch,
                integrator_options=integrator_options,
                solver_options=solver_options,
            )
            _require_converged(result, label=f"probe step {step_value:g}")
            residue = float(result.residue_diagnostic.residue)
            prediction = float(
                derivative.base_residue + step_value * derivative.derivative
            )
            residual = float(residue - prediction)
            samples.append(
                BiotSavartResidueTaylorSample(
                    step=step_value,
                    residue=residue,
                    first_order_prediction=prediction,
                    residual=residual,
                    absolute_residual=abs(residual),
                    status=result.status,
                    state=result.state,
                )
            )
    finally:
        field.x = original_x

    return BiotSavartBranchResidueTaylorDiagnostic(
        mode=BIOT_SAVART_BRANCH_RESOLVED_TAYLOR_MODE,
        derivative_step=derivative.step,
        direction_norm=direction_norm,
        direction=tuple(float(component) for component in direction_array),
        base_residue=derivative.base_residue,
        directional_derivative=derivative.derivative,
        base_status=derivative.base_status,
        base_state=derivative.base_state,
        samples=tuple(samples),
        observed_orders=_observed_orders(tuple(samples)),
    )


def branch_resolved_biot_savart_residue_vjp(
    biot_savart: BiotSavart,
    initial_state: Sequence[float],
    *,
    target: RationalTarget,
    chart: PoincareChart,
    branch: str,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    r_satisfied: float = DEFAULT_RESIDUE_SATISFIED_THRESHOLD,
    local_difference_step: float = 1.0e-6,
) -> BiotSavartBranchResidueVjpDiagnostic:
    field = _require_direct_biot_savart(biot_savart)
    satisfied_threshold = float(r_satisfied)
    if satisfied_threshold < 0.0:
        raise ValueError("Greene residue r_satisfied must be nonnegative")
    local_step = _positive_step(local_difference_step)
    orbit = _solve_rk4_periodic_orbit(
        field,
        initial_state,
        target=target,
        chart=chart,
        branch=branch,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    _require_converged(orbit, label="base")
    residue = float(orbit.residue_diagnostic.residue)
    monodromy = np.asarray(orbit.tangent_result.monodromy, dtype=float)
    if abs(residue) <= satisfied_threshold:
        derivative = Derivative()
        cotangent_stats = _CotangentStats(
            point_count=0,
            b_cotangent_norm=0.0,
            grad_b_cotangent_norm=0.0,
        )
        gradient = np.asarray(derivative(field), dtype=float)
        return _vjp_diagnostic_from_parts(
            orbit=orbit,
            derivative=derivative,
            gradient=gradient,
            dresidue_dstate=np.zeros(2, dtype=float),
            implicit_adjoint=np.zeros(2, dtype=float),
            cotangent_stats=cotangent_stats,
            gradient_status=BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_SATISFIED_FROZEN,
        )

    dresidue_dstate = _rk4_residue_state_gradient(
        field,
        orbit.state,
        target=target,
        integrator_options=integrator_options,
        local_difference_step=local_step,
    )
    fixed_point_jacobian = monodromy - np.eye(2, dtype=float)
    implicit_adjoint = np.linalg.solve(fixed_point_jacobian.T, dresidue_dstate)
    terminal_adjoint = np.asarray(
        [
            -implicit_adjoint[0],
            -implicit_adjoint[1],
            -0.25,
            0.0,
            0.0,
            -0.25,
        ],
        dtype=float,
    )
    derivative, cotangent_stats = _rk4_vjp_derivative(
        field,
        orbit.state,
        target=target,
        integrator_options=integrator_options,
        terminal_adjoint=terminal_adjoint,
        local_difference_step=local_step,
    )
    gradient = np.asarray(derivative(field), dtype=float)
    return _vjp_diagnostic_from_parts(
        orbit=orbit,
        derivative=derivative,
        gradient=gradient,
        dresidue_dstate=dresidue_dstate,
        implicit_adjoint=implicit_adjoint,
        cotangent_stats=cotangent_stats,
        gradient_status=BIOT_SAVART_BRANCH_RESIDUE_GRADIENT_ACTIVE,
    )


def branch_resolved_biot_savart_residue_vjp_taylor_diagnostic(
    biot_savart: BiotSavart,
    initial_state: Sequence[float],
    *,
    direction: Sequence[float],
    probe_steps: Sequence[float],
    target: RationalTarget,
    chart: PoincareChart,
    branch: str,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
    r_satisfied: float = DEFAULT_RESIDUE_SATISFIED_THRESHOLD,
    local_difference_step: float = 1.0e-6,
) -> BiotSavartBranchResidueTaylorDiagnostic:
    field = _require_direct_biot_savart(biot_savart)
    steps = _normalized_probe_steps(probe_steps)
    gradient_diagnostic = branch_resolved_biot_savart_residue_vjp(
        field,
        initial_state,
        target=target,
        chart=chart,
        branch=branch,
        integrator_options=integrator_options,
        solver_options=solver_options,
        r_satisfied=r_satisfied,
        local_difference_step=local_difference_step,
    )
    original_x = np.asarray(field.x, dtype=float).copy()
    direction_array, direction_norm = _normalized_direction(
        direction,
        expected_shape=original_x.shape,
    )
    directional_derivative = float(
        np.dot(np.asarray(gradient_diagnostic.gradient, dtype=float), direction_array)
    )
    samples: list[BiotSavartResidueTaylorSample] = []
    try:
        for step_value in steps:
            field.x = original_x + step_value * direction_array
            result = _solve_rk4_periodic_orbit(
                field,
                gradient_diagnostic.state,
                target=target,
                chart=chart,
                branch=branch,
                integrator_options=integrator_options,
                solver_options=solver_options,
            )
            _require_converged(result, label=f"probe step {step_value:g}")
            residue = float(result.residue_diagnostic.residue)
            prediction = float(
                gradient_diagnostic.residue + step_value * directional_derivative
            )
            residual = float(residue - prediction)
            samples.append(
                BiotSavartResidueTaylorSample(
                    step=step_value,
                    residue=residue,
                    first_order_prediction=prediction,
                    residual=residual,
                    absolute_residual=abs(residual),
                    status=result.status,
                    state=result.state,
                )
            )
    finally:
        field.x = original_x

    return BiotSavartBranchResidueTaylorDiagnostic(
        mode=BIOT_SAVART_BRANCH_RESOLVED_VJP_TAYLOR_MODE,
        derivative_step=local_difference_step,
        direction_norm=direction_norm,
        direction=tuple(float(component) for component in direction_array),
        base_residue=gradient_diagnostic.residue,
        directional_derivative=directional_derivative,
        base_status=gradient_diagnostic.base_status,
        base_state=gradient_diagnostic.state,
        samples=tuple(samples),
        observed_orders=_observed_orders(tuple(samples)),
    )


def solve_biot_savart_residue_branch(
    biot_savart: BiotSavart,
    initial_state: Sequence[float],
    *,
    target: RationalTarget,
    chart: PoincareChart,
    branch: str,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
) -> PeriodicOrbitResult:
    field = _require_direct_biot_savart(biot_savart)
    return _solve_rk4_periodic_orbit(
        field,
        initial_state,
        target=target,
        chart=chart,
        branch=branch,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )


@dataclass(frozen=True, slots=True)
class _FieldData:
    point: np.ndarray
    b: np.ndarray
    grad_b: np.ndarray


@dataclass(frozen=True, slots=True)
class _RhsEvaluation:
    field_data: _FieldData
    rhs: np.ndarray


@dataclass(frozen=True, slots=True)
class _Rk4StepRecord:
    phi: float
    step: float
    state: np.ndarray
    stage1: _RhsEvaluation
    stage2_state: np.ndarray
    stage2: _RhsEvaluation
    stage3_state: np.ndarray
    stage3: _RhsEvaluation
    stage4_state: np.ndarray
    stage4: _RhsEvaluation


@dataclass(frozen=True, slots=True)
class _CotangentStats:
    point_count: int
    b_cotangent_norm: float
    grad_b_cotangent_norm: float


def _solve_rk4_periodic_orbit(
    field: DifferentiableMagneticFieldLike,
    initial_state: Sequence[float],
    *,
    target: RationalTarget,
    chart: PoincareChart,
    branch: str,
    integrator_options: FieldlineIntegratorOptions,
    solver_options: PeriodicOrbitSolverOptions,
) -> PeriodicOrbitResult:
    branch_label = _validate_branch_for_target(target, branch)
    start_state = np.asarray(_normalize_state(initial_state), dtype=float)
    state = start_state.copy()

    for iteration in range(solver_options.max_iterations + 1):
        tangent_result, _records = _rk4_tangent_return_map(
            field,
            state,
            target=target,
            chart=chart,
            integrator_options=integrator_options,
            with_tape=False,
        )
        closure_residual = (
            np.asarray(tangent_result.return_map.final_state, dtype=float) - state
        )
        residual_norm = float(np.linalg.norm(closure_residual))
        if residual_norm <= solver_options.residual_tolerance:
            return _rk4_periodic_orbit_result(
                target=target,
                branch=branch_label,
                chart=chart,
                initial_state=(float(start_state[0]), float(start_state[1])),
                state=state,
                closure_residual=closure_residual,
                closure_residual_norm=residual_norm,
                iterations=iteration,
                status=_rk4_branch_status(
                    target=target,
                    branch=branch_label,
                    chart=chart,
                    tangent_result=tangent_result,
                    solver_options=solver_options,
                ),
                tangent_result=tangent_result,
            )
        if iteration == solver_options.max_iterations:
            return _rk4_periodic_orbit_result(
                target=target,
                branch=branch_label,
                chart=chart,
                initial_state=(float(start_state[0]), float(start_state[1])),
                state=state,
                closure_residual=closure_residual,
                closure_residual_norm=residual_norm,
                iterations=iteration,
                status=BRANCH_STATUS_MAX_ITERATIONS,
                tangent_result=tangent_result,
            )

        jacobian = np.asarray(tangent_result.monodromy, dtype=float) - np.eye(2)
        step = _trust_region_step(
            jacobian,
            closure_residual,
            max_step_norm=solver_options.max_step_norm,
        )
        next_state = _damped_rk4_newton_state(
            field,
            state,
            step,
            current_residual_norm=residual_norm,
            target=target,
            chart=chart,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
        if next_state is None:
            return _rk4_periodic_orbit_result(
                target=target,
                branch=branch_label,
                chart=chart,
                initial_state=(float(start_state[0]), float(start_state[1])),
                state=state,
                closure_residual=closure_residual,
                closure_residual_norm=residual_norm,
                iterations=iteration,
                status=BRANCH_STATUS_NEWTON_STALLED,
                tangent_result=tangent_result,
            )
        state = next_state

    raise RuntimeError("RK4 periodic-orbit solver exited unexpectedly")


def _rk4_tangent_return_map(
    field: DifferentiableMagneticFieldLike,
    initial_state: Sequence[float],
    *,
    target: RationalTarget,
    chart: PoincareChart,
    integrator_options: FieldlineIntegratorOptions,
    with_tape: bool,
) -> tuple[FieldlineTangentReturnResult, tuple[_Rk4StepRecord, ...]]:
    start_state = np.asarray(_normalize_state(initial_state), dtype=float)
    steps_per_torus = max(int(integrator_options.samples_per_full_torus), 96)
    steps = steps_per_torus * int(target.q)
    if steps <= 0:
        raise ValueError("RK4 return map requires a positive step count")
    phi_start = float(target.phi0)
    phi_stop = phi_start + 2.0 * np.pi * float(target.q)
    step = (phi_stop - phi_start) / float(steps)
    augmented = np.concatenate([start_state, np.eye(2, dtype=float).reshape(4)])
    phi_values = [phi_start]
    states = [start_state.copy()]
    bphi_ratios: list[float] = []
    records: list[_Rk4StepRecord] = []

    for step_index in range(steps):
        phi = phi_start + float(step_index) * step
        augmented, record = _rk4_step(
            field,
            phi,
            augmented,
            step,
            integrator_options=integrator_options,
        )
        bphi_ratios.extend(
            (
                _bphi_over_b(record.stage1.field_data.b, phi),
                _bphi_over_b(record.stage2.field_data.b, phi + 0.5 * step),
                _bphi_over_b(record.stage3.field_data.b, phi + 0.5 * step),
                _bphi_over_b(record.stage4.field_data.b, phi + step),
            )
        )
        if with_tape:
            records.append(record)
        phi_values.append(phi + step)
        states.append(np.asarray(augmented[:2], dtype=float).copy())

    phi_grid = np.asarray(phi_values, dtype=float)
    state_grid = np.asarray(states, dtype=float)
    monodromy = np.asarray(augmented[2:], dtype=float).reshape((2, 2))
    unwrapped_theta = chart.unwrapped_thetas(state_grid)
    return_section_theta = full_torus_return_section_unwrapped_thetas(
        chart,
        state_grid,
        samples_per_full_torus=steps_per_torus,
        expected_winding=target.expected_winding(),
    )
    return_map = FieldlineReturnResult(
        initial_state=(float(start_state[0]), float(start_state[1])),
        final_state=(float(state_grid[-1, 0]), float(state_grid[-1, 1])),
        phi_grid=phi_grid,
        states=state_grid,
        unwrapped_theta=unwrapped_theta,
        return_section_unwrapped_theta=return_section_theta,
        winding=float(
            (return_section_theta[-1] - return_section_theta[0]) / (2.0 * np.pi)
        ),
        min_bphi_over_b=float(np.min(np.asarray(bphi_ratios, dtype=float))),
    )
    return (
        FieldlineTangentReturnResult(
            return_map=return_map,
            monodromy=monodromy,
            trace_m=float(np.trace(monodromy)),
            det_m=float(np.linalg.det(monodromy)),
        ),
        tuple(records),
    )


def _rk4_step(
    field: DifferentiableMagneticFieldLike,
    phi: float,
    state: np.ndarray,
    step: float,
    *,
    integrator_options: FieldlineIntegratorOptions,
    stage1_override: tuple[np.ndarray, np.ndarray] | None = None,
    stage2_override: tuple[np.ndarray, np.ndarray] | None = None,
    stage3_override: tuple[np.ndarray, np.ndarray] | None = None,
    stage4_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, _Rk4StepRecord]:
    state_array = np.asarray(state, dtype=float)
    stage1 = _augmented_rhs_evaluation(
        field,
        float(phi),
        state_array,
        integrator_options=integrator_options,
        override=stage1_override,
    )
    stage2_state = state_array + 0.5 * float(step) * stage1.rhs
    stage2 = _augmented_rhs_evaluation(
        field,
        float(phi) + 0.5 * float(step),
        stage2_state,
        integrator_options=integrator_options,
        override=stage2_override,
    )
    stage3_state = state_array + 0.5 * float(step) * stage2.rhs
    stage3 = _augmented_rhs_evaluation(
        field,
        float(phi) + 0.5 * float(step),
        stage3_state,
        integrator_options=integrator_options,
        override=stage3_override,
    )
    stage4_state = state_array + float(step) * stage3.rhs
    stage4 = _augmented_rhs_evaluation(
        field,
        float(phi) + float(step),
        stage4_state,
        integrator_options=integrator_options,
        override=stage4_override,
    )
    next_state = (
        state_array
        + float(step)
        * (stage1.rhs + 2.0 * stage2.rhs + 2.0 * stage3.rhs + stage4.rhs)
        / 6.0
    )
    return (
        next_state,
        _Rk4StepRecord(
            phi=float(phi),
            step=float(step),
            state=state_array.copy(),
            stage1=stage1,
            stage2_state=stage2_state.copy(),
            stage2=stage2,
            stage3_state=stage3_state.copy(),
            stage3=stage3,
            stage4_state=stage4_state.copy(),
            stage4=stage4,
        ),
    )


def _augmented_rhs_evaluation(
    field: DifferentiableMagneticFieldLike,
    phi: float,
    state: np.ndarray,
    *,
    integrator_options: FieldlineIntegratorOptions,
    override: tuple[np.ndarray, np.ndarray] | None,
) -> _RhsEvaluation:
    if override is None:
        field_data = _field_data_at_augmented_state(field, float(phi), state)
    else:
        point = cartesian_from_cylindrical(
            float(state[0]),
            float(phi),
            float(state[1]),
        )
        field_data = _FieldData(
            point=point,
            b=np.asarray(override[0], dtype=float),
            grad_b=np.asarray(override[1], dtype=float),
        )
    rhs = _augmented_rhs_from_field_data(
        float(phi),
        state,
        field_data.b,
        field_data.grad_b,
        min_bphi_over_b=integrator_options.min_bphi_over_b,
    )
    return _RhsEvaluation(field_data=field_data, rhs=rhs)


def _field_data_at_augmented_state(
    field: DifferentiableMagneticFieldLike,
    phi: float,
    state: np.ndarray,
) -> _FieldData:
    point = cartesian_from_cylindrical(float(state[0]), float(phi), float(state[1]))
    field.set_points(point.reshape((1, 3)))
    b = np.asarray(field.B(), dtype=float)
    grad_b = np.asarray(field.dB_by_dX(), dtype=float)
    if b.shape != (1, 3):
        raise ValueError(f"Magnetic field B() must have shape (1, 3), got {b.shape}")
    if grad_b.shape != (1, 3, 3):
        raise ValueError(
            f"Magnetic field dB_by_dX() must have shape (1, 3, 3), got {grad_b.shape}"
        )
    return _FieldData(point=point, b=b[0].copy(), grad_b=grad_b[0].copy())


def _augmented_rhs_from_field_data(
    phi: float,
    augmented_state: np.ndarray,
    b_vector: np.ndarray,
    field_jacobian_xyz: np.ndarray,
    *,
    min_bphi_over_b: float,
) -> np.ndarray:
    state = np.asarray(augmented_state[:2], dtype=float)
    tangent_matrix = np.asarray(augmented_state[2:], dtype=float).reshape((2, 2))
    velocity, jacobian = _rhs_and_jacobian_from_field_data(
        float(phi),
        state,
        np.asarray(b_vector, dtype=float),
        np.asarray(field_jacobian_xyz, dtype=float),
        min_bphi_over_b=float(min_bphi_over_b),
    )
    return np.concatenate([velocity, (jacobian @ tangent_matrix).reshape(4)])


def _rhs_and_jacobian_from_field_data(
    phi: float,
    state: np.ndarray,
    b_vector: np.ndarray,
    field_jacobian_xyz: np.ndarray,
    *,
    min_bphi_over_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    radius = float(state[0])
    if radius <= 0.0:
        raise ValueError("RK4 Greene state requires R > 0")
    e_r, e_phi, e_z = cylindrical_basis(float(phi))
    b_norm = float(np.linalg.norm(b_vector))
    if b_norm <= 0.0 or not np.isfinite(b_norm):
        raise ValueError("Magnetic field norm must be finite and positive")
    b_phi = float(np.dot(b_vector, e_phi))
    if abs(b_phi) / b_norm <= float(min_bphi_over_b):
        raise ValueError("|B_phi|/|B| fell below the Greene RK4 threshold")
    b_r = float(np.dot(b_vector, e_r))
    b_z = float(np.dot(b_vector, e_z))
    dB_dR = e_r @ field_jacobian_xyz
    dB_dZ = e_z @ field_jacobian_xyz
    dbr_dr = float(np.dot(dB_dR, e_r))
    dbphi_dr = float(np.dot(dB_dR, e_phi))
    dbz_dr = float(np.dot(dB_dR, e_z))
    dbr_dz = float(np.dot(dB_dZ, e_r))
    dbphi_dz = float(np.dot(dB_dZ, e_phi))
    dbz_dz = float(np.dot(dB_dZ, e_z))

    inv_bphi = 1.0 / b_phi
    inv_bphi_squared = inv_bphi * inv_bphi
    velocity = np.asarray(
        [radius * b_r * inv_bphi, radius * b_z * inv_bphi],
        dtype=float,
    )
    jacobian = np.asarray(
        [
            [
                b_r * inv_bphi
                + radius * (dbr_dr * b_phi - b_r * dbphi_dr) * inv_bphi_squared,
                radius * (dbr_dz * b_phi - b_r * dbphi_dz) * inv_bphi_squared,
            ],
            [
                b_z * inv_bphi
                + radius * (dbz_dr * b_phi - b_z * dbphi_dr) * inv_bphi_squared,
                radius * (dbz_dz * b_phi - b_z * dbphi_dz) * inv_bphi_squared,
            ],
        ],
        dtype=float,
    )
    return velocity, jacobian


def _rk4_residue_state_gradient(
    field: DifferentiableMagneticFieldLike,
    state: SectionState,
    *,
    target: RationalTarget,
    integrator_options: FieldlineIntegratorOptions,
    local_difference_step: float,
) -> np.ndarray:
    base_state = np.asarray(state, dtype=float)
    gradient = np.empty(2, dtype=float)
    for index in range(2):
        step = _scaled_difference_step(base_state[index], local_difference_step)
        perturbation = np.zeros(2, dtype=float)
        perturbation[index] = step
        plus = _rk4_residue_at_state(
            field,
            base_state + perturbation,
            target=target,
            integrator_options=integrator_options,
        )
        minus = _rk4_residue_at_state(
            field,
            base_state - perturbation,
            target=target,
            integrator_options=integrator_options,
        )
        gradient[index] = (plus - minus) / (2.0 * step)
    return gradient


def _rk4_residue_at_state(
    field: DifferentiableMagneticFieldLike,
    state: np.ndarray,
    *,
    target: RationalTarget,
    integrator_options: FieldlineIntegratorOptions,
) -> float:
    chart = PoincareChart(axis_r=float(state[0]), axis_z=float(state[1]))
    tangent_result, _records = _rk4_tangent_return_map(
        field,
        state,
        target=target,
        chart=chart,
        integrator_options=integrator_options,
        with_tape=False,
    )
    return float(
        greene_residue_diagnostic_from_matrix(tangent_result.monodromy).residue
    )


def _rk4_vjp_derivative(
    field: BiotSavart,
    state: SectionState,
    *,
    target: RationalTarget,
    integrator_options: FieldlineIntegratorOptions,
    terminal_adjoint: np.ndarray,
    local_difference_step: float,
) -> tuple[Derivative, _CotangentStats]:
    _tangent_result, records = _rk4_tangent_return_map(
        field,
        state,
        target=target,
        chart=PoincareChart(axis_r=float(state[0]), axis_z=float(state[1])),
        integrator_options=integrator_options,
        with_tape=True,
    )
    adjoint = np.asarray(terminal_adjoint, dtype=float)
    points: list[np.ndarray] = []
    b_cotangents: list[np.ndarray] = []
    grad_b_cotangents: list[np.ndarray] = []
    for record in reversed(records):
        adjoint = _rk4_step_vjp(
            field,
            record,
            adjoint,
            integrator_options=integrator_options,
            local_difference_step=local_difference_step,
            points=points,
            b_cotangents=b_cotangents,
            grad_b_cotangents=grad_b_cotangents,
        )
    derivative, stats = _biot_savart_derivative_from_cotangents(
        field,
        points,
        b_cotangents,
        grad_b_cotangents,
    )
    return derivative, stats


def _rk4_step_vjp(
    field: DifferentiableMagneticFieldLike,
    record: _Rk4StepRecord,
    output_adjoint: np.ndarray,
    *,
    integrator_options: FieldlineIntegratorOptions,
    local_difference_step: float,
    points: list[np.ndarray],
    b_cotangents: list[np.ndarray],
    grad_b_cotangents: list[np.ndarray],
) -> np.ndarray:
    state_jacobian = _rk4_step_state_jacobian(
        field,
        record,
        integrator_options=integrator_options,
        local_difference_step=local_difference_step,
    )
    previous_adjoint = state_jacobian.T @ output_adjoint
    for stage_index in (1, 2, 3, 4):
        point, b_cotangent, grad_b_cotangent = _rk4_stage_cotangents(
            field,
            record,
            output_adjoint,
            stage_index=stage_index,
            integrator_options=integrator_options,
            local_difference_step=local_difference_step,
        )
        points.append(point)
        b_cotangents.append(b_cotangent)
        grad_b_cotangents.append(grad_b_cotangent)
    return previous_adjoint


def _rk4_step_state_jacobian(
    field: DifferentiableMagneticFieldLike,
    record: _Rk4StepRecord,
    *,
    integrator_options: FieldlineIntegratorOptions,
    local_difference_step: float,
) -> np.ndarray:
    jacobian = np.empty((6, 6), dtype=float)
    for index in range(6):
        step = _scaled_difference_step(record.state[index], local_difference_step)
        perturbation = np.zeros(6, dtype=float)
        perturbation[index] = step
        plus, _record = _rk4_step(
            field,
            record.phi,
            record.state + perturbation,
            record.step,
            integrator_options=integrator_options,
        )
        minus, _record = _rk4_step(
            field,
            record.phi,
            record.state - perturbation,
            record.step,
            integrator_options=integrator_options,
        )
        jacobian[:, index] = (plus - minus) / (2.0 * step)
    return jacobian


def _rk4_stage_cotangents(
    field: DifferentiableMagneticFieldLike,
    record: _Rk4StepRecord,
    output_adjoint: np.ndarray,
    *,
    stage_index: int,
    integrator_options: FieldlineIntegratorOptions,
    local_difference_step: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stage_index == 1:
        field_data = record.stage1.field_data
    elif stage_index == 2:
        field_data = record.stage2.field_data
    elif stage_index == 3:
        field_data = record.stage3.field_data
    elif stage_index == 4:
        field_data = record.stage4.field_data
    else:
        raise ValueError("RK4 stage index must be in 1..4")
    b_cotangent = np.empty(3, dtype=float)
    grad_b_cotangent = np.empty((3, 3), dtype=float)
    for component in range(3):
        step = _scaled_difference_step(field_data.b[component], local_difference_step)
        plus_b = field_data.b.copy()
        minus_b = field_data.b.copy()
        plus_b[component] += step
        minus_b[component] -= step
        plus = _rk4_step_with_stage_override(
            field,
            record,
            stage_index=stage_index,
            b_override=plus_b,
            grad_b_override=field_data.grad_b,
            integrator_options=integrator_options,
        )
        minus = _rk4_step_with_stage_override(
            field,
            record,
            stage_index=stage_index,
            b_override=minus_b,
            grad_b_override=field_data.grad_b,
            integrator_options=integrator_options,
        )
        b_cotangent[component] = float(
            np.dot(output_adjoint, (plus - minus) / (2.0 * step))
        )
    for row in range(3):
        for column in range(3):
            step = _scaled_difference_step(
                field_data.grad_b[row, column],
                local_difference_step,
            )
            plus_grad = field_data.grad_b.copy()
            minus_grad = field_data.grad_b.copy()
            plus_grad[row, column] += step
            minus_grad[row, column] -= step
            plus = _rk4_step_with_stage_override(
                field,
                record,
                stage_index=stage_index,
                b_override=field_data.b,
                grad_b_override=plus_grad,
                integrator_options=integrator_options,
            )
            minus = _rk4_step_with_stage_override(
                field,
                record,
                stage_index=stage_index,
                b_override=field_data.b,
                grad_b_override=minus_grad,
                integrator_options=integrator_options,
            )
            grad_b_cotangent[row, column] = float(
                np.dot(output_adjoint, (plus - minus) / (2.0 * step))
            )
    return field_data.point.copy(), b_cotangent, grad_b_cotangent


def _rk4_step_with_stage_override(
    field: DifferentiableMagneticFieldLike,
    record: _Rk4StepRecord,
    *,
    stage_index: int,
    b_override: np.ndarray,
    grad_b_override: np.ndarray,
    integrator_options: FieldlineIntegratorOptions,
) -> np.ndarray:
    override = (np.asarray(b_override, dtype=float), np.asarray(grad_b_override))
    stage1_override = override if stage_index == 1 else None
    stage2_override = override if stage_index == 2 else None
    stage3_override = override if stage_index == 3 else None
    stage4_override = override if stage_index == 4 else None
    value, _record = _rk4_step(
        field,
        record.phi,
        record.state,
        record.step,
        integrator_options=integrator_options,
        stage1_override=stage1_override,
        stage2_override=stage2_override,
        stage3_override=stage3_override,
        stage4_override=stage4_override,
    )
    return value


def _biot_savart_derivative_from_cotangents(
    field: BiotSavart,
    points: Sequence[np.ndarray],
    b_cotangents: Sequence[np.ndarray],
    grad_b_cotangents: Sequence[np.ndarray],
) -> tuple[Derivative, _CotangentStats]:
    if len(points) == 0:
        return Derivative(), _CotangentStats(
            point_count=0,
            b_cotangent_norm=0.0,
            grad_b_cotangent_norm=0.0,
        )
    point_array = np.asarray(points, dtype=float)
    b_cotangent_array = np.asarray(b_cotangents, dtype=float)
    grad_b_cotangent_array = np.asarray(grad_b_cotangents, dtype=float)
    field.set_points(point_array)
    dB_part, dgradB_part = field.B_and_dB_vjp(
        b_cotangent_array,
        grad_b_cotangent_array,
    )
    derivative = dB_part + dgradB_part
    return derivative, _CotangentStats(
        point_count=int(point_array.shape[0]),
        b_cotangent_norm=float(np.linalg.norm(b_cotangent_array)),
        grad_b_cotangent_norm=float(np.linalg.norm(grad_b_cotangent_array)),
    )


def _vjp_diagnostic_from_parts(
    *,
    orbit: PeriodicOrbitResult,
    derivative: Derivative,
    gradient: np.ndarray,
    dresidue_dstate: np.ndarray,
    implicit_adjoint: np.ndarray,
    cotangent_stats: _CotangentStats,
    gradient_status: str,
) -> BiotSavartBranchResidueVjpDiagnostic:
    return BiotSavartBranchResidueVjpDiagnostic(
        mode=BIOT_SAVART_BRANCH_RESOLVED_VJP_MODE,
        gradient_status=gradient_status,
        residue=float(orbit.residue_diagnostic.residue),
        base_status=orbit.status,
        state=orbit.state,
        final_state=_normalize_state(orbit.tangent_result.return_map.final_state),
        closure_residual=orbit.closure_residual,
        monodromy=_monodromy_matrix(orbit.tangent_result.monodromy),
        trace_m=float(orbit.tangent_result.trace_m),
        det_m=float(orbit.tangent_result.det_m),
        dresidue_dstate=(float(dresidue_dstate[0]), float(dresidue_dstate[1])),
        implicit_adjoint=(float(implicit_adjoint[0]), float(implicit_adjoint[1])),
        cotangent_point_count=cotangent_stats.point_count,
        b_cotangent_norm=cotangent_stats.b_cotangent_norm,
        grad_b_cotangent_norm=cotangent_stats.grad_b_cotangent_norm,
        gradient_norm=float(np.linalg.norm(gradient)),
        gradient=tuple(float(component) for component in gradient),
        derivative=derivative,
    )


def _rk4_periodic_orbit_result(
    *,
    target: RationalTarget,
    branch: str,
    chart: PoincareChart,
    initial_state: SectionState,
    state: np.ndarray,
    closure_residual: np.ndarray,
    closure_residual_norm: float,
    iterations: int,
    status: str,
    tangent_result: FieldlineTangentReturnResult,
) -> PeriodicOrbitResult:
    state_tuple = (float(state[0]), float(state[1]))
    return PeriodicOrbitResult(
        target=target,
        branch=branch,
        initial_state=initial_state,
        state=state_tuple,
        closure_residual=(float(closure_residual[0]), float(closure_residual[1])),
        closure_residual_norm=float(closure_residual_norm),
        iterations=int(iterations),
        status=status,
        tangent_result=tangent_result,
        residue_diagnostic=greene_residue_diagnostic_from_matrix(
            tangent_result.monodromy
        ),
        winding=tangent_result.return_map.winding,
        radial_label=chart.radial_label(state_tuple),
        min_bphi_over_b=tangent_result.return_map.min_bphi_over_b,
    )


def _rk4_branch_status(
    *,
    target: RationalTarget,
    branch: str,
    chart: PoincareChart,
    tangent_result: FieldlineTangentReturnResult,
    solver_options: PeriodicOrbitSolverOptions,
) -> str:
    return_map = tangent_result.return_map
    radial_label = chart.radial_label(return_map.initial_state)
    if not radial_label_in_target_window(
        radial_label,
        target,
        tolerance=solver_options.radial_window_tolerance,
    ):
        return BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW
    if (
        abs(target_winding_residual(return_map, target))
        > solver_options.winding_tolerance
    ):
        return BRANCH_STATUS_WRONG_WINDING
    if not tangent_map_determinant_within_tolerance(
        tangent_result,
        solver_options,
    ):
        return BRANCH_STATUS_BAD_DETERMINANT
    diagnostic = greene_residue_diagnostic_from_matrix(tangent_result.monodromy)
    if not residue_classification_matches_branch(branch, diagnostic):
        return BRANCH_STATUS_BRANCH_MISMATCH
    return BRANCH_STATUS_CONVERGED


def _damped_rk4_newton_state(
    field: DifferentiableMagneticFieldLike,
    state: np.ndarray,
    step: np.ndarray,
    *,
    current_residual_norm: float,
    target: RationalTarget,
    chart: PoincareChart,
    integrator_options: FieldlineIntegratorOptions,
    solver_options: PeriodicOrbitSolverOptions,
) -> np.ndarray | None:
    damping = 1.0
    while damping >= solver_options.min_damping:
        candidate = state + damping * step
        if candidate[0] > 0.0 and np.all(np.isfinite(candidate)):
            tangent_result, _records = _rk4_tangent_return_map(
                field,
                candidate,
                target=target,
                chart=chart,
                integrator_options=integrator_options,
                with_tape=False,
            )
            candidate_residual = (
                np.asarray(tangent_result.return_map.final_state, dtype=float)
                - candidate
            )
            if float(np.linalg.norm(candidate_residual)) < current_residual_norm:
                return candidate
        damping *= solver_options.damping_shrink
    return None


def _trust_region_step(
    jacobian: np.ndarray,
    closure_residual: np.ndarray,
    *,
    max_step_norm: float,
) -> np.ndarray:
    step = np.linalg.lstsq(jacobian, -closure_residual, rcond=None)[0]
    step_norm = float(np.linalg.norm(step))
    if step_norm > float(max_step_norm):
        return step * (float(max_step_norm) / step_norm)
    return step


def _validate_branch_for_target(target: RationalTarget, branch: str) -> str:
    branch_label = str(branch)
    if branch_label not in target.branches:
        raise ValueError(
            f"Greene branch {branch_label} is not enabled for target "
            f"{target.manifest_key()}"
        )
    return branch_label


def _bphi_over_b(b_vector: np.ndarray, phi: float) -> float:
    _e_r, e_phi, _e_z = cylindrical_basis(float(phi))
    b_norm = float(np.linalg.norm(b_vector))
    if b_norm <= 0.0:
        raise ValueError("Magnetic field norm must be positive")
    return abs(float(np.dot(b_vector, e_phi))) / b_norm


def _scaled_difference_step(value: float, local_difference_step: float) -> float:
    return float(local_difference_step) * max(1.0, abs(float(value)))


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


def _normalized_probe_steps(probe_steps: Sequence[float]) -> tuple[float, ...]:
    steps = tuple(_positive_step(step) for step in probe_steps)
    if len(steps) < 2:
        raise ValueError(
            "BiotSavart residue Taylor diagnostic requires at least two probe steps"
        )
    return steps


def _observed_orders(
    samples: tuple[BiotSavartResidueTaylorSample, ...],
) -> tuple[float, ...]:
    orders: list[float] = []
    for previous, current in zip(samples[:-1], samples[1:], strict=True):
        if previous.absolute_residual <= 0.0 or current.absolute_residual <= 0.0:
            raise ValueError("Taylor residuals must be positive to estimate order")
        orders.append(
            float(
                np.log(previous.absolute_residual / current.absolute_residual)
                / np.log(previous.step / current.step)
            )
        )
    return tuple(orders)


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
