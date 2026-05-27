from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

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


def _normalize_state(state: Sequence[float]) -> tuple[float, float]:
    if len(state) != 2:
        raise ValueError("Residue sensitivity state must be [R, Z]")
    return (float(state[0]), float(state[1]))


def _positive_step(step: float) -> float:
    step_value = float(step)
    if step_value <= 0.0:
        raise ValueError("Residue sensitivity finite-difference step must be positive")
    return step_value


def _require_converged(result: PeriodicOrbitResult, *, label: str) -> None:
    if result.status != BRANCH_STATUS_CONVERGED:
        raise ValueError(
            f"Branch-resolved residue finite difference requires {label} branch "
            f"status {BRANCH_STATUS_CONVERGED}, got {result.status}"
        )
