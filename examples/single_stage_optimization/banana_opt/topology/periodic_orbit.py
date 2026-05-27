from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

import numpy as np

from .fieldline_map import (
    DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    DifferentiableMagneticFieldLike,
    FieldlineIntegratorOptions,
    FieldlineTangentReturnResult,
    integrate_tangent_target_return_map,
    target_winding_residual,
)
from .greene_residue import (
    GREENE_RESIDUE_ELLIPTIC_O,
    GREENE_RESIDUE_HYPERBOLIC_X,
    GreeneResidueDiagnostic,
    greene_residue_diagnostic_from_matrix,
)
from .poincare_chart import PoincareChart
from .rational_target import (
    GREENE_BRANCHES,
    GREENE_BRANCH_O,
    RationalTarget,
)


BRANCH_STATUS_CONVERGED = "converged"
BRANCH_STATUS_MAX_ITERATIONS = "max_iterations"
BRANCH_STATUS_NEWTON_STALLED = "newton_stalled"
BRANCH_STATUS_OUTSIDE_RADIAL_WINDOW = "outside_radial_window"
BRANCH_STATUS_WRONG_WINDING = "wrong_winding"
BRANCH_STATUS_BRANCH_MISMATCH = "branch_mismatch"
BRANCH_STATUS_BAD_DETERMINANT = "bad_tangent_determinant"


@dataclass(frozen=True, slots=True)
class BranchState:
    """Accepted periodic branch state used as the next continuation seed."""

    target_id: str
    branch: str
    section_state: tuple[float, float]
    winding: float
    radial_label: float
    residue: float
    status: str
    generation: int = 0
    accepted_iteration_id: str = ""

    def __post_init__(self) -> None:
        target_id = str(self.target_id)
        branch = str(self.branch)
        if target_id == "":
            raise ValueError("BranchState target_id must be nonempty")
        if branch not in GREENE_BRANCHES:
            raise ValueError(f"Unknown Greene residue branch label: {branch}")
        object.__setattr__(self, "target_id", target_id)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(
            self,
            "section_state",
            _normalize_solver_state("BranchState section_state", self.section_state),
        )
        for field_name in ("winding", "radial_label", "residue"):
            value = float(getattr(self, field_name))
            if not isfinite(value):
                raise ValueError(f"BranchState {field_name} must be finite")
            object.__setattr__(self, field_name, value)
        status = str(self.status)
        if status != BRANCH_STATUS_CONVERGED:
            raise ValueError("BranchState status must be converged")
        generation = int(self.generation)
        if generation < 0:
            raise ValueError("BranchState generation must be nonnegative")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "generation", generation)
        object.__setattr__(
            self,
            "accepted_iteration_id",
            str(self.accepted_iteration_id),
        )


@dataclass(frozen=True, slots=True)
class PeriodicOrbitSolverOptions:
    """Newton policy for fixed-rational branch solves of P_c^q(z)-z=0."""

    residual_tolerance: float = 1.0e-9
    winding_tolerance: float = 1.0e-7
    radial_window_tolerance: float = 1.0e-8
    det_tolerance: float = 1.0e-5
    max_iterations: int = 12
    max_step_norm: float = 0.05
    min_damping: float = 0.0625
    damping_shrink: float = 0.5

    def __post_init__(self) -> None:
        residual_tolerance = float(self.residual_tolerance)
        winding_tolerance = float(self.winding_tolerance)
        radial_window_tolerance = float(self.radial_window_tolerance)
        det_tolerance = float(self.det_tolerance)
        max_step_norm = float(self.max_step_norm)
        min_damping = float(self.min_damping)
        damping_shrink = float(self.damping_shrink)
        max_iterations = int(self.max_iterations)
        if residual_tolerance <= 0.0 or not isfinite(residual_tolerance):
            raise ValueError("Periodic orbit residual_tolerance must be positive")
        if winding_tolerance <= 0.0 or not isfinite(winding_tolerance):
            raise ValueError("Periodic orbit winding_tolerance must be positive")
        if radial_window_tolerance < 0.0 or not isfinite(radial_window_tolerance):
            raise ValueError(
                "Periodic orbit radial_window_tolerance must be finite and nonnegative"
            )
        if det_tolerance < 0.0 or not isfinite(det_tolerance):
            raise ValueError(
                "Periodic orbit det_tolerance must be finite and nonnegative"
            )
        if max_iterations <= 0:
            raise ValueError("Periodic orbit max_iterations must be positive")
        if max_step_norm <= 0.0 or not isfinite(max_step_norm):
            raise ValueError("Periodic orbit max_step_norm must be positive")
        if min_damping <= 0.0 or min_damping > 1.0 or not isfinite(min_damping):
            raise ValueError("Periodic orbit min_damping must be in (0, 1]")
        if damping_shrink <= 0.0 or damping_shrink >= 1.0:
            raise ValueError("Periodic orbit damping_shrink must be in (0, 1)")
        object.__setattr__(self, "residual_tolerance", residual_tolerance)
        object.__setattr__(self, "winding_tolerance", winding_tolerance)
        object.__setattr__(self, "radial_window_tolerance", radial_window_tolerance)
        object.__setattr__(self, "det_tolerance", det_tolerance)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "max_step_norm", max_step_norm)
        object.__setattr__(self, "min_damping", min_damping)
        object.__setattr__(self, "damping_shrink", damping_shrink)


DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS = PeriodicOrbitSolverOptions()


@dataclass(frozen=True, slots=True)
class PeriodicOrbitResult:
    target: RationalTarget
    branch: str
    initial_state: tuple[float, float]
    state: tuple[float, float]
    closure_residual: tuple[float, float]
    closure_residual_norm: float
    iterations: int
    status: str
    tangent_result: FieldlineTangentReturnResult
    residue_diagnostic: GreeneResidueDiagnostic
    winding: float
    radial_label: float
    min_bphi_over_b: float

    @property
    def converged(self) -> bool:
        return self.status == BRANCH_STATUS_CONVERGED

    def to_branch_state(
        self,
        *,
        generation: int,
        accepted_iteration_id: str = "",
    ) -> BranchState:
        if not self.converged:
            raise ValueError("Only converged periodic-orbit results can be continued")
        return BranchState(
            target_id=self.target.manifest_key(),
            branch=self.branch,
            section_state=self.state,
            winding=self.winding,
            radial_label=self.radial_label,
            residue=self.residue_diagnostic.residue,
            status=self.status,
            generation=generation,
            accepted_iteration_id=accepted_iteration_id,
        )


def residue_classification_matches_branch(
    branch: str,
    diagnostic: GreeneResidueDiagnostic,
) -> bool:
    branch_label = _normalize_branch_label(branch)
    if branch_label == GREENE_BRANCH_O:
        return diagnostic.classification == GREENE_RESIDUE_ELLIPTIC_O
    return diagnostic.classification == GREENE_RESIDUE_HYPERBOLIC_X


def radial_label_in_target_window(
    radial_label: float,
    target: RationalTarget,
    *,
    tolerance: float,
) -> bool:
    label = float(radial_label)
    if not isfinite(label):
        raise ValueError("Periodic orbit radial label must be finite")
    radial_window = target.radial_window
    if radial_window is None:
        return True
    window_tolerance = float(tolerance)
    return (
        radial_window[0] - window_tolerance
        <= label
        <= radial_window[1] + window_tolerance
    )


def tangent_map_determinant_within_tolerance(
    tangent_result: FieldlineTangentReturnResult,
    solver_options: PeriodicOrbitSolverOptions,
) -> bool:
    det_m = float(tangent_result.det_m)
    return isfinite(det_m) and abs(det_m - 1.0) <= solver_options.det_tolerance


def solve_periodic_orbit(
    field: DifferentiableMagneticFieldLike,
    initial_state: Sequence[float],
    *,
    target: RationalTarget,
    chart: PoincareChart,
    branch: str,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
) -> PeriodicOrbitResult:
    branch_label = _validate_target_branch(target, branch)
    start_state = _normalize_solver_state("periodic orbit initial_state", initial_state)
    state_array = np.asarray(start_state, dtype=float)

    for iteration in range(solver_options.max_iterations + 1):
        tangent_result = integrate_tangent_target_return_map(
            field,
            state_array,
            target=target,
            chart=chart,
            options=integrator_options,
        )
        closure_residual = _closure_residual(tangent_result, state_array)
        closure_residual_norm = float(np.linalg.norm(closure_residual))
        if closure_residual_norm <= solver_options.residual_tolerance:
            return _periodic_orbit_result(
                target=target,
                branch=branch_label,
                chart=chart,
                initial_state=start_state,
                state=state_array,
                closure_residual=closure_residual,
                closure_residual_norm=closure_residual_norm,
                iterations=iteration,
                status=_validated_branch_status(
                    target=target,
                    branch=branch_label,
                    chart=chart,
                    tangent_result=tangent_result,
                    solver_options=solver_options,
                ),
                tangent_result=tangent_result,
            )
        if iteration == solver_options.max_iterations:
            return _periodic_orbit_result(
                target=target,
                branch=branch_label,
                chart=chart,
                initial_state=start_state,
                state=state_array,
                closure_residual=closure_residual,
                closure_residual_norm=closure_residual_norm,
                iterations=iteration,
                status=BRANCH_STATUS_MAX_ITERATIONS,
                tangent_result=tangent_result,
            )

        jacobian = np.asarray(tangent_result.monodromy, dtype=float) - np.eye(2)
        step = _trust_region_newton_step(
            jacobian,
            closure_residual,
            max_step_norm=solver_options.max_step_norm,
        )
        next_state = _damped_newton_state(
            field,
            state_array,
            step,
            current_residual_norm=closure_residual_norm,
            target=target,
            chart=chart,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
        if next_state is None:
            return _periodic_orbit_result(
                target=target,
                branch=branch_label,
                chart=chart,
                initial_state=start_state,
                state=state_array,
                closure_residual=closure_residual,
                closure_residual_norm=closure_residual_norm,
                iterations=iteration,
                status=BRANCH_STATUS_NEWTON_STALLED,
                tangent_result=tangent_result,
            )
        state_array = next_state

    raise RuntimeError("Periodic orbit solver loop exited unexpectedly")


def discover_periodic_orbit(
    field: DifferentiableMagneticFieldLike,
    initial_guesses: Sequence[Sequence[float]],
    *,
    target: RationalTarget,
    chart: PoincareChart,
    branch: str,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
) -> PeriodicOrbitResult:
    guesses = tuple(initial_guesses)
    if len(guesses) == 0:
        raise ValueError("Periodic orbit discovery requires at least one initial guess")

    best_result = solve_periodic_orbit(
        field,
        guesses[0],
        target=target,
        chart=chart,
        branch=branch,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )
    if best_result.converged:
        return best_result
    for initial_guess in guesses[1:]:
        result = solve_periodic_orbit(
            field,
            initial_guess,
            target=target,
            chart=chart,
            branch=branch,
            integrator_options=integrator_options,
            solver_options=solver_options,
        )
        if result.converged:
            return result
        if result.closure_residual_norm < best_result.closure_residual_norm:
            best_result = result
    return best_result


def continue_periodic_orbit(
    field: DifferentiableMagneticFieldLike,
    previous_branch_state: BranchState,
    *,
    target: RationalTarget,
    chart: PoincareChart,
    integrator_options: FieldlineIntegratorOptions = DEFAULT_FIELDLINE_INTEGRATOR_OPTIONS,
    solver_options: PeriodicOrbitSolverOptions = DEFAULT_PERIODIC_ORBIT_SOLVER_OPTIONS,
) -> PeriodicOrbitResult:
    branch_label = _validate_target_branch(target, previous_branch_state.branch)
    if previous_branch_state.target_id != target.manifest_key():
        raise ValueError("Cannot continue a Greene branch state across targets")
    return solve_periodic_orbit(
        field,
        previous_branch_state.section_state,
        target=target,
        chart=chart,
        branch=branch_label,
        integrator_options=integrator_options,
        solver_options=solver_options,
    )


def _normalize_solver_state(
    label: str,
    state: Sequence[float],
) -> tuple[float, float]:
    if len(state) != 2:
        raise ValueError(f"{label} must be [R, Z]")
    radius = float(state[0])
    z = float(state[1])
    if radius <= 0.0 or not isfinite(radius) or not isfinite(z):
        raise ValueError(f"{label} requires finite R > 0 and finite Z")
    return (radius, z)


def _normalize_branch_label(branch: str) -> str:
    branch_label = str(branch)
    if branch_label not in GREENE_BRANCHES:
        raise ValueError(f"Unknown Greene residue branch label: {branch_label}")
    return branch_label


def _validate_target_branch(target: RationalTarget, branch: str) -> str:
    branch_label = _normalize_branch_label(branch)
    if branch_label not in target.branches:
        raise ValueError(
            f"Greene branch {branch_label} is not enabled for target {target.manifest_key()}"
        )
    return branch_label


def _closure_residual(
    tangent_result: FieldlineTangentReturnResult,
    state: np.ndarray,
) -> np.ndarray:
    return np.asarray(tangent_result.return_map.final_state, dtype=float) - state


def _trust_region_newton_step(
    jacobian: np.ndarray,
    closure_residual: np.ndarray,
    *,
    max_step_norm: float,
) -> np.ndarray:
    step = np.linalg.lstsq(jacobian, -closure_residual, rcond=None)[0]
    step_norm = float(np.linalg.norm(step))
    if step_norm > max_step_norm:
        return step * (max_step_norm / step_norm)
    return step


def _damped_newton_state(
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
        candidate_state = state + damping * step
        if candidate_state[0] > 0.0 and np.all(np.isfinite(candidate_state)):
            tangent_result = integrate_tangent_target_return_map(
                field,
                candidate_state,
                target=target,
                chart=chart,
                options=integrator_options,
            )
            candidate_residual_norm = float(
                np.linalg.norm(_closure_residual(tangent_result, candidate_state))
            )
            if candidate_residual_norm < current_residual_norm:
                return candidate_state
        damping *= solver_options.damping_shrink
    return None


def _validated_branch_status(
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
    residue_diagnostic = greene_residue_diagnostic_from_matrix(tangent_result.monodromy)
    if not residue_classification_matches_branch(branch, residue_diagnostic):
        return BRANCH_STATUS_BRANCH_MISMATCH
    return BRANCH_STATUS_CONVERGED


def _periodic_orbit_result(
    *,
    target: RationalTarget,
    branch: str,
    chart: PoincareChart,
    initial_state: tuple[float, float],
    state: np.ndarray,
    closure_residual: np.ndarray,
    closure_residual_norm: float,
    iterations: int,
    status: str,
    tangent_result: FieldlineTangentReturnResult,
) -> PeriodicOrbitResult:
    state_tuple = (float(state[0]), float(state[1]))
    residue_diagnostic = greene_residue_diagnostic_from_matrix(tangent_result.monodromy)
    return PeriodicOrbitResult(
        target=target,
        branch=branch,
        initial_state=initial_state,
        state=state_tuple,
        closure_residual=(float(closure_residual[0]), float(closure_residual[1])),
        closure_residual_norm=float(closure_residual_norm),
        iterations=iterations,
        status=status,
        tangent_result=tangent_result,
        residue_diagnostic=residue_diagnostic,
        winding=tangent_result.return_map.winding,
        radial_label=chart.radial_label(state_tuple),
        min_bphi_over_b=tangent_result.return_map.min_bphi_over_b,
    )
