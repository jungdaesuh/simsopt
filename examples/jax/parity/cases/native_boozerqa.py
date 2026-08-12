"""Exact matched workflow for ``2_Intermediate/boozerQA.py``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, TypeVar, cast

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.measurement import MeasurementExecution
from examples.jax.parity.runtime import ParityLane
from simsopt.optimization_endpoint import (
    OptimizationEndpointCertificate,
    certify_optimization_endpoint,
)
from simsopt.optimization_trajectory import (
    OptimizationMeasurementWindow,
    OptimizationTrajectoryRecorder,
)
from simsopt.single_stage_boozer_vacuum import (
    JAX_FAST_DRIVER_ID,
    JAX_OPTAX_DRIVER_ID,
    JAX_PARITY_DRIVER_ID,
    OUTER_GRADIENT_TOLERANCE,
)
from simsopt_jax.examples import ExecutionScale, scalar_example_driver
from simsopt_jax.solve.driver import Driver

WORKFLOW_STAGES = (
    "construct_ncsx_coils_and_volume_labelled_surface",
    "solve_initial_boozer_surface",
    "assemble_nonqs_iota_radius_and_length_objective",
    "evaluate_initial_objective_and_gradient",
    "optimize_coils_and_currents_with_bfgs",
    "record_final_objective_gradient_and_physics_terms",
)

_TIMELINE_PHYSICS_OBSERVABLES = (
    "objective",
    "iota",
    "volume",
    "non_qs_ratio",
    "boozer_residual",
)
_TIMELINE_UNAVAILABLE_PHYSICS_OBSERVABLES = tuple(
    (name, None) for name in _TIMELINE_PHYSICS_OBSERVABLES
)


@dataclass(frozen=True, slots=True)
class BoozerSingleStageSpec:
    """Immutable scientific differences between Boozer single-stage examples."""

    case_id: str
    workflow_stages: tuple[str, ...]
    bounded_resolution: int
    native_resolution: int
    inner_tolerance: float
    bounded_outer_maxiter: int
    native_outer_maxiter: int
    bounded_non_qs_sdim: int
    native_non_qs_sdim: int
    residual_weight: float
    report_residual: bool
    enforce_endpoint_certificate: bool = False


BOOZER_QA_SPEC = BoozerSingleStageSpec(
    case_id="native-boozerqa",
    workflow_stages=WORKFLOW_STAGES,
    bounded_resolution=2,
    native_resolution=6,
    inner_tolerance=1.0e-10,
    bounded_outer_maxiter=5,
    native_outer_maxiter=1_000,
    bounded_non_qs_sdim=20,
    native_non_qs_sdim=20,
    residual_weight=0.0,
    report_residual=False,
)


@dataclass(frozen=True, slots=True)
class ChangedStateTimelineObservation:
    """Immutable numerical evidence from one already-completed evaluation."""

    evaluation_id: str
    evaluation_kind: str
    outer_iteration_id: int | None
    parameter_sha256: str
    parameters: tuple[float, ...]
    parameter_shape: tuple[int, ...]
    objective: float
    gradient: tuple[float, ...]
    gradient_dtype: str
    gradient_shape: tuple[int, ...]
    values_finite: bool
    forward_success: bool
    primal_success: bool
    actual_adjoint_success: bool
    gradient_source: str
    candidate_gradient_source: bool
    eligible: bool
    inner_residual_trace: tuple[float, ...]
    newton_step_accepted_trace: tuple[bool, ...]
    newton_linear_solve_success_trace: tuple[bool, ...]
    newton_iterations: int
    newton_attempted_iterations: int | None
    newton_trace_available: bool
    adjoint_route: str
    adjoint_output: tuple[float, ...]
    adjoint_residual: float
    adjoint_residual_relative: float
    dense_materializations: int
    lu_factorizations: int
    lu_solves: int
    refinement_corrections: int
    adjoint_executions: int
    observables: tuple[tuple[str, float | None], ...]


@dataclass(frozen=True, slots=True)
class ChangedStateTimelineDisposition:
    """Immutable optimizer decision joined to an existing evaluation."""

    evaluation_id: str
    parameter_sha256: str
    disposition: str
    accepted_iteration_id: int | None


ChangedStateTimelineRecord = (
    ChangedStateTimelineObservation | ChangedStateTimelineDisposition
)


@dataclass(frozen=True, slots=True)
class _PreparedJaxRuntime:
    """One constructed session and its exact reusable compiled callables."""

    session: object
    runtime: Mapping[str, object]
    reporting: Callable[..., Mapping[str, object]]
    value_and_grad: Callable[..., object]
    initial_parameters: np.ndarray
    initial_inner_success: bool
    iota_target: float
    initial_volume: float
    optimizer_backend: str | None
    incumbent_evaluator: Callable[..., object]
    incumbent_factory: Callable[[], object]
    exact_newton_variant: Literal["C0", "C1", "C2"]

    def fresh_incumbent_controller(self) -> object:
        """Mint isolated continuation state while retaining compiled callables."""

        return self.incumbent_factory()


class _NativeSurface(Protocol):
    x: np.ndarray

    def get_dofs(self) -> np.ndarray: ...


class _NativeSolver(Protocol):
    surface: _NativeSurface
    res: dict[str, object]


class _NativeObjective(Protocol):
    x: np.ndarray

    def J(self) -> float: ...

    def dJ(self) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class NativeBaselineAnchor:
    """Immutable identity and targets of the solved native baseline anchor."""

    parameter_sha256: str
    surface_sha256: str
    iota: float
    G: float
    iota_target: float
    volume_target: float
    major_radius_target: float
    total_length_target: float
    inner_solver_success: bool


@dataclass(frozen=True, slots=True)
class NativeCandidateEvaluation:
    """One native objective/gradient result at an explicitly supplied candidate."""

    objective: float
    gradient: np.ndarray
    inner_solver_success: bool
    solver_residual_l2: float
    solver_residual_inf: float


@dataclass(frozen=True, slots=True)
class _PreparedNativeRuntime:
    """Canonical native objective, mutable continuation state, and baseline anchor."""

    solver: _NativeSolver
    objective: _NativeObjective
    non_qs: _NativeObjective
    residual: _NativeObjective
    volume: _NativeObjective
    radius_penalty: _NativeObjective
    length_penalty: _NativeObjective
    initial_parameters: np.ndarray
    initial_solution_success: bool
    initial_iota: float
    initial_volume: float
    baseline_anchor: NativeBaselineAnchor

    def value_and_grad(self, parameters: np.ndarray) -> tuple[float, np.ndarray]:
        """Evaluate with the production native rollback behavior."""

        previous_surface = np.asarray(self.solver.surface.x, dtype=np.float64)
        previous_iota = float(self.solver.res["iota"])
        previous_G = float(self.solver.res["G"])
        self.objective.x = parameters
        value = float(self.objective.J())
        gradient = np.asarray(self.objective.dJ(), dtype=np.float64)
        if not bool(self.solver.res["success"]):
            value = 1.0e3
            self.solver.surface.x = previous_surface
            self.solver.res["iota"] = previous_iota
            self.solver.res["G"] = previous_G
        return value, gradient

    def evaluate_candidate(self, parameters: np.ndarray) -> NativeCandidateEvaluation:
        """Evaluate one outer candidate and report its native inner-solve residual."""

        objective, gradient = self.value_and_grad(parameters)
        solver_residual = np.asarray(
            self.solver.res["residual"], dtype=np.float64
        ).reshape(-1)
        return NativeCandidateEvaluation(
            objective=objective,
            gradient=gradient,
            inner_solver_success=bool(self.solver.res["success"]),
            solver_residual_l2=float(np.linalg.norm(solver_residual)),
            solver_residual_inf=float(np.linalg.norm(solver_residual, ord=np.inf)),
        )


@dataclass(frozen=True, slots=True)
class _PreparedJaxVariantExecution:
    """Prepared case execution that mints fresh optimizer state per invocation."""

    lane: ParityLane
    bundle: InputBundle
    arrays: dict[str, np.ndarray]
    spec: BoozerSingleStageSpec
    _runtime: _PreparedJaxRuntime

    @property
    def compilation_identity(self) -> tuple[int, int, int, int]:
        """Identity proof for the reused session, runtime, and value/grad callable."""

        return (
            id(self._runtime.session),
            id(self._runtime.runtime),
            id(self._runtime.value_and_grad),
            id(self._runtime.incumbent_evaluator),
        )

    def execute(self, measurement: MeasurementExecution) -> LaneObservation:
        """Run from a fresh incumbent controller on the prepared runtime."""

        if measurement.optimizer_backend != self._runtime.optimizer_backend:
            raise ValueError("prepared execution optimizer backend cannot change")
        return _jax(
            self.lane,
            self.bundle,
            self.arrays,
            self.spec,
            measurement,
            prepared=self._runtime,
        )


class _TimelineEvaluationKind(Protocol):
    value: str


class _TimelineTraceContext(Protocol):
    evaluation_id: str
    kind: _TimelineEvaluationKind
    outer_iteration_id: int | None
    parameter_sha256: str


class _TimelineExecutionCounts(Protocol):
    dense_materialization_count: object
    lu_factorization_count: object
    lu_solve_count: object
    refinement_correction_count: object
    adjoint_execution_count: object


class _PreparedIncumbentPrototype(Protocol):
    _compiled_evaluate: Callable[..., object]

    @property
    def current_inner_state(self) -> object: ...


@dataclass(frozen=True, slots=True)
class _AcceptedTimelineValueAndGradient:
    """Exact host value and gradient from the accepted timeline trial."""

    parameter_sha256: str
    value: float
    gradient: np.ndarray


@dataclass(frozen=True, slots=True)
class _DeferredChangedStateTimelineObservation:
    """Device evidence retained until the measured optimization has ended."""

    trace_context: _TimelineTraceContext
    parameters: tuple[float, ...]
    parameter_shape: tuple[int, ...]
    objective: float
    gradient: tuple[float, ...]
    gradient_shape: tuple[int, ...]
    forward_success: object
    primal_success: object
    actual_adjoint_success: object
    gradient_source: str
    candidate_gradient_source: bool
    eligible: object
    newton_iterations: object
    newton_attempted_iterations: object
    newton_trace_active_present: object
    inner_residual_trace_present: object
    newton_step_accepted_trace_present: object
    newton_linear_solve_success_trace_present: object
    newton_trace_active: object
    inner_residual_trace: object
    newton_step_accepted_trace: object
    newton_linear_solve_success_trace: object
    adjoint_output: object
    adjoint_residual: object
    adjoint_residual_relative: object
    execution_counts: _TimelineExecutionCounts
    observables: tuple[tuple[str, object | None], ...]


def _optimizer_final_value_and_gradient(
    parameters: np.ndarray,
    *,
    timeline_enabled: bool,
    accepted_timeline_evaluation: _AcceptedTimelineValueAndGradient | None,
    parameter_sha256: Callable[[np.ndarray], str],
    evaluate: Callable[[np.ndarray], tuple[float, np.ndarray]],
) -> tuple[float, np.ndarray]:
    """Reuse the accepted timeline result, or preserve ordinary evaluation."""

    if not timeline_enabled:
        return evaluate(parameters)
    if accepted_timeline_evaluation is None:
        raise RuntimeError("timeline optimizer final check has no accepted evaluation")
    if parameter_sha256(parameters) != accepted_timeline_evaluation.parameter_sha256:
        raise RuntimeError(
            "optimizer final parameters do not match the accepted timeline evaluation"
        )
    return (
        accepted_timeline_evaluation.value,
        accepted_timeline_evaluation.gradient,
    )


_TIMELINE_OBSERVATION_SINK: ContextVar[
    Callable[[ChangedStateTimelineRecord], None] | None
] = ContextVar("simsopt_jax_changed_state_timeline_observation_sink", default=None)


@contextmanager
def changed_state_timeline_observation_sink(
    sink: Callable[[ChangedStateTimelineRecord], None],
) -> Iterator[None]:
    """Collect already-materialized evaluation values without reevaluation."""

    token = _TIMELINE_OBSERVATION_SINK.set(sink)
    try:
        yield
    finally:
        _TIMELINE_OBSERVATION_SINK.reset(token)


_InitialEvaluation = TypeVar("_InitialEvaluation")


@contextmanager
def _measurement_optimization_window(
    measurement: MeasurementExecution | None,
    evaluate_initial: Callable[[], _InitialEvaluation],
) -> Iterator[tuple[_InitialEvaluation, OptimizationTrajectoryRecorder | None]]:
    """Start trajectory time before the lane's required initial evaluation."""
    with OptimizationMeasurementWindow(
        trajectory_path=(
            measurement.trajectory_path if measurement is not None else None
        ),
        timing_path=(
            measurement.optimization_timing_path if measurement is not None else None
        ),
    ) as trajectory:
        yield evaluate_initial(), trajectory


def _scale_configuration(
    scale: ExecutionScale,
    spec: BoozerSingleStageSpec,
) -> dict[str, object]:
    native_scale = scale == "native_default"
    resolution = spec.native_resolution if native_scale else spec.bounded_resolution
    return {
        "mpol": resolution,
        "ntor": resolution,
        "inner_maxiter": 20,
        "inner_tolerance": spec.inner_tolerance,
        "outer_maxiter": (
            spec.native_outer_maxiter if native_scale else spec.bounded_outer_maxiter
        ),
        "outer_rtol": 0.0,
        "outer_atol": OUTER_GRADIENT_TOLERANCE,
        "initial_iota": -0.406,
        "surface_distance": 0.10,
        "non_qs_sdim": (
            spec.native_non_qs_sdim if native_scale else spec.bounded_non_qs_sdim
        ),
        "residual_weight": spec.residual_weight,
        "report_residual": spec.report_residual,
        "reduced_coil_order": 3,
        "reduced_axis_order": 3,
        "reduced_points_per_period": 8,
    }


def _configuration_int(configuration: Mapping[str, object], name: str) -> int:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _configuration_float(configuration: Mapping[str, object], name: str) -> float:
    value = configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _problem(configuration: Mapping[str, object], scale: ExecutionScale):
    from simsopt.configs import get_data
    from simsopt.geo import SurfaceXYZTensorFourier

    options = (
        {}
        if scale == "native_default"
        else {
            "coil_order": _configuration_int(configuration, "reduced_coil_order"),
            "magnetic_axis_order": _configuration_int(
                configuration, "reduced_axis_order"
            ),
            "points_per_period": _configuration_int(
                configuration, "reduced_points_per_period"
            ),
        }
    )
    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data(
        "ncsx",
        **options,
    )
    base_currents[0].fix_all()
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))
    mpol = _configuration_int(configuration, "mpol")
    ntor = _configuration_int(configuration, "ntor")
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(
            0.0,
            1.0 / nfp,
            2 * ntor + 1,
            endpoint=False,
        ),
        quadpoints_theta=np.linspace(
            0.0,
            1.0,
            2 * mpol + 1,
            endpoint=False,
        ),
    )
    surface.fit_to_curve(
        magnetic_axis,
        _configuration_float(configuration, "surface_distance"),
        flip_theta=True,
    )
    return (
        base_curves,
        base_currents,
        magnetic_axis,
        int(nfp),
        native_field,
        surface,
        float(G0),
    )


def create_variant_input(
    root: Path,
    scale: ExecutionScale,
    spec: BoozerSingleStageSpec,
) -> InputBundle:
    """Freeze one configured Boozer single-stage problem for every lane."""
    configuration = _scale_configuration(scale, spec)
    (
        _base_curves,
        _base_currents,
        magnetic_axis,
        nfp,
        native_field,
        surface,
        G0,
    ) = _problem(configuration, scale)
    return create_input_bundle(
        root,
        case_id=spec.case_id,
        random_seed=1,
        arrays={
            "axis_dofs": np.asarray(
                magnetic_axis.local_full_x,
                dtype=np.float64,
            ),
            "coil_dofs": np.asarray(native_field.x, dtype=np.float64),
            "surface_dofs": np.asarray(surface.get_dofs(), dtype=np.float64),
        },
        configuration={
            **configuration,
            "nfp": nfp,
            "initial_G": G0,
        },
        scale=scale,
    )


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the bounded/full Boozer-QA state for every lane."""
    return create_variant_input(root, scale, BOOZER_QA_SPEC)


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()


def _canonical_fp64_digest(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_reconstructed_bundle_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    axis_dofs: np.ndarray,
    coil_dofs: np.ndarray,
    surface_dofs: np.ndarray,
) -> None:
    """Fail when reconstruction materially differs from the frozen bundle."""

    reconstructed = {
        "axis_dofs": np.asarray(axis_dofs, dtype=np.float64),
        "coil_dofs": np.asarray(coil_dofs, dtype=np.float64),
        "surface_dofs": np.asarray(surface_dofs, dtype=np.float64),
    }
    for name, expected in reconstructed.items():
        frozen = arrays.get(name)
        matches = frozen is not None and np.array_equal(frozen, expected)
        if frozen is not None and name == "surface_dofs":
            scale = max(1.0, float(np.max(np.abs(frozen))))
            matches = np.allclose(
                frozen,
                expected,
                rtol=0.0,
                atol=2.0 * np.finfo(np.float64).eps * scale,
            )
        if (
            frozen is None
            or frozen.dtype != np.dtype(np.float64)
            or frozen.shape != expected.shape
            or not matches
        ):
            raise ValueError(
                f"reconstructed {name} does not match the frozen input bundle"
            )


def _effective_fingerprint(
    bundle: InputBundle,
    surface_dofs: np.ndarray,
    coil_dofs: np.ndarray,
) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "surface_dofs": _array_digest(surface_dofs),
            "coil_dofs": _array_digest(coil_dofs),
            **bundle.configuration,
        },
    )


def _prepare_native_variant_runtime(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
) -> _PreparedNativeRuntime:
    """Construct the sole native objective and its solved baseline anchor."""

    from simsopt.field import BiotSavart
    from simsopt.geo import (
        BoozerResidual,
        BoozerSurface,
        CurveLength,
        Iotas,
        MajorRadius,
        NonQuasiSymmetricRatio,
        Volume,
    )
    from simsopt.objectives import QuadraticPenalty

    (
        base_curves,
        _base_currents,
        magnetic_axis,
        _nfp,
        native_field,
        surface,
        G0,
    ) = _problem(bundle.configuration, bundle.scale)
    _validate_reconstructed_bundle_arrays(
        arrays,
        axis_dofs=np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
        coil_dofs=np.asarray(native_field.x, dtype=np.float64),
        surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
    )
    surface.set_dofs(arrays["surface_dofs"])
    volume = Volume(surface)
    volume_target = float(volume.J())
    solver = BoozerSurface(
        native_field,
        surface,
        volume,
        volume_target,
        options={
            "newton_maxiter": _configuration_int(
                bundle.configuration,
                "inner_maxiter",
            ),
            "newton_tol": _configuration_float(
                bundle.configuration,
                "inner_tolerance",
            ),
            "verbose": False,
        },
    )
    initial_solution = solver.solve_residual_equation_exactly_newton(
        tol=_configuration_float(bundle.configuration, "inner_tolerance"),
        maxiter=_configuration_int(bundle.configuration, "inner_maxiter"),
        iota=_configuration_float(bundle.configuration, "initial_iota"),
        G=G0,
    )
    iota_target = float(initial_solution["iota"])
    initial_volume = float(volume.J())
    major_radius = MajorRadius(solver)
    lengths = [CurveLength(curve) for curve in base_curves]
    non_qs = NonQuasiSymmetricRatio(
        solver,
        BiotSavart(native_field.coils),
        sDIM=_configuration_int(bundle.configuration, "non_qs_sdim"),
    )
    residual = BoozerResidual(solver, native_field)
    iota_penalty = QuadraticPenalty(Iotas(solver), iota_target, "identity")
    major_radius_target = float(major_radius.J())
    radius_penalty = QuadraticPenalty(
        major_radius,
        major_radius_target,
        "identity",
    )
    total_length_target = float(sum(lengths).J())
    length_penalty = QuadraticPenalty(
        sum(lengths),
        total_length_target,
        "max",
    )
    objective = (
        non_qs
        + _configuration_float(bundle.configuration, "residual_weight") * residual
        + iota_penalty
        + radius_penalty
        + length_penalty
    )
    initial_parameters = np.asarray(objective.x, dtype=np.float64)
    if not np.array_equal(initial_parameters, arrays["coil_dofs"]):
        raise ValueError(
            "native objective parameters do not exactly match frozen coil_dofs"
        )
    return _PreparedNativeRuntime(
        solver=cast(_NativeSolver, solver),
        objective=cast(_NativeObjective, objective),
        non_qs=cast(_NativeObjective, non_qs),
        residual=cast(_NativeObjective, residual),
        volume=cast(_NativeObjective, volume),
        radius_penalty=cast(_NativeObjective, radius_penalty),
        length_penalty=cast(_NativeObjective, length_penalty),
        initial_parameters=initial_parameters,
        initial_solution_success=bool(initial_solution["success"]),
        initial_iota=iota_target,
        initial_volume=initial_volume,
        baseline_anchor=NativeBaselineAnchor(
            parameter_sha256=_canonical_fp64_digest(initial_parameters),
            surface_sha256=_canonical_fp64_digest(
                np.asarray(solver.surface.x, dtype=np.float64)
            ),
            iota=iota_target,
            G=float(initial_solution["G"]),
            iota_target=iota_target,
            volume_target=volume_target,
            major_radius_target=major_radius_target,
            total_length_target=total_length_target,
            inner_solver_success=bool(initial_solution["success"]),
        ),
    )


def _native(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
    measurement: MeasurementExecution | None = None,
) -> LaneObservation:
    from scipy.optimize import minimize

    prepared = _prepare_native_variant_runtime(bundle, arrays, spec)
    solver = prepared.solver
    objective = prepared.objective
    non_qs = prepared.non_qs
    residual = prepared.residual
    volume = prepared.volume
    radius_penalty = prepared.radius_penalty
    length_penalty = prepared.length_penalty
    initial_parameters = prepared.initial_parameters
    iota_target = prepared.initial_iota
    initial_volume = prepared.initial_volume

    value_and_grad = prepared.value_and_grad

    native_iteration = 0
    with _measurement_optimization_window(
        measurement,
        lambda: value_and_grad(initial_parameters),
    ) as ((initial_objective, initial_gradient), trajectory):
        if trajectory is None:
            record_iteration = None
        else:

            def record_iteration(intermediate_result) -> None:
                nonlocal native_iteration
                native_iteration += 1
                trajectory.record(native_iteration, float(intermediate_result.fun))

        optimizer_result = minimize(
            value_and_grad,
            initial_parameters,
            jac=True,
            method="BFGS",
            options={
                "maxiter": _configuration_int(bundle.configuration, "outer_maxiter"),
                "gtol": OUTER_GRADIENT_TOLERANCE,
            },
            callback=record_iteration,
        )
    final_parameters = np.asarray(optimizer_result.x, dtype=np.float64)
    objective.x = final_parameters
    final_objective = float(objective.J())
    final_gradient = np.asarray(objective.dJ(), dtype=np.float64)
    final_non_qs_ratio = float(non_qs.J())
    final_iota = float(solver.res["iota"])
    final_volume = float(volume.J())
    final_major_radius_penalty = float(radius_penalty.J())
    final_length_penalty = float(length_penalty.J())
    final_boozer_residual = float(residual.J())
    inner_solver_success = bool(solver.res["success"])
    outer_solver_success = bool(optimizer_result.success)
    endpoint_certificate = (
        certify_optimization_endpoint(
            status_convention="scipy-bfgs",
            provider_success=outer_solver_success,
            provider_status=int(optimizer_result.status),
            iterations=int(optimizer_result.nit),
            max_iterations=_configuration_int(
                bundle.configuration,
                "outer_maxiter",
            ),
            initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
            final_gradient_inf_norm=float(np.max(np.abs(final_gradient))),
            parameters_finite=bool(np.all(np.isfinite(final_parameters))),
            observables_finite=bool(
                np.isfinite(final_objective)
                and np.all(np.isfinite(final_gradient))
                and np.isfinite(final_non_qs_ratio)
                and np.isfinite(final_iota)
                and np.isfinite(final_volume)
                and np.isfinite(final_major_radius_penalty)
                and np.isfinite(final_length_penalty)
                and np.isfinite(final_boozer_residual)
            ),
            inner_success=bool(
                prepared.initial_solution_success and inner_solver_success
            ),
        )
        if spec.enforce_endpoint_certificate
        else None
    )
    values = _values(
        surface_dofs=arrays["surface_dofs"],
        coil_dofs=arrays["coil_dofs"],
        initial_parameters=initial_parameters,
        initial_objective=initial_objective,
        initial_gradient=initial_gradient,
        initial_iota=iota_target,
        initial_volume=initial_volume,
        final_parameters=final_parameters,
        final_objective=final_objective,
        final_gradient=final_gradient,
        final_non_qs_ratio=final_non_qs_ratio,
        final_iota=final_iota,
        final_volume=final_volume,
        final_major_radius_penalty=final_major_radius_penalty,
        final_length_penalty=final_length_penalty,
        final_boozer_residual=final_boozer_residual,
        inner_solver_success=inner_solver_success,
        outer_solver_success=outer_solver_success,
        outer_solver_status=int(optimizer_result.status),
        report_residual=spec.report_residual,
        endpoint_certificate=endpoint_certificate,
    )
    return _observation(
        "native-cpu",
        bundle,
        values,
        platform="cpu",
        precision="fp64",
        driver="simsopt_scipy_bfgs_with_boozer_newton",
        workflow_stages=spec.workflow_stages,
        solver_counts=(
            int(optimizer_result.nit),
            int(optimizer_result.nfev),
            int(optimizer_result.njev),
        ),
        endpoint_certificate=endpoint_certificate,
    )


def _host_float(value: object) -> float:
    import jax

    return float(np.asarray(jax.device_get(value), dtype=np.float64))


def _host_array(value: object) -> np.ndarray:
    import jax

    return np.asarray(jax.device_get(value), dtype=np.float64)


def _host_bool(value: object) -> bool:
    import jax

    return bool(np.asarray(jax.device_get(value), dtype=np.bool_))


def _host_int(value: object) -> int:
    import jax

    return int(np.asarray(jax.device_get(value), dtype=np.int64))


def _materialize_changed_state_timeline_observation(
    deferred: _DeferredChangedStateTimelineObservation,
) -> ChangedStateTimelineObservation:
    """Hostify deferred evidence only after the measured window has closed."""

    trace_presence = (
        _host_bool(deferred.newton_trace_active_present),
        _host_bool(deferred.inner_residual_trace_present),
        _host_bool(deferred.newton_step_accepted_trace_present),
        _host_bool(deferred.newton_linear_solve_success_trace_present),
    )
    if len(set(trace_presence)) != 1:
        raise RuntimeError("packed Newton detailed-trace presence is inconsistent")
    newton_trace_available = all(trace_presence)
    missing_int = int(np.iinfo(np.int32).min)
    attempted_value = _host_int(deferred.newton_attempted_iterations)
    newton_attempted_iterations = (
        None if attempted_value == missing_int else attempted_value
    )
    if newton_trace_available != (newton_attempted_iterations is not None):
        raise RuntimeError(
            "packed Newton attempted count and detailed-trace availability disagree"
        )
    if newton_trace_available:
        active = _host_array(deferred.newton_trace_active).astype(np.bool_)
        residual_trace = _host_array(deferred.inner_residual_trace).reshape(-1)
        step_accepted = _host_array(deferred.newton_step_accepted_trace).astype(
            np.bool_
        )
        linear_success = _host_array(deferred.newton_linear_solve_success_trace).astype(
            np.bool_
        )
        trimmed_residual_trace = tuple(float(value) for value in residual_trace[active])
        trimmed_step_accepted = tuple(bool(value) for value in step_accepted[active])
        trimmed_linear_success = tuple(bool(value) for value in linear_success[active])
    else:
        trimmed_residual_trace = ()
        trimmed_step_accepted = ()
        trimmed_linear_success = ()
    parameters = np.asarray(deferred.parameters, dtype=np.float64)
    gradient = np.asarray(deferred.gradient, dtype=np.float64)
    counts = deferred.execution_counts
    return ChangedStateTimelineObservation(
        evaluation_id=deferred.trace_context.evaluation_id,
        evaluation_kind=deferred.trace_context.kind.value,
        outer_iteration_id=deferred.trace_context.outer_iteration_id,
        parameter_sha256=deferred.trace_context.parameter_sha256,
        parameters=tuple(float(value) for value in parameters.reshape(-1)),
        parameter_shape=deferred.parameter_shape,
        objective=float(deferred.objective),
        gradient=tuple(float(value) for value in gradient.reshape(-1)),
        gradient_dtype=np.dtype(np.float64).str,
        gradient_shape=deferred.gradient_shape,
        values_finite=bool(
            np.isfinite(deferred.objective) and np.all(np.isfinite(gradient))
        ),
        forward_success=_host_bool(deferred.forward_success),
        primal_success=_host_bool(deferred.primal_success),
        actual_adjoint_success=_host_bool(deferred.actual_adjoint_success),
        gradient_source=deferred.gradient_source,
        candidate_gradient_source=deferred.candidate_gradient_source,
        eligible=_host_bool(deferred.eligible),
        inner_residual_trace=trimmed_residual_trace,
        newton_step_accepted_trace=trimmed_step_accepted,
        newton_linear_solve_success_trace=trimmed_linear_success,
        newton_iterations=_host_int(deferred.newton_iterations),
        newton_attempted_iterations=newton_attempted_iterations,
        newton_trace_available=newton_trace_available,
        adjoint_route="exact_jacobian_dense_fp64_lu",
        adjoint_output=tuple(
            float(value) for value in _host_array(deferred.adjoint_output).reshape(-1)
        ),
        adjoint_residual=_host_float(deferred.adjoint_residual),
        adjoint_residual_relative=_host_float(deferred.adjoint_residual_relative),
        dense_materializations=_host_int(counts.dense_materialization_count),
        lu_factorizations=_host_int(counts.lu_factorization_count),
        lu_solves=_host_int(counts.lu_solve_count),
        refinement_corrections=_host_int(counts.refinement_correction_count),
        adjoint_executions=_host_int(counts.adjoint_execution_count),
        observables=tuple(
            (name, None if value is None else _host_float(value))
            for name, value in deferred.observables
        ),
    )


def _prepare_jax_variant_runtime(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
    measurement: MeasurementExecution | None,
    *,
    exact_newton_variant: Literal["C0", "C1", "C2"] = "C0",
) -> _PreparedJaxRuntime:
    """Construct the single session whose compiled callables warm and measure."""

    if exact_newton_variant not in ("C0", "C1", "C2"):
        raise ValueError("exact_newton_variant must be C0, C1, or C2")

    from simsopt.geo import CurveLength, Volume
    from simsopt_jax.geo.optimizers.single_stage_routing import (
        resolve_single_stage_jax_boozer_optimizer_backend,
    )
    from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
    from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
    from simsopt_jax_adapters.geo.surface_objectives import (
        make_traceable_objective_runtime_bundle,
        make_traceable_objective_session,
    )
    from simsopt_jax_adapters.geo.surface_objectives_traceable import (
        AcceptedIncumbentHostValueAndGrad,
    )

    import jax

    (
        base_curves,
        _base_currents,
        magnetic_axis,
        nfp,
        native_field,
        surface,
        G0,
    ) = _problem(bundle.configuration, bundle.scale)
    _validate_reconstructed_bundle_arrays(
        arrays,
        axis_dofs=np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
        coil_dofs=np.asarray(native_field.x, dtype=np.float64),
        surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
    )
    surface.set_dofs(arrays["surface_dofs"])
    field = BiotSavartJAX(native_field.coils)
    volume = Volume(surface)
    optimizer_backend = (
        measurement.optimizer_backend if measurement is not None else None
    )
    solver_options: dict[str, object] = {
        "newton_maxiter": _configuration_int(bundle.configuration, "inner_maxiter"),
        "newton_tol": _configuration_float(bundle.configuration, "inner_tolerance"),
        "verbose": False,
    }
    if optimizer_backend is not None:
        solver_options["optimizer_backend"] = (
            resolve_single_stage_jax_boozer_optimizer_backend(
                "jax",
                optimizer_backend,
            )
        )
    if exact_newton_variant == "C0":
        solver_type = BoozerSurfaceJAX
    else:

        class _ComputeGraphCanaryBoozerSurface(BoozerSurfaceJAX):
            """Construction-scoped C1/C2 route; production C0 stays untouched."""

            def run_code_traceable(
                self,
                coil_source,
                sdofs,
                iota,
                G,
                *,
                certificate_coil_source=None,
                materialize_dense_linearization=None,
            ):
                del materialize_dense_linearization
                route = self._make_run_code_traceable_exact_benchmark_variant(
                    exact_newton_variant
                )
                array_result = route.compiled_kernel(
                    coil_source,
                    certificate_coil_source,
                    sdofs,
                    iota,
                    G,
                )
                return route.project_result(array_result)

        solver_type = _ComputeGraphCanaryBoozerSurface

    solver = solver_type(
        field,
        surface,
        volume,
        float(volume.J()),
        options=solver_options,
    )
    initial_solution = cast(
        Mapping[str, object],
        solver.run_code_traceable(
            field.coil_set_spec(),
            jax.device_put(np.asarray(surface.get_dofs(), dtype=np.float64)),
            jax.device_put(
                np.asarray(
                    _configuration_float(bundle.configuration, "initial_iota"),
                    dtype=np.float64,
                )
            ),
            jax.device_put(np.asarray(G0, dtype=np.float64)),
        ),
    )
    solver.install_traceable_solved_runtime_state(initial_solution)
    initial_inner_success = _host_bool(initial_solution["success"])
    iota_target = _host_float(initial_solution["iota"])
    initial_volume = float(volume.J())
    major_radius_target = float(surface.major_radius())
    total_length_target = float(sum(CurveLength(curve).J() for curve in base_curves))
    objective_configuration: dict[str, object] = {
        "non_qs_weight": 1.0,
        "residual_weight": _configuration_float(
            bundle.configuration,
            "residual_weight",
        ),
        "iota_weight": 1.0,
        "major_radius_weight": 1.0,
        "length_weight": 1.0,
        "curvature_weight": 0.0,
        "curve_curve_weight": 0.0,
        "curve_surface_weight": 0.0,
        "surface_vessel_weight": 0.0,
        "non_qs_quadpoints_phi": np.asarray(
            np.linspace(
                0.0,
                1.0 / nfp,
                2 * _configuration_int(bundle.configuration, "non_qs_sdim"),
                endpoint=False,
            ),
            dtype=np.float64,
        ),
        "non_qs_quadpoints_theta": np.asarray(
            np.linspace(
                0.0,
                1.0,
                2 * _configuration_int(bundle.configuration, "non_qs_sdim"),
                endpoint=False,
            ),
            dtype=np.float64,
        ),
        "non_qs_axis": 0,
        "optimized_coil_index": 0,
        "length_coil_indices": (0, 1, 2),
        "length_target": total_length_target,
        "curvature_threshold": 0.0,
        "curvature_p_norm": 2.0,
        "major_radius_target": major_radius_target,
        "curve_curve_threshold": 0.0,
        "curve_surface_threshold": 0.0,
        "vessel_gamma": np.asarray(surface.gamma(), dtype=np.float64),
        "surface_vessel_threshold": 0.0,
    }
    session = make_traceable_objective_session(
        solver,
        field,
        iota_target,
        outer_objective_config=objective_configuration,
    )
    runtime = make_traceable_objective_runtime_bundle(
        solver,
        field,
        iota_target,
        outer_objective_config=objective_configuration,
        session=session,
    )
    initial_parameters = np.asarray(field.x, dtype=np.float64)
    initial_parameters.setflags(write=False)
    prototype = cast(
        _PreparedIncumbentPrototype,
        session.accepted_incumbent_host_value_and_grad(),
    )
    incumbent_evaluator = prototype._compiled_evaluate
    baseline_inner_state = prototype.current_inner_state

    def mint_incumbent_controller() -> object:
        return AcceptedIncumbentHostValueAndGrad(
            incumbent_evaluator,
            baseline_inner_state,
        )

    return _PreparedJaxRuntime(
        session=session,
        runtime=runtime,
        reporting=cast(
            Callable[..., Mapping[str, object]],
            runtime["reporting_metrics_from_solution"],
        ),
        value_and_grad=cast(Callable[..., object], runtime["value_and_grad"]),
        initial_parameters=initial_parameters,
        initial_inner_success=initial_inner_success,
        iota_target=iota_target,
        initial_volume=initial_volume,
        optimizer_backend=optimizer_backend,
        incumbent_evaluator=incumbent_evaluator,
        incumbent_factory=mint_incumbent_controller,
        exact_newton_variant=exact_newton_variant,
    )


def _prepare_jax_variant_execution(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
    measurement: MeasurementExecution,
) -> _PreparedJaxVariantExecution:
    """Prepare one exact runtime for a warm run followed by a measured run."""

    if lane == "native-cpu":
        raise ValueError("prepared JAX execution requires a JAX lane")
    return _PreparedJaxVariantExecution(
        lane=lane,
        bundle=bundle,
        arrays=arrays,
        spec=spec,
        _runtime=_prepare_jax_variant_runtime(bundle, arrays, spec, measurement),
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
    measurement: MeasurementExecution | None = None,
    *,
    prepared: _PreparedJaxRuntime | None = None,
) -> LaneObservation:
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.geo.optimizer_host_lbfgs import (
        lbfgs_status_is_success,
        line_search_value_and_grad_more_thuente_host,
        minimize_bfgs_host_core,
        minimize_lbfgs_host_core,
    )
    from simsopt_jax.geo.optimizers.optimizer import (
        resolve_optimizer_backend_method,
        target_minimize,
    )
    from simsopt_jax.runtime.trace_annotations import (
        EvaluationDisposition,
        EvaluationKind,
        EvaluationTraceContext,
        PhaseId,
        annotations_enabled,
        evaluation_context,
        host_span,
        record_evaluation_disposition,
    )
    from simsopt_jax_adapters.geo.surface_objectives import (
        traceable_forward_result_outer_raw_terms,
    )
    from simsopt_jax_adapters.geo.surface_objectives_traceable import (
        _accepted_incumbent_host_observation_sink,
        _AcceptedIncumbentHostEvaluationObservation,
        _traceable_execution_evidence,
    )

    import jax

    optimizer_backend = (
        measurement.optimizer_backend if measurement is not None else None
    )
    if measurement is None:
        prepared = _prepare_jax_variant_runtime(
            bundle,
            arrays,
            spec,
            None,
        )
    elif prepared is None:
        prepared = _prepare_jax_variant_runtime(bundle, arrays, spec, measurement)
    session = prepared.session
    reporting = prepared.reporting
    initial_parameters = prepared.initial_parameters
    initial_inner_success = prepared.initial_inner_success
    iota_target = prepared.iota_target
    initial_volume = prepared.initial_volume
    incumbent_controller = prepared.fresh_incumbent_controller()
    value_and_grad = cast(Callable, prepared.value_and_grad)
    timeline_evaluation_count = 0
    timeline_accepted_iterations = 0
    timeline_pending_trials: list[EvaluationTraceContext] = []
    timeline_pending_values: dict[str, tuple[float, np.ndarray]] = {}
    accepted_timeline_evaluation: _AcceptedTimelineValueAndGradient | None = None
    timeline_contexts: list[EvaluationTraceContext] = []
    deferred_timeline_observations: list[_DeferredChangedStateTimelineObservation] = []

    def timeline_parameter_sha256(parameters: np.ndarray) -> str:
        canonical = np.ascontiguousarray(
            parameters,
            dtype=np.dtype("<f8"),
        ).reshape(-1)
        return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()

    def timeline_evaluate(parameters, kind: EvaluationKind, evaluate: Callable):
        nonlocal timeline_evaluation_count
        observation_sink = _TIMELINE_OBSERVATION_SINK.get()
        if not annotations_enabled() and observation_sink is None:
            return evaluate()
        context_id = f"evaluation-{timeline_evaluation_count:06d}"
        timeline_evaluation_count += 1
        with evaluation_context(
            context_id,
            timeline_parameter_sha256(np.asarray(parameters)),
            kind,
        ) as trace_context:
            evaluated = evaluate()
        timeline_contexts.append(trace_context)
        if (
            annotations_enabled() or observation_sink is not None
        ) and kind is EvaluationKind.TRIAL:
            timeline_pending_trials.append(trace_context)
        return evaluated

    def defer_timeline_observation(
        trace_context: EvaluationTraceContext,
        parameters: np.ndarray,
        objective: float,
        gradient: np.ndarray,
        *,
        forward_success: object,
        primal_success: object,
        actual_adjoint_success: object,
        gradient_source: str,
        candidate_gradient_source: bool,
        eligible: object,
        forward_result: Mapping[str, object],
        adjoint_output: object,
        adjoint_residual: object,
        adjoint_residual_relative: object,
        execution_counts: _TimelineExecutionCounts,
        observables: tuple[
            tuple[str, object | None], ...
        ] = _TIMELINE_UNAVAILABLE_PHYSICS_OBSERVABLES,
    ) -> None:
        canonical_parameters = np.asarray(parameters, dtype=np.float64)
        canonical_gradient = np.asarray(gradient, dtype=np.float64)
        deferred_timeline_observations.append(
            _DeferredChangedStateTimelineObservation(
                trace_context=trace_context,
                parameters=tuple(
                    float(value) for value in canonical_parameters.reshape(-1)
                ),
                parameter_shape=tuple(canonical_parameters.shape),
                objective=float(objective),
                gradient=tuple(
                    float(value) for value in canonical_gradient.reshape(-1)
                ),
                gradient_shape=tuple(canonical_gradient.shape),
                forward_success=forward_success,
                primal_success=primal_success,
                actual_adjoint_success=actual_adjoint_success,
                gradient_source=gradient_source,
                candidate_gradient_source=candidate_gradient_source,
                eligible=eligible,
                newton_iterations=forward_result["newton_iterations"],
                newton_attempted_iterations=forward_result[
                    "newton_attempted_iterations"
                ],
                newton_trace_active_present=forward_result[
                    "newton_trace_active_present"
                ],
                inner_residual_trace_present=forward_result[
                    "newton_trace_linear_residual_relative_present"
                ],
                newton_step_accepted_trace_present=forward_result[
                    "newton_trace_step_accepted_present"
                ],
                newton_linear_solve_success_trace_present=forward_result[
                    "newton_trace_linear_solve_success_present"
                ],
                newton_trace_active=forward_result["newton_trace_active"],
                inner_residual_trace=forward_result[
                    "newton_trace_linear_residual_relative"
                ],
                newton_step_accepted_trace=forward_result["newton_trace_step_accepted"],
                newton_linear_solve_success_trace=forward_result[
                    "newton_trace_linear_solve_success"
                ],
                adjoint_output=adjoint_output,
                adjoint_residual=adjoint_residual,
                adjoint_residual_relative=adjoint_residual_relative,
                execution_counts=execution_counts,
                observables=observables,
            )
        )

    def timeline_value_and_grad(
        parameters: np.ndarray,
        kind: EvaluationKind,
    ) -> tuple[float, np.ndarray]:
        observation_sink = _TIMELINE_OBSERVATION_SINK.get()
        scientific_evidence: list[_AcceptedIncumbentHostEvaluationObservation] = []

        def evaluate() -> tuple[float, np.ndarray]:
            if observation_sink is None:
                return incumbent_controller.value_and_grad(parameters)
            with _accepted_incumbent_host_observation_sink(scientific_evidence.append):
                return incumbent_controller.value_and_grad(parameters)

        evaluated = timeline_evaluate(
            parameters,
            kind,
            evaluate,
        )
        if (
            annotations_enabled() or observation_sink is not None
        ) and kind is EvaluationKind.TRIAL:
            value, gradient = evaluated
            timeline_pending_values[timeline_contexts[-1].evaluation_id] = (
                value,
                gradient,
            )
        if observation_sink is not None:
            if len(scientific_evidence) != 1:
                raise RuntimeError(
                    "timeline evaluation did not produce exactly one scientific "
                    "evidence record"
                )
            evidence = scientific_evidence[0]
            defer_timeline_observation(
                timeline_contexts[-1],
                np.asarray(parameters, dtype=np.float64),
                evidence.value,
                evidence.gradient,
                forward_success=evidence.forward_success,
                primal_success=evidence.primal_success,
                actual_adjoint_success=evidence.actual_adjoint_success,
                gradient_source=evidence.gradient_source,
                candidate_gradient_source=evidence.candidate_gradient_source,
                eligible=evidence.eligible,
                forward_result=evidence.forward_result,
                adjoint_output=evidence.adjoint_output,
                adjoint_residual=evidence.adjoint_residual,
                adjoint_residual_relative=evidence.adjoint_residual_relative,
                execution_counts=evidence.execution_counts,
            )
        return evaluated

    def emit_timeline_disposition(
        trace_context: EvaluationTraceContext,
        disposition: EvaluationDisposition,
        accepted_iteration_id: int | None,
    ) -> None:
        record_evaluation_disposition(
            trace_context,
            disposition,
            accepted_iteration_id=accepted_iteration_id,
        )
        observation_sink = _TIMELINE_OBSERVATION_SINK.get()
        if observation_sink is not None:
            observation_sink(
                ChangedStateTimelineDisposition(
                    evaluation_id=trace_context.evaluation_id,
                    parameter_sha256=trace_context.parameter_sha256,
                    disposition=disposition.value,
                    accepted_iteration_id=accepted_iteration_id,
                )
            )

    def evaluate_optimizer_trial(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        return timeline_value_and_grad(parameters, EvaluationKind.TRIAL)

    def evaluate_optimizer_final(
        parameters: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        return _optimizer_final_value_and_gradient(
            parameters,
            timeline_enabled=(
                annotations_enabled() or _TIMELINE_OBSERVATION_SINK.get() is not None
            ),
            accepted_timeline_evaluation=accepted_timeline_evaluation,
            parameter_sha256=timeline_parameter_sha256,
            evaluate=incumbent_controller.value_and_grad,
        )

    def accept_optimizer_trial(parameters: np.ndarray) -> None:
        nonlocal accepted_timeline_evaluation, timeline_accepted_iterations
        if not annotations_enabled() and _TIMELINE_OBSERVATION_SINK.get() is None:
            incumbent_controller.accept(parameters)
            return
        accepted_sha256 = timeline_parameter_sha256(np.asarray(parameters))
        accepted_indices = tuple(
            index
            for index, pending in enumerate(timeline_pending_trials)
            if pending.parameter_sha256 == accepted_sha256
        )
        if not accepted_indices:
            raise RuntimeError(
                "accepted parameters have no correlated timeline evaluation"
            )
        accepted_index = accepted_indices[-1]
        accepted_context = timeline_pending_trials[accepted_index]
        accepted_value, accepted_gradient = timeline_pending_values[
            accepted_context.evaluation_id
        ]
        accepted_iteration_id = timeline_accepted_iterations + 1
        with host_span(
            PhaseId.OPTIMIZER_LIFECYCLE,
            attributes={"accepted_iteration_id": accepted_iteration_id},
        ):
            incumbent_controller.accept(parameters)
            accepted_timeline_evaluation = _AcceptedTimelineValueAndGradient(
                parameter_sha256=accepted_context.parameter_sha256,
                value=accepted_value,
                gradient=accepted_gradient,
            )
            for index, pending in enumerate(timeline_pending_trials):
                accepted = index == accepted_index
                emit_timeline_disposition(
                    pending,
                    (
                        EvaluationDisposition.ACCEPTED
                        if accepted
                        else EvaluationDisposition.REJECTED
                    ),
                    accepted_iteration_id if accepted else None,
                )
        timeline_pending_trials.clear()
        timeline_pending_values.clear()
        timeline_accepted_iterations = accepted_iteration_id

    def reject_unresolved_optimizer_trials() -> None:
        if not timeline_pending_trials:
            return
        with host_span(
            PhaseId.OPTIMIZER_LIFECYCLE,
            attributes={"completion_rejections": len(timeline_pending_trials)},
        ):
            for pending in timeline_pending_trials:
                emit_timeline_disposition(
                    pending,
                    EvaluationDisposition.REJECTED,
                    None,
                )
        timeline_pending_trials.clear()
        timeline_pending_values.clear()

    def evaluate_initial() -> tuple[float, np.ndarray]:
        if optimizer_backend == "optax-lbfgs":
            initial_objective_device, initial_gradient_device = value_and_grad(
                jax.device_put(initial_parameters)
            )
            jax.block_until_ready((initial_objective_device, initial_gradient_device))
            return (
                _host_float(initial_objective_device),
                _host_array(initial_gradient_device),
            )
        return timeline_value_and_grad(initial_parameters, EvaluationKind.INITIAL)

    with _measurement_optimization_window(
        measurement,
        evaluate_initial,
    ) as ((initial_objective, initial_gradient), trajectory):
        if trajectory is None:
            progress_callback = None
        else:

            def record_iteration(
                iteration: int,
                objective: float,
                _grad_norm: float,
            ) -> None:
                trajectory.record(iteration, objective)

            progress_callback = record_iteration
        if optimizer_backend == "optax-lbfgs":
            optimizer_result = target_minimize(
                value_and_grad,
                jax.device_put(initial_parameters),
                method=resolve_optimizer_backend_method(
                    optimizer_backend,
                    limited_memory=True,
                ),
                tol=OUTER_GRADIENT_TOLERANCE,
                maxiter=_configuration_int(bundle.configuration, "outer_maxiter"),
                options={
                    "maxcor": min(
                        _configuration_int(bundle.configuration, "outer_maxiter"),
                        200,
                    ),
                    "maxls": 20,
                },
                value_and_grad=True,
                progress_callback=progress_callback,
            )
            jax.block_until_ready(optimizer_result.x)
            driver = Driver.OPTAX_LBFGS
            final_parameters = np.asarray(optimizer_result.x, dtype=np.float64)
            optimizer_iterations = int(optimizer_result.nit)
            optimizer_evaluations = int(optimizer_result.nfev)
            optimizer_gradient_evaluations = int(optimizer_result.njev)
            optimizer_status = int(optimizer_result.status)
            outer_solver_success = bool(optimizer_result.success)
            status_convention = "optax-lbfgs"
        else:
            driver = scalar_example_driver()
            if driver == Driver.SIMSOPT_LBFGSB:
                optimizer_result = minimize_lbfgs_host_core(
                    evaluate_optimizer_trial,
                    initial_parameters,
                    maxiter=_configuration_int(
                        bundle.configuration,
                        "outer_maxiter",
                    ),
                    maxcor=min(
                        _configuration_int(bundle.configuration, "outer_maxiter"),
                        200,
                    ),
                    ftol=0.0,
                    gtol=OUTER_GRADIENT_TOLERANCE,
                    maxls=20,
                    initial_value_and_grad=(initial_objective, initial_gradient),
                    final_eval_value_and_grad_host=evaluate_optimizer_final,
                    callback=accept_optimizer_trial,
                    progress_callback=progress_callback,
                )
            else:
                optimizer_result = minimize_bfgs_host_core(
                    evaluate_optimizer_trial,
                    initial_parameters,
                    maxiter=_configuration_int(
                        bundle.configuration,
                        "outer_maxiter",
                    ),
                    gtol=OUTER_GRADIENT_TOLERANCE,
                    maxls=20,
                    initial_value_and_grad=(initial_objective, initial_gradient),
                    line_search_value_and_grad=(
                        line_search_value_and_grad_more_thuente_host
                    ),
                    callback=accept_optimizer_trial,
                    progress_callback=progress_callback,
                )
            reject_unresolved_optimizer_trials()
            final_parameters = np.asarray(optimizer_result.x_k, dtype=np.float64)
            optimizer_iterations = int(optimizer_result.k)
            optimizer_evaluations = int(optimizer_result.nfev)
            optimizer_gradient_evaluations = int(optimizer_result.ngev)
            optimizer_status = int(optimizer_result.status)
            outer_solver_success = bool(
                lbfgs_status_is_success(optimizer_result.status, False)
                if driver == Driver.SIMSOPT_LBFGSB
                else optimizer_result.converged
            )
            status_convention = (
                "host-lbfgsb" if driver == Driver.SIMSOPT_LBFGSB else "host-bfgs"
            )

    timeline_final_lifecycle = (
        annotations_enabled() or _TIMELINE_OBSERVATION_SINK.get() is not None
    )

    def evaluate_final_candidate():
        if _TIMELINE_OBSERVATION_SINK.get() is None:
            if timeline_final_lifecycle:
                return session._evaluate_candidate_from_anchor_host(
                    final_parameters,
                    incumbent_controller.current_inner_state,
                )
            return session.evaluate_candidate_from_anchor(
                final_parameters, incumbent_controller.current_inner_state
            )
        with _traceable_execution_evidence():
            return session._evaluate_candidate_from_anchor_host(
                final_parameters, incumbent_controller.current_inner_state
            )

    final_result = timeline_evaluate(
        final_parameters,
        EvaluationKind.FINAL_REPORTING,
        evaluate_final_candidate,
    )
    if timeline_final_lifecycle:
        final_evaluation, final_objective, final_gradient = final_result
    else:
        final_evaluation = final_result
        final_forward = final_evaluation.forward_result
        final_objective = _host_float(final_forward["value"])
        final_gradient = _host_array(final_evaluation.gradient)
    final_forward = final_evaluation.forward_result
    if optimizer_backend != "optax-lbfgs":
        provider_state_invalid = bool(
            not np.isfinite(final_objective) or not np.all(np.isfinite(final_gradient))
        )
        outer_solver_success = bool(
            lbfgs_status_is_success(optimizer_status, provider_state_invalid)
            if driver == Driver.SIMSOPT_LBFGSB
            else optimizer_result.converged
        )
    final_metrics = reporting(
        final_evaluation.candidate_inner_state.coil_dofs,
        final_forward["x"],
        final_forward["success"],
        include_distance_metrics=False,
        outer_raw_terms=traceable_forward_result_outer_raw_terms(final_forward),
    )
    if _TIMELINE_OBSERVATION_SINK.get() is not None:
        final_adjoint_evidence = final_evaluation.adjoint_evidence
        final_execution_counts = final_evaluation.execution_counts
        if final_adjoint_evidence is None or final_execution_counts is None:
            raise RuntimeError("final timeline evaluation lacks execution evidence")
        defer_timeline_observation(
            timeline_contexts[-1],
            final_parameters,
            final_objective,
            final_gradient,
            forward_success=final_forward["success"],
            primal_success=final_forward["primal_success"],
            actual_adjoint_success=final_evaluation.actual_adjoint_success,
            gradient_source=final_evaluation.gradient_source,
            candidate_gradient_source=(final_evaluation.gradient_source == "candidate"),
            eligible=final_evaluation.candidate_inner_state.eligible,
            forward_result=final_forward,
            adjoint_output=final_adjoint_evidence.adjoint_output,
            adjoint_residual=final_adjoint_evidence.residual,
            adjoint_residual_relative=final_adjoint_evidence.residual_relative,
            execution_counts=final_execution_counts,
            observables=(
                ("objective", final_forward["value"]),
                ("iota", final_metrics["final_iota"]),
                ("volume", final_metrics["final_volume"]),
                ("non_qs_ratio", final_metrics["final_non_qs"]),
                ("boozer_residual", final_metrics["final_boozer_residual"]),
            ),
        )
    jax.block_until_ready((final_forward, final_metrics))
    observation_sink = _TIMELINE_OBSERVATION_SINK.get()
    if observation_sink is not None:
        for deferred in deferred_timeline_observations:
            observation_sink(_materialize_changed_state_timeline_observation(deferred))
    final_non_qs_ratio = _host_float(final_metrics["final_non_qs"])
    final_iota = _host_float(final_metrics["final_iota"])
    final_volume = _host_float(final_metrics["final_volume"])
    final_major_radius_penalty = _host_float(
        final_metrics["final_major_radius_penalty"]
    )
    final_length_penalty = _host_float(final_metrics["final_length_penalty"])
    final_boozer_residual = _host_float(final_metrics["final_boozer_residual"])
    inner_solver_success = _host_bool(final_metrics["solver_success"])
    endpoint_certificate = (
        certify_optimization_endpoint(
            status_convention=status_convention,
            provider_success=outer_solver_success,
            provider_status=optimizer_status,
            iterations=optimizer_iterations,
            max_iterations=_configuration_int(
                bundle.configuration,
                "outer_maxiter",
            ),
            initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
            final_gradient_inf_norm=float(np.max(np.abs(final_gradient))),
            parameters_finite=bool(np.all(np.isfinite(final_parameters))),
            observables_finite=bool(
                np.isfinite(final_objective)
                and np.all(np.isfinite(final_gradient))
                and np.isfinite(final_non_qs_ratio)
                and np.isfinite(final_iota)
                and np.isfinite(final_volume)
                and np.isfinite(final_major_radius_penalty)
                and np.isfinite(final_length_penalty)
                and np.isfinite(final_boozer_residual)
            ),
            inner_success=bool(initial_inner_success and inner_solver_success),
        )
        if spec.enforce_endpoint_certificate
        else None
    )
    values = _values(
        surface_dofs=arrays["surface_dofs"],
        coil_dofs=arrays["coil_dofs"],
        initial_parameters=initial_parameters,
        initial_objective=initial_objective,
        initial_gradient=initial_gradient,
        initial_iota=iota_target,
        initial_volume=initial_volume,
        final_parameters=final_parameters,
        final_objective=final_objective,
        final_gradient=final_gradient,
        final_non_qs_ratio=final_non_qs_ratio,
        final_iota=final_iota,
        final_volume=final_volume,
        final_major_radius_penalty=final_major_radius_penalty,
        final_length_penalty=final_length_penalty,
        final_boozer_residual=final_boozer_residual,
        inner_solver_success=inner_solver_success,
        outer_solver_success=outer_solver_success,
        outer_solver_status=optimizer_status,
        report_residual=spec.report_residual,
        endpoint_certificate=endpoint_certificate,
    )
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    return _observation(
        lane,
        bundle,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        driver=(
            JAX_OPTAX_DRIVER_ID
            if driver == Driver.OPTAX_LBFGS
            else JAX_FAST_DRIVER_ID
            if driver == Driver.SIMSOPT_LBFGSB
            else JAX_PARITY_DRIVER_ID
        ),
        workflow_stages=spec.workflow_stages,
        solver_counts=(
            optimizer_iterations,
            optimizer_evaluations,
            optimizer_gradient_evaluations,
        ),
        endpoint_certificate=endpoint_certificate,
    )


def _values(
    *,
    surface_dofs: np.ndarray,
    coil_dofs: np.ndarray,
    initial_parameters: np.ndarray,
    initial_objective: float,
    initial_gradient: np.ndarray,
    initial_iota: float,
    initial_volume: float,
    final_parameters: np.ndarray,
    final_objective: float,
    final_gradient: np.ndarray,
    final_non_qs_ratio: float,
    final_iota: float,
    final_volume: float,
    final_major_radius_penalty: float,
    final_length_penalty: float,
    final_boozer_residual: float,
    inner_solver_success: bool,
    outer_solver_success: bool,
    outer_solver_status: int,
    report_residual: bool,
    endpoint_certificate: OptimizationEndpointCertificate | None,
) -> dict[str, np.ndarray]:
    values = {
        "construction:surface_dofs": surface_dofs,
        "construction:coil_dofs": coil_dofs,
        "initial:parameters": initial_parameters,
        "initial:objective": np.asarray(initial_objective, dtype=np.float64),
        "initial:gradient": initial_gradient,
        "initial:iota": np.asarray(initial_iota, dtype=np.float64),
        "initial:volume": np.asarray(initial_volume, dtype=np.float64),
        "final:parameters": final_parameters,
        "final:objective": np.asarray(final_objective, dtype=np.float64),
        "final:gradient": final_gradient,
        "final:non_qs_ratio": np.asarray(final_non_qs_ratio, dtype=np.float64),
        "final:iota": np.asarray(final_iota, dtype=np.float64),
        "final:volume": np.asarray(final_volume, dtype=np.float64),
        "final:major_radius_penalty": np.asarray(
            final_major_radius_penalty,
            dtype=np.float64,
        ),
        "final:length_penalty": np.asarray(
            final_length_penalty,
            dtype=np.float64,
        ),
        "final:inner_solver_success": np.asarray(
            inner_solver_success,
            dtype=np.bool_,
        ),
        "final:outer_solver_success": np.asarray(
            outer_solver_success,
            dtype=np.bool_,
        ),
    }
    if endpoint_certificate is not None:
        values.update(
            {
                "final:endpoint_certificate_success": np.asarray(
                    endpoint_certificate.success,
                    dtype=np.bool_,
                ),
                "final:endpoint_initial_stationary": np.asarray(
                    endpoint_certificate.initial_stationary,
                    dtype=np.bool_,
                ),
                "final:endpoint_terminal_stationary": np.asarray(
                    endpoint_certificate.terminal_stationary,
                    dtype=np.bool_,
                ),
                "final:endpoint_constraints_satisfied": np.asarray(
                    endpoint_certificate.constraints_satisfied,
                    dtype=np.bool_,
                ),
                "final:outer_solver_status": np.asarray(
                    outer_solver_status,
                    dtype=np.int64,
                ),
            }
        )
    if report_residual:
        values["final:boozer_residual"] = np.asarray(
            final_boozer_residual,
            dtype=np.float64,
        )
        values["final:boozer_residual_rms"] = np.asarray(
            np.sqrt(2.0 * final_boozer_residual),
            dtype=np.float64,
        )
    return values


def _observation(
    lane: ParityLane,
    bundle: InputBundle,
    values: dict[str, np.ndarray],
    *,
    platform: str,
    precision: str,
    driver: str,
    workflow_stages: tuple[str, ...],
    solver_counts: tuple[int, int, int],
    endpoint_certificate: OptimizationEndpointCertificate | None = None,
) -> LaneObservation:
    nit, nfev, njev = solver_counts
    objective_decreased = bool(
        np.all(np.isfinite(values["initial:gradient"]))
        and np.all(np.isfinite(values["final:gradient"]))
        and np.all(np.isfinite(values["final:parameters"]))
        and np.isfinite(float(values["final:objective"]))
        and float(values["final:objective"]) <= float(values["initial:objective"])
        and bool(values["final:inner_solver_success"])
    )
    success = bool(
        objective_decreased
        and (endpoint_certificate is None or endpoint_certificate.success)
    )
    if endpoint_certificate is None:
        normalized_status = "converged" if success else "failed"
        raw_status = (
            f"inner={bool(values['final:inner_solver_success'])};"
            f"outer={bool(values['final:outer_solver_success'])}"
        )
    else:
        normalized_status = (
            "converged"
            if success
            else "budget_exhausted"
            if endpoint_certificate.stopping_reason
            in {"iteration-limit", "evaluation-limit"}
            else "failed"
        )
        raw_status = (
            f"inner={bool(values['final:inner_solver_success'])};"
            f"outer={bool(values['final:outer_solver_success'])};"
            f"certificate={endpoint_certificate.success};"
            f"stopping_reason={endpoint_certificate.stopping_reason};"
            f"initial_stationary={endpoint_certificate.initial_stationary};"
            f"terminal_stationary={endpoint_certificate.terminal_stationary};"
            f"constraints_satisfied={endpoint_certificate.constraints_satisfied}"
        )
    return LaneObservation(
        lane=lane,
        backend_mode=(
            "native_cpu" if lane == "native-cpu" else os.environ["SIMSOPT_BACKEND_MODE"]
        ),
        platform=platform,
        precision=precision,
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(
            bundle,
            values["construction:surface_dofs"],
            values["construction:coil_dofs"],
        ),
        driver=driver,
        normalized_status=normalized_status,
        raw_status=raw_status,
        success=success,
        nit=nit,
        nfev=nfev,
        njev=njev,
        completed_workflow_stages=workflow_stages,
        provenance=None,
        values=values,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the matched Boozer-QA coil optimization."""
    return execute_variant(lane, bundle, arrays, BOOZER_QA_SPEC)


def execute_variant(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
    *,
    measurement: MeasurementExecution | None = None,
) -> LaneObservation:
    """Execute one configured native/JAX Boozer single-stage workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays, spec, measurement)
    return _jax(lane, bundle, arrays, spec, measurement)
