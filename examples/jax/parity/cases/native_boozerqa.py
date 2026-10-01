"""Exact matched workflow for ``2_Intermediate/boozerQA.py``."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt.configs import get_data
from simsopt.geo import SurfaceXYZTensorFourier
from simsopt_contracts.optimization_endpoint import (
    OptimizationEndpointCertificate,
    StoppingReason,
    certify_optimization_endpoint,
    normalized_terminal_status,
)
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import ExecutionScale, scalar_example_driver
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_QA_INITIAL_IOTA,
    OFFICIAL_QA_LINE_SEARCH_MAXITER,
    OFFICIAL_QA_NEWTON_MAXITER,
    OFFICIAL_QA_NEWTON_TOLERANCE,
    OFFICIAL_QA_NON_QS_RESOLUTION,
    OFFICIAL_QA_OUTER_DRIVER,
    OFFICIAL_QA_OUTER_MAXITER,
    OFFICIAL_QA_RESIDUAL_WEIGHT,
    OFFICIAL_QA_SURFACE_DISTANCE,
    OFFICIAL_QA_SURFACE_RESOLUTION,
    boozer_qa_outer_objective_config,
)
from simsopt_jax.examples.single_stage_boozer_vacuum import (
    JAX_FAST_DRIVER_ID,
    JAX_PARITY_DRIVER_ID,
    OUTER_GRADIENT_TOLERANCE,
)
from simsopt_jax.geo.optimizer_host_lbfgs import (
    line_search_value_and_grad_more_thuente_host,
    minimize_bfgs_host_core,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.geo.boozer_qa_problem import BoozerQAProblem

import jax

WORKFLOW_STAGES = (
    "construct_ncsx_coils_and_volume_labelled_surface",
    "solve_initial_boozer_surface",
    "assemble_nonqs_iota_radius_and_length_objective",
    "evaluate_initial_objective_and_gradient",
    "optimize_coils_and_currents_with_bfgs",
    "record_final_objective_gradient_and_physics_terms",
)


@dataclass(frozen=True, slots=True)
class BoozerSingleStageSpec:
    """Immutable Boozer inputs; absent native tolerance reuses the base value."""

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
    native_inner_tolerance: float | None = None


#: The official values come from ``simsopt_jax.examples.boozer_official``, the
#: single owner the shipped ``examples/jax/2_Intermediate/boozerQA.py`` reads;
#: the ``bounded_*`` fields are this branch's reduced scale and have no upstream
#: counterpart.
BOOZER_QA_SPEC = BoozerSingleStageSpec(
    case_id="native-boozerqa",
    workflow_stages=WORKFLOW_STAGES,
    bounded_resolution=2,
    native_resolution=OFFICIAL_QA_SURFACE_RESOLUTION,
    inner_tolerance=1.0e-10,
    bounded_outer_maxiter=5,
    native_outer_maxiter=OFFICIAL_QA_OUTER_MAXITER,
    bounded_non_qs_sdim=OFFICIAL_QA_NON_QS_RESOLUTION,
    native_non_qs_sdim=OFFICIAL_QA_NON_QS_RESOLUTION,
    residual_weight=OFFICIAL_QA_RESIDUAL_WEIGHT,
    report_residual=False,
    native_inner_tolerance=OFFICIAL_QA_NEWTON_TOLERANCE,
)


def _outer_driver(spec: BoozerSingleStageSpec) -> Driver:
    """Keep the official BoozerQA BFGS method independent of device mode."""
    return (
        OFFICIAL_QA_OUTER_DRIVER
        if spec.case_id == BOOZER_QA_SPEC.case_id
        else scalar_example_driver()
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
    incumbent_factory: Callable[[], object]

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


class _PreparedIncumbentPrototype(Protocol):
    _compiled_evaluate: Callable[..., object]

    @property
    def current_inner_state(self) -> object: ...


def variant_scale_configuration(
    scale: ExecutionScale,
    spec: BoozerSingleStageSpec,
) -> dict[str, object]:
    """Resolve the frozen scientific inputs for one Boozer workflow variant."""
    native_scale = scale == "native_default"
    resolution = spec.native_resolution if native_scale else spec.bounded_resolution
    return {
        "mpol": resolution,
        "ntor": resolution,
        # Shared by both Boozer single-stage variants: the official BoozerQA
        # inner Newton budget, start iota and extrusion distance.
        "inner_maxiter": OFFICIAL_QA_NEWTON_MAXITER,
        "inner_tolerance": (
            spec.native_inner_tolerance
            if native_scale and spec.native_inner_tolerance is not None
            else spec.inner_tolerance
        ),
        "outer_maxiter": (
            spec.native_outer_maxiter if native_scale else spec.bounded_outer_maxiter
        ),
        "outer_rtol": 0.0,
        "outer_atol": OUTER_GRADIENT_TOLERANCE,
        "initial_iota": OFFICIAL_QA_INITIAL_IOTA,
        "surface_distance": OFFICIAL_QA_SURFACE_DISTANCE,
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


def build_variant_problem(configuration: Mapping[str, object], scale: ExecutionScale):
    """Construct the shared NCSX coil, axis, field, and initial surface state."""

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
    configuration = variant_scale_configuration(scale, spec)
    (
        _base_curves,
        _base_currents,
        magnetic_axis,
        nfp,
        native_field,
        surface,
        G0,
    ) = build_variant_problem(configuration, scale)
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


def validate_variant_bundle_arrays(
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
    ) = build_variant_problem(bundle.configuration, bundle.scale)
    validate_variant_bundle_arrays(
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

    initial_objective, initial_gradient = value_and_grad(initial_parameters)
    optimizer_result = minimize(
        value_and_grad,
        initial_parameters,
        jac=True,
        method="BFGS",
        options={
            "maxiter": _configuration_int(bundle.configuration, "outer_maxiter"),
            "gtol": OUTER_GRADIENT_TOLERANCE,
        },
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
    outer_certificate = certify_optimization_endpoint(
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
        inner_success=bool(prepared.initial_solution_success and inner_solver_success),
    )
    endpoint_certificate = (
        outer_certificate if spec.enforce_endpoint_certificate else None
    )
    values = variant_observable_values(
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
    return variant_lane_observation(
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
        outer_stopping_reason=outer_certificate.stopping_reason,
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


def _prepare_jax_variant_runtime(
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
) -> _PreparedJaxRuntime:
    """Construct the single session and its reusable compiled callables."""

    from simsopt.geo import CurveLength, Volume
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
    ) = build_variant_problem(bundle.configuration, bundle.scale)
    validate_variant_bundle_arrays(
        arrays,
        axis_dofs=np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
        coil_dofs=np.asarray(native_field.x, dtype=np.float64),
        surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
    )
    surface.set_dofs(arrays["surface_dofs"])
    field = BiotSavartJAX(native_field.coils)
    volume = Volume(surface)
    solver_options: dict[str, object] = {
        "newton_maxiter": _configuration_int(bundle.configuration, "inner_maxiter"),
        "newton_tol": _configuration_float(bundle.configuration, "inner_tolerance"),
        "verbose": False,
    }
    solver = BoozerSurfaceJAX(
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
    objective_configuration: dict[str, object] = boozer_qa_outer_objective_config(
        nfp=nfp,
        non_qs_resolution=_configuration_int(bundle.configuration, "non_qs_sdim"),
        length_target=total_length_target,
        major_radius_target=major_radius_target,
        vessel_gamma=surface.gamma(),
        residual_weight=_configuration_float(
            bundle.configuration,
            "residual_weight",
        ),
    )
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
        incumbent_factory=mint_incumbent_controller,
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
) -> LaneObservation:
    from simsopt_jax.backend.runtime import get_runtime_jax_device
    from simsopt_jax.geo.optimizer_host_lbfgs import (
        lbfgs_status_is_success,
        line_search_value_and_grad_more_thuente_host,
        minimize_bfgs_host_core,
        minimize_lbfgs_host_core,
    )
    from simsopt_jax_adapters.geo.surface_objectives import (
        traceable_forward_result_outer_raw_terms,
    )

    import jax

    prepared = _prepare_jax_variant_runtime(bundle, arrays, spec)
    session = prepared.session
    reporting = prepared.reporting
    initial_parameters = prepared.initial_parameters
    initial_inner_success = prepared.initial_inner_success
    iota_target = prepared.iota_target
    initial_volume = prepared.initial_volume
    incumbent_controller = prepared.fresh_incumbent_controller()
    initial_objective, initial_gradient = incumbent_controller.value_and_grad(
        initial_parameters
    )
    driver = _outer_driver(spec)
    if driver == Driver.SIMSOPT_LBFGSB:
        optimizer_result = minimize_lbfgs_host_core(
            incumbent_controller.value_and_grad,
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
            final_eval_value_and_grad_host=incumbent_controller.value_and_grad,
            callback=incumbent_controller.accept,
        )
    else:
        optimizer_result = minimize_bfgs_host_core(
            incumbent_controller.value_and_grad,
            initial_parameters,
            maxiter=_configuration_int(
                bundle.configuration,
                "outer_maxiter",
            ),
            gtol=OUTER_GRADIENT_TOLERANCE,
            maxls=20,
            initial_value_and_grad=(initial_objective, initial_gradient),
            line_search_value_and_grad=(line_search_value_and_grad_more_thuente_host),
            callback=incumbent_controller.accept,
        )
    final_parameters = np.asarray(optimizer_result.x_k, dtype=np.float64)
    optimizer_iterations = int(optimizer_result.k)
    optimizer_evaluations = int(optimizer_result.nfev)
    optimizer_gradient_evaluations = int(optimizer_result.ngev)
    optimizer_status = int(optimizer_result.status)
    status_convention = (
        "host-lbfgsb" if driver == Driver.SIMSOPT_LBFGSB else "host-bfgs"
    )
    final_evaluation = session.evaluate_candidate_from_anchor(
        final_parameters, incumbent_controller.current_inner_state
    )
    final_forward = final_evaluation.forward_result
    final_objective = _host_float(final_forward["value"])
    final_gradient = _host_array(final_evaluation.gradient)
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
    jax.block_until_ready((final_forward, final_metrics))
    final_non_qs_ratio = _host_float(final_metrics["final_non_qs"])
    final_iota = _host_float(final_metrics["final_iota"])
    final_volume = _host_float(final_metrics["final_volume"])
    final_major_radius_penalty = _host_float(
        final_metrics["final_major_radius_penalty"]
    )
    final_length_penalty = _host_float(final_metrics["final_length_penalty"])
    final_boozer_residual = _host_float(final_metrics["final_boozer_residual"])
    inner_solver_success = _host_bool(final_metrics["solver_success"])
    outer_certificate = certify_optimization_endpoint(
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
    endpoint_certificate = (
        outer_certificate if spec.enforce_endpoint_certificate else None
    )
    values = variant_observable_values(
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
    return variant_lane_observation(
        lane,
        bundle,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        driver=(
            JAX_FAST_DRIVER_ID
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
        outer_stopping_reason=outer_certificate.stopping_reason,
    )


def variant_observable_values(
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
    """Encode construction and endpoint quantities in the parity observable schema."""
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


def variant_lane_observation(
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
    outer_stopping_reason: StoppingReason | None = None,
) -> LaneObservation:
    """Apply common endpoint status semantics to a completed Boozer variant.

    A variant that enforces an endpoint certificate is labelled by it. Every
    other variant is labelled by its outer optimizer's own stopping reason: a
    decreased objective never turns an iteration-limit stop into convergence.
    """
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
        if outer_stopping_reason is None:
            raise ValueError(
                "a variant without an endpoint certificate reports its outer "
                "stopping reason"
            )
        terminal = normalized_terminal_status(
            scientific_predicate=objective_decreased,
            stage_stopping_reasons=(outer_stopping_reason,),
        )
        normalized_status = terminal.normalized_status
        success = terminal.success
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


def _execute_official_jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Run the official BoozerQA workflow on the analytic exact Boozer route.

    Upstream's ``solve_residual_equation_exactly_newton`` assembles the dense
    analytic Jacobian, solves it directly, takes the undamped Newton step and, in
    the example's objective, returns ``J = 1e3`` with the previous surface
    restored when the solve fails (upstream boozerQA.py:102-109).  That is what
    :class:`~simsopt_jax_adapters.geo.boozer_qa_problem.BoozerQAProblem` runs, and
    it is the same evaluator the shipped ``examples/jax/2_Intermediate/boozerQA.py``
    drives, so the script and this matched workflow cannot diverge.

    The traceable-session route (``_prepare_jax_variant_runtime`` / ``_jax``) is
    separate; ``execute_variant`` still dispatches JAX lanes to it.
    """
    configuration = bundle.configuration
    (
        base_curves,
        _base_currents,
        magnetic_axis,
        nfp,
        native_field,
        surface,
        G0,
    ) = build_variant_problem(configuration, bundle.scale)
    validate_variant_bundle_arrays(
        arrays,
        axis_dofs=np.asarray(magnetic_axis.local_full_x, dtype=np.float64),
        coil_dofs=np.asarray(native_field.x, dtype=np.float64),
        surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
    )
    surface.set_dofs(arrays["surface_dofs"])
    problem = BoozerQAProblem(
        base_curves=base_curves,
        native_field=native_field,
        surface=surface,
        nfp=nfp,
        initial_G=G0,
        initial_iota=_configuration_float(configuration, "initial_iota"),
        boozer_options={
            "newton_maxiter": _configuration_int(configuration, "inner_maxiter"),
            "newton_tol": _configuration_float(configuration, "inner_tolerance"),
            "verbose": False,
        },
        non_qs_resolution=_configuration_int(configuration, "non_qs_sdim"),
        residual_weight=_configuration_float(configuration, "residual_weight"),
    )
    initial_parameters = problem.initial_coil_dofs
    if not np.array_equal(initial_parameters, arrays["coil_dofs"]):
        raise ValueError("official JAX parameters do not match the frozen coil dofs")
    initial_objective, initial_gradient = problem.value_and_gradient(initial_parameters)
    outer_maxiter = _configuration_int(configuration, "outer_maxiter")
    optimizer_result = minimize_bfgs_host_core(
        problem.value_and_gradient,
        initial_parameters,
        maxiter=outer_maxiter,
        gtol=OUTER_GRADIENT_TOLERANCE,
        maxls=OFFICIAL_QA_LINE_SEARCH_MAXITER,
        initial_value_and_grad=(initial_objective, initial_gradient),
        line_search_value_and_grad=line_search_value_and_grad_more_thuente_host,
    )
    final_parameters = np.asarray(optimizer_result.x_k, dtype=np.float64)
    endpoint = problem.endpoint(final_parameters)
    final_gradient = endpoint.gradient
    outer_solver_success = bool(optimizer_result.converged)
    optimizer_status = int(optimizer_result.status)
    outer_certificate = certify_optimization_endpoint(
        # ``minimize_bfgs_host_core`` is the host core of the official outer
        # method, so its termination integers follow the same ``host-bfgs``
        # convention the session route applies to this spec.
        status_convention="host-bfgs",
        provider_success=outer_solver_success,
        provider_status=optimizer_status,
        iterations=int(optimizer_result.k),
        max_iterations=outer_maxiter,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        final_gradient_inf_norm=float(np.max(np.abs(final_gradient))),
        parameters_finite=bool(np.all(np.isfinite(final_parameters))),
        observables_finite=bool(
            np.isfinite(endpoint.value)
            and np.all(np.isfinite(final_gradient))
            and np.isfinite(endpoint.non_qs_ratio)
            and np.isfinite(endpoint.iota)
            and np.isfinite(endpoint.volume)
            and np.isfinite(endpoint.major_radius_penalty)
            and np.isfinite(endpoint.length_penalty)
            and np.isfinite(endpoint.boozer_residual)
        ),
        inner_success=bool(problem.initial_inner_success and endpoint.inner_success),
    )
    endpoint_certificate = (
        outer_certificate if BOOZER_QA_SPEC.enforce_endpoint_certificate else None
    )
    values = variant_observable_values(
        surface_dofs=arrays["surface_dofs"],
        coil_dofs=arrays["coil_dofs"],
        initial_parameters=initial_parameters,
        initial_objective=float(initial_objective),
        initial_gradient=np.asarray(initial_gradient, dtype=np.float64),
        initial_iota=problem.iota_target,
        initial_volume=problem.initial_volume,
        final_parameters=final_parameters,
        final_objective=endpoint.value,
        final_gradient=final_gradient,
        final_non_qs_ratio=endpoint.non_qs_ratio,
        final_iota=endpoint.iota,
        final_volume=endpoint.volume,
        final_major_radius_penalty=endpoint.major_radius_penalty,
        final_length_penalty=endpoint.length_penalty,
        final_boozer_residual=endpoint.boozer_residual,
        inner_solver_success=endpoint.inner_success,
        outer_solver_success=outer_solver_success,
        outer_solver_status=optimizer_status,
        report_residual=BOOZER_QA_SPEC.report_residual,
        endpoint_certificate=endpoint_certificate,
    )
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    return variant_lane_observation(
        lane,
        bundle,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        driver=JAX_PARITY_DRIVER_ID,
        workflow_stages=BOOZER_QA_SPEC.workflow_stages,
        solver_counts=(
            int(optimizer_result.k),
            int(optimizer_result.nfev),
            int(optimizer_result.ngev),
        ),
        endpoint_certificate=endpoint_certificate,
        outer_stopping_reason=outer_certificate.stopping_reason,
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the matched Boozer-QA coil optimization."""
    if lane == "native-cpu":
        return execute_variant(lane, bundle, arrays, BOOZER_QA_SPEC)
    return _execute_official_jax(lane, bundle, arrays)


def execute_variant(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    spec: BoozerSingleStageSpec,
) -> LaneObservation:
    """Execute one configured native/JAX Boozer single-stage workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays, spec)
    return _jax(lane, bundle, arrays, spec)
