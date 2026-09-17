"""Matched VMEC-free implicit-Boozer single-stage workflow.

The parity JAX lane uses the shipped example's exact analytic evaluator and
SciPy BFGS policy.  The separate measurement entry point retains the historical
traceable-session optimizer for trajectory and Optax instrumentation; its
observations are not the authority receipts published by ``run_parity.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from examples.jax.parity.arbiter import (
    SHIPPED_SINGLE_STAGE_SCIPY_DRIVER_ID,
    LaneObservation,
)
from examples.jax.parity.cases.native_boozerqa import (
    BoozerSingleStageSpec,
    build_variant_problem,
    create_variant_input,
    execute_variant,
    validate_variant_bundle_arrays,
    variant_lane_observation,
    variant_observable_values,
    variant_scale_configuration,
)
from examples.jax.parity.contracts import QualityBand
from examples.jax.parity.input_bundle import InputBundle
from examples.jax.parity.measurement import MeasurementExecution
from examples.jax.parity.runtime import ParityLane
from simsopt.geo import Volume
from simsopt.geo.curve import Curve
from simsopt.single_stage_boozer_vacuum import (
    NATIVE_ITERATIONS,
    OUTER_GRADIENT_TOLERANCE,
)
from simsopt_contracts.optimization_endpoint import certify_optimization_endpoint
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import ExecutionScale
from simsopt_jax.solve import Driver, ScipyBFGSOptions
from simsopt_jax.solve.dispatch import minimize
from simsopt_jax_adapters.geo.single_stage_boozer_vacuum_problem import (
    BOUNDED_SCALE,
    NATIVE_SCALE,
    SingleStageVacuumProblem,
)

import jax

WORKFLOW_STAGES = (
    "construct_ncsx_coils_and_volume_labelled_surface",
    "solve_initial_boozer_surface",
    "assemble_nonqs_residual_iota_radius_and_length_objective",
    "evaluate_initial_objective_and_gradient",
    "optimize_coils_and_currents_with_bfgs",
    "record_final_objective_gradient_and_implicit_physics_state",
)

SPEC = BoozerSingleStageSpec(
    case_id="native-single-stage-boozer-vacuum-optimization",
    workflow_stages=WORKFLOW_STAGES,
    bounded_resolution=1,
    native_resolution=6,
    inner_tolerance=1.0e-13,
    bounded_outer_maxiter=2,
    native_outer_maxiter=NATIVE_ITERATIONS,
    bounded_non_qs_sdim=4,
    native_non_qs_sdim=20,
    residual_weight=1.0,
    report_residual=True,
    enforce_endpoint_certificate=True,
)

# Rule 3 of the 2026-08-15 native_default certification-gate ruling: a
# budget-exhausted continuous optimizer supports endpoint quality only, not
# optimizer convergence or final-value equivalence. Bounded scale has no band.
#
# The 1e-7 ceiling was set above the worst 2026-08-14 three-lane endpoint
# (4.5614e-8) on the former traceable-session route. Pass-5b's shipped-example
# native/GPU pairs also ended below that ceiling at the same 1000-step budget.
# Neither packet establishes the new three-lane harness verdict; the ceiling
# remains an endpoint-quality limit until that run is audited.
NATIVE_DEFAULT_QUALITY_BAND = QualityBand(
    observable="final:objective",
    max_value=1.0e-07,
    derivation=(
        "2026-08-14 traceable-session three-lane native_default packet: "
        "worst 1000-step endpoint 4.5614e-08; ceiling set at 1e-07. "
        "2026-09-15 pass-5b shipped-example native/GPU pairs also met this "
        "ceiling; new three-lane shipped-route authority remains to be audited"
    ),
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Freeze the exact native/JAX single-stage construction."""
    return create_variant_input(root, scale, SPEC)


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the VMEC-free single-stage workflow in one isolated lane."""
    if lane == "native-cpu":
        return execute_variant(lane, bundle, arrays, SPEC)
    return _execute_shipped_jax(lane, bundle, arrays)


def _execute_shipped_jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    configuration = variant_scale_configuration(bundle.scale, SPEC)
    for key, expected in configuration.items():
        if bundle.configuration[key] != expected:
            raise ValueError(f"frozen single-stage {key} differs from shipped scale")
    (
        _base_curves,
        _base_currents,
        magnetic_axis,
        nfp,
        native_field,
        surface,
        initial_G,
    ) = build_variant_problem(bundle.configuration, bundle.scale)
    if (
        bundle.configuration["nfp"] != nfp
        or bundle.configuration["initial_G"] != initial_G
    ):
        raise ValueError("frozen single-stage NCSX construction differs")
    validate_variant_bundle_arrays(
        arrays,
        axis_dofs=np.asarray(cast(Curve, magnetic_axis).local_full_x, dtype=np.float64),
        coil_dofs=np.asarray(native_field.x, dtype=np.float64),
        surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
    )
    scale = NATIVE_SCALE if bundle.scale == "native_default" else BOUNDED_SCALE
    problem = SingleStageVacuumProblem(scale)
    initial_parameters = problem.initial_coil_dofs
    if not np.array_equal(initial_parameters, arrays["coil_dofs"]):
        raise ValueError("shipped JAX parameters do not match frozen coil dofs")
    initial_volume = float(Volume(surface).J())
    initial_objective, initial_gradient = problem.value_and_gradient(initial_parameters)
    max_steps = (
        SPEC.native_outer_maxiter
        if bundle.scale == "native_default"
        else SPEC.bounded_outer_maxiter
    )
    optimizer_result = minimize(
        problem.value_and_gradient,
        initial_parameters,
        driver=Driver.SCIPY_BFGS,
        options=ScipyBFGSOptions(
            maxiter=max_steps,
            gtol=OUTER_GRADIENT_TOLERANCE,
        ),
    )
    final_parameters = np.asarray(optimizer_result.x, dtype=np.float64)
    endpoint = problem.endpoint(final_parameters)
    final_gradient = endpoint.gradient
    parameters_finite = bool(
        np.all(np.isfinite(final_parameters)) and np.all(np.isfinite(final_gradient))
    )
    observables_finite = bool(
        np.isfinite(endpoint.value)
        and np.isfinite(endpoint.iota)
        and np.isfinite(endpoint.volume)
        and np.isfinite(endpoint.non_qs_ratio)
        and np.isfinite(endpoint.boozer_residual)
        and np.isfinite(endpoint.boozer_residual_rms)
    )
    certificate = certify_optimization_endpoint(
        status_convention="scipy-bfgs",
        provider_success=bool(optimizer_result.success),
        provider_status=int(optimizer_result.status),
        iterations=int(optimizer_result.nit),
        max_iterations=max_steps,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        final_gradient_inf_norm=float(np.max(np.abs(final_gradient))),
        parameters_finite=parameters_finite,
        observables_finite=observables_finite,
        inner_success=endpoint.inner_success,
    )
    values = variant_observable_values(
        surface_dofs=arrays["surface_dofs"],
        coil_dofs=arrays["coil_dofs"],
        initial_parameters=initial_parameters,
        initial_objective=float(initial_objective),
        initial_gradient=np.asarray(initial_gradient, dtype=np.float64),
        initial_iota=problem.iota_target,
        initial_volume=initial_volume,
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
        outer_solver_success=bool(optimizer_result.success),
        outer_solver_status=int(optimizer_result.status),
        report_residual=True,
        endpoint_certificate=certificate,
    )
    device = get_runtime_jax_device()
    platform = "cpu" if device is None else device.platform
    return variant_lane_observation(
        lane,
        bundle,
        values,
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.jax_enable_x64) else "fp32",
        driver=SHIPPED_SINGLE_STAGE_SCIPY_DRIVER_ID,
        workflow_stages=WORKFLOW_STAGES,
        solver_counts=(
            int(optimizer_result.nit),
            int(optimizer_result.nfev),
            int(optimizer_result.njev),
        ),
        endpoint_certificate=certificate,
    )


def execute_measurement(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
    measurement: MeasurementExecution,
) -> LaneObservation:
    """Execute one instrumented single-stage measurement lane."""
    return execute_variant(lane, bundle, arrays, SPEC, measurement=measurement)
