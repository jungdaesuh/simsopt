"""Exact matched workflow for ``1_Simple/minimize_curve_length.py``."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from examples.jax.official_tiny_least_squares import (
    CONTROLLED_CURVE_INITIAL_FULL,
    CONTROLLED_CURVE_REPLAY_SEED,
    DRIVER_CURVE,
    curve_length_residual,
    guard_finite_endpoint,
    solve_jax_residual,
    trf_outcome,
    value_and_jacobian,
)
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases._official_least_squares import (
    declared_work_budget,
    official_stopping_keywords,
    solve_official_least_squares,
)
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt.geo import CurveLength, CurveRZFourier
from simsopt.objectives import LeastSquaresProblem
from simsopt_jax.examples import ExecutionScale

import jax

WORKFLOW_STAGES = (
    "construct_fourier_curve_problem",
    "evaluate_initial_length_residual_jacobian_and_objective",
    "solve_curve_length_least_squares_problem",
    "evaluate_final_length_residual_jacobian_and_objective",
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Materialize one deterministic realization of the native random state."""
    # The upstream example draws from the process-global RNG without seeding.
    # This parity case persists the official controlled replay realization so native
    # and JAX lanes consume identical curve inputs on every run.
    initial_full_parameters = np.asarray(CONTROLLED_CURVE_INITIAL_FULL)
    return create_input_bundle(
        root,
        case_id="native-minimize-curve-length",
        random_seed=CONTROLLED_CURVE_REPLAY_SEED,
        arrays={"initial_full_parameters": initial_full_parameters},
        configuration={
            "nquadrature": 100,
            "nfourier": 4,
            "nfp": 5,
            "stellsym": True,
            "fixed_dof": 0,
            "max_steps": 512 if scale == "bounded" else 2048,
        },
        scale=scale,
    )


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


def _build_curve(bundle: InputBundle, initial_full_parameters: np.ndarray):
    curve = CurveRZFourier(
        _configuration_int(bundle, "nquadrature"),
        _configuration_int(bundle, "nfourier"),
        _configuration_int(bundle, "nfp"),
        bool(bundle.configuration["stellsym"]),
    )
    curve.x = initial_full_parameters
    curve.fix(_configuration_int(bundle, "fixed_dof"))
    return curve


def _effective_fingerprint(bundle: InputBundle, curve) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "full_dofs": np.asarray(curve.local_full_x).tolist(),
            "free_positions": np.flatnonzero(curve.local_dofs_free_status).tolist(),
            "quadpoints": np.asarray(curve.quadpoints).tolist(),
            "nfourier": curve.order,
            "nfp": curve.nfp,
            "stellsym": curve.stellsym,
            "max_steps": bundle.configuration["max_steps"],
        },
    )


def _state(
    prefix: str,
    parameters: np.ndarray,
    length: float,
    residual_jacobian: np.ndarray,
) -> dict[str, np.ndarray]:
    residual = np.asarray((length,), dtype=np.float64)
    return {
        f"{prefix}:parameters": parameters,
        f"{prefix}:length": np.asarray(length, dtype=np.float64),
        f"{prefix}:residual": residual,
        f"{prefix}:residual_jacobian": residual_jacobian[np.newaxis, :],
        f"{prefix}:objective_sum_squares": np.asarray(
            length * length,
            dtype=np.float64,
        ),
        f"{prefix}:objective_gradient": 2.0 * length * residual_jacobian,
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    curve = _build_curve(bundle, arrays["initial_full_parameters"])
    objective = CurveLength(curve)
    problem = LeastSquaresProblem.from_tuples([(objective.J, 0.0, 1.0)])
    effective_fingerprint = _effective_fingerprint(bundle, curve)
    initial_parameters = np.asarray(problem.x, dtype=np.float64)
    initial_length = float(objective.J())
    initial_residual_jacobian = np.asarray(objective.dJ(), dtype=np.float64)
    budget = declared_work_budget(bundle.scale, _configuration_int(bundle, "max_steps"))
    result = solve_official_least_squares(problem, **official_stopping_keywords(budget))
    final_parameters = np.asarray(problem.x, dtype=np.float64)
    final_length = float(objective.J())
    final_residual_jacobian = np.asarray(objective.dJ(), dtype=np.float64)
    # The official wrapper discards SciPy's result; the recorder keeps it, so this
    # lane publishes the provider's own outcome. Official capture
    # A/reference-simple/runs/native-minimize-curve-length/captured-controlled-omp1:
    # status 2 `ftol`, nfev 62, njev 55, fun = 18.849556246039942.
    outcome = guard_finite_endpoint(
        trf_outcome(result),
        (
            np.asarray(final_length, dtype=np.float64),
            final_parameters,
            final_residual_jacobian,
        ),
    )
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=effective_fingerprint,
        driver="simsopt_least_squares_serial_solve",
        normalized_status=outcome.normalized_status,
        raw_status=outcome.raw_status,
        success=outcome.success,
        nit=None,
        nfev=outcome.nfev,
        njev=outcome.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **_state(
                "initial",
                initial_parameters,
                initial_length,
                initial_residual_jacobian,
            ),
            **_state(
                "final",
                final_parameters,
                final_length,
                final_residual_jacobian,
            ),
        },
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    curve = _build_curve(bundle, arrays["initial_full_parameters"])
    effective_fingerprint = _effective_fingerprint(bundle, curve)
    full_dofs = np.asarray(curve.local_full_x, dtype=np.float64)
    free_positions = np.flatnonzero(curve.local_dofs_free_status)
    residual = curve_length_residual(
        full_dofs,
        np.asarray(curve.quadpoints, dtype=np.float64),
        free_positions,
        order=curve.order,
        nfp=curve.nfp,
        stellsym=curve.stellsym,
    )
    initial_parameters_host = full_dofs[free_positions]
    initial_value, initial_jacobian = value_and_jacobian(
        residual, initial_parameters_host
    )
    budget = declared_work_budget(bundle.scale, _configuration_int(bundle, "max_steps"))
    optimizer = solve_jax_residual(residual, initial_parameters_host, max_nfev=budget)
    initial_length = float(initial_value[0])
    initial_jacobian_host = initial_jacobian[0]
    final_parameters_host = np.asarray(optimizer.x, dtype=np.float64)
    final_value, final_jacobian = value_and_jacobian(residual, final_parameters_host)
    final_length = float(final_value[0])
    final_jacobian_host = final_jacobian[0]
    outcome = guard_finite_endpoint(
        trf_outcome(optimizer),
        (
            np.asarray(final_length, dtype=np.float64),
            final_parameters_host,
            final_jacobian_host,
        ),
    )
    platform = jax.devices()[0].platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if bool(jax.config.read("jax_enable_x64")) else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=effective_fingerprint,
        driver=DRIVER_CURVE,
        normalized_status=outcome.normalized_status,
        raw_status=outcome.raw_status,
        success=outcome.success,
        nit=None,
        nfev=outcome.nfev,
        njev=outcome.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **_state(
                "initial",
                initial_parameters_host,
                initial_length,
                initial_jacobian_host,
            ),
            **_state(
                "final",
                final_parameters_host,
                final_length,
                final_jacobian_host,
            ),
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact source-adjacent native or JAX curve workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
