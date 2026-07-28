"""Matched native/JAX reconstruction of ``just_a_quadratic.py``."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from simsopt_jax.examples import ExecutionScale
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane

WORKFLOW_STAGES = (
    "construct_weighted_least_squares_problem",
    "evaluate_initial_residual_and_jacobian",
    "solve_least_squares_problem",
    "evaluate_final_residual_and_jacobian",
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Create the quadratic's canonical arrays and shared solve configuration."""
    return create_input_bundle(
        root,
        case_id="traceable-least-squares",
        random_seed=0,
        arrays={
            "initial_parameters": np.array([0.25, -1.0, 4.0], dtype=np.float64),
            "targets": np.array([1.0, 2.0, 3.0], dtype=np.float64),
            "weights": np.array([1.0, 2.0, 3.0], dtype=np.float64),
        },
        configuration={
            "rtol": 1.0e-10,
            "atol": 1.0e-10,
            "max_steps": 20 if scale == "bounded" else 100,
        },
        scale=scale,
    )


def _effective_fingerprint(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> str:
    payload = {
        "initial_parameters": arrays["initial_parameters"].tolist(),
        "initial_parameters_dtype": str(arrays["initial_parameters"].dtype),
        "targets": arrays["targets"].tolist(),
        "targets_dtype": str(arrays["targets"].dtype),
        "weights": arrays["weights"].tolist(),
        "weights_dtype": str(arrays["weights"].dtype),
        "rtol": bundle.configuration["rtol"],
        "atol": bundle.configuration["atol"],
        "max_steps": bundle.configuration["max_steps"],
    }
    return effective_construction_fingerprint(bundle, payload)


def _values(
    initial_parameters: np.ndarray,
    final_parameters: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray,
) -> dict[str, np.ndarray]:
    residual_jacobian = np.diag(np.sqrt(weights))

    def state(prefix: str, parameters: np.ndarray) -> dict[str, np.ndarray]:
        residual = np.sqrt(weights) * (parameters - targets)
        objective = np.asarray(np.dot(residual, residual), dtype=np.float64)
        gradient = 2.0 * residual_jacobian.T @ residual
        return {
            f"{prefix}:residual": residual,
            f"{prefix}:residual_jacobian": residual_jacobian,
            f"{prefix}:objective_sum_squares": objective,
            f"{prefix}:solver_cost": 0.5 * objective,
            f"{prefix}:objective_gradient": gradient,
        }

    return {
        **state("initial", initial_parameters),
        "final:parameters": final_parameters,
        **state("final", final_parameters),
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from scipy.optimize import least_squares
    from simsopt.objectives import LeastSquaresProblem
    from simsopt.objectives.functions import Identity

    initial = arrays["initial_parameters"]
    targets = arrays["targets"]
    weights = arrays["weights"]
    identities = [Identity(float(value)) for value in initial]
    problem = LeastSquaresProblem.from_tuples(
        [
            (identity.f, float(target), float(weight))
            for identity, target, weight in zip(identities, targets, weights)
        ]
    )
    residual_jacobian = np.diag(np.sqrt(weights))
    result = least_squares(
        problem.residuals,
        initial,
        jac=lambda parameters: residual_jacobian,
        ftol=float(bundle.configuration["rtol"]),
        xtol=float(bundle.configuration["atol"]),
        gtol=min(
            float(bundle.configuration["rtol"]),
            float(bundle.configuration["atol"]),
        ),
        max_nfev=int(bundle.configuration["max_steps"]),
    )
    normalized = (
        "converged"
        if result.success
        else "budget_exhausted"
        if result.status == 0
        else "failed"
    )
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver="scipy_least_squares",
        normalized_status=normalized,
        raw_status=str(result.status),
        success=bool(result.success),
        nit=None,
        nfev=int(result.nfev),
        njev=None if result.njev is None else int(result.njev),
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=_values(initial, np.asarray(result.x), targets, weights),
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax.solve.serial import (
        TraceableLeastSquaresProblem,
        least_squares_serial_solve_jax,
    )

    import jax
    import jax.numpy as jnp

    initial = arrays["initial_parameters"]
    targets = arrays["targets"]
    weights = arrays["weights"]
    square_root_weights = np.sqrt(weights)

    def residual(parameters):
        return square_root_weights * (parameters - targets)

    problem = TraceableLeastSquaresProblem(
        residual_fn=residual,
        x=jnp.asarray(initial),
    )
    result = least_squares_serial_solve_jax(
        problem,
        rtol=float(bundle.configuration["rtol"]),
        atol=float(bundle.configuration["atol"]),
        max_steps=int(bundle.configuration["max_steps"]),
    )
    final = np.asarray(jax.device_get(jax.block_until_ready(problem.x)))
    platform = jax.devices()[0].platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if jax.config.jax_enable_x64 else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver=result.driver.value,
        normalized_status="converged" if result.success else "failed",
        raw_status=str(result.status),
        success=result.success,
        nit=result.nit,
        nfev=result.nfev,
        njev=result.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=_values(initial, final, targets, weights),
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the matched quadratic using the selected public lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
