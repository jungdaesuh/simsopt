"""Exact matched workflow for ``1_Simple/just_a_quadratic.py``."""

from __future__ import annotations

import os
from numbers import Real
from pathlib import Path
from typing import cast

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples import ExecutionScale

WORKFLOW_STAGES = (
    "construct_three_identity_optimizable_objects",
    "construct_weighted_least_squares_problem",
    "evaluate_initial_residual_jacobian_and_objective",
    "solve_least_squares_problem",
    "evaluate_final_parameters_residual_jacobian_and_objective",
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Materialize the exact native constants once for every execution lane."""
    return create_input_bundle(
        root,
        case_id="native-just-a-quadratic",
        random_seed=0,
        arrays={
            "initial_parameters": np.zeros(3, dtype=np.float64),
            "targets": np.asarray((1.0, 2.0, 3.0), dtype=np.float64),
            "weights": np.asarray((1.0, 2.0, 3.0), dtype=np.float64),
        },
        configuration={
            "rtol": 1.0e-12,
            "atol": 1.0e-12,
            "max_steps": 32 if scale == "bounded" else 128,
        },
        scale=scale,
    )


def _effective_fingerprint(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> str:
    return effective_construction_fingerprint(
        bundle,
        {
            "initial_parameters": arrays["initial_parameters"].tolist(),
            "targets": arrays["targets"].tolist(),
            "weights": arrays["weights"].tolist(),
            "dtype": "float64",
            "rtol": bundle.configuration["rtol"],
            "atol": bundle.configuration["atol"],
            "max_steps": bundle.configuration["max_steps"],
        },
    )


def _configuration_float(bundle: InputBundle, name: str) -> float:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"configuration {name} must be numeric")
    return float(value)


def _configuration_int(bundle: InputBundle, name: str) -> int:
    value = bundle.configuration[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"configuration {name} must be an integer")
    return value


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
            f"{prefix}:objective_gradient": gradient,
        }

    return {
        "initial:parameters": initial_parameters,
        **state("initial", initial_parameters),
        "final:parameters": final_parameters,
        **state("final", final_parameters),
    }


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from simsopt.objectives import LeastSquaresProblem
    from simsopt.objectives.functions import Identity
    from simsopt.solve import least_squares_serial_solve

    initial = arrays["initial_parameters"]
    targets = arrays["targets"]
    weights = arrays["weights"]
    identities = tuple(Identity() for _ in initial)
    problem = LeastSquaresProblem.from_tuples(
        [
            (
                identity.f,
                cast(Real, float(target)),
                cast(Real, float(weight)),
            )
            for identity, target, weight in zip(
                identities, targets, weights, strict=True
            )
        ]
    )
    least_squares_serial_solve(
        problem,
        ftol=_configuration_float(bundle, "rtol"),
        xtol=_configuration_float(bundle, "atol"),
        gtol=min(
            _configuration_float(bundle, "rtol"),
            _configuration_float(bundle, "atol"),
        ),
        max_nfev=_configuration_int(bundle, "max_steps"),
    )
    final = np.asarray(problem.x, dtype=np.float64)
    final_residual = np.sqrt(weights) * (final - targets)
    success = bool(
        np.linalg.norm(final_residual) <= 1.0e-8
        and np.allclose(final, targets, rtol=1.0e-10, atol=1.0e-12)
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
        driver="simsopt_least_squares_serial_solve",
        normalized_status="converged" if success else "failed",
        raw_status="objective_threshold",
        success=success,
        nit=None,
        nfev=None,
        njev=None,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=_values(initial, final, targets, weights),
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax.examples import solve_weighted_quadratic

    import jax
    import jax.numpy as jnp

    initial = arrays["initial_parameters"]
    targets = arrays["targets"]
    weights = arrays["weights"]
    device_result = solve_weighted_quadratic(
        initial_parameters=jnp.asarray(initial, dtype=jnp.float64),
        targets=jnp.asarray(targets, dtype=jnp.float64),
        weights=jnp.asarray(weights, dtype=jnp.float64),
        rtol=_configuration_float(bundle, "rtol"),
        atol=_configuration_float(bundle, "atol"),
        max_steps=_configuration_int(bundle, "max_steps"),
    )
    final = np.asarray(
        jax.device_get(device_result.final_parameters),
        dtype=np.float64,
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
        effective_construction_fingerprint=_effective_fingerprint(bundle, arrays),
        driver=device_result.optimizer.driver.value,
        normalized_status=(
            "converged" if device_result.optimizer.success else "failed"
        ),
        raw_status=str(device_result.optimizer.status),
        success=device_result.optimizer.success,
        nit=device_result.optimizer.nit,
        nfev=device_result.optimizer.nfev,
        njev=device_result.optimizer.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values=_values(initial, final, targets, weights),
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the exact native source-adjacent or public JAX workflow."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
