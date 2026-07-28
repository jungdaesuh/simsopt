"""Exact matched workflow for ``1_Simple/minimize_curve_length.py``."""

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
    "construct_fourier_curve_problem",
    "evaluate_initial_length_residual_jacobian_and_objective",
    "solve_curve_length_least_squares_problem",
    "evaluate_final_length_residual_jacobian_and_objective",
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Materialize one deterministic realization of the native random state."""
    initial_full_parameters = np.random.RandomState(0).rand(9) - 0.5
    initial_full_parameters[0] = 3.0
    return create_input_bundle(
        root,
        case_id="native-minimize-curve-length",
        random_seed=0,
        arrays={"initial_full_parameters": initial_full_parameters},
        configuration={
            "nquadrature": 100,
            "nfourier": 4,
            "nfp": 5,
            "stellsym": True,
            "fixed_dof": 0,
            "rtol": 1.0e-10,
            "atol": 1.0e-8,
            "max_steps": 512 if scale == "bounded" else 2048,
        },
        scale=scale,
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


def _build_curve(bundle: InputBundle, initial_full_parameters: np.ndarray):
    from simsopt.geo import CurveRZFourier

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
            "rtol": bundle.configuration["rtol"],
            "atol": bundle.configuration["atol"],
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
    from simsopt.geo import CurveLength
    from simsopt.objectives import LeastSquaresProblem
    from simsopt.solve import least_squares_serial_solve

    curve = _build_curve(bundle, arrays["initial_full_parameters"])
    objective = CurveLength(curve)
    problem = LeastSquaresProblem.from_tuples([(objective.J, 0.0, 1.0)])
    effective_fingerprint = _effective_fingerprint(bundle, curve)
    initial_parameters = np.asarray(problem.x, dtype=np.float64)
    initial_length = float(objective.J())
    initial_residual_jacobian = np.asarray(objective.dJ(), dtype=np.float64)
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
    final_parameters = np.asarray(problem.x, dtype=np.float64)
    final_length = float(objective.J())
    final_residual_jacobian = np.asarray(objective.dJ(), dtype=np.float64)
    circle_oracle = 2.0 * np.pi * arrays["initial_full_parameters"][0]
    final_objective_gradient = 2.0 * final_length * final_residual_jacobian
    success = bool(
        np.isclose(final_length, circle_oracle, rtol=1.0e-9, atol=1.0e-9)
        and np.linalg.norm(final_objective_gradient, ord=np.inf) <= 2.0e-2
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
        normalized_status="converged" if success else "failed",
        raw_status="circle_and_gradient_threshold",
        success=success,
        nit=None,
        nfev=None,
        njev=None,
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
    import jax
    import jax.numpy as jnp

    from simsopt_jax.core.curve_geometry import (
        curve_incremental_arclength_from_spec,
        curve_spec_with_dofs,
    )
    from simsopt_jax.core.specs import make_curve_rzfourier_spec
    from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax
    from simsopt_jax_adapters.geo.curve_objectives import curve_length_pure

    curve = _build_curve(bundle, arrays["initial_full_parameters"])
    effective_fingerprint = _effective_fingerprint(bundle, curve)
    full_dofs = np.asarray(curve.local_full_x, dtype=np.float64)
    free_positions = np.flatnonzero(curve.local_dofs_free_status)
    fixed_dofs = full_dofs.copy()
    fixed_dofs[free_positions] = 0.0
    expansion = np.zeros((full_dofs.size, free_positions.size), dtype=np.float64)
    expansion[free_positions, np.arange(free_positions.size)] = 1.0
    spec = make_curve_rzfourier_spec(
        dofs=jnp.asarray(full_dofs, dtype=jnp.float64),
        quadpoints=jnp.asarray(curve.quadpoints, dtype=jnp.float64),
        order=curve.order,
        nfp=curve.nfp,
        stellsym=curve.stellsym,
    )
    fixed_dofs_device = jnp.asarray(fixed_dofs, dtype=jnp.float64)
    expansion_device = jnp.asarray(expansion, dtype=jnp.float64)

    def residual(parameters: jax.Array) -> jax.Array:
        current_full = fixed_dofs_device + expansion_device @ parameters
        current_spec = curve_spec_with_dofs(spec, current_full)
        incremental_arclength = curve_incremental_arclength_from_spec(current_spec)
        return jnp.reshape(curve_length_pure(incremental_arclength), (1,))

    def objective(parameters: jax.Array) -> jax.Array:
        residual_value = residual(parameters)
        return jnp.vdot(residual_value, residual_value).real

    initial_device = jnp.asarray(curve.x, dtype=jnp.float64)
    residual_jacobian = jax.jit(jax.jacfwd(residual))
    initial_residual = residual(initial_device)
    initial_jacobian = residual_jacobian(initial_device)
    problem = TraceableScalarProblem(objective_fn=objective, x=initial_device)
    optimizer = serial_solve_jax(
        problem,
        rtol=_configuration_float(bundle, "rtol"),
        atol=_configuration_float(bundle, "atol"),
        max_steps=_configuration_int(bundle, "max_steps"),
        require_success=False,
    )
    final_device = problem.x
    final_residual = residual(final_device)
    final_jacobian = residual_jacobian(final_device)
    completed = jax.block_until_ready(
        (
            initial_device,
            initial_residual,
            initial_jacobian,
            final_device,
            final_residual,
            final_jacobian,
        )
    )

    def host(value: jax.Array) -> np.ndarray:
        return np.asarray(jax.device_get(value), dtype=np.float64)

    initial_parameters_host = host(completed[0])
    initial_residual_host = host(completed[1])
    initial_jacobian_host = host(completed[2])[0]
    final_parameters_host = host(completed[3])
    final_residual_host = host(completed[4])
    final_jacobian_host = host(completed[5])[0]
    circle_oracle = 2.0 * np.pi * arrays["initial_full_parameters"][0]
    final_objective_gradient = 2.0 * float(final_residual_host[0]) * final_jacobian_host
    scientific_success = bool(
        np.isclose(
            final_residual_host[0],
            circle_oracle,
            rtol=1.0e-9,
            atol=1.0e-9,
        )
        and np.linalg.norm(final_objective_gradient, ord=np.inf) <= 2.0e-5
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
        driver=optimizer.driver.value,
        normalized_status="converged" if scientific_success else "failed",
        raw_status=str(optimizer.status),
        success=scientific_success,
        nit=optimizer.nit,
        nfev=optimizer.nfev,
        njev=optimizer.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **_state(
                "initial",
                initial_parameters_host,
                float(initial_residual_host[0]),
                initial_jacobian_host,
            ),
            **_state(
                "final",
                final_parameters_host,
                float(final_residual_host[0]),
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
