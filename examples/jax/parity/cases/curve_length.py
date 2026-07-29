"""Matched native/JAX bounded curve-length minimization."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.input_bundle import (
    InputBundle,
    create_input_bundle,
    effective_construction_fingerprint,
)
from examples.jax.parity.runtime import ParityLane
from simsopt_jax.examples import ExecutionScale

_DEFORMATION_NAMES = ("xs(2)", "yc(2)", "zs(2)")
WORKFLOW_STAGES = (
    "construct_fourier_curve_problem",
    "evaluate_initial_length_and_gradient",
    "optimize_curve_length",
    "evaluate_final_length_and_gradient",
)


def create_input(root: Path, scale: ExecutionScale) -> InputBundle:
    """Create the curve's canonical free state and solve configuration."""
    return create_input_bundle(
        root,
        case_id="curve-length-optimization",
        random_seed=0,
        arrays={"initial_parameters": np.array([0.3, -0.2, 0.4], dtype=np.float64)},
        configuration={
            "radius": 2.0,
            "nquadpoints": 64,
            "order": 2,
            "deformation_names": list(_DEFORMATION_NAMES),
            "rtol": 1.0e-10,
            "atol": 1.0e-8,
            "max_steps": 32 if scale == "bounded" else 128,
        },
        scale=scale,
    )


def _build_curve(bundle: InputBundle, initial: np.ndarray):
    from simsopt.geo.curvexyzfourier import CurveXYZFourier

    curve = CurveXYZFourier(
        int(bundle.configuration["nquadpoints"]),
        order=int(bundle.configuration["order"]),
    )
    radius = float(bundle.configuration["radius"])
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    curve.set("xs(2)", float(initial[0]))
    curve.set("yc(2)", float(initial[1]))
    curve.set("zs(2)", float(initial[2]))
    curve.fix_all()
    deformation_names = tuple(bundle.configuration["deformation_names"])
    if deformation_names != _DEFORMATION_NAMES:
        raise ValueError("effective construction rejected deformation_names drift")
    for name in deformation_names:
        curve.unfix(name)
    curve.x = initial
    return curve


def _effective_fingerprint(bundle: InputBundle, curve) -> str:
    payload = {
        "full_dofs": np.asarray(curve.local_full_x).tolist(),
        "free_positions": np.flatnonzero(curve.local_dofs_free_status).tolist(),
        "deformation_names": list(bundle.configuration["deformation_names"]),
        "nquadpoints": int(bundle.configuration["nquadpoints"]),
        "order": int(bundle.configuration["order"]),
        "rtol": bundle.configuration["rtol"],
        "atol": bundle.configuration["atol"],
        "max_steps": bundle.configuration["max_steps"],
    }
    return effective_construction_fingerprint(bundle, payload)


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from scipy.optimize import minimize
    from simsopt.geo import CurveLength

    initial = arrays["initial_parameters"]
    curve = _build_curve(bundle, initial)
    objective = CurveLength(curve)
    effective_fingerprint = _effective_fingerprint(bundle, curve)

    def value(parameters: np.ndarray) -> float:
        curve.x = parameters
        return float(objective.J())

    def gradient(parameters: np.ndarray) -> np.ndarray:
        curve.x = parameters
        return np.asarray(objective.dJ(), dtype=np.float64)

    initial_value = np.asarray(value(initial), dtype=np.float64)
    initial_gradient = gradient(initial)
    result = minimize(
        value,
        initial,
        jac=gradient,
        method="BFGS",
        options={
            "gtol": float(bundle.configuration["atol"]),
            "maxiter": int(bundle.configuration["max_steps"]),
        },
    )
    final = np.asarray(result.x, dtype=np.float64)
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=effective_fingerprint,
        driver="scipy_bfgs",
        normalized_status="converged" if result.success else "failed",
        raw_status=str(result.status),
        success=bool(result.success),
        nit=int(result.nit),
        nfev=int(result.nfev),
        njev=int(result.njev),
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            "initial:objective": initial_value,
            "initial:objective_gradient": initial_gradient,
            "final:parameters": final,
            "final:objective": np.asarray(value(final), dtype=np.float64),
            "final:objective_gradient": gradient(final),
        },
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax.core.curve_geometry import curve_incremental_arclength_from_spec
    from simsopt_jax.core.specs import CurveXYZFourierSpec
    from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax
    from simsopt_jax_adapters.geo.curve_objectives import curve_length_pure

    import jax
    import jax.numpy as jnp

    initial = arrays["initial_parameters"]
    curve = _build_curve(bundle, initial)
    effective_fingerprint = _effective_fingerprint(bundle, curve)
    full_dofs = np.asarray(curve.local_full_x, dtype=np.float64)
    curve_spec = CurveXYZFourierSpec(
        dofs=full_dofs,
        quadpoints=np.asarray(curve.quadpoints, dtype=np.float64),
        order=curve.order,
    )
    free_positions = np.flatnonzero(curve.local_dofs_free_status)
    fixed = full_dofs.copy()
    fixed[free_positions] = 0.0
    expansion = np.zeros((full_dofs.size, free_positions.size), dtype=np.float64)
    expansion[free_positions, np.arange(free_positions.size)] = 1.0

    def value(parameters):
        current_dofs = fixed + expansion @ parameters
        current_spec = dataclasses.replace(curve_spec, dofs=current_dofs)
        return curve_length_pure(curve_incremental_arclength_from_spec(current_spec))

    initial_device = jnp.asarray(initial)
    value_and_gradient = jax.jit(jax.value_and_grad(value))
    initial_value, initial_gradient = value_and_gradient(initial_device)
    problem = TraceableScalarProblem(objective_fn=value, x=initial_device)
    result = serial_solve_jax(
        problem,
        rtol=float(bundle.configuration["rtol"]),
        atol=float(bundle.configuration["atol"]),
        max_steps=int(bundle.configuration["max_steps"]),
    )
    final_device = jax.block_until_ready(problem.x)
    final_value, final_gradient = value_and_gradient(final_device)

    def host(value):
        return np.asarray(jax.device_get(jax.block_until_ready(value)))

    platform = jax.devices()[0].platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if jax.config.jax_enable_x64 else "fp32",
        scale=bundle.scale,
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=effective_fingerprint,
        driver=result.driver.value,
        normalized_status="converged" if result.success else "failed",
        raw_status=str(result.status),
        success=result.success,
        nit=result.nit,
        nfev=result.nfev,
        njev=result.njev,
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            "initial:objective": host(initial_value),
            "initial:objective_gradient": host(initial_gradient),
            "final:parameters": host(final_device),
            "final:objective": host(final_value),
            "final:objective_gradient": host(final_gradient),
        },
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the matched curve problem using the selected public lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
