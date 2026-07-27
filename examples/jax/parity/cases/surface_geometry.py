"""Matched native/JAX bounded torus area-volume least squares."""

from __future__ import annotations

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

WORKFLOW_STAGES = (
    "construct_area_volume_problem",
    "evaluate_initial_area_volume_residual_and_jacobian",
    "solve_area_volume_least_squares_problem",
    "evaluate_final_area_volume_state",
)


def create_input(root: Path, smoke: bool) -> InputBundle:
    """Create the torus free state, quadrature, targets, and solve policy."""
    major_radius = 1.0
    minor_radius = 0.2
    return create_input_bundle(
        root,
        case_id="surface-geometry-optimization",
        random_seed=0,
        arrays={
            "initial_parameters": np.array([0.15, 0.25], dtype=np.float64),
            "quadrature": np.linspace(0.0, 1.0, 32, endpoint=False),
            "targets": np.array(
                [
                    4.0 * np.pi**2 * major_radius * minor_radius,
                    2.0 * np.pi**2 * major_radius * minor_radius**2,
                ],
                dtype=np.float64,
            ),
        },
        configuration={
            "major_radius": major_radius,
            "mpol": 1,
            "ntor": 0,
            "nfp": 1,
            "stellsym": True,
            "rtol": 1.0e-12,
            "atol": 1.0e-12,
            "max_steps": 24 if smoke else 96,
        },
    )


def _build_surface(bundle: InputBundle, arrays: dict[str, np.ndarray]):
    from simsopt.geo import SurfaceRZFourier

    quadrature = arrays["quadrature"]
    surface = SurfaceRZFourier(
        mpol=int(bundle.configuration["mpol"]),
        ntor=int(bundle.configuration["ntor"]),
        nfp=int(bundle.configuration["nfp"]),
        stellsym=bool(bundle.configuration["stellsym"]),
        quadpoints_phi=quadrature,
        quadpoints_theta=quadrature,
    )
    surface.set_rc(0, 0, float(bundle.configuration["major_radius"]))
    surface.set_rc(1, 0, float(arrays["initial_parameters"][0]))
    surface.set_zs(1, 0, float(arrays["initial_parameters"][1]))
    surface.fix_all()
    surface.unfix("rc(1,0)")
    surface.unfix("zs(1,0)")
    surface.x = arrays["initial_parameters"]
    return surface


def _effective_fingerprint(
    bundle: InputBundle, arrays: dict[str, np.ndarray], surface
) -> str:
    payload = {
        "full_dofs": np.asarray(surface.local_full_x).tolist(),
        "free_positions": np.flatnonzero(surface.local_dofs_free_status).tolist(),
        "quadpoints_phi": np.asarray(surface.quadpoints_phi).tolist(),
        "quadpoints_theta": np.asarray(surface.quadpoints_theta).tolist(),
        "targets": arrays["targets"].tolist(),
        "mpol": surface.mpol,
        "ntor": surface.ntor,
        "nfp": surface.nfp,
        "stellsym": surface.stellsym,
        "rtol": bundle.configuration["rtol"],
        "atol": bundle.configuration["atol"],
        "max_steps": bundle.configuration["max_steps"],
    }
    return effective_construction_fingerprint(bundle, payload)


def _parameter_invariants(parameters: np.ndarray) -> np.ndarray:
    """Return the quotient coordinates for interchangeable ellipse semi-axes."""
    values = np.asarray(parameters, dtype=np.float64)
    return np.asarray((np.sum(values), np.prod(values)), dtype=np.float64)


def _native(bundle: InputBundle, arrays: dict[str, np.ndarray]) -> LaneObservation:
    from scipy.optimize import least_squares

    surface = _build_surface(bundle, arrays)
    initial = arrays["initial_parameters"]
    targets = arrays["targets"]
    free_positions = np.flatnonzero(surface.local_dofs_free_status)
    effective_fingerprint = _effective_fingerprint(bundle, arrays, surface)

    def residual(parameters: np.ndarray) -> np.ndarray:
        surface.x = parameters
        return np.array([surface.area() - targets[0], surface.volume() - targets[1]])

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        surface.x = parameters
        return np.stack(
            (
                np.asarray(surface.darea())[free_positions],
                np.asarray(surface.dvolume())[free_positions],
            )
        )

    def state(prefix: str, parameters: np.ndarray) -> dict[str, np.ndarray]:
        current_residual = residual(parameters)
        current_jacobian = jacobian(parameters)
        objective = np.asarray(np.dot(current_residual, current_residual))
        return {
            f"{prefix}:residual": current_residual,
            f"{prefix}:residual_jacobian": current_jacobian,
            f"{prefix}:objective_sum_squares": objective,
            f"{prefix}:solver_cost": 0.5 * objective,
            f"{prefix}:objective_gradient": (
                2.0 * current_jacobian.T @ current_residual
            ),
            f"{prefix}:area": np.asarray(surface.area()),
            f"{prefix}:volume": np.asarray(surface.volume()),
        }

    initial_state = state("initial", initial)
    result = least_squares(
        residual,
        initial,
        jac=jacobian,
        ftol=float(bundle.configuration["rtol"]),
        xtol=float(bundle.configuration["atol"]),
        gtol=float(bundle.configuration["atol"]),
        max_nfev=int(bundle.configuration["max_steps"]),
    )
    final = np.asarray(result.x)
    return LaneObservation(
        lane="native-cpu",
        backend_mode="native_cpu",
        platform="cpu",
        precision="fp64",
        input_fingerprint=bundle.input_fingerprint,
        configuration_fingerprint=bundle.configuration_fingerprint,
        effective_construction_fingerprint=effective_fingerprint,
        driver="scipy_least_squares",
        normalized_status="converged" if result.success else "failed",
        raw_status=str(result.status),
        success=bool(result.success),
        nit=None,
        nfev=int(result.nfev),
        njev=None if result.njev is None else int(result.njev),
        completed_workflow_stages=WORKFLOW_STAGES,
        provenance=None,
        values={
            **initial_state,
            "final:parameters": final,
            "final:parameter_invariants": _parameter_invariants(final),
            **state("final", final),
        },
        applicability={"final:parameters": False},
    )


def _jax(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    from simsopt_jax.core.specs import SurfaceRZFourierSpec
    from simsopt_jax.solve.serial import (
        TraceableLeastSquaresProblem,
        least_squares_serial_solve_jax,
    )
    from simsopt_jax_adapters.geo.surface_objectives import (
        surface_area_jax_from_dofs,
        surface_volume_jax_from_dofs,
    )

    import jax
    import jax.numpy as jnp

    surface = _build_surface(bundle, arrays)
    effective_fingerprint = _effective_fingerprint(bundle, arrays, surface)
    full_dofs = np.asarray(surface.local_full_x)
    free_positions = np.flatnonzero(surface.local_dofs_free_status)
    fixed = full_dofs.copy()
    fixed[free_positions] = 0.0
    expansion = np.zeros((full_dofs.size, free_positions.size))
    expansion[free_positions, np.arange(free_positions.size)] = 1.0
    spec = SurfaceRZFourierSpec(
        rc=np.asarray(surface.rc),
        rs=np.asarray(surface.rs),
        zc=np.asarray(surface.zc),
        zs=np.asarray(surface.zs),
        quadpoints_phi=np.asarray(arrays["quadrature"]),
        quadpoints_theta=np.asarray(arrays["quadrature"]),
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        mpol=surface.mpol,
        ntor=surface.ntor,
    )
    targets = arrays["targets"]

    def quantities(parameters):
        current_dofs = fixed + expansion @ parameters
        return jnp.stack(
            (
                surface_area_jax_from_dofs(spec, current_dofs),
                surface_volume_jax_from_dofs(spec, current_dofs),
            )
        )

    def residual(parameters):
        return quantities(parameters) - targets

    def state_device(parameters):
        current_quantities = quantities(parameters)
        current_residual = residual(parameters)
        current_jacobian = jax.jacfwd(residual)(parameters)
        objective = jnp.vdot(current_residual, current_residual)
        gradient = 2.0 * current_jacobian.T @ current_residual
        return (
            current_quantities[0],
            current_quantities[1],
            current_residual,
            current_jacobian,
            objective,
            gradient,
        )

    compiled_state = jax.jit(state_device)

    def state(prefix: str, parameters) -> dict[str, np.ndarray]:
        (
            area,
            volume,
            current_residual,
            current_jacobian,
            objective,
            gradient,
        ) = compiled_state(parameters)

        def host(value):
            return np.asarray(jax.device_get(jax.block_until_ready(value)))

        objective_value = host(objective)
        return {
            f"{prefix}:residual": host(current_residual),
            f"{prefix}:residual_jacobian": host(current_jacobian),
            f"{prefix}:objective_sum_squares": objective_value,
            f"{prefix}:solver_cost": 0.5 * objective_value,
            f"{prefix}:objective_gradient": host(gradient),
            f"{prefix}:area": host(area),
            f"{prefix}:volume": host(volume),
        }

    initial = jnp.asarray(arrays["initial_parameters"])
    initial_state = state("initial", initial)
    problem = TraceableLeastSquaresProblem(residual_fn=residual, x=initial)
    result = least_squares_serial_solve_jax(
        problem,
        rtol=float(bundle.configuration["rtol"]),
        atol=float(bundle.configuration["atol"]),
        max_steps=int(bundle.configuration["max_steps"]),
    )
    final = jax.block_until_ready(problem.x)
    platform = jax.devices()[0].platform
    return LaneObservation(
        lane=lane,
        backend_mode=os.environ["SIMSOPT_BACKEND_MODE"],
        platform="gpu" if platform in {"cuda", "gpu"} else platform,
        precision="fp64" if jax.config.jax_enable_x64 else "fp32",
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
            **initial_state,
            "final:parameters": np.asarray(jax.device_get(final)),
            "final:parameter_invariants": _parameter_invariants(
                np.asarray(jax.device_get(final))
            ),
            **state("final", final),
        },
        applicability={"final:parameters": False},
    )


def execute(
    lane: ParityLane,
    bundle: InputBundle,
    arrays: dict[str, np.ndarray],
) -> LaneObservation:
    """Execute the matched surface problem using the selected public lane."""
    if lane == "native-cpu":
        return _native(bundle, arrays)
    return _jax(lane, bundle, arrays)
