"""JAX mirror of ``examples/1_Simple/stage_two_optimization_minimal.py``.

Native SIMSOPT constructs the fixed target surface, Fourier coils, currents,
and symmetry ownership.  ``BiotSavartJAX`` snapshots those objects into an
immutable DOF-to-field program.  Quadratic flux, the one-sided total-length
penalty, gradients, and the complete bounded optimization then execute on the
selected JAX device.  The accepted DOFs are published back only for reporting.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "native-stage-two-optimization-minimal"
LENGTH_WEIGHT = 1.0
LENGTH_TARGET = 8.0


def _build_problem() -> tuple[BiotSavartJAX, SquaredFluxJAX]:
    curves = create_equally_spaced_curves(
        2,
        1,
        stellsym=False,
        R0=1.0,
        R1=0.5,
        order=1,
        numquadpoints=16,
    )
    currents = [Current(1.0) * 1.0e5 for _ in curves]
    currents[0].fix_all()
    coils = coils_via_symmetries(curves, currents, 1, False)
    quadrature = np.linspace(0.0, 1.0, 8, endpoint=False)
    surface = SurfaceRZFourier(
        nfp=1,
        stellsym=False,
        mpol=1,
        ntor=0,
        quadpoints_phi=quadrature,
        quadpoints_theta=quadrature,
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)
    surface.fix_all()
    field = BiotSavartJAX(coils)
    return field, SquaredFluxJAX(surface, field)


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    field, flux = _build_problem()
    flux_objective = flux.traceable_objective()

    @jax.jit
    def objective(parameters: jax.Array) -> jax.Array:
        coil_set = field.coil_set_spec_from_dofs(parameters)
        total_length = jnp.asarray(0.0, dtype=jnp.float64)
        for _gammas, gammadashs, _currents in coil_set.field_inputs():
            total_length = total_length + jnp.sum(
                jnp.mean(jnp.linalg.norm(gammadashs, axis=-1), axis=-1)
            )
        excess_length = jnp.maximum(total_length - LENGTH_TARGET, 0.0)
        return flux_objective(parameters) + (
            0.5 * LENGTH_WEIGHT * excess_length * excess_length
        )

    value_and_gradient = jax.jit(jax.value_and_grad(objective))

    @jax.jit
    def components(parameters: jax.Array) -> tuple[jax.Array, jax.Array]:
        coil_set = field.coil_set_spec_from_dofs(parameters)
        total_length = jnp.asarray(0.0, dtype=jnp.float64)
        for _gammas, gammadashs, _currents in coil_set.field_inputs():
            total_length = total_length + jnp.sum(
                jnp.mean(jnp.linalg.norm(gammadashs, axis=-1), axis=-1)
            )
        excess_length = jnp.maximum(total_length - LENGTH_TARGET, 0.0)
        return flux_objective(parameters), (
            0.5 * LENGTH_WEIGHT * excess_length * excess_length
        )

    initial_device = jax.device_put(np.asarray(field.x, dtype=np.float64))
    initial_objective_device, initial_gradient_device = value_and_gradient(
        initial_device
    )
    problem = TraceableScalarProblem(objective_fn=objective, x=initial_device)
    solver_result = serial_solve_jax(
        problem,
        max_steps=max_steps,
        rtol=1.0e-9,
        atol=1.0e-7,
    )
    solution_device = jax.block_until_ready(problem.x)
    final_objective_device, final_gradient_device = value_and_gradient(solution_device)
    flux_device, length_penalty_device = components(solution_device)
    initial = np.asarray(jax.device_get(initial_device), dtype=np.float64)
    initial_gradient = np.asarray(
        jax.device_get(initial_gradient_device), dtype=np.float64
    )
    solution = np.asarray(jax.device_get(solution_device), dtype=np.float64)
    final_gradient = np.asarray(jax.device_get(final_gradient_device), dtype=np.float64)
    initial_objective = float(jax.device_get(initial_objective_device))
    final_objective = float(jax.device_get(final_objective_device))
    squared_flux = float(jax.device_get(flux_device))
    length_penalty = float(jax.device_get(length_penalty_device))
    field.x = solution
    scientific_success = bool(
        solver_result.success
        and np.isfinite(final_objective)
        and final_objective < initial_objective
        and np.linalg.norm(final_gradient, ord=np.inf) <= 1.0e-5
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_parameters": tuple(float(value) for value in initial),
            "initial_objective": initial_objective,
            "initial_gradient": tuple(float(value) for value in initial_gradient),
            "solution": tuple(float(value) for value in solution),
            "final_objective": final_objective,
            "final_gradient": tuple(float(value) for value in final_gradient),
            "squared_flux": squared_flux,
            "curve_length_penalty": length_penalty,
            "solver_success": solver_result.success,
            "solver_driver": solver_result.driver.value,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-stage-two-minimal-",
        bounded_steps=40,
        native_default_steps=300,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
