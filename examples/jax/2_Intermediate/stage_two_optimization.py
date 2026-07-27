"""JAX port of ``examples/2_Intermediate/stage_two_optimization.py``.

The host constructs the native Landreman-Paul QA surface, Fourier coils,
currents, and symmetry graph.  ``BiotSavartJAX`` snapshots that graph once;
quadratic flux, length, separation, curvature, and mean-squared-curvature
terms plus both BFGS stages then execute on the selected JAX device.  Only the
accepted state and final diagnostics return to the host.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.examples import ExampleResult, run_example, scalar_example_driver
from simsopt_jax.objectives import StageTwoObjectiveConfig, make_stage_two_objective
from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "native-stage-two-optimization"
NATIVE_ITERATIONS = 400
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _build_problem(
    max_steps: int,
) -> tuple[BiotSavartJAX, SquaredFluxJAX, jax.Array, jax.Array]:
    native_scale = max_steps >= NATIVE_ITERATIONS
    surface_resolution = 32 if native_scale else 4
    curve_order = 5 if native_scale else 3
    curve_quadrature = 100 if native_scale else 24
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="half period",
        nphi=surface_resolution,
        ntheta=surface_resolution,
    )
    base_curves = create_equally_spaced_curves(
        4,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=curve_order,
        numquadpoints=curve_quadrature,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    base_currents[0].fix_all()
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        True,
    )
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3))
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3))
    )
    return field, flux, surface_gamma, surface_normal


def _objective_config(length_weight: float) -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(
        num_base_curves=4,
        length_weight=length_weight,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=1000.0,
        curve_surface_minimum_distance=0.3,
        curve_surface_weight=10.0,
        curvature_threshold=5.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=1.0e-6,
    )


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    field, flux, surface_gamma, surface_normal = _build_problem(max_steps)
    flux_objective = flux.traceable_objective()
    first_objective = make_stage_two_objective(
        field,
        flux_objective,
        surface_gamma,
        surface_normal,
        _objective_config(1.0e-6),
    )
    second_objective = make_stage_two_objective(
        field,
        flux_objective,
        surface_gamma,
        surface_normal,
        _objective_config(1.0e-7),
    )
    initial_device = jax.device_put(np.asarray(field.x, dtype=np.float64))
    flux_problem = TraceableScalarProblem(
        objective_fn=flux_objective,
        x=initial_device,
    )
    driver = scalar_example_driver()
    problem = TraceableScalarProblem(objective_fn=first_objective, x=initial_device)
    initial_objective_device, initial_gradient_device = problem.value_and_grad(
        initial_device
    )
    first_result = serial_solve_jax(
        problem,
        driver=driver,
        max_steps=max_steps,
        rtol=1.0e-8,
        atol=1.0e-7,
    )
    problem = TraceableScalarProblem(objective_fn=second_objective, x=problem.x)
    second_result = serial_solve_jax(
        problem,
        driver=driver,
        max_steps=max_steps,
        rtol=1.0e-8,
        atol=1.0e-7,
    )
    solution_device = jax.block_until_ready(problem.x)
    final_objective_device, final_gradient_device = problem.value_and_grad(
        solution_device
    )
    squared_flux_device = flux_problem.objective(solution_device)
    initial = np.asarray(jax.device_get(initial_device), dtype=np.float64)
    initial_gradient = np.asarray(
        jax.device_get(initial_gradient_device), dtype=np.float64
    )
    solution = np.asarray(jax.device_get(solution_device), dtype=np.float64)
    final_gradient = np.asarray(jax.device_get(final_gradient_device), dtype=np.float64)
    scalars = np.asarray(
        jax.device_get(
            jnp.stack(
                (
                    initial_objective_device,
                    final_objective_device,
                    squared_flux_device,
                )
            )
        ),
        dtype=np.float64,
    )
    initial_objective, final_objective, squared_flux = (
        float(value) for value in scalars
    )
    field.x = solution
    solver_success = bool(first_result.success and second_result.success)
    scientific_success = bool(
        solver_success
        and np.all(np.isfinite(solution))
        and np.all(np.isfinite(final_gradient))
        and np.isfinite(squared_flux)
        and final_objective < initial_objective
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
            "solver_success": solver_success,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-stage-two-optimization-",
        bounded_steps=200,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
