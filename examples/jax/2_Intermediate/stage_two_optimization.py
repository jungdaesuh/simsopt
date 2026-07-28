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
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import (
    ExampleResult,
    run_example,
    solve_standard_stage_two,
)
from simsopt_jax.objectives import StageTwoObjectiveConfig
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
    curve_order = 5 if native_scale else 2
    curve_quadrature = 100 if native_scale else 16
    surface = SurfaceRZFourier.from_vmec_input(
        str(TEST_DATA / "input.LandremanPaul2021_QA"),
        range="half period",
        nphi=surface_resolution,
        ntheta=surface_resolution,
    )
    surface.fix_all()
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


def _objective_config() -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(
        num_base_curves=4,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=1000.0,
        curve_surface_minimum_distance=0.3,
        curve_surface_weight=10.0,
        curvature_threshold=5.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=1.0e-6,
    )


def solve(output_directory: Path, max_steps: int) -> ExampleResult:
    field, flux, surface_gamma, surface_normal = _build_problem(max_steps)
    device = get_runtime_jax_device()
    initial_device = jax.device_put(
        np.asarray(field.x, dtype=np.float64),
        device,
    )
    taylor_direction_device = jax.device_put(
        np.random.RandomState(1).uniform(size=initial_device.shape),
        device,
    )
    first_length_weight_device = jax.device_put(
        np.asarray(1.0e-6, dtype=np.float64),
        device,
    )
    second_length_weight_device = jax.device_put(
        np.asarray(1.0e-7, dtype=np.float64),
        device,
    )
    device_result = solve_standard_stage_two(
        field=field,
        flux_spec=flux.fixed_surface_flux_spec(),
        surface_gamma=surface_gamma,
        surface_normal=surface_normal,
        initial_parameters=initial_device,
        taylor_direction=taylor_direction_device,
        regularization_config=_objective_config(),
        first_length_weight=first_length_weight_device,
        second_length_weight=second_length_weight_device,
        max_steps=max_steps,
        rtol=1.0e-12,
        atol=1.0e-10,
    )
    initial, first, final, taylor_errors = jax.device_get(
        (
            device_result.initial,
            device_result.first,
            device_result.final,
            device_result.taylor_errors,
        )
    )
    solution = np.asarray(final.parameters, dtype=np.float64)
    final_gradient = np.asarray(final.objective_gradient, dtype=np.float64)
    field.x = solution
    field.save(str(output_directory / "biot_savart_opt.json"))
    solver_success = bool(
        device_result.first_optimizer.success and device_result.second_optimizer.success
    )
    scientific_success = bool(
        device_result.first_optimizer.status in (0, 1)
        and device_result.second_optimizer.status in (0, 1)
        and np.all(np.isfinite(solution))
        and np.all(np.isfinite(final_gradient))
        and np.isfinite(final.squared_flux)
        and first.objective < initial.objective
        and final.objective < initial.objective
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_parameters": tuple(float(value) for value in initial.parameters),
            "initial_objective": float(initial.objective),
            "initial_gradient": tuple(
                float(value) for value in initial.objective_gradient
            ),
            "first_solution": tuple(float(value) for value in first.parameters),
            "first_objective": float(first.objective),
            "first_gradient": tuple(float(value) for value in first.objective_gradient),
            "solution": tuple(float(value) for value in solution),
            "final_objective": float(final.objective),
            "final_gradient": tuple(float(value) for value in final_gradient),
            "squared_flux": float(final.squared_flux),
            "geometric_penalty": float(final.geometric_penalty),
            "maximum_normal_field": float(final.maximum_normal_field),
            "total_curve_length": float(final.total_curve_length),
            "taylor_errors": tuple(float(value) for value in taylor_errors),
            "solver_success": solver_success,
            "solver_status": (
                device_result.first_optimizer.status,
                device_result.second_optimizer.status,
            ),
            "solver_iterations": (
                device_result.first_optimizer.nit,
                device_result.second_optimizer.nit,
            ),
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-stage-two-optimization-",
        bounded_steps=50,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
