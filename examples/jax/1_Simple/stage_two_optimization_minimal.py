"""JAX mirror of ``examples/1_Simple/stage_two_optimization_minimal.py``.

Native SIMSOPT constructs the Landreman-Paul QA surface, four base Fourier
coils, currents, and stellarator symmetry ownership. ``BiotSavartJAX`` freezes
that graph once. Quadratic flux, the one-sided total-length penalty, Taylor
test, diagnostics, gradients, and optimization then execute on the selected
JAX device. The accepted DOFs are published back only for reporting.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
    solve_minimal_stage_two,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "native-stage-two-optimization-minimal"
LENGTH_WEIGHT = 1.0
LENGTH_TARGET = 18.0
NATIVE_ITERATIONS = 300
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _build_problem(
    scale: ExecutionScale,
) -> tuple[BiotSavartJAX, SquaredFluxJAX, jax.Array, jax.Array]:
    native_scale = scale == "native_default"
    surface_resolution = 32 if native_scale else 4
    curve_order = 5 if native_scale else 2
    curve_quadrature = 100 if native_scale else 16
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="half period",
        nphi=surface_resolution,
        ntheta=surface_resolution,
    )
    curves = create_equally_spaced_curves(
        4,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=curve_order,
        numquadpoints=curve_quadrature,
    )
    currents = [Current(1.0) * 1.0e5 for _ in curves]
    currents[0].fix_all()
    coils = coils_via_symmetries(
        curves,
        currents,
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


def solve(
    output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    field, flux, surface_gamma, surface_normal = _build_problem(scale)
    initial_device = jax.device_put(np.asarray(field.x, dtype=np.float64))
    taylor_direction = jax.device_put(
        np.random.RandomState(1).uniform(size=initial_device.shape)
    )
    device_result = solve_minimal_stage_two(
        field=field,
        flux_spec=flux.fixed_surface_flux_spec(),
        surface_gamma=surface_gamma,
        surface_normal=surface_normal,
        initial_parameters=initial_device,
        taylor_direction=taylor_direction,
        num_base_curves=4,
        length_weight=LENGTH_WEIGHT,
        length_target=LENGTH_TARGET,
        max_steps=max_steps,
        rtol=1.0e-12,
        atol=1.0e-10,
    )
    initial, final, taylor_errors = jax.device_get(
        (
            device_result.initial,
            device_result.final,
            device_result.taylor_errors,
        )
    )
    solution = np.asarray(final.parameters, dtype=np.float64)
    final_gradient = np.asarray(final.objective_gradient, dtype=np.float64)
    field.x = solution
    field.save(str(output_directory / "biot_savart_opt.json"))
    scientific_success = bool(
        device_result.optimizer.success
        and np.isfinite(final.objective)
        and final.objective < initial.objective
        and np.linalg.norm(final_gradient, ord=np.inf) <= 1.0e-5
        and float(final.total_curve_length) <= 1.1 * LENGTH_TARGET
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_parameters": tuple(float(value) for value in initial.parameters),
            "initial_objective": float(initial.objective),
            "initial_gradient": tuple(
                float(value) for value in initial.objective_gradient
            ),
            "solution": tuple(float(value) for value in solution),
            "final_objective": float(final.objective),
            "final_gradient": tuple(float(value) for value in final_gradient),
            "squared_flux": float(final.squared_flux),
            "curve_length_penalty": float(final.length_penalty),
            "maximum_normal_field": float(final.maximum_normal_field),
            "total_curve_length": float(final.total_curve_length),
            "taylor_errors": tuple(float(value) for value in taylor_errors),
            "solver_success": device_result.optimizer.success,
            "solver_driver": device_result.optimizer.driver.value,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-stage-two-minimal-",
        bounded_steps=80,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
