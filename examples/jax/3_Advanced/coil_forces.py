"""JAX port of ``examples/3_Advanced/coil_forces.py``.

The host builds the native Landreman-Paul surface, regularized coil graph, and
symmetry-expanded currents.  Quadratic flux, engineering penalties, Lorentz
force, vacuum energy, gradients, and both optimization stages then execute on
the selected JAX CPU or GPU.  The stage length weight is an explicit device
parameter, so one compiled graph serves both stages.  Fast mode uses
bounded-memory SIMSOPT L-BFGS; parity mode uses SIMSOPT BFGS.  Only accepted
endpoint diagnostics return to the host.

Splitting the length penalty out of the stage objective is bitwise invariant
against the pre-change formulation (and so is the aligned parity mirror) only
while the total base-curve length stays under the 17.4 m target and the penalty
is therefore exactly zero; once it is active the two spellings reassociate the
same sum and their values differ at the ~1 ULP level.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
    scalar_example_driver,
)
from simsopt_jax.objectives import StageTwoObjectiveConfig
from simsopt_jax.solve.contracts import OptimizerResult
from simsopt_jax.solve.driver import Driver
from simsopt_jax.solve.serial import (
    TraceableArrayFunction,
    TraceableParametricScalarProblem,
    serial_solve_jax,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives import (
    ForceStageTwoConfig,
    force_stage_two_diagnostics,
    make_force_stage_two_length_penalty,
    make_force_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "native-coil-forces"
NATIVE_ITERATIONS = 400
FIRST_LENGTH_WEIGHT = 1.0e-3
SECOND_LENGTH_WEIGHT = 1.0e-4
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _stage_config() -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(
        num_base_curves=3,
        length_target=17.4,
        length_target_mode="max",
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=1000.0,
        curve_surface_minimum_distance=0.3,
        curve_surface_weight=10.0,
        curvature_threshold=5.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=5.0,
        mean_squared_curvature_weight=1.0e-6,
    )


def _force_config() -> ForceStageTwoConfig:
    return ForceStageTwoConfig(
        num_force_coils=3,
        force_weight=1.0e-2,
        vacuum_energy_weight=1.0e-4,
        force_power=4.0,
        force_threshold=0.0,
        downsample=1,
    )


def _build_problem(
    scale: ExecutionScale,
) -> tuple[
    BiotSavartJAX,
    SquaredFluxJAX,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    native_scale = scale == "native_default"
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="half period",
        nphi=32 if native_scale else 4,
        ntheta=32 if native_scale else 4,
    )
    base_curves = create_equally_spaced_curves(
        3,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.5,
        order=5 if native_scale else 2,
        numquadpoints=75 if native_scale else 8,
        use_jax_curve=False,
    )
    base_currents = [Current(1.0e5) for _ in base_curves]
    base_currents[0].fix_all()
    regularization = 0.05**2 / np.sqrt(np.e)
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        surface.stellsym,
        [regularization for _ in base_curves],
    )
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3))
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3))
    )
    target_quadpoints = jax.device_put(
        np.stack(
            tuple(
                np.asarray(curve.quadpoints, dtype=np.float64) for curve in base_curves
            )
        )
    )
    regularizations = jax.device_put(
        np.full(len(coils), regularization, dtype=np.float64)
    )
    return (
        field,
        flux,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
    )


def _run_stage(
    problem: TraceableParametricScalarProblem,
    *,
    driver: Driver,
    max_steps: int,
) -> OptimizerResult:
    if driver == Driver.SIMSOPT_LBFGSB:
        return serial_solve_jax(
            problem,
            driver=driver,
            max_steps=max_steps,
            maxcor=min(max_steps, 300),
            rtol=1.0e-15,
            atol=1.0e-8,
            require_success=False,
        )
    return serial_solve_jax(
        problem,
        driver=driver,
        max_steps=max_steps,
        line_search_max_steps=40,
        rtol=1.0e-15,
        atol=1.0e-8,
        require_success=False,
    )


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    (
        field,
        flux,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
    ) = _build_problem(scale)
    flux_objective = flux.traceable_objective()
    force_config = _force_config()
    stage_config = _stage_config()
    objective = make_force_stage_two_objective(
        field,
        flux_objective,
        surface_gamma,
        surface_normal,
        target_quadpoints,
        regularizations,
        stage_config,
        force_config,
    )
    length_penalty = make_force_stage_two_length_penalty(field, stage_config)

    def weighted_objective(
        parameters: jax.Array,
        length_weight: jax.Array,
    ) -> jax.Array:
        return objective(parameters) + length_penalty(parameters, length_weight)

    initial_device = jax.device_put(np.asarray(field.x, dtype=np.float64))
    problem = TraceableParametricScalarProblem(
        objective_fn=weighted_objective,
        objective_parameter=jax.device_put(
            np.asarray(FIRST_LENGTH_WEIGHT, dtype=np.float64)
        ),
        x=initial_device,
    )
    initial_objective_device = problem.objective(initial_device)
    driver = scalar_example_driver()
    first_result = _run_stage(problem, driver=driver, max_steps=max_steps)
    problem.set_objective_parameter(
        jax.device_put(np.asarray(SECOND_LENGTH_WEIGHT, dtype=np.float64))
    )
    second_result = _run_stage(problem, driver=driver, max_steps=max_steps)
    solution_device = jax.block_until_ready(problem.x)
    final_objective_device, gradient_device = problem.value_and_grad(solution_device)
    force_diagnostics = force_stage_two_diagnostics(
        field,
        target_quadpoints,
        regularizations,
        force_config,
    )
    diagnostics = TraceableArrayFunction(
        function_fn=lambda parameters: jnp.concatenate(
            (
                force_diagnostics(parameters),
                flux_objective(parameters)[None],
            )
        ),
        x=solution_device,
    )
    force_values_device = jax.block_until_ready(diagnostics(solution_device))
    solution = np.asarray(jax.device_get(solution_device), dtype=np.float64)
    gradient = np.asarray(jax.device_get(gradient_device), dtype=np.float64)
    scalar_values = np.asarray(
        jax.device_get(
            jnp.concatenate(
                (
                    jnp.stack((initial_objective_device, final_objective_device)),
                    force_values_device,
                )
            )
        ),
        dtype=np.float64,
    )
    (
        initial_objective,
        final_objective,
        force_objective,
        maximum_force,
        vacuum_energy,
        squared_flux,
    ) = (float(value) for value in scalar_values)
    solver_accepted = bool(
        first_result.status in (0, 1) and second_result.status in (0, 1)
    )
    scientific_success = bool(
        solver_accepted
        and np.all(np.isfinite(solution))
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(scalar_values))
        and final_objective < initial_objective
        and force_objective >= 0.0
        and maximum_force >= 0.0
        and vacuum_energy >= 0.0
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_objective": initial_objective,
            "solution": tuple(float(value) for value in solution),
            "final_objective": final_objective,
            "squared_flux": squared_flux,
            "force_objective": force_objective,
            "maximum_force": maximum_force,
            "vacuum_energy": vacuum_energy,
            "gradient": tuple(float(value) for value in gradient),
            "solver_success": bool(first_result.success and second_result.success),
            "solver_status": (first_result.status, second_result.status),
            "solver_iterations": (first_result.nit, second_result.nit),
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-coil-forces-",
        bounded_steps=3,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
