"""JAX port of ``examples/3_Advanced/stage_two_optimization_finitebuild.py``.

The host constructs the native Landreman-Paul surface and immutable
multifilament coil topology.  Pack-frame geometry, all filament copies,
Biot-Savart field evaluation, quadratic flux, engineering penalties,
gradients, and optimization then execute on the selected JAX CPU or GPU.
Each shared pack frame is evaluated once per objective call.  The objective
scale is an explicit device parameter, so one compiled graph serves both the
scaled solve and the published unscaled gradient.  Fast mode uses
bounded-memory SIMSOPT L-BFGS; parity mode uses SIMSOPT BFGS.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import (
    Coil,
    Current,
    apply_symmetries_to_currents,
    apply_symmetries_to_curves,
)
from simsopt.geo import (
    CurveLength,
    SurfaceRZFourier,
    create_equally_spaced_curves,
    create_multifilament_grid,
)
from simsopt_jax.core import compute_filament_offsets
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
    scalar_example_driver,
)
from simsopt_jax.examples.stage_two_finitebuild import (
    prepare_finite_build_stage_two,
    solve_finite_build_stage_two,
)
from simsopt_jax.solve.driver import Driver
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives import (
    FiniteBuildStageTwoConfig,
    finite_build_stage_two_diagnostics,
    make_finite_build_stage_two_objective,
)
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "native-stage-two-optimization-finitebuild"
NATIVE_ITERATIONS = 400
SOLVE_OBJECTIVE_SCALE = 1.0e-4
PUBLISHED_OBJECTIVE_SCALE = 1.0
NUM_BASE_CURVES = 4
NUM_FILAMENTS_N = 2
NUM_FILAMENTS_B = 3
GAP_SIZE_N = 0.02
GAP_SIZE_B = 0.04
ROTATION_ORDER = 1
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _build_problem(
    scale: ExecutionScale,
) -> tuple[
    BiotSavartJAX,
    SquaredFluxJAX,
    FiniteBuildStageTwoConfig,
]:
    native_scale = scale == "native_default"
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="half period",
        nphi=32 if native_scale else 4,
        ntheta=32 if native_scale else 4,
    )
    base_curves = create_equally_spaced_curves(
        NUM_BASE_CURVES,
        surface.nfp,
        stellsym=True,
        R0=1.0,
        R1=0.7,
        order=5 if native_scale else 2,
        numquadpoints=75 if native_scale else 8,
        use_jax_curve=False,
    )
    filament_count = NUM_FILAMENTS_N * NUM_FILAMENTS_B
    base_currents = []
    for index in range(NUM_BASE_CURVES):
        current = Current(1.0)
        if index == 0:
            current.fix_all()
        base_currents.append(current * (1.0e5 / filament_count))
    base_filaments = list(
        itertools.chain.from_iterable(
            create_multifilament_grid(
                curve,
                NUM_FILAMENTS_N,
                NUM_FILAMENTS_B,
                GAP_SIZE_N,
                GAP_SIZE_B,
                rotation_order=ROTATION_ORDER,
            )
            for curve in base_curves
        )
    )
    filament_currents = list(
        itertools.chain.from_iterable(
            [current] * filament_count for current in base_currents
        )
    )
    filament_curves = apply_symmetries_to_curves(
        base_filaments,
        surface.nfp,
        True,
    )
    currents = apply_symmetries_to_currents(
        filament_currents,
        surface.nfp,
        True,
    )
    coils = [
        Coil(curve, current)
        for curve, current in zip(filament_curves, currents, strict=True)
    ]
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    initial_lengths = tuple(float(CurveLength(curve).J()) for curve in base_curves)
    config = FiniteBuildStageTwoConfig(
        num_base_curves=NUM_BASE_CURVES,
        filament_offsets=compute_filament_offsets(
            numfilaments_n=NUM_FILAMENTS_N,
            numfilaments_b=NUM_FILAMENTS_B,
            gapsize_n=GAP_SIZE_N,
            gapsize_b=GAP_SIZE_B,
        ),
        symmetry_copies=surface.nfp * 2,
        length_targets=initial_lengths,
        length_weight=1.0e-2,
        curve_curve_minimum_distance=0.1,
        curve_curve_weight=10.0,
    )
    return field, flux, config


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    field, flux, config = _build_problem(scale)
    objective = make_finite_build_stage_two_objective(
        field,
        flux.fixed_surface_flux_spec(),
        config,
    )

    initial_device = jax.device_put(np.asarray(field.x, dtype=np.float64))
    prepared = prepare_finite_build_stage_two(
        objective_fn=objective,
        diagnostics_fn=finite_build_stage_two_diagnostics(
            field,
            flux.fixed_surface_flux_spec(),
            config,
        ),
        initial_parameters=initial_device,
        objective_scale=jax.device_put(
            np.asarray(SOLVE_OBJECTIVE_SCALE, dtype=np.float64)
        ),
    )
    initial_values_device = prepared.diagnostics(initial_device)
    driver = scalar_example_driver()
    result = solve_finite_build_stage_two(
        prepared,
        driver=driver,
        max_steps=max_steps,
        rtol=1.0e-15,
        atol=1.0e-12,
        line_search_max_steps=None if driver == Driver.SIMSOPT_LBFGSB else 40,
    )
    solution_device = jax.block_until_ready(prepared.problem.x)
    prepared.problem.set_objective_parameter(
        jax.device_put(np.asarray(PUBLISHED_OBJECTIVE_SCALE, dtype=np.float64))
    )
    _final_objective_device, gradient_device = prepared.problem.value_and_grad(
        solution_device
    )
    final_values_device = jax.block_until_ready(prepared.diagnostics(solution_device))
    solution = np.asarray(jax.device_get(solution_device), dtype=np.float64)
    gradient = np.asarray(jax.device_get(gradient_device), dtype=np.float64)
    values = np.asarray(
        jax.device_get(jnp.concatenate((initial_values_device, final_values_device))),
        dtype=np.float64,
    )
    metric_count = 4 + NUM_BASE_CURVES
    initial_values = values[:metric_count]
    final_values = values[metric_count:]
    initial_objective = float(np.sum(initial_values[:3]))
    final_objective = float(np.sum(final_values[:3]))
    squared_flux, length_penalty, distance_penalty, minimum_clearance = (
        float(value) for value in final_values[:4]
    )
    coil_lengths = tuple(float(value) for value in final_values[4:])
    solver_accepted = result.status in (0, 1)
    scientific_success = bool(
        solver_accepted
        and np.all(np.isfinite(solution))
        and np.all(np.isfinite(gradient))
        and np.all(np.isfinite(values))
        and np.isfinite(final_objective)
        and final_objective < initial_objective
        and squared_flux >= 0.0
        and length_penalty >= 0.0
        and distance_penalty >= 0.0
        and minimum_clearance > 0.0
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_objective": initial_objective,
            "solution": tuple(float(value) for value in solution),
            "final_objective": final_objective,
            "squared_flux": squared_flux,
            "length_penalty": length_penalty,
            "distance_penalty": distance_penalty,
            "minimum_clearance": minimum_clearance,
            "coil_lengths": coil_lengths,
            "gradient": tuple(float(value) for value in gradient),
            "solver_success": bool(result.success),
            "solver_status": result.status,
            "solver_iterations": result.nit,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-stage-two-finitebuild-",
        bounded_steps=3,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
