"""JAX port of ``examples/2_Intermediate/stage_two_optimization_planar_coils.py``.

The host constructs the Landreman-Paul QA surface and native planar-Fourier
coil graph.  Quadratic flux, equality-target length, clearance, curvature,
mean-squared-curvature, and the native discrete linking-number term execute
through one reusable parametric pure-JAX Stage-II objective.  Both optimization
stages run on the selected device, and independent endpoint topology checks
fail closed if a solver crosses into a linked component.  Initial and final
state trees return through one batched host publication.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_planar_curves
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.core import CoilSetDofExtractionSpec
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
    solve_standard_stage_two,
)
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    stage_two_planar_topology_values,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "native-stage-two-optimization-planar-coils"
NATIVE_ITERATIONS = 400
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _build_problem(
    scale: ExecutionScale,
    device: jax.Device | None,
) -> tuple[BiotSavartJAX, SquaredFluxJAX, jax.Array, jax.Array]:
    native_scale = scale == "native_default"
    surface_resolution = 32 if native_scale else 4
    curve_order = 5 if native_scale else 2
    curve_quadrature = 100 if native_scale else 32
    surface = SurfaceRZFourier.from_vmec_input(
        TEST_DATA / "input.LandremanPaul2021_QA",
        range="half period",
        nphi=surface_resolution,
        ntheta=surface_resolution,
    )
    base_curves = create_equally_spaced_planar_curves(
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
    field = BiotSavartJAX(
        coils_via_symmetries(base_curves, base_currents, surface.nfp, True)
    )
    flux = SquaredFluxJAX(surface, field)
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    return field, flux, surface_gamma, surface_normal


def _regularization_config() -> StageTwoObjectiveConfig:
    return StageTwoObjectiveConfig(
        num_base_curves=4,
        length_target=10.4,
        length_target_mode="identity",
        curve_curve_minimum_distance=0.08,
        curve_curve_weight=1000.0,
        curve_surface_minimum_distance=0.12,
        curve_surface_weight=10.0,
        curvature_threshold=10.0,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=10.0,
        mean_squared_curvature_weight=1.0e-6,
        linking_number_weight=1.0,
    )


@jax.jit
def _topology_diagnostics(
    extraction: CoilSetDofExtractionSpec,
    parameter_states: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    return jax.vmap(
        lambda parameters: stage_two_planar_topology_values(
            extraction,
            parameters,
            4,
        )
    )(parameter_states)


def _topology_state(
    topology_states: tuple[np.ndarray, np.ndarray, np.ndarray],
    index: int,
) -> tuple[float, float, np.ndarray]:
    return (
        float(topology_states[0][index]),
        float(topology_states[1][index]),
        np.asarray(topology_states[2][index], dtype=np.float64),
    )


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    device = get_runtime_jax_device()
    field, flux, surface_gamma, surface_normal = _build_problem(scale, device)
    initial_device = jax.device_put(
        np.asarray(field.x, dtype=np.float64),
        device,
    )
    taylor_direction_device = jax.device_put(
        np.random.RandomState(1).uniform(size=initial_device.shape),
        device,
    )
    first_length_weight_device = jax.device_put(
        np.asarray(10.0, dtype=np.float64),
        device,
    )
    second_length_weight_device = jax.device_put(
        np.asarray(1.0, dtype=np.float64),
        device,
    )
    device_result = solve_standard_stage_two(
        field=field,
        flux_spec=flux.fixed_surface_flux_spec(),
        surface_gamma=surface_gamma,
        surface_normal=surface_normal,
        initial_parameters=initial_device,
        taylor_direction=taylor_direction_device,
        regularization_config=_regularization_config(),
        first_length_weight=first_length_weight_device,
        second_length_weight=second_length_weight_device,
        max_steps=max_steps,
        rtol=1.0e-8,
        atol=1.0e-7,
    )
    topology_device = _topology_diagnostics(
        field.coil_dof_extraction_spec(),
        jnp.stack(
            (
                device_result.initial.parameters,
                device_result.final.parameters,
            )
        ),
    )
    initial_state, final_state, topology_states = jax.device_get(
        jax.block_until_ready(
            (
                device_result.initial,
                device_result.final,
                topology_device,
            )
        )
    )
    initial = np.asarray(initial_state.parameters, dtype=np.float64)
    solution = np.asarray(final_state.parameters, dtype=np.float64)
    initial_objective = float(initial_state.objective)
    final_objective = float(final_state.objective)
    squared_flux = float(final_state.squared_flux)
    (
        initial_planarity_penalty,
        initial_linking_number,
        initial_canonical_geometry,
    ) = _topology_state(topology_states, 0)
    planarity_penalty, linking_number, canonical_geometry = _topology_state(
        topology_states,
        1,
    )
    solver_success = bool(
        device_result.first_optimizer.success and device_result.second_optimizer.success
    )
    solver_accepted = bool(
        device_result.first_optimizer.status in (0, 1)
        and device_result.second_optimizer.status in (0, 1)
    )
    scientific_success = bool(
        solver_accepted
        and np.all(np.isfinite(solution))
        and final_objective < initial_objective
        and initial_planarity_penalty <= 1.0e-24
        and planarity_penalty <= 1.0e-24
        and initial_linking_number == 0.0
        and linking_number == 0.0
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_parameters": tuple(float(value) for value in initial),
            "initial_objective": initial_objective,
            "solution": tuple(float(value) for value in solution),
            "initial_canonical_geometry": tuple(
                float(value) for value in initial_canonical_geometry
            ),
            "canonical_geometry": tuple(float(value) for value in canonical_geometry),
            "final_objective": final_objective,
            "planarity_penalty": planarity_penalty,
            "linking_number": linking_number,
            "squared_flux": squared_flux,
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
        temporary_prefix="simsopt-jax-stage-two-planar-",
        bounded_steps=100,
        native_default_steps=NATIVE_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
