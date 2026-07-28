"""JAX port of ``examples/2_Intermediate/stage_two_optimization_planar_coils.py``.

The host constructs the Landreman-Paul QA surface and native planar-Fourier
coil graph.  Quadratic flux, equality-target length, clearance, curvature,
mean-squared-curvature, and the native discrete linking-number term execute
through the public pure-JAX Stage-II objective.  Both optimization stages run
on the selected device, and independent endpoint topology checks fail closed
if a solver crosses into a linked component.  Only the accepted state and
final diagnostics return to the host.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_planar_curves
from simsopt_jax.core import (
    CoilSetDofExtractionSpec,
    coil_specs_from_dof_extraction_spec,
    curve_geometry_from_spec,
)
from simsopt_jax.examples import ExampleResult, run_example, scalar_example_driver
from simsopt_jax.objectives import (
    StageTwoObjectiveConfig,
    make_stage_two_objective,
    stage_two_linking_number,
)
from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "native-stage-two-optimization-planar-coils"
NATIVE_ITERATIONS = 400
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _build_problem(
    max_steps: int,
) -> tuple[BiotSavartJAX, SquaredFluxJAX, jax.Array, jax.Array]:
    native_scale = max_steps == NATIVE_ITERATIONS
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
    parameters: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    coil_specs = coil_specs_from_dof_extraction_spec(
        extraction,
        parameters,
    )
    geometry_by_curve: dict[int, tuple[jax.Array, jax.Array, jax.Array]] = {}
    geometry = []
    for spec in coil_specs:
        curve_id = id(spec.curve)
        curve_geometry = geometry_by_curve.get(curve_id)
        if curve_geometry is None:
            curve_geometry = curve_geometry_from_spec(spec.curve)
            geometry_by_curve[curve_id] = curve_geometry
        gamma, gammadash, gammadashdash = curve_geometry
        if spec.symmetry.has_rotation:
            gamma = gamma @ spec.symmetry.rotmat
            gammadash = gammadash @ spec.symmetry.rotmat
            gammadashdash = gammadashdash @ spec.symmetry.rotmat
        geometry.append((gamma, gammadash, gammadashdash))
    gamma = jnp.stack(tuple(terms[0] for terms in geometry))
    gammadash = jnp.stack(tuple(terms[1] for terms in geometry))
    base_gamma = gamma[:4]
    centered = base_gamma - jnp.mean(base_gamma, axis=1, keepdims=True)
    covariance = jnp.einsum("nqi,nqj->nij", centered, centered) / base_gamma.shape[1]
    minimum_variance = jnp.linalg.eigvalsh(covariance)[:, 0]
    lengths = jnp.mean(
        jnp.linalg.norm(gammadash[:4], axis=2),
        axis=1,
    )
    canonical_geometry = jnp.concatenate(
        (
            lengths[:, None],
            jnp.mean(base_gamma, axis=1),
            covariance.reshape((base_gamma.shape[0], 9)),
        ),
        axis=1,
    ).reshape((-1,))
    return (
        jnp.sum(jnp.square(minimum_variance)),
        stage_two_linking_number(gamma, gammadash),
        canonical_geometry,
    )


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    field, flux, surface_gamma, surface_normal = _build_problem(max_steps)
    flux_objective = flux.traceable_objective()
    first_objective = make_stage_two_objective(
        field,
        flux_objective,
        surface_gamma,
        surface_normal,
        _objective_config(10.0),
    )
    second_objective = make_stage_two_objective(
        field,
        flux_objective,
        surface_gamma,
        surface_normal,
        _objective_config(1.0),
    )
    initial_device = jax.device_put(np.asarray(field.x, dtype=np.float64))
    flux_problem = TraceableScalarProblem(objective_fn=flux_objective, x=initial_device)
    driver = scalar_example_driver()
    problem = TraceableScalarProblem(objective_fn=first_objective, x=initial_device)
    initial_objective_device = problem.objective(initial_device)
    (
        initial_planarity_device,
        initial_linking_device,
        initial_canonical_geometry_device,
    ) = _topology_diagnostics(
        field.coil_dof_extraction_spec(),
        initial_device,
    )
    first_result = serial_solve_jax(
        problem,
        driver=driver,
        max_steps=max_steps,
        rtol=1.0e-8,
        atol=1.0e-7,
        line_search_max_steps=40 if driver.value == "simsopt_bfgs" else None,
        require_success=False,
    )
    problem = TraceableScalarProblem(objective_fn=second_objective, x=problem.x)
    second_result = serial_solve_jax(
        problem,
        driver=driver,
        max_steps=max_steps,
        rtol=1.0e-8,
        atol=1.0e-7,
        line_search_max_steps=40 if driver.value == "simsopt_bfgs" else None,
        require_success=False,
    )
    solution_device = jax.block_until_ready(problem.x)
    (
        final_planarity_device,
        final_linking_device,
        canonical_geometry_device,
    ) = _topology_diagnostics(
        field.coil_dof_extraction_spec(),
        solution_device,
    )
    diagnostics_device = jnp.concatenate(
        (
            jnp.stack(
                (
                    initial_objective_device,
                    problem.objective(solution_device),
                    flux_problem.objective(solution_device),
                    initial_planarity_device,
                    final_planarity_device,
                    initial_linking_device,
                    final_linking_device,
                )
            ),
            initial_canonical_geometry_device,
            canonical_geometry_device,
        )
    )
    initial = np.asarray(jax.device_get(initial_device), dtype=np.float64)
    solution = np.asarray(jax.device_get(solution_device), dtype=np.float64)
    diagnostics = np.asarray(
        jax.device_get(jax.block_until_ready(diagnostics_device)),
        dtype=np.float64,
    )
    scalar_diagnostics = diagnostics[:7]
    canonical_size = int(canonical_geometry_device.shape[0])
    initial_canonical_geometry = diagnostics[7 : 7 + canonical_size]
    canonical_geometry = diagnostics[7 + canonical_size :]
    (
        initial_objective,
        final_objective,
        squared_flux,
        initial_planarity_penalty,
        planarity_penalty,
        initial_linking_number,
        linking_number,
    ) = (float(value) for value in scalar_diagnostics)
    solver_success = bool(first_result.success and second_result.success)
    solver_accepted = bool(
        first_result.status in (0, 1) and second_result.status in (0, 1)
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
            "solver_status": (first_result.status, second_result.status),
            "solver_iterations": (first_result.nit, second_result.nit),
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
