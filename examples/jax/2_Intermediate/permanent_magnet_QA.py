"""JAX port of ``examples/2_Intermediate/permanent_magnet_QA.py``.

The host constructs the native Landreman-Paul QA surfaces and initial TF-coil
graph.  Coil-current optimization runs with the public pure-JAX Stage-II
objective on the selected device.  The accepted coil state is published once
to construct the native cylindrical permanent-magnet grid, which is then
snapshotted as an immutable ``PermanentMagnetGridJAX`` payload.  Both
relax-and-split continuation stages execute on the selected JAX device.
Plotting, VTK/FAMUS output, and the native example's disabled VMEC post-check
are reporting concerns outside this numerical parity claim.
"""

from __future__ import annotations

import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.geo import PermanentMagnetGrid, SurfaceRZFourier
from simsopt.util.permanent_magnet_helper_functions import (
    initialize_coils_for_pm_optimization,
)
from simsopt_jax.core import grouped_biot_savart_B_from_spec
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
    scalar_example_driver,
)
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.objectives import StageTwoObjectiveConfig, make_stage_two_objective
from simsopt_jax.solve.permanent_magnet import relax_and_split_jax
from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "native-permanent-magnet-qa"
NATIVE_COIL_ITERATIONS = 500
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coil_objective_config(surface: SurfaceRZFourier) -> StageTwoObjectiveConfig:
    major_radius = surface.get_rc(0, 0)
    return StageTwoObjectiveConfig(
        num_base_curves=8,
        length_weight=1.0,
        length_target=18.0 * major_radius,
        curve_curve_minimum_distance=0.1 * major_radius,
        curve_curve_weight=1.0,
        curve_surface_minimum_distance=0.15 * major_radius,
        curve_surface_weight=1.0e-2,
        curvature_threshold=0.1 * major_radius,
        curvature_weight=1.0e-6,
        mean_squared_curvature_threshold=0.1 * major_radius,
        mean_squared_curvature_weight=1.0e-6,
    )


def _build_grid(
    output_directory: Path,
    max_steps: int,
    scale: ExecutionScale,
) -> tuple[PermanentMagnetGridJAX, dict[str, str]]:
    native_scale = scale == "native_default"
    resolution = 16 if native_scale else 4
    radial_extent = 0.02 if native_scale else 0.05
    surface_path = TEST_DATA / "input.LandremanPaul2021_QA_lowres"
    surface = SurfaceRZFourier.from_vmec_input(
        surface_path,
        range="half period",
        nphi=resolution,
        ntheta=resolution,
    )
    inner_surface = SurfaceRZFourier.from_vmec_input(
        surface_path,
        range="half period",
        nphi=resolution,
        ntheta=resolution,
    )
    outer_surface = SurfaceRZFourier.from_vmec_input(
        surface_path,
        range="half period",
        nphi=resolution,
        ntheta=resolution,
    )
    inner_surface.extend_via_projected_normal(0.05)
    outer_surface.extend_via_projected_normal(0.15)

    with redirect_stdout(io.StringIO()):
        _, _, coils = initialize_coils_for_pm_optimization(
            "qa",
            TEST_DATA,
            surface,
            output_directory,
        )
    field = BiotSavartJAX(coils)
    flux = SquaredFluxJAX(surface, field)
    surface_gamma = jax.device_put(
        np.asarray(surface.gamma(), dtype=np.float64).reshape((-1, 3))
    )
    surface_normal = jax.device_put(
        np.asarray(surface.normal(), dtype=np.float64).reshape((-1, 3))
    )
    objective = make_stage_two_objective(
        field,
        flux.traceable_objective(),
        surface_gamma,
        surface_normal,
        _coil_objective_config(surface),
    )
    initial = jax.device_put(np.asarray(field.x, dtype=np.float64))
    problem = TraceableScalarProblem(objective_fn=objective, x=initial)
    serial_solve_jax(
        problem,
        driver=scalar_example_driver(),
        max_steps=max_steps,
        rtol=1.0e-8,
        atol=1.0e-7,
        require_success=False,
    )

    accepted_coils = field.coil_set_spec_from_dofs(problem.x)
    normal_field_device = jnp.sum(
        jnp.reshape(
            grouped_biot_savart_B_from_spec(surface_gamma, accepted_coils),
            (resolution, resolution, 3),
        )
        * jax.device_put(np.asarray(surface.unitnormal(), dtype=np.float64)),
        axis=2,
    )
    normal_field = np.asarray(
        jax.device_get(jax.block_until_ready(normal_field_device)),
        dtype=np.float64,
    )
    with redirect_stdout(io.StringIO()):
        cpu_grid = PermanentMagnetGrid.geo_setup_between_toroidal_surfaces(
            surface,
            normal_field,
            inner_surface,
            outer_surface,
            dr=radial_extent,
            coordinate_flag="cylindrical",
        )
    grid = PermanentMagnetGridJAX.from_cpu(cpu_grid)
    return grid, {surface_path.name: _sha256(surface_path)}


def solve(
    output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    grid, input_sha256 = _build_grid(output_directory, max_steps, scale)
    native_scale = scale == "native_default"
    inner_steps = 10 if native_scale else 3
    outer_steps = 10 if native_scale else 2
    nu = 1.0e10
    base_reg_l0 = 0.05 / (2.0 * nu)
    initial_error_device = jnp.linalg.norm(grid.b_obj)
    first = relax_and_split_jax(
        grid,
        max_iter=inner_steps,
        max_iter_RS=outer_steps,
        nu=nu,
        reg_l0=0.5 * base_reg_l0,
    )
    second = relax_and_split_jax(
        grid,
        m0=first.m,
        max_iter=inner_steps,
        max_iter_RS=outer_steps,
        nu=nu,
        reg_l0=base_reg_l0,
    )
    proxy = second.m_proxy
    final_error_device = jnp.linalg.norm(grid.A_obj @ jnp.ravel(proxy) - grid.b_obj)
    moments = np.asarray(jax.device_get(jax.block_until_ready(proxy)), dtype=np.float64)
    errors = np.asarray(
        jax.device_get(
            jax.block_until_ready(jnp.stack((initial_error_device, final_error_device)))
        ),
        dtype=np.float64,
    )
    selected = np.flatnonzero(np.linalg.norm(moments, axis=1) > 0.0)
    selected_moments = moments[selected]
    initial_error, final_error = (float(value) for value in errors)
    solver_success = bool(
        np.all(np.isfinite(selected_moments))
        and selected.size > 0
        and final_error < initial_error
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_normal_error": initial_error,
            "final_normal_error": final_error,
            "selected_dipoles": tuple(int(index) for index in selected),
            "moments": tuple(
                tuple(float(component) for component in moment)
                for moment in selected_moments
            ),
            "input_sha256": input_sha256,
            "solver_success": solver_success,
        },
        status="ok" if solver_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-permanent-magnet-qa-",
        bounded_steps=100,
        native_default_steps=NATIVE_COIL_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
