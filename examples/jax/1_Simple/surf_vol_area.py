"""JAX mirror of ``examples/1_Simple/surf_vol_area.py``.

Host construction owns ``SurfaceRZFourier`` creation and accepted-state
publication.  Immutable surface geometry, free-DOF expansion, both sequential
area/volume target problems, and their derivatives execute in JAX.  The second
stage consumes the first stage's materialized accepted state, matching the
native save/load workflow without hiding a numerical host solve.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_spec_from_dofs
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax.solve.serial import (
    TraceableLeastSquaresProblem,
    least_squares_serial_solve_jax,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    surface_area_jax_from_dofs,
    surface_volume_jax_from_dofs,
)

EXAMPLE_ID = "native-surf-vol-area"
FIRST_TARGETS = (8.0, 0.6)
SECOND_TARGETS = (9.0, 0.8)


def _build_surface() -> SurfaceRZFourier:
    quadrature = np.linspace(0.0, 1.0, 32, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=0,
        nfp=1,
        stellsym=True,
        quadpoints_phi=quadrature,
        quadpoints_theta=quadrature,
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.1)
    surface.set_zs(1, 0, 0.1)
    surface.fix_all()
    surface.unfix("rc(1,0)")
    surface.unfix("zs(1,0)")
    return surface


def _solve_stage(
    surface: SurfaceRZFourier,
    targets: tuple[float, float],
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    full_dofs = np.asarray(surface.local_full_x, dtype=np.float64)
    free_positions = np.flatnonzero(surface.local_dofs_free_status)
    fixed_dofs = full_dofs.copy()
    fixed_dofs[free_positions] = 0.0
    expansion = np.zeros((full_dofs.size, free_positions.size), dtype=np.float64)
    expansion[free_positions, np.arange(free_positions.size)] = 1.0
    spec = surface_rz_fourier_spec_from_dofs(
        jax.device_put(full_dofs),
        quadpoints_phi=jax.device_put(
            np.asarray(surface.quadpoints_phi, dtype=np.float64)
        ),
        quadpoints_theta=jax.device_put(
            np.asarray(surface.quadpoints_theta, dtype=np.float64)
        ),
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )
    fixed_device = jax.device_put(fixed_dofs)
    expansion_device = jax.device_put(expansion)
    targets_device = jax.device_put(np.asarray(targets, dtype=np.float64))

    @jax.jit
    def residuals(parameters: jax.Array) -> jax.Array:
        current_dofs = fixed_device + expansion_device @ parameters
        return (
            jnp.stack(
                (
                    surface_area_jax_from_dofs(spec, current_dofs),
                    surface_volume_jax_from_dofs(spec, current_dofs),
                )
            )
            - targets_device
        )

    initial_device = jax.device_put(np.asarray(surface.x, dtype=np.float64))
    initial_residuals = np.asarray(
        jax.device_get(residuals(initial_device)), dtype=np.float64
    )
    problem = TraceableLeastSquaresProblem(residual_fn=residuals, x=initial_device)
    solver_result = least_squares_serial_solve_jax(
        problem,
        max_steps=max_steps,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    solution = np.asarray(jax.device_get(problem.x), dtype=np.float64)
    final_residuals = np.asarray(jax.device_get(residuals(problem.x)), dtype=np.float64)
    surface.x = solution
    return initial_residuals, solution, final_residuals, solver_result.success


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    surface = _build_surface()
    first_initial, first_solution, first_final, first_success = _solve_stage(
        surface, FIRST_TARGETS, max_steps
    )
    accepted_full_state = np.asarray(surface.local_full_x, dtype=np.float64).copy()
    second_surface = _build_surface()
    second_surface.local_full_x = accepted_full_state
    second_initial, second_solution, second_final, second_success = _solve_stage(
        second_surface, SECOND_TARGETS, max_steps
    )
    scientific_success = bool(
        first_success
        and second_success
        and np.linalg.norm(first_final) <= 1.0e-8
        and np.linalg.norm(second_final) <= 1.0e-8
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "first_initial_residuals": tuple(float(value) for value in first_initial),
            "first_solution": tuple(float(value) for value in first_solution),
            "first_final_residuals": tuple(float(value) for value in first_final),
            "second_initial_residuals": tuple(float(value) for value in second_initial),
            "second_solution": tuple(float(value) for value in second_solution),
            "second_final_residuals": tuple(float(value) for value in second_final),
            "first_solver_success": first_success,
            "second_solver_success": second_success,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-surf-vol-area-",
        bounded_steps=32,
        native_default_steps=128,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
