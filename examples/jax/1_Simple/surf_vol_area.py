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
import numpy as np
from simsopt import load
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.examples import (
    ExampleResult,
    run_example,
    solve_rz_surface_area_volume_sequence,
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


def solve(output_directory: Path, max_steps: int) -> ExampleResult:
    surface = _build_surface()
    full_dofs = np.asarray(surface.local_full_x, dtype=np.float64)
    free_positions = np.flatnonzero(surface.local_dofs_free_status)
    device_result = solve_rz_surface_area_volume_sequence(
        full_dofs=jax.device_put(full_dofs),
        quadpoints_phi=jax.device_put(
            np.asarray(surface.quadpoints_phi, dtype=np.float64)
        ),
        quadpoints_theta=jax.device_put(
            np.asarray(surface.quadpoints_theta, dtype=np.float64)
        ),
        free_positions=jax.device_put(free_positions),
        first_targets=jax.device_put(np.asarray(FIRST_TARGETS, dtype=np.float64)),
        second_targets=jax.device_put(np.asarray(SECOND_TARGETS, dtype=np.float64)),
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        max_steps=max_steps,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    host_values = jax.device_get(
        (
            device_result.first.initial_residuals,
            device_result.first.final_parameters,
            device_result.first.final_residuals,
            device_result.second.initial_residuals,
            device_result.second.final_parameters,
            device_result.second.final_residuals,
        )
    )
    (
        first_initial,
        first_solution,
        first_final,
        second_initial,
        second_solution,
        second_final,
    ) = (np.asarray(value, dtype=np.float64) for value in host_values)

    surface.x = first_solution
    first_state_path = output_directory / "surf_fw.json"
    surface.save(str(first_state_path), indent=2)
    second_surface = load(str(first_state_path))
    second_surface.x = second_solution
    second_surface.save(
        str(output_directory / "surf_centered.json"),
        indent=2,
    )

    scientific_success = bool(
        device_result.first.optimizer.success
        and device_result.second.optimizer.success
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
            "first_solver_success": device_result.first.optimizer.success,
            "second_solver_success": device_result.second.optimizer.success,
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
