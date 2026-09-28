"""JAX mirror of ``examples/1_Simple/surf_vol_area.py``.

Host construction owns ``SurfaceRZFourier`` creation and accepted-state
publication.  Immutable surface geometry, free-DOF expansion, both sequential
area/volume target problems, and their derivatives execute in JAX.  The second
stage consumes the first stage's materialized accepted state, matching the
native save/load workflow without hiding a numerical host solve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from simsopt import load
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    run_example,
)
from simsopt_jax.examples.official_tiny_least_squares import (
    DRIVER_SURFACE,
    combine_trf_outcomes,
    guard_finite_endpoint,
    official_default_max_nfev,
    solve_jax_residual,
    surface_area_volume_residual,
    trf_outcome,
    value_and_jacobian,
)

EXAMPLE_ID = "native-surf-vol-area"
FIRST_TARGETS = (8.0, 0.6)
SECOND_TARGETS = (9.0, 0.8)


def _build_surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier()
    surface.fix("rc(0,0)")
    return surface


def solve(
    output_directory: Path, max_steps: int, _scale: ExecutionScale
) -> ExampleResult:
    surface = _build_surface()

    def stage(current: SurfaceRZFourier, targets: tuple[float, float]):
        residual = surface_area_volume_residual(
            np.asarray(current.local_full_x, dtype=np.float64),
            np.asarray(current.quadpoints_phi, dtype=np.float64),
            np.asarray(current.quadpoints_theta, dtype=np.float64),
            np.flatnonzero(current.local_dofs_free_status),
            np.asarray(targets, dtype=np.float64),
            mpol=current.mpol,
            ntor=current.ntor,
            nfp=current.nfp,
            stellsym=current.stellsym,
        )
        initial = np.asarray(current.x, dtype=np.float64)
        initial_residual, _ = value_and_jacobian(residual, initial)
        optimizer = solve_jax_residual(residual, initial, max_nfev=max_steps)
        final_residual, _ = value_and_jacobian(residual, np.asarray(optimizer.x))
        current.x = optimizer.x
        outcome = guard_finite_endpoint(
            trf_outcome(optimizer), (np.asarray(optimizer.x), final_residual)
        )
        return initial_residual, final_residual, optimizer, outcome

    first_initial, first_final, first_optimizer, first_outcome = stage(
        surface, FIRST_TARGETS
    )
    first_solution = np.asarray(surface.x, dtype=np.float64)

    first_state_path = output_directory / "surf_fw.json"
    surface.save(str(first_state_path), indent=2)
    second_surface = load(str(first_state_path))
    second_initial, second_final, second_optimizer, second_outcome = stage(
        second_surface, SECOND_TARGETS
    )
    second_solution = np.asarray(second_surface.x, dtype=np.float64)
    second_surface.save(
        str(output_directory / "surf_centered.json"),
        indent=2,
    )

    # Both official solves must report convergence. The residual norms are
    # published diagnostics, not a gate of our own. Official capture
    # A/reference-simple/runs/native-surf-vol-area/captured-natural-omp1:
    # status 1 `gtol` nfev 9 njev 8, then status 1 `gtol` nfev 5 njev 5.
    outcome = combine_trf_outcomes(first_outcome, second_outcome)
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "first_initial_residuals": tuple(float(value) for value in first_initial),
            "first_solution": tuple(float(value) for value in first_solution),
            "first_final_residuals": tuple(float(value) for value in first_final),
            "second_initial_residuals": tuple(float(value) for value in second_initial),
            "second_solution": tuple(float(value) for value in second_solution),
            "second_final_residuals": tuple(float(value) for value in second_final),
            "first_solver_success": bool(first_optimizer.success),
            "second_solver_success": bool(second_optimizer.success),
            "first_solver_status": int(first_optimizer.status),
            "second_solver_status": int(second_optimizer.status),
            "solver_driver": DRIVER_SURFACE,
            "first_function_evaluations": first_outcome.nfev,
            "second_function_evaluations": second_outcome.nfev,
            "first_jacobian_evaluations": first_outcome.njev,
            "second_jacobian_evaluations": second_outcome.njev,
            "first_residual_norm": float(np.linalg.norm(first_final)),
            "second_residual_norm": float(np.linalg.norm(second_final)),
        },
        status="ok" if outcome.success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    # Neither scale caps the official workflow: the declared budget is SciPy's
    # own default for this problem size, so only --max-steps can bind.
    official_budget = official_default_max_nfev(_build_surface().x.size)
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-surf-vol-area-",
        bounded_steps=official_budget,
        native_default_steps=official_budget,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
