"""JAX port of ``examples/2_Intermediate/boozer.py``.

The host constructs the canonical NCSX coils and tensor-Fourier surface.  A
``BoozerSurfaceJAX`` first reduces the area-constrained Boozer residual, then
polishes it with the least-squares solver, and finally relabels the surface by
toroidal flux and expands it to three times the converged flux.  The numerical
solves execute on the selected JAX device.

Every official setting and every stage comes from
``simsopt_jax.examples.boozer_official``, so this script cannot drift off
upstream's settings.

GPU memory: ``jit(solve)`` at mpol=ntor=16 on a 48² quadrature requested
243.07 GiB and OOM'd on a 32 GB card.  The largest resolution known to fit
on that card is this example's ``native_default`` scale, mpol=ntor=5 on an
11² grid; smoke (mpol=ntor=2, 5²) also fits.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
from simsopt.configs import get_data
from simsopt.geo import (
    Area,
    SurfaceXYZTensorFourier,
    ToroidalFlux,
    Volume,
    boozer_surface_residual,
)
from simsopt.geo.curve import Curve
from simsopt_jax.examples import ExampleResult, ExecutionScale, run_example
from simsopt_jax.examples.boozer_official import (
    OFFICIAL_CONSTRAINT_WEIGHT,
    OFFICIAL_FLUX_MULTIPLIER,
    OFFICIAL_INITIAL_IOTA,
    OFFICIAL_LS_MAXITER,
    OFFICIAL_SOLVER_TOLERANCE,
    OFFICIAL_SURFACE_DISTANCE,
    OFFICIAL_SURFACE_RESOLUTION,
    BoozerStageState,
    boozer_first_stage_budget,
    boozer_official_options,
    run_boozer_lbfgs_stage,
    run_boozer_manual_stage,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

EXAMPLE_ID = "native-boozer"


def _residual_norm(
    surface: SurfaceXYZTensorFourier,
    iota: float,
    G: float,
    native_field: object,
) -> float:
    residual = boozer_surface_residual(
        surface,
        iota,
        G,
        native_field,
        derivatives=0,
    )
    return float(np.linalg.norm(np.asarray(residual, dtype=np.float64)))


def solve(
    _output_directory: Path, max_steps: int, scale: ExecutionScale
) -> ExampleResult:
    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    del base_curves
    magnetic_axis = cast(Curve, magnetic_axis)
    field = BiotSavartJAX(native_field.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))

    native_scale = scale == "native_default"
    mpol = OFFICIAL_SURFACE_RESOLUTION if native_scale else 2
    ntor = OFFICIAL_SURFACE_RESOLUTION if native_scale else 2
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False),
    )
    surface.fit_to_curve(magnetic_axis, OFFICIAL_SURFACE_DISTANCE, flip_theta=True)
    start = BoozerStageState(
        surface_dofs=np.asarray(surface.get_dofs(), dtype=np.float64),
        iota=OFFICIAL_INITIAL_IOTA,
        G=G0,
    )
    initial_residual = _residual_norm(surface, start.iota, start.G, native_field)

    # The first stage's budget is read from its owner, never rebuilt here:
    # upstream's own ``OFFICIAL_LBFGS_MAXITER`` at ``native_default`` (where
    # stages two and three already run ``OFFICIAL_LS_MAXITER`` whatever
    # ``max_steps`` says), and the official ratio of the two official budgets
    # applied to ``max_steps`` at a reduced scale.
    rough_maxiter = boozer_first_stage_budget(
        least_squares_steps=max_steps,
        native_default=native_scale,
    )
    options = boozer_official_options(
        rough_maxiter=rough_maxiter,
        ls_maxiter=OFFICIAL_LS_MAXITER,
        tolerance=OFFICIAL_SOLVER_TOLERANCE,
    )
    area = Area(surface)
    area_solver = BoozerSurfaceJAX(
        field,
        surface,
        area,
        float(area.J()),
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
        options=options,
    )
    rough = run_boozer_lbfgs_stage(
        area_solver,
        start,
        tol=OFFICIAL_SOLVER_TOLERANCE,
        maxiter=rough_maxiter,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
    )
    polished = run_boozer_manual_stage(
        area_solver,
        rough.state,
        tol=OFFICIAL_SOLVER_TOLERANCE,
        maxiter=OFFICIAL_LS_MAXITER,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
    )

    toroidal_flux = ToroidalFlux(surface, native_field)
    target_flux = OFFICIAL_FLUX_MULTIPLIER * float(toroidal_flux.J())
    flux_field = BiotSavartJAX(native_field.coils)
    flux_solver = BoozerSurfaceJAX(
        flux_field,
        surface,
        toroidal_flux,
        target_flux,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
        options=options,
        surface_runtime_state=area_solver.surface_runtime_state,
    )
    expanded = run_boozer_manual_stage(
        flux_solver,
        polished.state,
        tol=OFFICIAL_SOLVER_TOLERANCE,
        maxiter=OFFICIAL_LS_MAXITER,
        constraint_weight=OFFICIAL_CONSTRAINT_WEIGHT,
    )

    final_iota = expanded.state.iota
    final_G = expanded.state.G
    final_residual = _residual_norm(surface, final_iota, final_G, native_field)
    volume = float(Volume(surface).J())
    solver_success = polished.success and expanded.success
    scientific_success = bool(
        solver_success
        and np.isfinite(final_residual)
        and final_residual < initial_residual
        and np.isfinite(final_iota)
        and np.isfinite(volume)
        and abs(volume) > 0.0
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_residual": initial_residual,
            "final_residual": final_residual,
            "iota": final_iota,
            "volume": volume,
            "solver_success": solver_success,
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-boozer-",
        bounded_steps=20,
        native_default_steps=OFFICIAL_LS_MAXITER,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
