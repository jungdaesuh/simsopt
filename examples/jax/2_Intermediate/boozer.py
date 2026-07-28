"""JAX port of ``examples/2_Intermediate/boozer.py``.

The host constructs the canonical NCSX coils and tensor-Fourier surface.  A
``BoozerSurfaceJAX`` first reduces the area-constrained Boozer residual, then
polishes it with the least-squares solver, and finally relabels the surface by
toroidal flux and expands it to three times the converged flux.  The numerical
solves execute on the selected JAX device.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import cast

import jax
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
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX

EXAMPLE_ID = "native-boozer"
NATIVE_LS_ITERATIONS = 100


def _host_float(value: object) -> float:
    return float(np.asarray(jax.device_get(value), dtype=np.float64))


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


def _options(max_steps: int) -> dict[str, object]:
    return {
        "bfgs_maxiter": 3 * max_steps,
        "bfgs_tol": 1.0e-10,
        "newton_maxiter": NATIVE_LS_ITERATIONS,
        "newton_tol": 1.0e-10,
        "verbose": False,
    }


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    base_curves, base_currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    del base_curves
    magnetic_axis = cast(Curve, magnetic_axis)
    field = BiotSavartJAX(native_field.coils)
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    G0 = 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))

    native_scale = max_steps >= NATIVE_LS_ITERATIONS
    mpol = 5 if native_scale else 2
    ntor = 5 if native_scale else 2
    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, 2 * ntor + 1, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 2 * mpol + 1, endpoint=False),
    )
    surface.fit_to_curve(magnetic_axis, 0.10, flip_theta=True)
    initial_iota = -0.4
    initial_residual = _residual_norm(surface, initial_iota, G0, native_field)

    area = Area(surface)
    area_solver = BoozerSurfaceJAX(
        field,
        surface,
        area,
        float(area.J()),
        constraint_weight=100.0,
        options=_options(max_steps),
    )
    rough = cast(
        Mapping[str, object],
        area_solver.minimize_boozer_penalty_constraints_LBFGS(
            tol=1.0e-10,
            maxiter=3 * max_steps,
            constraint_weight=100.0,
            iota=initial_iota,
            G=G0,
        ),
    )
    area_solver.need_to_run_code = True
    polished = cast(
        Mapping[str, object],
        area_solver.minimize_boozer_penalty_constraints_ls(
            tol=1.0e-10,
            maxiter=NATIVE_LS_ITERATIONS,
            constraint_weight=100.0,
            iota=_host_float(rough["iota"]),
            G=_host_float(rough["G"]),
            method="manual",
        ),
    )

    toroidal_flux = ToroidalFlux(surface, native_field)
    target_flux = 3.0 * float(toroidal_flux.J())
    flux_field = BiotSavartJAX(native_field.coils)
    flux_solver = BoozerSurfaceJAX(
        flux_field,
        surface,
        toroidal_flux,
        target_flux,
        constraint_weight=100.0,
        options=_options(max_steps),
        surface_runtime_state=area_solver.surface_runtime_state,
    )
    expanded = cast(
        Mapping[str, object],
        flux_solver.minimize_boozer_penalty_constraints_ls(
            tol=1.0e-10,
            maxiter=NATIVE_LS_ITERATIONS,
            constraint_weight=100.0,
            iota=_host_float(polished["iota"]),
            G=_host_float(polished["G"]),
            method="manual",
        ),
    )

    final_iota = _host_float(expanded["iota"])
    final_G = _host_float(expanded["G"])
    final_residual = _residual_norm(surface, final_iota, final_G, native_field)
    volume = float(Volume(surface).J())
    solver_success = bool(polished["success"]) and bool(expanded["success"])
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
        native_default_steps=NATIVE_LS_ITERATIONS,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
