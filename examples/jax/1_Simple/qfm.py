"""JAX mirror of ``examples/1_Simple/qfm.py``.

Host construction loads the canonical NCSX coils and constructs each native
surface label.  Immutable surface and coil specifications enter
``QfmSurfaceJAX``.  For volume, toroidal flux, and area in sequence, the example
runs the penalty solve followed by SIMSOPT's JAX augmented-Lagrangian equality
solve and publishes the accepted surface state once per solve.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from simsopt.configs.zoo import get_data
from simsopt.geo import SurfaceRZFourier
from simsopt.geo.surfaceobjectives import Area, ToroidalFlux, Volume
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.qfm_surface import QfmSurfaceJAX

EXAMPLE_ID = "native-qfm"


@dataclass(frozen=True)
class StageResult:
    label: Literal["volume", "toroidal_flux", "area"]
    penalty_success: bool
    exact_success: bool
    qfm_value: float
    constraint_objective: float


def _build_surface() -> tuple[SurfaceRZFourier, BiotSavartJAX]:
    _curves, _currents, magnetic_axis, nfp, native_field = get_data("ncsx")
    phis = np.linspace(0.0, 1.0 / nfp, 6, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 6, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    return surface, BiotSavartJAX(native_field.coils)


def _state(qfm: QfmSurfaceJAX) -> tuple[float, np.ndarray, float, np.ndarray]:
    parameters = np.asarray(qfm.surface.get_dofs(), dtype=np.float64)
    qfm_value, qfm_gradient = qfm.qfm_objective(parameters, derivatives=1)
    constraint_value, constraint_gradient = qfm.qfm_label_constraint(
        parameters, derivatives=1
    )
    return (
        float(qfm_value),
        np.asarray(qfm_gradient, dtype=np.float64),
        float(constraint_value),
        np.asarray(constraint_gradient, dtype=np.float64),
    )


def _solve_stage(
    qfm: QfmSurfaceJAX,
    label: Literal["volume", "toroidal_flux", "area"],
    max_steps: int,
) -> StageResult:
    penalty = qfm.minimize_qfm_penalty_jax(
        tol=1.0e-8,
        maxiter=max_steps,
        constraint_weight=1.0,
    )
    exact = qfm.minimize_qfm_exact_jax(
        tol=1.0e-8,
        maxiter=max_steps,
    )
    qfm_value, _qfm_gradient, constraint, _constraint_gradient = _state(qfm)
    return StageResult(
        label=label,
        penalty_success=bool(penalty["success"]),
        exact_success=bool(exact["success"]),
        qfm_value=qfm_value,
        constraint_objective=constraint,
    )


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    surface, field = _build_surface()
    initial_parameters = np.asarray(surface.get_dofs(), dtype=np.float64)
    volume = Volume(surface)
    volume_qfm = QfmSurfaceJAX(field, surface, volume, float(volume.J()))
    (
        initial_qfm,
        initial_qfm_gradient,
        initial_constraint,
        initial_constraint_gradient,
    ) = _state(volume_qfm)
    volume_result = _solve_stage(volume_qfm, "volume", max_steps)

    toroidal_flux = ToroidalFlux(surface, field)
    flux_qfm = QfmSurfaceJAX(
        field,
        surface,
        toroidal_flux,
        float(toroidal_flux.J()),
    )
    flux_result = _solve_stage(flux_qfm, "toroidal_flux", max_steps)

    area = Area(surface)
    area_qfm = QfmSurfaceJAX(field, surface, area, float(area.J()))
    area_result = _solve_stage(area_qfm, "area", max_steps)
    final_parameters = np.asarray(surface.get_dofs(), dtype=np.float64)
    final_qfm, final_qfm_gradient, final_constraint, final_constraint_gradient = _state(
        area_qfm
    )
    initial_residuals = np.asarray((initial_qfm, initial_constraint), dtype=np.float64)
    initial_jacobian = np.stack((initial_qfm_gradient, initial_constraint_gradient))
    final_residuals = np.asarray((final_qfm, final_constraint), dtype=np.float64)
    final_jacobian = np.stack((final_qfm_gradient, final_constraint_gradient))
    stages = (volume_result, flux_result, area_result)
    scientific_success = bool(
        all(stage.penalty_success and stage.exact_success for stage in stages)
        and np.all(np.isfinite(final_parameters))
        and final_qfm < initial_qfm
        and final_constraint <= 1.0e-10
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_parameters": tuple(float(value) for value in initial_parameters),
            "initial_residuals": tuple(float(value) for value in initial_residuals),
            "initial_jacobian": tuple(
                tuple(float(value) for value in row) for row in initial_jacobian
            ),
            "solution": tuple(float(value) for value in final_parameters),
            "final_residuals": tuple(float(value) for value in final_residuals),
            "final_jacobian": tuple(
                tuple(float(value) for value in row) for row in final_jacobian
            ),
            "constraint_residual": final_constraint,
            "solver_success": scientific_success,
            "stages": tuple(
                {
                    "label": stage.label,
                    "penalty_success": stage.penalty_success,
                    "exact_success": stage.exact_success,
                    "qfm_value": stage.qfm_value,
                    "constraint_objective": stage.constraint_objective,
                }
                for stage in stages
            ),
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-qfm-",
        bounded_steps=80,
        native_default_steps=1000,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
