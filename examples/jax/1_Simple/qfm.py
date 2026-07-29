"""JAX mirror of ``examples/1_Simple/qfm.py``.

Host construction loads the canonical NCSX coils and fitted surface. Immutable
surface and coil specifications then enter one device-resident volume,
toroidal-flux, and area sequence. Each stage runs SIMSOPT's JAX penalty solve
followed by its augmented-Lagrangian equality solve. The completed result tree
crosses to the host once for reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import jax
import numpy as np
from simsopt.configs.zoo import get_data
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import (
    ExampleResult,
    ExecutionScale,
    QfmStageDeviceResult,
    run_example,
    solve_qfm_sequence,
)
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX

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


def _stage_result(
    stage: QfmStageDeviceResult,
    label: Literal["volume", "toroidal_flux", "area"],
) -> StageResult:
    label_residual = float(stage.exact.label_residual_abs)
    return StageResult(
        label=label,
        penalty_success=bool(stage.penalty_optimizer.success),
        exact_success=bool(stage.exact_optimizer.success),
        qfm_value=float(stage.exact.qfm_value),
        constraint_objective=0.5 * label_residual * label_residual,
    )


def solve(
    _output_directory: Path, max_steps: int, _scale: ExecutionScale
) -> ExampleResult:
    surface, field = _build_surface()
    device = get_runtime_jax_device()
    coil_set_spec = field.coil_set_spec_from_dofs(
        jax.device_put(np.asarray(field.x, dtype=np.float64), device)
    )
    device_result = solve_qfm_sequence(
        initial_parameters=jax.device_put(
            np.asarray(surface.get_dofs(), dtype=np.float64),
            device,
        ),
        quadpoints_phi=jax.device_put(
            np.asarray(surface.quadpoints_phi, dtype=np.float64),
            device,
        ),
        quadpoints_theta=jax.device_put(
            np.asarray(surface.quadpoints_theta, dtype=np.float64),
            device,
        ),
        coil_set_spec=coil_set_spec,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        max_steps=max_steps,
        tolerance=1.0e-8,
        constraint_weight=1.0,
    )
    result = jax.device_get(device_result)
    initial = result.volume.initial
    final = result.area.exact
    initial_label_residual = float(initial.label_value - result.volume.target)
    final_label_residual = float(final.label_value - result.area.target)
    initial_constraint = 0.5 * initial_label_residual * initial_label_residual
    final_constraint = 0.5 * final_label_residual * final_label_residual
    initial_residuals = np.asarray(
        (initial.qfm_value, initial_constraint),
        dtype=np.float64,
    )
    initial_jacobian = np.stack(
        (
            initial.qfm_gradient,
            initial_label_residual * initial.label_gradient,
        )
    )
    final_residuals = np.asarray(
        (final.qfm_value, final_constraint),
        dtype=np.float64,
    )
    final_jacobian = np.stack(
        (
            final.qfm_gradient,
            final_label_residual * final.label_gradient,
        )
    )
    stages = (
        _stage_result(result.volume, "volume"),
        _stage_result(result.toroidal_flux, "toroidal_flux"),
        _stage_result(result.area, "area"),
    )
    scientific_success = bool(
        all(stage.penalty_success and stage.exact_success for stage in stages)
        and np.all(np.isfinite(final.parameters))
        and final.qfm_value < initial.qfm_value
        and final_constraint <= 1.0e-10
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_parameters": tuple(float(value) for value in initial.parameters),
            "initial_residuals": tuple(float(value) for value in initial_residuals),
            "initial_jacobian": tuple(
                tuple(float(value) for value in row) for row in initial_jacobian
            ),
            "solution": tuple(float(value) for value in final.parameters),
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
