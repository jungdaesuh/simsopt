"""Solve a bounded NCSX Boozer-surface problem with public JAX adapters.

Host boundary: canonical coils, surface construction, and final reporting stay
on the host. ``BoozerSurfaceJAX`` snapshots the fixed coil state, executes the
inner solve, and publishes one accepted surface state.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from simsopt.configs.zoo import get_data
from simsopt.geo import Area, SurfaceRZFourier
from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.surface_objectives import BoozerResidualJAX

EXAMPLE_ID = "boozer-surface-optimization"


@dataclass(frozen=True)
class ExampleResult:
    solver_success: bool
    iterations: int
    residual_norm: float
    residual_objective: float
    final_gradient_inf_norm: float
    surface_update_norm: float
    iota: float
    status: Literal["ok", "failed"]

    def json_object(self) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            "backend_mode": get_backend_mode(),
            "platform": jax.devices()[0].platform,
            "precision": get_resolved_precision(),
            "status": self.status,
            "observables": {
                "solver_success": self.solver_success,
                "iterations": self.iterations,
                "residual_norm": self.residual_norm,
                "residual_objective": self.residual_objective,
                "final_gradient_inf_norm": self.final_gradient_inf_norm,
                "surface_update_norm": self.surface_update_norm,
                "iota": self.iota,
            },
        }


def _build_problem(max_steps: int) -> tuple[BoozerSurfaceJAX, BiotSavartJAX]:
    _base_curves, _base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    for coil in biotsavart.coils:
        coil.curve.fix_all()
        coil.current.fix_all()
    quadrature_phi = np.linspace(0.0, 1.0 / nfp, 3, endpoint=False)
    quadrature_theta = np.linspace(0.0, 1.0, 3, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=quadrature_phi,
        quadpoints_theta=quadrature_theta,
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    label = Area(surface)
    field = BiotSavartJAX(biotsavart.coils)
    return (
        BoozerSurfaceJAX(
            field,
            surface,
            label,
            float(label.J()),
            constraint_weight=1.0,
            options={
                "bfgs_maxiter": max_steps,
                "newton_maxiter": max_steps,
                "newton_tol": 1.0e-8,
                "verbose": False,
            },
        ),
        field,
    )


def _solve(max_steps: int) -> ExampleResult:
    boozer, field = _build_problem(max_steps)
    initial_dofs = np.asarray(boozer.surface.get_dofs(), dtype=np.float64)
    result = boozer.run_code(iota=0.4)
    if result is None:
        raise RuntimeError("fresh BoozerSurfaceJAX unexpectedly returned no result")
    final_dofs = np.asarray(boozer.surface.get_dofs(), dtype=np.float64)
    residual_norm = float(
        np.linalg.norm(np.asarray(jax.device_get(result["residual"])))
    )
    residual_objective = float(jax.device_get(BoozerResidualJAX(boozer, field).J()))
    gradient_inf_norm = float(jax.device_get(result["final_gradient_inf_norm"]))
    update_norm = float(np.linalg.norm(final_dofs - initial_dofs))
    solver_success = bool(result["success"])
    is_correct = bool(
        solver_success
        and residual_norm < 1.0
        and np.isfinite(residual_objective)
        and gradient_inf_norm <= 1.0e-8
        and update_norm > 0.0
    )
    return ExampleResult(
        solver_success=solver_success,
        iterations=int(jax.device_get(result["iter"])),
        residual_norm=residual_norm,
        residual_objective=residual_objective,
        final_gradient_inf_norm=gradient_inf_norm,
        surface_update_norm=update_norm,
        iota=float(jax.device_get(result["iota"])),
        status="ok" if is_correct else "failed",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-steps", type=int)
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    result = _solve(options.max_steps or (20 if options.smoke else 60))
    if options.json:
        print(json.dumps(result.json_object(), sort_keys=True))
    else:
        print(f"success={result.solver_success}, iota={result.iota:.8f}")
        print(f"residual norm={result.residual_norm:.6e}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
