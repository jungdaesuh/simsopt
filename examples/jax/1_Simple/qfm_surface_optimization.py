"""Optimize a bounded NCSX QFM surface with public JAX adapters.

Host boundary: canonical NCSX coils and a native surface are constructed on
the host. ``QfmSurfaceJAX`` snapshots both, runs the compiled penalty solve,
and publishes the accepted surface coefficients once at completion.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from simsopt.configs.zoo import get_data
from simsopt.geo import SurfaceRZFourier
from simsopt.geo.surfaceobjectives import Area
from simsopt_jax.examples import ExecutionScale, example_runtime_metadata
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.qfm_surface import QfmSurfaceJAX

EXAMPLE_ID = "qfm-surface-optimization"


@dataclass(frozen=True)
class ExampleResult:
    initial_penalty: float
    final_penalty: float
    initial_gradient_norm: float
    gradient_norm: float
    surface_update_norm: float
    solver_success: bool
    iterations: int
    status: Literal["ok", "failed"]

    def json_object(self, scale: ExecutionScale) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            **example_runtime_metadata(scale),
            "status": self.status,
            "observables": {
                "initial_penalty": self.initial_penalty,
                "final_penalty": self.final_penalty,
                "initial_gradient_norm": self.initial_gradient_norm,
                "gradient_norm": self.gradient_norm,
                "surface_update_norm": self.surface_update_norm,
                "solver_success": self.solver_success,
                "iterations": self.iterations,
            },
        }


def _build_problem() -> QfmSurfaceJAX:
    _base_curves, _base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
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
    label = Area(surface)
    return QfmSurfaceJAX(
        BiotSavartJAX(biotsavart.coils),
        surface,
        label,
        0.98 * float(label.J()),
    )


def _solve(max_steps: int) -> ExampleResult:
    qfm = _build_problem()
    initial_dofs = np.asarray(qfm.surface.get_dofs(), dtype=np.float64)
    initial_penalty, initial_gradient = qfm.qfm_penalty_constraints(
        initial_dofs,
        derivatives=1,
    )
    result = qfm.minimize_qfm_penalty_jax(
        tol=1.0e-8,
        maxiter=max_steps,
        constraint_weight=1.0,
    )
    final_dofs = np.asarray(qfm.surface.get_dofs(), dtype=np.float64)
    final_penalty = float(jax.device_get(result["fun"]))
    final_gradient = np.asarray(jax.device_get(result["gradient"]), dtype=np.float64)
    initial_gradient_norm = float(np.linalg.norm(initial_gradient))
    gradient_norm = float(np.linalg.norm(final_gradient))
    update_norm = float(np.linalg.norm(final_dofs - initial_dofs))
    solver_success = bool(result["success"])
    is_correct = bool(
        solver_success
        and final_penalty < float(jax.device_get(initial_penalty))
        and gradient_norm < initial_gradient_norm
        and gradient_norm <= 1.0e-8
        and update_norm > 0.0
    )
    return ExampleResult(
        initial_penalty=float(jax.device_get(initial_penalty)),
        final_penalty=final_penalty,
        initial_gradient_norm=initial_gradient_norm,
        gradient_norm=gradient_norm,
        surface_update_norm=update_norm,
        solver_success=solver_success,
        iterations=int(jax.device_get(result["iter"])),
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
    result = _solve(options.max_steps or (75 if options.smoke else 200))
    if options.json:
        scale: ExecutionScale = "bounded" if options.smoke else "native_default"
        print(json.dumps(result.json_object(scale), sort_keys=True))
    else:
        print(f"initial penalty={result.initial_penalty:.6e}")
        print(f"final penalty={result.final_penalty:.6e}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
