"""Minimize fixed-surface quadratic flux by updating one coil current.

Host boundary: native curve, current, coil, and surface objects are constructed
once. JAX adapters snapshot their layout before tracing the complete
current-to-field-to-flux calculation. Accepted current states are then
published back to the native owner.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from simsopt.field import Current, coils_via_symmetries
from simsopt.geo import SurfaceRZFourier, create_equally_spaced_curves
from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.curve_objectives import CurveLengthJAX
from simsopt_jax_adapters.objectives.flux import SquaredFluxJAX

EXAMPLE_ID = "coil-flux-optimization"
INITIAL_CURRENT = 1.0e5
COIL_MINOR_RADIUS = 0.5


@dataclass(frozen=True)
class ExampleResult:
    initial_flux: float
    final_flux: float
    optimized_current: float
    gradient_fd_error: float
    coil_length: float
    coil_length_oracle: float
    status: Literal["ok", "failed"]

    def json_object(self) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            "backend_mode": get_backend_mode(),
            "platform": jax.devices()[0].platform,
            "precision": get_resolved_precision(),
            "status": self.status,
            "observables": {
                "initial_flux": self.initial_flux,
                "final_flux": self.final_flux,
                "optimized_current": self.optimized_current,
                "gradient_fd_error": self.gradient_fd_error,
                "coil_length": self.coil_length,
                "coil_length_oracle": self.coil_length_oracle,
            },
        }


def _build_objective() -> tuple[SquaredFluxJAX, CurveLengthJAX]:
    curves = create_equally_spaced_curves(
        1,
        1,
        stellsym=False,
        R0=1.0,
        R1=COIL_MINOR_RADIUS,
        order=1,
        numquadpoints=8,
    )
    curves[0].fix_all()
    coils = coils_via_symmetries(curves, [Current(INITIAL_CURRENT)], 1, False)
    quadrature = np.linspace(0.0, 1.0, 4, endpoint=False)
    surface = SurfaceRZFourier(
        nfp=1,
        stellsym=False,
        mpol=1,
        ntor=0,
        quadpoints_phi=quadrature,
        quadpoints_theta=quadrature,
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)
    surface.fix_all()
    flux = SquaredFluxJAX(
        surface,
        BiotSavartJAX(coils),
        definition="quadratic flux",
    )
    return flux, CurveLengthJAX(curves[0])


def _solve() -> ExampleResult:
    objective, length_objective = _build_objective()
    initial = np.asarray(objective.x, dtype=np.float64)
    initial_flux = float(objective.J())
    analytic_gradient = float(np.asarray(objective.dJ(), dtype=np.float64)[0])
    epsilon = 1.0
    objective.x = initial + epsilon
    value_plus = float(objective.J())
    objective.x = initial - epsilon
    value_minus = float(objective.J())
    finite_difference = (value_plus - value_minus) / (2.0 * epsilon)
    gradient_fd_error = abs(analytic_gradient - finite_difference) / max(
        abs(analytic_gradient), 1.0e-30
    )

    # With fixed geometry and one current, quadratic flux is homogeneous of
    # degree two. This is its exact one-dimensional Newton/line-minimizer step.
    optimized_current = float(initial[0] - 2.0 * initial_flux / analytic_gradient)
    objective.x = np.asarray((optimized_current,), dtype=np.float64)
    final_flux = float(objective.J())
    coil_length = float(length_objective.J())
    coil_length_oracle = 2.0 * np.pi * COIL_MINOR_RADIUS
    is_correct = bool(
        gradient_fd_error <= 1.0e-8
        and final_flux <= 1.0e-6 * initial_flux
        and abs(optimized_current) <= 1.0e-6
        and np.isclose(coil_length, coil_length_oracle, rtol=1.0e-12, atol=1.0e-12)
    )
    return ExampleResult(
        initial_flux=initial_flux,
        final_flux=final_flux,
        optimized_current=optimized_current,
        gradient_fd_error=gradient_fd_error,
        coil_length=coil_length,
        coil_length_oracle=coil_length_oracle,
        status="ok" if is_correct else "failed",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    result = _solve()
    if options.json:
        print(json.dumps(result.json_object(), sort_keys=True))
    else:
        print(f"initial flux={result.initial_flux:.6e}")
        print(f"final flux={result.final_flux:.6e}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
