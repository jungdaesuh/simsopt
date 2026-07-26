"""Minimize Fourier deformations of a circle with ``CurveLengthJAX``.

Host boundary: a native ``CurveXYZFourier`` owns coefficients and receives
accepted optimizer states. Objective values and derivatives are evaluated by
the public JAX-backed adapter.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from scipy.optimize import minimize
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision
from simsopt_jax_adapters.geo.curve_objectives import CurveLengthJAX

EXAMPLE_ID = "curve-length-optimization"
RADIUS = 2.0
DEFORMATION_NAMES = ("xs(2)", "yc(2)", "zs(2)")


@dataclass(frozen=True)
class ExampleResult:
    final_length: float
    circle_oracle: float
    gradient_fd_error: float
    optimizer_success: bool
    status: Literal["ok", "failed"]

    def json_object(self) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            "backend_mode": get_backend_mode(),
            "platform": jax.devices()[0].platform,
            "precision": get_resolved_precision(),
            "status": self.status,
            "observables": {
                "final_length": self.final_length,
                "circle_oracle": self.circle_oracle,
                "gradient_fd_error": self.gradient_fd_error,
                "optimizer_success": self.optimizer_success,
            },
        }


def _build_problem() -> tuple[CurveXYZFourier, CurveLengthJAX]:
    curve = CurveXYZFourier(64, order=2)
    curve.set("xc(1)", RADIUS)
    curve.set("ys(1)", RADIUS)
    curve.set("xs(2)", 0.3)
    curve.set("yc(2)", -0.2)
    curve.set("zs(2)", 0.4)
    curve.fix_all()
    for name in DEFORMATION_NAMES:
        curve.unfix(name)
    return curve, CurveLengthJAX(curve)


def _solve(max_steps: int) -> ExampleResult:
    curve, objective = _build_problem()

    def value(parameters: np.ndarray) -> float:
        curve.x = parameters
        return float(objective.J())

    def gradient(parameters: np.ndarray) -> np.ndarray:
        curve.x = parameters
        return np.asarray(objective.dJ(), dtype=np.float64)

    initial = np.asarray(curve.x, dtype=np.float64)
    direction = np.asarray((0.3, -0.4, 0.5), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    epsilon = 1.0e-6
    analytic_directional = float(np.dot(gradient(initial), direction))
    finite_difference = (
        value(initial + epsilon * direction) - value(initial - epsilon * direction)
    ) / (2.0 * epsilon)
    gradient_fd_error = abs(analytic_directional - finite_difference)
    curve.x = initial

    result = minimize(
        value,
        initial,
        jac=gradient,
        method="BFGS",
        options={"maxiter": max_steps, "gtol": 1.0e-10},
    )
    curve.x = result.x
    final_length = float(objective.J())
    circle_oracle = 2.0 * np.pi * RADIUS
    is_correct = bool(
        result.success
        and np.isclose(final_length, circle_oracle, rtol=1.0e-10, atol=1.0e-10)
        and gradient_fd_error <= 1.0e-6
    )
    return ExampleResult(
        final_length=final_length,
        circle_oracle=circle_oracle,
        gradient_fd_error=gradient_fd_error,
        optimizer_success=bool(result.success),
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
    result = _solve(options.max_steps or (32 if options.smoke else 128))
    if options.json:
        print(json.dumps(result.json_object(), sort_keys=True))
    else:
        print(f"final length={result.final_length:.12f}")
        print(f"circle oracle={result.circle_oracle:.12f}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
