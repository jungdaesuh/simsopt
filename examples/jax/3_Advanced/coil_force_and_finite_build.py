"""Evaluate a finite-build coil-force objective and its material frame.

Native curves, currents, and regularized coils form the host problem. JAX
adapters evaluate the differentiable force objective and Frenet frame.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from simsopt.field import Current, RegularizedCoil
from simsopt.geo import CurveXYZFourier
from simsopt_jax.examples import ExecutionScale, example_runtime_metadata
from simsopt_jax_adapters.field.force import LpCurveForce
from simsopt_jax_adapters.geo.framed_curve import (
    FramedCurveFrenetJAX,
    ZeroRotationJAX,
)

EXAMPLE_ID = "coil-force-and-finite-build"


@dataclass(frozen=True)
class ExampleResult:
    force_objective: float
    force_objective_oracle_error: float
    gradient_fd_error: float
    frame_orthonormality_error: float
    planar_torsion_max: float
    status: Literal["ok", "failed"]

    def json_object(self, scale: ExecutionScale) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            **example_runtime_metadata(scale),
            "status": self.status,
            "observables": {
                "force_objective": self.force_objective,
                "force_objective_oracle_error": self.force_objective_oracle_error,
                "gradient_fd_error": self.gradient_fd_error,
                "frame_orthonormality_error": self.frame_orthonormality_error,
                "planar_torsion_max": self.planar_torsion_max,
            },
        }


def _circle(radius: float, quadrature_points: int) -> CurveXYZFourier:
    curve = CurveXYZFourier(quadrature_points, order=1)
    curve.x = radius * np.asarray(
        (0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        dtype=np.float64,
    )
    return curve


def _directional_fd_error(objective: LpCurveForce) -> float:
    initial_dofs = np.asarray(objective.x, dtype=np.float64).copy()
    direction = np.sin(np.arange(initial_dofs.size, dtype=np.float64) + 1.0)
    direction /= np.linalg.norm(direction)
    directional_gradient = float(
        np.dot(np.asarray(jax.device_get(objective.dJ())), direction)
    )
    step = 1.0e-5
    objective.x = initial_dofs + step * direction
    value_plus = float(jax.device_get(objective.J()))
    objective.x = initial_dofs - step * direction
    value_minus = float(jax.device_get(objective.J()))
    objective.x = initial_dofs
    return abs(directional_gradient - (value_plus - value_minus) / (2.0 * step))


def _solve() -> ExampleResult:
    current = 1.7e4
    regularization = 0.05**2 / np.sqrt(np.e)
    target = RegularizedCoil(
        _circle(1.0, 32),
        Current(current),
        regularization,
    )
    source = RegularizedCoil(
        _circle(1.3, 32),
        Current(-current),
        regularization,
    )
    force_objective = LpCurveForce(target, [target, source], p=2.0, threshold=0.0)
    objective_value = float(jax.device_get(force_objective.J()))

    tangent_norm = np.linalg.norm(target.curve.gammadash(), axis=1)
    force_norm = np.linalg.norm(target.force([target, source]), axis=1) / 1.0e6
    objective_oracle = 0.5 * np.sum(force_norm**2 * tangent_norm) / force_norm.size
    objective_oracle_error = abs(objective_value - float(objective_oracle))
    gradient_fd_error = _directional_fd_error(force_objective)

    framed_curve = FramedCurveFrenetJAX(
        target.curve, ZeroRotationJAX(target.curve.quadpoints)
    )
    tangent, normal, binormal = (
        np.asarray(jax.device_get(component), dtype=np.float64)
        for component in framed_curve.rotated_frame()
    )
    frame = np.stack((tangent, normal, binormal), axis=1)
    gram = frame @ np.swapaxes(frame, 1, 2)
    identity = np.broadcast_to(np.eye(3), gram.shape)
    frame_orthonormality_error = float(np.max(np.abs(gram - identity)))
    planar_torsion_max = float(
        np.max(
            np.abs(
                np.asarray(
                    jax.device_get(framed_curve.frame_torsion()),
                    dtype=np.float64,
                )
            )
        )
    )
    is_correct = bool(
        objective_oracle_error <= 1.0e-12
        and gradient_fd_error <= 1.0e-6
        and frame_orthonormality_error <= 1.0e-12
        and planar_torsion_max <= 1.0e-12
    )
    return ExampleResult(
        force_objective=objective_value,
        force_objective_oracle_error=objective_oracle_error,
        gradient_fd_error=gradient_fd_error,
        frame_orthonormality_error=frame_orthonormality_error,
        planar_torsion_max=planar_torsion_max,
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
        scale: ExecutionScale = "bounded" if options.smoke else "native_default"
        print(json.dumps(result.json_object(scale), sort_keys=True))
    else:
        print(f"force objective={result.force_objective:.8e}")
        print(f"gradient FD error={result.gradient_fd_error:.3e}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
