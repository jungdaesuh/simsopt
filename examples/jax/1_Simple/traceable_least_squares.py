"""Solve an immutable weighted quadratic with the public JAX serial solver."""

from __future__ import annotations

import argparse
import json
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import jax
import numpy as np
from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision
from simsopt_jax.solve.serial import (
    TraceableLeastSquaresProblem,
    least_squares_serial_solve_jax,
)

EXAMPLE_ID = "traceable-least-squares"
TARGET = np.asarray((1.0, 2.0, 3.0), dtype=np.float64)
WEIGHTS = np.sqrt(np.asarray((1.0, 2.0, 3.0), dtype=np.float64))
RESIDUAL_JACOBIAN = np.diag(WEIGHTS)


@dataclass(frozen=True)
class ExampleResult:
    solution: tuple[float, ...]
    objective: float
    residual_norm: float
    gradient_inf_norm: float
    solver_driver: str
    solver_status: int
    solver_success: bool
    iterations: int
    function_evaluations: int
    jacobian_evaluations: int
    status: Literal["ok", "failed"]

    def json_object(self) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            "backend_mode": get_backend_mode(),
            "platform": jax.devices()[0].platform,
            "precision": get_resolved_precision(),
            "status": self.status,
            "observables": {
                "solution": self.solution,
                "objective": self.objective,
                "residual_norm": self.residual_norm,
                "gradient_inf_norm": self.gradient_inf_norm,
                "solver_driver": self.solver_driver,
                "solver_status": self.solver_status,
                "solver_success": self.solver_success,
                "iterations": self.iterations,
                "function_evaluations": self.function_evaluations,
                "jacobian_evaluations": self.jacobian_evaluations,
            },
        }


def _weighted_residuals(parameters: jax.Array) -> jax.Array:
    return WEIGHTS * (parameters - TARGET)


def _solve(output_directory: Path, max_steps: int) -> ExampleResult:
    problem = TraceableLeastSquaresProblem(
        residual_fn=_weighted_residuals,
        x=jax.device_put(np.zeros(3, dtype=np.float64)),
    )
    with chdir(output_directory):
        solver_result = least_squares_serial_solve_jax(
            problem,
            max_steps=max_steps,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    solution = np.asarray(jax.device_get(problem.x), dtype=np.float64)
    objective = float(jax.device_get(problem.objective()))
    residual = np.asarray(
        jax.device_get(problem.residuals()),
        dtype=np.float64,
    )
    residual_jacobian = np.asarray(RESIDUAL_JACOBIAN, dtype=np.float64)
    gradient = residual_jacobian.T @ residual
    residual_norm = float(np.linalg.norm(residual))
    gradient_inf_norm = float(np.linalg.norm(gradient, ord=np.inf))
    is_correct = bool(
        np.allclose(solution, np.asarray(TARGET), rtol=1.0e-10, atol=1.0e-10)
        and objective <= 1.0e-16
        and solver_result.success
    )
    return ExampleResult(
        solution=tuple(float(value) for value in solution),
        objective=objective,
        residual_norm=residual_norm,
        gradient_inf_norm=gradient_inf_norm,
        solver_driver=solver_result.driver.value,
        solver_status=solver_result.status,
        solver_success=solver_result.success,
        iterations=solver_result.nit,
        function_evaluations=solver_result.nfev,
        jacobian_evaluations=solver_result.njev,
        status="ok" if is_correct else "failed",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    max_steps = options.max_steps or (32 if options.smoke else 128)
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        result = _solve(options.output_dir, max_steps)
    elif options.smoke:
        with TemporaryDirectory(prefix="simsopt-jax-example-") as temporary:
            result = _solve(Path(temporary), max_steps)
    else:
        result = _solve(Path.cwd(), max_steps)

    if options.json:
        print(json.dumps(result.json_object(), sort_keys=True))
    else:
        print(f"solution={result.solution}")
        print(f"objective={result.objective:.6e}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
