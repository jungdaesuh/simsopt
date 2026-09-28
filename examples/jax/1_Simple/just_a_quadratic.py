"""JAX mirror of ``examples/1_Simple/just_a_quadratic.py``.

The three optimization variables are independent.  Their weighted residuals
have targets 1, 2, and 3 and weights 1, 2, and 3, exactly as in the native
SIMSOPT example.  The same implementation runs on the selected JAX CPU or GPU
device; ``SIMSOPT_BACKEND_MODE`` selects fast or parity solver intent.
"""

from __future__ import annotations

import argparse
import json
from contextlib import chdir
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import numpy as np
from simsopt_jax.examples import (
    ExecutionScale,
    example_runtime_metadata,
)
from simsopt_jax.examples.official_tiny_least_squares import (
    DRIVER_QUADRATIC,
    guard_finite_endpoint,
    quadratic_residual,
    solve_jax_residual,
    trf_outcome,
    value_and_jacobian,
)

EXAMPLE_ID = "native-just-a-quadratic"
INITIAL_PARAMETERS = (0.0, 0.0, 0.0)
TARGETS = (1.0, 2.0, 3.0)
WEIGHTS = (1.0, 2.0, 3.0)


@dataclass(frozen=True)
class ExampleResult:
    initial_parameters: tuple[float, ...]
    targets: tuple[float, ...]
    weights: tuple[float, ...]
    initial_residuals: tuple[float, ...]
    initial_jacobian: tuple[tuple[float, ...], ...]
    initial_objective: float
    solution: tuple[float, ...]
    objective: float
    residual_norm: float
    gradient_inf_norm: float
    solver_driver: str
    solver_status: int
    solver_success: bool
    iterations: int | None
    function_evaluations: int
    jacobian_evaluations: int
    status: Literal["ok", "failed"]

    def json_object(self, scale: ExecutionScale) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            **example_runtime_metadata(scale),
            "status": self.status,
            "observables": {
                "initial_parameters": self.initial_parameters,
                "targets": self.targets,
                "weights": self.weights,
                "initial_residuals": self.initial_residuals,
                "initial_jacobian": self.initial_jacobian,
                "initial_objective": self.initial_objective,
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


def solve(output_directory: Path, max_steps: int | None) -> ExampleResult:
    """Solve the quadratic and collect scientific parity observables."""

    initial = np.asarray(INITIAL_PARAMETERS, dtype=np.float64)
    residual = quadratic_residual(
        np.asarray(TARGETS, dtype=np.float64),
        np.asarray(WEIGHTS, dtype=np.float64),
    )
    initial_residuals, initial_jacobian = value_and_jacobian(residual, initial)
    with chdir(output_directory):
        optimizer = solve_jax_residual(residual, initial, max_nfev=max_steps)
    solution = np.asarray(optimizer.x, dtype=np.float64)
    residuals, final_jacobian = value_and_jacobian(residual, solution)
    gradient = 2.0 * final_jacobian.T @ residuals
    initial_objective = float(np.dot(initial_residuals, initial_residuals))
    objective = float(np.dot(residuals, residuals))
    residual_norm = float(np.linalg.norm(residuals))
    gradient_inf_norm = float(np.linalg.norm(gradient, ord=np.inf))
    # The outcome is the SciPy provider's, never an accuracy threshold of our
    # own: the official run itself stops 7.90e-10 away from the exact minimizer
    # (A/reference-simple/runs/native-just-a-quadratic/captured-natural-omp1:
    # status 1 `gtol`, nfev 4, x = (1.0, 1.9999999992096138, 2.9999999998773283)).
    # The endpoint accuracy is published below as a diagnostic instead.
    outcome = guard_finite_endpoint(
        trf_outcome(optimizer), (solution, residuals, gradient)
    )

    return ExampleResult(
        initial_parameters=INITIAL_PARAMETERS,
        targets=TARGETS,
        weights=WEIGHTS,
        initial_residuals=tuple(float(value) for value in initial_residuals),
        initial_jacobian=tuple(
            tuple(float(value) for value in row) for row in initial_jacobian
        ),
        initial_objective=initial_objective,
        solution=tuple(float(value) for value in solution),
        objective=objective,
        residual_norm=residual_norm,
        gradient_inf_norm=gradient_inf_norm,
        solver_driver=DRIVER_QUADRATIC,
        solver_status=int(optimizer.status),
        solver_success=bool(optimizer.success),
        iterations=None,
        function_evaluations=outcome.nfev,
        jacobian_evaluations=outcome.njev,
        status="ok" if outcome.success else "failed",
    )


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = argument_parser().parse_args(arguments)
    # No scale caps the official workflow: without an explicit --max-steps the
    # solve runs SciPy's own default budget (100 * n), the official policy.
    max_steps = options.max_steps
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        result = solve(options.output_dir, max_steps)
    elif options.smoke:
        with TemporaryDirectory(prefix="simsopt-jax-just-a-quadratic-") as temporary:
            result = solve(Path(temporary), max_steps)
    else:
        result = solve(Path.cwd(), max_steps)

    if options.json:
        scale: ExecutionScale = "bounded" if options.smoke else "native_default"
        print(json.dumps(result.json_object(scale), sort_keys=True))
    else:
        print(f"solution={result.solution}")
        print(f"objective={result.objective:.6e}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
