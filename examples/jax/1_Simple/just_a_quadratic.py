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

import jax
import numpy as np
from simsopt_jax.examples import (
    ExecutionScale,
    example_runtime_metadata,
    solve_weighted_quadratic,
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
    iterations: int
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


def solve(output_directory: Path, max_steps: int) -> ExampleResult:
    """Solve the quadratic and collect scientific parity observables."""

    with chdir(output_directory):
        device_result = solve_weighted_quadratic(
            initial_parameters=jax.device_put(
                np.asarray(INITIAL_PARAMETERS, dtype=np.float64)
            ),
            targets=jax.device_put(np.asarray(TARGETS, dtype=np.float64)),
            weights=jax.device_put(np.asarray(WEIGHTS, dtype=np.float64)),
            max_steps=max_steps,
            rtol=1.0e-12,
            atol=1.0e-12,
        )

    initial_residuals = np.asarray(
        jax.device_get(device_result.initial_residuals), dtype=np.float64
    )
    initial_jacobian = np.asarray(
        jax.device_get(device_result.initial_jacobian), dtype=np.float64
    )
    solution = np.asarray(
        jax.device_get(device_result.final_parameters), dtype=np.float64
    )
    residuals = np.asarray(
        jax.device_get(device_result.final_residuals), dtype=np.float64
    )
    gradient = np.asarray(
        jax.device_get(device_result.final_gradient), dtype=np.float64
    )
    initial_objective = float(jax.device_get(device_result.initial_objective))
    objective = float(jax.device_get(device_result.final_objective))
    residual_norm = float(np.linalg.norm(residuals))
    gradient_inf_norm = float(np.linalg.norm(gradient, ord=np.inf))
    scientific_success = bool(
        np.allclose(solution, np.asarray(TARGETS), rtol=1.0e-10, atol=1.0e-10)
        and objective <= 1.0e-16
        and device_result.optimizer.success
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
        solver_driver=device_result.optimizer.driver.value,
        solver_status=device_result.optimizer.status,
        solver_success=device_result.optimizer.success,
        iterations=device_result.optimizer.nit,
        function_evaluations=device_result.optimizer.nfev,
        jacobian_evaluations=device_result.optimizer.njev,
        status="ok" if scientific_success else "failed",
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
    max_steps = options.max_steps or (32 if options.smoke else 128)
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
