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
import jax.numpy as jnp
import numpy as np
from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision
from simsopt_jax.solve.serial import (
    TraceableLeastSquaresProblem,
    least_squares_serial_solve_jax,
)

EXAMPLE_ID = "traceable-least-squares"
TARGET = jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float64)
WEIGHTS = jnp.sqrt(jnp.asarray((1.0, 2.0, 3.0), dtype=jnp.float64))


@dataclass(frozen=True)
class ExampleResult:
    solution: tuple[float, ...]
    objective: float
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
            },
        }


def _weighted_residuals(parameters: jax.Array) -> jax.Array:
    return WEIGHTS * (parameters - TARGET)


def _solve(output_directory: Path, max_steps: int) -> ExampleResult:
    problem = TraceableLeastSquaresProblem(
        residual_fn=_weighted_residuals,
        x=jnp.zeros(3, dtype=jnp.float64),
    )
    with chdir(output_directory):
        least_squares_serial_solve_jax(problem, max_steps=max_steps)
    solution = np.asarray(problem.x, dtype=np.float64)
    objective = float(np.asarray(problem.objective()))
    is_correct = bool(
        np.allclose(solution, np.asarray(TARGET), rtol=1.0e-10, atol=1.0e-10)
        and objective <= 1.0e-16
    )
    return ExampleResult(
        solution=tuple(float(value) for value in solution),
        objective=objective,
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
