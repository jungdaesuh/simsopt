"""JAX mirror of ``examples/1_Simple/minimize_curve_length.py``.

Host construction creates the native ``CurveRZFourier`` and fixes its major
radius.  An immutable RZ-Fourier specification and free-DOF expansion then
enter the JAX numerical region, which owns every published number: the native
curve is a construction input and is never written back to.
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
from simsopt.geo import CurveRZFourier
from simsopt_jax.examples import (
    ExecutionScale,
    example_runtime_metadata,
)
from simsopt_jax.examples.official_tiny_least_squares import (
    CONTROLLED_CURVE_INITIAL_FULL,
    DRIVER_CURVE,
    curve_length_residual,
    guard_finite_endpoint,
    solve_jax_residual,
    trf_outcome,
    value_and_jacobian,
)

EXAMPLE_ID = "native-minimize-curve-length"
NQUADRATURE = 100
NFOURIER = 4
NFP = 5
MAJOR_RADIUS = 3.0


@dataclass(frozen=True)
class ExampleResult:
    initial_parameters: tuple[float, ...]
    initial_length: float
    initial_gradient: tuple[float, ...]
    solution: tuple[float, ...]
    final_length: float
    final_gradient: tuple[float, ...]
    circle_oracle: float
    solver_success: bool
    optimizer_success: bool
    optimizer_status: int
    solver_driver: str
    iterations: int | None
    function_evaluations: int
    gradient_evaluations: int
    status: Literal["ok", "failed"]

    def json_object(self, scale: ExecutionScale) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            **example_runtime_metadata(scale),
            "status": self.status,
            "observables": {
                "initial_parameters": self.initial_parameters,
                "initial_length": self.initial_length,
                "initial_gradient": self.initial_gradient,
                "solution": self.solution,
                "final_length": self.final_length,
                "final_gradient": self.final_gradient,
                "circle_oracle": self.circle_oracle,
                "solver_success": self.solver_success,
                "optimizer_success": self.optimizer_success,
                "optimizer_status": self.optimizer_status,
                "solver_driver": self.solver_driver,
                "iterations": self.iterations,
                "function_evaluations": self.function_evaluations,
                "gradient_evaluations": self.gradient_evaluations,
            },
        }


def _build_curve() -> CurveRZFourier:
    curve = CurveRZFourier(NQUADRATURE, NFOURIER, NFP, True)
    curve.x = np.asarray(CONTROLLED_CURVE_INITIAL_FULL, dtype=np.float64)
    curve.fix(0)
    return curve


def solve(output_directory: Path, max_steps: int | None) -> ExampleResult:
    curve = _build_curve()
    full_dofs = np.asarray(curve.local_full_x, dtype=np.float64)
    free_positions = np.flatnonzero(curve.local_dofs_free_status)
    residual = curve_length_residual(
        full_dofs,
        np.asarray(curve.quadpoints, dtype=np.float64),
        free_positions,
        order=curve.order,
        nfp=curve.nfp,
        stellsym=curve.stellsym,
    )
    initial = full_dofs[free_positions]
    initial_value, initial_jacobian = value_and_jacobian(residual, initial)
    with chdir(output_directory):
        optimizer = solve_jax_residual(residual, initial, max_nfev=max_steps)
    solution = np.asarray(optimizer.x, dtype=np.float64)
    final_value, final_jacobian = value_and_jacobian(residual, solution)
    initial_length = float(initial_value[0])
    final_length = float(final_value[0])
    initial_gradient = initial_jacobian[0]
    final_gradient = final_jacobian[0]
    # ``circle_oracle`` is a published diagnostic, not a gate: the official run
    # stops on ftol 3.25e-07 away from it
    # (A/reference-simple/runs/native-minimize-curve-length/captured-controlled-omp1:
    # status 2 `ftol`, nfev 62, njev 55, fun = 18.849556246039942).
    circle_oracle = 2.0 * np.pi * MAJOR_RADIUS
    outcome = guard_finite_endpoint(
        trf_outcome(optimizer),
        (np.asarray(final_length, dtype=np.float64), solution, final_gradient),
    )
    return ExampleResult(
        initial_parameters=tuple(float(value) for value in initial),
        initial_length=initial_length,
        initial_gradient=tuple(float(value) for value in initial_gradient),
        solution=tuple(float(value) for value in solution),
        final_length=final_length,
        final_gradient=tuple(float(value) for value in final_gradient),
        circle_oracle=circle_oracle,
        solver_success=outcome.success,
        optimizer_success=bool(optimizer.success),
        optimizer_status=int(optimizer.status),
        solver_driver=DRIVER_CURVE,
        iterations=None,
        function_evaluations=outcome.nfev,
        gradient_evaluations=outcome.njev,
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
    # solve runs SciPy's own default budget (100 * n), the official policy. The
    # official run needs 62 evaluations, which the previous smoke cap of 32 cut
    # short, so the mirror reported a truncated solve as its own outcome.
    max_steps = options.max_steps
    if options.output_dir is not None:
        options.output_dir.mkdir(parents=True, exist_ok=True)
        result = solve(options.output_dir, max_steps)
    elif options.smoke:
        with TemporaryDirectory(prefix="simsopt-jax-minimize-curve-length-") as tmp:
            result = solve(Path(tmp), max_steps)
    else:
        result = solve(Path.cwd(), max_steps)
    if options.json:
        scale: ExecutionScale = "bounded" if options.smoke else "native_default"
        print(json.dumps(result.json_object(scale), sort_keys=True))
    else:
        print(f"initial curve length: {result.initial_length:.12e}")
        print(f"final curve length: {result.final_length:.12e}")
        print(f"expected final length: {result.circle_oracle:.12e}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
