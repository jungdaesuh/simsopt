"""JAX mirror of ``examples/1_Simple/minimize_curve_length.py``.

Host construction creates the native ``CurveRZFourier`` and fixes its major
radius.  An immutable RZ-Fourier specification and free-DOF expansion then
enter the JAX numerical region.  Only the accepted final free state is
published back to the native curve for reporting.
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
from simsopt.geo import CurveRZFourier
from simsopt_jax.examples import ExecutionScale, example_runtime_metadata
from simsopt_jax.core.curve_geometry import (
    curve_incremental_arclength_from_spec,
    curve_spec_with_dofs,
)
from simsopt_jax.core.specs import make_curve_rzfourier_spec
from simsopt_jax.solve.serial import (
    TraceableScalarProblem,
    serial_solve_jax,
)
from simsopt_jax_adapters.geo.curve_objectives import curve_length_pure

EXAMPLE_ID = "native-minimize-curve-length"
NQUADRATURE = 100
NFOURIER = 4
NFP = 5
RANDOM_SEED = 0
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
    solver_driver: str
    iterations: int
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
                "solver_driver": self.solver_driver,
                "iterations": self.iterations,
                "function_evaluations": self.function_evaluations,
                "gradient_evaluations": self.gradient_evaluations,
            },
        }


def _build_curve() -> CurveRZFourier:
    curve = CurveRZFourier(NQUADRATURE, NFOURIER, NFP, True)
    random_state = np.random.default_rng(RANDOM_SEED)
    initial_full = np.zeros(curve.dof_size, dtype=np.float64)
    initial_full[1] = float(random_state.uniform(0.05, 0.15))
    initial_full[0] = MAJOR_RADIUS
    curve.x = initial_full
    curve.fix("rc(0)")
    return curve


def solve(output_directory: Path, max_steps: int) -> ExampleResult:
    curve = _build_curve()
    full_dofs = np.asarray(curve.local_full_x, dtype=np.float64)
    free_positions = np.flatnonzero(curve.local_dofs_free_status)
    fixed_dofs = full_dofs.copy()
    fixed_dofs[free_positions] = 0.0
    expansion = np.zeros((full_dofs.size, free_positions.size), dtype=np.float64)
    expansion[free_positions, np.arange(free_positions.size)] = 1.0
    spec = make_curve_rzfourier_spec(
        dofs=jax.device_put(full_dofs),
        quadpoints=jax.device_put(np.asarray(curve.quadpoints, dtype=np.float64)),
        order=curve.order,
        nfp=curve.nfp,
        stellsym=curve.stellsym,
    )
    fixed_dofs_device = jax.device_put(fixed_dofs)
    expansion_device = jax.device_put(expansion)

    @jax.jit
    def curve_length(free_dofs: jax.Array) -> jax.Array:
        current_full = fixed_dofs_device + expansion_device @ free_dofs
        current_spec = curve_spec_with_dofs(spec, current_full)
        incremental_arclength = curve_incremental_arclength_from_spec(current_spec)
        return curve_length_pure(incremental_arclength)

    value_and_gradient = jax.jit(jax.value_and_grad(curve_length))

    initial_device = jax.device_put(np.asarray(curve.x, dtype=np.float64))
    initial_length_device, initial_gradient_device = value_and_gradient(initial_device)
    problem = TraceableScalarProblem(objective_fn=curve_length, x=initial_device)
    with chdir(output_directory):
        solver_result = serial_solve_jax(
            problem,
            max_steps=max_steps,
            rtol=1.0e-10,
            atol=1.0e-8,
        )
    final_device = jax.block_until_ready(problem.x)
    final_length_device, final_gradient_device = value_and_gradient(final_device)

    initial = np.asarray(jax.device_get(initial_device), dtype=np.float64)
    initial_gradient = np.asarray(
        jax.device_get(initial_gradient_device), dtype=np.float64
    )
    solution = np.asarray(jax.device_get(final_device), dtype=np.float64)
    final_gradient = np.asarray(jax.device_get(final_gradient_device), dtype=np.float64)
    initial_length = float(jax.device_get(initial_length_device))
    final_length = float(jax.device_get(final_length_device))
    curve.x = solution
    circle_oracle = 2.0 * np.pi * MAJOR_RADIUS
    scientific_success = bool(
        solver_result.success
        and np.isclose(final_length, circle_oracle, rtol=1.0e-9, atol=1.0e-9)
        and np.linalg.norm(final_gradient, ord=np.inf) <= 1.0e-7
    )
    return ExampleResult(
        initial_parameters=tuple(float(value) for value in initial),
        initial_length=initial_length,
        initial_gradient=tuple(float(value) for value in initial_gradient),
        solution=tuple(float(value) for value in solution),
        final_length=final_length,
        final_gradient=tuple(float(value) for value in final_gradient),
        circle_oracle=circle_oracle,
        solver_success=solver_result.success,
        solver_driver=solver_result.driver.value,
        iterations=solver_result.nit,
        function_evaluations=solver_result.nfev,
        gradient_evaluations=solver_result.njev,
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
