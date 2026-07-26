"""Minimize Fourier deformations of a circle with ``CurveLengthJAX``.

Host boundary: a native ``CurveXYZFourier`` supplies one immutable spec and
receives only the completed optimizer state.
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
import jax.numpy as jnp
import numpy as np
from simsopt.geo.curvexyzfourier import CurveXYZFourier
from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision
from simsopt_jax.core.curve_geometry import (
    curve_incremental_arclength_from_spec,
    curve_spec_with_dofs,
)
from simsopt_jax.solve.serial import TraceableScalarProblem, serial_solve_jax
from simsopt_jax_adapters.geo.curve_objectives import CurveLengthJAX, curve_length_pure
from simsopt_jax_adapters.geo.curve_specs import curve_spec_from_adapter_curve

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


def _solve(output_directory: Path, max_steps: int) -> ExampleResult:
    curve, objective = _build_problem()
    curve_spec = jax.tree.map(
        lambda leaf: (
            np.asarray(jax.device_get(leaf)) if isinstance(leaf, jax.Array) else leaf
        ),
        curve_spec_from_adapter_curve(curve),
    )
    full_dofs_host = np.asarray(curve.local_full_x, dtype=np.float64)
    free_positions = np.flatnonzero(curve.local_dofs_free_status)
    fixed_full_dofs_host = full_dofs_host.copy()
    fixed_full_dofs_host[free_positions] = 0.0
    expansion_host = np.zeros((full_dofs_host.size, free_positions.size))
    expansion_host[free_positions, np.arange(free_positions.size)] = 1.0
    fixed_full_dofs = fixed_full_dofs_host
    expansion = expansion_host

    def value(parameters: jax.Array) -> jax.Array:
        current_full_dofs = fixed_full_dofs + expansion @ parameters
        current_spec = curve_spec_with_dofs(curve_spec, current_full_dofs)
        return curve_length_pure(curve_incremental_arclength_from_spec(current_spec))

    initial = jax.device_put(np.asarray(curve.x, dtype=np.float64))
    direction = jax.device_put(np.asarray((0.3, -0.4, 0.5), dtype=np.float64))
    direction /= jnp.linalg.norm(direction)

    @jax.jit
    def directional_derivative_error(
        parameters: jax.Array,
        tangent: jax.Array,
    ) -> jax.Array:
        epsilon = 1.0e-6
        analytic_directional = jnp.dot(jax.grad(value)(parameters), tangent)
        finite_difference = (
            value(parameters + epsilon * tangent)
            - value(parameters - epsilon * tangent)
        ) / (2.0 * epsilon)
        return jnp.abs(analytic_directional - finite_difference)

    gradient_fd_error = float(
        jax.device_get(directional_derivative_error(initial, direction))
    )

    problem = TraceableScalarProblem(objective_fn=value, x=initial)
    with chdir(output_directory):
        result = serial_solve_jax(
            problem,
            max_steps=max_steps,
            rtol=1.0e-10,
            atol=1.0e-8,
        )
    curve.x = np.asarray(jax.device_get(problem.x), dtype=np.float64)
    final_length = float(jax.device_get(objective.J()))
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
        print(f"final length={result.final_length:.12f}")
        print(f"circle oracle={result.circle_oracle:.12f}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
