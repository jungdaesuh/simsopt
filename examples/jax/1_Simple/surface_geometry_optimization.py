"""Fit an axisymmetric torus to analytic area and volume targets.

Host boundary: a native ``SurfaceRZFourier`` supplies one immutable spec and
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
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_spec_from_dofs
from simsopt_jax.solve.serial import (
    TraceableLeastSquaresProblem,
    least_squares_serial_solve_jax,
)
from simsopt_jax_adapters.geo.surface_objectives import (
    AreaJAX,
    VolumeJAX,
    surface_area_jax_from_dofs,
    surface_volume_jax_from_dofs,
)

EXAMPLE_ID = "surface-geometry-optimization"
MAJOR_RADIUS = 1.0
MINOR_RADIUS = 0.2


@dataclass(frozen=True)
class ExampleResult:
    area: float
    volume: float
    area_oracle: float
    volume_oracle: float
    residual_norm: float
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
                "area": self.area,
                "volume": self.volume,
                "area_oracle": self.area_oracle,
                "volume_oracle": self.volume_oracle,
                "residual_norm": self.residual_norm,
                "optimizer_success": self.optimizer_success,
            },
        }


def _build_surface() -> SurfaceRZFourier:
    quadrature = np.linspace(0.0, 1.0, 32, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=0,
        nfp=1,
        stellsym=True,
        quadpoints_phi=quadrature,
        quadpoints_theta=quadrature,
    )
    surface.set_rc(0, 0, MAJOR_RADIUS)
    surface.set_rc(1, 0, 0.15)
    surface.set_zs(1, 0, 0.25)
    surface.fix_all()
    surface.unfix("rc(1,0)")
    surface.unfix("zs(1,0)")
    return surface


def _solve(output_directory: Path, max_steps: int) -> ExampleResult:
    surface = _build_surface()
    area_objective = AreaJAX(surface)
    volume_objective = VolumeJAX(surface)
    area_oracle = 4.0 * np.pi**2 * MAJOR_RADIUS * MINOR_RADIUS
    volume_oracle = 2.0 * np.pi**2 * MAJOR_RADIUS * MINOR_RADIUS**2
    area_target = np.asarray(area_oracle, dtype=np.float64)
    volume_target = np.asarray(volume_oracle, dtype=np.float64)
    full_dofs_host = np.asarray(surface.local_full_x, dtype=np.float64)
    free_positions = np.flatnonzero(surface.local_dofs_free_status)
    fixed_full_dofs_host = full_dofs_host.copy()
    fixed_full_dofs_host[free_positions] = 0.0
    expansion_host = np.zeros((full_dofs_host.size, free_positions.size))
    expansion_host[free_positions, np.arange(free_positions.size)] = 1.0
    full_dofs = jax.device_put(full_dofs_host)
    fixed_full_dofs = fixed_full_dofs_host
    expansion = expansion_host
    surface_spec = jax.tree.map(
        lambda leaf: (
            np.asarray(jax.device_get(leaf)) if isinstance(leaf, jax.Array) else leaf
        ),
        surface_rz_fourier_spec_from_dofs(
            full_dofs,
            quadpoints_phi=jax.device_put(np.asarray(surface.quadpoints_phi)),
            quadpoints_theta=jax.device_put(np.asarray(surface.quadpoints_theta)),
            mpol=surface.mpol,
            ntor=surface.ntor,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
        ),
    )

    def residuals(parameters: jax.Array) -> jax.Array:
        current_full_dofs = fixed_full_dofs + expansion @ parameters
        return jnp.stack(
            (
                surface_area_jax_from_dofs(surface_spec, current_full_dofs)
                - area_target,
                surface_volume_jax_from_dofs(surface_spec, current_full_dofs)
                - volume_target,
            )
        )

    @jax.jit
    def residual_norm_fn(parameters: jax.Array) -> jax.Array:
        return jnp.linalg.norm(residuals(parameters))

    problem = TraceableLeastSquaresProblem(
        residual_fn=residuals,
        x=jax.device_put(np.asarray(surface.x, dtype=np.float64)),
    )
    with chdir(output_directory):
        result = least_squares_serial_solve_jax(
            problem,
            max_steps=max_steps,
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    surface.x = np.asarray(jax.device_get(problem.x), dtype=np.float64)
    area = float(jax.device_get(area_objective.J()))
    volume = float(jax.device_get(volume_objective.J()))
    residual_norm = float(jax.device_get(residual_norm_fn(problem.x)))
    is_correct = bool(
        result.success
        and np.isclose(area, area_oracle, rtol=1.0e-9, atol=1.0e-10)
        and np.isclose(volume, volume_oracle, rtol=1.0e-9, atol=1.0e-10)
        and residual_norm <= 1.0e-9
    )
    return ExampleResult(
        area=area,
        volume=volume,
        area_oracle=area_oracle,
        volume_oracle=volume_oracle,
        residual_norm=residual_norm,
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
    max_steps = options.max_steps or (24 if options.smoke else 96)
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
        print(f"area={result.area:.12f}")
        print(f"volume={result.volume:.12f}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
