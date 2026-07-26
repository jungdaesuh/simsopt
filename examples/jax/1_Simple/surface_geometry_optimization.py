"""Fit an axisymmetric torus to analytic area and volume targets.

Host boundary: a native ``SurfaceRZFourier`` owns coefficients and receives
accepted optimizer states. ``AreaJAX`` and ``VolumeJAX`` own metric evaluation
and derivatives.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from scipy.optimize import least_squares
from simsopt.geo import SurfaceRZFourier
from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision
from simsopt_jax_adapters.geo.surface_objectives import AreaJAX, VolumeJAX

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


def _solve(max_steps: int) -> ExampleResult:
    surface = _build_surface()
    area_objective = AreaJAX(surface)
    volume_objective = VolumeJAX(surface)
    area_oracle = 4.0 * np.pi**2 * MAJOR_RADIUS * MINOR_RADIUS
    volume_oracle = 2.0 * np.pi**2 * MAJOR_RADIUS * MINOR_RADIUS**2

    def residuals(parameters: np.ndarray) -> np.ndarray:
        surface.x = parameters
        return np.asarray(
            (
                float(area_objective.J()) - area_oracle,
                float(volume_objective.J()) - volume_oracle,
            ),
            dtype=np.float64,
        )

    def jacobian(parameters: np.ndarray) -> np.ndarray:
        surface.x = parameters
        return np.stack(
            (
                np.asarray(area_objective.dJ(), dtype=np.float64),
                np.asarray(volume_objective.dJ(), dtype=np.float64),
            )
        )

    result = least_squares(
        residuals,
        np.asarray(surface.x, dtype=np.float64),
        jac=jacobian,
        max_nfev=max_steps,
        xtol=1.0e-12,
        ftol=1.0e-12,
        gtol=1.0e-12,
    )
    surface.x = result.x
    area = float(area_objective.J())
    volume = float(volume_objective.J())
    residual_norm = float(np.linalg.norm(residuals(result.x)))
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
    return parser


def main(arguments: list[str] | None = None) -> int:
    options = _parser().parse_args(arguments)
    result = _solve(options.max_steps or (24 if options.smoke else 96))
    if options.json:
        print(json.dumps(result.json_object(), sort_keys=True))
    else:
        print(f"area={result.area:.12f}")
        print(f"volume={result.volume:.12f}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
