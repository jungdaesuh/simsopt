"""Solve a two-dipole fixed-state permanent-magnet problem in pure JAX."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from simsopt_jax.backend.runtime import get_backend_mode, get_resolved_precision
from simsopt_jax.geo.permanent_magnet_grid import PermanentMagnetGridJAX
from simsopt_jax.solve.permanent_magnet import GPMO_baseline_jax

EXAMPLE_ID = "permanent-magnet-optimization"


@dataclass(frozen=True)
class ExampleResult:
    moments: tuple[tuple[float, ...], ...]
    residual_norm: float
    selected_dipoles: tuple[int, ...]
    status: Literal["ok", "failed"]

    def json_object(self) -> dict[str, object]:
        return {
            "example_id": EXAMPLE_ID,
            "backend_mode": get_backend_mode(),
            "platform": jax.devices()[0].platform,
            "precision": get_resolved_precision(),
            "status": self.status,
            "observables": {
                "moments": self.moments,
                "residual_norm": self.residual_norm,
                "selected_dipoles": self.selected_dipoles,
            },
        }


def _fixed_grid() -> PermanentMagnetGridJAX:
    response = jax.device_put(np.eye(6, dtype=np.float64))
    target = jax.device_put(
        np.asarray((0.8, 0.0, 0.0, -0.6, 0.0, 0.0), dtype=np.float64)
    )
    zero_moments = jax.device_put(np.zeros((2, 3), dtype=np.float64))
    return PermanentMagnetGridJAX(
        A_obj=response,
        b_obj=target,
        ATb=jnp.reshape(response.T @ target, (2, 3)),
        ATA_scale=jax.device_put(np.asarray(1.0, dtype=np.float64)),
        m0=zero_moments,
        m=zero_moments,
        m_proxy=zero_moments,
        m_maxima=jax.device_put(np.ones(2, dtype=np.float64)),
        dipole_grid_xyz=jax.device_put(
            np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float64)
        ),
        coordinate_flag="cartesian",
        R0=0.0,
        nfp=1,
        stellsym=False,
        nphi=1,
        ntheta=6,
        ndipoles=2,
    )


def _solve() -> ExampleResult:
    grid = _fixed_grid()
    result = GPMO_baseline_jax(grid, K=2)
    moments_array = np.asarray(jax.device_get(result.m), dtype=np.float64)
    residual_norm = float(
        np.linalg.norm(np.asarray(jax.device_get(result.core_result.residual)))
    )
    selected = tuple(
        int(value) for value in np.asarray(jax.device_get(result.selected_dipoles))
    )
    expected = np.asarray(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)))
    is_correct = bool(
        np.array_equal(moments_array, expected)
        and np.isclose(residual_norm, np.sqrt(5.0) / 5.0)
        and selected == (0, 1)
    )
    return ExampleResult(
        moments=tuple(
            tuple(float(component) for component in row) for row in moments_array
        ),
        residual_norm=residual_norm,
        selected_dipoles=selected,
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
        print(json.dumps(result.json_object(), sort_keys=True))
    else:
        print(f"moments={result.moments}")
        print(f"residual norm={result.residual_norm:.6e}")
    return 0 if result.status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
