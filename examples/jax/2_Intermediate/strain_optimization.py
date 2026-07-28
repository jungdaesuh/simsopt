"""JAX port of ``examples/2_Intermediate/strain_optimization.py``.

The host loads and scales the same HSX coil, then freezes its quadrature
geometry.  Fourier tape-frame rotation, centroid-frame strains, their
gradients, and the complete optimization execute on the selected JAX device.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from simsopt.configs import get_data
from simsopt.geo import CurveXYZFourier
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import (
    ExampleResult,
    run_example,
    solve_strain_rotation,
)

EXAMPLE_ID = "native-strain-optimization"
ROTATION_ORDER = 10
OBJECTIVE_WIDTH = 1.0e-3
REPORTING_WIDTH = 3.0e-3
TORSIONAL_THRESHOLD = 2.0e-3
CURVATURE_THRESHOLD = 2.0e-3


def _fixed_hsx_geometry() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    base_curves, _currents, _axis, _nfp, _field = get_data(
        "hsx",
        coil_order=10,
        points_per_period=10,
    )
    source_curve = base_curves[1]
    curve = CurveXYZFourier(source_curve.quadpoints, source_curve.order)
    curve.x = np.asarray(source_curve.x, dtype=np.float64) * 0.1
    return (
        np.asarray(curve.quadpoints, dtype=np.float64),
        np.asarray(curve.gamma(), dtype=np.float64),
        np.asarray(curve.gammadash(), dtype=np.float64),
        np.asarray(curve.gammadashdash(), dtype=np.float64),
    )


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    geometry = _fixed_hsx_geometry()
    device = get_runtime_jax_device()
    device_result = solve_strain_rotation(
        quadpoints=jax.device_put(geometry[0], device),
        gamma=jax.device_put(geometry[1], device),
        gammadash=jax.device_put(geometry[2], device),
        gammadashdash=jax.device_put(geometry[3], device),
        initial_parameters=jax.device_put(
            np.zeros(2 * ROTATION_ORDER + 1, dtype=np.float64),
            device,
        ),
        rotation_order=ROTATION_ORDER,
        objective_width=OBJECTIVE_WIDTH,
        reporting_width=REPORTING_WIDTH,
        torsional_threshold=TORSIONAL_THRESHOLD,
        curvature_threshold=CURVATURE_THRESHOLD,
        maxiter=max_steps,
        maxfun=15000,
        gtol=1.0e-20,
        ftol=1.0e-20,
        maxcor=10,
        maxls=20,
    )
    result = jax.device_get(device_result)
    maximum_strain = max(
        float(result.final.maximum_torsional_strain),
        float(result.final.maximum_binormal_curvature_strain),
    )
    scientific_success = bool(
        np.isfinite(result.final.objective)
        and result.final.objective < result.initial.objective
        and np.linalg.norm(result.final.gradient, ord=np.inf) <= 1.0e-8
        and np.isfinite(maximum_strain)
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_parameters": tuple(
                float(value) for value in result.initial.parameters
            ),
            "initial_objective": float(result.initial.objective),
            "initial_gradient": tuple(
                float(value) for value in result.initial.gradient
            ),
            "solution": tuple(float(value) for value in result.final.parameters),
            "final_objective": float(result.final.objective),
            "final_gradient": tuple(
                float(value) for value in result.final.gradient
            ),
            "maximum_strain": maximum_strain,
            "solver_success": bool(result.success),
            "solver_driver": "simsopt_lbfgsb",
            "solver_status": int(result.status),
            "iterations": int(result.iterations),
            "function_evaluations": int(result.function_evaluations),
            "gradient_evaluations": int(result.gradient_evaluations),
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-strain-optimization-",
        bounded_steps=50,
        native_default_steps=400,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
