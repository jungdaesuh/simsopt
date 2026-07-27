"""JAX port of ``examples/2_Intermediate/strain_optimization.py``.

The host loads and scales the same HSX coil, then freezes its quadrature
geometry.  Fourier tape-frame rotation, centroid-frame strains, their
gradients, and the complete optimization execute on the selected JAX device.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from simsopt.configs import get_data
from simsopt.geo import CurveXYZFourier
from simsopt_jax.core.framedcurve import (
    rotated_centroid_frame,
    rotated_centroid_frame_dash,
    rotation_alpha,
    rotation_alphadash,
)
from simsopt_jax.examples import ExampleResult, run_example
from simsopt_jax.solve import Driver, SimsoptLBFGSBOptions
from simsopt_jax.solve.dispatch import minimize

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


def _strains_from_rotation(
    quadpoints: jax.Array,
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    rotation_dofs: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    alpha = rotation_alpha(rotation_dofs, quadpoints, ROTATION_ORDER)
    alphadash = rotation_alphadash(
        rotation_dofs,
        quadpoints,
        ROTATION_ORDER,
    )
    _tangent, _normal, binormal = rotated_centroid_frame(
        gamma,
        gammadash,
        alpha,
    )
    tangent_dash, normal_dash, _binormal_dash = rotated_centroid_frame_dash(
        gamma,
        gammadash,
        gammadashdash,
        alpha,
        alphadash,
    )
    torsion = jnp.sum(
        (normal_dash / arc_length[:, None]) * binormal,
        axis=1,
    )
    binormal_curvature = jnp.sum(
        (tangent_dash / arc_length[:, None]) * binormal,
        axis=1,
    )
    torsional_strain = torsion**2 * REPORTING_WIDTH**2 / 12.0
    binormal_strain = REPORTING_WIDTH * jnp.abs(binormal_curvature) / 2.0
    return torsional_strain, binormal_strain


def _objective(
    quadpoints: jax.Array,
    gamma: jax.Array,
    gammadash: jax.Array,
    gammadashdash: jax.Array,
    rotation_dofs: jax.Array,
) -> jax.Array:
    reporting_torsion, reporting_binormal = _strains_from_rotation(
        quadpoints,
        gamma,
        gammadash,
        gammadashdash,
        rotation_dofs,
    )
    width_ratio = OBJECTIVE_WIDTH / REPORTING_WIDTH
    objective_torsion = reporting_torsion * width_ratio**2
    objective_binormal = reporting_binormal * width_ratio
    torsional_excess = jnp.maximum(
        objective_torsion - TORSIONAL_THRESHOLD,
        0.0,
    )
    binormal_excess = jnp.maximum(
        objective_binormal - CURVATURE_THRESHOLD,
        0.0,
    )
    arc_length = jnp.linalg.norm(gammadash, axis=1)
    return 0.5 * jnp.mean((torsional_excess**2 + binormal_excess**2) * arc_length)


def solve(_output_directory: Path, max_steps: int) -> ExampleResult:
    geometry = _fixed_hsx_geometry()
    objective = jax.tree_util.Partial(_objective, *geometry)
    strains = jax.jit(jax.tree_util.Partial(_strains_from_rotation, *geometry))
    value_and_gradient = jax.jit(jax.value_and_grad(objective))
    initial_device = jax.device_put(np.zeros(2 * ROTATION_ORDER + 1, dtype=np.float64))
    initial_objective_device, initial_gradient_device = value_and_gradient(
        initial_device
    )
    solver_result = minimize(
        value_and_gradient,
        initial_device,
        driver=Driver.SIMSOPT_LBFGSB,
        options=SimsoptLBFGSBOptions(
            maxiter=max_steps,
            maxfun=4 * max_steps,
            gtol=1.0e-10,
            ftol=1.0e-12,
            maxcor=10,
            maxls=20,
        ),
    )
    solution_device = jax.block_until_ready(jax.device_put(solver_result.x))
    final_objective_device, final_gradient_device = value_and_gradient(solution_device)
    torsional_strain_device, binormal_strain_device = strains(solution_device)

    initial = np.asarray(jax.device_get(initial_device), dtype=np.float64)
    initial_gradient = np.asarray(
        jax.device_get(initial_gradient_device),
        dtype=np.float64,
    )
    solution = np.asarray(jax.device_get(solution_device), dtype=np.float64)
    final_gradient = np.asarray(
        jax.device_get(final_gradient_device),
        dtype=np.float64,
    )
    initial_objective = float(jax.device_get(initial_objective_device))
    final_objective = float(jax.device_get(final_objective_device))
    maximum_strain = float(
        max(
            np.max(np.asarray(jax.device_get(torsional_strain_device))),
            np.max(np.asarray(jax.device_get(binormal_strain_device))),
        )
    )
    scientific_success = bool(
        solver_result.success
        and np.isfinite(final_objective)
        and final_objective < initial_objective
        and np.linalg.norm(final_gradient, ord=np.inf) <= 1.0e-7
        and np.isfinite(maximum_strain)
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_parameters": tuple(float(value) for value in initial),
            "initial_objective": initial_objective,
            "initial_gradient": tuple(float(value) for value in initial_gradient),
            "solution": tuple(float(value) for value in solution),
            "final_objective": final_objective,
            "final_gradient": tuple(float(value) for value in final_gradient),
            "maximum_strain": maximum_strain,
            "solver_success": solver_result.success,
            "solver_driver": solver_result.driver.value,
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
