"""JAX port of ``examples/2_Intermediate/wireframe_rcls_basic.py``.

The host constructs the same Landreman-Paul QA boundary, offset wireframe, and
poloidal-current constraint as the native example.  The fixed normal-field
response is then handed to the immutable JAX RCLS solve on the selected device.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt_jax.examples import ExampleResult, run_example, solve_wireframe_rcls
from simsopt_jax_adapters.solve.wireframe import (
    bnorm_obj_matrices_jax,
)

EXAMPLE_ID = "native-wireframe-rcls-basic"
TEST_DATA = Path(__file__).resolve().parents[3] / "tests" / "test_files"
REGULARIZATION_WEIGHT = 1.0e-10


def _minimum_norm_feasible_currents(
    wireframe: ToroidalWireframe,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    constraint, target = wireframe.constraint_matrices(
        assume_no_crossings=False,
        remove_constrained_segments=True,
    )
    constraint_array = np.asarray(constraint, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64).reshape((-1, 1))
    free_segments = np.asarray(wireframe.unconstrained_segments(), dtype=np.intp)
    gram = constraint_array @ constraint_array.T
    free_currents = constraint_array.T @ np.linalg.solve(gram, target_array)
    currents = np.zeros((wireframe.n_segments, 1), dtype=np.float64)
    currents[free_segments] = free_currents
    return currents, constraint_array, target_array, free_segments


def solve(_output_directory: Path, resolution: int) -> ExampleResult:
    plasma_surface = SurfaceRZFourier.from_vmec_input(
        str(TEST_DATA / "input.LandremanPaul2021_QA"),
        nphi=4 * resolution,
        ntheta=4 * resolution,
        range="half period",
    )
    wireframe_surface = SurfaceRZFourier.from_vmec_input(
        str(TEST_DATA / "input.LandremanPaul2021_QA")
    )
    wireframe_surface.extend_via_projected_normal(0.3)
    wireframe = ToroidalWireframe(
        wireframe_surface,
        resolution,
        3 * resolution // 2,
    )

    mu0 = 4.0 * np.pi * 1.0e-7
    poloidal_current = -2.0 * np.pi * plasma_surface.get_rc(0, 0) * 1.0 / mu0
    wireframe.set_poloidal_current(poloidal_current)

    response, target = bnorm_obj_matrices_jax(
        wireframe,
        plasma_surface,
        area_weighted=True,
        verbose=False,
    )
    initial_currents, _, constraint_target, _ = _minimum_norm_feasible_currents(
        wireframe
    )
    normal = np.asarray(plasma_surface.normal(), dtype=np.float64)
    device_result = solve_wireframe_rcls(
        wireframe=wireframe,
        response=response,
        target=target,
        regularization=REGULARIZATION_WEIGHT,
        initial_currents=initial_currents,
        plasma_points=np.asarray(plasma_surface.gamma(), dtype=np.float64).reshape(
            (-1, 3)
        ),
        plasma_unit_normal=np.asarray(
            plasma_surface.unitnormal(), dtype=np.float64
        ).reshape((-1, 3)),
        plasma_area_weights=(
            np.linalg.norm(normal, axis=2).reshape(-1)
            / normal.shape[0]
            / normal.shape[1]
        ),
        wireframe_nodes=np.stack(wireframe.nodes),
        wireframe_segments=np.asarray(wireframe.segments, dtype=np.int32),
        wireframe_segment_signs=np.asarray(wireframe.seg_signs, dtype=np.float64),
        assume_no_crossings=False,
    )
    result = jax.device_get(device_result)
    solution = np.asarray(result.final.currents, dtype=np.float64)
    initial_normal_error = float(np.sqrt(2.0 * float(result.initial.normal_objective)))
    final_normal_error = float(np.sqrt(2.0 * float(result.final.normal_objective)))
    regularization_objective = float(result.final.regularization_objective)
    constraint_error = float(result.final.constraint_max_abs)
    maximum_current = float(result.maximum_current)
    constraint_scale = max(1.0, float(np.linalg.norm(constraint_target, ord=np.inf)))
    scientific_success = bool(
        result.finite_currents
        and final_normal_error < initial_normal_error
        and constraint_error <= 1.0e-11 * constraint_scale
        and maximum_current > 0.0
    )
    return ExampleResult(
        example_id=EXAMPLE_ID,
        observables={
            "initial_normal_error": initial_normal_error,
            "final_normal_error": final_normal_error,
            "normal_objective": float(result.final.normal_objective),
            "regularization_objective": regularization_objective,
            "total_objective": float(result.final.total_objective),
            "constraint_residual": constraint_error,
            "mean_relative_normal_field": float(result.mean_relative_normal_field),
            "maximum_current": maximum_current,
            "solver_success": scientific_success,
            "solution": tuple(float(value) for value in solution.ravel()),
        },
        status="ok" if scientific_success else "failed",
    )


def main(arguments: list[str] | None = None) -> int:
    return run_example(
        arguments,
        description=__doc__,
        temporary_prefix="simsopt-jax-wireframe-rcls-basic-",
        bounded_steps=4,
        native_default_steps=8,
        solve=solve,
    )


if __name__ == "__main__":
    raise SystemExit(main())
