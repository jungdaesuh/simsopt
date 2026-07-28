"""Strict-transfer coverage for the exact wireframe RCLS workflow."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    _build_geometry,
    _scale_configuration,
)
from simsopt.solve.wireframe_optimization import bnorm_obj_matrices
from simsopt_jax.backend.runtime import get_runtime_jax_device
from simsopt_jax.examples import solve_wireframe_rcls

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_wireframe_rcls_example_has_one_batched_numerical_host_boundary() -> None:
    source = (
        REPOSITORY_ROOT / "examples/jax/2_Intermediate/wireframe_rcls_basic.py"
    ).read_text()

    assert source.count("jax.device_get(") == 1


def test_wireframe_rcls_and_field_postprocessing_stay_on_device() -> None:
    configuration = _scale_configuration("bounded")
    plasma_surface, wireframe = _build_geometry(configuration)
    response, target = bnorm_obj_matrices(
        wireframe,
        plasma_surface,
        area_weighted=True,
        verbose=False,
    )
    constraint, constraint_target = wireframe.constraint_matrices(
        assume_no_crossings=False,
        remove_constrained_segments=True,
    )
    free_segments = np.asarray(wireframe.unconstrained_segments(), dtype=np.int64)
    constraint_array = np.asarray(constraint, dtype=np.float64)
    constraint_target_array = np.asarray(constraint_target, dtype=np.float64).reshape(
        (-1, 1)
    )
    free_currents = constraint_array.T @ np.linalg.solve(
        constraint_array @ constraint_array.T,
        constraint_target_array,
    )
    initial_currents = np.zeros((wireframe.n_segments, 1), dtype=np.float64)
    initial_currents[free_segments] = free_currents
    normal = np.asarray(plasma_surface.normal(), dtype=np.float64)
    device = get_runtime_jax_device()
    response_device = jax.device_put(response, device)
    target_device = jax.device_put(target, device)
    initial_currents_device = jax.device_put(initial_currents, device)
    plasma_points_device = jax.device_put(
        np.asarray(plasma_surface.gamma(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    plasma_unit_normal_device = jax.device_put(
        np.asarray(plasma_surface.unitnormal(), dtype=np.float64).reshape((-1, 3)),
        device,
    )
    plasma_area_weights_device = jax.device_put(
        np.linalg.norm(normal, axis=2).reshape(-1) / normal.shape[0] / normal.shape[1],
        device,
    )
    wireframe_nodes_device = jax.device_put(np.stack(wireframe.nodes), device)
    wireframe_segments_device = jax.device_put(
        np.asarray(wireframe.segments, dtype=np.int32),
        device,
    )
    wireframe_segment_signs_device = jax.device_put(
        np.asarray(wireframe.seg_signs, dtype=np.float64),
        device,
    )

    with jax.transfer_guard("disallow"):
        result = solve_wireframe_rcls(
            wireframe=wireframe,
            response=response_device,
            target=target_device,
            regularization=1.0e-10,
            initial_currents=initial_currents_device,
            plasma_points=plasma_points_device,
            plasma_unit_normal=plasma_unit_normal_device,
            plasma_area_weights=plasma_area_weights_device,
            wireframe_nodes=wireframe_nodes_device,
            wireframe_segments=wireframe_segments_device,
            wireframe_segment_signs=wireframe_segment_signs_device,
            assume_no_crossings=False,
        )

    initial_objective, final_objective, constraint_residual, mean_relative = (
        jax.device_get(
            (
                result.initial.normal_objective,
                result.final.normal_objective,
                result.final.constraint_residual,
                result.mean_relative_normal_field,
            )
        )
    )
    assert float(final_objective) < float(initial_objective)
    assert np.linalg.norm(constraint_residual, ord=np.inf) <= 1.0e-3
    assert float(mean_relative) < 0.04
