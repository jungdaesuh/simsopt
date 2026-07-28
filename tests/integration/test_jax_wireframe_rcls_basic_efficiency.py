"""Efficiency contracts for the exact wireframe RCLS example workflow."""

from __future__ import annotations

import jax
import numpy as np
from examples.jax.parity.cases.native_wireframe_rcls_basic import (
    _build_geometry,
    _minimum_norm_feasible_currents,
    _scale_configuration,
)
from simsopt.solve.wireframe_optimization import bnorm_obj_matrices
from simsopt_jax.examples import solve_wireframe_rcls


class _CountingWireframe:
    """Expose the required wireframe API while counting constraint snapshots."""

    def __init__(self, wireframe) -> None:
        self._wireframe = wireframe
        self.constraint_matrix_calls = 0

    @property
    def n_segments(self) -> int:
        return int(self._wireframe.n_segments)

    def constraint_matrices(
        self,
        *,
        assume_no_crossings: bool,
        remove_constrained_segments: bool,
    ):
        self.constraint_matrix_calls += 1
        return self._wireframe.constraint_matrices(
            assume_no_crossings=assume_no_crossings,
            remove_constrained_segments=remove_constrained_segments,
        )

    def unconstrained_segments(self):
        return self._wireframe.unconstrained_segments()


def test_wireframe_rcls_uses_one_immutable_constraint_snapshot() -> None:
    configuration = _scale_configuration("bounded")
    plasma_surface, native_wireframe = _build_geometry(configuration)
    response, target = bnorm_obj_matrices(
        native_wireframe,
        plasma_surface,
        area_weighted=True,
        verbose=False,
    )
    initial_currents, _, _, _ = _minimum_norm_feasible_currents(native_wireframe)
    normal = np.asarray(plasma_surface.normal(), dtype=np.float64)
    wireframe = _CountingWireframe(native_wireframe)

    result = solve_wireframe_rcls(
        wireframe=wireframe,
        response=response,
        target=target,
        regularization=1.0e-10,
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
        wireframe_nodes=np.stack(native_wireframe.nodes),
        wireframe_segments=np.asarray(native_wireframe.segments, dtype=np.int32),
        wireframe_segment_signs=np.asarray(
            native_wireframe.seg_signs,
            dtype=np.float64,
        ),
        assume_no_crossings=False,
    )
    jax.block_until_ready(result)

    assert wireframe.constraint_matrix_calls == 1
