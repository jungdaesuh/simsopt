"""Efficiency contracts for the exact wireframe RCLS example workflow."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from pathlib import Path

import jax
import numpy as np
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt.solve.wireframe_optimization import bnorm_obj_matrices
from simsopt_jax.examples import solve_wireframe_rcls

_SURFACE_INPUT = (
    Path(__file__).resolve().parents[1] / "test_files" / "input.LandremanPaul2021_QA"
)


def _build_geometry() -> tuple[SurfaceRZFourier, ToroidalWireframe]:
    """The ``wireframe_rcls_basic`` workflow at its reduced resolution."""
    plasma_surface = SurfaceRZFourier.from_vmec_input(
        str(_SURFACE_INPUT),
        nphi=16,
        ntheta=16,
        range="half period",
    )
    wireframe_surface = SurfaceRZFourier.from_vmec_input(str(_SURFACE_INPUT))
    wireframe_surface.extend_via_projected_normal(0.3)
    wireframe = ToroidalWireframe(wireframe_surface, 4, 6)
    mu0 = 4.0 * np.pi * 1.0e-7
    wireframe.set_poloidal_current(-2.0 * np.pi * plasma_surface.get_rc(0, 0) / mu0)
    return plasma_surface, wireframe


def _minimum_norm_feasible_currents(wireframe: ToroidalWireframe) -> np.ndarray:
    constraint, target = wireframe.constraint_matrices(
        assume_no_crossings=False,
        remove_constrained_segments=True,
    )
    constraint_array = np.asarray(constraint, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64).reshape((-1, 1))
    free_segments = np.asarray(wireframe.unconstrained_segments(), dtype=np.int64)
    gram = constraint_array @ constraint_array.T
    currents = np.zeros((wireframe.n_segments, 1), dtype=np.float64)
    currents[free_segments] = constraint_array.T @ np.linalg.solve(gram, target_array)
    return currents


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
    plasma_surface, native_wireframe = _build_geometry()
    response, target = bnorm_obj_matrices(
        native_wireframe,
        plasma_surface,
        area_weighted=True,
        verbose=False,
    )
    initial_currents = _minimum_norm_feasible_currents(native_wireframe)
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
