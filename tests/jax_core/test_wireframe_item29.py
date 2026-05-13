"""Item 29 parity tests for ``simsopt.jax_core.wireframe``.

Oracle: the existing C++-backed ``simsopt.field.WireframeField`` path through
``simsoptpp/wireframe_field_impl.h`` and
``simsoptpp/magneticfield_wireframe.cpp``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import WireframeField
from simsopt.geo import SurfaceRZFourier, ToroidalWireframe
from simsopt.jax_core.wireframe import (
    wireframe_B,
    wireframe_B_and_dB_by_dX,
    wireframe_dB_by_dX,
    wireframe_segment_B_contributions,
)

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


def _surf_torus(nfp: int, rmaj: float, rmin: float) -> SurfaceRZFourier:
    surface = SurfaceRZFourier(nfp=nfp, mpol=1, ntor=0)
    surface.set_rc(0, 0, rmaj)
    surface.set_rc(1, 0, rmin)
    surface.set_zs(1, 0, rmin)
    return surface


def _wireframe_case() -> tuple[ToroidalWireframe, np.ndarray]:
    wireframe = ToroidalWireframe(_surf_torus(nfp=2, rmaj=2.0, rmin=0.7), 4, 6)
    wireframe.currents[:] = np.linspace(
        -2.0e5,
        3.5e5,
        wireframe.n_segments,
        dtype=np.float64,
    )
    points = np.array(
        [
            [1.25, 0.10, -0.18],
            [1.70, 0.35, 0.22],
            [2.15, -0.20, 0.05],
            [2.55, 0.42, -0.14],
            [1.55, -0.48, 0.31],
            [2.30, 0.18, -0.35],
            [1.88, 0.62, 0.16],
        ],
        dtype=np.float64,
    )
    return wireframe, points


def _wireframe_arrays(
    wireframe: ToroidalWireframe,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.ascontiguousarray(np.stack(wireframe.nodes).astype(np.float64)),
        np.ascontiguousarray(wireframe.segments.astype(np.int32)),
        np.ascontiguousarray(np.asarray(wireframe.seg_signs, dtype=np.float64)),
        np.ascontiguousarray(wireframe.currents.astype(np.float64)),
    )


def test_wireframe_total_B_and_dB_match_cpp_wireframefield():
    """Total field and first spatial derivative match the C++ oracle."""
    wireframe, points = _wireframe_case()
    nodes, segments, seg_signs, currents = _wireframe_arrays(wireframe)

    field = WireframeField(wireframe)
    field.set_points(points)
    B_cpu = np.asarray(field.B(), dtype=np.float64)
    dB_cpu = np.asarray(field.dB_by_dX(), dtype=np.float64)

    B_jax, dB_jax = wireframe_B_and_dB_by_dX(
        points,
        nodes,
        segments,
        seg_signs,
        currents,
    )

    np.testing.assert_allclose(np.asarray(B_jax), B_cpu, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(dB_jax), dB_cpu, rtol=_RTOL, atol=_ATOL)


def test_wireframe_separate_total_kernels_match_combined_kernel():
    """Separate ``B`` / ``dB`` entry points share the combined-kernel contract."""
    wireframe, points = _wireframe_case()
    nodes, segments, seg_signs, currents = _wireframe_arrays(wireframe)

    B_combined, dB_combined = wireframe_B_and_dB_by_dX(
        points,
        nodes,
        segments,
        seg_signs,
        currents,
    )
    B_separate = wireframe_B(points, nodes, segments, seg_signs, currents)
    dB_separate = wireframe_dB_by_dX(points, nodes, segments, seg_signs, currents)

    np.testing.assert_array_equal(np.asarray(B_separate), np.asarray(B_combined))
    np.testing.assert_array_equal(np.asarray(dB_separate), np.asarray(dB_combined))


def test_wireframe_segment_B_contributions_match_cpp_fieldcache():
    """Unit-current segment contributions match ``WireframeField`` cache arrays."""
    wireframe, points = _wireframe_case()
    nodes, segments, seg_signs, _ = _wireframe_arrays(wireframe)

    field = WireframeField(wireframe)
    field.set_points(points)
    segment_B_cpu = np.stack(field.dB_by_dsegmentcurrents(0), axis=0)
    segment_B_jax = wireframe_segment_B_contributions(
        points,
        nodes,
        segments,
        seg_signs,
    )

    np.testing.assert_allclose(
        np.asarray(segment_B_jax),
        segment_B_cpu,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_wireframe_jitted_device_arrays_under_strict_transfer_guard():
    """The pure JAX kernel consumes device-resident arrays without host transfer."""
    wireframe, points = _wireframe_case()
    nodes, segments, seg_signs, currents = _wireframe_arrays(wireframe)
    points_dev = jnp.asarray(points, dtype=jnp.float64)
    nodes_dev = jnp.asarray(nodes, dtype=jnp.float64)
    segments_dev = jnp.asarray(segments, dtype=jnp.int32)
    seg_signs_dev = jnp.asarray(seg_signs, dtype=jnp.float64)
    currents_dev = jnp.asarray(currents, dtype=jnp.float64)
    points_dev.block_until_ready()
    nodes_dev.block_until_ready()
    segments_dev.block_until_ready()
    seg_signs_dev.block_until_ready()
    currents_dev.block_until_ready()

    with jax.transfer_guard("disallow"):
        B, dB = jax.jit(wireframe_B_and_dB_by_dX)(
            points_dev,
            nodes_dev,
            segments_dev,
            seg_signs_dev,
            currents_dev,
        )
        B.block_until_ready()
        dB.block_until_ready()
