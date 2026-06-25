"""Item 29 parity tests for ``simsopt_jax.core.wireframe``.

Oracle: the existing C++-backed ``simsopt.field.WireframeField`` path through
``simsoptpp/wireframe_field_impl.h`` and
``simsoptpp/magneticfield_wireframe.cpp``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field import WireframeField
from simsopt.geo import (
    CircularPort,
    SurfaceRZFourier,
    ToroidalWireframe,
    windowpane_wireframe,
)
from simsopt_jax.core.wireframe import (
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


def _constraint_wireframe() -> ToroidalWireframe:
    return ToroidalWireframe(_surf_torus(nfp=3, rmaj=2.0, rmin=1.0), 4, 4)


def _assert_wireframe_state_is_jax_kernel_compatible(
    wireframe: ToroidalWireframe,
) -> None:
    points = np.asarray(
        [
            [1.25, 0.10, -0.18],
            [1.70, 0.35, 0.22],
            [2.15, -0.20, 0.05],
        ],
        dtype=np.float64,
    )
    nodes, segments, seg_signs, currents = _wireframe_arrays(wireframe)
    B = wireframe_B(points, nodes, segments, seg_signs, currents)
    assert np.asarray(B).shape == points.shape


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


def test_toroidal_wireframe_constraints_remain_jax_kernel_compatible():
    """Mirror legacy constraint mutations and keep the JAX input contract live."""
    wireframe = _constraint_wireframe()
    test_current = 1.0e6

    assert len(wireframe.constraints) == wireframe.n_phi * wireframe.n_theta - 2

    wireframe.currents[0] = test_current
    assert not wireframe.check_constraints()
    wireframe.currents[0] = 0.0
    assert wireframe.check_constraints()

    wireframe.add_segment_constraints(1)
    with pytest.raises(ValueError):
        wireframe.add_segment_constraints(1)
    wireframe.remove_segment_constraints(1)
    with pytest.raises(ValueError):
        wireframe.remove_segment_constraints(1)

    wireframe.set_segments_constrained([0, 4, 18])
    constrained_segments = wireframe.constrained_segments()
    assert len(constrained_segments) == 4
    for segment in [0, 4, 18, 21]:
        assert segment in constrained_segments
        if segment == 21:
            assert f"implicit_segment_{segment}" in wireframe.constraints
        else:
            assert f"segment_{segment}" in wireframe.constraints

    wireframe.set_segments_free([4])
    constrained_segments = wireframe.constrained_segments()
    assert len(constrained_segments) == 2
    for segment in [0, 18]:
        assert segment in constrained_segments
        assert f"segment_{segment}" in wireframe.constraints

    wireframe.currents[18:22] = test_current
    assert not wireframe.check_constraints()
    wireframe.free_all_segments()
    assert wireframe.check_constraints()
    assert len(wireframe.constrained_segments()) == 0
    wireframe.currents[:] = 0.0

    wireframe.add_poloidal_current_constraint(test_current)
    assert not wireframe.check_constraints()
    tf_count = wireframe.n_phi // 2
    with pytest.raises(ValueError):
        wireframe.add_tfcoil_currents(
            tf_count,
            2.0 * test_current * (1.0 / (2.0 * tf_count * wireframe.nfp)),
        )
    wireframe.add_tfcoil_currents(
        tf_count,
        test_current * (1.0 / (2.0 * tf_count * wireframe.nfp)),
    )
    assert wireframe.check_constraints()
    wireframe.remove_poloidal_current_constraint()
    wireframe.currents[:] = 0.0
    assert wireframe.check_constraints()

    wireframe.add_toroidal_current_constraint(test_current)
    wireframe.currents[1 : wireframe.n_tor_segments : wireframe.n_theta] = (
        test_current
    )
    assert not wireframe.check_constraints()
    wireframe.currents[3 : wireframe.n_tor_segments : wireframe.n_theta] = (
        test_current
    )
    assert not wireframe.check_constraints()
    wireframe.currents[1 : wireframe.n_tor_segments : wireframe.n_theta] = (
        0.5 * test_current
    )
    wireframe.currents[3 : wireframe.n_tor_segments : wireframe.n_theta] = (
        0.5 * test_current
    )
    assert wireframe.check_constraints()
    wireframe.remove_toroidal_current_constraint()
    wireframe.currents[:] = 0.0
    assert wireframe.check_constraints()

    with pytest.raises(ValueError):
        wireframe.set_toroidal_breaks(2, 1)
    with pytest.raises(ValueError):
        wireframe.set_toroidal_breaks(1, 3)

    wireframe.set_toroidal_breaks(1, 1)
    assert set(wireframe.constrained_segments()) == {4, 5, 6, 7}
    wireframe.free_all_segments()

    wireframe.set_toroidal_breaks(1, 2)
    assert set(wireframe.constrained_segments()) == set(range(4, 12)) | set(
        range(22, 26)
    )
    wireframe.free_all_segments()

    wireframe.set_toroidal_breaks(1, 2, allow_pol_current=True)
    assert set(wireframe.constrained_segments()) == {4, 5, 6, 7, 8, 9, 10, 11}
    wireframe.free_all_segments()

    wireframe.set_toroidal_current(test_current)
    with pytest.raises(ValueError):
        wireframe.set_toroidal_breaks(1, 2)
    wireframe.remove_toroidal_current_constraint()

    wireframe.add_tfcoil_currents(1, test_current)
    wireframe.set_toroidal_breaks(1, 1, allow_pol_current=False)
    assert wireframe.check_constraints()
    wireframe.free_all_segments()

    wireframe.set_toroidal_breaks(1, 2, allow_pol_current=False)
    assert not wireframe.check_constraints()
    wireframe.free_all_segments()

    wireframe.set_toroidal_breaks(1, 2, allow_pol_current=True)
    assert wireframe.check_constraints()
    wireframe.currents[:] = 0.0
    wireframe.free_all_segments()

    wireframe.set_toroidal_breaks(1, 2)
    free_cells = wireframe.get_free_cells(form="indices")
    free_cells_mask = wireframe.get_free_cells()
    expected_free_cells = {0, 1, 2, 3, 12, 13, 14, 15}
    assert len(free_cells) == 8
    for cell_index in range(16):
        expected_free = cell_index in expected_free_cells
        assert (cell_index in free_cells) == expected_free
        assert bool(free_cells_mask[cell_index]) == expected_free
    wireframe.free_all_segments()

    wireframe.set_segments_constrained(np.arange(wireframe.n_tor_segments))
    assert not np.any(wireframe.get_free_cells())
    wireframe.free_all_segments()
    _assert_wireframe_state_is_jax_kernel_compatible(wireframe)


def test_toroidal_wireframe_constraint_matrices_match_legacy_shapes():
    """Constraint matrices keep the legacy layout used by JAX RCLS wrappers."""
    wireframe = _constraint_wireframe()

    base_matrix, base_rhs = wireframe.constraint_matrices()
    assert np.max(np.abs(base_rhs)) == 0.0
    assert base_matrix.shape == (wireframe.n_phi * wireframe.n_theta - 2, 32)
    assert base_rhs.shape == (base_matrix.shape[0], 1)

    full_matrix, full_rhs = wireframe.constraint_matrices(
        remove_redundancies=False
    )
    np.testing.assert_array_equal(full_matrix, base_matrix)
    np.testing.assert_array_equal(full_rhs, base_rhs)

    reduced_matrix, reduced_rhs = wireframe.constraint_matrices(
        remove_redundancies=True
    )
    np.testing.assert_array_equal(reduced_matrix, base_matrix)
    np.testing.assert_array_equal(reduced_rhs, base_rhs)

    with pytest.raises(RuntimeError):
        wireframe.constraint_matrices(assume_no_crossings=True)

    wireframe.set_segments_constrained(np.arange(wireframe.n_tor_segments))
    wireframe.set_segments_constrained([16, 17, 22, 23, 24, 25, 30, 31])
    constrained_count = len(wireframe.constrained_segments())

    no_crossing_matrix, no_crossing_rhs = wireframe.constraint_matrices(
        assume_no_crossings=True
    )
    assert np.max(np.abs(no_crossing_rhs)) == 0.0
    assert no_crossing_matrix.shape == (constrained_count + 6, 32)
    assert no_crossing_rhs.shape == (no_crossing_matrix.shape[0], 1)

    free_matrix, free_rhs = wireframe.constraint_matrices(
        assume_no_crossings=True,
        remove_constrained_segments=True,
    )
    assert free_matrix.shape == (6, 8)
    assert free_rhs.shape == (6, 1)

    default_matrix, default_rhs = wireframe.constraint_matrices()
    assert np.max(np.abs(default_rhs)) == 0.0
    assert default_matrix.shape == (constrained_count + 8, 32)

    redundant_matrix, redundant_rhs = wireframe.constraint_matrices(
        remove_redundancies=False
    )
    assert np.max(np.abs(redundant_rhs)) == 0.0
    assert redundant_matrix.shape == (
        wireframe.n_phi * wireframe.n_theta - 2 + constrained_count,
        32,
    )
    assert redundant_rhs.shape == (redundant_matrix.shape[0], 1)
    _assert_wireframe_state_is_jax_kernel_compatible(wireframe)


def test_toroidal_wireframe_collision_constraints_match_legacy_segments():
    """Collision-derived constraints preserve the legacy constrained segments."""
    wireframe = _constraint_wireframe()

    port_near_node = CircularPort(
        ox=2.0,
        oy=0.0,
        oz=0.0,
        ax=0.0,
        ay=0.0,
        az=1.0,
        ir=0.001,
        thick=0.001,
        l0=0.0,
        l1=0.999,
    )
    wireframe.constrain_colliding_segments(port_near_node.collides)
    assert len(wireframe.constrained_segments()) == 0

    wireframe.constrain_colliding_segments(port_near_node.collides, gap=0.01)
    assert set(wireframe.constrained_segments()) == {1, 3, 16, 17}
    wireframe.free_all_segments()

    enclosing_port = CircularPort(
        ox=0.0,
        oy=0.0,
        oz=0.0,
        ax=0.0,
        ay=0.0,
        az=1.0,
        ir=3.5,
        thick=0.0,
        l0=-1.5,
        l1=1.5,
    )
    wireframe.constrain_colliding_segments(enclosing_port.collides)
    assert len(wireframe.unconstrained_segments()) == 0
    _assert_wireframe_state_is_jax_kernel_compatible(wireframe)


def test_windowpane_wireframe_unconstrained_segments_are_jax_kernel_compatible():
    """Windowpane construction keeps the legacy free-segment count and JAX shape."""
    wireframe = windowpane_wireframe(
        _surf_torus(nfp=3, rmaj=2.0, rmin=1.0),
        4,
        6,
        2,
        2,
        2,
        2,
    )

    unconstrained_segments = wireframe.unconstrained_segments()
    expected_count = 2 * (2 + 2) * 4 * 6
    assert len(unconstrained_segments) == expected_count
    _assert_wireframe_state_is_jax_kernel_compatible(wireframe)
