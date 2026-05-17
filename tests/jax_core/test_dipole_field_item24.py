"""Item 24 parity tests for ``simsopt.jax_core.dipole_field``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import simsoptpp as sopp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.field.magneticfieldclasses import DipoleField
from simsopt.jax_core.dipole_field import (
    define_a_uniform_cartesian_grid_between_two_toroidal_surfaces,
    dipole_field_A,
    dipole_field_A_from_spec,
    dipole_field_B,
    dipole_field_B_from_spec,
    dipole_field_Bn,
    dipole_field_dA,
    dipole_field_dA_from_spec,
    dipole_field_dB,
    dipole_field_dB_from_spec,
    make_dipole_field_spec,
)
from .jaxpr_utils import count_jaxpr_primitives

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_DIRECT_RTOL = _DIRECT_KERNEL["rtol"]
_DIRECT_ATOL = _DIRECT_KERNEL["atol"]

_POINTS = np.ascontiguousarray(
    np.array(
        [
            [1.15, 0.22, 0.43],
            [-0.72, 0.61, -0.31],
            [0.33, -0.84, 0.58],
            [1.44, -0.18, -0.67],
        ],
        dtype=np.float64,
    )
)
_DIPOLE_POINTS = np.ascontiguousarray(
    np.array(
        [
            [0.13, -0.25, 0.91],
            [-0.42, 0.37, -0.54],
            [0.64, 0.18, 0.27],
            [-0.18, -0.66, 0.72],
        ],
        dtype=np.float64,
    )
)
_DIPOLE_MOMENTS = np.ascontiguousarray(
    np.array(
        [
            [0.51, -0.32, 0.27],
            [-0.14, 0.63, 0.46],
            [0.38, 0.19, -0.58],
            [-0.41, -0.22, 0.36],
        ],
        dtype=np.float64,
    )
)
_UNITNORMAL = np.ascontiguousarray(
    np.array(
        [
            [0.3, -0.4, 0.5],
            [-0.2, 0.7, 0.1],
            [0.6, 0.2, -0.3],
            [-0.5, -0.1, 0.4],
        ],
        dtype=np.float64,
    )
)
_UNITNORMAL = np.ascontiguousarray(
    _UNITNORMAL / np.linalg.norm(_UNITNORMAL, axis=1)[:, None]
)
_BN_TARGET = np.ascontiguousarray(np.linspace(-0.2, 0.3, _POINTS.shape[0]))
_GRID_NORMAL_INNER = np.ascontiguousarray(np.array([[1.0, 0.0, 0.0]]))
_GRID_NORMAL_OUTER = np.ascontiguousarray(np.array([[1.0, 0.0, 0.0]]))
_GRID_XYZ_INNER = np.ascontiguousarray(np.array([[1.0, 0.0, 0.0]]))
_GRID_XYZ_OUTER = np.ascontiguousarray(np.array([[2.0, 0.0, 0.0]]))
_GRID_XYZ_UNIFORM = np.ascontiguousarray(
    np.array(
        [
            [0.5, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
        ]
    )
)


def _assert_direct_kernel_close(actual: np.ndarray, expected: np.ndarray) -> None:
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=_DIRECT_RTOL,
        atol=_DIRECT_ATOL,
    )


def test_direct_cpp_parity_for_field_and_derivative_kernels() -> None:
    """Raw JAX kernels match the exposed ``simsoptpp`` dipole-field oracles."""

    expected_B = np.asarray(
        sopp.dipole_field_B(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS)
    )
    expected_A = np.asarray(
        sopp.dipole_field_A(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS)
    )
    expected_dB = np.asarray(
        sopp.dipole_field_dB(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS)
    )
    expected_dA = np.asarray(
        sopp.dipole_field_dA(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS)
    )

    actual_B = np.asarray(dipole_field_B(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS))
    actual_A = np.asarray(dipole_field_A(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS))
    actual_dB = np.asarray(dipole_field_dB(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS))
    actual_dA = np.asarray(dipole_field_dA(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS))

    assert actual_B.shape == expected_B.shape == (_POINTS.shape[0], 3)
    assert actual_A.shape == expected_A.shape == (_POINTS.shape[0], 3)
    assert actual_dB.shape == expected_dB.shape == (_POINTS.shape[0], 3, 3)
    assert actual_dA.shape == expected_dA.shape == (_POINTS.shape[0], 3, 3)
    _assert_direct_kernel_close(actual_B, expected_B)
    _assert_direct_kernel_close(actual_A, expected_A)
    _assert_direct_kernel_close(actual_dB, expected_dB)
    _assert_direct_kernel_close(actual_dA, expected_dA)


def test_total_field_kernels_stream_over_dipoles() -> None:
    """Total B/A/dB/dA kernels scan over dipoles instead of staging dense pairs."""

    points = jnp.asarray(_POINTS, dtype=jnp.float64)
    dipole_points = jnp.asarray(_DIPOLE_POINTS, dtype=jnp.float64)
    dipole_moments = jnp.asarray(_DIPOLE_MOMENTS, dtype=jnp.float64)

    kernels = (
        dipole_field_B,
        dipole_field_A,
        dipole_field_dB,
        dipole_field_dA,
    )
    for kernel in kernels:
        jaxpr = jax.make_jaxpr(kernel)(points, dipole_points, dipole_moments)
        assert count_jaxpr_primitives(jaxpr, "scan") == 1, kernel.__name__


def test_dipole_field_Bn_stages_as_static_jit_kernel() -> None:
    """The PM matrix path compiles once for fixed symmetry/basis metadata."""

    points = jnp.asarray(_POINTS, dtype=jnp.float64)
    dipole_points = jnp.asarray(_DIPOLE_POINTS, dtype=jnp.float64)
    unitnormal = jnp.asarray(_UNITNORMAL, dtype=jnp.float64)
    b_obj = jnp.asarray(_BN_TARGET, dtype=jnp.float64)

    def Bn_kernel(
        points_arg: jax.Array,
        dipoles_arg: jax.Array,
        normals_arg: jax.Array,
        b_arg: jax.Array,
    ) -> jax.Array:
        return dipole_field_Bn(
            points_arg,
            dipoles_arg,
            normals_arg,
            3,
            1,
            b_arg,
            "cartesian",
            1.05,
        )

    jaxpr = jax.make_jaxpr(Bn_kernel)(points, dipole_points, unitnormal, b_obj)

    assert count_jaxpr_primitives(jaxpr, "jit") == 1


def test_immutable_spec_jits_without_host_oracle_dependency() -> None:
    """Spec-based entry points are traceable and reuse the raw kernel outputs."""

    spec = make_dipole_field_spec(_DIPOLE_POINTS, _DIPOLE_MOMENTS)
    expected = (
        dipole_field_B(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS),
        dipole_field_A(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS),
        dipole_field_dB(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS),
        dipole_field_dA(_POINTS, _DIPOLE_POINTS, _DIPOLE_MOMENTS),
    )

    actual = jax.jit(
        lambda points, spec: (
            dipole_field_B_from_spec(points, spec),
            dipole_field_A_from_spec(points, spec),
            dipole_field_dB_from_spec(points, spec),
            dipole_field_dA_from_spec(points, spec),
        )
    )(_POINTS, spec)

    for actual_leaf, expected_leaf in zip(actual, expected, strict=True):
        _assert_direct_kernel_close(np.asarray(actual_leaf), np.asarray(expected_leaf))


@pytest.mark.parametrize("coordinate_flag", ["cartesian", "cylindrical", "toroidal"])
def test_python_dipolefield_preprocessing_usage_matches_jax_raw_kernels(
    coordinate_flag: str,
) -> None:
    """Existing ``DipoleField`` symmetry/orientation preprocessing feeds JAX."""

    field = DipoleField(
        _DIPOLE_POINTS,
        _DIPOLE_MOMENTS.reshape(-1),
        stellsym=True,
        nfp=2,
        coordinate_flag=coordinate_flag,
        R0=1.05,
    )
    field.set_points(_POINTS)

    actual_B = np.asarray(dipole_field_B(_POINTS, field.dipole_grid, field.m_vec))
    actual_A = np.asarray(dipole_field_A(_POINTS, field.dipole_grid, field.m_vec))
    actual_dB = np.asarray(dipole_field_dB(_POINTS, field.dipole_grid, field.m_vec))
    actual_dA = np.asarray(dipole_field_dA(_POINTS, field.dipole_grid, field.m_vec))

    _assert_direct_kernel_close(actual_B, np.asarray(field.B()))
    _assert_direct_kernel_close(actual_A, np.asarray(field.A()))
    _assert_direct_kernel_close(actual_dB, np.asarray(field.dB_by_dX()))
    _assert_direct_kernel_close(actual_dA, np.asarray(field.dA_by_dX()))


@pytest.mark.parametrize("coordinate_flag", ["cartesian", "cylindrical", "toroidal"])
def test_dipole_field_Bn_cpp_parity_for_production_matrix(
    coordinate_flag: str,
) -> None:
    """``dipole_field_Bn`` matches the C++ PM optimization matrix kernel."""

    expected = np.asarray(
        sopp.dipole_field_Bn(
            _POINTS,
            _DIPOLE_POINTS,
            _UNITNORMAL,
            3,
            1,
            _BN_TARGET,
            coordinate_flag,
            1.05,
        )
    )
    actual = np.asarray(
        dipole_field_Bn(
            _POINTS,
            _DIPOLE_POINTS,
            _UNITNORMAL,
            3,
            1,
            _BN_TARGET,
            coordinate_flag,
            1.05,
        )
    )

    assert (
        actual.shape == expected.shape == (_POINTS.shape[0], _DIPOLE_POINTS.shape[0], 3)
    )
    _assert_direct_kernel_close(actual, expected)


def test_uniform_cartesian_grid_between_toroidal_surfaces_cpp_parity() -> None:
    """The production grid-filter helper matches the C++ zero-row contract."""

    expected = np.asarray(
        sopp.define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
            _GRID_NORMAL_INNER,
            _GRID_NORMAL_OUTER,
            _GRID_XYZ_UNIFORM,
            _GRID_XYZ_INNER,
            _GRID_XYZ_OUTER,
        )
    )
    actual = np.asarray(
        define_a_uniform_cartesian_grid_between_two_toroidal_surfaces(
            _GRID_NORMAL_INNER,
            _GRID_NORMAL_OUTER,
            _GRID_XYZ_UNIFORM,
            _GRID_XYZ_INNER,
            _GRID_XYZ_OUTER,
        )
    )

    _assert_direct_kernel_close(actual, expected)


def test_uniform_cartesian_grid_filter_streams_candidate_points() -> None:
    """The grid filter scans candidate points instead of vmapping ray batches."""

    normal_inner = jnp.asarray(_GRID_NORMAL_INNER, dtype=jnp.float64)
    normal_outer = jnp.asarray(_GRID_NORMAL_OUTER, dtype=jnp.float64)
    xyz_uniform = jnp.asarray(_GRID_XYZ_UNIFORM, dtype=jnp.float64)
    xyz_inner = jnp.asarray(_GRID_XYZ_INNER, dtype=jnp.float64)
    xyz_outer = jnp.asarray(_GRID_XYZ_OUTER, dtype=jnp.float64)

    jaxpr = jax.make_jaxpr(
        define_a_uniform_cartesian_grid_between_two_toroidal_surfaces
    )(normal_inner, normal_outer, xyz_uniform, xyz_inner, xyz_outer)

    assert count_jaxpr_primitives(jaxpr, "scan") == 1
