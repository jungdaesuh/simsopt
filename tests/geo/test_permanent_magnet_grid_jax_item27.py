"""Fixed-state JAX payload tests for permanent-magnet grids."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np
import pytest

import simsoptpp as sopp

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.permanent_magnet_grid_jax import (
    MWPGP_ALPHA_SAFETY_FACTOR,
    PermanentMagnetGridJAX,
    mwpgp_alpha_from_grid,
    permanent_magnet_grid_to_jax,
)

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]


@dataclass(frozen=True)
class _PlasmaBoundary:
    nfp: int
    stellsym: bool


@dataclass(frozen=True)
class _CPUGrid:
    A_obj: np.ndarray
    b_obj: np.ndarray
    ATb: np.ndarray
    ATA_scale: float
    m0: np.ndarray
    m: np.ndarray
    m_proxy: np.ndarray
    m_maxima: np.ndarray
    dipole_grid_xyz: np.ndarray
    coordinate_flag: str
    R0: float
    plasma_boundary: _PlasmaBoundary
    nphi: int
    ntheta: int
    ndipoles: int


def _fixed_state_fixture():
    points = np.ascontiguousarray(
        np.array(
            [
                [1.10, 0.20, -0.15],
                [1.35, -0.10, 0.22],
                [1.75, 0.30, 0.05],
                [2.05, -0.25, -0.18],
            ],
            dtype=np.float64,
        )
    )
    normal = np.ascontiguousarray(
        np.array(
            [
                [0.40, -0.10, 0.20],
                [-0.25, 0.55, 0.12],
                [0.30, 0.25, -0.35],
                [-0.45, -0.20, 0.30],
            ],
            dtype=np.float64,
        )
    )
    Bn = np.ascontiguousarray(
        np.array([[0.12, -0.18], [0.22, -0.05]], dtype=np.float64)
    )
    dipoles = np.ascontiguousarray(
        np.array(
            [
                [0.42, 0.18, -0.20],
                [0.75, -0.22, 0.31],
                [1.05, 0.12, 0.08],
            ],
            dtype=np.float64,
        )
    )
    m_maxima = np.ascontiguousarray(np.array([0.4, 0.6, 0.8], dtype=np.float64))
    return points, normal, Bn, dipoles, m_maxima


def _expected_cpu_matrix(points, normal, Bn, dipoles, coordinate_flag):
    b_obj = -Bn.reshape(-1)
    unitnormal = normal / np.linalg.norm(normal, axis=1)[:, None]
    A_raw = np.asarray(
        sopp.dipole_field_Bn(
            points,
            dipoles,
            np.ascontiguousarray(unitnormal),
            2,
            1,
            np.ascontiguousarray(b_obj),
            coordinate_flag,
            1.25,
        )
    ).reshape(Bn.size, dipoles.shape[0] * 3)
    scale = np.sqrt(np.linalg.norm(normal, axis=1) / float(Bn.size))
    A_obj = A_raw * scale[:, None]
    b_scaled = b_obj * scale
    return A_obj, b_scaled


@pytest.mark.parametrize("coordinate_flag", ("cartesian", "cylindrical", "toroidal"))
def test_from_fixed_state_matches_cpu_pm_matrix(coordinate_flag):
    points, normal, Bn, dipoles, m_maxima = _fixed_state_fixture()

    grid = PermanentMagnetGridJAX.from_fixed_state(
        plasma_points=points,
        normal=normal,
        Bn=Bn,
        dipole_grid_xyz=dipoles,
        m_maxima=m_maxima,
        nfp=2,
        stellsym=True,
        coordinate_flag=coordinate_flag,
        R0=1.25,
    )
    A_expected, b_expected = _expected_cpu_matrix(
        points, normal, Bn, dipoles, coordinate_flag
    )

    np.testing.assert_allclose(
        np.asarray(grid.A_obj), A_expected, rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(grid.b_obj), b_expected, rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(
        np.asarray(grid.ATb),
        (A_expected.T @ b_expected).reshape(dipoles.shape[0], 3),
        rtol=_RTOL,
        atol=_ATOL,
    )
    expected_singular_values = np.linalg.svd(
        A_expected, full_matrices=False, compute_uv=False
    )
    assert np.asarray(grid.ATA_scale) == pytest.approx(
        expected_singular_values[0] ** 2,
        rel=_RTOL,
        abs=_ATOL,
    )
    assert grid.ndipoles == dipoles.shape[0]
    assert grid.nphi == 2
    assert grid.ntheta == 2


def test_from_fixed_state_preserves_pol_vectors():
    points, normal, Bn, dipoles, m_maxima = _fixed_state_fixture()
    pol_vectors = np.ascontiguousarray(
        np.array(
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            ],
            dtype=np.float64,
        )
    )

    grid = PermanentMagnetGridJAX.from_fixed_state(
        plasma_points=points,
        normal=normal,
        Bn=Bn,
        dipole_grid_xyz=dipoles,
        m_maxima=m_maxima,
        nfp=2,
        stellsym=True,
        coordinate_flag="cartesian",
        R0=1.25,
        pol_vectors=pol_vectors,
    )

    assert grid.pol_vectors is not None
    np.testing.assert_allclose(
        np.asarray(grid.pol_vectors),
        pol_vectors,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_from_cpu_preserves_staged_payload_and_alpha_rule():
    A_obj = np.ascontiguousarray(
        np.array(
            [
                [1.2, -0.4, 0.5, 0.7, -0.1, 0.3],
                [0.0, 1.7, -0.3, -0.6, 0.2, 1.1],
                [2.1, 0.8, 1.4, 0.9, -1.5, 0.7],
            ],
            dtype=np.float64,
        )
    )
    b_obj = np.ascontiguousarray(np.array([0.3, -0.7, 1.1], dtype=np.float64))
    ATb = A_obj.T @ b_obj
    ATA_scale = float(np.linalg.svd(A_obj, compute_uv=False)[0] ** 2)
    m0 = np.zeros(6, dtype=np.float64)
    m = np.linspace(-0.1, 0.2, 6, dtype=np.float64)
    m_proxy = np.linspace(0.05, -0.15, 6, dtype=np.float64)
    m_maxima = np.array([0.4, 0.8], dtype=np.float64)
    dipoles = np.array([[0.2, 0.1, -0.3], [0.6, -0.2, 0.4]], dtype=np.float64)
    cpu = _CPUGrid(
        A_obj=A_obj,
        b_obj=b_obj,
        ATb=ATb,
        ATA_scale=ATA_scale,
        m0=m0,
        m=m,
        m_proxy=m_proxy,
        m_maxima=m_maxima,
        dipole_grid_xyz=dipoles,
        coordinate_flag="cartesian",
        R0=1.1,
        plasma_boundary=_PlasmaBoundary(nfp=2, stellsym=True),
        nphi=1,
        ntheta=3,
        ndipoles=2,
    )

    grid = permanent_magnet_grid_to_jax(cpu)

    np.testing.assert_allclose(np.asarray(grid.A_obj), A_obj, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(np.asarray(grid.b_obj), b_obj, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(
        np.asarray(grid.ATb), ATb.reshape(2, 3), rtol=_RTOL, atol=_ATOL
    )
    np.testing.assert_allclose(np.asarray(grid.m0), m0.reshape(2, 3))
    np.testing.assert_allclose(np.asarray(grid.m), m.reshape(2, 3))
    np.testing.assert_allclose(np.asarray(grid.m_proxy), m_proxy.reshape(2, 3))
    np.testing.assert_allclose(np.asarray(grid.m_maxima), m_maxima)
    np.testing.assert_allclose(np.asarray(grid.dipole_grid_xyz), dipoles)
    assert grid.coordinate_flag == "cartesian"
    assert grid.R0 == pytest.approx(1.1)
    assert grid.nfp == 2
    assert grid.stellsym is True
    np.testing.assert_allclose(
        np.asarray(mwpgp_alpha_from_grid(grid)),
        2.0 * MWPGP_ALPHA_SAFETY_FACTOR / ATA_scale,
        rtol=_RTOL,
        atol=_ATOL,
    )


def test_grid_payload_is_jit_compatible_under_transfer_guard():
    points, normal, Bn, dipoles, m_maxima = _fixed_state_fixture()
    grid = PermanentMagnetGridJAX.from_fixed_state(
        plasma_points=points,
        normal=normal,
        Bn=Bn,
        dipole_grid_xyz=dipoles,
        m_maxima=m_maxima,
        nfp=2,
        stellsym=True,
        coordinate_flag="cartesian",
        R0=1.25,
    )

    @jax.jit
    def _normal_equation_value(grid_data: PermanentMagnetGridJAX):
        return grid_data.A_obj.T @ (grid_data.A_obj @ grid_data.m0.reshape(-1))

    _normal_equation_value(grid).block_until_ready()
    with jax.transfer_guard("disallow"):
        out = _normal_equation_value(grid)
        out.block_until_ready()

    assert out.shape == (grid.ndipoles * 3,)
    np.testing.assert_allclose(
        np.asarray(out),
        np.zeros(grid.ndipoles * 3),
        rtol=_RTOL,
        atol=_ATOL,
    )
