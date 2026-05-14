"""N04a parity tests for ``SurfaceGarabedian`` → JAX spec routing.

The host class is at ``src/simsopt/geo/surfacegarabedian.py`` and the
SSOT CPU conversion to ``SurfaceRZFourier`` lives at
``surfacegarabedian.py:to_RZFourier``. The JAX route adds:

- ``SurfaceGarabedianSpec`` (immutable Δ + shape pytree)
- ``make_surface_garabedian_spec`` factory
- ``garabedian_to_rzfourier_spec`` converter (pure JAX)
- ``SurfaceGarabedian.to_spec()`` method

This test file exercises byte-identity parity of the converter against
the CPU ``to_RZFourier()`` output, the ``to_spec`` round-trip, and the
strict-mode transfer-guard behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax

from simsopt.geo.surfacegarabedian import SurfaceGarabedian
from simsopt.jax_core import (
    SurfaceGarabedianSpec,
    garabedian_to_rzfourier_spec,
)

_SHAPE_FIXTURES = [
    pytest.param(0, 1, 0, 0, id="torus-base"),
    pytest.param(0, 2, -1, 1, id="1mode-2nmode"),
    pytest.param(-1, 2, -2, 2, id="1m1m-2n2n"),
    pytest.param(0, 3, -2, 3, id="0to3-m2to3-n"),
]


@pytest.fixture(autouse=True)
def _require_simsoptpp():
    pytest.importorskip("simsoptpp")


def _delta_index(surface: SurfaceGarabedian, m: int, n: int) -> int:
    return surface.ndim * (m - surface.mmin) + n - surface.nmin


def _seed_delta(surface: SurfaceGarabedian, rng: np.random.Generator) -> None:
    """Replace Δ_{m,n} with a randomised but stellsym-consistent payload."""
    rng_scale = 1e-2
    full = np.asarray(surface.local_full_x, dtype=np.float64).copy()
    full = full + rng_scale * rng.standard_normal(full.shape)
    full[_delta_index(surface, 1, 0)] = 1.0
    full[_delta_index(surface, 0, 0)] = 0.1
    surface.local_full_x = full


def _assert_byte_equal(actual: object, expected: object) -> None:
    actual_array = np.asarray(actual)
    expected_array = np.asarray(expected)
    np.testing.assert_array_equal(actual_array, expected_array)
    assert actual_array.dtype == expected_array.dtype
    assert actual_array.tobytes() == expected_array.tobytes()


@pytest.mark.parametrize(("mmin", "mmax", "nmin", "nmax"), _SHAPE_FIXTURES)
def test_to_spec_round_trips_fields(mmin: int, mmax: int, nmin: int, nmax: int) -> None:
    """``to_spec()`` returns a spec whose fields mirror the host class state."""
    surface = SurfaceGarabedian(nfp=3, mmin=mmin, mmax=mmax, nmin=nmin, nmax=nmax)
    rng = np.random.default_rng(2026)
    _seed_delta(surface, rng)

    spec = surface.to_spec()

    assert isinstance(spec, SurfaceGarabedianSpec)
    assert spec.nfp == surface.nfp
    assert spec.mmin == surface.mmin
    assert spec.mmax == surface.mmax
    assert spec.nmin == surface.nmin
    assert spec.nmax == surface.nmax

    np.testing.assert_array_equal(
        np.asarray(spec.dofs), np.asarray(surface.get_dofs(), dtype=np.float64)
    )
    np.testing.assert_array_equal(
        np.asarray(spec.quadpoints_phi),
        np.asarray(surface.quadpoints_phi, dtype=np.float64),
    )
    np.testing.assert_array_equal(
        np.asarray(spec.quadpoints_theta),
        np.asarray(surface.quadpoints_theta, dtype=np.float64),
    )


def test_to_spec_captures_fixed_geometry_dofs() -> None:
    """``to_spec()`` captures all geometry coefficients, including fixed ones."""
    surface = SurfaceGarabedian(nfp=3, mmin=0, mmax=2, nmin=-1, nmax=1)
    rng = np.random.default_rng(2029)
    _seed_delta(surface, rng)
    surface.fix("Delta(1,0)")

    spec = surface.to_spec()

    np.testing.assert_array_equal(
        np.asarray(spec.dofs), np.asarray(surface.local_full_x, dtype=np.float64)
    )


@pytest.mark.parametrize(("mmin", "mmax", "nmin", "nmax"), _SHAPE_FIXTURES)
def test_jax_conversion_matches_cpu_to_rzfourier(
    mmin: int, mmax: int, nmin: int, nmax: int
) -> None:
    """``garabedian_to_rzfourier_spec`` matches CPU ``to_RZFourier()`` byte-identical."""
    surface = SurfaceGarabedian(nfp=3, mmin=mmin, mmax=mmax, nmin=nmin, nmax=nmax)
    rng = np.random.default_rng(2027)
    _seed_delta(surface, rng)

    cpu_rz = surface.to_RZFourier()
    cpu_rc = np.asarray(cpu_rz.rc, dtype=np.float64)
    cpu_zs = np.asarray(cpu_rz.zs, dtype=np.float64)

    spec = surface.to_spec()
    rz_spec = garabedian_to_rzfourier_spec(spec)

    jax_rc = np.asarray(rz_spec.rc, dtype=np.float64)
    jax_zs = np.asarray(rz_spec.zs, dtype=np.float64)

    assert jax_rc.shape == cpu_rc.shape
    assert jax_zs.shape == cpu_zs.shape

    _assert_byte_equal(jax_rc, cpu_rc)
    _assert_byte_equal(jax_zs, cpu_zs)
    np.testing.assert_array_equal(rz_spec.rs, np.zeros_like(cpu_rc))
    np.testing.assert_array_equal(rz_spec.zc, np.zeros_like(cpu_rc))
    assert rz_spec.nfp == cpu_rz.nfp
    assert rz_spec.stellsym is True
    assert rz_spec.mpol == cpu_rz.mpol
    assert rz_spec.ntor == cpu_rz.ntor


def test_garabedian_conversion_under_strict_transfer_guard() -> None:
    """The JAX converter runs cleanly under ``jax.transfer_guard('disallow')``.

    Construction happens outside the guard (the host class touches
    NumPy + simsoptpp); the converter itself must stay on-device.
    """
    surface = SurfaceGarabedian(nfp=2, mmin=0, mmax=2, nmin=-1, nmax=1)
    rng = np.random.default_rng(2028)
    _seed_delta(surface, rng)
    spec = surface.to_spec()

    with jax.transfer_guard("disallow"):
        rz_spec = garabedian_to_rzfourier_spec(spec)

    assert rz_spec.mpol >= 1
    assert rz_spec.ntor >= 1
    assert rz_spec.stellsym is True
