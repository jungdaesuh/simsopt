"""JAX port goal item 06 closeout coverage witness.

The existing repo-wide JAX SurfaceRZFourier parity suite at
``tests/geo/test_surface_rzfourier_jax.py`` covers ``stellsym=False`` only
at ``nphi=9, ntheta=10`` (see ``_make_surface``). The production-scale
HLO probe in the same module raises ``nphi`` and ``ntheta`` (17/18,
32/33) but pins ``stellsym=True``, leaving the non-stellsym SurfaceRZFourier
geometry kernel without a production-scale CPU/JAX parity fixture.

This module installs that missing witness for prompt item 06 of the
JAX port goal manifest. It exercises the JAX SurfaceRZFourier adapter
at ``stellsym=False``, ``nphi=32``, ``ntheta=16`` (above the
``nphi >= 16, ntheta >= 8`` production-scale floor in section 4c of
``jax_port_goal_prompt_2026-05-12.md``), covering ``gamma``,
``gammadash1``, ``gammadash2``, ``normal``, ``area``, and ``volume``
against the simsoptpp-backed CPU oracle. Tolerances come from the
``direct_kernel`` parity-ladder lane
(``benchmarks.validation_ladder_contract.PARITY_LADDER_TOLERANCES``).
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import host_array, host_scalar, parity_default_device, parity_rng

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.jax_core import (
    surface_rz_fourier_area_from_dofs,
    surface_rz_fourier_area_from_spec,
    surface_rz_fourier_gamma_from_dofs,
    surface_rz_fourier_gamma_from_spec,
    surface_rz_fourier_gammadash1_from_dofs,
    surface_rz_fourier_gammadash1_from_spec,
    surface_rz_fourier_gammadash2_from_dofs,
    surface_rz_fourier_gammadash2_from_spec,
    surface_rz_fourier_normal_from_dofs,
    surface_rz_fourier_normal_from_spec,
    surface_rz_fourier_volume_from_dofs,
    surface_rz_fourier_volume_from_spec,
)

_DIRECT_KERNEL = parity_ladder_tolerances("direct_kernel")
_RTOL = _DIRECT_KERNEL["rtol"]
_ATOL = _DIRECT_KERNEL["atol"]

_NPHI = 32
_NTHETA = 16
_MPOL = 4
_NTOR = 3
_NFP = 2


@pytest.fixture(autouse=True)
def _parity_device_scope():
    """Pin the JAX default device to the CPU parity lane.

    Item 06 declared CPU-only validation; user requested no GPU runs.
    """
    with parity_default_device("cpu"):
        yield


def _make_production_non_stellsym_surface() -> SurfaceRZFourier:
    """Build a 32x16 non-stellsym SurfaceRZFourier at production scale.

    Above the production-scale floor (nphi >= 16, ntheta >= 8) and
    activates every sin/cos branch of the SurfaceRZFourier DOF layout by
    setting ``rs`` and ``zc`` perturbations alongside ``rc`` and ``zs``.
    """
    rng = parity_rng(13)
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=_NFP,
        stellsym=False,
        mpol=_MPOL,
        ntor=_NTOR,
        nphi=_NPHI,
        ntheta=_NTHETA,
        range="field period",
    )
    surface.rc[:, :] = rng.normal(scale=0.05, size=surface.rc.shape)
    surface.zs[:, :] = rng.normal(scale=0.05, size=surface.zs.shape)
    surface.rs[:, :] = rng.normal(scale=0.03, size=surface.rs.shape)
    surface.zc[:, :] = rng.normal(scale=0.03, size=surface.zc.shape)
    surface.rc[0, surface.ntor] = 1.25
    surface.rc[1, surface.ntor] += 0.18
    surface.zs[1, surface.ntor] += 0.12
    surface.rc[0, : surface.ntor] = 0.0
    surface.zs[0, : surface.ntor + 1] = 0.0
    surface.rs[0, : surface.ntor + 1] = 0.0
    surface.zc[0, : surface.ntor] = 0.0
    surface.local_full_x = surface.get_dofs()
    return surface


def _assert_production_scale_non_stellsym_parity(
    surface: SurfaceRZFourier,
) -> None:
    spec = surface.surface_spec()
    dofs = surface.get_dofs()

    cpu_gamma = surface.gamma()
    cpu_gd1 = surface.gammadash1()
    cpu_gd2 = surface.gammadash2()
    cpu_normal = surface.normal()
    cpu_area = surface.area()
    cpu_volume = surface.volume()

    assert cpu_gamma.shape == (_NPHI, _NTHETA, 3)
    assert cpu_normal.shape == (_NPHI, _NTHETA, 3)

    np.testing.assert_allclose(
        host_array(surface_rz_fourier_gamma_from_spec(spec)),
        cpu_gamma,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        host_array(surface_rz_fourier_gammadash1_from_spec(spec)),
        cpu_gd1,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        host_array(surface_rz_fourier_gammadash2_from_spec(spec)),
        cpu_gd2,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        host_array(surface_rz_fourier_normal_from_spec(spec)),
        cpu_normal,
        rtol=_RTOL,
        atol=_ATOL,
    )

    np.testing.assert_allclose(
        host_array(surface_rz_fourier_gamma_from_dofs(spec, dofs)),
        cpu_gamma,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        host_array(surface_rz_fourier_gammadash1_from_dofs(spec, dofs)),
        cpu_gd1,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        host_array(surface_rz_fourier_gammadash2_from_dofs(spec, dofs)),
        cpu_gd2,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        host_array(surface_rz_fourier_normal_from_dofs(spec, dofs)),
        cpu_normal,
        rtol=_RTOL,
        atol=_ATOL,
    )

    assert host_scalar(surface_rz_fourier_area_from_spec(spec)) == pytest.approx(
        cpu_area, rel=_RTOL, abs=_ATOL
    )
    assert host_scalar(surface_rz_fourier_area_from_dofs(spec, dofs)) == pytest.approx(
        cpu_area, rel=_RTOL, abs=_ATOL
    )
    assert host_scalar(surface_rz_fourier_volume_from_spec(spec)) == pytest.approx(
        cpu_volume, rel=_RTOL, abs=_ATOL
    )
    assert host_scalar(
        surface_rz_fourier_volume_from_dofs(spec, dofs)
    ) == pytest.approx(cpu_volume, rel=_RTOL, abs=_ATOL)

    np.testing.assert_allclose(
        host_array(surface.gamma_jax()),
        cpu_gamma,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        host_array(surface.normal_jax()),
        cpu_normal,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        host_array(surface.gammadash1_jax(dofs)),
        cpu_gd1,
        rtol=_RTOL,
        atol=_ATOL,
    )
    np.testing.assert_allclose(
        host_array(surface.gammadash2_jax(dofs)),
        cpu_gd2,
        rtol=_RTOL,
        atol=_ATOL,
    )
    assert host_scalar(surface.area_jax()) == pytest.approx(
        cpu_area, rel=_RTOL, abs=_ATOL
    )
    assert host_scalar(surface.volume_jax(dofs)) == pytest.approx(
        cpu_volume, rel=_RTOL, abs=_ATOL
    )


def test_surface_rzfourier_jax_production_scale_non_stellsym_parity():
    """Closeout coverage witness: production-scale non-stellsym parity.

    Closes the gap left by ``_make_surface(stellsym=False)`` in
    ``tests/geo/test_surface_rzfourier_jax.py`` which runs at ``nphi=9,
    ntheta=10`` only. This test runs the same JAX SurfaceRZFourier paths
    at the prompt's production-scale floor (``nphi=32, ntheta=16``)
    against the simsoptpp CPU oracle at the ``direct_kernel`` parity-ladder
    lane.
    """
    surface = _make_production_non_stellsym_surface()
    _assert_production_scale_non_stellsym_parity(surface)
