"""Parity tests for the JAX SurfaceXYZTensorFourier ``clamped_dims`` path.

These tests guard the JAX-native implementation of the boundary-condition
enforcer that the C++ ``SurfaceXYZTensorFourier`` applies when any of
``clamped_dims`` is ``True``. The CPU C++ class multiplies the basis
function on the ``(m <= mpol, n <= ntor)`` cos-cos block by
``E(phi, theta) = sin(nfp*phi/2)^2 + sin(theta/2)^2`` for the dim flagged
as clamped. The JAX kernel adds the same correction term inside
``surface_*_from_dofs`` and the spec adapter threads
``clamped_dims`` through ``SurfaceXYZTensorFourierSpec``.

The tests exercise all 8 ``clamped_dims`` combinations under both
stellsym and non-stellsym surfaces, the spec round-trip, the strict
transfer guard, and the JIT cache contract.
"""

from __future__ import annotations

import itertools

import jax
import numpy as np
import pytest

from simsopt_jax.core import (
    SurfaceXYZTensorFourierSpec,
    make_surface_xyz_tensor_fourier_spec,
)

CLAMPED_COMBINATIONS = list(itertools.product((False, True), repeat=3))


@pytest.fixture(autouse=True)
def _require_simsoptpp():
    pytest.importorskip("simsoptpp")
    pytest.importorskip("simsopt")


def _build_surface(*, clamped_dims, stellsym, seed):
    """Build a CPU SurfaceXYZTensorFourier and a matching JAX spec."""
    from simsopt.geo import SurfaceXYZTensorFourier

    mpol, ntor, nfp = 2, 1, 2
    quadpoints_phi = np.linspace(0.0, 1.0 / nfp, 7, endpoint=False)
    quadpoints_theta = np.linspace(0.0, 1.0, 6, endpoint=False)

    surface = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        clamped_dims=list(clamped_dims),
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    rng = np.random.default_rng(seed)
    dofs = surface.get_dofs().copy()
    dofs[:] = rng.normal(scale=0.1, size=dofs.shape)
    # Bias the major-radius mode away from zero so the surface is non-trivial.
    dofs[0] += 1.0
    surface.set_dofs(dofs)
    return surface


def _surface_spec(surface) -> SurfaceXYZTensorFourierSpec:
    return make_surface_xyz_tensor_fourier_spec(
        dofs=surface.get_dofs(),
        quadpoints_phi=surface.quadpoints_phi,
        quadpoints_theta=surface.quadpoints_theta,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
        mpol=surface.mpol,
        ntor=surface.ntor,
        clamped_dims=tuple(surface.clamped_dims),
    )


@pytest.mark.parametrize("stellsym", [True, False])
def test_unclamped_baseline_gamma_parity(stellsym):
    """Regression: unclamped path still matches CPU within rtol=1e-12."""
    from simsopt_jax.core import (
        surface_xyz_tensor_fourier_gamma_from_spec,
        surface_xyz_tensor_fourier_gammadash1_from_spec,
        surface_xyz_tensor_fourier_gammadash2_from_spec,
    )

    surface = _build_surface(
        clamped_dims=(False, False, False),
        stellsym=stellsym,
        seed=31 + int(stellsym),
    )
    spec = _surface_spec(surface)

    np.testing.assert_allclose(
        np.asarray(jax.jit(surface_xyz_tensor_fourier_gamma_from_spec)(spec)),
        surface.gamma(),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(surface_xyz_tensor_fourier_gammadash1_from_spec)(spec)),
        surface.gammadash1(),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(surface_xyz_tensor_fourier_gammadash2_from_spec)(spec)),
        surface.gammadash2(),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize(
    "clamped_dims",
    CLAMPED_COMBINATIONS,
    ids=lambda v: "c={}{}{}".format(*[int(x) for x in v]),
)
@pytest.mark.parametrize("stellsym", [True, False])
def test_clamped_combination_gamma_parity(clamped_dims, stellsym):
    """JAX gamma/gammadash1/gammadash2 match CPU for every clamped combo."""
    from simsopt_jax.core import (
        surface_xyz_tensor_fourier_gamma_from_spec,
        surface_xyz_tensor_fourier_gammadash1_from_spec,
        surface_xyz_tensor_fourier_gammadash2_from_spec,
    )

    surface = _build_surface(
        clamped_dims=clamped_dims,
        stellsym=stellsym,
        seed=101 + sum(int(v) for v in clamped_dims) * 7 + int(stellsym),
    )
    spec = _surface_spec(surface)

    np.testing.assert_allclose(
        np.asarray(jax.jit(surface_xyz_tensor_fourier_gamma_from_spec)(spec)),
        surface.gamma(),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(surface_xyz_tensor_fourier_gammadash1_from_spec)(spec)),
        surface.gammadash1(),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(surface_xyz_tensor_fourier_gammadash2_from_spec)(spec)),
        surface.gammadash2(),
        rtol=1e-12,
        atol=1e-14,
    )


@pytest.mark.parametrize("clamped_dims", CLAMPED_COMBINATIONS)
@pytest.mark.parametrize("stellsym", [True, False])
def test_clamped_normal_and_unitnormal_parity(clamped_dims, stellsym):
    """Normal and unitnormal must match CPU for every clamped combination.

    The C++ ``apply_bc_enforcer`` at
    ``src/simsoptpp/surfacexyztensorfourier.h:903-913`` activates the
    clamping per dimension, so a bug that only hits, say, ``dim=2``
    normal would slip past a single ``(True, True, True)`` test. Full
    8-combo coverage on both stellsym branches closes this gap.
    """
    from simsopt_jax.core import (
        surface_xyz_tensor_fourier_normal_from_spec,
        surface_xyz_tensor_fourier_unitnormal_from_spec,
    )

    surface = _build_surface(
        clamped_dims=clamped_dims,
        stellsym=stellsym,
        seed=211 + int(stellsym) + 11 * sum(1 << i for i, c in enumerate(clamped_dims) if c),
    )
    spec = _surface_spec(surface)

    np.testing.assert_allclose(
        np.asarray(jax.jit(surface_xyz_tensor_fourier_normal_from_spec)(spec)),
        surface.normal(),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(surface_xyz_tensor_fourier_unitnormal_from_spec)(spec)),
        surface.unitnormal(),
        rtol=1e-12,
        atol=1e-14,
    )


def test_explicit_spec_no_longer_raises_clamped_dims():
    """Explicit spec construction must not raise for any clamped combination."""
    from simsopt.geo import SurfaceXYZTensorFourier

    for clamped_dims in CLAMPED_COMBINATIONS:
        surface = SurfaceXYZTensorFourier(
            mpol=2,
            ntor=1,
            nfp=2,
            stellsym=True,
            clamped_dims=list(clamped_dims),
            quadpoints_phi=np.linspace(0.0, 0.5, 7, endpoint=False),
            quadpoints_theta=np.linspace(0.0, 1.0, 6, endpoint=False),
        )
        # Must not raise.
        spec = _surface_spec(surface)
        assert spec.clamped_dims == tuple(bool(v) for v in clamped_dims)


def test_explicit_spec_round_trips_clamped_dims():
    """Spec stores `clamped_dims` as a hashable bool tuple."""
    from simsopt.geo import SurfaceXYZTensorFourier

    surface = SurfaceXYZTensorFourier(
        mpol=2,
        ntor=1,
        nfp=2,
        stellsym=True,
        clamped_dims=[True, False, True],
        quadpoints_phi=np.linspace(0.0, 0.5, 7, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 6, endpoint=False),
    )
    spec = _surface_spec(surface)
    assert spec.clamped_dims == (True, False, True)
    # Ensure it is a Python bool tuple (hashable, JIT-cache friendly).
    assert isinstance(spec.clamped_dims, tuple)
    assert all(isinstance(v, bool) for v in spec.clamped_dims)


def test_clamped_gamma_runs_under_strict_transfer_guard():
    """JIT-compiled clamped gamma evaluation respects ``transfer_guard("disallow")``."""
    from simsopt_jax.core import surface_xyz_tensor_fourier_gamma_from_spec

    surface = _build_surface(
        clamped_dims=(True, False, True),
        stellsym=False,
        seed=331,
    )
    spec = _surface_spec(surface)
    compiled = jax.jit(surface_xyz_tensor_fourier_gamma_from_spec)
    # Warm-up to amortize the device staging of compiled code.
    compiled(spec).block_until_ready()

    with jax.transfer_guard("disallow"):
        compiled(spec).block_until_ready()


def test_clamped_dims_invalidates_jit_cache():
    """Different clamped_dims must produce numerically different results
    when the underlying coefficient block is nonzero. This proxies the
    JIT cache key: if the cache reused a stale compile, the clamped and
    unclamped outputs would coincide.
    """
    from simsopt_jax.core import surface_xyz_tensor_fourier_gamma_from_spec

    unclamped = _build_surface(
        clamped_dims=(False, False, False), stellsym=False, seed=4001
    )
    spec_uc = _surface_spec(unclamped)

    clamped = _build_surface(clamped_dims=(True, True, True), stellsym=False, seed=4001)
    spec_c = _surface_spec(clamped)

    gamma_uc = np.asarray(jax.jit(surface_xyz_tensor_fourier_gamma_from_spec)(spec_uc))
    gamma_c = np.asarray(jax.jit(surface_xyz_tensor_fourier_gamma_from_spec)(spec_c))

    # The two surfaces share the same DOF vector but differ in clamped
    # flags, so the gammas must differ by more than floating-point noise.
    assert not np.allclose(gamma_uc, gamma_c, rtol=1e-8, atol=1e-10)
