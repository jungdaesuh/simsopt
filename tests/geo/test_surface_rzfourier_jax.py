import numpy as np
import jax

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.jax_core import (
    SurfaceRZFourierSpec,
    make_surface_rzfourier_spec,
    surface_rz_fourier_area_from_spec,
    surface_rz_fourier_gamma_from_spec,
    surface_rz_fourier_gammadash1_from_spec,
    surface_rz_fourier_gammadash2_from_spec,
    surface_rz_fourier_normal_from_spec,
    surface_rz_fourier_volume_from_spec,
)


jax.config.update("jax_enable_x64", True)


def _make_surface(*, stellsym: bool) -> SurfaceRZFourier:
    rng = np.random.default_rng(7 if stellsym else 11)
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=2,
        stellsym=stellsym,
        mpol=2,
        ntor=1,
        nphi=9,
        ntheta=10,
        range="field period",
    )
    surface.rc[:, :] = rng.normal(scale=0.05, size=surface.rc.shape)
    surface.zs[:, :] = rng.normal(scale=0.05, size=surface.zs.shape)
    surface.rc[0, surface.ntor] = 1.2
    surface.rc[1, surface.ntor] += 0.15
    surface.zs[1, surface.ntor] += 0.08
    if not stellsym:
        surface.rs[:, :] = rng.normal(scale=0.03, size=surface.rs.shape)
        surface.zc[:, :] = rng.normal(scale=0.03, size=surface.zc.shape)
    surface.local_full_x = surface.get_dofs()
    return surface


def _assert_surface_parity(surface: SurfaceRZFourier) -> None:
    spec = surface.surface_spec()
    assert isinstance(spec, SurfaceRZFourierSpec)

    gamma_jax = np.asarray(surface_rz_fourier_gamma_from_spec(spec))
    gd1_jax = np.asarray(surface_rz_fourier_gammadash1_from_spec(spec))
    gd2_jax = np.asarray(surface_rz_fourier_gammadash2_from_spec(spec))
    normal_jax = np.asarray(surface_rz_fourier_normal_from_spec(spec))

    np.testing.assert_allclose(gamma_jax, surface.gamma(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gd1_jax, surface.gammadash1(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gd2_jax, surface.gammadash2(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(normal_jax, surface.normal(), rtol=1e-12, atol=1e-12)

    np.testing.assert_allclose(
        np.asarray(surface.gamma_jax()),
        surface.gamma(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(surface.normal_jax()),
        surface.normal(),
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        float(surface_rz_fourier_area_from_spec(spec)),
        surface.area(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        float(surface_rz_fourier_volume_from_spec(spec)),
        surface.volume(),
        rtol=1e-12,
        atol=1e-12,
    )


def test_surface_rzfourier_jax_parity_stellsym():
    _assert_surface_parity(_make_surface(stellsym=True))


def test_surface_rzfourier_jax_parity_non_stellsym():
    _assert_surface_parity(_make_surface(stellsym=False))


def test_surface_rzfourier_spec_is_jittable():
    spec = make_surface_rzfourier_spec(
        rc=np.asarray([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        zs=np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        quadpoints_phi=np.linspace(0.0, 0.5, 4, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 5, endpoint=False),
        nfp=2,
        stellsym=True,
    )
    gamma = jax.jit(surface_rz_fourier_gamma_from_spec)(spec)
    normal = jax.jit(surface_rz_fourier_normal_from_spec)(spec)
    assert gamma.shape == (4, 5, 3)
    assert normal.shape == (4, 5, 3)
