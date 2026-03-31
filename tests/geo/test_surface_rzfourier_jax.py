import numpy as np
import jax
import jax.numpy as jnp

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.boozer_residual_jax import _surface_geometry_from_dofs
from simsopt.jax_core import (
    SurfaceRZFourierSpec,
    make_surface_rzfourier_spec,
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
    surface_rz_fourier_spec_from_dofs,
    surface_rz_fourier_volume_from_dofs,
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
    surface.rc[0, : surface.ntor] = 0.0
    surface.zs[0, : surface.ntor + 1] = 0.0
    if not stellsym:
        surface.rs[:, :] = rng.normal(scale=0.03, size=surface.rs.shape)
        surface.zc[:, :] = rng.normal(scale=0.03, size=surface.zc.shape)
        surface.rs[0, : surface.ntor + 1] = 0.0
        surface.zc[0, : surface.ntor] = 0.0
    surface.local_full_x = surface.get_dofs()
    return surface


def _assert_surface_parity(surface: SurfaceRZFourier) -> None:
    spec = surface.surface_spec()
    dofs = surface.get_dofs()
    assert isinstance(spec, SurfaceRZFourierSpec)

    gamma_jax = np.asarray(surface_rz_fourier_gamma_from_spec(spec))
    gd1_jax = np.asarray(surface_rz_fourier_gammadash1_from_spec(spec))
    gd2_jax = np.asarray(surface_rz_fourier_gammadash2_from_spec(spec))
    normal_jax = np.asarray(surface_rz_fourier_normal_from_spec(spec))
    gamma_from_dofs = np.asarray(surface_rz_fourier_gamma_from_dofs(spec, dofs))
    gd1_from_dofs = np.asarray(surface_rz_fourier_gammadash1_from_dofs(spec, dofs))
    gd2_from_dofs = np.asarray(surface_rz_fourier_gammadash2_from_dofs(spec, dofs))
    normal_from_dofs = np.asarray(surface_rz_fourier_normal_from_dofs(spec, dofs))

    np.testing.assert_allclose(gamma_jax, surface.gamma(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gd1_jax, surface.gammadash1(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gd2_jax, surface.gammadash2(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(normal_jax, surface.normal(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(gamma_from_dofs, surface.gamma(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        gd1_from_dofs, surface.gammadash1(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        gd2_from_dofs, surface.gammadash2(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        normal_from_dofs, surface.normal(), rtol=1e-12, atol=1e-12
    )

    np.testing.assert_allclose(
        np.asarray(surface.gamma_jax()),
        surface.gamma(),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(surface.gamma_jax(dofs)),
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
        np.asarray(surface.normal_jax(dofs)),
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
        float(surface_rz_fourier_area_from_dofs(spec, dofs)),
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
    np.testing.assert_allclose(
        float(surface_rz_fourier_volume_from_dofs(spec, dofs)),
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


def test_surface_rzfourier_geometry_from_dofs_matches_boozer_hot_path():
    surface = _make_surface(stellsym=False)
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    gamma, xphi, xtheta = _surface_geometry_from_dofs(
        dofs,
        jnp.asarray(surface.quadpoints_phi),
        jnp.asarray(surface.quadpoints_theta),
        surface.mpol,
        surface.ntor,
        surface.nfp,
        surface.stellsym,
        None,
        surface_kind="rzfourier",
    )
    np.testing.assert_allclose(
        np.asarray(gamma), surface.gamma(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(xphi), surface.gammadash1(), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(xtheta), surface.gammadash2(), rtol=1e-12, atol=1e-12
    )


def _assert_area_volume_gradient_parity(surface: SurfaceRZFourier) -> None:
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    spec = surface.surface_spec()

    area_grad = np.asarray(
        jax.grad(lambda x: surface_rz_fourier_area_from_dofs(spec, x))(dofs)
    )
    volume_grad = np.asarray(
        jax.grad(lambda x: surface_rz_fourier_volume_from_dofs(spec, x))(dofs)
    )

    np.testing.assert_allclose(
        area_grad,
        np.asarray(surface.darea_by_dcoeff()),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        volume_grad,
        np.asarray(surface.dvolume_by_dcoeff()),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(surface.darea_by_dcoeff_jax(dofs)),
        area_grad,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(surface.dvolume_by_dcoeff_jax(dofs)),
        volume_grad,
        rtol=1e-12,
        atol=1e-12,
    )


def test_surface_rzfourier_area_volume_gradient_parity_stellsym():
    _assert_area_volume_gradient_parity(_make_surface(stellsym=True))


def test_surface_rzfourier_area_volume_gradient_parity_non_stellsym():
    _assert_area_volume_gradient_parity(_make_surface(stellsym=False))


def test_surface_rzfourier_spec_from_dofs_round_trip():
    surface = _make_surface(stellsym=False)
    spec = surface_rz_fourier_spec_from_dofs(
        surface.get_dofs(),
        quadpoints_phi=surface.quadpoints_phi,
        quadpoints_theta=surface.quadpoints_theta,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )
    np.testing.assert_allclose(
        np.asarray(spec.rc), np.asarray(surface.rc), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(spec.rs), np.asarray(surface.rs), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(spec.zc), np.asarray(surface.zc), rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(spec.zs), np.asarray(surface.zs), rtol=0.0, atol=1e-12
    )
