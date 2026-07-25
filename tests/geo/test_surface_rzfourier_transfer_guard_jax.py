import jax
import jax.numpy as jnp
import numpy as np

from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt_jax.core import (
    surface_rz_fourier_darea_from_dofs,
    surface_rz_fourier_daspect_ratio_from_dofs,
    surface_rz_fourier_dmajor_radius_from_dofs,
    surface_rz_fourier_dmean_cross_sectional_area_from_dofs,
    surface_rz_fourier_dminor_radius_from_dofs,
    surface_rz_fourier_dvolume_from_dofs,
    surface_rz_fourier_spec_from_dofs,
)
from simsopt_jax.core.surface_rzfourier import (
    _surface_rz_fourier_derivative_lin_from_spec,
)


def _make_surface() -> SurfaceRZFourier:
    surface = SurfaceRZFourier.from_nphi_ntheta(
        nfp=2,
        stellsym=True,
        mpol=2,
        ntor=1,
        nphi=9,
        ntheta=10,
        range="field period",
    )
    surface.rc[:, :] = 0.0
    surface.zs[:, :] = 0.0
    surface.rc[0, surface.ntor] = 1.2
    surface.rc[1, surface.ntor] = 0.15
    surface.zs[1, surface.ntor] = 0.08
    surface.rc[1, surface.ntor + 1] = 0.02
    surface.zs[1, surface.ntor + 1] = -0.03
    surface.local_full_x = surface.get_dofs()
    return surface


def _surface_spec_from_surface(surface: SurfaceRZFourier):
    return surface_rz_fourier_spec_from_dofs(
        surface.get_dofs(),
        quadpoints_phi=surface.quadpoints_phi,
        quadpoints_theta=surface.quadpoints_theta,
        mpol=surface.mpol,
        ntor=surface.ntor,
        nfp=surface.nfp,
        stellsym=surface.stellsym,
    )


def test_surface_rzfourier_scalar_gradients_allow_strict_transfer_guard():
    surface = _make_surface()
    spec = _surface_spec_from_surface(surface)
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)

    gradient_fns = (
        surface_rz_fourier_darea_from_dofs,
        surface_rz_fourier_dvolume_from_dofs,
        surface_rz_fourier_dmean_cross_sectional_area_from_dofs,
        surface_rz_fourier_dminor_radius_from_dofs,
        surface_rz_fourier_dmajor_radius_from_dofs,
        surface_rz_fourier_daspect_ratio_from_dofs,
    )

    compiled_fns = tuple(
        jax.jit(lambda x, fn=gradient_fn: fn(spec, x)) for gradient_fn in gradient_fns
    )

    with jax.transfer_guard("disallow"):
        for compiled_fn in compiled_fns:
            compiled_fn(dofs).block_until_ready()


def test_surface_rzfourier_eager_linear_derivatives_allow_strict_transfer_guard():
    surface = _make_surface()
    spec = _surface_spec_from_surface(surface)
    quadpoints_phi = jax.device_put(np.asarray([0.0, 0.125], dtype=np.float64))
    quadpoints_theta = jax.device_put(np.asarray([0.0, 0.25], dtype=np.float64))

    with jax.transfer_guard("disallow"):
        gammadash1 = _surface_rz_fourier_derivative_lin_from_spec(
            spec,
            quadpoints_phi,
            quadpoints_theta,
            1,
            0,
        )
        gammadash2 = _surface_rz_fourier_derivative_lin_from_spec(
            spec,
            quadpoints_phi,
            quadpoints_theta,
            0,
            1,
        )
        gammadash1.block_until_ready()
        gammadash2.block_until_ready()
