"""JAX-specific ToroidalFlux Taylor coverage.

These tests exercise the pure JAX label/objective ingredients directly:

1. Surface-DOF Hessian Taylor convergence for toroidal flux.
2. Coil-family DOF gradient Taylor convergence for toroidal flux.
3. Object-level CPU/JAX parity for ToroidalFlux value and derivatives.
"""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import (
    enable_strict_parity_backend,
    parity_default_device,
    parity_lane,
    parity_rng,
)

# Add the src root so pure-JAX simsopt modules resolve from this repo
# without reloading the entire simsopt package during test collection.
_REPO_SRC_ROOT = str(Path(__file__).resolve().parents[2] / "src")
if _REPO_SRC_ROOT not in sys.path:
    sys.path.insert(0, _REPO_SRC_ROOT)

from simsopt.field.biotsavart_jax import biot_savart_A
from simsopt.field.biotsavart import BiotSavart
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.field.coil import Current, coils_via_symmetries
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.surfaceobjectives import ToroidalFlux
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.label_constraints_jax import toroidal_flux_jax
from simsopt.geo.surface_fourier_jax import (
    surface_gamma_from_dofs,
    surface_gammadash2_from_dofs,
    stellsym_scatter_indices,
)

_MPOL = 1
_NTOR = 1
_NFP = 1
_NPHI = 15
_NTHETA = 16
_QP_PHI = jnp.linspace(0, 1, _NPHI, endpoint=False)
_QP_THETA = jnp.linspace(0, 1, _NTHETA, endpoint=False)
_TF_COIL_DOFS = jnp.array(
    [
        0.02,
        -0.03,
        0.01,
        -0.02,
        0.03,
        -0.01,
        0.04,
        0.02,
        -0.03,
        0.01,
        -0.04,
        0.03,
        -0.02,
        0.01,
    ],
    dtype=jnp.float64,
)


def _make_torus_dofs(R=1.0, r=0.1, mpol=1, ntor=1, nfp=1, stellsym=False):
    ncols = 2 * ntor + 1
    xc = np.zeros((2 * mpol + 1, ncols))
    yc = np.zeros((2 * mpol + 1, ncols))
    zc = np.zeros((2 * mpol + 1, ncols))
    xc[0, 0] = R
    xc[1, 0] = r
    zc[mpol + 1, 0] = r
    full = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])

    if stellsym:
        scatter_idx = stellsym_scatter_indices(mpol, ntor)
        return full[scatter_idx], scatter_idx
    return full.copy(), None


def _surface_slice_from_dofs(surface_dofs, stellsym, scatter_idx):
    gamma = surface_gamma_from_dofs(
        surface_dofs,
        _QP_PHI,
        _QP_THETA,
        _MPOL,
        _NTOR,
        _NFP,
        stellsym,
        scatter_idx,
    )
    gammadash2 = surface_gammadash2_from_dofs(
        surface_dofs,
        _QP_PHI,
        _QP_THETA,
        _MPOL,
        _NTOR,
        _NFP,
        stellsym,
        scatter_idx,
    )
    return gamma[0], gammadash2[0]


def _make_surface_dofs(stellsym):
    surface_dofs_np, scatter_idx = _make_torus_dofs(
        R=1.0,
        r=0.1,
        mpol=_MPOL,
        ntor=_NTOR,
        nfp=_NFP,
        stellsym=stellsym,
    )
    return jnp.array(surface_dofs_np), scatter_idx


def _make_tf_coils_from_dofs(
    dofs,
    *,
    n_coils=6,
    nquad=48,
):
    twopi = 2 * np.pi
    t = jnp.linspace(0.0, 1.0, nquad, endpoint=False)
    angle = twopi * t

    R_center = 1.0 + 0.04 * dofs[0]
    r_coil = 0.28 + 0.02 * dofs[1]
    phase_offsets = (
        twopi * (jnp.arange(n_coils) / n_coils) + 0.12 * dofs[2 : 2 + n_coils]
    )
    currents = 1e5 * (1.0 + 0.05 * dofs[2 + n_coils : 2 + 2 * n_coils])

    cos_angle = jnp.cos(angle)
    sin_angle = jnp.sin(angle)
    coil_R = R_center + r_coil * cos_angle
    dcoil_R = -r_coil * twopi * sin_angle
    coil_z = r_coil * sin_angle
    dcoil_z = r_coil * twopi * cos_angle

    cos_phi = jnp.cos(phase_offsets)[:, None]
    sin_phi = jnp.sin(phase_offsets)[:, None]

    gammas = jnp.stack(
        [
            coil_R[None, :] * cos_phi,
            coil_R[None, :] * sin_phi,
            jnp.broadcast_to(coil_z, (n_coils, nquad)),
        ],
        axis=-1,
    )
    gammadashs = jnp.stack(
        [
            dcoil_R[None, :] * cos_phi,
            dcoil_R[None, :] * sin_phi,
            jnp.broadcast_to(dcoil_z, (n_coils, nquad)),
        ],
        axis=-1,
    )
    return gammas, gammadashs, currents


def _make_object_level_toroidal_flux_case():
    ncoils = 2
    nfp = 1
    stellsym = False

    base_curves = create_equally_spaced_curves(
        ncoils,
        nfp,
        stellsym=stellsym,
        R0=1.0,
        R1=0.5,
        order=3,
    )
    base_currents = [Current(1e5) for _ in range(ncoils)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0, 19, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 21, endpoint=False),
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)
    return coils, surface


def _taylor_test_first_order(
    f, grad_fn, x, *, epsilons=None, direction=None, atol=1e-9
):
    rng = parity_rng(3)
    if direction is None:
        direction = jnp.array(rng.rand(*x.shape) - 0.5)
    if epsilons is None:
        epsilons = np.power(2.0, -np.arange(10, 20, dtype=float))

    df0 = float(jnp.dot(grad_fn(x), direction))
    err_old = 1e9
    for eps in epsilons:
        f_plus = float(f(x + eps * direction))
        f_minus = float(f(x - eps * direction))
        fd_est = (f_plus - f_minus) / (2 * eps)
        err = abs(fd_est - df0)
        assert err < max(atol, 0.35 * err_old), (
            f"Taylor convergence stalled: err={err:.2e}, "
            f"prev={err_old:.2e}, ratio={err / err_old:.3f}"
        )
        err_old = err


def _taylor_test_second_order(f, grad_fn, hess_fn, x, *, epsilons=None):
    rng = parity_rng(5)
    direction1 = jnp.array(rng.rand(*x.shape) - 0.5)
    direction2 = jnp.array(rng.rand(*x.shape) - 0.5)
    if epsilons is None:
        epsilons = np.power(2.0, -np.arange(7, 20, dtype=float))

    df0 = float(jnp.dot(grad_fn(x), direction1))
    hess = hess_fn(x)
    d2f0 = float(direction2 @ (hess @ direction1))

    err_old = 1e9
    for eps in epsilons:
        df_eps = float(jnp.dot(grad_fn(x + eps * direction2), direction1))
        err = abs((df_eps - df0) / eps - d2f0)
        assert err <= 0.56 * err_old, (
            f"Second-order Taylor convergence stalled: err={err:.2e}, "
            f"prev={err_old:.2e}, ratio={err / err_old:.3f}"
        )
        err_old = err


class TestToroidalFluxJAXTaylor:
    @pytest.mark.parametrize("stellsym", [False, True])
    def test_toroidal_flux_surface_hessian_taylor(self, stellsym):
        """Pure-JAX ToroidalFlux Hessian gate for surface DOFs."""
        surface_dofs, scatter_idx = _make_surface_dofs(stellsym)
        coil_gammas, coil_gammadashs, coil_currents = _make_tf_coils_from_dofs(
            _TF_COIL_DOFS
        )

        def flux(surface_dofs_inner):
            points, gammadash2 = _surface_slice_from_dofs(
                surface_dofs_inner,
                stellsym,
                scatter_idx,
            )
            A = biot_savart_A(points, coil_gammas, coil_gammadashs, coil_currents)
            return toroidal_flux_jax(A, gammadash2, _NTHETA)

        _taylor_test_second_order(
            flux,
            jax.grad(flux),
            jax.hessian(flux),
            surface_dofs,
        )

    @pytest.mark.parametrize("stellsym", [False, True])
    def test_toroidal_flux_coil_dofs_taylor(self, stellsym):
        """Pure-JAX ToroidalFlux gradient gate for a traceable TF coil family."""
        surface_dofs, scatter_idx = _make_surface_dofs(stellsym)
        points, gammadash2 = _surface_slice_from_dofs(
            surface_dofs, stellsym, scatter_idx
        )

        def flux(coil_dofs_inner):
            coil_gammas, coil_gammadashs, coil_currents = _make_tf_coils_from_dofs(
                coil_dofs_inner
            )
            A = biot_savart_A(points, coil_gammas, coil_gammadashs, coil_currents)
            return toroidal_flux_jax(A, gammadash2, _NTHETA)

        _taylor_test_first_order(
            flux,
            jax.grad(flux),
            _TF_COIL_DOFS,
        )


class TestToroidalFluxObjectParity:
    @pytest.fixture(autouse=True)
    def _strict_parity_lane(self, monkeypatch, request, parity_lane):
        enable_strict_parity_backend(monkeypatch, request, parity_lane)
        with parity_default_device(parity_lane):
            yield

    def test_value_parity(self, parity_lane):
        del parity_lane
        coils, surface = _make_object_level_toroidal_flux_case()

        tf_cpu = ToroidalFlux(surface, BiotSavart(coils))
        tf_jax = ToroidalFlux(surface, BiotSavartJAX(coils))

        np.testing.assert_allclose(tf_jax.J(), tf_cpu.J(), rtol=1e-10, atol=1e-12)

    def test_surface_gradient_parity(self, parity_lane):
        del parity_lane
        coils, surface = _make_object_level_toroidal_flux_case()

        tf_cpu = ToroidalFlux(surface, BiotSavart(coils))
        tf_jax = ToroidalFlux(surface, BiotSavartJAX(coils))

        np.testing.assert_allclose(
            tf_jax.dJ_by_dsurfacecoefficients(),
            tf_cpu.dJ_by_dsurfacecoefficients(),
            rtol=1e-9,
            atol=1e-11,
        )

    def test_surface_hessian_parity(self, parity_lane):
        del parity_lane
        coils, surface = _make_object_level_toroidal_flux_case()

        tf_cpu = ToroidalFlux(surface, BiotSavart(coils))
        tf_jax = ToroidalFlux(surface, BiotSavartJAX(coils))

        np.testing.assert_allclose(
            tf_jax.d2J_by_dsurfacecoefficientsdsurfacecoefficients(),
            tf_cpu.d2J_by_dsurfacecoefficientsdsurfacecoefficients(),
            rtol=1e-8,
            atol=1e-10,
        )

    def test_coil_gradient_parity(self, parity_lane):
        del parity_lane
        coils, surface = _make_object_level_toroidal_flux_case()

        tf_cpu = ToroidalFlux(surface, BiotSavart(coils))
        tf_jax = ToroidalFlux(surface, BiotSavartJAX(coils))

        deriv_jax = tf_jax.dJ_by_dcoils()
        deriv_cpu = tf_cpu.dJ_by_dcoils()
        for coil in coils:
            np.testing.assert_allclose(
                deriv_jax(coil),
                deriv_cpu(coil),
                rtol=1e-9,
                atol=1e-8,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
