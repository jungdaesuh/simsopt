"""
Pure-JAX label constraint parity tests matching upstream C++ test suite.

Section 8b of jax_gpu_remaining_todos.md.  These tests exercise
``toroidal_flux_jax``, ``volume_jax`` (``surface_volume``), and
``area_jax`` (``surface_area``) gradient correctness via central
finite differences, matching the upstream tests in
``tests/geo/test_surface_objectives.py``.

Tests:
1. Toroidal flux invariance across phi slices
2. Toroidal flux gradient FD (Taylor test w.r.t. surface DOFs)
3. Volume gradient FD (Taylor test w.r.t. surface DOFs)
4. Area gradient FD (Taylor test w.r.t. surface DOFs)

No simsoptpp dependency — all tests use pure JAX functions.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import numpy as np

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

# ---------------------------------------------------------------------------
# Load JAX modules directly (avoids simsopt/__init__.py → simsoptpp dep).
# label_constraints_jax.py uses relative imports, so we temporarily inject
# stub packages into sys.modules, load everything, then clean up.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parents[2] / "src" / "simsopt"


def _load(name, relpath, package=None):
    spec = importlib.util.spec_from_file_location(name, str(_SRC / relpath))
    mod = importlib.util.module_from_spec(spec)
    if package is not None:
        mod.__package__ = package
    spec.loader.exec_module(mod)
    return mod


# label_constraints_jax.py has top-level relative imports that need stub
# parent packages in sys.modules and __package__ set on the module.
_stubs_added = []
for _pkg in ["simsopt", "simsopt.geo", "simsopt.field"]:
    if _pkg not in sys.modules:
        _stub = types.ModuleType(_pkg)
        _stub.__path__ = []
        sys.modules[_pkg] = _stub
        _stubs_added.append(_pkg)

_sf = _load("surface_fourier_jax", "geo/surface_fourier_jax.py")
sys.modules["simsopt.geo.surface_fourier_jax"] = _sf

_bs = _load("biotsavart_jax", "field/biotsavart_jax.py")
sys.modules["simsopt.field.biotsavart_jax"] = _bs

_lc = _load(
    "label_constraints_jax", "geo/label_constraints_jax.py", package="simsopt.geo"
)

# Clean up stubs so the real simsopt package isn't shadowed for other tests.
for _entry in ["simsopt.geo.surface_fourier_jax", "simsopt.field.biotsavart_jax"]:
    sys.modules.pop(_entry, None)
for _pkg in reversed(_stubs_added):
    sys.modules.pop(_pkg, None)

surface_gamma_from_dofs = _sf.surface_gamma_from_dofs
surface_gammadash2_from_dofs = _sf.surface_gammadash2_from_dofs
surface_normal_from_dofs = _sf.surface_normal_from_dofs
surface_volume = _sf.surface_volume
surface_area = _sf.surface_area
stellsym_scatter_indices = _sf.stellsym_scatter_indices

biot_savart_A = _bs.biot_savart_A
toroidal_flux_jax = _lc.toroidal_flux_jax


def _make_tf_coils(n_coils=16, R_center=1.0, r_coil=0.3, nquad=64, I=1e5):
    """Create toroidal-field coils evenly distributed in toroidal angle.

    Each coil is a circle in its poloidal plane that encircles the torus
    cross-section, producing a toroidal magnetic field inside the torus.

    Returns (gammas, gammadashs, currents).
    """
    twopi = 2 * np.pi
    gammas_list = []
    gds_list = []
    for k in range(n_coils):
        phi_k = twopi * k / n_coils
        t = np.linspace(0, 1, nquad, endpoint=False)
        R_t = R_center + r_coil * np.cos(twopi * t)
        z_t = r_coil * np.sin(twopi * t)
        gamma = np.stack([R_t * np.cos(phi_k), R_t * np.sin(phi_k), z_t], axis=-1)
        dR = -r_coil * twopi * np.sin(twopi * t)
        dz = r_coil * twopi * np.cos(twopi * t)
        gd = np.stack([dR * np.cos(phi_k), dR * np.sin(phi_k), dz], axis=-1)
        gammas_list.append(gamma)
        gds_list.append(gd)

    gammas = jnp.array(np.stack(gammas_list))  # (n_coils, nquad, 3)
    gds = jnp.array(np.stack(gds_list))
    currents = jnp.array([I] * n_coils)
    return gammas, gds, currents


def _make_torus_dofs(R=1.0, r=0.1, mpol=1, ntor=1, nfp=1, stellsym=False):
    """Create surface DOFs for a near-circular-cross-section torus.

    Returns (dofs, scatter_indices_or_None).
    """
    ncols = 2 * ntor + 1

    # Build full coefficient matrices
    xc = np.zeros((2 * mpol + 1, ncols))
    yc = np.zeros((2 * mpol + 1, ncols))  # all-zero by symmetry; needed for DOF layout
    zc = np.zeros((2 * mpol + 1, ncols))
    xc[0, 0] = R  # constant × constant
    xc[1, 0] = r  # cos(θ) × constant
    zc[mpol + 1, 0] = r  # sin(θ) × constant

    full = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])

    if stellsym:
        scatter_idx = stellsym_scatter_indices(mpol, ntor)
        dofs = full[scatter_idx]
        return dofs, scatter_idx
    return full.copy(), None


def _taylor_test_central(
    f, grad_fn, x, epsilons=None, direction=None, atol=1e-9, rate=0.35
):
    """Central-FD Taylor test: |(f(x+εd) − f(x−εd))/(2ε) − df·d| → 0.

    Central FD is O(ε²), so error should quarter when ε halves.
    """
    rng = np.random.RandomState(1)
    if direction is None:
        direction = jnp.array(rng.rand(*x.shape) - 0.5)
    if epsilons is None:
        epsilons = np.power(2.0, -np.arange(10, 20, dtype=float))

    dfx = float(jnp.sum(grad_fn(x) * direction))

    err_old = 1e9
    for eps in epsilons:
        fp = float(f(x + eps * direction))
        fm = float(f(x - eps * direction))
        fd_est = (fp - fm) / (2 * eps)
        err = abs(fd_est - dfx)
        assert err < max(atol, rate * err_old), (
            f"Taylor convergence stalled: err={err:.2e}, "
            f"prev={err_old:.2e}, ratio={err / err_old:.3f}"
        )
        err_old = err


# ---------------------------------------------------------------------------
# Shared surface + field configuration
# ---------------------------------------------------------------------------

_MPOL = 1
_NTOR = 1
_NFP = 1
_NPHI = 15
_NTHETA = 16
_QP_PHI = jnp.linspace(0, 1, _NPHI, endpoint=False)
_QP_THETA = jnp.linspace(0, 1, _NTHETA, endpoint=False)

_TF_GAMMAS, _TF_GDS, _TF_CURRENTS = _make_tf_coils(
    n_coils=16, R_center=1.0, r_coil=0.3, nquad=64, I=1e5
)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestLabelConstraintsParity:
    """Parity tests matching upstream tests/geo/test_surface_objectives.py."""

    def test_toroidal_flux_invariance(self):
        """Toroidal flux is approximately constant across phi slices.

        Matches ``test_toroidal_flux_is_constant``.
        Uses 48 evenly-spaced TF coils for sub-1% field ripple
        (ripple ~ exp(-π r_coil N / (2π R)) ≈ 0.07% for N=48).
        """
        mpol, ntor, nfp = 2, 1, 1
        nphi, ntheta = 30, 32
        qp_phi = jnp.linspace(0, 1, nphi, endpoint=False)
        qp_theta = jnp.linspace(0, 1, ntheta, endpoint=False)

        dofs, _ = _make_torus_dofs(
            R=1.0, r=0.1, mpol=mpol, ntor=ntor, nfp=nfp, stellsym=False
        )
        dofs = jnp.array(dofs)

        gamma = surface_gamma_from_dofs(
            dofs, qp_phi, qp_theta, mpol, ntor, nfp, stellsym=False
        )
        gd2 = surface_gammadash2_from_dofs(
            dofs, qp_phi, qp_theta, mpol, ntor, nfp, stellsym=False
        )

        # Dense TF coil set for smooth toroidal field
        tf_g, tf_gd, tf_I = _make_tf_coils(
            n_coils=48, R_center=1.0, r_coil=0.3, nquad=32, I=1e5
        )

        A_all = biot_savart_A(gamma.reshape(-1, 3), tf_g, tf_gd, tf_I).reshape(
            nphi, ntheta, 3
        )
        tf_list = np.array(
            [float(toroidal_flux_jax(A_all[i], gd2[i], ntheta)) for i in range(nphi)]
        )

        mean_tf = np.mean(tf_list)
        assert abs(mean_tf) > 1e-12, "Toroidal flux is zero — bad test config"
        max_err = np.max(np.abs(mean_tf - tf_list)) / abs(mean_tf)
        assert max_err < 1e-2, f"Toroidal flux varies {max_err:.4f} across phi"

    @pytest.mark.parametrize("stellsym", [False, True])
    def test_toroidal_flux_gradient_fd(self, stellsym):
        """dΦ_tor/d(surface DOFs) matches central finite differences.

        Matches ``test_toroidal_flux_first_derivative``.
        """
        dofs_np, scatter_idx = _make_torus_dofs(
            R=1.0, r=0.1, mpol=_MPOL, ntor=_NTOR, nfp=_NFP, stellsym=stellsym
        )
        dofs = jnp.array(dofs_np)

        def f(d):
            gamma = surface_gamma_from_dofs(
                d, _QP_PHI, _QP_THETA, _MPOL, _NTOR, _NFP, stellsym, scatter_idx
            )
            gd2 = surface_gammadash2_from_dofs(
                d, _QP_PHI, _QP_THETA, _MPOL, _NTOR, _NFP, stellsym, scatter_idx
            )
            pts = gamma[0]  # phi slice 0
            A = biot_savart_A(pts, _TF_GAMMAS, _TF_GDS, _TF_CURRENTS)
            return toroidal_flux_jax(A, gd2[0], _NTHETA)

        grad_fn = jax.grad(f)
        _taylor_test_central(f, grad_fn, dofs)

    @pytest.mark.parametrize("stellsym", [False, True])
    def test_volume_gradient_fd(self, stellsym):
        """dVolume/d(surface DOFs) matches central finite differences.

        Matches ``test_parameter_derivatives_volume``.
        """
        dofs_np, scatter_idx = _make_torus_dofs(
            R=1.0, r=0.1, mpol=_MPOL, ntor=_NTOR, nfp=_NFP, stellsym=stellsym
        )
        dofs = jnp.array(dofs_np)

        def f(d):
            gamma = surface_gamma_from_dofs(
                d, _QP_PHI, _QP_THETA, _MPOL, _NTOR, _NFP, stellsym, scatter_idx
            )
            normal = surface_normal_from_dofs(
                d, _QP_PHI, _QP_THETA, _MPOL, _NTOR, _NFP, stellsym, scatter_idx
            )
            return surface_volume(gamma, normal)

        grad_fn = jax.grad(f)
        _taylor_test_central(f, grad_fn, dofs)

    @pytest.mark.parametrize("stellsym", [False, True])
    def test_area_gradient_fd(self, stellsym):
        """dArea/d(surface DOFs) matches central finite differences.

        Matches ``test_label_surface_derivative1(Area)``.
        """
        dofs_np, scatter_idx = _make_torus_dofs(
            R=1.0, r=0.1, mpol=_MPOL, ntor=_NTOR, nfp=_NFP, stellsym=stellsym
        )
        dofs = jnp.array(dofs_np)

        def f(d):
            normal = surface_normal_from_dofs(
                d, _QP_PHI, _QP_THETA, _MPOL, _NTOR, _NFP, stellsym, scatter_idx
            )
            return surface_area(normal)

        grad_fn = jax.grad(f)
        _taylor_test_central(f, grad_fn, dofs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
