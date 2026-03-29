"""Shared helpers for BoozerSurfaceJAX test modules."""

import importlib.util
import sys
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_SRC = Path(__file__).resolve().parents[2] / "src" / "simsopt"


def _ensure_package(pkg, path):
    if pkg in sys.modules:
        return
    try:
        __import__(pkg)
    except ImportError:
        module = types.ModuleType(pkg)
        module.__path__ = [str(path)]
        sys.modules[pkg] = module


def _load_and_register(module_fqn, relpath):
    spec = importlib.util.spec_from_file_location(module_fqn, str(_SRC / relpath))
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_fqn] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("simsopt", _SRC)
_ensure_package("simsopt.geo", _SRC / "geo")
_ensure_package("simsopt.field", _SRC / "field")
_ensure_package("simsopt.objectives", _SRC / "objectives")

_sf = _load_and_register("simsopt.geo.surface_fourier_jax", "geo/surface_fourier_jax.py")
_bs_jax = _load_and_register("simsopt.field.biotsavart_jax", "field/biotsavart_jax.py")
_bs_backend = _load_and_register(
    "simsopt.field.biotsavart_jax_backend", "field/biotsavart_jax_backend.py"
)
_obj_utils = _load_and_register("simsopt.objectives.utilities", "objectives/utilities.py")
_br = _load_and_register("simsopt.geo.boozer_residual_jax", "geo/boozer_residual_jax.py")
_lc = _load_and_register("simsopt.geo.label_constraints_jax", "geo/label_constraints_jax.py")
_opt = _load_and_register("simsopt.geo.optimizer_jax", "geo/optimizer_jax.py")
_bsj = _load_and_register("simsopt.geo.boozersurface_jax", "geo/boozersurface_jax.py")
_soj = _load_and_register(
    "simsopt.geo.surfaceobjectives_jax", "geo/surfaceobjectives_jax.py"
)

surface_gamma = _sf.surface_gamma
surface_gammadash1 = _sf.surface_gammadash1
surface_gammadash2 = _sf.surface_gammadash2
surface_normal = _sf.surface_normal
surface_volume = _sf.surface_volume
surface_area = _sf.surface_area
stellsym_scatter_indices = _sf.stellsym_scatter_indices
dofs_to_xyzc = _sf.dofs_to_xyzc
surface_gamma_from_dofs = _sf.surface_gamma_from_dofs

biot_savart_B = _bs_jax.biot_savart_B
biot_savart_A = _bs_jax.biot_savart_A
biot_savart_dA_by_dX = _bs_jax.biot_savart_dA_by_dX

boozer_residual_scalar = _br.boozer_residual_scalar
volume_jax = _lc.volume_jax
area_jax = _lc.area_jax
toroidal_flux_jax = _lc.toroidal_flux_jax
compute_G_from_currents = _lc.compute_G_from_currents

jax_minimize = _opt.jax_minimize
newton_polish = _opt.newton_polish
newton_exact = _opt.newton_exact
PRIVATE_OPTIMIZER_JAX_VERSION = _opt.PRIVATE_OPTIMIZER_JAX_VERSION

_boozer_penalty_objective = _bsj._boozer_penalty_objective
_boozer_exact_coil_vjp = _bsj._boozer_exact_coil_vjp
_boozer_ls_coil_vjp = _bsj._boozer_ls_coil_vjp
require_target_backend_x64 = _bsj.require_target_backend_x64
resolve_optimizer_backend_method = _bsj.resolve_optimizer_backend_method
BoozerSurfaceJAX = _bsj.BoozerSurfaceJAX

_ensure_solved_jax = _soj._ensure_solved
_resolved_boozer_G_jax = _soj._resolved_boozer_G


def _make_simple_torus_coeffs(R0=1.0, r=0.1, mpol=1, ntor=1, nfp=1):
    """Create coefficient matrices for a circular-cross-section torus."""
    shape = (2 * mpol + 1, 2 * ntor + 1)
    xc = np.zeros(shape)
    yc = np.zeros(shape)
    zc = np.zeros(shape)
    xc[0, 0] = R0
    xc[1, 0] = r
    zc[mpol + 1, 0] = r
    return xc, yc, zc


def _circular_coil_geometry(*, radius, z_offset, nquad):
    """Return circular-coil geometry arrays for a fixed radius and height."""
    phi = np.linspace(0, 2 * np.pi, nquad, endpoint=False)
    gamma = np.stack(
        [radius * np.cos(phi), radius * np.sin(phi), z_offset * np.ones_like(phi)],
        axis=-1,
    )
    gammadash = np.stack(
        [
            -radius * np.sin(phi) * 2 * np.pi,
            radius * np.cos(phi) * 2 * np.pi,
            np.zeros_like(phi),
        ],
        axis=-1,
    )
    return gamma, gammadash


def _make_circular_coil(R=1.0, z=0.0, nquad=128, current=1e5):
    """Create a single circular coil at fixed radius and height."""
    gamma, gammadash = _circular_coil_geometry(radius=R, z_offset=z, nquad=nquad)
    return (
        jnp.array(gamma[None]),
        jnp.array(gammadash[None]),
        jnp.array([current]),
    )


def _make_two_coils(nquad=128):
    """Two circular coils at z=±0.3 for a minimal field."""
    radius = 1.0

    coils = []
    for z_offset, current in [(0.3, 1e5), (-0.3, 1e5)]:
        gamma, gammadash = _circular_coil_geometry(
            radius=radius,
            z_offset=z_offset,
            nquad=nquad,
        )
        coils.append((gamma, gammadash, current))

    gammas = jnp.array(np.stack([coil[0] for coil in coils]))
    gammadashs = jnp.array(np.stack([coil[1] for coil in coils]))
    currents = jnp.array([coil[2] for coil in coils])
    return gammas, gammadashs, currents


def _simple_torus_geometry_values(*, R0, r, mpol, ntor, nfp, nphi, ntheta):
    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)
    gamma = surface_gamma(
        qphi,
        qtheta,
        jnp.array(xc),
        jnp.array(yc),
        jnp.array(zc),
        mpol,
        ntor,
        nfp,
    )
    normal = surface_normal(
        qphi,
        qtheta,
        jnp.array(xc),
        jnp.array(yc),
        jnp.array(zc),
        mpol,
        ntor,
        nfp,
    )
    return {
        "volume": float(surface_volume(gamma, normal)),
        "area": float(surface_area(normal)),
        "expected_volume": 2.0 * np.pi**2 * R0 * r**2,
        "expected_area": 4.0 * np.pi**2 * R0 * r,
    }


def _build_penalty_problem(
    *,
    nphi=8,
    ntheta=8,
    mpol=1,
    ntor=1,
    nfp=1,
    R0=1.0,
    r=0.1,
    label_type="volume",
    targetlabel=None,
    constraint_weight=1.0,
    optimize_G=True,
    weight_inv_modB=True,
):
    """Build the repeated torus/coil setup used by Boozer penalty tests."""
    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = jnp.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = jnp.linspace(0, 1.0, ntheta, endpoint=False)
    sdofs = jnp.concatenate(
        [
            jnp.array(xc).ravel(),
            jnp.array(yc).ravel(),
            jnp.array(zc).ravel(),
        ]
    )

    gammas, gammadashs, currents = _make_two_coils()
    G = float(compute_G_from_currents(currents))
    iota = 0.3
    if optimize_G:
        x = jnp.concatenate([sdofs, jnp.array([iota, G])])
    else:
        x = jnp.concatenate([sdofs, jnp.array([iota])])

    if targetlabel is None:
        if label_type == "volume":
            targetlabel = 2.0 * np.pi**2 * R0 * r**2
        elif label_type == "area":
            targetlabel = 4.0 * np.pi**2 * R0 * r
        else:
            raise ValueError("targetlabel must be provided for non-analytic label types.")

    coil_arrays = [(gammas, gammadashs, currents)]

    def objective(xx):
        return _boozer_penalty_objective(
            xx,
            coil_arrays,
            qphi,
            qtheta,
            mpol,
            ntor,
            nfp,
            False,
            None,
            targetlabel,
            constraint_weight,
            label_type,
            0,
            optimize_G,
            weight_inv_modB,
        )

    return {
        "R0": R0,
        "r": r,
        "x": x,
        "iota": iota,
        "G": G,
        "gammas": gammas,
        "gammadashs": gammadashs,
        "currents": currents,
        "coil_arrays": coil_arrays,
        "qphi": qphi,
        "qtheta": qtheta,
        "mpol": mpol,
        "ntor": ntor,
        "nfp": nfp,
        "targetlabel": targetlabel,
        "constraint_weight": constraint_weight,
        "optimize_G": optimize_G,
        "weight_inv_modB": weight_inv_modB,
        "objective": objective,
    }


def _successful_minimize_result(
    x0,
    *,
    nit=0,
    nfev=1,
    njev=1,
):
    return types.SimpleNamespace(
        x=jnp.asarray(x0),
        fun=0.0,
        jac=jnp.zeros_like(x0),
        nit=nit,
        nfev=nfev,
        njev=njev,
        success=True,
        status=0,
    )


def _successful_newton_polish_result(x0, *, nit=0):
    n = x0.shape[0]
    return {
        "x": x0,
        "fun": jnp.asarray(0.0),
        "grad": jnp.zeros_like(x0),
        "hessian": jnp.eye(n, dtype=x0.dtype),
        "nit": nit,
        "success": True,
    }


def _patch_newton_polish_runner(monkeypatch, fake_newton_polish):
    """Patch the centralized Newton-polish dispatch seam used by run_code()."""

    def fake_runner(
        self,
        method,
        obj_fn,
        x0,
        *,
        maxiter,
        tol,
        stab,
        progress_callback=None,
    ):
        del self, method
        return fake_newton_polish(
            obj_fn,
            x0,
            maxiter=maxiter,
            tol=tol,
            stab=stab,
            progress_callback=progress_callback,
        )

    monkeypatch.setattr(
        _bsj.BoozerSurfaceJAX,
        "_run_newton_polish_for_method",
        fake_runner,
    )


class _MockCurrent:
    """Minimal mock for coil current."""

    def __init__(self, value):
        self._value = value
        self.dofs = self

    def get_value(self):
        return self._value

    def all_fixed(self):
        return True


class _MockCurve:
    """Minimal mock for coil curve."""

    def __init__(self, gamma, gammadash):
        self._gamma = gamma
        self._gammadash = gammadash

    def gamma(self):
        return self._gamma

    def gammadash(self):
        return self._gammadash


class _MockCoil:
    def __init__(self, gamma, gammadash, current):
        self.curve = _MockCurve(gamma, gammadash)
        self.current = _MockCurrent(current)


class _MockBiotSavart(_bsj.Optimizable):
    """Minimal mock for BiotSavartJAX."""

    def __init__(self, coils):
        super().__init__(x0=np.asarray([]))
        self._coils = coils


class _MockSurface:
    """Minimal mock for SurfaceXYZTensorFourier."""

    def __init__(self, dofs, mpol, ntor, nfp, stellsym, qphi, qtheta):
        self._dofs = np.array(dofs, dtype=np.float64)
        self.mpol = mpol
        self.ntor = ntor
        self.nfp = nfp
        self.stellsym = stellsym
        self.quadpoints_phi = qphi
        self.quadpoints_theta = qtheta

    def get_dofs(self):
        return self._dofs.copy()

    def set_dofs(self, dofs):
        self._dofs = np.array(dofs, dtype=np.float64)

    def get_stellsym_mask(self):
        nphi = len(self.quadpoints_phi)
        ntheta = len(self.quadpoints_theta)
        return np.ones((nphi, ntheta), dtype=bool)


class _MockVolumeLabel:
    """Minimal mock for Volume label."""

    def J(self):
        return 0.0


def _make_mock_coils(nquad=64):
    """Create two mock coils at z=+/-0.3 for BoozerSurfaceJAX tests."""
    radius = 1.0
    coils = []
    for z_offset, current in [(0.3, 1e5), (-0.3, 1e5)]:
        gamma, gammadash = _circular_coil_geometry(
            radius=radius,
            z_offset=z_offset,
            nquad=nquad,
        )
        coils.append(_MockCoil(gamma, gammadash, current))
    return coils


def _make_mixed_quad_mock_coils():
    """Two coils with different quadrature counts."""
    radius = 1.0
    coils = []
    for z_offset, current, nquad in [(0.3, 1e5, 64), (-0.3, 1e5, 128)]:
        gamma, gammadash = _circular_coil_geometry(
            radius=radius,
            z_offset=z_offset,
            nquad=nquad,
        )
        coils.append(_MockCoil(gamma, gammadash, current))
    return coils


def _make_mock_boozer_surface(
    nphi=8,
    ntheta=8,
    mpol=1,
    ntor=1,
    nfp=1,
    *,
    stellsym=False,
):
    """Build a BoozerSurfaceJAX from mock objects."""
    R0, r = 1.0, 0.1
    xc, yc, zc = _make_simple_torus_coeffs(R0, r, mpol, ntor, nfp)
    qphi = np.linspace(0, 1.0 / nfp, nphi, endpoint=False)
    qtheta = np.linspace(0, 1.0, ntheta, endpoint=False)
    full_sdofs = np.concatenate([xc.ravel(), yc.ravel(), zc.ravel()])
    if stellsym:
        scatter = np.asarray(stellsym_scatter_indices(mpol, ntor), dtype=np.int32)
        sdofs = full_sdofs[scatter]
    else:
        sdofs = full_sdofs

    biot_savart = _MockBiotSavart(_make_mock_coils())
    surface = _MockSurface(sdofs, mpol, ntor, nfp, stellsym, qphi, qtheta)
    label = _MockVolumeLabel()
    target = 2.0 * np.pi**2 * R0 * r**2

    return BoozerSurfaceJAX(
        biot_savart,
        surface,
        label,
        target,
        constraint_weight=1.0,
    )
