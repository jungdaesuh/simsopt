"""Shared helpers for BoozerSurfaceJAX test modules."""

import inspect
import sys
from dataclasses import dataclass
from pathlib import Path
import types

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

jax.config.update("jax_enable_x64", True)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
_REPO_ROOT_STR = str(_REPO_ROOT)
if _REPO_ROOT_STR not in sys.path:
    sys.path.insert(0, _REPO_ROOT_STR)

from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(_SRC_ROOT)

import simsopt.field.biotsavart_jax as _bs_jax
import simsopt.field.biotsavart_jax_backend as _bs_backend
import simsopt.geo.boozer_residual_jax as _br
import simsopt.geo.boozersurface_jax as _bsj
import simsopt.geo.label_constraints_jax as _lc
import simsopt.geo.optimizer_jax as _opt
import simsopt.geo.surface_fourier_jax as _sf
import simsopt.geo.surfaceobjectives_jax as _soj

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
jax_least_squares = _opt.jax_least_squares
newton_polish = _opt.newton_polish
newton_exact = _opt.newton_exact
PRIVATE_OPTIMIZER_JAX_VERSION = _opt.PRIVATE_OPTIMIZER_JAX_VERSION

_boozer_penalty_objective = _bsj._boozer_penalty_objective
_boozer_exact_coil_vjp = _bsj._boozer_exact_coil_vjp
_boozer_ls_coil_vjp = _bsj._boozer_ls_coil_vjp
require_target_backend_x64 = _bsj.require_target_backend_x64
resolve_least_squares_optimizer_method = _opt.resolve_least_squares_optimizer_method
resolve_optimizer_backend_method = _opt.resolve_optimizer_backend_method
BoozerSurfaceJAX = _bsj.BoozerSurfaceJAX
BiotSavartJAX = _bs_backend.BiotSavartJAX

_ensure_solved_jax = _soj._ensure_solved


UPSTREAM_BOOZER_SURFACE_TYPES = (
    "SurfaceXYZFourier",
    "SurfaceXYZTensorFourier",
)
UPSTREAM_BOOZER_STELLSYM = (True, False)
UPSTREAM_BOOZER_OPTIMIZE_G = (True, False)

_UPSTREAM_BOOZER_CONSTRAINT_WEIGHT = 11.1232
_UPSTREAM_BOOZER_TF_TARGET = 0.1
_UPSTREAM_BOOZER_IOTA0 = -0.3
_UPSTREAM_EXACT_IOTA0 = -0.44856192
_UPSTREAM_EXACT_TF_TARGET = 0.41431152


@dataclass(frozen=True)
class UpstreamBoozerPenaltyCase:
    surfacetype: str
    stellsym: bool
    optimize_G: bool
    cpu_boozer: object
    jax_boozer: BoozerSurfaceJAX
    x: np.ndarray
    constraint_weight: float


@dataclass(frozen=True)
class UpstreamExactSurfaceCase:
    surfacetype: str
    jax_boozer: BoozerSurfaceJAX
    initial_iota: float
    initial_G: float


def _upstream_initial_G(current_values, nfp):
    current_sum = nfp * sum(abs(current) for current in current_values)
    return 2.0 * np.pi * current_sum * (4.0 * np.pi * 1e-7 / (2.0 * np.pi))


def _current_values(currents):
    return [current.get_value() for current in currents]


def _build_upstream_penalty_decision_vector(
    surface_dofs, current_values, nfp, *, optimize_G
):
    x = np.concatenate([surface_dofs, [_UPSTREAM_BOOZER_IOTA0]])
    if not optimize_G:
        return x
    return np.concatenate([x, [_upstream_initial_G(current_values, nfp)]])


def _make_toroidal_flux_label(surface, coils):
    from simsopt.field import BiotSavart
    from simsopt.geo import ToroidalFlux

    return ToroidalFlux(surface, BiotSavart(coils), nphi=51, ntheta=51)


def _build_upstream_jax_boozer(surface, bs, label, target, *, constraint_weight=None):
    return BoozerSurfaceJAX(
        BiotSavartJAX(bs.coils),
        surface,
        label,
        target,
        constraint_weight=constraint_weight,
    )


def _build_upstream_boozer_pair(bs, surface, label, target, *, constraint_weight=None):
    from simsopt.geo import BoozerSurface

    return (
        BoozerSurface(bs, surface, label, target),
        _build_upstream_jax_boozer(
            surface,
            bs,
            label,
            target,
            constraint_weight=constraint_weight,
        ),
    )


def _build_upstream_ncsx_surface_context(surfacetype, stellsym):
    from simsopt.configs.zoo import get_data

    from .surface_test_helpers import get_surface

    _, base_currents, magnetic_axis, nfp, bs = get_data("ncsx")
    surface = get_surface(surfacetype, stellsym)
    surface.fit_to_curve(magnetic_axis, 0.1)
    return {
        "base_currents": base_currents,
        "nfp": nfp,
        "bs": bs,
        "surface": surface,
        "label": _make_toroidal_flux_label(surface, bs.coils),
    }


def _build_upstream_exact_surface_case(surfacetype):
    from simsopt.configs.zoo import get_data

    from .surface_test_helpers import get_exact_surface

    _, base_currents, _, nfp, bs = get_data("ncsx")
    current_values = _current_values(base_currents)
    surface = get_exact_surface(surface_type=surfacetype)
    label = _make_toroidal_flux_label(surface, bs.coils)
    _, jax_boozer = _build_upstream_boozer_pair(
        bs,
        surface,
        label,
        _UPSTREAM_EXACT_TF_TARGET,
    )
    return UpstreamExactSurfaceCase(
        surfacetype=surfacetype,
        jax_boozer=jax_boozer,
        initial_iota=_UPSTREAM_EXACT_IOTA0,
        initial_G=_upstream_initial_G(current_values, nfp),
    )


def _build_upstream_boozer_penalty_case(surfacetype, stellsym, optimize_G):
    case_data = _build_upstream_ncsx_surface_context(surfacetype, stellsym)
    base_currents = case_data["base_currents"]
    current_values = _current_values(base_currents)
    bs = case_data["bs"]
    surface = case_data["surface"]
    label = case_data["label"]

    cpu_boozer, jax_boozer = _build_upstream_boozer_pair(
        bs,
        surface,
        label,
        _UPSTREAM_BOOZER_TF_TARGET,
        constraint_weight=_UPSTREAM_BOOZER_CONSTRAINT_WEIGHT,
    )

    x = _build_upstream_penalty_decision_vector(
        surface.get_dofs(),
        current_values,
        case_data["nfp"],
        optimize_G=optimize_G,
    )

    return UpstreamBoozerPenaltyCase(
        surfacetype=surfacetype,
        stellsym=stellsym,
        optimize_G=optimize_G,
        cpu_boozer=cpu_boozer,
        jax_boozer=jax_boozer,
        x=x,
        constraint_weight=_UPSTREAM_BOOZER_CONSTRAINT_WEIGHT,
    )


def _evaluate_upstream_boozer_penalty_case(case):
    cpu_value, cpu_gradient = case.cpu_boozer.boozer_penalty_constraints_vectorized(
        case.x,
        derivatives=1,
        constraint_weight=case.constraint_weight,
        optimize_G=case.optimize_G,
    )
    jax_objective = case.jax_boozer._make_penalty_objective_with(
        case.optimize_G,
        case.jax_boozer.options["weight_inv_modB"],
        case.constraint_weight,
    )
    jax_value, jax_gradient = jax.value_and_grad(jax_objective)(jnp.asarray(case.x))
    return {
        "cpu_value": float(cpu_value),
        "cpu_gradient": np.asarray(cpu_gradient),
        "jax_value": float(jax_value),
        "jax_gradient": np.asarray(jax_gradient),
    }


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
    stellsym=False,
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
    full_sdofs = jnp.concatenate(
        [
            jnp.array(xc).ravel(),
            jnp.array(yc).ravel(),
            jnp.array(zc).ravel(),
        ]
    )

    if stellsym:
        scatter = jnp.asarray(stellsym_scatter_indices(mpol, ntor), dtype=jnp.int32)
        sdofs = full_sdofs[scatter]
    else:
        scatter = None
        sdofs = full_sdofs

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
            raise ValueError(
                "targetlabel must be provided for non-analytic label types."
            )

    coil_arrays = [(gammas, gammadashs, currents)]

    def objective(xx):
        return _boozer_penalty_objective(
            xx,
            coil_arrays=coil_arrays,
            quadpoints_phi=qphi,
            quadpoints_theta=qtheta,
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            scatter_indices=scatter,
            surface_kind="generic",
            targetlabel=targetlabel,
            constraint_weight=constraint_weight,
            label_type=label_type,
            phi_idx=0,
            optimize_G=optimize_G,
            weight_inv_modB=weight_inv_modB,
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
    flat_x0, _ = ravel_pytree(x0)
    return types.SimpleNamespace(
        x=x0,
        fun=0.0,
        jac=jnp.zeros_like(flat_x0),
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
    supports_objective_args = (
        "objective_args" in inspect.signature(fake_newton_polish).parameters
    )
    supports_materialize_hessian = (
        "materialize_hessian" in inspect.signature(fake_newton_polish).parameters
    )
    supports_max_dense_hessian_bytes = (
        "max_dense_hessian_bytes" in inspect.signature(fake_newton_polish).parameters
    )

    def fake_runner(
        self,
        method,
        obj_fn,
        x0,
        *,
        maxiter,
        tol,
        stab,
        materialize_hessian=True,
        max_dense_hessian_bytes=None,
        progress_callback=None,
        objective_args=(),
    ):
        del self, method
        kwargs = {
            "maxiter": maxiter,
            "tol": tol,
            "stab": stab,
            "progress_callback": progress_callback,
        }
        if supports_materialize_hessian:
            kwargs["materialize_hessian"] = materialize_hessian
        if supports_max_dense_hessian_bytes:
            kwargs["max_dense_hessian_bytes"] = max_dense_hessian_bytes
        if supports_objective_args:
            kwargs["objective_args"] = objective_args
        return fake_newton_polish(obj_fn, x0, **kwargs)

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
        self._coil_spec = _bs_backend.grouped_coil_set_spec_from_lists(
            [coil.curve.gamma() for coil in coils],
            [coil.curve.gammadash() for coil in coils],
            [coil.current.get_value() for coil in coils],
        )

    def coil_set_spec(self):
        return self._coil_spec


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
