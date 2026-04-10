"""Mirrored CPU/JAX parity tests for SquaredFluxJAX."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import enable_strict_parity_backend, parity_default_device, parity_rng

from simsopt._core.derivative import Derivative
from simsopt._core.optimizable import Optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.fluxobjective_jax import SquaredFluxJAX

_SQUARED_FLUX_DEFINITIONS = (
    "quadratic flux",
    "normalized",
    "local",
)
_VALUE_RTOL = 1e-12
_VALUE_ATOL = 1e-15
_GRADIENT_RTOL = 1e-11
_GRADIENT_ATOL = 1e-14
_FALLBACK_GRADIENT_RTOL = 1e-12
_FALLBACK_GRADIENT_ATOL = 1e-15


class _FluxObjectiveFakeSurface:
    def __init__(self, normal):
        self._normal = np.asarray(normal, dtype=np.float64)

    def gamma(self):
        return np.zeros_like(self._normal)

    def normal(self):
        return self._normal


class _FluxObjectiveFakeField(Optimizable):
    def __init__(self, B, *, supports_fallback):
        self._B = np.asarray(B, dtype=np.float64)
        self._jax_native = False
        self._supports_fallback = bool(supports_fallback)
        self._points = None
        super().__init__(x0=np.zeros(self._B.size, dtype=np.float64))

    def recompute_bell(self, parent=None):
        del parent

    def set_points(self, xyz):
        self._points = np.asarray(xyz, dtype=np.float64)

    def set_points_from_spec(self, field_eval_spec):
        self._points = np.asarray(field_eval_spec.points, dtype=np.float64)

    def supports_jax_objective_fallback(self):
        return self._supports_fallback

    def B(self):
        return jnp.asarray(self._B.reshape((-1, 3)), dtype=jnp.float64)

    def B_vjp(self, dJdB):
        return Derivative({self: np.asarray(dJdB, dtype=np.float64).reshape((-1,))})


def _make_fake_flux_objective(*, definition, normal, B, target, supports_fallback):
    surface = _FluxObjectiveFakeSurface(normal)
    field = _FluxObjectiveFakeField(B, supports_fallback=supports_fallback)
    target_array = np.asarray(target)
    objective = SquaredFlux(surface, field, target=target_array, definition=definition)
    objective_jax = SquaredFluxJAX(
        surface,
        field,
        target=target_array,
        definition=definition,
    )
    return objective, objective_jax, field


def _finite_fallback_flux_case(definition):
    return _make_fake_flux_objective(
        definition=definition,
        normal=[[[3.0, 0.0, 4.0], [0.0, 2.0, 0.0]]],
        B=[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
        target=[[0.25, -0.5]],
        supports_fallback=True,
    )


def _make_native_flux_parity_case():
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
        quadpoints_phi=np.linspace(0.0, 1.0, 32, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 32, endpoint=False),
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.2)
    surface.set_zs(1, 0, 0.2)
    surface.fix_all()
    return coils, surface


def _make_native_flux_objectives(definition, *, target=None):
    coils, surface = _make_native_flux_parity_case()

    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(surface.gamma().reshape((-1, 3)))
    objective_cpu = SquaredFlux(surface, bs_cpu, target=target, definition=definition)

    bs_jax = BiotSavartJAX(coils)
    objective_jax = SquaredFluxJAX(
        surface,
        bs_jax,
        target=target,
        definition=definition,
        parity_mode="native_only",
    )
    return objective_cpu, objective_jax


def _degenerate_point_quadrature_scale(definition):
    if definition in {"quadratic flux", "local"}:
        return 0.5
    return 1.0


def _assert_flux_value_parity(actual, reference):
    np.testing.assert_allclose(actual, reference, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)


def _assert_flux_gradient_parity(actual, reference):
    np.testing.assert_allclose(
        actual,
        reference,
        rtol=_GRADIENT_RTOL,
        atol=_GRADIENT_ATOL,
    )


def _assert_fallback_flux_gradient_parity(actual, reference):
    np.testing.assert_allclose(
        actual,
        reference,
        rtol=_FALLBACK_GRADIENT_RTOL,
        atol=_FALLBACK_GRADIENT_ATOL,
    )


@pytest.fixture(autouse=True)
def _strict_parity_lane(monkeypatch, request, parity_lane):
    enable_strict_parity_backend(monkeypatch, request, parity_lane)
    with parity_default_device(parity_lane):
        yield


@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_fluxobjective_value_parity(definition):
    objective_cpu, objective_jax = _make_native_flux_objectives(definition)

    assert objective_jax._use_jax_native
    _assert_flux_value_parity(objective_jax.J(), objective_cpu.J())


@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_fluxobjective_gradient_parity(definition):
    objective_cpu, objective_jax = _make_native_flux_objectives(definition)
    _assert_flux_gradient_parity(objective_jax.dJ(), objective_cpu.dJ())


@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_fallback_fluxobjective_value_and_gradient_parity(definition):
    objective_cpu, objective_jax, _unused_field = _finite_fallback_flux_case(definition)

    assert not objective_jax._use_jax_native
    assert objective_jax._uses_jax_objective_fallback
    _assert_flux_value_parity(objective_jax.J(), objective_cpu.J())
    _assert_fallback_flux_gradient_parity(objective_jax.dJ(), objective_cpu.dJ())


def test_fluxobjective_target_parity():
    _, surface = _make_native_flux_parity_case()
    rng = parity_rng(11)
    target = rng.standard_normal(surface.normal().shape[:2]) * 1e-2

    objective_cpu, objective_jax = _make_native_flux_objectives(
        "quadratic flux",
        target=target,
    )
    _assert_flux_value_parity(objective_jax.J(), objective_cpu.J())
    _assert_flux_gradient_parity(objective_jax.dJ(), objective_cpu.dJ())


def test_quadratic_flux_zero_normals_contract():
    objective_cpu, objective_jax, field = _make_fake_flux_objective(
        definition="quadratic flux",
        normal=np.zeros((1, 1, 3)),
        B=np.zeros((1, 1, 3)),
        target=[[1.0]],
        supports_fallback=True,
    )

    assert np.isnan(objective_cpu.J())
    np.testing.assert_allclose(objective_jax.J(), 0.0, atol=0.0)
    np.testing.assert_allclose(objective_cpu.dJ(), np.zeros(field.local_dof_size))
    np.testing.assert_allclose(objective_jax.dJ(), np.zeros(field.local_dof_size))


@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_degenerate_normals_do_not_perturb_valid_flux_contracts(definition):
    full_objective_cpu, full_objective_jax, _field = _make_fake_flux_objective(
        definition=definition,
        normal=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        B=[[[7.0, -3.0, 2.0], [2.0, 0.0, 0.0]]],
        target=[[11.0, 0.5]],
        supports_fallback=True,
    )
    reduced_objective_cpu, _reduced_objective_jax, _reduced_field = _make_fake_flux_objective(
        definition=definition,
        normal=[[[1.0, 0.0, 0.0]]],
        B=[[[2.0, 0.0, 0.0]]],
        target=[[0.5]],
        supports_fallback=True,
    )

    quadrature_scale = _degenerate_point_quadrature_scale(definition)
    expected_value = quadrature_scale * reduced_objective_cpu.J()
    expected_gradient = np.concatenate(
        (np.zeros(3), quadrature_scale * reduced_objective_cpu.dJ())
    )

    np.testing.assert_allclose(
        full_objective_jax.J(), expected_value, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        full_objective_cpu.dJ(), expected_gradient, rtol=1e-12, atol=1e-15
    )
    np.testing.assert_allclose(
        full_objective_jax.dJ(), expected_gradient, rtol=1e-12, atol=1e-15
    )


@pytest.mark.parametrize("definition", ("normalized", "local"))
def test_singular_zero_field_contract(definition):
    objective_cpu, objective_jax, _field = _make_fake_flux_objective(
        definition=definition,
        normal=[[[1.0, 0.0, 0.0]]],
        B=[[[0.0, 0.0, 0.0]]],
        target=[[1.0]],
        supports_fallback=True,
    )

    assert np.isinf(objective_cpu.J())
    assert np.isinf(objective_jax.J())
    with pytest.raises(ObjectiveFailure, match="gradient is singular"):
        objective_cpu.dJ()
    with pytest.raises(ObjectiveFailure, match="gradient is singular"):
        objective_jax.dJ()


def test_native_only_mode_rejects_fallback_seams():
    surface = _FluxObjectiveFakeSurface([[[1.0, 0.0, 0.0]]])
    field = _FluxObjectiveFakeField([[[0.0, 0.0, 0.0]]], supports_fallback=True)

    with pytest.raises(RuntimeError, match="parity_mode='native_only'.*fallback seam"):
        SquaredFluxJAX(
            surface,
            field,
            target=np.asarray([[0.0]], dtype=np.float64),
            parity_mode="native_only",
        )
