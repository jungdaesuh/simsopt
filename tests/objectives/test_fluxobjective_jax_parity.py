"""Mirrored CPU/JAX parity tests for SquaredFluxJAX."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import enable_strict_parity_backend, parity_default_device, parity_rng

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt._core.optimizable import Optimizable
from simsopt.backend import invalidate_backend_cache
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.jax_core import (
    fixed_surface_flux_integral_from_B,
    make_fixed_surface_flux_spec,
)
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
_FD_GRADIENT_TOLS = parity_ladder_tolerances("fd-gradient")


class _NonNativeFakeField(Optimizable):
    def __init__(self):
        self._uses_uniform_curve_xyz_fourier_fastpath = False
        self._points = None
        super().__init__(x0=np.zeros(1, dtype=np.float64))

    def recompute_bell(self, parent=None):
        del parent

    def set_points_from_spec(self, field_eval_spec):
        self._points = np.asarray(field_eval_spec.points, dtype=np.float64)


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


def _make_large_grouped_flux_objective():
    nfp = 1
    stellsym = False
    base_curves = [
        *create_equally_spaced_curves(
            2,
            nfp,
            stellsym=stellsym,
            R0=1.0,
            R1=0.45,
            order=3,
            numquadpoints=24,
        ),
        *create_equally_spaced_curves(
            1,
            nfp,
            stellsym=stellsym,
            R0=1.08,
            R1=0.38,
            order=4,
            numquadpoints=28,
        ),
    ]
    base_currents = [Current(value) for value in (1.0e5, -0.8e5, 0.6e5)]
    coils = coils_via_symmetries(base_curves, base_currents, nfp, stellsym)

    surface = SurfaceRZFourier(
        nfp=nfp,
        stellsym=stellsym,
        mpol=1,
        ntor=1,
        quadpoints_phi=np.linspace(0.0, 1.0, 48, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 48, endpoint=False),
    )
    surface.set_rc(0, 0, 1.0)
    surface.set_rc(1, 0, 0.22)
    surface.set_zs(1, 0, 0.2)
    surface.fix_all()

    objective = SquaredFluxJAX(
        surface,
        BiotSavartJAX(coils),
        definition="quadratic flux",
    )
    assert not objective.field._uses_uniform_curve_xyz_fourier_fastpath
    return objective


def _set_field_kernel_tuning(
    monkeypatch,
    *,
    coil_chunk_size: int,
    quadrature_block_size: int,
) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_JAX_COIL_CHUNK_SIZE", str(coil_chunk_size))
    monkeypatch.setenv(
        "SIMSOPT_JAX_QUADRATURE_BLOCK_SIZE",
        str(quadrature_block_size),
    )
    invalidate_backend_cache()


def _large_grouped_flux_value_and_gradient(
    monkeypatch,
    *,
    coil_chunk_size: int,
    quadrature_block_size: int,
):
    _set_field_kernel_tuning(
        monkeypatch,
        coil_chunk_size=coil_chunk_size,
        quadrature_block_size=quadrature_block_size,
    )
    objective = _make_large_grouped_flux_objective()
    return objective.J(), objective.dJ()


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
    )
    return objective_cpu, objective_jax


def _make_non_rz_fixed_surface(surface_cls):
    surface = surface_cls(
        nfp=1,
        stellsym=True,
        mpol=1,
        ntor=0,
        quadpoints_phi=np.linspace(0.0, 1.0, 16, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 16, endpoint=False),
    )
    surface.fix_all()
    return surface


def _make_flux_objectives_for_surface(definition, surface, *, target=None):
    coils, _ = _make_native_flux_parity_case()

    bs_cpu = BiotSavart(coils)
    bs_cpu.set_points(surface.gamma().reshape((-1, 3)))
    objective_cpu = SquaredFlux(surface, bs_cpu, target=target, definition=definition)

    objective_jax = SquaredFluxJAX(
        surface,
        BiotSavartJAX(coils),
        target=target,
        definition=definition,
    )
    return objective_cpu, objective_jax


def _single_valid_flux_value_and_gradient(definition):
    """Independent NumPy oracle for the second point in the degenerate fixture."""
    normal = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    field = np.array([2.0, 0.0, 0.0], dtype=np.float64)
    target = 0.5
    grid_size = 2.0

    bdotn = float(np.dot(field, normal) - target)
    norm_n = float(np.linalg.norm(normal))
    field_norm_sq = float(np.dot(field, field))

    if definition == "quadratic flux":
        value = 0.5 * bdotn**2 * norm_n / grid_size
        gradient = bdotn * normal * norm_n / grid_size
    elif definition == "normalized":
        numerator = bdotn**2 * norm_n
        denominator = field_norm_sq * norm_n
        value = 0.5 * numerator / denominator
        gradient = 0.5 * (
            (2.0 * bdotn * normal) * denominator - numerator * (2.0 * field)
        ) / denominator**2
    elif definition == "local":
        value = 0.5 * bdotn**2 * norm_n / field_norm_sq / grid_size
        gradient = 0.5 * norm_n * (
            (2.0 * bdotn * normal) * field_norm_sq
            - bdotn**2 * (2.0 * field)
        ) / (field_norm_sq**2 * grid_size)
    else:
        raise ValueError(f"Unknown flux definition {definition!r}.")

    return value, np.concatenate((np.zeros(3, dtype=np.float64), gradient))


def _assert_flux_value_parity(actual, reference):
    np.testing.assert_allclose(actual, reference, rtol=_VALUE_RTOL, atol=_VALUE_ATOL)


def _assert_flux_gradient_parity(actual, reference):
    np.testing.assert_allclose(
        actual,
        reference,
        rtol=_GRADIENT_RTOL,
        atol=_GRADIENT_ATOL,
    )


def _assert_squared_flux_directional_fd(objective):
    x0 = np.asarray(objective.x, dtype=np.float64).copy()
    direction = np.linspace(-1.0, 1.0, x0.size, dtype=np.float64)
    direction /= np.linalg.norm(direction)
    gradient = np.asarray(objective.dJ(), dtype=np.float64)
    directional_gradient = float(np.dot(gradient, direction))

    eps = 1e-5
    objective.x = x0 + eps * direction
    value_plus = float(objective.J())
    objective.x = x0 - eps * direction
    value_minus = float(objective.J())
    objective.x = x0

    directional_fd = (value_plus - value_minus) / (2.0 * eps)
    np.testing.assert_allclose(
        directional_gradient,
        directional_fd,
        rtol=float(_FD_GRADIENT_TOLS["directional_fd_rtol"]),
        atol=float(_FD_GRADIENT_TOLS["directional_fd_atol"]),
    )


def _flux_kernel_value_and_grad(*, definition, normal, B, target):
    normal_array = np.asarray(normal, dtype=np.float64)
    flux_spec = make_fixed_surface_flux_spec(
        points=np.zeros_like(normal_array.reshape((-1, 3))),
        normal=normal_array,
        target=np.asarray(target, dtype=np.float64),
        definition=definition,
    )
    B_flat = jnp.asarray(np.asarray(B, dtype=np.float64).reshape((-1, 3)))

    def objective(B_arg):
        return fixed_surface_flux_integral_from_B(B_arg, flux_spec)

    return jax.value_and_grad(objective)(B_flat)



@pytest.fixture(autouse=True)
def _strict_parity_lane(monkeypatch, request, parity_lane):
    enable_strict_parity_backend(monkeypatch, request, parity_lane)
    with parity_default_device(parity_lane):
        yield


@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_fluxobjective_value_parity(definition):
    objective_cpu, objective_jax = _make_native_flux_objectives(definition)

    _assert_flux_value_parity(objective_jax.J(), objective_cpu.J())


@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_fluxobjective_gradient_parity(definition):
    objective_cpu, objective_jax = _make_native_flux_objectives(definition)
    _assert_flux_gradient_parity(objective_jax.dJ(), objective_cpu.dJ())


@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_squaredfluxjax_gradient_matches_directional_taylor_fd(definition):
    _, objective_jax = _make_native_flux_objectives(definition)
    _assert_squared_flux_directional_fd(objective_jax)


def test_squaredfluxjax_large_point_cloud_grouped_vjp_matches_dense(monkeypatch):
    try:
        dense_value, dense_gradient = _large_grouped_flux_value_and_gradient(
            monkeypatch,
            coil_chunk_size=0,
            quadrature_block_size=0,
        )
        chunked_value, chunked_gradient = _large_grouped_flux_value_and_gradient(
            monkeypatch,
            coil_chunk_size=2,
            quadrature_block_size=17,
        )
    finally:
        invalidate_backend_cache()

    _assert_flux_value_parity(chunked_value, dense_value)
    _assert_flux_gradient_parity(chunked_gradient, dense_gradient)


@pytest.mark.parametrize(
    "surface_cls",
    (SurfaceXYZFourier, SurfaceXYZTensorFourier),
)
@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_non_rz_fixed_surface_value_and_gradient_parity(surface_cls, definition):
    surface = _make_non_rz_fixed_surface(surface_cls)
    objective_cpu, objective_jax = _make_flux_objectives_for_surface(
        definition,
        surface,
    )

    _assert_flux_value_parity(objective_jax.J(), objective_cpu.J())
    _assert_flux_gradient_parity(objective_jax.dJ(), objective_cpu.dJ())


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
    value, grad = _flux_kernel_value_and_grad(
        definition="quadratic flux",
        normal=np.zeros((1, 1, 3)),
        B=np.zeros((1, 1, 3)),
        target=[[1.0]],
    )

    np.testing.assert_allclose(value, 0.0, atol=0.0)
    np.testing.assert_allclose(grad, np.zeros((1, 3)), atol=0.0)


@pytest.mark.parametrize("definition", _SQUARED_FLUX_DEFINITIONS)
def test_degenerate_normals_do_not_perturb_valid_flux_contracts(definition):
    full_value, full_grad = _flux_kernel_value_and_grad(
        definition=definition,
        normal=[[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]],
        B=[[[7.0, -3.0, 2.0], [2.0, 0.0, 0.0]]],
        target=[[11.0, 0.5]],
    )
    expected_value, expected_gradient = _single_valid_flux_value_and_gradient(
        definition
    )

    np.testing.assert_allclose(full_value, expected_value, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        np.asarray(full_grad).reshape(-1),
        expected_gradient,
        rtol=1e-12,
        atol=1e-15,
    )


@pytest.mark.parametrize("definition", ("normalized", "local"))
def test_singular_zero_field_contract(definition):
    value, grad = _flux_kernel_value_and_grad(
        definition=definition,
        normal=[[[1.0, 0.0, 0.0]]],
        B=[[[0.0, 0.0, 0.0]]],
        target=[[1.0]],
    )

    assert np.isinf(value)
    np.testing.assert_allclose(np.asarray(grad), np.zeros((1, 3)), atol=0.0)


def test_squaredfluxjax_requires_native_field_contract():
    _coils, surface = _make_native_flux_parity_case()
    field = _NonNativeFakeField()

    with pytest.raises(NotImplementedError, match="coil_dof_extraction_spec"):
        SquaredFluxJAX(
            surface,
            field,
            target=np.asarray([[0.0]], dtype=np.float64),
        )
