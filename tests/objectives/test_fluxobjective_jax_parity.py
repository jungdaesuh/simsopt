"""Mirrored CPU/JAX parity tests for SquaredFluxJAX."""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import enable_strict_parity_backend, parity_default_device, parity_rng

from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from simsopt._core.optimizable import Optimizable
from simsopt._core.util import ObjectiveFailure
from simsopt.backend import invalidate_backend_cache
from simsopt.field import BiotSavart, Current, coils_via_symmetries
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.geo.curve import create_equally_spaced_curves
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.geo.surfacexyzfourier import SurfaceXYZFourier
from simsopt.geo.surfacexyztensorfourier import SurfaceXYZTensorFourier
from simsopt.jax_core import (
    fixed_surface_flux_integral_from_B,
    fixed_surface_flux_specs_from_surface,
    make_fixed_surface_flux_spec,
)
from simsopt.objectives.fluxobjective import SquaredFlux
from simsopt.objectives.fluxobjective_jax import (
    SquaredFluxJAX,
    coil_current_fixed_geometry_flux_jax,
    coil_current_fixed_geometry_value_and_grad_jax,
)

_SQUARED_FLUX_DEFINITIONS = (
    "quadratic flux",
    "normalized",
    "local",
)
_STAGE2_MINIMAL_INPUT = (
    Path(__file__).resolve().parents[1]
    / "test_files"
    / "input.LandremanPaul2021_QA"
)
_STAGE2_MINIMAL_NCOILS = 4
_STAGE2_MINIMAL_NPHI = 32
_STAGE2_MINIMAL_NTHETA = 32
_STAGE2_MINIMAL_COIL_ORDER = 5
_STAGE2_MINIMAL_COIL_MAJOR_RADIUS = 1.0
_STAGE2_MINIMAL_COIL_MINOR_RADIUS = 0.5
_STAGE2_MINIMAL_CURRENT = 1.0e5
# SSOT parity tolerances from the validation-ladder contract.
# Value parity (CPU C++ ``SquaredFlux.J()`` oracle vs ``SquaredFluxJAX.J()``)
# uses the ``direct_kernel`` lane; gradient parity uses the first-derivative
# row of the ``derivative_heavy`` lane.
_DIRECT_KERNEL_TOLS = parity_ladder_tolerances("direct_kernel")
_DERIVATIVE_HEAVY_TOLS = parity_ladder_tolerances("derivative_heavy")
_FD_GRADIENT_TOLS = parity_ladder_tolerances("fd-gradient")

# Inline tolerances for chunked-vs-dense JAX self-consistency probes (Tier-4
# per tests/REVIEWER_ORACLE_LINT.md). These intentionally sit *tighter* than
# the parity-lane SSOT so chunking bugs cannot hide behind the cross-backend
# parity floor — see test_squaredfluxjax_large_point_cloud_grouped_vjp_matches_dense.
_CHUNKED_SELF_CONSISTENCY_VALUE_RTOL = 1e-12
_CHUNKED_SELF_CONSISTENCY_VALUE_ATOL = 1e-15
_CHUNKED_SELF_CONSISTENCY_GRADIENT_RTOL = 1e-11
_CHUNKED_SELF_CONSISTENCY_GRADIENT_ATOL = 1e-14


class _NonNativeFakeField(Optimizable):
    def __init__(self):
        self._uses_uniform_curve_xyz_fourier_fastpath = False
        self._points = None
        self._points_version = 0
        self._dof_layout_version = 0
        super().__init__(x0=np.zeros(1, dtype=np.float64))

    def recompute_bell(self, parent=None):
        del parent

    def set_points_from_spec(self, field_eval_spec):
        self._points = np.asarray(field_eval_spec.points, dtype=np.float64)
        self._points_version += 1


class _FixedGeometryCurrentFluxCase(NamedTuple):
    points: object
    gammas: object
    gammadashs: object
    currents: object
    flux_spec: object
    coils: object
    surface: SurfaceRZFourier


def _make_native_flux_parity_case(current_values=(1e5, 1e5)):
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
    base_currents = [Current(value) for value in current_values]
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
        gradient = (
            0.5
            * ((2.0 * bdotn * normal) * denominator - numerator * (2.0 * field))
            / denominator**2
        )
    elif definition == "local":
        value = 0.5 * bdotn**2 * norm_n / field_norm_sq / grid_size
        gradient = (
            0.5
            * norm_n
            * ((2.0 * bdotn * normal) * field_norm_sq - bdotn**2 * (2.0 * field))
            / (field_norm_sq**2 * grid_size)
        )
    else:
        raise ValueError(f"Unknown flux definition {definition!r}.")

    return value, np.concatenate((np.zeros(3, dtype=np.float64), gradient))


def _assert_flux_value_parity(actual, reference):
    """Value parity assertion using the ``direct_kernel`` lane SSOT."""
    np.testing.assert_allclose(
        actual,
        reference,
        rtol=_DIRECT_KERNEL_TOLS["rtol"],
        atol=_DIRECT_KERNEL_TOLS["atol"],
    )


def _assert_flux_gradient_parity(actual, reference):
    """Gradient parity assertion using the ``derivative_heavy`` lane SSOT."""
    np.testing.assert_allclose(
        actual,
        reference,
        rtol=_DERIVATIVE_HEAVY_TOLS["first_derivative_rtol"],
        atol=_DERIVATIVE_HEAVY_TOLS["first_derivative_atol"],
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
    """Fixed-surface edge-contract helper for the flux kernel.

    NOT a CPU/JAX parity oracle. Constructs a fixed-surface flux spec
    around a small hand-built ``(normal, B, target)`` triplet and
    returns ``(value, grad_B)`` from the JAX flux integrand. Used only
    by the three edge-contract tests below — zero normals,
    degenerate-normal masking, and singular zero-field handling — to
    pin specific finite-or-infinite contract behaviour on hand-built
    fixtures. Those tests cite either oracle type 2 (closed-form NumPy
    expressions, e.g. ``_single_valid_flux_value_and_gradient``) or
    documented contract semantics (NaN/Inf masking), not JAX-vs-JAX
    parity.

    Call sites:
      * ``test_quadratic_flux_zero_normals_contract``
      * ``test_degenerate_normals_do_not_perturb_valid_flux_contracts``
      * ``test_singular_zero_field_contract``

    Do not cite this helper as a parity oracle in new tests; route
    CPU/JAX parity through ``SquaredFlux``/``SquaredFluxJAX`` and the
    ``_assert_flux_value_parity`` / ``_assert_flux_gradient_parity``
    lane-driven helpers above.
    """
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


def _fixed_geometry_current_flux_case():
    stellsym = True
    surface = SurfaceRZFourier.from_vmec_input(
        _STAGE2_MINIMAL_INPUT,
        range="half period",
        nphi=_STAGE2_MINIMAL_NPHI,
        ntheta=_STAGE2_MINIMAL_NTHETA,
    )
    base_curves = create_equally_spaced_curves(
        _STAGE2_MINIMAL_NCOILS,
        surface.nfp,
        stellsym=stellsym,
        R0=_STAGE2_MINIMAL_COIL_MAJOR_RADIUS,
        R1=_STAGE2_MINIMAL_COIL_MINOR_RADIUS,
        order=_STAGE2_MINIMAL_COIL_ORDER,
    )
    base_currents = [
        Current(_STAGE2_MINIMAL_CURRENT) for _ in range(_STAGE2_MINIMAL_NCOILS)
    ]
    coils = coils_via_symmetries(
        base_curves,
        base_currents,
        surface.nfp,
        stellsym,
    )
    bs_jax = BiotSavartJAX(coils)
    coil_dofs = jnp.asarray(bs_jax.x, dtype=jnp.float64)
    groups = bs_jax.grouped_coil_arrays_from_dofs(coil_dofs)
    assert len(groups) == 1
    gammas, gammadashs, currents = groups[0]
    field_eval_spec, flux_spec = fixed_surface_flux_specs_from_surface(
        surface,
        definition="quadratic flux",
    )
    return _FixedGeometryCurrentFluxCase(
        points=field_eval_spec.points,
        gammas=gammas,
        gammadashs=gammadashs,
        currents=currents,
        flux_spec=flux_spec,
        coils=coils,
        surface=surface,
    )


def test_coil_current_fixed_geometry_flux_matches_cpu_squaredflux_oracle():
    """Current-only helper must match the CPU/C++ ``SquaredFlux`` oracle.

    Oracle: type 1 — CPU ``SquaredFlux.J()`` delegates to the C++
    ``simsoptpp.integral_BdotN`` reference for the
    ``examples/1_Simple/stage_two_optimization_minimal.py`` QA surface and
    symmetry-expanded Stage-II coil set.
    Lane: parity, direct-kernel tolerance from
    ``PARITY_LADDER_TOLERANCES["direct_kernel"]``.
    """
    case = _fixed_geometry_current_flux_case()

    actual = coil_current_fixed_geometry_flux_jax(
        case.points,
        case.gammas,
        case.gammadashs,
        case.currents,
        case.flux_spec,
    )
    bs_cpu = BiotSavart(case.coils)
    bs_cpu.set_points(case.surface.gamma().reshape((-1, 3)))
    expected = SquaredFlux(case.surface, bs_cpu, definition="quadratic flux").J()

    np.testing.assert_allclose(
        np.asarray(actual),
        np.asarray(expected),
        rtol=_DIRECT_KERNEL_TOLS["rtol"],
        atol=_DIRECT_KERNEL_TOLS["atol"],
    )


def test_coil_current_fixed_geometry_value_and_grad_matches_finite_difference():
    """Validate the current-gradient contract without reusing reverse-mode AD.

    Oracle: type 4 — central finite-difference directional derivative of the
    scalar current-only flux objective on the fixed Stage-II QA fixture.
    Lane: parity, fd-gradient tolerance from
    ``PARITY_LADDER_TOLERANCES["fd-gradient"]``.
    """
    case = _fixed_geometry_current_flux_case()
    value, grad = coil_current_fixed_geometry_value_and_grad_jax(
        case.points,
        case.gammas,
        case.gammadashs,
        case.currents,
        case.flux_spec,
    )
    direction = jnp.linspace(-0.75, 1.0, case.currents.size, dtype=jnp.float64)
    step = jnp.asarray(10.0, dtype=jnp.float64)
    forward = coil_current_fixed_geometry_flux_jax(
        case.points,
        case.gammas,
        case.gammadashs,
        case.currents + step * direction,
        case.flux_spec,
    )
    backward = coil_current_fixed_geometry_flux_jax(
        case.points,
        case.gammas,
        case.gammadashs,
        case.currents - step * direction,
        case.flux_spec,
    )
    finite_difference = (forward - backward) / (2.0 * step)

    assert np.isfinite(np.asarray(value))
    np.testing.assert_allclose(
        np.asarray(jnp.vdot(grad, direction)),
        np.asarray(finite_difference),
        rtol=_FD_GRADIENT_TOLS["directional_fd_rtol"],
        atol=_FD_GRADIENT_TOLS["directional_fd_atol"],
    )


def test_coil_current_fixed_geometry_value_and_grad_jits_under_strict_transfer_guard():
    """The current-only Stage-II QA kernel stays usable under JIT."""
    case = _fixed_geometry_current_flux_case()

    @jax.jit
    def compiled(current_values):
        return coil_current_fixed_geometry_value_and_grad_jax(
            case.points,
            case.gammas,
            case.gammadashs,
            current_values,
            case.flux_spec,
        )

    with jax.transfer_guard("disallow"):
        value, grad = compiled(case.currents)

    assert np.isfinite(np.asarray(value))
    assert np.asarray(grad).shape == tuple(case.currents.shape)


def test_squaredfluxjax_large_point_cloud_grouped_vjp_matches_dense(monkeypatch):
    """JAX chunked-vs-dense self-consistency on the flux integral.

    Tier-4 self-consistency check (JAX-vs-JAX through the same kernel
    under different chunk parameters) per ``tests/REVIEWER_ORACLE_LINT.md``;
    NOT a CPU/JAX parity test. Bound by
    ``_CHUNKED_SELF_CONSISTENCY_*`` constants, which are intentionally
    tighter than the parity-lane SSOT so chunking bugs cannot hide
    behind the cross-backend parity floor.
    """
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

    # Chunked-vs-dense JAX self-consistency: tighter than parity-lane SSOT
    # on purpose so a chunking bug cannot hide behind the parity floor.
    np.testing.assert_allclose(
        chunked_value,
        dense_value,
        rtol=_CHUNKED_SELF_CONSISTENCY_VALUE_RTOL,
        atol=_CHUNKED_SELF_CONSISTENCY_VALUE_ATOL,
    )
    np.testing.assert_allclose(
        chunked_gradient,
        dense_gradient,
        rtol=_CHUNKED_SELF_CONSISTENCY_GRADIENT_RTOL,
        atol=_CHUNKED_SELF_CONSISTENCY_GRADIENT_ATOL,
    )


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


@pytest.mark.parametrize("definition", ("normalized", "local"))
def test_squaredfluxjax_zero_current_gradient_raises_objective_failure(definition):
    coils, surface = _make_native_flux_parity_case(current_values=(0.0, 0.0))
    objective = SquaredFluxJAX(
        surface,
        BiotSavartJAX(coils),
        definition=definition,
    )

    with pytest.raises(ObjectiveFailure, match="gradient is singular"):
        objective.dJ()


def test_squaredfluxjax_rejects_field_point_mutation_after_construction():
    _, objective = _make_native_flux_objectives("quadratic flux")
    mutated_points = np.asarray(
        objective.surface.gamma().reshape((-1, 3)), dtype=np.float64
    )
    mutated_points[:, 0] += 1.0e-3
    objective.field.set_points(mutated_points)

    with pytest.raises(RuntimeError, match="Do not call field.set_points"):
        objective.J()
    with pytest.raises(RuntimeError, match="Do not call field.set_points"):
        objective.dJ()


def test_squaredfluxjax_rejects_field_dof_layout_mutation_after_construction():
    objective = _make_large_grouped_flux_objective()
    assert not objective.field._uses_uniform_curve_xyz_fourier_fastpath
    lineage_opt = next(
        opt for opt in objective.field.unique_dof_lineage if opt.local_dof_size > 1
    )

    lineage_opt.fix(0)
    try:
        with pytest.raises(RuntimeError, match="free/fixed DOF layout"):
            objective.J()
        with pytest.raises(RuntimeError, match="free/fixed DOF layout"):
            objective.dJ()
    finally:
        lineage_opt.unfix(0)


def test_squaredfluxjax_requires_native_field_contract():
    _coils, surface = _make_native_flux_parity_case()
    field = _NonNativeFakeField()

    with pytest.raises(NotImplementedError, match="coil_dof_extraction_spec"):
        SquaredFluxJAX(
            surface,
            field,
            target=np.asarray([[0.0]], dtype=np.float64),
        )
