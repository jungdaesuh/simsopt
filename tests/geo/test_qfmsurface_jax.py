"""QFM JAX solver orchestration tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import simsopt_jax.geo.optimizers.private._bfgs as _private_bfgs

from conftest import enable_strict_jax_backend, host_array, host_scalar

from simsopt.configs.zoo import get_data
from simsopt.field.biotsavart import BiotSavart
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt.field.coil import Coil
import simsopt_jax_adapters.geo.qfm_surface as qfmsurface_jax_module
from simsopt.geo.qfmsurface import QfmSurface
from simsopt_jax_adapters.geo.qfm_surface import QfmSurfaceJAX
from simsopt.geo.surfaceobjectives import Area, QfmResidual, ToroidalFlux, Volume
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt_jax.core import qfm_solver as qfm_solver_module
from simsopt_jax.core.qfm_solver import (
    QfmAugmentedLagrangianInfo,
    QfmPenaltySolveInfo,
    qfm_augmented_lagrangian_solve_jax,
    qfm_exact_kkt_residual_jax_from_dofs,
    qfm_penalty_jax_from_dofs,
    qfm_penalty_solve_jax,
    qfm_penalty_value_and_grad_jax_from_dofs,
    qfm_residual_jax_from_dofs,
)
from simsopt_jax.core.specs import (
    make_surface_xyz_fourier_spec,
    make_surface_xyz_tensor_fourier_spec,
)
from simsopt_jax.core.surface_rzfourier import surface_rz_fourier_spec_from_dofs

from .surface_test_helpers import get_surface


def _make_ncsx_rz_qfm_surface():
    _base_curves, _base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    phis = np.linspace(0.0, 1.0 / nfp, 6, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 6, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    return biotsavart, surface


def _make_qfm_case():
    biotsavart, surface = _make_ncsx_rz_qfm_surface()
    return BiotSavartJAX(biotsavart.coils), surface


def _make_label_grid_qfm_cpu_case():
    _base_curves, _base_currents, _magnetic_axis, nfp, biotsavart = get_data("ncsx")
    phis = np.linspace(0.0, 1.0 / nfp, 6, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 5, endpoint=False)
    surface = SurfaceRZFourier(
        mpol=3,
        ntor=2,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    surface.x = dofs + 0.02 * np.sin(np.arange(dofs.size, dtype=np.float64))
    return biotsavart, surface


def _make_label_grid_qfm_case():
    biotsavart, surface = _make_label_grid_qfm_cpu_case()
    return BiotSavartJAX(biotsavart.coils), surface


def _coil_set_spec(biotsavart):
    return biotsavart.coil_set_spec_from_dofs(
        jnp.asarray(biotsavart.x, dtype=jnp.float64)
    )


def _surface_spec(surface):
    surface_type = type(surface).__name__
    if surface_type == "SurfaceRZFourier":
        return surface_rz_fourier_spec_from_dofs(
            surface.get_dofs(),
            quadpoints_phi=surface.quadpoints_phi,
            quadpoints_theta=surface.quadpoints_theta,
            mpol=surface.mpol,
            ntor=surface.ntor,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
        )
    if surface_type == "SurfaceXYZFourier":
        return make_surface_xyz_fourier_spec(
            dofs=surface.get_dofs(),
            quadpoints_phi=surface.quadpoints_phi,
            quadpoints_theta=surface.quadpoints_theta,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
        )
    if surface_type == "SurfaceXYZTensorFourier":
        return make_surface_xyz_tensor_fourier_spec(
            dofs=surface.get_dofs(),
            quadpoints_phi=surface.quadpoints_phi,
            quadpoints_theta=surface.quadpoints_theta,
            nfp=surface.nfp,
            stellsym=surface.stellsym,
            mpol=surface.mpol,
            ntor=surface.ntor,
            clamped_dims=tuple(getattr(surface, "clamped_dims", (False, False, False))),
        )
    raise TypeError(f"Unsupported surface type for explicit spec: {surface_type}")


def _make_qfm_inputs():
    biotsavart, surface = _make_qfm_case()
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    return biotsavart, surface, dofs, _coil_set_spec(biotsavart)


def _make_test_qfm_xyz_volume_case():
    _base_curves, _base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    phis = np.linspace(0.0, 1.0 / nfp, 20, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 20, endpoint=False)
    surface = get_surface(
        "SurfaceXYZFourier",
        True,
        phis=phis,
        thetas=thetas,
        ntor=4,
        mpol=4,
    )
    surface.fit_to_curve(magnetic_axis, 0.2)
    return biotsavart, surface


def _make_qfm_gradient_case(surface_type: str, stellsym: bool):
    _base_curves, _base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    surface_cpu = get_surface(
        surface_type,
        stellsym,
        mpol=1,
        ntor=1,
        nfp=nfp,
        nphi=7,
        ntheta=8,
        full=True,
    )
    surface_cpu.fit_to_curve(magnetic_axis, 0.1)
    surface_jax = get_surface(
        surface_type,
        stellsym,
        mpol=1,
        ntor=1,
        nfp=nfp,
        nphi=7,
        ntheta=8,
        full=True,
    )
    surface_jax.x = np.asarray(surface_cpu.get_dofs(), dtype=np.float64)

    label_cpu = Area(surface_cpu)
    label_jax = Area(surface_jax)
    target = 0.97 * label_cpu.J()
    qfm_cpu = QfmSurface(BiotSavart(biotsavart.coils), surface_cpu, label_cpu, target)
    qfm_jax = QfmSurfaceJAX(
        BiotSavartJAX(biotsavart.coils),
        surface_jax,
        label_jax,
        target,
    )
    base_dofs = np.asarray(surface_cpu.get_dofs(), dtype=np.float64)
    trial_dofs = base_dofs + 1.0e-3 * np.cos(
        np.arange(base_dofs.size, dtype=np.float64)
    )
    return qfm_cpu, qfm_jax, trial_dofs


def _assert_value_and_gradient_match(cpu_result, jax_result):
    value_cpu, gradient_cpu = cpu_result
    value_jax, gradient_jax = jax_result
    np.testing.assert_allclose(value_jax, value_cpu, rtol=1.0e-10, atol=1.0e-12)
    np.testing.assert_allclose(
        gradient_jax,
        gradient_cpu,
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def _scaled_biotsavart(coils, scale: float):
    return BiotSavart([Coil(coil.curve, scale * coil.current) for coil in coils])


def test_qfm_bfgs_curvature_floor_rejects_float32_boundary() -> None:
    s = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    y = jnp.asarray([1.0e-5, 1.0], dtype=jnp.float32)

    qfm_valid = qfm_solver_module._bfgs_has_valid_curvature(s, y)
    _, _, private_valid, _ = _private_bfgs._bfgs_curvature_terms(
        s,
        y,
        x_dtype=s.dtype,
    )

    assert bool(qfm_valid) is False
    assert bool(private_valid) is False


@pytest.mark.parametrize(
    "surface_type",
    [
        "SurfaceXYZFourier",
        "SurfaceXYZTensorFourier",
    ],
)
@pytest.mark.parametrize("stellsym", [True, False])
def test_qfm_surface_jax_gradients_match_cpu_legacy_contracts(
    surface_type: str,
    stellsym: bool,
) -> None:
    qfm_cpu, qfm_jax, trial_dofs = _make_qfm_gradient_case(surface_type, stellsym)

    _assert_value_and_gradient_match(
        qfm_cpu.qfm_objective(trial_dofs, derivatives=1),
        qfm_jax.qfm_objective(trial_dofs, derivatives=1),
    )
    _assert_value_and_gradient_match(
        qfm_cpu.qfm_label_constraint(trial_dofs, derivatives=1),
        qfm_jax.qfm_label_constraint(trial_dofs, derivatives=1),
    )
    _assert_value_and_gradient_match(
        qfm_cpu.qfm_penalty_constraints(
            trial_dofs,
            derivatives=1,
            constraint_weight=1.7,
        ),
        qfm_jax.qfm_penalty_constraints(
            trial_dofs,
            derivatives=1,
            constraint_weight=1.7,
        ),
    )


def _cpu_label_constraint_value_and_grad(label, target: float, dofs: np.ndarray):
    label.surface.x = np.asarray(dofs, dtype=np.float64)
    residual = label.J() - target
    return 0.5 * residual**2, residual * label.dJ_by_dsurfacecoefficients()


def _cpu_qfm_penalty_value_and_grad(
    biotsavart,
    surface,
    label,
    target: float,
    dofs: np.ndarray,
    constraint_weight: float,
):
    surface.x = np.asarray(dofs, dtype=np.float64)
    qfm = QfmResidual(surface, biotsavart)
    qfm_value = qfm.J()
    qfm_gradient = qfm.dJ_by_dsurfacecoefficients()
    label.surface.x = np.asarray(dofs, dtype=np.float64)
    label_residual = label.J() - target
    label_gradient = label.dJ_by_dsurfacecoefficients()
    return (
        qfm_value + 0.5 * constraint_weight * label_residual**2,
        qfm_gradient + constraint_weight * label_residual * label_gradient,
    )


def _cpu_label_value(label, dofs: object) -> float:
    label.surface.x = host_array(dofs)
    return label.J()


def _qfm_surface_with_distinct_label_field():
    _base_curves, _base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    phis = np.linspace(0.0, 1.0 / nfp, 6, endpoint=False)
    thetas = np.linspace(0.0, 1.0, 6, endpoint=False)
    surface_cpu = SurfaceRZFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface_cpu.fit_to_curve(magnetic_axis, 0.2, flip_theta=True)
    surface_jax = SurfaceRZFourier(
        mpol=1,
        ntor=1,
        stellsym=True,
        nfp=nfp,
        quadpoints_phi=phis,
        quadpoints_theta=thetas,
    )
    surface_jax.x = np.asarray(surface_cpu.get_dofs(), dtype=np.float64)
    qfm_field_cpu = BiotSavart(biotsavart.coils)
    label_field_cpu = _scaled_biotsavart(biotsavart.coils, 0.73)
    qfm_field_jax = BiotSavartJAX(qfm_field_cpu.coils)
    label_field_jax = BiotSavartJAX(label_field_cpu.coils)
    return (
        qfm_field_cpu,
        label_field_cpu,
        surface_cpu,
        qfm_field_jax,
        label_field_jax,
        surface_jax,
    )


def _penalty_info(dofs: jax.Array) -> QfmPenaltySolveInfo:
    return QfmPenaltySolveInfo(
        success=jnp.asarray(True),
        status=jnp.asarray(0),
        fun=jnp.asarray(0.25, dtype=dofs.dtype),
        gradient=jnp.ones_like(dofs),
        nit=jnp.asarray(2),
        nfev=jnp.asarray(3),
        njev=jnp.asarray(4),
        label_value=jnp.asarray(1.0, dtype=dofs.dtype),
        label_residual=jnp.asarray(0.0, dtype=dofs.dtype),
        qfm_value=jnp.asarray(0.125, dtype=dofs.dtype),
        penalty_value=jnp.asarray(0.25, dtype=dofs.dtype),
    )


def _augmented_info(dofs: jax.Array) -> QfmAugmentedLagrangianInfo:
    return QfmAugmentedLagrangianInfo(
        success=jnp.asarray(True),
        status=jnp.asarray(0),
        fun=jnp.asarray(0.2, dtype=dofs.dtype),
        gradient=jnp.ones_like(dofs),
        nit=jnp.asarray(1),
        nfev=jnp.asarray(2),
        njev=jnp.asarray(3),
        label_value=jnp.asarray(1.0, dtype=dofs.dtype),
        label_residual=jnp.asarray(0.0, dtype=dofs.dtype),
        qfm_value=jnp.asarray(0.1, dtype=dofs.dtype),
        augmented_value=jnp.asarray(0.2, dtype=dofs.dtype),
        multiplier=jnp.asarray(0.0, dtype=dofs.dtype),
        penalty_weight=jnp.asarray(10.0, dtype=dofs.dtype),
    )


def test_qfm_penalty_solve_jax_reduces_fixed_state_penalty() -> None:
    """Oracle: the same pure QFM penalty kernel before and after the solve."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.98 * Area(surface).J()
    initial = qfm_penalty_jax_from_dofs(
        _surface_spec(surface),
        dofs,
        coil_set_spec,
        label="area",
        label_spec=_surface_spec(surface),
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
        constraint_weight=1.0,
    )

    final_dofs, info = qfm_penalty_solve_jax(
        _surface_spec(surface),
        coil_set_spec,
        "area",
        target,
        1.0,
        dofs,
        label_spec=_surface_spec(surface),
        label_coil_set_spec=coil_set_spec,
        max_iter=5,
        tol=1e-8,
    )

    assert final_dofs.shape == dofs.shape
    assert host_scalar(info.penalty_value) < host_scalar(initial)
    assert host_array(info.gradient).shape == tuple(dofs.shape)


def test_qfm_penalty_solve_jax_not_worse_than_host_lbfgsb_diagnostic() -> None:
    """Diagnostic: JAX penalty solve is not worse than host LBFGS-B residual."""
    biotsavart_cpu, surface_cpu = _make_test_qfm_xyz_volume_case()
    _biotsavart_src, surface_jax = _make_test_qfm_xyz_volume_case()
    label_cpu = Volume(surface_cpu)
    target = label_cpu.J()
    qfm_cpu = QfmSurface(
        BiotSavart(biotsavart_cpu.coils),
        surface_cpu,
        label_cpu,
        target,
    )

    cpu_result = qfm_cpu.minimize_qfm_penalty_constraints_LBFGS(
        tol=1e-8,
        maxiter=200,
        constraint_weight=1.0,
    )
    cpu_qfm_residual = qfm_cpu.qfm_objective(surface_cpu.get_dofs())
    biotsavart_jax = BiotSavartJAX(biotsavart_cpu.coils)
    coil_set_spec = _coil_set_spec(biotsavart_jax)
    final_dofs, info = qfm_penalty_solve_jax(
        _surface_spec(surface_jax),
        coil_set_spec,
        "volume",
        target,
        1.0,
        jnp.asarray(surface_jax.get_dofs(), dtype=jnp.float64),
        label_spec=_surface_spec(surface_jax),
        label_coil_set_spec=coil_set_spec,
        max_iter=400,
        tol=1e-8,
    )

    assert cpu_result["success"]
    assert final_dofs.shape == tuple(surface_jax.get_dofs().shape)
    assert host_scalar(info.qfm_value) <= cpu_qfm_residual * (1.0 + 1.0e-6)


def test_qfm_augmented_lagrangian_meets_upstream_exact_acceptance() -> None:
    """AL exact path meets upstream QFM acceptance after the host LBFGS warm start."""
    biotsavart_cpu, warm_surface = _make_test_qfm_xyz_volume_case()
    label = Volume(warm_surface)
    target = label.J()
    warm_qfm = QfmSurface(
        BiotSavart(biotsavart_cpu.coils),
        warm_surface,
        label,
        target,
    )
    warm_result = warm_qfm.minimize_qfm_penalty_constraints_LBFGS(
        tol=1e-8,
        maxiter=1000,
        constraint_weight=1.0,
    )
    warm_dofs = np.asarray(warm_surface.get_dofs(), dtype=np.float64)

    _biotsavart_src, host_surface = _make_test_qfm_xyz_volume_case()
    host_surface.x = warm_dofs.copy()
    host_qfm = QfmSurface(
        BiotSavart(biotsavart_cpu.coils),
        host_surface,
        Volume(host_surface),
        target,
    )
    host_result = host_qfm.minimize_qfm_exact_constraints_SLSQP(tol=1e-9, maxiter=1000)
    host_qfm_residual = host_qfm.qfm_objective(host_surface.get_dofs())
    host_label_residual = Volume(host_surface).J() - target

    _biotsavart_src, jax_surface = _make_test_qfm_xyz_volume_case()
    jax_surface.x = warm_dofs.copy()
    biotsavart_jax = BiotSavartJAX(biotsavart_cpu.coils)
    coil_set_spec = _coil_set_spec(biotsavart_jax)
    al_dofs, al_info = qfm_augmented_lagrangian_solve_jax(
        _surface_spec(jax_surface),
        coil_set_spec,
        "volume",
        target,
        jnp.asarray(jax_surface.get_dofs(), dtype=jnp.float64),
        label_spec=_surface_spec(jax_surface),
        label_coil_set_spec=coil_set_spec,
        max_outer=3,
        inner_max_iter=200,
        tol=1e-8,
    )

    assert warm_result["success"]
    assert host_result["success"]
    assert host_qfm_residual < 1.0e-5
    assert abs(host_label_residual) < 3.0e-5
    assert abs(host_scalar(al_info.label_residual)) <= 1.0e-6
    assert host_scalar(al_info.qfm_value) < 1.0e-5
    assert host_scalar(al_info.qfm_value) <= host_qfm_residual
    assert al_dofs.shape == tuple(jax_surface.get_dofs().shape)


def test_qfm_augmented_lagrangian_kkt_diagnostic_no_worse_than_host_slsqp() -> None:
    """Diagnostic: natural equality KKT residual, not host SLSQP DOF identity."""
    biotsavart_cpu, warm_surface = _make_test_qfm_xyz_volume_case()
    label = Volume(warm_surface)
    target = label.J()
    warm_qfm = QfmSurface(
        BiotSavart(biotsavart_cpu.coils),
        warm_surface,
        label,
        target,
    )
    warm_result = warm_qfm.minimize_qfm_penalty_constraints_LBFGS(
        tol=1e-8,
        maxiter=1000,
        constraint_weight=1.0,
    )
    warm_dofs = np.asarray(warm_surface.get_dofs(), dtype=np.float64)

    _biotsavart_src, host_surface = _make_test_qfm_xyz_volume_case()
    host_surface.x = warm_dofs.copy()
    host_qfm = QfmSurface(
        BiotSavart(biotsavart_cpu.coils),
        host_surface,
        Volume(host_surface),
        target,
    )
    host_result = host_qfm.minimize_qfm_exact_constraints_SLSQP(tol=1e-9, maxiter=1000)

    _biotsavart_src, jax_surface = _make_test_qfm_xyz_volume_case()
    jax_surface.x = warm_dofs.copy()
    biotsavart_jax = BiotSavartJAX(biotsavart_cpu.coils)
    coil_set_spec = _coil_set_spec(biotsavart_jax)
    surface_spec = _surface_spec(jax_surface)
    al_dofs, al_info = qfm_augmented_lagrangian_solve_jax(
        surface_spec,
        coil_set_spec,
        "volume",
        target,
        jnp.asarray(jax_surface.get_dofs(), dtype=jnp.float64),
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        max_outer=3,
        inner_max_iter=200,
        tol=1e-8,
    )

    host_kkt = qfm_exact_kkt_residual_jax_from_dofs(
        surface_spec,
        jnp.asarray(host_surface.get_dofs(), dtype=jnp.float64),
        coil_set_spec,
        label="volume",
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
    )
    al_kkt = qfm_exact_kkt_residual_jax_from_dofs(
        surface_spec,
        al_dofs,
        coil_set_spec,
        label="volume",
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
    )

    assert warm_result["success"]
    assert host_result["success"]
    assert abs(host_scalar(al_info.label_residual)) <= 1.0e-6
    assert host_scalar(host_kkt.label_gradient_norm) > 1.0
    assert host_scalar(al_kkt.label_gradient_norm) > 1.0
    assert host_scalar(al_kkt.feasibility_abs) <= host_scalar(host_kkt.feasibility_abs)
    assert host_scalar(al_kkt.stationarity_inf) <= host_scalar(
        host_kkt.stationarity_inf
    ) * (1.0 + 1.0e-8)


def test_qfm_augmented_lagrangian_success_uses_absolute_kkt() -> None:
    """KKT-passing AL results are successful at the public solver surface."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.99 * Area(surface).J()
    surface_spec = _surface_spec(surface)
    perturbation = 1.0e-3 * jnp.sin(jnp.arange(dofs.size, dtype=jnp.float64))
    final_dofs, info = qfm_augmented_lagrangian_solve_jax(
        surface_spec,
        coil_set_spec,
        "area",
        target,
        dofs + perturbation,
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        max_outer=5,
        inner_max_iter=100,
        tol=1.0e-6,
    )
    kkt = qfm_exact_kkt_residual_jax_from_dofs(
        surface_spec,
        final_dofs,
        coil_set_spec,
        label="area",
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
    )

    assert bool(host_scalar(info.success))
    assert host_scalar(kkt.feasibility_abs) <= 1.0e-6
    assert host_scalar(kkt.stationarity_inf) <= 1.0e-6


def test_qfm_augmented_lagrangian_rejects_feasible_nonstationary_state() -> None:
    """Label feasibility alone is not enough for exact-path AL success."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.99 * Area(surface).J()
    surface_spec = _surface_spec(surface)
    final_dofs, info = qfm_augmented_lagrangian_solve_jax(
        surface_spec,
        coil_set_spec,
        "area",
        target,
        dofs,
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        max_outer=2,
        inner_max_iter=20,
        tol=2.0e-4,
    )
    kkt = qfm_exact_kkt_residual_jax_from_dofs(
        surface_spec,
        final_dofs,
        coil_set_spec,
        label="area",
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
    )

    assert host_scalar(kkt.feasibility_abs) <= 2.0e-4
    assert host_scalar(kkt.stationarity_inf) > 2.0e-4
    assert not bool(host_scalar(info.success))


def test_qfm_augmented_lagrangian_branch_stability_uses_kkt_invariants() -> None:
    """Small warm-start perturbations preserve objective, label, and KKT invariants."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.99 * Area(surface).J()
    surface_spec = _surface_spec(surface)
    perturbation = jnp.sin(jnp.arange(dofs.size, dtype=jnp.float64))
    results = []
    for scale in (0.0, 1.0e-4, 1.0e-3):
        final_dofs, info = qfm_augmented_lagrangian_solve_jax(
            surface_spec,
            coil_set_spec,
            "area",
            target,
            dofs + scale * perturbation,
            label_spec=surface_spec,
            label_coil_set_spec=coil_set_spec,
            max_outer=5,
            inner_max_iter=100,
            tol=1.0e-6,
        )
        kkt = qfm_exact_kkt_residual_jax_from_dofs(
            surface_spec,
            final_dofs,
            coil_set_spec,
            label="area",
            label_spec=surface_spec,
            label_coil_set_spec=coil_set_spec,
            targetlabel=target,
        )
        assert bool(host_scalar(info.success))
        assert host_scalar(kkt.feasibility_abs) <= 1.0e-6
        assert host_scalar(kkt.stationarity_inf) <= 1.0e-6
        results.append(
            (
                host_scalar(info.qfm_value),
                host_scalar(info.label_residual),
                host_scalar(kkt.stationarity_inf),
            )
        )

    qfm_values, label_residuals, stationarity_values = zip(*results, strict=True)
    np.testing.assert_allclose(qfm_values, qfm_values[0], rtol=1.0e-7, atol=1.0e-12)
    np.testing.assert_allclose(
        label_residuals,
        label_residuals[0],
        rtol=0.0,
        atol=1.0e-9,
    )
    assert np.ptp(np.asarray(stationarity_values)) <= 5.0e-7


def test_qfm_penalty_fixed_state_gradient_matches_centered_fd() -> None:
    """Derivative-heavy lane: fixed-state JAX gradient matches FD."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.98 * Area(surface).J()
    surface_spec = _surface_spec(surface)
    value, gradient = qfm_penalty_value_and_grad_jax_from_dofs(
        surface_spec,
        dofs,
        coil_set_spec,
        label="area",
        label_spec=surface_spec,
        label_coil_set_spec=coil_set_spec,
        targetlabel=target,
        constraint_weight=1.0,
    )
    step = 2.0**-18
    finite_difference_gradient = []
    for idx in range(dofs.size):
        basis = np.zeros(dofs.size, dtype=np.float64)
        basis[idx] = 1.0
        basis_jax = jnp.asarray(basis, dtype=jnp.float64)
        value_plus = qfm_penalty_jax_from_dofs(
            surface_spec,
            dofs + step * basis_jax,
            coil_set_spec,
            label="area",
            label_spec=surface_spec,
            label_coil_set_spec=coil_set_spec,
            targetlabel=target,
            constraint_weight=1.0,
        )
        value_minus = qfm_penalty_jax_from_dofs(
            surface_spec,
            dofs - step * basis_jax,
            coil_set_spec,
            label="area",
            label_spec=surface_spec,
            label_coil_set_spec=coil_set_spec,
            targetlabel=target,
            constraint_weight=1.0,
        )
        finite_difference_gradient.append(
            (host_scalar(value_plus) - host_scalar(value_minus)) / (2.0 * step)
        )

    assert np.isfinite(host_scalar(value))
    np.testing.assert_allclose(
        np.asarray(finite_difference_gradient),
        host_array(gradient),
        rtol=1.0e-8,
        atol=1.0e-10,
    )


def test_qfm_penalty_solve_jax_transfer_guard_clean() -> None:
    """The BFGS solver core does not enter JAX's host-staging optimizer path."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = jnp.asarray(0.98 * Area(surface).J(), dtype=dofs.dtype)
    constraint_weight = jnp.asarray(1.0, dtype=dofs.dtype)

    with jax.transfer_guard("disallow"):
        final_dofs, info = qfm_penalty_solve_jax(
            _surface_spec(surface),
            coil_set_spec,
            "area",
            target,
            constraint_weight,
            dofs,
            label_spec=_surface_spec(surface),
            label_coil_set_spec=coil_set_spec,
            max_iter=1,
            tol=1e-8,
        )

    assert final_dofs.shape == dofs.shape
    assert host_array(info.gradient).shape == tuple(dofs.shape)


def test_qfm_augmented_lagrangian_solve_jax_transfer_guard_clean() -> None:
    """The AL wrapper keeps scalar updates and inner BFGS staging on device."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = jnp.asarray(0.99 * Area(surface).J(), dtype=dofs.dtype)

    with jax.transfer_guard("disallow"):
        final_dofs, info = qfm_augmented_lagrangian_solve_jax(
            _surface_spec(surface),
            coil_set_spec,
            "area",
            target,
            dofs,
            label_spec=_surface_spec(surface),
            label_coil_set_spec=coil_set_spec,
            max_outer=2,
            inner_max_iter=1,
            tol=1e-8,
        )

    assert final_dofs.shape == dofs.shape
    assert host_array(info.gradient).shape == tuple(dofs.shape)
    np.testing.assert_allclose(host_scalar(info.penalty_weight), 100.0)
    assert host_scalar(info.multiplier) != 0.0


def test_qfm_augmented_lagrangian_info_reports_qfm_gradient() -> None:
    """The exact-path result pairs QFM ``fun`` with the QFM objective gradient."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = Area(surface).J()

    final_dofs, info = qfm_augmented_lagrangian_solve_jax(
        _surface_spec(surface),
        coil_set_spec,
        "area",
        target,
        dofs,
        label_spec=_surface_spec(surface),
        label_coil_set_spec=coil_set_spec,
        max_outer=1,
        inner_max_iter=1,
        tol=1e-8,
    )
    expected_gradient = jax.grad(
        lambda surface_dofs: qfm_residual_jax_from_dofs(
            _surface_spec(surface),
            surface_dofs,
            coil_set_spec,
        )
    )(final_dofs)

    np.testing.assert_allclose(
        host_array(info.gradient),
        host_array(expected_gradient),
        rtol=1e-10,
        atol=1e-12,
    )


def test_qfm_augmented_lagrangian_info_keeps_augmented_value_separate() -> None:
    """AL diagnostics keep public QFM ``fun`` separate from augmented value."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.99 * Area(surface).J()

    _final_dofs, info = qfm_augmented_lagrangian_solve_jax(
        _surface_spec(surface),
        coil_set_spec,
        "area",
        target,
        dofs,
        label_spec=_surface_spec(surface),
        label_coil_set_spec=coil_set_spec,
        max_outer=2,
        inner_max_iter=1,
        tol=1e-8,
    )

    np.testing.assert_allclose(host_scalar(info.fun), host_scalar(info.qfm_value))
    assert not np.isclose(
        host_scalar(info.augmented_value),
        host_scalar(info.fun),
        rtol=1e-8,
        atol=1e-12,
    )
    assert host_scalar(info.multiplier) != 0.0
    np.testing.assert_allclose(
        host_scalar(info.penalty_weight),
        100.0,
        rtol=0.0,
        atol=0.0,
    )


def test_qfm_surface_jax_penalty_value_and_gradient_do_not_mutate_surface() -> None:
    """Oracle: CPU QFM residual plus CPU label value/gradient at trial DOFs."""
    biotsavart_cpu, surface_cpu = _make_ncsx_rz_qfm_surface()
    _biotsavart_src, surface = _make_ncsx_rz_qfm_surface()
    biotsavart = BiotSavartJAX(biotsavart_cpu.coils)
    initial_dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    trial_dofs = initial_dofs.copy()
    trial_dofs[0] += 1.0e-3
    label_cpu = Area(surface_cpu)
    target = 0.99 * label_cpu.J()
    qfm_surface = QfmSurfaceJAX(biotsavart, surface, Area(surface), target)

    value, gradient = qfm_surface.qfm_penalty_constraints(
        trial_dofs,
        derivatives=1,
        constraint_weight=1.5,
    )
    expected_value, expected_gradient = _cpu_qfm_penalty_value_and_grad(
        biotsavart_cpu,
        surface_cpu,
        label_cpu,
        target,
        trial_dofs,
        1.5,
    )

    np.testing.assert_allclose(value, expected_value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(surface.get_dofs(), initial_dofs, rtol=0.0, atol=0.0)


def test_qfm_surface_jax_label_constraint_uses_label_surface_spec() -> None:
    """Label residuals use the label-owned quadrature grid, matching CPU QFM."""
    biotsavart_cpu, surface_cpu = _make_label_grid_qfm_cpu_case()
    _biotsavart_src, surface = _make_label_grid_qfm_cpu_case()
    biotsavart = BiotSavartJAX(biotsavart_cpu.coils)
    label_cpu = Area(
        surface_cpu,
        nphi=4,
        ntheta=9,
        range=SurfaceRZFourier.RANGE_FULL_TORUS,
    )
    label = Area(
        surface,
        nphi=4,
        ntheta=9,
        range=SurfaceRZFourier.RANGE_FULL_TORUS,
    )
    target = label_cpu.J()
    trial_dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    trial_dofs = trial_dofs + 0.01 * np.cos(np.arange(trial_dofs.size))
    qfm_surface = QfmSurfaceJAX(biotsavart, surface, label, target)

    value, gradient = qfm_surface.qfm_label_constraint(trial_dofs, derivatives=1)
    expected_value, expected_gradient = _cpu_label_constraint_value_and_grad(
        label_cpu,
        target,
        trial_dofs,
    )
    wrong_label = Area(surface_cpu)
    wrong_value, _wrong_gradient = _cpu_label_constraint_value_and_grad(
        wrong_label,
        target,
        trial_dofs,
    )

    np.testing.assert_allclose(value, expected_value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-10)
    assert not np.isclose(value, wrong_value, rtol=1e-8, atol=1e-12)


def test_qfm_surface_jax_penalty_constraints_use_label_surface_spec() -> None:
    """Penalty value/gradient keep QFM and label quadrature specs separate."""
    biotsavart_cpu, surface_cpu = _make_label_grid_qfm_cpu_case()
    _biotsavart_src, surface = _make_label_grid_qfm_cpu_case()
    biotsavart = BiotSavartJAX(biotsavart_cpu.coils)
    label_cpu = Area(
        surface_cpu,
        nphi=4,
        ntheta=9,
        range=SurfaceRZFourier.RANGE_FULL_TORUS,
    )
    label = Area(
        surface,
        nphi=4,
        ntheta=9,
        range=SurfaceRZFourier.RANGE_FULL_TORUS,
    )
    target = label_cpu.J()
    trial_dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    trial_dofs = trial_dofs + 0.01 * np.cos(np.arange(trial_dofs.size))
    qfm_surface = QfmSurfaceJAX(biotsavart, surface, label, target)

    value, gradient = qfm_surface.qfm_penalty_constraints(
        trial_dofs,
        derivatives=1,
        constraint_weight=1.5,
    )
    expected_value, expected_gradient = _cpu_qfm_penalty_value_and_grad(
        biotsavart_cpu,
        surface_cpu,
        label_cpu,
        target,
        trial_dofs,
        1.5,
    )
    wrong_value, _wrong_gradient = _cpu_qfm_penalty_value_and_grad(
        biotsavart_cpu,
        surface_cpu,
        Area(surface_cpu),
        target,
        trial_dofs,
        1.5,
    )

    np.testing.assert_allclose(value, expected_value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-10)
    assert not np.isclose(value, wrong_value, rtol=1e-8, atol=1e-12)


def test_qfm_surface_jax_toroidal_flux_uses_label_owned_biotsavart() -> None:
    """Oracle: CPU QFM keeps residual and toroidal-flux label fields separate."""
    (
        qfm_field_cpu,
        label_field_cpu,
        surface_cpu,
        qfm_field_jax,
        label_field_jax,
        surface_jax,
    ) = _qfm_surface_with_distinct_label_field()
    target = 0.94 * ToroidalFlux(surface_cpu, label_field_cpu).J()
    trial_dofs = np.asarray(surface_cpu.get_dofs(), dtype=np.float64)
    trial_dofs = trial_dofs + 0.01 * np.sin(np.arange(trial_dofs.size))
    label_cpu = ToroidalFlux(surface_cpu, label_field_cpu)
    label_jax = ToroidalFlux(surface_jax, label_field_jax)
    qfm_surface = QfmSurfaceJAX(qfm_field_jax, surface_jax, label_jax, target)

    value, gradient = qfm_surface.qfm_penalty_constraints(
        trial_dofs,
        derivatives=1,
        constraint_weight=1.2,
    )
    expected_value, expected_gradient = _cpu_qfm_penalty_value_and_grad(
        qfm_field_cpu,
        surface_cpu,
        label_cpu,
        target,
        trial_dofs,
        1.2,
    )
    wrong_value, _wrong_gradient = _cpu_qfm_penalty_value_and_grad(
        qfm_field_cpu,
        surface_cpu,
        ToroidalFlux(surface_cpu, BiotSavart(qfm_field_cpu.coils)),
        target,
        trial_dofs,
        1.2,
    )

    np.testing.assert_allclose(value, expected_value, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(gradient, expected_gradient, rtol=1e-8, atol=1e-10)
    assert not np.isclose(value, wrong_value, rtol=1e-8, atol=1e-12)


def test_qfm_solvers_report_label_value_on_label_surface_spec() -> None:
    """Real penalty and AL solver info use the label-owned quadrature grid."""
    biotsavart, surface = _make_label_grid_qfm_case()
    _biotsavart_cpu, label_probe_surface = _make_label_grid_qfm_cpu_case()
    label = Area(
        surface,
        nphi=4,
        ntheta=9,
        range=SurfaceRZFourier.RANGE_FULL_TORUS,
    )
    surface_spec = _surface_spec(surface)
    label_spec = _surface_spec(label.surface)
    coil_set_spec = _coil_set_spec(biotsavart)
    init_dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    target = label.J()
    label_probe = Area(
        label_probe_surface,
        nphi=4,
        ntheta=9,
        range=SurfaceRZFourier.RANGE_FULL_TORUS,
    )
    wrong_probe = Area(label_probe_surface)

    penalty_dofs, penalty_info = qfm_penalty_solve_jax(
        surface_spec,
        coil_set_spec,
        "area",
        target,
        1.0,
        init_dofs,
        label_spec=label_spec,
        label_coil_set_spec=coil_set_spec,
        max_iter=0,
        tol=1e-8,
    )
    al_dofs, al_info = qfm_augmented_lagrangian_solve_jax(
        surface_spec,
        coil_set_spec,
        "area",
        target,
        init_dofs,
        label_spec=label_spec,
        label_coil_set_spec=coil_set_spec,
        max_outer=1,
        inner_max_iter=1,
        tol=1e-8,
    )

    expected_penalty_label = _cpu_label_value(label_probe, penalty_dofs)
    wrong_penalty_label = _cpu_label_value(wrong_probe, penalty_dofs)
    expected_al_label = _cpu_label_value(label_probe, al_dofs)
    wrong_al_label = _cpu_label_value(wrong_probe, al_dofs)

    np.testing.assert_allclose(
        host_scalar(penalty_info.label_value),
        expected_penalty_label,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        host_scalar(al_info.label_value),
        expected_al_label,
        rtol=1e-10,
        atol=1e-12,
    )
    assert not np.isclose(
        host_scalar(penalty_info.label_value),
        wrong_penalty_label,
        rtol=1e-8,
        atol=1e-12,
    )
    assert not np.isclose(
        host_scalar(al_info.label_value),
        wrong_al_label,
        rtol=1e-8,
        atol=1e-12,
    )


def test_qfm_surface_jax_solver_receives_label_surface_spec(
    monkeypatch,
    request,
) -> None:
    """Penalty solves preserve label-specific quadrature/range metadata."""
    enable_strict_jax_backend(monkeypatch, request, mode="jax_cpu_parity")
    biotsavart, surface = _make_label_grid_qfm_case()
    label = Area(
        surface,
        nphi=4,
        ntheta=9,
        range=SurfaceRZFourier.RANGE_FULL_TORUS,
    )
    observed = {}

    def fake_penalty_solve(
        spec,
        coil_set_spec,
        label_name,
        targetlabel,
        constraint_weight,
        init_dofs,
        *,
        max_iter,
        tol,
        optimizer,
        toroidal_flux_idx,
        label_spec,
        label_coil_set_spec,
    ):
        observed["surface_grid"] = (
            spec.quadpoints_phi.shape,
            spec.quadpoints_theta.shape,
        )
        observed["label_grid"] = (
            label_spec.quadpoints_phi.shape,
            label_spec.quadpoints_theta.shape,
        )
        final_dofs = jnp.asarray(init_dofs, dtype=jnp.float64)
        return final_dofs, _penalty_info(final_dofs)

    monkeypatch.setattr(
        qfmsurface_jax_module,
        "qfm_penalty_solve_jax",
        fake_penalty_solve,
    )
    qfm_surface = QfmSurfaceJAX(biotsavart, surface, label, label.J())

    qfm_surface.minimize_qfm(method="BFGS", maxiter=3)

    assert observed["surface_grid"] == ((6,), (5,))
    assert observed["label_grid"] == ((4,), (9,))


def test_qfm_surface_jax_penalty_writeback_happens_after_solver(
    monkeypatch,
    request,
) -> None:
    """Adapter writes final DOFs once, outside the pure QFM solve."""
    enable_strict_jax_backend(monkeypatch, request, mode="jax_cpu_parity")
    biotsavart, surface = _make_qfm_case()
    initial_dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    observed = {}

    def fake_penalty_solve(
        spec,
        coil_set_spec,
        label,
        targetlabel,
        constraint_weight,
        init_dofs,
        *,
        max_iter,
        tol,
        optimizer,
        toroidal_flux_idx,
        label_spec,
        label_coil_set_spec,
    ):
        observed["label"] = label
        observed["surface_dofs_during_solver"] = np.asarray(surface.get_dofs())
        final_dofs = jnp.asarray(init_dofs, dtype=jnp.float64) + 1.0e-4
        return final_dofs, _penalty_info(final_dofs)

    monkeypatch.setattr(
        qfmsurface_jax_module,
        "qfm_penalty_solve_jax",
        fake_penalty_solve,
    )
    qfm_surface = QfmSurfaceJAX(
        biotsavart,
        surface,
        Area(surface),
        Area(surface).J(),
    )

    result = qfm_surface.minimize_qfm(method="BFGS", maxiter=3)

    assert observed["label"] == "area"
    np.testing.assert_allclose(observed["surface_dofs_during_solver"], initial_dofs)
    np.testing.assert_allclose(surface.get_dofs(), initial_dofs + 1.0e-4)
    assert result["success"] is True
    assert result["iter"] == 2


def test_qfm_surface_jax_augmented_lagrangian_dispatches_without_slsqp_fallback(
    monkeypatch,
    request,
) -> None:
    """Strict JAX ``AL`` dispatch uses the augmented-Lagrangian solver."""
    enable_strict_jax_backend(monkeypatch, request, mode="jax_cpu_parity")
    biotsavart, surface = _make_label_grid_qfm_case()
    label = Volume(
        surface,
        nphi=4,
        ntheta=9,
        range=SurfaceRZFourier.RANGE_FULL_TORUS,
    )
    calls = []

    def fake_augmented_solve(
        spec,
        coil_set_spec,
        label,
        targetlabel,
        init_dofs,
        *,
        max_outer,
        inner_max_iter,
        tol,
        optimizer,
        toroidal_flux_idx,
        label_spec,
        label_coil_set_spec,
    ):
        calls.append((label, max_outer, inner_max_iter, optimizer))
        calls.append(
            (label_spec.quadpoints_phi.shape, label_spec.quadpoints_theta.shape)
        )
        final_dofs = jnp.asarray(init_dofs, dtype=jnp.float64)
        return final_dofs, _augmented_info(final_dofs)

    def forbidden_native_minimize(self, *args, **kwargs):
        raise AssertionError("QfmSurfaceJAX used native SLSQP in JAX backend mode.")

    monkeypatch.setattr(
        qfmsurface_jax_module,
        "qfm_augmented_lagrangian_solve_jax",
        fake_augmented_solve,
    )
    monkeypatch.setattr(
        qfmsurface_jax_module.QfmSurface,
        "minimize_qfm",
        forbidden_native_minimize,
    )
    qfm_surface = QfmSurfaceJAX(
        biotsavart,
        surface,
        label,
        label.J(),
    )

    result = qfm_surface.minimize_qfm(method="AL", maxiter=4)

    assert calls == [("volume", 4, 1, "bfgs"), ((4,), (9,))]
    assert result["success"] is True
    assert result["fun"] == 0.1


def test_qfm_surface_jax_native_dispatch_rejects_unwired_lm(
    monkeypatch,
) -> None:
    """Native dispatch does not silently route the JAX-only LM method to SLSQP."""
    biotsavart, surface = _make_qfm_case()
    qfm_surface = QfmSurfaceJAX(
        biotsavart,
        surface,
        Area(surface),
        Area(surface).J(),
    )

    def forbidden_native_minimize(self, *args, **kwargs):
        raise AssertionError("LM should fail before entering native SLSQP.")

    monkeypatch.setattr(qfmsurface_jax_module, "is_jax_backend", lambda: False)
    monkeypatch.setattr(
        qfmsurface_jax_module.QfmSurface,
        "minimize_qfm",
        forbidden_native_minimize,
    )

    with np.testing.assert_raises(ValueError):
        qfm_surface.minimize_qfm(method="LM", maxiter=1)


def test_qfm_surface_jax_lazy_geo_export() -> None:
    """Smoke: canonical JAX module can construct and evaluate ``QfmSurfaceJAX``."""
    biotsavart, surface = _make_qfm_case()
    label = Area(surface)
    qfm_surface = QfmSurfaceJAX(biotsavart, surface, label, label.J())

    np.testing.assert_allclose(
        qfm_surface.qfm_label_constraint(surface.x),
        0.0,
        rtol=0.0,
        atol=1.0e-28,
    )
