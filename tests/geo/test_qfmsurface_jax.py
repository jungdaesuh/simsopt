"""QFM JAX solver orchestration tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from conftest import enable_strict_jax_backend, host_array, host_scalar

from simsopt.configs.zoo import get_data
from simsopt.field.biotsavart_jax_backend import BiotSavartJAX
from simsopt.geo import qfmsurface_jax as qfmsurface_jax_module
from simsopt.geo.qfmsurface_jax import QfmSurfaceJAX
from simsopt.geo.surfaceobjectives import Area, Volume
from simsopt.geo.surfacerzfourier import SurfaceRZFourier
from simsopt.jax_core.qfm_solver import (
    QfmAugmentedLagrangianInfo,
    QfmPenaltySolveInfo,
    qfm_augmented_lagrangian_solve_jax,
    qfm_penalty_jax_from_dofs,
    qfm_penalty_solve_jax,
    qfm_penalty_value_and_grad_jax_from_dofs,
    qfm_residual_jax_from_dofs,
)


def _make_qfm_case():
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
    return BiotSavartJAX(biotsavart.coils), surface


def _coil_set_spec(biotsavart):
    return biotsavart.coil_set_spec_from_dofs(
        jnp.asarray(biotsavart.x, dtype=jnp.float64)
    )


def _make_qfm_inputs():
    biotsavart, surface = _make_qfm_case()
    dofs = jnp.asarray(surface.get_dofs(), dtype=jnp.float64)
    return biotsavart, surface, dofs, _coil_set_spec(biotsavart)


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
        surface.surface_spec(),
        dofs,
        coil_set_spec,
        label="area",
        targetlabel=target,
        constraint_weight=1.0,
    )

    final_dofs, info = qfm_penalty_solve_jax(
        surface.surface_spec(),
        coil_set_spec,
        "area",
        target,
        1.0,
        dofs,
        max_iter=5,
        tol=1e-8,
    )

    assert final_dofs.shape == dofs.shape
    assert host_scalar(info.penalty_value) < host_scalar(initial)
    assert host_array(info.gradient).shape == tuple(dofs.shape)


def test_qfm_penalty_solve_jax_transfer_guard_clean() -> None:
    """The BFGS solver core does not enter JAX's host-staging optimizer path."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.98 * Area(surface).J()

    with jax.transfer_guard("disallow"):
        final_dofs, info = qfm_penalty_solve_jax(
            surface.surface_spec(),
            coil_set_spec,
            "area",
            target,
            1.0,
            dofs,
            max_iter=1,
            tol=1e-8,
        )

    assert final_dofs.shape == dofs.shape
    assert host_array(info.gradient).shape == tuple(dofs.shape)


def test_qfm_augmented_lagrangian_solve_jax_transfer_guard_clean() -> None:
    """The AL wrapper keeps scalar updates and inner BFGS staging on device."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = Area(surface).J()

    with jax.transfer_guard("disallow"):
        final_dofs, info = qfm_augmented_lagrangian_solve_jax(
            surface.surface_spec(),
            coil_set_spec,
            "area",
            target,
            dofs,
            max_outer=1,
            inner_max_iter=1,
            tol=1e-8,
        )

    assert final_dofs.shape == dofs.shape
    assert host_array(info.gradient).shape == tuple(dofs.shape)


def test_qfm_augmented_lagrangian_info_reports_qfm_gradient() -> None:
    """The exact-path result pairs QFM ``fun`` with the QFM objective gradient."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = Area(surface).J()

    final_dofs, info = qfm_augmented_lagrangian_solve_jax(
        surface.surface_spec(),
        coil_set_spec,
        "area",
        target,
        dofs,
        max_outer=1,
        inner_max_iter=1,
        tol=1e-8,
    )
    expected_gradient = jax.grad(
        lambda surface_dofs: qfm_residual_jax_from_dofs(
            surface.surface_spec(),
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


def test_qfm_augmented_lagrangian_info_uses_final_inner_objective_state() -> None:
    """AL diagnostics describe the objective minimized by the final inner solve."""
    _biotsavart, surface, dofs, coil_set_spec = _make_qfm_inputs()
    target = 0.99 * Area(surface).J()

    _final_dofs, info = qfm_augmented_lagrangian_solve_jax(
        surface.surface_spec(),
        coil_set_spec,
        "area",
        target,
        dofs,
        max_outer=1,
        inner_max_iter=1,
        tol=1e-8,
    )

    np.testing.assert_allclose(
        host_scalar(info.augmented_value),
        host_scalar(info.fun),
        rtol=1e-12,
        atol=1e-14,
    )
    np.testing.assert_allclose(host_scalar(info.multiplier), 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        host_scalar(info.penalty_weight),
        10.0,
        rtol=0.0,
        atol=0.0,
    )


def test_qfm_surface_jax_penalty_value_and_gradient_do_not_mutate_surface() -> None:
    """Oracle: pure value/grad helper for the same trial surface DOFs."""
    biotsavart, surface = _make_qfm_case()
    initial_dofs = np.asarray(surface.get_dofs(), dtype=np.float64)
    trial_dofs = initial_dofs.copy()
    trial_dofs[0] += 1.0e-3
    target = 0.99 * Area(surface).J()
    qfm_surface = QfmSurfaceJAX(biotsavart, surface, Area(surface), target)

    value, gradient = qfm_surface.qfm_penalty_constraints(
        trial_dofs,
        derivatives=1,
        constraint_weight=1.5,
    )
    expected_value, expected_gradient = qfm_penalty_value_and_grad_jax_from_dofs(
        surface.surface_spec(),
        jnp.asarray(trial_dofs, dtype=jnp.float64),
        _coil_set_spec(biotsavart),
        label="area",
        targetlabel=target,
        constraint_weight=1.5,
    )

    np.testing.assert_allclose(value, host_scalar(expected_value), rtol=1e-12)
    np.testing.assert_allclose(gradient, host_array(expected_gradient), rtol=1e-10)
    np.testing.assert_allclose(surface.get_dofs(), initial_dofs, rtol=0.0, atol=0.0)


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
    biotsavart, surface = _make_qfm_case()
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
    ):
        calls.append((label, max_outer, inner_max_iter, optimizer))
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
        Volume(surface),
        Volume(surface).J(),
    )

    result = qfm_surface.minimize_qfm(method="AL", maxiter=4)

    assert calls == [("volume", 4, 1, "bfgs")]
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
    """The public geo package lazily exports ``QfmSurfaceJAX``."""
    from simsopt.geo import QfmSurfaceJAX as exported

    assert exported is QfmSurfaceJAX
