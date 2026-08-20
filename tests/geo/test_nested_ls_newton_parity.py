"""Nested Boozer LS Newton: native C++ vs JAX on frozen coils.

Canary-1 lock for GPU nested-LS parity. This is the reconstruct inner
problem (scalar ``J_LS``, ``∇J_LS``, Volume penalty, free ``G``), not
flat-675 QR-in-``J`` and not F3's 7.7× timing claim.
"""

from __future__ import annotations

import numpy as np
import pytest
from simsopt.configs.zoo import get_data
from simsopt.geo import BoozerSurface, Volume
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.nested_ls_newton_parity import (
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_NEWTON_MAXITER,
    NESTED_LS_NEWTON_STAB,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_WEIGHT_INV_MODB,
    assert_nested_ls_newton_pair,
    assert_nested_ls_penalty_pair,
    evaluate_nested_ls_penalty_pair,
    jax_newton_residual_is_long_vector,
    pack_nested_ls_decision,
    run_nested_ls_newton_pair,
)

from .boozersurface_jax_test_helpers import _clone_upstream_surface
from .surface_test_helpers import get_surface

_IOTA0 = -0.406


def _g_from_currents(base_currents, nfp: int) -> float:
    current_sum = nfp * sum(abs(current.get_value()) for current in base_currents)
    return 2.0 * np.pi * current_sum * (4.0 * np.pi * 1.0e-7 / (2.0 * np.pi))


def _nested_ls_volume_pair():
    _, base_currents, magnetic_axis, nfp, biotsavart = get_data("ncsx")
    surface = get_surface(
        "SurfaceXYZTensorFourier",
        True,
        mpol=2,
        ntor=2,
        nphi=7,
        ntheta=7,
        nfp=nfp,
    )
    surface.fit_to_curve(magnetic_axis, 0.1, flip_theta=True)
    native_surface = _clone_upstream_surface(surface)
    jax_surface = _clone_upstream_surface(surface)
    native_label = Volume(native_surface)
    jax_label = Volume(jax_surface)
    target = float(native_label.J())
    newton_options = {
        "verbose": False,
        "newton_tol": NESTED_LS_NEWTON_TOL,
        "newton_maxiter": NESTED_LS_NEWTON_MAXITER,
        "weight_inv_modB": NESTED_LS_WEIGHT_INV_MODB,
    }
    native = BoozerSurface(
        biotsavart,
        native_surface,
        native_label,
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options=newton_options,
    )
    jax_boozer = BoozerSurfaceJAX(
        BiotSavartJAX(biotsavart.coils),
        jax_surface,
        jax_label,
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options={
            **newton_options,
            "optimizer_backend": "scipy",
            "materialize_dense_linearization": True,
        },
    )
    g0 = _g_from_currents(base_currents, nfp)
    decision = pack_nested_ls_decision(native_surface.get_dofs(), _IOTA0, g0)
    return native, jax_boozer, decision, _IOTA0, g0


def _seed_both_lanes_from_native_lbfgs(native, jax_boozer, iota, g0):
    native.need_to_run_code = True
    polished = native.minimize_boozer_penalty_constraints_LBFGS(
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        iota=float(iota),
        G=float(g0),
        tol=1e-10,
        maxiter=400,
        verbose=False,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )
    if not bool(polished["success"]):
        raise AssertionError("native LBFGS seed for nested-LS Newton failed.")
    jax_boozer.surface.set_dofs(native.surface.get_dofs())
    return float(polished["iota"]), float(polished["G"])


def _initial_ls_newton_value_and_grad(jax_boozer, x0):
    from simsopt_jax_adapters.geo.boozer_surface import (
        _ls_newton_objective_value_and_grad,
    )

    return _ls_newton_objective_value_and_grad(
        jax_boozer._make_penalty_objective_with(
            True,
            NESTED_LS_WEIGHT_INV_MODB,
            NESTED_LS_CONSTRAINT_WEIGHT,
            decision_split_mode="jvp",
        ),
        x0,
        (),
    )


def _shifted_decision(x0, iota, g0, *, d_iota=0.5, d_g=0.25, d_s0=1.0e-3):
    moved = np.array(x0, dtype=np.float64, copy=True)
    moved[0] = float(moved[0]) + d_s0
    moved[-2] = float(iota) + d_iota
    moved[-1] = float(g0) + d_g
    return moved


def _run_mocked_jax_ls_newton(jax_boozer, iota, g0, monkeypatch, polish):
    monkeypatch.setattr(
        jax_boozer,
        "_run_newton_polish_for_method",
        lambda *_args, **_kwargs: polish,
    )
    jax_boozer.need_to_run_code = True
    return jax_boozer.minimize_boozer_penalty_constraints_newton(
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        iota=float(iota),
        G=float(g0),
        tol=NESTED_LS_NEWTON_TOL,
        maxiter=NESTED_LS_NEWTON_MAXITER,
        stab=NESTED_LS_NEWTON_STAB,
        verbose=False,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )


@pytest.mark.boozer
def test_nested_ls_penalty_matches_native_at_packed_state():
    native, jax_boozer, decision, _iota, _g0 = _nested_ls_volume_pair()
    pair = evaluate_nested_ls_penalty_pair(
        native_boozer=native,
        jax_boozer=jax_boozer,
        decision=decision,
    )
    assert pair.native.gradient.shape[0] == decision.size
    assert_nested_ls_penalty_pair(pair)


@pytest.mark.boozer
def test_nested_ls_newton_matches_native_on_frozen_coils():
    native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    seed_iota, seed_g = _seed_both_lanes_from_native_lbfgs(native, jax_boozer, iota, g0)
    pair = run_nested_ls_newton_pair(
        native_boozer=native,
        jax_boozer=jax_boozer,
        iota=seed_iota,
        G=seed_g,
    )
    assert_nested_ls_newton_pair(pair)


@pytest.mark.boozer
def test_jax_ls_newton_rolls_back_when_gradient_worsens(monkeypatch):
    _native, jax_boozer, decision, iota, g0 = _nested_ls_volume_pair()
    del _native
    trusted_dofs = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
    x0 = np.asarray(jax_boozer._pack_decision_vector(iota, g0), dtype=np.float64)
    initial_value, initial_grad = _initial_ls_newton_value_and_grad(jax_boozer, x0)
    worse_grad = np.asarray(initial_grad, dtype=np.float64) * 10.0
    result = _run_mocked_jax_ls_newton(
        jax_boozer,
        iota,
        g0,
        monkeypatch,
        {
            "x": _shifted_decision(x0, iota, g0),
            "fun": 1.0,
            "grad": worse_grad,
            "hessian": np.eye(decision.size, dtype=np.float64),
            "nit": 1,
            "success": False,
        },
    )
    assert result["success"] is False
    np.testing.assert_allclose(float(result["iota"]), float(iota))
    np.testing.assert_allclose(float(result["G"]), float(g0))
    np.testing.assert_allclose(float(result["fun"]), float(initial_value))
    np.testing.assert_allclose(
        np.asarray(result["jacobian"], dtype=np.float64),
        np.asarray(initial_grad, dtype=np.float64),
    )
    assert result["hessian"] is None
    np.testing.assert_allclose(
        float(result["final_gradient_norm"]),
        float(np.linalg.norm(np.asarray(initial_grad, dtype=np.float64))),
    )
    np.testing.assert_allclose(
        np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64),
        trusted_dofs,
    )


@pytest.mark.boozer
def test_jax_ls_newton_commits_successful_iterate_even_if_gradient_worsens(
    monkeypatch,
):
    _native, jax_boozer, decision, iota, g0 = _nested_ls_volume_pair()
    del _native
    x0 = np.asarray(jax_boozer._pack_decision_vector(iota, g0), dtype=np.float64)
    _initial_value, initial_grad = _initial_ls_newton_value_and_grad(jax_boozer, x0)
    del _initial_value
    worse_grad = np.asarray(initial_grad, dtype=np.float64) * 10.0
    result = _run_mocked_jax_ls_newton(
        jax_boozer,
        iota,
        g0,
        monkeypatch,
        {
            "x": _shifted_decision(x0, iota, g0),
            "fun": 1.0,
            "grad": worse_grad,
            "hessian": np.eye(decision.size, dtype=np.float64),
            "nit": 1,
            "success": True,
        },
    )
    assert result["success"] is True
    np.testing.assert_allclose(float(result["iota"]), float(iota) + 0.5)
    np.testing.assert_allclose(float(result["G"]), float(g0) + 0.25)
    np.testing.assert_allclose(float(result["fun"]), 1.0)
    np.testing.assert_allclose(
        np.asarray(result["jacobian"], dtype=np.float64),
        worse_grad,
    )
    np.testing.assert_allclose(
        float(jax_boozer.surface.get_dofs()[0]),
        float(x0[0]) + 1.0e-3,
    )
    assert result["hessian"] is not None


@pytest.mark.boozer
def test_jax_ls_newton_commits_unmoved_iterate_even_if_gradient_worsens(
    monkeypatch,
):
    _native, jax_boozer, decision, iota, g0 = _nested_ls_volume_pair()
    del _native
    trusted_dofs = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
    x0 = np.asarray(jax_boozer._pack_decision_vector(iota, g0), dtype=np.float64)
    _initial_value, initial_grad = _initial_ls_newton_value_and_grad(jax_boozer, x0)
    del _initial_value
    worse_grad = np.asarray(initial_grad, dtype=np.float64) * 10.0
    hessian = np.eye(decision.size, dtype=np.float64)
    result = _run_mocked_jax_ls_newton(
        jax_boozer,
        iota,
        g0,
        monkeypatch,
        {
            "x": np.array(x0, dtype=np.float64, copy=True),
            "fun": 1.0,
            "grad": worse_grad,
            "hessian": hessian,
            "nit": 0,
            "success": False,
        },
    )
    assert result["success"] is False
    np.testing.assert_allclose(float(result["iota"]), float(iota))
    np.testing.assert_allclose(float(result["G"]), float(g0))
    np.testing.assert_allclose(
        np.asarray(result["jacobian"], dtype=np.float64),
        worse_grad,
    )
    assert result["hessian"] is not None
    np.testing.assert_allclose(
        np.asarray(result["hessian"], dtype=np.float64),
        hessian,
    )
    np.testing.assert_allclose(
        np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64),
        trusted_dofs,
    )


@pytest.mark.boozer
def test_jax_ls_newton_commits_when_hessian_is_nonfinite_but_iterate_is_finite(
    monkeypatch,
):
    _native, jax_boozer, decision, iota, g0 = _nested_ls_volume_pair()
    del _native
    x0 = np.asarray(jax_boozer._pack_decision_vector(iota, g0), dtype=np.float64)
    _initial_value, initial_grad = _initial_ls_newton_value_and_grad(jax_boozer, x0)
    del _initial_value
    better_grad = np.asarray(initial_grad, dtype=np.float64) * 0.1
    result = _run_mocked_jax_ls_newton(
        jax_boozer,
        iota,
        g0,
        monkeypatch,
        {
            "x": _shifted_decision(x0, iota, g0),
            "fun": 0.1,
            "grad": better_grad,
            "hessian": np.full(
                (decision.size, decision.size), np.nan, dtype=np.float64
            ),
            "nit": 1,
            "success": True,
        },
    )
    assert result["success"] is True
    np.testing.assert_allclose(float(result["iota"]), float(iota) + 0.5)
    np.testing.assert_allclose(float(result["G"]), float(g0) + 0.25)
    np.testing.assert_allclose(
        np.asarray(result["jacobian"], dtype=np.float64),
        better_grad,
    )
    assert result["hessian"] is None
    np.testing.assert_allclose(
        float(jax_boozer.surface.get_dofs()[0]),
        float(x0[0]) + 1.0e-3,
    )


@pytest.mark.boozer
def test_jax_ls_newton_rolls_back_nonfinite_gradient_even_if_polish_reports_success(
    monkeypatch,
):
    _native, jax_boozer, decision, iota, g0 = _nested_ls_volume_pair()
    del _native
    trusted_dofs = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
    x0 = np.asarray(jax_boozer._pack_decision_vector(iota, g0), dtype=np.float64)
    initial_value, initial_grad = _initial_ls_newton_value_and_grad(jax_boozer, x0)
    nan_grad = np.full(decision.size, np.nan, dtype=np.float64)
    result = _run_mocked_jax_ls_newton(
        jax_boozer,
        iota,
        g0,
        monkeypatch,
        {
            "x": _shifted_decision(x0, iota, g0),
            "fun": 0.0,
            "grad": nan_grad,
            "hessian": np.eye(decision.size, dtype=np.float64),
            "nit": 1,
            "success": True,
        },
    )
    assert result["success"] is False
    np.testing.assert_allclose(float(result["iota"]), float(iota))
    np.testing.assert_allclose(float(result["G"]), float(g0))
    np.testing.assert_allclose(float(result["fun"]), float(initial_value))
    np.testing.assert_allclose(
        np.asarray(result["jacobian"], dtype=np.float64),
        np.asarray(initial_grad, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64),
        trusted_dofs,
    )


@pytest.mark.boozer
def test_jax_ls_newton_rolls_back_nonfinite_iterate_even_if_polish_reports_success(
    monkeypatch,
):
    _native, jax_boozer, decision, iota, g0 = _nested_ls_volume_pair()
    del _native
    trusted_dofs = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
    x0 = np.asarray(jax_boozer._pack_decision_vector(iota, g0), dtype=np.float64)
    initial_value, initial_grad = _initial_ls_newton_value_and_grad(jax_boozer, x0)
    moved = _shifted_decision(x0, iota, g0)
    moved[-2] = np.nan
    result = _run_mocked_jax_ls_newton(
        jax_boozer,
        iota,
        g0,
        monkeypatch,
        {
            "x": moved,
            "fun": 0.0,
            "grad": np.asarray(initial_grad, dtype=np.float64) * 0.1,
            "hessian": np.eye(decision.size, dtype=np.float64),
            "nit": 1,
            "success": True,
        },
    )
    assert result["success"] is False
    np.testing.assert_allclose(float(result["iota"]), float(iota))
    np.testing.assert_allclose(float(result["G"]), float(g0))
    np.testing.assert_allclose(float(result["fun"]), float(initial_value))
    np.testing.assert_allclose(
        np.asarray(result["jacobian"], dtype=np.float64),
        np.asarray(initial_grad, dtype=np.float64),
    )
    np.testing.assert_allclose(
        np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64),
        trusted_dofs,
    )


@pytest.mark.boozer
def test_jax_ls_newton_commits_when_gradient_improves_without_success(monkeypatch):
    _native, jax_boozer, decision, iota, g0 = _nested_ls_volume_pair()
    del _native
    x0 = np.asarray(jax_boozer._pack_decision_vector(iota, g0), dtype=np.float64)
    _initial_value, initial_grad = _initial_ls_newton_value_and_grad(jax_boozer, x0)
    del _initial_value
    better_grad = np.asarray(initial_grad, dtype=np.float64) * 0.1
    result = _run_mocked_jax_ls_newton(
        jax_boozer,
        iota,
        g0,
        monkeypatch,
        {
            "x": _shifted_decision(x0, iota, g0),
            "fun": 0.1,
            "grad": better_grad,
            "hessian": np.eye(decision.size, dtype=np.float64),
            "nit": 1,
            "success": False,
        },
    )
    assert result["success"] is False
    np.testing.assert_allclose(float(result["iota"]), float(iota) + 0.5)
    np.testing.assert_allclose(float(result["G"]), float(g0) + 0.25)
    np.testing.assert_allclose(float(result["fun"]), 0.1)
    np.testing.assert_allclose(
        np.asarray(result["jacobian"], dtype=np.float64),
        better_grad,
    )
    np.testing.assert_allclose(
        float(jax_boozer.surface.get_dofs()[0]),
        float(x0[0]) + 1.0e-3,
    )


@pytest.mark.boozer
def test_jax_ls_newton_persist_uses_gradient_norm_not_objective(monkeypatch):
    _native, jax_boozer, decision, iota, g0 = _nested_ls_volume_pair()
    del _native
    x0 = np.asarray(jax_boozer._pack_decision_vector(iota, g0), dtype=np.float64)
    initial_value, initial_grad = _initial_ls_newton_value_and_grad(jax_boozer, x0)
    better_grad = np.asarray(initial_grad, dtype=np.float64) * 0.1
    worse_fun = float(initial_value) + 1.0e6
    result = _run_mocked_jax_ls_newton(
        jax_boozer,
        iota,
        g0,
        monkeypatch,
        {
            "x": _shifted_decision(x0, iota, g0),
            "fun": worse_fun,
            "grad": better_grad,
            "hessian": np.eye(decision.size, dtype=np.float64),
            "nit": 1,
            "success": False,
        },
    )
    assert result["success"] is False
    np.testing.assert_allclose(float(result["iota"]), float(iota) + 0.5)
    np.testing.assert_allclose(float(result["G"]), float(g0) + 0.25)
    np.testing.assert_allclose(float(result["fun"]), worse_fun)
    np.testing.assert_allclose(
        np.asarray(result["jacobian"], dtype=np.float64),
        better_grad,
    )
    np.testing.assert_allclose(
        float(jax_boozer.surface.get_dofs()[0]),
        float(x0[0]) + 1.0e-3,
    )


@pytest.mark.boozer
def test_nested_ls_newton_matches_native_from_nearby_iota():
    native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    seed_iota, seed_g = _seed_both_lanes_from_native_lbfgs(native, jax_boozer, iota, g0)
    pair = run_nested_ls_newton_pair(
        native_boozer=native,
        jax_boozer=jax_boozer,
        iota=seed_iota + 1.0e-3,
        G=seed_g,
    )
    assert pair.native.iteration_count >= 1
    assert pair.jax.iteration_count >= 1
    assert_nested_ls_newton_pair(pair)


@pytest.mark.boozer
def test_newton_residual_semantics_differ_across_lanes():
    native, jax_boozer, decision, iota, g0 = _nested_ls_volume_pair()
    seed_iota, seed_g = _seed_both_lanes_from_native_lbfgs(native, jax_boozer, iota, g0)
    native.need_to_run_code = True
    native_result = native.minimize_boozer_penalty_constraints_newton(
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        iota=seed_iota,
        G=seed_g,
        tol=NESTED_LS_NEWTON_TOL,
        maxiter=NESTED_LS_NEWTON_MAXITER,
        stab=NESTED_LS_NEWTON_STAB,
        verbose=False,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )
    jax_boozer.need_to_run_code = True
    jax_result = jax_boozer.minimize_boozer_penalty_constraints_newton(
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        iota=seed_iota,
        G=seed_g,
        tol=NESTED_LS_NEWTON_TOL,
        maxiter=NESTED_LS_NEWTON_MAXITER,
        stab=NESTED_LS_NEWTON_STAB,
        verbose=False,
        weight_inv_modB=NESTED_LS_WEIGHT_INV_MODB,
    )
    native_residual = np.asarray(native_result["residual"], dtype=np.float64)
    native_jacobian = np.asarray(native_result["jacobian"], dtype=np.float64)
    np.testing.assert_array_equal(native_residual, native_jacobian)
    assert native_residual.shape == (decision.size,)
    assert jax_newton_residual_is_long_vector(jax_result, decision_size=decision.size)
