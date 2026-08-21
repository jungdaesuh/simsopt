"""Reduced nested-LS operator: exact (ι, G) QR projection and Newton on s.

This is the reconstruct physics bar, not F3 fused L-BFGS-B and not banana
``run_code``. The 7×7 NCSX fixture is the same operator as 255×64, not the
same scale.
"""

from __future__ import annotations

import ast
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from simsopt.configs.zoo import get_data
from simsopt.geo import BoozerSurface, Volume
from simsopt_jax.parity_tolerances import parity_ladder_tolerances
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.boozer_surface import BoozerSurfaceJAX
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_BANANA_NEWTON_MAXITER,
    NESTED_LS_BANANA_NEWTON_STAB,
    NESTED_LS_BANANA_NEWTON_TOL,
    NESTED_LS_BANANA_USES_BFGS_THEN_NEWTON,
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_NEWTON_MAXITER,
    NESTED_LS_NEWTON_STAB,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_PHYSICS_BAR,
    NESTED_LS_TIMING_BAR,
    NESTED_LS_WEIGHT_INV_MODB,
    nested_ls_physics_newton_kwargs,
)
from simsopt_jax_adapters.geo.nested_ls_newton_parity import (
    assert_nested_ls_newton_pair,
    evaluate_nested_ls_penalty_pair,
    pack_nested_ls_decision,
    run_nested_ls_newton_pair,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    NESTED_LS_SCHUR_GMRES_RTOL,
    NestedLsReducedRankError,
    compare_ad_qr_and_schur_hvp,
    nested_ls_reduced_closures,
    pack_surface_and_y,
    projected_y_system,
    reduced_penalty_gradient,
    reduced_penalty_gradient_envelope,
    reduced_penalty_hvp,
    require_full_y_rank,
    run_reduced_nested_ls_newton,
    run_reduced_nested_ls_schur_newton,
    solve_projected_y,
    split_surface_and_y,
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
        raise AssertionError("native LBFGS seed for reduced nested-LS failed.")
    jax_boozer.surface.set_dofs(native.surface.get_dofs())
    return float(polished["iota"]), float(polished["G"])


def test_nested_ls_contract_keeps_banana_off_the_physics_bar():
    assert NESTED_LS_PHYSICS_BAR == "reconstruct_newton"
    assert NESTED_LS_TIMING_BAR == "banana_run_code"
    assert NESTED_LS_BANANA_USES_BFGS_THEN_NEWTON is True
    assert NESTED_LS_NEWTON_STAB != NESTED_LS_BANANA_NEWTON_STAB
    assert NESTED_LS_NEWTON_TOL != NESTED_LS_BANANA_NEWTON_TOL
    assert NESTED_LS_NEWTON_MAXITER != NESTED_LS_BANANA_NEWTON_MAXITER
    assert NESTED_LS_BANANA_NEWTON_STAB == 0.0
    assert NESTED_LS_BANANA_NEWTON_TOL == 1.0e-11
    kwargs = nested_ls_physics_newton_kwargs()
    assert kwargs["stab"] == NESTED_LS_NEWTON_STAB
    assert kwargs["tol"] == NESTED_LS_NEWTON_TOL
    assert kwargs["maxiter"] == NESTED_LS_NEWTON_MAXITER


def test_rank_gate_rejects_rank_deficient_iota_g_columns():
    def residual_fn(packed):
        y = packed[-2:]
        return y[0] * jnp.ones(6, dtype=jnp.float64)

    surface = jnp.ones(4, dtype=jnp.float64)
    solution = solve_projected_y(
        residual_fn, surface, jnp.asarray([0.4, -0.2], dtype=jnp.float64)
    )
    with pytest.raises(NestedLsReducedRankError, match="rank=1"):
        require_full_y_rank(solution)


@pytest.mark.boozer
def test_nested_ls_residual_is_affine_in_iota_and_g():
    _native, jax_boozer, decision, _iota, _g0 = _nested_ls_volume_pair()
    del _native
    residual_fn, _objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    surface, y0 = split_surface_and_y(decision)
    y1 = y0 + jnp.asarray([0.07, -0.03], dtype=jnp.float64)
    matrix0, rhs0 = projected_y_system(residual_fn, surface, y0)
    matrix1, rhs1 = projected_y_system(residual_fn, surface, y1)
    value_tol = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(
        np.asarray(matrix1),
        np.asarray(matrix0),
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="nested-LS residual columns changed with y; they must be affine",
    )
    np.testing.assert_allclose(
        np.asarray(rhs1),
        np.asarray(rhs0),
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="nested-LS residual intercept changed with y; they must be affine",
    )
    residual0 = np.asarray(residual_fn(pack_surface_and_y(surface, y0)))
    predicted = np.asarray(matrix0 @ y0 - rhs0)
    np.testing.assert_allclose(
        residual0,
        predicted,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="r(s, y) != A y - b",
    )


@pytest.mark.boozer
def test_projected_y_minimizes_penalty_at_frozen_surface():
    _native, jax_boozer, decision, _iota, _g0 = _nested_ls_volume_pair()
    del _native
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    surface, y0 = split_surface_and_y(decision)
    solution = solve_projected_y(residual_fn, surface, y0)
    require_full_y_rank(solution)
    assert int(np.asarray(solution.numerical_rank)) == 2
    y_star = solution.solution
    phi_star = float(objective_fn(pack_surface_and_y(surface, y_star)))
    for shift in ((0.02, 0.0), (0.0, -0.03), (0.015, 0.02)):
        trial = y_star + jnp.asarray(shift, dtype=jnp.float64)
        phi_trial = float(objective_fn(pack_surface_and_y(surface, trial)))
        assert phi_star <= phi_trial + 1.0e-12


@pytest.mark.boozer
def test_reduced_gradient_and_hvp_match_finite_differences():
    _native, jax_boozer, decision, _iota, _g0 = _nested_ls_volume_pair()
    del _native
    _residual_fn, _objective_fn, phi_hat = nested_ls_reduced_closures(jax_boozer)
    surface, _y = split_surface_and_y(decision)
    gradient = np.asarray(reduced_penalty_gradient(phi_hat, surface), dtype=np.float64)
    step = 1.0e-7
    probe_indices = (0, min(3, int(surface.size) - 1), int(surface.size) - 1)
    for index in probe_indices:
        basis = np.zeros(int(surface.size), dtype=np.float64)
        basis[index] = step
        plus = float(phi_hat(surface + basis))
        minus = float(phi_hat(surface - basis))
        np.testing.assert_allclose(
            gradient[index],
            (plus - minus) / (2.0 * step),
            rtol=1.0e-5,
            atol=1.0e-7,
            err_msg=f"reduced ∇_s Φ̂ finite-difference mismatch at index {index}",
        )
    tangent = np.zeros(int(surface.size), dtype=np.float64)
    tangent[0] = 1.0
    hvp = np.asarray(reduced_penalty_hvp(phi_hat, surface, tangent), dtype=np.float64)
    plus_grad = np.asarray(
        reduced_penalty_gradient(phi_hat, surface + step * tangent), dtype=np.float64
    )
    minus_grad = np.asarray(
        reduced_penalty_gradient(phi_hat, surface - step * tangent), dtype=np.float64
    )
    np.testing.assert_allclose(
        hvp,
        (plus_grad - minus_grad) / (2.0 * step),
        rtol=1.0e-4,
        atol=1.0e-6,
        err_msg="reduced H_ss v finite-difference mismatch",
    )


@pytest.mark.boozer
def test_envelope_gradient_matches_ad_through_qr():
    _native, jax_boozer, decision, _iota, _g0 = _nested_ls_volume_pair()
    del _native
    residual_fn, objective_fn, phi_hat = nested_ls_reduced_closures(jax_boozer)
    surface, _y = split_surface_and_y(decision)
    solution = solve_projected_y(residual_fn, surface)
    require_full_y_rank(solution)
    ad_grad = np.asarray(reduced_penalty_gradient(phi_hat, surface), dtype=np.float64)
    envelope = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, surface, solution.solution),
        dtype=np.float64,
    )
    packed = pack_surface_and_y(surface, solution.solution)
    phi_y = np.asarray(jax.grad(objective_fn)(packed)[-2:], dtype=np.float64)
    value_tol = parity_ladder_tolerances("ls_wrapper_gradient")
    np.testing.assert_allclose(
        envelope,
        ad_grad,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="envelope ∇_s Φ(s, y*) missed AD-through-QR ∇_s Φ̂",
    )
    np.testing.assert_allclose(
        phi_y,
        0.0,
        atol=1.0e-10,
        err_msg="Φ_y at QR y* is not a stationary inner point",
    )


@pytest.mark.boozer
def test_schur_hvp_matches_ad_through_qr():
    _native, jax_boozer, decision, _iota, _g0 = _nested_ls_volume_pair()
    del _native
    residual_fn, objective_fn, phi_hat = nested_ls_reduced_closures(jax_boozer)
    surface, _y = split_surface_and_y(decision)
    gradient = np.asarray(reduced_penalty_gradient(phi_hat, surface), dtype=np.float64)
    directions = [
        np.eye(int(surface.size), dtype=np.float64)[0],
        gradient / np.linalg.norm(gradient),
    ]
    derivative_tol = parity_ladder_tolerances("derivative_heavy")
    for tangent in directions:
        comparison = compare_ad_qr_and_schur_hvp(
            residual_fn, objective_fn, phi_hat, surface, tangent
        )
        assert comparison.phi_yy_condition > 0.0
        np.testing.assert_allclose(
            comparison.schur,
            comparison.ad_through_qr,
            rtol=float(derivative_tol["second_derivative_rtol"]),
            atol=float(derivative_tol["second_derivative_atol"]),
            err_msg="Schur Ĥ_ss v missed the AD-through-QR HVP",
        )


@pytest.mark.boozer
def test_schur_newton_one_step_matches_ad_through_qr():
    _native_ad, jax_ad, _decision, iota, g0 = _nested_ls_volume_pair()
    _native_schur, jax_schur, _decision_schur, _iota_s, _g_s = _nested_ls_volume_pair()
    del _native_ad, _native_schur, _decision_schur, _iota_s, _g_s
    shifted = np.array(jax_ad.surface.get_dofs(), dtype=np.float64, copy=True)
    shifted[0] += 1.0e-3
    jax_ad.surface.set_dofs(shifted)
    jax_schur.surface.set_dofs(shifted)
    ad = run_reduced_nested_ls_newton(jax_ad, iota=iota, G=g0, maxiter=1)
    schur = run_reduced_nested_ls_schur_newton(
        jax_schur,
        iota=iota,
        G=g0,
        maxiter=1,
        gmres_restart=64,
        gmres_maxiter=10,
        gmres_rtol=1.0e-10,
    )
    assert ad.coil_delta_inf == 0.0
    assert schur.coil_delta_inf == 0.0
    assert schur.step_accepted is True
    assert schur.iteration_count == 1
    value_tol = parity_ladder_tolerances("direct_kernel")
    grad_tol = parity_ladder_tolerances("ls_wrapper_gradient")
    np.testing.assert_allclose(
        schur.iota,
        ad.iota,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="Schur Newton iota missed AD-through-QR Newton after one step",
    )
    np.testing.assert_allclose(
        schur.surface_dofs,
        ad.surface_dofs,
        rtol=float(grad_tol["rtol"]),
        atol=float(grad_tol["atol"]),
        err_msg="Schur Newton surface missed AD-through-QR Newton after one step",
    )
    assert schur.gmres_rtol == pytest.approx(1.0e-10, rel=0.0, abs=0.0)
    assert schur.gmres_info in (0, -1)
    assert schur.gmres_forcing_eta <= 1.0e-8


@pytest.mark.boozer
def test_schur_newton_forcing_eta_is_explicit_linear_residual_ratio():
    _native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    del _native, _decision
    shifted = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
    shifted[0] += 1.0e-3
    jax_boozer.surface.set_dofs(shifted)
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    y_start = np.array([float(iota), float(g0)], dtype=np.float64)
    y_star = solve_projected_y(residual_fn, shifted, y_start).solution
    gradient = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, shifted, y_star),
        dtype=np.float64,
    )
    grad_l2 = float(np.linalg.norm(gradient))
    assert grad_l2 > 0.0
    assert NESTED_LS_SCHUR_GMRES_RTOL == pytest.approx(0.24, rel=0.0, abs=0.0)
    schur = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=iota,
        G=g0,
        maxiter=1,
    )
    assert schur.gmres_rtol == pytest.approx(0.24, rel=0.0, abs=0.0)
    assert schur.gmres_info in (0, -1)
    assert schur.gmres_matvecs == 0
    assert schur.gmres_forcing_eta == pytest.approx(
        schur.gmres_residual_l2 / grad_l2,
        rel=1.0e-12,
        abs=1.0e-12,
    )


def test_schur_newton_module_does_not_import_host_scipy_gmres():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "simsopt_jax_adapters"
        / "geo"
        / "nested_ls_reduced.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    scipy_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (
            node.module == "scipy.sparse.linalg"
            or (
                node.module is not None
                and node.module.startswith("scipy.sparse.linalg.")
            )
        ):
            scipy_names.extend(alias.name for alias in node.names)
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "scipy.sparse.linalg" or alias.name.startswith(
                    "scipy.sparse.linalg."
                ):
                    scipy_names.append(alias.name)
    assert "gmres" not in scipy_names
    assert "LinearOperator" not in scipy_names
    assert all(not name.startswith("scipy.sparse.linalg") for name in scipy_names)


@pytest.mark.boozer
def test_reduced_newton_matches_native_on_manifold():
    native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    seed_iota, seed_g = _seed_both_lanes_from_native_lbfgs(native, jax_boozer, iota, g0)
    pair = run_nested_ls_newton_pair(
        native_boozer=native,
        jax_boozer=jax_boozer,
        iota=seed_iota,
        G=seed_g,
    )
    assert_nested_ls_newton_pair(pair)
    jax_boozer.surface.set_dofs(pair.native.surface_dofs)
    reduced = run_reduced_nested_ls_newton(jax_boozer, iota=seed_iota, G=seed_g)
    assert reduced.coil_delta_inf == 0.0
    assert reduced.success is True
    assert reduced.persisted is True
    assert reduced.y_rank == 2
    assert reduced.reduced_gradient.shape == reduced.surface_dofs.shape
    value_tol = parity_ladder_tolerances("direct_kernel")
    grad_tol = parity_ladder_tolerances("ls_wrapper_gradient")
    np.testing.assert_allclose(
        reduced.iota,
        pair.native.iota,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    np.testing.assert_allclose(
        reduced.G,
        pair.native.G,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    np.testing.assert_allclose(
        reduced.surface_dofs,
        pair.native.surface_dofs,
        rtol=float(grad_tol["rtol"]),
        atol=float(grad_tol["atol"]),
    )
    packed = pack_nested_ls_decision(reduced.surface_dofs, reduced.iota, reduced.G)
    penalty = evaluate_nested_ls_penalty_pair(
        native_boozer=native,
        jax_boozer=jax_boozer,
        decision=packed,
    )
    np.testing.assert_allclose(
        penalty.jax.objective,
        pair.native.objective,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )


@pytest.mark.boozer
def test_reduced_newton_matches_native_from_a_surface_step():
    native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    seed_iota, seed_g = _seed_both_lanes_from_native_lbfgs(native, jax_boozer, iota, g0)
    shifted = np.array(native.surface.get_dofs(), dtype=np.float64, copy=True)
    shifted[0] += 1.0e-3
    native.surface.set_dofs(shifted)
    jax_boozer.surface.set_dofs(shifted)
    pair = run_nested_ls_newton_pair(
        native_boozer=native,
        jax_boozer=jax_boozer,
        iota=seed_iota,
        G=seed_g,
    )
    assert_nested_ls_newton_pair(pair)
    jax_boozer.surface.set_dofs(shifted)
    reduced = run_reduced_nested_ls_newton(jax_boozer, iota=seed_iota, G=seed_g)
    assert pair.native.iteration_count >= 1
    assert reduced.coil_delta_inf == 0.0
    assert reduced.success is True
    assert reduced.reduced_gradient.shape == reduced.surface_dofs.shape
    value_tol = parity_ladder_tolerances("direct_kernel")
    grad_tol = parity_ladder_tolerances("ls_wrapper_gradient")
    np.testing.assert_allclose(
        reduced.iota,
        pair.native.iota,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    np.testing.assert_allclose(
        reduced.G,
        pair.native.G,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    np.testing.assert_allclose(
        reduced.surface_dofs,
        pair.native.surface_dofs,
        rtol=float(grad_tol["rtol"]),
        atol=float(grad_tol["atol"]),
    )
    np.testing.assert_allclose(
        np.linalg.norm(reduced.reduced_gradient),
        0.0,
        atol=NESTED_LS_NEWTON_TOL,
        err_msg="reduced Newton did not reach ||∇_s Φ̂||_2 ≤ 1e-13",
    )
