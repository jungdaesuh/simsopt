"""Reduced nested-LS operator: exact (ι, G) QR projection and Newton on s.

This is the reconstruct physics bar, not F3 fused L-BFGS-B and not banana
``run_code``. The 7×7 NCSX fixture is the same operator as 255×64, not the
same scale.
"""

from __future__ import annotations

import ast
import inspect
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
    NESTED_LS_BANANA_BFGS_TOL,
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
    nested_ls_banana_run_code_options,
    nested_ls_physics_newton_kwargs,
)
from simsopt_jax_adapters.geo.nested_ls_newton_parity import (
    assert_nested_ls_newton_pair,
    evaluate_nested_ls_penalty_pair,
    pack_nested_ls_decision,
    run_nested_ls_newton_pair,
)
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    NESTED_LS_EW_ALPHA,
    NESTED_LS_EW_GAMMA,
    NESTED_LS_EW_SAFEGUARD,
    NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES,
    NESTED_LS_SCHUR_GMRES_MAXITER,
    NESTED_LS_SCHUR_GMRES_MAXITER_CAP,
    NESTED_LS_SCHUR_GMRES_RESTART,
    NESTED_LS_SCHUR_GMRES_RTOL,
    NestedLsReducedRankError,
    apply_reduced_mixed_schur_coil_tangent,
    compare_ad_qr_and_schur_hvp,
    dense_schur_inverse_preconditioner,
    eisenstat_walker_forcing_eta,
    factor_reduced_nested_ls_schur,
    factor_schur_fourier_block_preconditioner,
    implicit_adjoint_coil_gradient,
    linear_solve_meets_forcing,
    materialize_stabilized_schur_dense,
    nested_ls_reduced_closures,
    nested_ls_runtime_coil_closures,
    pack_surface_and_y,
    projected_y_system,
    reduced_penalty_gradient,
    reduced_penalty_gradient_envelope,
    reduced_penalty_hvp,
    require_full_y_rank,
    run_banana_run_code_pair,
    run_reduced_nested_ls_newton,
    run_reduced_nested_ls_schur_newton,
    schur_dense_operator_bytes,
    solve_operator_gmres_with_forcing,
    solve_projected_y,
    solve_stabilized_schur_dense_lu,
    split_surface_and_y,
    tensor_fourier_mode_blocks,
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
    banana = nested_ls_banana_run_code_options()
    assert banana["newton_tol"] == NESTED_LS_BANANA_NEWTON_TOL
    assert banana["newton_maxiter"] == NESTED_LS_BANANA_NEWTON_MAXITER
    assert banana["bfgs_tol"] == NESTED_LS_BANANA_BFGS_TOL
    assert "stab" not in banana


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


class _DummySchur:
    surface_size = 2

    def apply(self, tangent):
        vector = jnp.asarray(tangent, dtype=jnp.float64).reshape(-1)
        return vector


def test_schur_dense_bytes_cap_refuses_oversize():
    assert schur_dense_operator_bytes(37) == 37 * 37 * 8
    assert (
        schur_dense_operator_bytes(661) > NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES
    )

    class _F3DummySchur:
        surface_size = 661

        def apply(self, tangent):
            return jnp.asarray(tangent, dtype=jnp.float64).reshape(-1)

    with pytest.raises(MemoryError, match="max_dense_linearization_bytes="):
        materialize_stabilized_schur_dense(
            _F3DummySchur(),
            1.0e-4,
            max_dense_linearization_bytes=NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES,
        )
    with pytest.raises(MemoryError, match="max_dense_linearization_bytes=1"):
        materialize_stabilized_schur_dense(
            _DummySchur(),
            1.0e-4,
            max_dense_linearization_bytes=1,
        )


def test_materialize_chunk_batch_size_defaults_to_production_ssot():
    signature = inspect.signature(materialize_stabilized_schur_dense)
    assert signature.parameters["chunk_batch_size"].default is None
    newton = inspect.signature(run_reduced_nested_ls_schur_newton)
    assert newton.parameters["linear_solver"].default == "gmres"


def test_implicit_adjoint_defaults_and_dense_lu_honours_1mib_cap():
    signature = inspect.signature(implicit_adjoint_coil_gradient)
    assert signature.parameters["stab"].default == 0.0
    assert signature.parameters["linear_solver"].default == "gmres"
    assert (
        signature.parameters["max_dense_linearization_bytes"].default
        == NESTED_LS_IMPLICIT_ADJOINT_DEFAULT_DENSE_BYTES
        == 1_048_576
    )
    newton = inspect.signature(run_reduced_nested_ls_schur_newton)
    assert newton.parameters["linear_solver"].default == "gmres"
    assert newton.parameters["max_dense_linearization_bytes"].default is None
    assert schur_dense_operator_bytes(661) == 3_495_368

    class _F3DummySchur:
        surface_size = 661

        def apply(self, tangent):
            return jnp.asarray(tangent, dtype=jnp.float64).reshape(-1)

    cotangent = jnp.zeros((661,), dtype=jnp.float64).at[0].set(1.0)
    with pytest.raises(MemoryError, match="max_dense_linearization_bytes="):
        implicit_adjoint_coil_gradient(
            lambda packed, coil: packed,
            lambda packed, coil: jnp.sum(packed),
            jnp.zeros((661,), dtype=jnp.float64),
            jnp.zeros((3,), dtype=jnp.float64),
            cotangent,
            linear_solver="dense_lu",
            operator=_F3DummySchur(),
        )


@pytest.mark.boozer
def test_chunked_dense_hss_a_and_b_match_exact_schur():
    _native_ad, jax_ad, _decision, iota, g0 = _nested_ls_volume_pair()
    _native_lu, jax_lu, _d2, _i2, _g2 = _nested_ls_volume_pair()
    del _native_ad, _native_lu, _d2, _i2, _g2, _decision
    shifted = np.array(jax_ad.surface.get_dofs(), dtype=np.float64, copy=True)
    shifted[0] += 1.0e-3
    jax_ad.surface.set_dofs(shifted)
    jax_lu.surface.set_dofs(shifted)
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_lu)
    del _phi_hat
    y_star = solve_projected_y(
        residual_fn, shifted, np.array([iota, g0], dtype=np.float64)
    ).solution
    operator = factor_reduced_nested_ls_schur(
        residual_fn, objective_fn, shifted, y_probe=y_star
    )
    dense = materialize_stabilized_schur_dense(operator, NESTED_LS_NEWTON_STAB)
    tangent = np.zeros(shifted.size, dtype=np.float64)
    tangent[0] = 1.0
    hv = np.asarray(operator.apply(tangent), dtype=np.float64).reshape(-1)
    hv = hv + NESTED_LS_NEWTON_STAB * tangent
    np.testing.assert_allclose(
        np.asarray(dense, dtype=np.float64) @ tangent,
        hv,
        rtol=1.0e-10,
        atol=1.0e-10,
        err_msg="chunked dense Ĥ_ss missed the exact Schur matvec",
    )
    apply_m = dense_schur_inverse_preconditioner(dense)
    recovered = np.asarray(apply_m(jnp.asarray(hv, dtype=jnp.float64)))
    np.testing.assert_allclose(
        recovered,
        tangent,
        rtol=1.0e-9,
        atol=1.0e-9,
        err_msg="dense inverse M missed B: M(Ĥ+stab I)v = v",
    )
    rhs = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, shifted, y_star),
        dtype=np.float64,
    )
    lu_delta = np.asarray(
        solve_stabilized_schur_dense_lu(dense, jnp.asarray(rhs, dtype=jnp.float64)),
        dtype=np.float64,
    )
    lu_residual = np.asarray(dense, dtype=np.float64) @ lu_delta - rhs
    assert float(np.linalg.norm(lu_residual)) <= 1.0e-8 * float(np.linalg.norm(rhs))
    ad = run_reduced_nested_ls_newton(jax_ad, iota=iota, G=g0, maxiter=1)
    schur = run_reduced_nested_ls_schur_newton(
        jax_lu,
        iota=iota,
        G=g0,
        maxiter=1,
        linear_solver="dense_lu",
    )
    value_tol = parity_ladder_tolerances("direct_kernel")
    grad_tol = parity_ladder_tolerances("ls_wrapper_gradient")
    np.testing.assert_allclose(
        schur.surface_dofs,
        ad.surface_dofs,
        rtol=float(grad_tol["rtol"]),
        atol=float(grad_tol["atol"]),
        err_msg="dense-LU Schur Newton missed AD-through-QR after one step",
    )
    np.testing.assert_allclose(
        schur.iota,
        ad.iota,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    assert schur.gmres_forcing_eta <= 1.0e-8
    assert schur.step_accepted is True


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
    operator = factor_reduced_nested_ls_schur(
        residual_fn, objective_fn, shifted, y_probe=y_star
    )
    predicted = np.asarray(
        operator.apply(schur.gmres_solution), dtype=np.float64
    ).reshape(-1)
    predicted = predicted + NESTED_LS_NEWTON_STAB * np.asarray(
        schur.gmres_solution, dtype=np.float64
    ).reshape(-1)
    independent_residual = predicted - gradient
    independent_l2 = float(np.linalg.norm(independent_residual))
    assert schur.gmres_residual_l2 == pytest.approx(
        independent_l2, rel=1.0e-12, abs=1.0e-12
    )
    assert schur.gmres_forcing_eta == pytest.approx(
        independent_l2 / grad_l2,
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert schur.steps[0].gmres_rtol <= NESTED_LS_SCHUR_GMRES_RTOL
    if linear_solve_meets_forcing(schur.gmres_forcing_eta, schur.steps[0].gmres_rtol):
        assert schur.step_accepted is True
    else:
        assert schur.step_accepted is False
        np.testing.assert_array_equal(schur.surface_dofs, shifted)


def test_eisenstat_walker_choice2_caps_safeguards_and_floors():
    eta0 = eisenstat_walker_forcing_eta(
        1.0,
        previous_grad_norm=None,
        previous_eta=None,
        eta_max=0.24,
        nonlinear_tol=1.0e-13,
    )
    assert eta0 == pytest.approx(0.24, rel=0.0, abs=0.0)
    eta_fast = eisenstat_walker_forcing_eta(
        0.05,
        previous_grad_norm=1.0,
        previous_eta=0.24,
        eta_max=0.24,
        nonlinear_tol=1.0e-13,
    )
    freeze_small = NESTED_LS_EW_GAMMA * 0.24**NESTED_LS_EW_ALPHA
    assert freeze_small < NESTED_LS_EW_SAFEGUARD
    assert eta_fast == pytest.approx(
        NESTED_LS_EW_GAMMA * (0.05) ** NESTED_LS_EW_ALPHA,
        rel=0.0,
        abs=1.0e-15,
    )
    eta_frozen = eisenstat_walker_forcing_eta(
        0.05,
        previous_grad_norm=1.0,
        previous_eta=0.9,
        eta_max=0.9,
        nonlinear_tol=1.0e-13,
    )
    freeze = NESTED_LS_EW_GAMMA * 0.9**NESTED_LS_EW_ALPHA
    assert freeze > NESTED_LS_EW_SAFEGUARD
    assert eta_frozen == pytest.approx(freeze, rel=0.0, abs=1.0e-15)
    eta_floor = eisenstat_walker_forcing_eta(
        1.0e-12,
        previous_grad_norm=1.0,
        previous_eta=0.24,
        eta_max=0.24,
        nonlinear_tol=1.0e-13,
    )
    assert eta_floor == pytest.approx(0.5 * 1.0e-13 / 1.0e-12, rel=0.0, abs=1.0e-15)
    assert linear_solve_meets_forcing(0.24, 0.24) is True
    assert linear_solve_meets_forcing(0.2400001, 0.24) is False


def test_eisenstat_walker_step2_matches_recorded_f3_walk():
    """Choice 2 at the accepted F3 B37 point must request η≈0.05."""

    eta = eisenstat_walker_forcing_eta(
        0.003795976041011014,
        previous_grad_norm=0.01609557303688532,
        previous_eta=0.24,
        eta_max=NESTED_LS_SCHUR_GMRES_RTOL,
        nonlinear_tol=NESTED_LS_NEWTON_TOL,
    )
    assert eta == pytest.approx(0.05005835207396105, rel=0.0, abs=1.0e-12)
    assert eta < 0.24
    assert not linear_solve_meets_forcing(0.1514095713193289, eta)


def test_eisenstat_walker_step4_requested_eta_is_not_the_achieved_eta():
    """Cap-64 walk step 4 requested 0.04071795 and achieved 0.12038811."""

    requested = eisenstat_walker_forcing_eta(
        0.000780997243354473,
        previous_grad_norm=0.003671789660022825,
        previous_eta=0.24,
        eta_max=NESTED_LS_SCHUR_GMRES_RTOL,
        nonlinear_tol=NESTED_LS_NEWTON_TOL,
    )
    achieved = 0.1203881060498997
    assert requested == pytest.approx(0.04071795165373735, rel=0.0, abs=0.0)
    assert requested == pytest.approx(0.0407179517, rel=0.0, abs=5.0e-11)
    assert achieved != pytest.approx(requested, rel=0.0, abs=1.0e-6)
    assert not linear_solve_meets_forcing(achieved, requested)


def test_schur_gmres_defaults_are_one_restart_cycle_not_one_krylov_step():
    """JAX ``gmres`` ``maxiter`` counts restart cycles, not Krylov vectors.

    Default ``maxiter=1``, ``restart=8`` is one cycle of up to eight
    Krylov iterations. ``gmres_matvecs`` stays 0 because incremental
    GMRES does not report operator applications: unavailable telemetry,
    not zero work.
    """

    assert NESTED_LS_SCHUR_GMRES_RESTART == 8
    assert NESTED_LS_SCHUR_GMRES_MAXITER == 1
    assert NESTED_LS_SCHUR_GMRES_MAXITER_CAP == 512


def test_operator_gmres_rejects_unit_interval_eta_then_retries():
    weights = jnp.linspace(1.0, 1.0e-3, 16)

    def matvec(tangent: jax.Array) -> jax.Array:
        return weights * tangent

    rhs = jnp.ones((16,), dtype=jnp.float64)
    _delta, _residual, _info, _residual_l2, eta_one, used_one = (
        solve_operator_gmres_with_forcing(
            matvec,
            rhs,
            eta_requested=NESTED_LS_SCHUR_GMRES_RTOL,
            restart=2,
            maxiter=1,
            maxiter_cap=1,
        )
    )
    assert used_one == 1
    assert eta_one > NESTED_LS_SCHUR_GMRES_RTOL
    assert eta_one <= 1.0
    assert linear_solve_meets_forcing(eta_one, NESTED_LS_SCHUR_GMRES_RTOL) is False
    _delta2, _residual2, _info2, _residual_l2_2, eta_retry, used_retry = (
        solve_operator_gmres_with_forcing(
            matvec,
            rhs,
            eta_requested=NESTED_LS_SCHUR_GMRES_RTOL,
            restart=2,
            maxiter=1,
            maxiter_cap=NESTED_LS_SCHUR_GMRES_MAXITER_CAP,
        )
    )
    assert used_retry > 1
    assert eta_retry <= NESTED_LS_SCHUR_GMRES_RTOL
    assert linear_solve_meets_forcing(eta_retry, NESTED_LS_SCHUR_GMRES_RTOL) is True


@pytest.mark.boozer
def test_schur_newton_rejects_eta_above_requested_forcing():
    _native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    del _native, _decision
    shifted = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
    shifted[0] += 1.0e-3
    jax_boozer.surface.set_dofs(shifted)
    schur = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=iota,
        G=g0,
        maxiter=1,
        gmres_restart=8,
        gmres_maxiter=1,
        gmres_maxiter_cap=1,
        gmres_rtol=1.0e-12,
    )
    assert schur.gmres_forcing_eta > 1.0e-12
    assert schur.step_accepted is False
    assert schur.step_alpha == pytest.approx(1.0, rel=0.0, abs=0.0)
    np.testing.assert_array_equal(schur.surface_dofs, shifted)
    assert schur.gmres_maxiter == 1


@pytest.mark.boozer
def test_schur_newton_retries_krylov_until_forcing_is_met():
    _native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    del _native, _decision
    shifted = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
    shifted[0] += 1.0e-3
    jax_boozer.surface.set_dofs(shifted)
    schur = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=iota,
        G=g0,
        maxiter=1,
        gmres_restart=8,
        gmres_maxiter=1,
        gmres_maxiter_cap=NESTED_LS_SCHUR_GMRES_MAXITER_CAP,
        gmres_rtol=NESTED_LS_SCHUR_GMRES_RTOL,
    )
    assert schur.gmres_forcing_eta <= NESTED_LS_SCHUR_GMRES_RTOL
    assert schur.step_accepted is True
    assert schur.gmres_maxiter > 1
    assert schur.steps[0].gmres_rtol == pytest.approx(
        NESTED_LS_SCHUR_GMRES_RTOL, rel=0.0, abs=0.0
    )


@pytest.mark.boozer
def test_fourier_block_m_canary_on_exact_schur():
    _native_plain, jax_plain, _decision_p, iota, g0 = _nested_ls_volume_pair()
    _native_prec, jax_prec, _decision_m, _iota_m, _g_m = _nested_ls_volume_pair()
    del _native_plain, _native_prec, _decision_p, _decision_m, _iota_m, _g_m
    shifted = np.array(jax_plain.surface.get_dofs(), dtype=np.float64, copy=True)
    shifted[0] += 1.0e-3
    jax_plain.surface.set_dofs(shifted)
    jax_prec.surface.set_dofs(shifted)
    names = tuple(jax_plain.surface.local_full_dof_names)
    assert len(names) == shifted.size
    mpol = int(jax_plain.surface.mpol)
    ntor = int(jax_plain.surface.ntor)
    blocks = tensor_fourier_mode_blocks(names, mpol=mpol, ntor=ntor)
    covered = [index for block in blocks for index in block]
    assert sorted(covered) == list(range(shifted.size))
    assert {len(block) for block in blocks} == {1, 3, 6}
    assert len(blocks) == (mpol + 1) * (ntor + 1)

    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_prec)
    del _phi_hat
    y_start = np.array([float(iota), float(g0)], dtype=np.float64)
    y_star = solve_projected_y(residual_fn, shifted, y_start).solution
    gradient = np.asarray(
        reduced_penalty_gradient_envelope(objective_fn, shifted, y_star),
        dtype=np.float64,
    )
    operator = factor_reduced_nested_ls_schur(
        residual_fn, objective_fn, shifted, y_probe=y_star
    )
    preconditioner = factor_schur_fourier_block_preconditioner(
        operator,
        NESTED_LS_NEWTON_STAB,
        names,
        mpol=mpol,
        ntor=ntor,
    )
    probe_block = max(preconditioner.blocks, key=len)
    inverse = preconditioner.inverses[preconditioner.blocks.index(probe_block)]
    index_array = np.asarray(probe_block)
    columns = []
    for dof_index in probe_block:
        unit = np.zeros(shifted.size, dtype=np.float64)
        unit[int(dof_index)] = 1.0
        hv = np.asarray(operator.apply(unit), dtype=np.float64).reshape(-1)
        hv = hv + NESTED_LS_NEWTON_STAB * unit
        columns.append(hv[index_array])
    probed = np.stack(columns, axis=1)
    block_size = len(probe_block)
    np.testing.assert_allclose(
        np.asarray(inverse, dtype=np.float64) @ probed,
        np.eye(block_size, dtype=np.float64),
        rtol=1.0e-10,
        atol=1.0e-10,
        err_msg="Fourier-block inverse missed the exact Schur block",
    )

    plain = run_reduced_nested_ls_schur_newton(
        jax_plain,
        iota=iota,
        G=g0,
        maxiter=1,
        gmres_maxiter_cap=1,
    )
    prec = run_reduced_nested_ls_schur_newton(
        jax_prec,
        iota=iota,
        G=g0,
        maxiter=1,
        gmres_maxiter_cap=1,
        gmres_preconditioner=preconditioner.apply,
    )
    assert plain.gmres_forcing_eta > 1.0
    assert plain.step_accepted is False
    assert prec.gmres_forcing_eta < plain.gmres_forcing_eta
    predicted = np.asarray(
        operator.apply(prec.gmres_solution), dtype=np.float64
    ).reshape(-1)
    predicted = predicted + NESTED_LS_NEWTON_STAB * np.asarray(
        prec.gmres_solution, dtype=np.float64
    ).reshape(-1)
    independent_l2 = float(np.linalg.norm(predicted - gradient))
    assert prec.gmres_residual_l2 == pytest.approx(
        independent_l2, rel=1.0e-12, abs=1.0e-12
    )
    assert prec.gmres_forcing_eta < plain.gmres_forcing_eta


def test_dense_lu_newton_certifies_live_matvec_not_dense_product():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "simsopt_jax_adapters"
        / "geo"
        / "nested_ls_reduced.py"
    )
    source = source_path.read_text(encoding="utf-8")
    branch = source.split('elif linear_solver == "dense_lu":', 1)[1]
    newton_part = branch.split("else:", 1)[0]
    assert "residual = matvec(newton_jax) - rhs" in newton_part
    assert "residual = dense @ newton_jax - rhs" not in newton_part


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


def _nested_ls_banana_pair():
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
    banana_options = nested_ls_banana_run_code_options()
    native = BoozerSurface(
        biotsavart,
        native_surface,
        native_label,
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options=dict(banana_options),
    )
    jax_boozer = BoozerSurfaceJAX(
        BiotSavartJAX(biotsavart.coils),
        jax_surface,
        jax_label,
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options={
            **banana_options,
            "optimizer_backend": "scipy",
            "materialize_dense_linearization": True,
        },
    )
    g0 = _g_from_currents(base_currents, nfp)
    return native, jax_boozer, _IOTA0, g0


@pytest.mark.boozer
def test_schur_newton_walk_records_accepted_steps():
    _native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    del _native, _decision
    shifted = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
    shifted[0] += 1.0e-3
    jax_boozer.surface.set_dofs(shifted)
    walk = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=iota,
        G=g0,
        maxiter=3,
        gmres_restart=64,
        gmres_maxiter=10,
        gmres_rtol=1.0e-10,
    )
    assert walk.coil_delta_inf == 0.0
    assert walk.iteration_count >= 2
    assert walk.steps
    assert walk.steps[0].step_accepted is True
    assert walk.steps[0].iteration == 1
    assert walk.steps[-1].grad_l2 < walk.steps[0].grad_l2
    assert len(walk.steps) == walk.iteration_count or (
        len(walk.steps) == walk.iteration_count + 1
        and walk.steps[-1].step_accepted is False
    )
    assert all(
        step.gmres_forcing_eta <= step.gmres_rtol
        for step in walk.steps
        if step.step_accepted
    )


@pytest.mark.boozer
def test_schur_newton_reaches_reconstruct_gradient_tol():
    native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    del _decision
    seed_iota, seed_g = _seed_both_lanes_from_native_lbfgs(native, jax_boozer, iota, g0)
    shifted = np.array(jax_boozer.surface.get_dofs(), dtype=np.float64, copy=True)
    shifted[0] += 1.0e-4
    jax_boozer.surface.set_dofs(shifted)
    walk = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=seed_iota,
        G=seed_g,
        maxiter=NESTED_LS_NEWTON_MAXITER,
        gmres_restart=64,
        gmres_maxiter=10,
        gmres_rtol=1.0e-10,
    )
    assert walk.coil_delta_inf == 0.0
    grad_l2 = float(np.linalg.norm(walk.reduced_gradient))
    assert walk.success is True
    assert grad_l2 <= NESTED_LS_NEWTON_TOL
    assert all(
        step.gmres_forcing_eta <= step.gmres_rtol
        for step in walk.steps
        if step.step_accepted
    )


@pytest.mark.boozer
def test_runtime_coil_closures_match_captured_and_mixed_hvp():
    _native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    del _native, _decision, iota, g0
    residual_fn, objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    residual_rt, objective_rt, phi_hat_rt = nested_ls_runtime_coil_closures(jax_boozer)
    surface = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    coil = np.asarray(jax.device_get(jax_boozer.biotsavart.x), dtype=np.float64)
    solution = solve_projected_y(residual_fn, surface)
    require_full_y_rank(solution)
    packed = pack_surface_and_y(surface, solution.solution)
    value_tol = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(
        np.asarray(objective_rt(packed, coil), dtype=np.float64),
        np.asarray(objective_fn(packed), dtype=np.float64),
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="runtime-coil Φ missed captured-coil Φ",
    )
    np.testing.assert_allclose(
        np.asarray(residual_rt(packed, coil), dtype=np.float64),
        np.asarray(residual_fn(packed), dtype=np.float64),
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
        err_msg="runtime-coil residual missed captured-coil residual",
    )
    tangent = None
    mixed = None
    phi_unperturbed = float(np.asarray(objective_rt(packed, coil), dtype=np.float64))
    for index in range(min(12, coil.size)):
        candidate = np.zeros_like(coil)
        candidate[index] = 1.0
        phi_perturbed = float(
            np.asarray(
                objective_rt(packed, coil + 1.0e-3 * candidate), dtype=np.float64
            )
        )
        mixed_candidate = np.asarray(
            apply_reduced_mixed_schur_coil_tangent(
                residual_rt,
                objective_rt,
                surface,
                coil,
                candidate,
            ),
            dtype=np.float64,
        )
        if (
            abs(phi_perturbed - phi_unperturbed) > 0.0
            and float(np.linalg.norm(mixed_candidate)) > 1.0e-8
        ):
            tangent = candidate
            mixed = mixed_candidate
            break
    assert tangent is not None
    assert mixed is not None
    epsilon = 1.0e-6
    plus = jax.grad(lambda s: phi_hat_rt(s, coil + epsilon * tangent))(
        jnp.asarray(surface, dtype=jnp.float64)
    )
    minus = jax.grad(lambda s: phi_hat_rt(s, coil))(
        jnp.asarray(surface, dtype=jnp.float64)
    )
    finite_difference = np.asarray((plus - minus) / epsilon, dtype=np.float64)
    derivative_tol = parity_ladder_tolerances("derivative_heavy")
    np.testing.assert_allclose(
        mixed,
        finite_difference,
        rtol=float(derivative_tol["second_derivative_rtol"]),
        atol=max(float(derivative_tol["second_derivative_atol"]), 1.0e-6),
        err_msg="mixed Ĥ_sc v_c missed FD of ∇_s Φ̂",
    )


@pytest.mark.boozer
def test_implicit_adjoint_matches_surface_response_to_coil_step():
    native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    del _decision
    seed_iota, seed_g = _seed_both_lanes_from_native_lbfgs(native, jax_boozer, iota, g0)
    polished = run_reduced_nested_ls_newton(jax_boozer, iota=seed_iota, G=seed_g)
    assert polished.success is True
    residual_rt, objective_rt, _phi_hat = nested_ls_runtime_coil_closures(jax_boozer)
    surface0 = np.array(polished.surface_dofs, dtype=np.float64, copy=True)
    coil0 = np.asarray(jax.device_get(jax_boozer.biotsavart.x), dtype=np.float64)
    cotangent = np.zeros_like(surface0)
    cotangent[0] = 1.0
    tangent = None
    mixed_dc = None
    mixed_norm = -1.0
    for index in range(min(8, coil0.size)):
        candidate = np.zeros_like(coil0)
        candidate[index] = 1.0
        mixed_candidate = np.asarray(
            apply_reduced_mixed_schur_coil_tangent(
                residual_rt,
                objective_rt,
                surface0,
                coil0,
                candidate,
            ),
            dtype=np.float64,
        )
        candidate_norm = float(np.linalg.norm(mixed_candidate))
        if candidate_norm > mixed_norm:
            mixed_norm = candidate_norm
            tangent = candidate
            mixed_dc = mixed_candidate
    assert tangent is not None
    assert mixed_dc is not None
    assert mixed_norm > 1.0e-8
    adjoint = np.asarray(
        implicit_adjoint_coil_gradient(
            residual_rt,
            objective_rt,
            surface0,
            coil0,
            cotangent,
            stab=NESTED_LS_NEWTON_STAB,
            linear_solver="dense_lu",
        ),
        dtype=np.float64,
    )
    operator = factor_reduced_nested_ls_schur(
        lambda packed: residual_rt(packed, coil0),
        lambda packed: objective_rt(packed, coil0),
        surface0,
    )
    dense = materialize_stabilized_schur_dense(operator, NESTED_LS_NEWTON_STAB)
    lambda_s = np.asarray(
        solve_stabilized_schur_dense_lu(dense, jnp.asarray(cotangent)),
        dtype=np.float64,
    )
    predicted = -float(np.dot(lambda_s, mixed_dc))
    predicted_step = np.asarray(
        -solve_stabilized_schur_dense_lu(dense, jnp.asarray(mixed_dc)),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        float(np.dot(adjoint, tangent)),
        predicted,
        rtol=1.0e-8,
        atol=1.0e-10,
        err_msg="coil-gradient VJP missed -λᵀ Ĥ_sc v_c",
    )
    predicted_step_norm = float(np.linalg.norm(predicted_step))
    assert predicted_step_norm > 1.0e-8
    epsilon = 1.0e-6
    jax_boozer.surface.set_dofs(surface0)
    jax_boozer.biotsavart.x = coil0
    jax_boozer._refresh_coil_data()
    control = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(polished.iota),
        G=float(polished.G),
        maxiter=1,
        gmres_restart=64,
        gmres_maxiter=10,
        gmres_rtol=1.0e-10,
    )
    jax_boozer.surface.set_dofs(surface0)
    jax_boozer.biotsavart.x = coil0 + epsilon * tangent
    jax_boozer._refresh_coil_data()
    perturbed = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(polished.iota),
        G=float(polished.G),
        maxiter=1,
        gmres_restart=64,
        gmres_maxiter=10,
        gmres_rtol=1.0e-10,
    )
    finite_difference = (perturbed.surface_dofs - control.surface_dofs) / epsilon
    assert control.coil_delta_inf == 0.0
    assert perturbed.coil_delta_inf == 0.0
    jax_boozer.biotsavart.x = coil0
    jax_boozer._refresh_coil_data()
    np.testing.assert_allclose(
        predicted_step,
        finite_difference,
        rtol=5.0e-2,
        atol=max(1.0e-9, 1.0e-2 * predicted_step_norm),
        err_msg="implicit adjoint missed one Schur step of ds/dc",
    )


@pytest.mark.boozer
def test_unregularized_ift_adjoint_matches_reconverged_surface_fd():
    native, jax_boozer, _decision, iota, g0 = _nested_ls_volume_pair()
    del _decision
    seed_iota, seed_g = _seed_both_lanes_from_native_lbfgs(native, jax_boozer, iota, g0)
    polished = run_reduced_nested_ls_newton(jax_boozer, iota=seed_iota, G=seed_g)
    assert polished.success is True
    residual_rt, objective_rt, _phi_hat = nested_ls_runtime_coil_closures(jax_boozer)
    surface0 = np.array(polished.surface_dofs, dtype=np.float64, copy=True)
    coil0 = np.asarray(jax.device_get(jax_boozer.biotsavart.x), dtype=np.float64)
    cotangent = np.zeros_like(surface0)
    cotangent[0] = 1.0
    tangent = None
    mixed_dc = None
    mixed_norm = -1.0
    for index in range(min(8, coil0.size)):
        candidate = np.zeros_like(coil0)
        candidate[index] = 1.0
        mixed_candidate = np.asarray(
            apply_reduced_mixed_schur_coil_tangent(
                residual_rt,
                objective_rt,
                surface0,
                coil0,
                candidate,
            ),
            dtype=np.float64,
        )
        candidate_norm = float(np.linalg.norm(mixed_candidate))
        if candidate_norm > mixed_norm:
            mixed_norm = candidate_norm
            tangent = candidate
            mixed_dc = mixed_candidate
    assert tangent is not None
    assert mixed_dc is not None
    assert mixed_norm > 1.0e-8
    adjoint = np.asarray(
        implicit_adjoint_coil_gradient(
            residual_rt,
            objective_rt,
            surface0,
            coil0,
            cotangent,
            max_dense_linearization_bytes=1,
        ),
        dtype=np.float64,
    )
    operator = factor_reduced_nested_ls_schur(
        lambda packed: residual_rt(packed, coil0),
        lambda packed: objective_rt(packed, coil0),
        surface0,
    )
    dense = materialize_stabilized_schur_dense(operator, 0.0)
    lambda_s = np.asarray(
        solve_stabilized_schur_dense_lu(dense, jnp.asarray(cotangent)),
        dtype=np.float64,
    )
    predicted = -float(np.dot(lambda_s, mixed_dc))
    predicted_step = np.asarray(
        -solve_stabilized_schur_dense_lu(dense, jnp.asarray(mixed_dc)),
        dtype=np.float64,
    )
    np.testing.assert_allclose(
        float(np.dot(adjoint, tangent)),
        predicted,
        rtol=1.0e-6,
        atol=1.0e-8,
        err_msg="unregularized coil VJP missed -λᵀ Ĥ_ss⁻¹ Ĥ_sc v_c",
    )
    predicted_step_norm = float(np.linalg.norm(predicted_step))
    assert predicted_step_norm > 1.0e-8
    epsilon = 1.0e-6
    jax_boozer.surface.set_dofs(surface0)
    jax_boozer.biotsavart.x = coil0
    jax_boozer._refresh_coil_data()
    control = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(polished.iota),
        G=float(polished.G),
        maxiter=NESTED_LS_NEWTON_MAXITER,
        gmres_restart=64,
        gmres_maxiter=10,
        gmres_rtol=1.0e-10,
    )
    assert control.success is True
    jax_boozer.surface.set_dofs(surface0)
    jax_boozer.biotsavart.x = coil0 + epsilon * tangent
    jax_boozer._refresh_coil_data()
    perturbed = run_reduced_nested_ls_schur_newton(
        jax_boozer,
        iota=float(polished.iota),
        G=float(polished.G),
        maxiter=NESTED_LS_NEWTON_MAXITER,
        gmres_restart=64,
        gmres_maxiter=10,
        gmres_rtol=1.0e-10,
    )
    assert perturbed.success is True
    finite_difference = (perturbed.surface_dofs - control.surface_dofs) / epsilon
    assert control.coil_delta_inf == 0.0
    assert perturbed.coil_delta_inf == 0.0
    jax_boozer.biotsavart.x = coil0
    jax_boozer._refresh_coil_data()
    np.testing.assert_allclose(
        predicted_step,
        finite_difference,
        rtol=5.0e-2,
        atol=max(1.0e-9, 1.0e-2 * predicted_step_norm),
        err_msg="unregularized IFT missed FD of reconverged ds/dc",
    )


@pytest.mark.boozer
def test_b3_banana_run_code_matches_native():
    native, jax_boozer, iota, g0 = _nested_ls_banana_pair()
    pair = run_banana_run_code_pair(native, jax_boozer, iota=iota, G=g0)
    assert pair.native.coil_delta_inf == 0.0
    assert pair.jax.coil_delta_inf == 0.0
    assert pair.native.success is True
    assert pair.jax.success is True
    assert pair.physics_matched is True
    value_tol = parity_ladder_tolerances("direct_kernel")
    np.testing.assert_allclose(
        pair.jax.iota,
        pair.native.iota,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
    np.testing.assert_allclose(
        pair.jax.G,
        pair.native.G,
        rtol=float(value_tol["rtol"]),
        atol=float(value_tol["atol"]),
    )
