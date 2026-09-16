"""A real small-grid correction and the public acceptance boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import simsopt_jax_adapters.geo.flat675.polish as polish_module
from simsopt.configs.zoo import get_data
from simsopt.geo import BoozerSurface, SurfaceRZFourier, SurfaceXYZTensorFourier, Volume
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX
from simsopt_jax_adapters.geo.flat675 import (
    Flat675AcceptanceLimits,
    build_flat675_problem,
    polish_flat675,
)
from simsopt_jax_adapters.geo.flat675.nested_bridge import nested_view_from_flat675
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_BANANA_NEWTON_MAXITER,
    NESTED_LS_CONSTRAINT_WEIGHT,
    NESTED_LS_JAX_INNER_STAB,
    NESTED_LS_NEWTON_TOL,
    NESTED_LS_WEIGHT_INV_MODB,
    nested_ls_banana_run_code_options,
)


def _near_stationary_problem():
    _, base_currents, axis, nfp, biotsavart = get_data("ncsx")
    grid = 7
    surface = SurfaceXYZTensorFourier(
        mpol=2,
        ntor=2,
        nfp=nfp,
        stellsym=True,
        quadpoints_phi=np.linspace(0.0, 1.0 / nfp, grid, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, grid, endpoint=False),
    )
    surface.fit_to_curve(axis, 0.1, flip_theta=False)
    target = float(Volume(surface).J())
    native = BoozerSurface(
        biotsavart,
        surface,
        Volume(surface),
        target,
        constraint_weight=NESTED_LS_CONSTRAINT_WEIGHT,
        options=dict(nested_ls_banana_run_code_options()),
    )
    current_sum = 2 * nfp * sum(abs(current.get_value()) for current in base_currents)
    G_seed = 4.0e-7 * np.pi * current_sum
    seed = native.run_code(0.406, G=G_seed)
    assert seed["success"], "NCSX native seed failed to converge"

    boundary = SurfaceRZFourier(
        nfp=nfp,
        stellsym=True,
        mpol=2,
        ntor=2,
        quadpoints_phi=np.linspace(0.0, 1.0, 16, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 16, endpoint=False),
    )
    boundary.set_rc(0, 0, 1.5)
    boundary.set_rc(1, 0, 0.3)
    boundary.set_zs(1, 0, 0.3)
    problem = build_flat675_problem(
        boundary=boundary,
        field=BiotSavartJAX(biotsavart.coils),
        mpol=2,
        ntor=2,
        nphi=grid,
        ntheta=grid,
    )
    policy = replace(problem.objective_policy, boozer_target_label=target)
    problem = replace(problem, objective_policy=policy)
    incoming = np.asarray(problem.start_candidate.outer_vector(), dtype=np.float64)
    incoming[problem.material.layout.surface_slice] = np.asarray(surface.get_dofs()) + (
        1.0e-4
        * np.random.default_rng(20260915).standard_normal(surface.get_dofs().size)
    )
    return problem, incoming


@pytest.fixture(scope="module")
def real_polish():
    problem, incoming = _near_stationary_problem()
    retained = incoming.copy()
    polished = polish_flat675(problem, incoming)
    assert np.array_equal(incoming, retained), "polish changed its caller's vector"
    assert incoming.flags.writeable, "polish froze its caller's vector"
    assert not np.shares_memory(incoming, polished.outer_vector)
    assert polished.solver_success, polished.rejection_reasons
    assert polished.acceptance_status == "not_assessed", polished.rejection_reasons
    return problem, incoming, polished


@pytest.mark.boozer
def test_polish_reports_real_native_full_decision_residual(real_polish) -> None:
    problem, incoming, polished = real_polish
    view = nested_view_from_flat675(problem, incoming)
    gamma_before = np.array(view.surface_native.gamma(), dtype=np.float64, copy=True)
    native_boozer = BoozerSurface(
        view.biotsavart_native,
        view.surface_native,
        Volume(view.surface_native),
        float(problem.objective_policy.boozer_target_label),
        constraint_weight=view.jax_inputs.constraint_weight,
    )
    packed = np.concatenate(
        (
            polished.outer_vector[view.layout.surface_slice],
            (polished.iota_after, polished.G_after),
        )
    )
    _, native_gradient = native_boozer.boozer_penalty_constraints_vectorized(
        packed,
        derivatives=1,
        constraint_weight=view.jax_inputs.constraint_weight,
        optimize_G=True,
        weight_inv_modB=True,
    )
    native_l2 = float(np.linalg.norm(native_gradient))
    assert np.isclose(native_l2, polished.full_gradient_l2_after, rtol=1e-8, atol=1e-12)
    view.surface_native.set_dofs(polished.outer_vector[view.layout.surface_slice])
    gamma = np.array(view.surface_native.gamma(), dtype=np.float64, copy=True)
    point_moves = np.linalg.norm(gamma - gamma_before, axis=-1)
    assert np.isclose(
        polished.surface_displacement_max_m, np.max(point_moves), rtol=1e-12
    )
    assert np.isclose(
        polished.surface_displacement_rms_m,
        np.sqrt(np.mean(point_moves**2)),
        rtol=1e-12,
    )
    view.biotsavart_native.set_points(gamma.reshape((-1, 3)))
    field = np.asarray(view.biotsavart_native.B()).reshape(gamma.shape)
    magnitude = np.linalg.norm(field, axis=-1)
    residual = (
        polished.G_after * field
        - magnitude[..., None] ** 2
        * (
            np.asarray(view.surface_native.gammadash1())
            + polished.iota_after * np.asarray(view.surface_native.gammadash2())
        )
    ) / magnitude[..., None]
    assert np.isclose(
        np.sqrt(np.mean(residual**2)), polished.boozer_weighted_rms_after, rtol=1e-9
    )
    assert np.isclose(
        abs(
            float(Volume(view.surface_native).J())
            - problem.objective_policy.boozer_target_label
        ),
        polished.absolute_label_error_after,
        rtol=1e-12,
        atol=1e-13,
    )
    assert np.array_equal(
        polished.outer_vector[view.layout.coil_slice], incoming[view.layout.coil_slice]
    )
    assert np.array_equal(
        polished.outer_vector[view.layout.vessel_slice],
        incoming[view.layout.vessel_slice],
    )
    assert polished.outer_vector.flags.writeable is False
    assert (
        polished.as_dict()["corrected_outer_vector"] == polished.outer_vector.tolist()
    )


def test_acceptance_limits_are_explicit_and_finite() -> None:
    with pytest.raises(ValueError, match="max_boozer_weighted_rms"):
        Flat675AcceptanceLimits(float("nan"), 1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="max_objective_increase"):
        Flat675AcceptanceLimits(1.0, 1.0, 1.0, -1.0)


@pytest.mark.boozer
def test_strict_design_limit_rejects_real_correction(real_polish) -> None:
    problem, incoming, _ = real_polish
    accepted = polish_flat675(
        problem, incoming, limits=Flat675AcceptanceLimits(1e9, 1e9, 1e9, 1e9)
    )
    assert accepted.acceptance_status == "accepted", accepted.rejection_reasons
    limits = Flat675AcceptanceLimits(
        max_boozer_weighted_rms=0.0,
        max_absolute_label_error=0.0,
        max_surface_displacement_m=0.0,
        max_objective_increase=0.0,
    )
    judged = polish_flat675(problem, incoming, limits=limits)
    assert judged.acceptance_status == "rejected"
    assert "boozer_weighted_rms" in judged.rejection_reasons


def _policy_checked_solver(result: SimpleNamespace):
    """Constrain the production call's keyword-only Newton policy seam."""

    def solve(
        jax_boozer,
        *,
        iota,
        G,
        constraint_weight,
        weight_inv_modB,
        stab,
        tol,
        maxiter,
        linear_solver,
        max_dense_linearization_bytes,
        residual_fn,
        objective_fn,
    ) -> SimpleNamespace:
        assert jax_boozer is not None
        assert np.isfinite(iota) and np.isfinite(G)
        assert constraint_weight == NESTED_LS_CONSTRAINT_WEIGHT
        assert weight_inv_modB is NESTED_LS_WEIGHT_INV_MODB
        assert stab == NESTED_LS_JAX_INNER_STAB
        assert tol == NESTED_LS_NEWTON_TOL
        assert maxiter == NESTED_LS_BANANA_NEWTON_MAXITER
        assert linear_solver == "dense_lu"
        assert max_dense_linearization_bytes is None
        assert callable(residual_fn) and callable(objective_fn)
        return result

    return solve


@pytest.mark.parametrize(
    ("limited_field", "reason"),
    [
        ("max_boozer_weighted_rms", "boozer_weighted_rms"),
        ("max_absolute_label_error", "label_error"),
        ("max_surface_displacement_m", "surface_displacement"),
        ("max_objective_increase", "objective_increase"),
    ],
)
@pytest.mark.boozer
def test_each_design_limit_rejects_its_measured_quantity(
    real_polish, monkeypatch: pytest.MonkeyPatch, limited_field, reason
) -> None:
    problem, incoming, baseline = real_polish
    surface = baseline.outer_vector[problem.material.layout.surface_slice]
    monkeypatch.setattr(
        polish_module,
        "run_reduced_nested_ls_schur_newton",
        _policy_checked_solver(
            SimpleNamespace(
                surface_dofs=surface,
                iota=baseline.iota_after,
                G=baseline.G_after,
                reduced_gradient=np.zeros_like(surface),
                success=True,
                exit_status="converged",
                persisted=True,
                iteration_count=1,
            )
        ),
    )
    if limited_field == "max_objective_increase":
        original_terms = polish_module._weighted_terms
        calls = 0

        def larger_after_terms(*args, **kwargs):
            nonlocal calls
            calls += 1
            values = original_terms(*args, **kwargs)
            if calls == 2:
                values["residual"] += abs(baseline.objective_increase) + 1.0
            return values

        monkeypatch.setattr(polish_module, "_weighted_terms", larger_after_terms)
    caps = dict.fromkeys(
        (
            "max_boozer_weighted_rms",
            "max_absolute_label_error",
            "max_surface_displacement_m",
            "max_objective_increase",
        ),
        1e9,
    )
    caps[limited_field] = 0.0
    judged = polish_flat675(problem, incoming, limits=Flat675AcceptanceLimits(**caps))
    assert judged.acceptance_status == "rejected"
    assert reason in judged.rejection_reasons


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"success": False, "persisted": False}, "solver_failed"),
        ({"G": float("nan")}, "nonfinite"),
        ({"iota": 1.0}, "branch_changed"),
    ],
)
@pytest.mark.boozer
def test_failed_solver_outputs_cannot_be_accepted(
    real_polish, monkeypatch: pytest.MonkeyPatch, override, reason
) -> None:
    problem, incoming, baseline = real_polish
    surface = baseline.outer_vector[problem.material.layout.surface_slice]
    values = {
        "surface_dofs": surface,
        "iota": baseline.iota_after,
        "G": baseline.G_after,
        "reduced_gradient": np.zeros_like(surface),
        "success": True,
        "exit_status": "converged",
        "persisted": True,
        "iteration_count": 1,
    }
    values.update(override)
    monkeypatch.setattr(
        polish_module,
        "run_reduced_nested_ls_schur_newton",
        _policy_checked_solver(SimpleNamespace(**values)),
    )
    permissive = Flat675AcceptanceLimits(1e9, 1e9, 1e9, 1e9)
    judged = polish_flat675(problem, incoming, limits=permissive)
    assert judged.acceptance_status == "rejected"
    assert reason in judged.rejection_reasons
    if reason == "nonfinite":
        json.dumps(judged.as_dict(), allow_nan=False)
