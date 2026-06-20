"""Phase-1 incumbent-recovery graduation + repair-objective coverage.

Regression tests for the 2026-06-11 revert-wall fix: a phase-1 attempt whose
FINAL accepted state fails the regime rule used to roll everything back to
the seed anchor (erasing any hardware-clean incumbent found mid-attempt), so
hardware-imperfect rebuilds shipped bitwise donor coils. The fix graduates
from the best in-attempt incumbent (best_feasible for every regime;
best_hardware_near_miss additionally for repair_first) and completes the
repair objective with the keep-out / vessel / self-intersect / width /
poloidal axes it was previously blind to.

These tests fail against the pre-fix behavior: the recovery helper did not
exist, the repair objective ignored J_hardware_keepout, and the repair-state
encoding was a literal tuple duplicated across modules.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES_ROOT = (
    Path(__file__).resolve().parents[2] / "examples" / "single_stage_optimization"
)
sys.path.insert(0, str(EXAMPLES_ROOT))
from banana_opt.incumbents import SingleStageIncumbentState  # noqa: E402
from banana_opt.single_stage_phase1 import (  # noqa: E402
    REPAIR_STATE_BLOCKED,
    REPAIR_STATE_CLEAN,
    _SEED_REGIME_PRESERVE_FIRST,
    _SEED_REGIME_REPAIR_FIRST,
    _phase1_incumbent_recovery,
    _repair_phase1_total_grad,
    _step_norm_limited_phase1_objective,
    build_phase1_config,
    repair_state_is_clean,
    run_penalty_phase1,
)

del sys.path[0]

NDOF = 4


def make_incumbent(x, *, hardware_success):
    x = np.asarray(x, dtype=float)
    return SingleStageIncumbentState(
        x=x.copy(),
        surface_state={"frozen": True},
        objective_total=1.0,
        objective_grad=np.zeros_like(x),
        search_eval={"total": 1.0},
        surface_status={"success": True, "self_intersections": [False]},
        search_surface_status={"success": True},
        accepted_hardware_status={"success": bool(hardware_success)},
        topology_gate_status={"enabled": False, "success": True},
    )


def make_run_dict(anchor_x):
    anchor_x = np.asarray(anchor_x, dtype=float)
    return {
        "accepted_x": anchor_x.copy(),
        "x_prev": anchor_x.copy(),
        "surface_state": {"frozen": True},
        "J": 1.0,
        "dJ": np.zeros_like(anchor_x),
        "search_eval": {"total": 1.0},
        "surface_status": {"success": True, "self_intersections": [False]},
        "search_surface_status": {"success": True},
        "accepted_hardware_status": {"success": False},
        "topology_gate_status": {"enabled": False, "success": True},
        "accepted_iterations": 0,
        "invalid_state_rejects_total": 0,
        "surface_solve_rejects": 0,
        "hardware_rejects": 0,
        "topology_gate_rejects": 0,
        "best_feasible_incumbent": None,
        "best_feasible_metric": None,
        "best_feasible_stage": None,
        "best_hardware_near_miss_incumbent": None,
        "best_hardware_near_miss_metric": None,
        "best_hardware_near_miss_stage": None,
    }


class RepairStateEncodingTests:
    pass


def test_repair_state_clean_predicate_matches_constants():
    assert repair_state_is_clean(REPAIR_STATE_CLEAN)
    assert repair_state_is_clean((0, 0.0))
    assert not repair_state_is_clean(REPAIR_STATE_BLOCKED)
    assert not repair_state_is_clean((0, 0.25))


def test_repair_objective_includes_keepout_axis():
    grad_dim = 3
    eval_dict = {
        "total": 99.0,
        "grad": np.full(grad_dim, 7.0),
        "J_cc": 1.0,
        "dJ_cc": np.array([1.0, 0.0, 0.0]),
        "J_hardware_keepout": 5.0,
        "dJ_hardware_keepout": np.array([0.0, 1.0, 0.0]),
    }
    config = build_phase1_config(cc_weight=2.0, hardware_keepout_weight=3.0)
    total, grad = _repair_phase1_total_grad(eval_dict, phase1_config=config)
    assert total == pytest.approx(2.0 * 1.0 + 3.0 * 5.0)
    np.testing.assert_allclose(grad, np.array([2.0, 3.0, 0.0]))


def test_step_norm_limited_phase1_objective_rejects_before_expensive_eval():
    calls = []

    def objective(x):
        calls.append(np.asarray(x, dtype=float))
        return 1.0, np.ones(2)

    limited_objective = _step_norm_limited_phase1_objective(
        objective,
        np.array([0.0, 0.0]),
        step_norm_limit=0.1,
    )

    total, grad = limited_objective(np.array([0.2, 0.0]))

    assert not calls
    assert total > 1.0
    assert grad[0] > 0.0
    np.testing.assert_allclose(grad[1], 0.0)


def test_repair_objective_raises_loudly_instead_of_falling_back():
    """No silent degradation: a misconfigured repair phase (all weights zero)
    and a broken eval contract (nonzero weight, None/missing term) both raise
    instead of quietly optimizing something else."""
    eval_dict = {
        "total": 42.0,
        "grad": np.array([4.0, 4.0]),
        "J_cc": 1.0,
        "dJ_cc": np.array([1.0, 0.0]),
        "J_coil_width": None,
        "dJ_coil_width": None,
    }
    all_zero = build_phase1_config()
    with pytest.raises(ValueError, match="no live hardware term"):
        _repair_phase1_total_grad(eval_dict, phase1_config=all_zero)

    width_only = build_phase1_config(width_weight=10.0)
    with pytest.raises(ValueError, match="J_coil_width"):
        _repair_phase1_total_grad(eval_dict, phase1_config=width_only)

    missing_key = build_phase1_config(hardware_keepout_weight=1000.0)
    with pytest.raises(ValueError, match="J_hardware_keepout"):
        _repair_phase1_total_grad(eval_dict, phase1_config=missing_key)

    zero_weight_none_term_ok = build_phase1_config(cc_weight=2.0)
    total, grad = _repair_phase1_total_grad(
        eval_dict, phase1_config=zero_weight_none_term_ok
    )
    assert total == pytest.approx(2.0)
    np.testing.assert_allclose(grad, np.array([2.0, 0.0]))


def test_repair_objective_passes_rejection_evals_through():
    """Rejected-step evals ({"total", "grad"} only — the documented
    backtracking contract) pass through unchanged; they are not a broken
    contract and must not raise."""
    rejection_eval = {"total": 1.0e9, "grad": np.array([3.0, 3.0])}
    config = build_phase1_config(cc_weight=2.0, hardware_keepout_weight=1000.0)
    total, grad = _repair_phase1_total_grad(rejection_eval, phase1_config=config)
    assert total == pytest.approx(1.0e9)
    np.testing.assert_allclose(grad, np.array([3.0, 3.0]))


def test_recovery_prefers_feasible_incumbent_within_radius():
    anchor_x = np.zeros(NDOF)
    run_dict = make_run_dict(anchor_x)
    incumbent_x = np.full(NDOF, 0.01)
    run_dict["best_feasible_incumbent"] = make_incumbent(
        incumbent_x, hardware_success=True
    )
    run_dict["best_feasible_metric"] = 0.5
    result = _phase1_incumbent_recovery(
        run_dict,
        anchor_state={"best_feasible_metric": None},
        anchor_x=anchor_x,
        anchor_near_miss_metric=None,
        seed_regime=_SEED_REGIME_PRESERVE_FIRST,
        local_radius=0.05,
        phase1_config=build_phase1_config(),
    )
    assert result is not None
    incumbent, outcome, step_rms = result
    assert outcome == "feasible_incumbent_recovery"
    np.testing.assert_allclose(incumbent.x, incumbent_x)
    assert 0.0 < step_rms <= 0.05


def test_recovery_rejects_outside_radius_and_stale_metric():
    anchor_x = np.zeros(NDOF)
    run_dict = make_run_dict(anchor_x)
    run_dict["best_feasible_incumbent"] = make_incumbent(
        np.full(NDOF, 1.0), hardware_success=True
    )
    run_dict["best_feasible_metric"] = 0.5
    outside = _phase1_incumbent_recovery(
        run_dict,
        anchor_state={"best_feasible_metric": None},
        anchor_x=anchor_x,
        anchor_near_miss_metric=None,
        seed_regime=_SEED_REGIME_PRESERVE_FIRST,
        local_radius=0.05,
        phase1_config=build_phase1_config(),
    )
    assert outside is None

    run_dict["best_feasible_incumbent"] = make_incumbent(
        np.full(NDOF, 0.01), hardware_success=True
    )
    stale = _phase1_incumbent_recovery(
        run_dict,
        anchor_state={"best_feasible_metric": 0.4},
        anchor_x=anchor_x,
        anchor_near_miss_metric=None,
        seed_regime=_SEED_REGIME_PRESERVE_FIRST,
        local_radius=0.05,
        phase1_config=build_phase1_config(),
    )
    assert stale is None


def test_recovery_near_miss_is_repair_first_only():
    anchor_x = np.zeros(NDOF)
    run_dict = make_run_dict(anchor_x)
    run_dict["best_hardware_near_miss_incumbent"] = make_incumbent(
        np.full(NDOF, 0.01), hardware_success=False
    )
    run_dict["best_hardware_near_miss_metric"] = (1.5, 2.0)
    common = dict(
        anchor_state={"best_feasible_metric": None},
        anchor_x=anchor_x,
        anchor_near_miss_metric=(3.0, 2.0),
        local_radius=0.05,
        phase1_config=build_phase1_config(),
    )
    repair = _phase1_incumbent_recovery(
        run_dict, seed_regime=_SEED_REGIME_REPAIR_FIRST, **common
    )
    assert repair is not None
    assert repair[1] == "repair_incumbent_recovery"

    preserve = _phase1_incumbent_recovery(
        run_dict, seed_regime=_SEED_REGIME_PRESERVE_FIRST, **common
    )
    assert preserve is None

    worse = dict(common)
    worse["anchor_near_miss_metric"] = (1.0, 1.0)
    not_improved = _phase1_incumbent_recovery(
        run_dict, seed_regime=_SEED_REGIME_REPAIR_FIRST, **worse
    )
    assert not_improved is None


def _run_phase1_with_dirty_final_state(run_dict, *, seed_regime):
    """Drive run_penalty_phase1 with an injected minimizer whose accepted
    trajectory plants a hardware-clean best_feasible incumbent mid-attempt
    but leaves the FINAL accepted state hardware-dirty — the live revert-wall
    shape from the 06-11 campaign rebuilds."""
    anchor_x = np.asarray(run_dict["accepted_x"], dtype=float).copy()
    incumbent_x = anchor_x + 0.01

    def fake_callback(xk):
        run_dict["accepted_iterations"] += 1
        run_dict["accepted_x"] = np.asarray(xk, dtype=float).copy()

    def fake_minimize(fun, x0, jac, method, bounds, callback, options):
        clean_x = incumbent_x.copy()
        callback(clean_x)
        run_dict["best_feasible_incumbent"] = make_incumbent(
            clean_x, hardware_success=True
        )
        run_dict["best_feasible_metric"] = 0.5
        run_dict["best_feasible_stage"] = "test"
        dirty_x = anchor_x + 0.02
        callback(dirty_x)

        class Result:
            nit = 2
            success = True
            message = "fake"
            status = 0

        return Result()

    return run_penalty_phase1(
        anchor_x.copy(),
        total_maxiter=10,
        maxcor=10,
        ftol=1e-12,
        gtol=1e-12,
        initial_step_scale=1.0,
        initial_step_maxiter=0,
        enable_local_preservation=True,
        seed_regime=seed_regime,
        is_frontier_mode=False,
        lower_bounds=np.full(NDOF, -1.0),
        upper_bounds=np.full(NDOF, 1.0),
        run_dict=run_dict,
        objective_fn=lambda x: (0.0, np.zeros(NDOF)),
        callback_fn=fake_callback,
        refinement_eligible_fn=lambda rd: False,
        repair_progress_state_fn=lambda rd: REPAIR_STATE_BLOCKED,
        phase1_config=build_phase1_config(cc_weight=1.0),
        objective_eval_fn=None,
        normalize_message_fn=lambda message, **kwargs: str(message),
        restore_accepted_state_fn=lambda: None,
        minimize_fn=fake_minimize,
    )


def test_run_penalty_phase1_graduates_from_feasible_incumbent():
    anchor_x = np.zeros(NDOF)
    run_dict = make_run_dict(anchor_x)
    result = _run_phase1_with_dirty_final_state(
        run_dict, seed_regime=_SEED_REGIME_PRESERVE_FIRST
    )
    assert result["phase1_outcome"] == "feasible_incumbent_recovery", (
        "phase-1 must graduate from the in-attempt hardware-clean incumbent "
        f"instead of rolling back to the anchor (got {result['phase1_outcome']!r})"
    )
    assert result["continue_search"] is True
    np.testing.assert_allclose(
        np.asarray(run_dict["accepted_x"], dtype=float), anchor_x + 0.01
    )
    assert np.max(np.abs(result["next_dofs"] - anchor_x)) > 0.0
    assert result["local_preservation_radius"] is not None


def test_run_penalty_phase1_still_fails_closed_without_incumbent():
    anchor_x = np.zeros(NDOF)
    run_dict = make_run_dict(anchor_x)

    def fake_callback(xk):
        run_dict["accepted_iterations"] += 1
        run_dict["accepted_x"] = np.asarray(xk, dtype=float).copy()

    def fake_minimize(fun, x0, jac, method, bounds, callback, options):
        callback(anchor_x + 0.02)

        class Result:
            nit = 2
            success = True
            message = "fake"
            status = 0

        return Result()

    result = run_penalty_phase1(
        anchor_x.copy(),
        total_maxiter=10,
        maxcor=10,
        ftol=1e-12,
        gtol=1e-12,
        initial_step_scale=1.0,
        initial_step_maxiter=0,
        enable_local_preservation=True,
        seed_regime=_SEED_REGIME_PRESERVE_FIRST,
        is_frontier_mode=False,
        lower_bounds=np.full(NDOF, -1.0),
        upper_bounds=np.full(NDOF, 1.0),
        run_dict=run_dict,
        objective_fn=lambda x: (0.0, np.zeros(NDOF)),
        callback_fn=fake_callback,
        refinement_eligible_fn=lambda rd: False,
        repair_progress_state_fn=lambda rd: REPAIR_STATE_BLOCKED,
        phase1_config=build_phase1_config(cc_weight=1.0),
        objective_eval_fn=None,
        normalize_message_fn=lambda message, **kwargs: str(message),
        restore_accepted_state_fn=lambda: None,
        minimize_fn=fake_minimize,
    )
    assert result["phase1_outcome"] != "feasible_incumbent_recovery"
    assert bool(result["continue_search"]) is False or result[
        "phase1_outcome"
    ].startswith("preserved")
    np.testing.assert_allclose(
        np.asarray(run_dict["accepted_x"], dtype=float), anchor_x
    )
