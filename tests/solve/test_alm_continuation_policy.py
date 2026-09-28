"""Continuation policies decide each step of the ALM loop; the loop executes.

The loop measures the iterate, runs L-BFGS-B, keeps the books and executes
decisions. An ``ALMContinuationPolicy`` makes the decisions that differ between
users (``inner_plan``, ``retry_stalled_trial``, ``before_inner``,
``after_inner``); ``DefaultContinuationPolicy`` (``simsopt.solve.alm.policy``)
is the library default.

Every library call the golden scenarios make is intercepted and given the
policy under test. The pure-decision tests cover the default branches the
goldens do not reach.
"""

import ast
import dataclasses
import inspect
import sys
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Union
from unittest.mock import patch

import numpy as np

import simsopt.solve.alm as alm
from simsopt.solve.alm import control as alm_control
from simsopt.solve.alm import inner as alm_inner
from simsopt.solve.alm import core as alm_core
from simsopt.solve.alm import policy as alm_policy
from simsopt.solve.alm.continuation import (
    ALMContinue,
    ALMConverge,
    ALMDualUpdateStep,
    ALMHold,
    ALMInnerPlanView,
    ALMInnerSolveOutcome,
    ALMIterateMeasurement,
    ALMPostInnerView,
    ALMPreInnerView,
    ALMRaisePenalty,
    ALMStalledTrialView,
    ALMStop,
)
from simsopt.solve.alm.policy import DefaultContinuationPolicy

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "test_files" / "alm_golden"
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))
import alm_golden_scenarios as golden  # noqa: E402

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "src" / "simsopt" / "solve" / "alm"

@contextmanager
def library_policy(policy):
    """Run every library ``minimize_alm`` call in this block with ``policy``;
    yields the ``(trust_radius_init, policy the caller passed)`` of each call."""
    calls: list = []
    library_minimize_alm = alm_control.minimize_alm

    def with_policy(*args, **kwargs):
        settings = args[3] if len(args) > 3 else kwargs["settings"]
        calls.append(
            (settings.trust_radius_init, kwargs.get("continuation_policy"))
        )
        return library_minimize_alm(
            *args, **{**kwargs, "continuation_policy": policy}
        )

    with patch.object(alm_control, "minimize_alm", with_policy):
        yield calls


class AlmPolicyGoldenReplayTests(unittest.TestCase):
    def test_default_policy_passed_explicitly_replays_every_golden(self):
        for scenario in golden.SCENARIOS:
            with self.subTest(scenario=scenario.name):
                with library_policy(DefaultContinuationPolicy()) as calls:
                    trajectory = scenario.run()
                self.assertGreater(len(calls), 0)
                self.assertEqual({passed for _radius, passed in calls}, {None})
                self.assertIsNone(golden.golden_difference(scenario.name, trajectory))


class AlmPolicyWiringTests(unittest.TestCase):
    def test_library_default_is_the_default_policy(self):
        parameter = inspect.signature(alm_control.minimize_alm).parameters[
            "continuation_policy"
        ]
        self.assertIs(parameter.default, alm_policy.DEFAULT_CONTINUATION_POLICY)
        self.assertIsInstance(parameter.default, DefaultContinuationPolicy)

    def test_the_default_policy_is_a_stateless_frozen_dataclass(self):
        self.assertTrue(dataclasses.is_dataclass(DefaultContinuationPolicy))
        self.assertTrue(DefaultContinuationPolicy.__dataclass_params__.frozen)
        for method in (
            "inner_plan",
            "retry_stalled_trial",
            "before_inner",
            "after_inner",
        ):
            self.assertTrue(callable(getattr(DefaultContinuationPolicy, method)))
        self.assertEqual(dataclasses.fields(DefaultContinuationPolicy), ())

    def test_decisions_and_views_are_frozen(self):
        for carrier in (
            ALMConverge,
            ALMStop,
            ALMDualUpdateStep,
            ALMRaisePenalty,
            ALMHold,
            ALMContinue,
            ALMPreInnerView,
            ALMPostInnerView,
            ALMInnerPlanView,
            ALMStalledTrialView,
        ):
            with self.subTest(carrier=carrier.__name__):
                self.assertTrue(carrier.__dataclass_params__.frozen)

    def test_loop_module_defines_no_policy_rule(self):
        # The loop executes decisions; the rules live in the policies.
        policy_symbols = (
            "ALMInnerSolveProfile",
            "_BOXED_INNER_PROFILES",
            "_select_inner_solve_profile",
            "_build_inner_options",
            "_PLATEAU_STALL_LIMIT",
            "_strict_feasibility_satisfied",
            "_grow_continuation_trust_radius",
            "_termination_reason_from_last_step",
            "_emit_alm_subproblem_continue",
            "_emit_alm_penalty_increase_arm",
            "_emit_alm_converged_step",
            "_apply_continuation_penalty_increase",
        )
        tree = ast.parse((PACKAGE_ROOT / "control.py").read_text(encoding="utf-8"))
        defined = {
            target.id
            for node in tree.body
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        } | {
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        }
        for symbol in policy_symbols:
            with self.subTest(symbol=symbol):
                self.assertNotIn(symbol, defined)


# --------------------------------------------------------------------------
# Pure decisions: views measured the way the loop measures them
# --------------------------------------------------------------------------

SETTINGS = alm.ALMSettings(
    max_outer_iterations=4,
    max_subproblem_continuations=2,
    penalty_init=10.0,
    feasibility_tol=1.0e-6,
    stationarity_tol=1.0e-6,
    trust_radius_min=1.0e-3,
    trust_radius_grow=1.5,
)


def _measure(
    evaluation,
    *,
    settings=SETTINGS,
    multipliers=(0.0,),
    penalty=10.0,
    update_feasibility_tol=1.0e-2,
    update_stationarity_tol=1.0e-2,
) -> ALMIterateMeasurement:
    multipliers = np.asarray(multipliers, dtype=float)
    gate = max(
        settings.feasibility_tol,
        min(update_feasibility_tol, settings.relaxed_feasibility_gate_cap),
    )
    solver, feasibility, _dual, max_violation = alm_core._extract_constraint_state(
        evaluation
    )
    routing = alm_core._constraint_routing_state(
        evaluation, multipliers, penalty, gate
    )
    stationarity, kkt, mismatch = alm_core._stationarity_metrics(
        evaluation, routing, gate
    )
    return ALMIterateMeasurement(
        evaluation=evaluation,
        multipliers=multipliers,
        penalty=float(penalty),
        solver_constraint_values=solver,
        feasibility_values=feasibility,
        max_feasibility_violation=max_violation,
        routing_state=routing,
        stationarity_norm=stationarity,
        kkt_stationarity_norm=kkt,
        signal_mismatch_active=mismatch,
        update_feasibility_tol=update_feasibility_tol,
        update_stationarity_tol=update_stationarity_tol,
        effective_feasibility_tol=gate,
    )


def _inner(
    measured,
    *,
    meaningful_progress=True,
    infeasible_stall=False,
    infeasible_stall_reason=None,
    feasibility_delta=0.0,
    sufficient_decrease_measure=1.0,
    sufficient_decrease_reference=1.0,
) -> ALMInnerSolveOutcome:
    return ALMInnerSolveOutcome(
        x=np.zeros(2),
        measured=measured,
        iterations=1,
        attempts=1,
        optimizer_success=True,
        optimizer_message="CONVERGENCE",
        bounds=None,
        inner_options=MappingProxyType({}),
        inner_profile="unbounded",
        infeasible_stall=infeasible_stall,
        infeasible_stall_reason=infeasible_stall_reason,
        inner_false_success=False,
        nonfinite_candidate_evaluation=False,
        nonfinite_candidate_fields=None,
        meaningful_progress=meaningful_progress,
        feasibility_delta=feasibility_delta,
        feasibility_delta_tolerance=1.0e-12,
        sufficient_decrease_measure=sufficient_decrease_measure,
        sufficient_decrease_reference=sufficient_decrease_reference,
    )


def _view(
    measured,
    *,
    after_inner=True,
    settings=SETTINGS,
    continuation_iteration=0,
    last_cap_binding_active=False,
    feasible_stall_count=0,
    trust_radius=None,
    **inner_fields,
) -> Union[ALMPreInnerView, ALMPostInnerView]:
    shared = dict(
        settings=settings,
        outer_iteration=1,
        continuation_iteration=continuation_iteration,
        start=measured,
        last_cap_binding_active=last_cap_binding_active,
        feasible_stall_count=feasible_stall_count,
        trust_radius=trust_radius,
    )
    if not after_inner:
        return ALMPreInnerView(**shared)
    return ALMPostInnerView(inner=_inner(measured, **inner_fields), **shared)


def _plain(target, x0, **overrides):
    """min (x0 - target)^2 s.t. x0 - 1 <= 0 at x = (x0, 0)."""
    evaluation = alm.augmented_inequality_objective(
        (x0 - target) ** 2,
        np.array([2.0 * (x0 - target), 0.0]),
        np.array([x0 - 1.0]),
        [np.array([1.0, 0.0])],
        np.zeros(1),
        10.0,
    )
    evaluation.update(overrides)
    return evaluation


def _hybrid(x, *, surrogate_bias, activity_tolerance=None, **overrides):
    evaluation = golden.hybrid_band_evaluate(
        surrogate_bias=surrogate_bias, activity_tolerance=activity_tolerance
    )(np.asarray(x, dtype=float), np.zeros(1), 10.0)
    evaluation.update(overrides)
    return evaluation


# A KKT point: x0 = 0 minimizes x0^2 with the constraint inactive.
CONVERGED = _plain(0.0, 0.0)
# Hard constraints inactive under hybrid signals, stationary by construction.
CONSTRAINTS_INACTIVE = _hybrid((0.0, 1.0), surrogate_bias=0.05, stationarity_norm=0.0)
# Hard-feasible, surrogate-active (live surrogate shift): signal mismatch.
MISMATCH_LIVE_SHIFT = _hybrid((0.99, 1.0), surrogate_bias=0.05)
# Hard-active, surrogate-inactive (zero surrogate shift): signal mismatch.
MISMATCH_ZERO_SHIFT = _hybrid(
    (0.99, 1.0), surrogate_bias=-0.05, activity_tolerance=0.02
)
# Feasible, far from stationary.
FEASIBLE_NOT_STATIONARY = _plain(3.0, 0.5)
# Violated by 1.
INFEASIBLE = _plain(3.0, 2.0)
STALL_REASON = "failed_inner_solve_without_feasibility_gain"


class AlmDefaultBeforeInnerTests(unittest.TestCase):
    def test_converged_start_returns_converge(self):
        view = _view(_measure(CONVERGED), after_inner=False, feasible_stall_count=1)
        self.assertEqual(
            DefaultContinuationPolicy().before_inner(view),
            ALMConverge(
                action="converged",
                termination_reason="converged",
                message="ALM converged: max_violation=0.000e+00, stationarity=0.000e+00",
                feasible_stall_count=1,
            ),
        )

    def test_cap_binding_blocks_the_start_shortcut(self):
        # Golden gap: the cap-only block before the inner solve.
        view = _view(
            _measure(CONVERGED), after_inner=False, last_cap_binding_active=True
        )
        self.assertIsNone(DefaultContinuationPolicy().before_inner(view))

    def test_constraints_inactive_start_runs_the_inner_solve(self):
        measured = _measure(CONSTRAINTS_INACTIVE)
        self.assertFalse(measured.signal_mismatch_active)
        view = _view(measured, after_inner=False)
        self.assertIsNone(DefaultContinuationPolicy().before_inner(view))


class AlmDefaultAfterInnerTests(unittest.TestCase):
    def after_inner(self, view):
        return DefaultContinuationPolicy().after_inner(view)

    def test_converged_iterate_returns_converge(self):
        self.assertEqual(
            self.after_inner(_view(_measure(CONVERGED))),
            ALMConverge(
                action="converged",
                termination_reason="converged",
                message="ALM converged: max_violation=0.000e+00, stationarity=0.000e+00",
                feasible_stall_count=0,
            ),
        )

    def test_cap_binding_turns_convergence_into_a_dual_update(self):
        # Golden gap: the cap-only block after the inner solve.
        decision = self.after_inner(
            _view(_measure(CONVERGED), last_cap_binding_active=True)
        )
        self.assertIsInstance(decision, ALMDualUpdateStep)
        self.assertIsNone(decision.penalty_reason)
        self.assertEqual(decision.feasible_stall_count, 0)

    def test_constraints_inactive_iterate_converges_with_the_hard_violation(self):
        measured = _measure(CONSTRAINTS_INACTIVE)
        self.assertEqual(
            self.after_inner(_view(measured)),
            ALMConverge(
                action="constraints_inactive_converged",
                termination_reason="constraints_inactive_converged",
                message=(
                    "ALM converged with inactive hard constraints: "
                    f"max_violation={measured.routing_state.hard_max_violation:.3e}, "
                    "stationarity=0.000e+00"
                ),
                feasible_stall_count=0,
            ),
        )

    def test_cap_binding_turns_constraints_inactive_convergence_into_a_stall(self):
        # Golden gap: the cap-only block on the constraints-inactive arm.
        decision = self.after_inner(
            _view(
                _measure(CONSTRAINTS_INACTIVE),
                last_cap_binding_active=True,
                continuation_iteration=1,
                meaningful_progress=False,
                feasible_stall_count=1,
            )
        )
        self.assertEqual(
            decision,
            ALMStop(
                action="constraints_inactive_stall",
                termination_reason="constraints_inactive_stall",
                message_prefix=(
                    "ALM stopped after hard constraints became inactive without "
                    "further stationarity progress"
                ),
                feasible_stall_count=1,
            ),
        )

    def test_infeasible_stall_with_the_dual_gate_met_raises_with_the_update(self):
        decision = self.after_inner(
            _view(
                _measure(_plain(0.0, 0.0, stationarity_norm=1.0e-3)),
                infeasible_stall=True,
                infeasible_stall_reason=STALL_REASON,
            )
        )
        self.assertEqual(
            decision,
            ALMDualUpdateStep(penalty_reason=STALL_REASON, feasible_stall_count=0),
        )

    def test_unimproved_hard_violation_raises_with_the_update(self):
        # The dual gate (update tolerance 0.1) holds, but the hard violation
        # exceeds the clamped gate (1e-2) and did not shrink.
        measured = _measure(
            _plain(0.0, 1.05, stationarity_norm=1.0e-3),
            update_feasibility_tol=0.1,
            update_stationarity_tol=10.0,
        )
        decision = self.after_inner(_view(measured, feasibility_delta=0.0))
        self.assertEqual(
            decision,
            ALMDualUpdateStep(
                penalty_reason="hard_feasibility_not_improved_after_dual_update",
                feasible_stall_count=0,
            ),
        )

    def test_infeasible_stall_raises_the_penalty(self):
        # Golden gap: the cap return of this arm is the loop's.
        decision = self.after_inner(
            _view(
                _measure(INFEASIBLE),
                infeasible_stall=True,
                infeasible_stall_reason=STALL_REASON,
            )
        )
        self.assertIsInstance(decision, ALMRaisePenalty)
        self.assertEqual(decision.action, "infeasible_stall_penalty_increase")
        self.assertIsNone(decision.subproblem_limit_reason)
        self.assertFalse(decision.signal_mismatch_repair)

    def test_stalled_signal_mismatch_with_a_live_shift_raises_the_penalty(self):
        # Golden gap: the cap return of this arm is the loop's. The
        # raise keeps the stall count it came with.
        measured = _measure(MISMATCH_LIVE_SHIFT, update_stationarity_tol=1.0e-6)
        self.assertTrue(measured.signal_mismatch_active)
        self.assertFalse(measured.routing_state.surrogate_positive_shift_zero)
        decision = self.after_inner(
            _view(measured, meaningful_progress=False, feasible_stall_count=1)
        )
        self.assertIsInstance(decision, ALMRaisePenalty)
        self.assertEqual(decision.action, "signal_mismatch_penalty_increase")
        self.assertEqual(decision.feasible_stall_count, 1)

    def test_repair_flag_with_a_zero_shift_stops_on_the_mismatch(self):
        # Golden gap: continue_on_signal_mismatch=True cannot repair a
        # mismatch whose surrogate shift is zero.
        settings = dataclasses.replace(SETTINGS, continue_on_signal_mismatch=True)
        measured = _measure(
            MISMATCH_ZERO_SHIFT, settings=settings, update_stationarity_tol=1.0e-6
        )
        self.assertTrue(measured.signal_mismatch_active)
        self.assertTrue(measured.routing_state.surrogate_positive_shift_zero)
        decision = self.after_inner(
            _view(
                measured,
                settings=settings,
                meaningful_progress=False,
                feasible_stall_count=1,
            )
        )
        self.assertEqual(
            decision,
            ALMStop(
                action="signal_mismatch_stall",
                termination_reason="signal_mismatch_stall",
                message_prefix=(
                    "ALM stopped after hard-feasible and surrogate-active "
                    "signals repeated without corrective progress"
                ),
                feasible_stall_count=1,
            ),
        )

    def test_repaired_mismatch_continues_then_raises_at_the_limit(self):
        settings = dataclasses.replace(SETTINGS, continue_on_signal_mismatch=True)
        measured = _measure(
            MISMATCH_LIVE_SHIFT, settings=settings, update_stationarity_tol=1.0e-6
        )
        continued = self.after_inner(
            _view(
                measured,
                settings=settings,
                continuation_iteration=1,
                feasible_stall_count=1,
                trust_radius=0.3,
            )
        )
        self.assertEqual(
            continued,
            ALMContinue(
                trust_radius=0.3,
                update_stationarity_tol=1.0e-6,
                feasible_stall_count=0,
                signal_mismatch_repair=True,
            ),
        )
        at_limit = self.after_inner(
            _view(
                measured,
                settings=settings,
                continuation_iteration=settings.max_subproblem_continuations,
                feasible_stall_count=1,
            )
        )
        self.assertIsInstance(at_limit, ALMRaisePenalty)
        self.assertEqual(
            at_limit.action, "signal_mismatch_subproblem_limit_penalty_increase"
        )
        self.assertEqual(at_limit.subproblem_limit_reason, "max_subproblem_continuations")
        self.assertTrue(at_limit.signal_mismatch_repair)
        self.assertEqual(at_limit.feasible_stall_count, 0)

    def test_second_feasible_stall_stops_on_the_plateau(self):
        decision = self.after_inner(
            _view(
                _measure(FEASIBLE_NOT_STATIONARY),
                meaningful_progress=False,
                feasible_stall_count=1,
            )
        )
        self.assertEqual(
            decision,
            ALMStop(
                action="subproblem_limit",
                termination_reason="plateau_stall",
                message_prefix=(
                    "ALM stopped after repeated feasible stationarity "
                    "plateau without meaningful progress"
                ),
                feasible_stall_count=2,
                subproblem_limit_reason="plateau_stall",
                marks_max_outer=True,
            ),
        )

    def test_feasible_stall_continues_with_a_tighter_stationarity_tolerance(self):
        # Stationarity 5 misses the update tolerance 4 but is within twice
        # it, so the tolerance tightens to half the stationarity.
        measured = _measure(FEASIBLE_NOT_STATIONARY, update_stationarity_tol=4.0)
        self.assertEqual(measured.stationarity_norm, 5.0)
        decision = self.after_inner(
            _view(measured, meaningful_progress=False, trust_radius=0.3)
        )
        self.assertEqual(
            decision,
            ALMContinue(
                trust_radius=0.3,
                update_stationarity_tol=min(
                    4.0,
                    max(SETTINGS.stationarity_tol, 0.5 * measured.stationarity_norm),
                ),
                feasible_stall_count=1,
            ),
        )

    def test_feasible_step_at_the_continuation_limit_raises_the_penalty(self):
        decision = self.after_inner(
            _view(
                _measure(FEASIBLE_NOT_STATIONARY),
                continuation_iteration=SETTINGS.max_subproblem_continuations,
                feasible_stall_count=1,
            )
        )
        self.assertIsInstance(decision, ALMRaisePenalty)
        self.assertEqual(decision.action, "subproblem_limit_penalty_increase")
        self.assertEqual(decision.subproblem_limit_reason, "max_subproblem_continuations")
        self.assertEqual(decision.feasible_stall_count, 0)

    def test_sufficient_decrease_holds_the_penalty(self):
        self.assertEqual(
            self.after_inner(
                _view(
                    _measure(INFEASIBLE),
                    sufficient_decrease_measure=0.4,
                    sufficient_decrease_reference=1.0,
                    feasible_stall_count=1,
                )
            ),
            ALMHold(feasible_stall_count=1),
        )

    def test_insufficient_decrease_raises_the_penalty(self):
        # Golden gap: the cap return of this arm is the loop's.
        decision = self.after_inner(_view(_measure(INFEASIBLE)))
        self.assertIsInstance(decision, ALMRaisePenalty)
        self.assertEqual(decision.action, "penalty_increase")


def _slack_row(x0, multiplier, penalty=1.0):
    """min -0.9 x0 s.t. x0 - 0.1 <= 0 (KKT point x0 = 0.1, lambda = 0.9),
    evaluated at ``x0`` with ``multiplier`` and ``penalty``."""
    evaluation = alm.augmented_inequality_objective(
        -0.9 * x0,
        np.array([-0.9]),
        np.array([x0 - 0.1]),
        [np.array([1.0])],
        np.array([multiplier]),
        penalty,
    )
    return _measure(evaluation, multipliers=(multiplier,), penalty=penalty)


# At x0 = 0 with lambda = rho = 1 the row is inactive (g = -0.1), yet its
# shifted multiplier max(0, lambda + rho g) = 0.9 cancels grad f = -0.9: the
# augmented gradient is 0 and no row is active, so the KKT diagnostic is None.
# Complementarity fails by min(0.9, 0.1) = 0.1.
SLACK_ROW_CANCELS_GRADIENT = _slack_row(0.0, 1.0)


class AlmDefaultComplementarityTests(unittest.TestCase):
    """Default convergence certifies the KKT conditions at the shifted
    multipliers the augmented gradient carries, complementarity included."""

    def test_the_cancelled_gradient_has_no_active_row_to_certify(self):
        measured = SLACK_ROW_CANCELS_GRADIENT
        self.assertEqual(measured.stationarity_norm, 0.0)
        self.assertEqual(measured.max_feasibility_violation, 0.0)
        self.assertIsNone(measured.kkt_stationarity_norm)

    def test_a_multiplier_on_an_inactive_row_blocks_the_start_shortcut(self):
        view = _view(SLACK_ROW_CANCELS_GRADIENT, after_inner=False)
        self.assertIsNone(DefaultContinuationPolicy().before_inner(view))

    def test_a_multiplier_on_an_inactive_row_is_updated_instead_of_converging(self):
        self.assertEqual(
            DefaultContinuationPolicy().after_inner(_view(SLACK_ROW_CANCELS_GRADIENT)),
            ALMDualUpdateStep(penalty_reason=None, feasible_stall_count=0),
        )

    def test_the_multiplier_of_an_active_row_converges(self):
        measured = _slack_row(0.1, 0.9)
        self.assertEqual(measured.stationarity_norm, 0.0)
        for view in (_view(measured, after_inner=False), _view(measured)):
            with self.subTest(view=type(view).__name__):
                decision = (
                    DefaultContinuationPolicy().before_inner(view)
                    if isinstance(view, ALMPreInnerView)
                    else DefaultContinuationPolicy().after_inner(view)
                )
                self.assertIsInstance(decision, ALMConverge)

    def test_a_run_started_at_the_cancelled_gradient_ends_at_the_kkt_point(self):
        def evaluate(x, multipliers, penalty):
            return alm.augmented_inequality_objective(
                float(-0.9 * x[0]), np.array([-0.9]), np.array([x[0] - 0.1]),
                [np.array([1.0])], multipliers, penalty,
            )

        result = alm.minimize_alm(
            np.array([0.0]), ["x0_le_0p1"], evaluate,
            alm.ALMSettings(max_outer_iterations=5), {"maxiter": 10},
            initial_multipliers=np.array([1.0]), initial_penalty=1.0,
        )
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, [0.1], atol=1.0e-6)
        np.testing.assert_allclose(result.multipliers, [0.9], atol=1.0e-6)


def _row_scaled(*, row_scale=1.0, band=0.0, hybrid=False):
    """min -x s.t. M (x - 0.1) <= 0 (M = ``row_scale``; KKT point x = 0.1,
    lambda = 1/M) with activity band ``band``; ``hybrid`` returns the quartet
    with every channel equal to the signed row."""

    def evaluate(x, multipliers, penalty):
        signed = np.array([row_scale * (x[0] - 0.1)])
        evaluation = alm.augmented_inequality_objective(
            -float(x[0]), np.array([-1.0]), signed, [np.array([row_scale])],
            multipliers, penalty,
        )
        evaluation["constraint_activity_tolerances"] = np.array([band])
        if hybrid:
            evaluation.update(
                hard_signed_constraint_values=signed.copy(),
                hard_violation_values=np.maximum(signed, 0.0),
                surrogate_signed_constraint_values=signed.copy(),
                hard_dual_update_values=signed.copy(),
            )
        return evaluation

    return evaluate


def _solve_from(evaluate, x0, multiplier, penalty, *, rows=("c",), settings=None, **kwargs):
    return alm.minimize_alm(
        np.array([x0]), list(rows), evaluate,
        settings or alm.ALMSettings(max_outer_iterations=2), {"maxiter": 10},
        initial_multipliers=(
            None if multiplier is None
            else np.atleast_1d(np.asarray(multiplier, dtype=float))
        ),
        initial_penalty=penalty, **kwargs,
    )


# Row rescaling g -> M g, lambda -> lambda / M, rho -> rho / M^2 leaves the
# augmented Lagrangian, its gradient and every iterate unchanged; the caps are
# lifted so that each M in the sweep states the same problem.
UNCAPPED = alm.ALMSettings(max_outer_iterations=4, penalty_max=None, multiplier_max=None)


class AlmInvariantComplementarityTests(unittest.TestCase):
    """The complementarity certificate uses the actual slack, never an
    activity band, and is invariant under an equivalent row rescaling."""

    def assert_reaches_the_kkt_point(self, result):
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.termination_reason, "converged")
        np.testing.assert_allclose(result.x, [0.1], rtol=0.0, atol=1.0e-9)

    def test_an_activity_band_does_not_certify_an_interior_point(self):
        # x = 0.08, g = -0.02 inside a 0.02 band; lambda+ = 1 cancels grad f.
        self.assert_reaches_the_kkt_point(
            _solve_from(_row_scaled(band=0.02), 0.08, 1.02, 1.0)
        )

    def test_a_rescaled_row_does_not_certify_an_interior_point(self):
        # lambda+ = 5e-7 is below feasibility_tol only in the rescaled units.
        scale = 2.0e6
        self.assert_reaches_the_kkt_point(
            _solve_from(_row_scaled(row_scale=scale), 0.0, 1.1 / scale, 1.0 / scale**2)
        )

    def test_success_and_x_do_not_depend_on_row_scale_or_band(self):
        outcomes = {}
        for scale in (1.0e-6, 1.0e-3, 1.0, 1.0e3, 1.0e6):
            for band in (0.0, 0.01, 0.02, 0.05):
                for x0, multiplier in ((0.0, 1.1), (0.08, 1.02)):
                    result = _solve_from(
                        _row_scaled(row_scale=scale, band=band), x0,
                        multiplier / scale, 1.0 / scale**2, settings=UNCAPPED,
                    )
                    outcomes[scale, band, x0] = (result.success, result.x[0])
        self.assertEqual({success for success, _x in outcomes.values()}, {True}, outcomes)
        spread = np.ptp([x for _success, x in outcomes.values()])
        self.assertLessEqual(spread, 1.0e-9, outcomes)
        self.assertAlmostEqual(next(iter(outcomes.values()))[1], 0.1, delta=1.0e-9)

    def test_an_exactly_active_row_with_a_positive_multiplier_converges(self):
        result = _solve_from(_row_scaled(), 0.1, 1.0, 1.0)
        self.assert_reaches_the_kkt_point(result)
        self.assertEqual(result.outer_iterations, 1)

    def test_a_weakly_active_row_with_a_zero_multiplier_converges(self):
        def evaluate(x, multipliers, penalty):
            return alm.augmented_inequality_objective(
                float((x[0] - 0.1) ** 2), np.array([2.0 * (x[0] - 0.1)]),
                np.array([x[0] - 0.1]), [np.array([1.0])], multipliers, penalty,
            )

        self.assert_reaches_the_kkt_point(_solve_from(evaluate, 0.1, 0.0, 1.0))

    def test_an_equality_like_pair_converges(self):
        def evaluate(x, multipliers, penalty):
            return alm.augmented_inequality_objective(
                -float(x[0]), np.array([-1.0]),
                np.array([x[0] - 0.1, 0.1 - x[0]]),
                [np.array([1.0]), np.array([-1.0])], multipliers, penalty,
            )

        result = _solve_from(
            evaluate, 0.0, (0.0, 0.0), 1.0, rows=("upper", "lower"),
            settings=alm.ALMSettings(max_outer_iterations=40),
        )
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, [0.1], atol=1.0e-6)

    def test_rows_with_zero_gradients_do_not_block_convergence(self):
        # A constant slack row and an identically zero row next to the active one.
        def evaluate(x, multipliers, penalty):
            return alm.augmented_inequality_objective(
                -float(x[0]), np.array([-1.0]),
                np.array([x[0] - 0.1, -1.0, 0.0]),
                [np.array([1.0]), np.zeros(1), np.zeros(1)], multipliers, penalty,
            )

        result = _solve_from(
            evaluate, 0.1, (1.0, 0.5, 0.5), 1.0, rows=("active", "slack", "zero"),
        )
        self.assert_reaches_the_kkt_point(result)

    def test_base_bounds_do_not_block_convergence(self):
        result = _solve_from(
            _row_scaled(), 0.0, 1.1, 1.0, base_bounds=[(-1.0, 1.0)],
        )
        self.assert_reaches_the_kkt_point(result)


class AlmComplementarityGapTests(unittest.TestCase):
    """Convergence bounds the complementarity gap sum_i lambda+_i max(0, -g_i)
    term by term: no row classification, no cancellation between rows. Each
    run below starts where the augmented gradient is exactly 0 at a point that
    is not the optimum; it must reach the optimum or end without success."""

    def assert_no_false_convergence(self, result, optimum, atol):
        if result.success:
            np.testing.assert_allclose(result.x, optimum, rtol=0.0, atol=atol)

    def test_a_steep_row_that_never_reaches_its_boundary_does_not_certify(self):
        # g = -1 + 0.1 sin(w x) lies in [-1.1, -0.9]: never active, optimum
        # x = 1. At x = 0, lambda+ = 2 / (0.1 w) cancels grad f = -2.
        steepness, penalty = 1.5e7, 1.0e-8

        def evaluate(x, multipliers, penalty_value):
            phase = steepness * x[0]
            return alm.augmented_inequality_objective(
                float((x[0] - 1.0) ** 2), np.array([2.0 * (x[0] - 1.0)]),
                np.array([-1.0 + 0.1 * np.sin(phase)]),
                [np.array([0.1 * steepness * np.cos(phase)])],
                multipliers, penalty_value,
            )

        result = _solve_from(evaluate, 0.0, 2.0 / (0.1 * steepness) + penalty, penalty)
        self.assert_no_false_convergence(result, [1.0], atol=1.0e-6)

    def test_opposing_multipliers_do_not_cancel_into_a_certificate(self):
        # f = x^2, g1 = x - 1, g2 = -x - 1: both rows slack by 1 with
        # lambda+ = 1, whose gradient terms +1 and -1 cancel. The dual update
        # takes both multipliers to 0, which certifies x = 0.
        def evaluate(x, multipliers, penalty):
            return alm.augmented_inequality_objective(
                float(x[0] ** 2), np.array([2.0 * x[0]]),
                np.array([x[0] - 1.0, -x[0] - 1.0]),
                [np.array([1.0]), np.array([-1.0])], multipliers, penalty,
            )

        result = _solve_from(evaluate, 0.0, (2.0, 2.0), 1.0, rows=("upper", "lower"))
        self.assertTrue(result.success, result.message)
        np.testing.assert_array_equal(result.x, [0.0])
        self.assertEqual(result.outer_iterations, 2)
        np.testing.assert_array_equal(result.evaluation["positive_shift_values"], [0.0, 0.0])

    def test_a_large_multiplier_just_inside_its_boundary_does_not_certify(self):
        # f = -5e5 x, g = x - 5e-7: lambda+ = 5e5 at slack 5e-7 is a gap of
        # 0.25 in f, which moving to the boundary would recover.
        def evaluate(x, multipliers, penalty):
            return alm.augmented_inequality_objective(
                -5.0e5 * float(x[0]), np.array([-5.0e5]), np.array([x[0] - 5.0e-7]),
                [np.array([1.0])], multipliers, penalty,
            )

        result = _solve_from(evaluate, 0.0, 5.0e5 + 5.0e-7, 1.0)
        self.assert_no_false_convergence(result, [5.0e-7], atol=1.0e-9)

    def test_the_gap_is_the_lagrangian_gap_on_the_feasible_side(self):
        evaluation = alm.augmented_inequality_objective(
            0.3, np.zeros(1), np.array([-0.5, 0.0, -2.0]),
            [np.ones(1), np.ones(1), np.ones(1)], np.array([2.0, 1.0, 1.0]), 1.0,
        )
        routing = alm_core._constraint_routing_state(
            evaluation, np.array([2.0, 1.0, 1.0]), 1.0, 1.0e-6
        )
        # lambda+ = (1.5, 1, 0), slack = (0.5, 0, 2): f - l(x, lambda+) = 0.75.
        self.assertEqual(alm_core._complementarity_gap(routing), 0.75)


class AlmObjectiveOffsetTests(unittest.TestCase):
    """Adding a constant to f changes neither the minimizer nor any gradient,
    so it must not change a termination decision: the complementarity-gap
    tolerance is absolute, in f's units."""

    OFFSETS = (0.0, 1.0e3, -1.0e3, 1.0e7, -1.0e7)

    def decisions(self, make_evaluate, x0, multiplier):
        return {
            offset: (lambda r: (r.success, r.termination_reason, float(r.x[0])))(
                _solve_from(make_evaluate(offset), x0, multiplier, 1.0)
            )
            for offset in self.OFFSETS
        }

    def test_an_affine_objective_offset_does_not_change_the_decision(self):
        # f = C - x, g = x - 1: at x = 0, lambda+ = 1 cancels grad f with a gap of 1.
        def make_evaluate(offset):
            def evaluate(x, multipliers, penalty):
                return alm.augmented_inequality_objective(
                    offset - float(x[0]), np.array([-1.0]), np.array([x[0] - 1.0]),
                    [np.array([1.0])], multipliers, penalty,
                )
            return evaluate

        decisions = self.decisions(make_evaluate, 0.0, 2.0)
        self.assertEqual(set(decisions.values()), {(True, "converged", 1.0)}, decisions)

    def test_a_quadratic_objective_offset_does_not_change_the_decision(self):
        # f = (x - 1)^2 + C, g = x - 2: at x = 0, lambda+ = 2 cancels grad f
        # with a gap of 4; the optimum x = 1 leaves the row inactive.
        def make_evaluate(offset):
            def evaluate(x, multipliers, penalty):
                return alm.augmented_inequality_objective(
                    float((x[0] - 1.0) ** 2) + offset, np.array([2.0 * (x[0] - 1.0)]),
                    np.array([x[0] - 2.0]), [np.array([1.0])], multipliers, penalty,
                )
            return evaluate

        decisions = self.decisions(make_evaluate, 0.0, 4.0)
        self.assertEqual(len(set(decisions.values())), 1, decisions)
        self.assertFalse(decisions[0.0][0] and decisions[0.0][2] != 1.0, decisions)


def _bounded(objective_slope):
    """min slope * x on 0 <= x <= 1 with an inert row g = -1 (zero gradient)."""

    def evaluate(x, multipliers, penalty):
        return alm.augmented_inequality_objective(
            objective_slope * float(x[0]), np.array([objective_slope]),
            np.array([-1.0]), [np.zeros(1)], multipliers, penalty,
        )

    return evaluate


class AlmBoundStationarityTests(unittest.TestCase):
    """With base bounds, stationarity is the bound-constrained measure: a
    gradient component pointing out of the box at an active bound is held by
    the bound's multiplier and does not count."""

    def solve(self, slope, x0):
        return _solve_from(_bounded(slope), x0, 0.0, 1.0, base_bounds=[(0.0, 1.0)])

    def test_an_optimum_on_the_upper_bound_converges(self):
        result = self.solve(-1.0, 1.0)
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.termination_reason, "converged")
        np.testing.assert_array_equal(result.x, [1.0])
        self.assertEqual(result.stationarity_norm, 0.0)
        # The unprojected augmented-gradient norm stays in the evaluation.
        self.assertEqual(result.evaluation["stationarity_norm"], 1.0)

    def test_an_optimum_on_the_lower_bound_converges(self):
        result = self.solve(1.0, 0.0)
        self.assertTrue(result.success, result.message)
        np.testing.assert_array_equal(result.x, [0.0])
        self.assertEqual(result.stationarity_norm, 0.0)

    def test_a_gradient_pointing_into_the_box_still_counts(self):
        # At x = 1 the descent direction of f = x points into the box.
        result = self.solve(1.0, 1.0)
        self.assertTrue(result.success, result.message)
        np.testing.assert_allclose(result.x, [0.0], atol=1.0e-12)

    def test_the_reduction_zeroes_only_blocked_components(self):
        bounds = [(0.0, 1.0), (0.0, 1.0), (-np.inf, np.inf)]
        x = np.array([1.0, 0.0, 0.5])
        for grad, expected in (
            (np.array([-3.0, 4.0, 0.0]), 0.0),
            (np.array([3.0, -4.0, 0.0]), 5.0),
            (np.array([-3.0, -4.0, 12.0]), np.hypot(4.0, 12.0)),
        ):
            with self.subTest(grad=grad):
                evaluation = {"grad": grad, "stationarity_norm": float(np.linalg.norm(grad))}
                self.assertEqual(
                    alm_core._bound_reduced_stationarity_norm(evaluation, x, bounds),
                    expected,
                )

    def test_a_point_outside_the_box_is_not_on_its_bound(self):
        bounds = [(0.0, 1.0)]
        for x, grad in ((2.0, -1.0), (-1.0, 1.0)):
            with self.subTest(x=x):
                evaluation = {"grad": np.array([grad])}
                self.assertEqual(
                    alm_core._bound_reduced_stationarity_norm(
                        evaluation, np.array([x]), bounds
                    ),
                    1.0,
                )

    def test_a_point_just_inside_the_box_is_not_on_its_bound(self):
        # Only x equal to the bound is on it: the x0 clip and L-BFGS-B's
        # projection put coordinates exactly on the bound values.
        for bound in (1.0, 1.0e12):
            inside = np.nextafter(bound, -np.inf)
            with self.subTest(bound=bound):
                self.assertEqual(
                    alm_core._bound_reduced_stationarity_norm(
                        {"grad": np.array([-1.0])}, np.array([inside]), [(-np.inf, bound)]
                    ),
                    1.0,
                )

    def test_a_steep_objective_just_inside_the_upper_bound_reaches_it(self):
        # f = 1000 (U - x) with x0 = U - 5e-4 (stored 2^-11 below U = 1e12):
        # the point is interior, f = 0.49, and the optimum is U.
        upper = 1.0e12
        result = _solve_from(
            lambda x, m, p: alm.augmented_inequality_objective(
                1000.0 * float(upper - x[0]), np.array([-1000.0]), np.array([-1.0]),
                [np.zeros(1)], m, p,
            ),
            upper - 5.0e-4, 0.0, 1.0, base_bounds=[(-np.inf, upper)],
        )
        self.assertTrue(result.success, result.message)
        np.testing.assert_array_equal(result.x, [upper])

    def test_a_smooth_objective_two_ulp_inside_the_bound_reaches_it(self):
        # f = tanh(1e16 (1 - x)) at x0 = 1 - 2 eps: f = 0.9997, optimum f(1) = 0.
        scale = 1.0e16

        def evaluate(x, multipliers, penalty):
            phase = scale * (1.0 - x[0])
            return alm.augmented_inequality_objective(
                float(np.tanh(phase)), np.array([-scale / np.cosh(phase) ** 2]),
                np.array([-1.0]), [np.zeros(1)], multipliers, penalty,
            )

        result = _solve_from(
            evaluate, 1.0 - 2.0 * np.finfo(float).eps, 0.0, 1.0, base_bounds=[(0.0, 1.0)],
        )
        self.assertTrue(result.success, result.message)
        np.testing.assert_array_equal(result.x, [1.0])

    def test_an_optimum_on_a_fixed_coordinate_converges(self):
        for slope in (-1.0, 1.0):
            with self.subTest(slope=slope):
                result = _solve_from(
                    _bounded(slope), 0.5, 0.0, 1.0, base_bounds=[(0.5, 0.5)],
                )
                self.assertTrue(result.success, result.message)
                np.testing.assert_array_equal(result.x, [0.5])
                self.assertEqual(result.stationarity_norm, 0.0)

    def test_an_inner_result_within_rounding_of_a_bound_is_placed_on_it(self):
        # L-BFGS-B's last step x + stp * d can stop a few ulps off a bound it
        # reached; the loop snaps such a coordinate onto the bound only where
        # the gradient pushes out, and clips any coordinate outside the box.
        bounds = [(0.0, 1.0), (0.0, 1.0), (-2.0e9, 2.0e9), (0.0, 1.0)]
        few_ulps_below_1 = 1.0 - 4.0 * np.spacing(1.0)
        x = np.array([few_ulps_below_1, 1.0e-3, -2.0e9 - 19 * np.spacing(2.0e9), few_ulps_below_1])
        grad = np.array([-1.0, 1.0, -1.0, 1.0])
        np.testing.assert_array_equal(
            alm_inner._snap_onto_bounds(x, grad, bounds),
            [1.0, 1.0e-3, -2.0e9, few_ulps_below_1],
        )
        np.testing.assert_array_equal(alm_inner._snap_onto_bounds(x, grad, None), x)

    def test_a_snap_that_raises_the_objective_is_not_kept(self):
        # f = C - t + sin(2 pi t), t = (1 - x) / h, h = 64 ulp: the gradient
        # points out of the box at x0 = 1 - h and at 1, but f(1) = f(x0) + 1.
        # L-BFGS-B stays at x0; a snap to 1 would be a worse point that passes
        # the bound test. The decision must not depend on the offset C.
        h = 64.0 * np.spacing(1.0)

        def make_evaluate(offset):
            def evaluate(x, multipliers, penalty):
                t = (1.0 - x[0]) / h
                return alm.augmented_inequality_objective(
                    offset - t + np.sin(2.0 * np.pi * t),
                    np.array([(1.0 - 2.0 * np.pi * np.cos(2.0 * np.pi * t)) / h]),
                    np.array([-1.0]), [np.zeros(1)], multipliers, penalty,
                )
            return evaluate

        decisions = {}
        for offset in (0.0, 1.0e6):
            result = _solve_from(make_evaluate(offset), 1.0 - h, 0.0, 1.0, base_bounds=[(0.0, 1.0)])
            decisions[offset] = (result.success, result.termination_reason, float(result.x[0]))
        self.assertEqual(decisions[0.0], decisions[1.0e6], decisions)
        self.assertNotEqual(decisions[0.0][2], 1.0, decisions)

    def test_a_snap_is_kept_unless_the_total_rises_beyond_round_off(self):
        eps = np.finfo(float).eps
        for before, after, kept in (
            (1.0, 0.5, True), (1.0, 1.0 + 4.0 * eps, True), (1.0, 1.0 + 1.0e-12, False),
            (1.0e6, 1.0e6 + 1.0, False), (-1.0, 0.0, False),
        ):
            with self.subTest(before=before, after=after):
                self.assertIs(
                    alm_inner._snap_keeps_total({"total": before}, {"total": after}), kept
                )

    def test_an_evaluator_stationarity_norm_is_kept(self):
        # Its own measure (not ||grad||): the evaluator accounts for the bounds.
        evaluation = {"grad": np.array([-1.0]), "stationarity_norm": 100.0}
        self.assertEqual(
            alm_core._bound_reduced_stationarity_norm(evaluation, np.array([1.0]), [(0.0, 1.0)]),
            100.0,
        )

    def test_an_x0_outside_the_box_is_projected_onto_it(self):
        # L-BFGS-B clips x0 the same way; the run ends at the bound optimum.
        for slope, x0, optimum in ((-1.0, 2.0, 1.0), (1.0, -1.0, 0.0)):
            with self.subTest(x0=x0):
                result = self.solve(slope, x0)
                self.assertTrue(result.success, result.message)
                np.testing.assert_array_equal(result.x, [optimum])

    def test_a_mixed_out_of_box_x0_ends_at_the_bound_optimum(self):
        def evaluate(x, multipliers, penalty):
            return alm.augmented_inequality_objective(
                float(x[1] - x[0]), np.array([-1.0, 1.0]), np.array([-1.0]),
                [np.zeros(2)], multipliers, penalty,
            )

        result = alm.minimize_alm(
            np.array([2.0, -1.0]), ["inert"], evaluate,
            alm.ALMSettings(max_outer_iterations=2), {"maxiter": 10},
            base_bounds=[(0.0, 1.0), (0.0, 1.0)],
        )
        self.assertTrue(result.success, result.message)
        np.testing.assert_array_equal(result.x, [1.0, 0.0])

    def test_a_resumed_x_outside_the_box_is_rejected(self):
        # A checkpoint of a run with these bounds lies inside them: an x
        # outside means other bounds, and projecting it would not continue
        # the run that wrote it.
        boundaries = []
        _solve_from(
            _row_scaled(), 0.0, 1.1, 1.0, base_bounds=[(-1.0, 1.0)],
            on_outer_boundary=boundaries.append,
        )
        boundary = boundaries[0]
        self.assertIsNone(boundary.termination_reason)
        np.testing.assert_array_equal(boundary.state.x, [0.0])
        for bounds in ([(0.5, 1.0)], [(-1.0, -0.5)]):
            with self.subTest(bounds=bounds):
                with self.assertRaisesRegex(ValueError, "base_bounds"):
                    _solve_from(
                        _row_scaled(), 0.0, None, None, base_bounds=bounds,
                        resume_from=boundary,
                    )


class AlmSignalMismatchTests(unittest.TestCase):
    """A signal mismatch is an actual disagreement of the hard and surrogate
    channels, not a live surrogate shift."""

    def test_identical_channels_at_an_active_boundary_converge(self):
        result = _solve_from(_row_scaled(hybrid=True), 0.1, 1.0, 1.0)
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.termination_reason, "converged")
        np.testing.assert_allclose(result.x, [0.1], atol=1.0e-12)

    def test_identical_channels_are_no_mismatch(self):
        evaluation = _row_scaled(hybrid=True)(np.array([0.1]), np.array([1.0]), 1.0)
        routing = alm_core._constraint_routing_state(evaluation, np.array([1.0]), 1.0, 1.0e-6)
        self.assertFalse(routing.signal_mismatch_active)

    def test_a_surrogate_that_pushes_while_the_hard_row_is_slack_is_a_mismatch(self):
        measured = _measure(MISMATCH_LIVE_SHIFT)
        self.assertTrue(measured.signal_mismatch_active)
        self.assertFalse(measured.routing_state.surrogate_positive_shift_zero)
        self.assertTrue(measured.routing_state.hard_positive_shift_zero)


class AlmExhaustedOuterLabelTests(unittest.TestCase):
    """Each decision that can end the final outer carries its termination
    reason; most of them are golden gaps."""

    def test_raise_decisions_carry_their_max_outer_label(self):
        settings = dataclasses.replace(SETTINGS, continue_on_signal_mismatch=True)
        cases = (
            (
                _view(
                    _measure(INFEASIBLE),
                    infeasible_stall=True,
                    infeasible_stall_reason=STALL_REASON,
                ),
                "max_outer_after_infeasible_stall",
            ),
            (
                _view(
                    _measure(MISMATCH_LIVE_SHIFT, update_stationarity_tol=1.0e-6),
                    meaningful_progress=False,
                ),
                "max_outer_after_signal_mismatch_penalty_increase",
            ),
            (
                _view(
                    _measure(
                        MISMATCH_LIVE_SHIFT,
                        settings=settings,
                        update_stationarity_tol=1.0e-6,
                    ),
                    settings=settings,
                    continuation_iteration=settings.max_subproblem_continuations,
                ),
                "max_outer",
            ),
            (
                _view(
                    _measure(FEASIBLE_NOT_STATIONARY),
                    continuation_iteration=SETTINGS.max_subproblem_continuations,
                ),
                "max_outer_after_subproblem_limit_penalty_increase",
            ),
            (_view(_measure(INFEASIBLE)), "max_outer_after_penalty_increase"),
        )
        for view, label in cases:
            with self.subTest(label=label):
                decision = DefaultContinuationPolicy().after_inner(view)
                self.assertIsInstance(decision, ALMRaisePenalty)
                self.assertEqual(decision.max_outer_termination, label)

    def test_other_outer_ending_decisions_carry_their_label(self):
        self.assertEqual(
            ALMDualUpdateStep(
                penalty_reason=None, feasible_stall_count=0
            ).max_outer_termination,
            "max_outer_after_dual_update",
        )
        self.assertEqual(
            ALMHold(feasible_stall_count=0).max_outer_termination,
            "max_outer_after_sufficient_decrease_hold",
        )
        self.assertEqual(
            ALMContinue(
                trust_radius=None, update_stationarity_tol=1.0, feasible_stall_count=0
            ).max_outer_termination,
            "max_outer",
        )


class AlmInnerPlanTests(unittest.TestCase):
    OPTIONS = MappingProxyType({"maxiter": 300, "ftol": 1e-15, "gtol": 1e-15})

    def _view(self, *, attempt_radius, continuation_iteration=1, start_feasible=True):
        return ALMInnerPlanView(
            settings=SETTINGS,
            inner_options=self.OPTIONS,
            update_stationarity_tol=1.0,
            attempt_radius=attempt_radius,
            continuation_iteration=continuation_iteration,
            start_feasible=start_feasible,
        )

    def test_default_plan_stages_gtol_and_caps_nothing(self):
        expected = {"maxiter": 300, "ftol": 1e-15, "gtol": 1e-4, "maxls": 20}
        for radius, profile in ((None, "unbounded"), (0.05, "boxed")):
            with self.subTest(radius=radius):
                plan = DefaultContinuationPolicy().inner_plan(
                    self._view(attempt_radius=radius)
                )
                self.assertEqual(plan.profile, profile)
                self.assertEqual(dict(plan.options), expected)
                self.assertNotIsInstance(plan.options, dict)


class AlmStalledTrialTests(unittest.TestCase):
    SETTINGS = dataclasses.replace(SETTINGS, max_inner_attempts=2, trust_radius_grow=2.0)

    def _view(self, *, attempt_index=1, attempt_radius=0.1, inner_false_success=True):
        return ALMStalledTrialView(
            settings=self.SETTINGS,
            attempt_index=attempt_index,
            attempt_radius=attempt_radius,
            inner_false_success=inner_false_success,
        )

    def test_default_keeps_the_start_iterate(self):
        self.assertIsNone(DefaultContinuationPolicy().retry_stalled_trial(self._view()))


# --------------------------------------------------------------------------
# Loop-level golden gaps and the trust-radius characterization
# --------------------------------------------------------------------------


def _pushed_away(x, multipliers, penalty):
    """min (x + 3)^2 s.t. x - 1 <= 0: the objective never balances the
    constraint, so an infeasible iterate keeps a large KKT residual."""
    return alm.augmented_inequality_objective(
        (x[0] + 3.0) ** 2,
        np.array([2.0 * (x[0] + 3.0)]),
        np.array([x[0] - 1.0]),
        [np.array([1.0])],
        multipliers,
        penalty,
    )


def _pulled_out(x, multipliers, penalty):
    """min (x - 3)^2 s.t. x - 1 <= 0."""
    return alm.augmented_inequality_objective(
        (x[0] - 3.0) ** 2,
        np.array([2.0 * (x[0] - 3.0)]),
        np.array([x[0] - 1.0]),
        [np.array([1.0])],
        multipliers,
        penalty,
    )


def _scripted_minimize(*points):
    """A stand-in for SciPy's L-BFGS-B that returns ``points`` in order."""
    remaining = list(points)

    def minimize(fun, x, jac, method, bounds, callback, options):
        return SimpleNamespace(
            x=np.array([remaining.pop(0)]),
            nit=1,
            success=True,
            message="CONVERGENCE",
        )

    return minimize


class AlmLoopGapTests(unittest.TestCase):
    def test_hold_on_the_final_outer_labels_the_exhausted_run(self):
        # Golden gap: a sufficient-decrease hold ends the final outer.
        settings = alm.ALMSettings(
            max_outer_iterations=1, max_subproblem_continuations=0, penalty_init=100.0
        )
        with patch.object(alm_inner, "minimize", _scripted_minimize(1.5)):
            result = alm.minimize_alm(
                np.array([3.0]), ["x_cap"], _pushed_away, settings, {"maxiter": 5}
            )
        self.assertEqual(
            result.termination_reason, "max_outer_after_sufficient_decrease_hold"
        )
        self.assertFalse(result.restored_best_feasible)

    def test_budget_stop_restores_the_best_feasible_incumbent(self):
        # Golden gap: the process-budget stop after a hard-feasible incumbent,
        # with a hard-infeasible final iterate.
        settings = alm.ALMSettings(
            max_outer_iterations=3, max_subproblem_continuations=2, penalty_init=100.0
        )
        accepted = []

        def accepted_callback(x):
            accepted.append(float(x[0]))
            if len(accepted) == 2:
                raise alm_control.ALMProcessBudgetExhausted()

        with patch.object(alm_inner, "minimize", _scripted_minimize(0.5, 1.2)):
            result = alm.minimize_alm(
                np.array([0.0]),
                ["x_cap"],
                _pulled_out,
                settings,
                {"maxiter": 5},
                accepted_callback=accepted_callback,
            )
        self.assertEqual(accepted, [0.5, 1.2])
        self.assertEqual(result.termination_reason, "process_budget_exhausted")
        self.assertTrue(result.restored_best_feasible)
        self.assertEqual(result.restored_best_feasible_reason, "final_iterate_infeasible")
        np.testing.assert_array_equal(result.x, [0.5])
        self.assertEqual(
            result.message,
            "ALM stopped after exhausting the this-process accepted-iteration "
            "budget: max_violation=0.000e+00, stationarity=5.000e+00",
        )


class AlmLibraryTrustRadiusCharacterizationTests(unittest.TestCase):
    """A library caller that sets ``trust_radius_init`` and passes no policy
    gets ``DefaultContinuationPolicy``: it boxes each inner solve and shrinks
    or grows the radius by the one core trust rule, with no inner-work caps, no
    false-success retry (``max_inner_attempts=1`` leaves it no attempt to use)
    and no radius growth on a continuation. The pinned path is exact (dyadic
    radii and iterates); the result digest is pinned for the numpy and SciPy
    the goldens were recorded with.
    """

    SETTINGS = alm.ALMSettings(
        max_outer_iterations=6,
        max_subproblem_continuations=2,
        penalty_init=1.0,
        penalty_scale=10.0,
        penalty_max=1.0e6,
        feasibility_tol=1.0e-6,
        stationarity_tol=1.0e-6,
        trust_radius_init=0.25,
        trust_radius_min=1.0e-3,
        trust_radius_shrink=0.5,
        trust_radius_grow=1.5,
        max_inner_attempts=1,
    )
    INNER_OPTIONS = {"maxiter": 50, "maxcor": 10, "ftol": 1.0e-12, "gtol": 1.0e-12}

    def _run(self, **policy):
        events = []
        result = alm.minimize_alm(
            np.zeros(2),
            list(golden.FAR_TARGET_CONSTRAINTS),
            golden.far_target_evaluate,
            self.SETTINGS,
            dict(self.INNER_OPTIONS),
            on_outer_step=events.append,
            **policy,
        )
        steps = [
            (
                event.outer_iteration,
                event.continuation_iteration,
                event.action,
                event.inner.inner_profile,
                event.inner.attempts,
                float(event.after.trust_radius).hex(),
            )
            for event in events
        ]
        return result, events[-1].after.trust_radius, steps

    def assert_pinned(self, result, trust_radius, steps, pins):
        self.assertEqual(result.termination_reason, pins["termination_reason"])
        self.assertEqual(result.restored_best_feasible, pins["restored"])
        self.assertEqual((result.nit, result.outer_iterations), pins["counts"])
        self.assertEqual([float(v).hex() for v in result.x], pins["x"])
        self.assertEqual(float(trust_radius).hex(), pins["trust_radius"])
        self.assertEqual(float(result.penalty).hex(), pins["penalty"])
        self.assertEqual(steps, pins["steps"])
        if golden.bitwise_environment():
            self.assertEqual(golden.encoded_digest(result), pins["digest"])

    DEFAULT_PINS = {
        "termination_reason": "max_outer_restored_best_feasible",
        "restored": True,
        "counts": (11, 6),
        "x": ["0x1.4000000000000p-1", "0x1.4000000000000p-1"],
        "trust_radius": "0x1.b000000000000p-1",
        "penalty": "0x1.0000000000000p+0",
        "steps": [
            (1, 0, "subproblem_continue", "boxed", 1, "0x1.8000000000000p-2"),
            (1, 1, "subproblem_continue", "boxed", 1, "0x1.2000000000000p-1"),
            (1, 2, "dual_update", "boxed", 1, "0x1.b000000000000p-1"),
            (2, 0, "dual_update", "boxed", 1, "0x1.b000000000000p-1"),
            (3, 0, "dual_update", "boxed", 1, "0x1.b000000000000p-1"),
            (4, 0, "dual_update", "boxed", 1, "0x1.b000000000000p-1"),
            (5, 0, "dual_update", "boxed", 1, "0x1.b000000000000p-1"),
            (6, 0, "dual_update", "boxed", 1, "0x1.b000000000000p-1"),
        ],
        "digest": "559233cc244b0d0fdd0ca52921cc51ed0d0c03e37f70821627e63be954abb7b9",
    }

    def test_library_radius_without_a_policy_takes_the_default_path(self):
        self.assert_pinned(*self._run(), self.DEFAULT_PINS)

    def test_explicit_default_policy_takes_the_same_path(self):
        self.assert_pinned(
            *self._run(continuation_policy=DefaultContinuationPolicy()),
            self.DEFAULT_PINS,
        )


if __name__ == "__main__":
    unittest.main()
