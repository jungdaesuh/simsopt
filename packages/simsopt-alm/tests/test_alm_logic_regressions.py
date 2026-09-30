"""Regression tests for the logic bugs of the 2026-09-30 cross-lab review of
1b7387b74 (Codex Astra B1-B8, Grok G1-G4). Each test is the reviewer's
reproduction on an analytic problem with a known KKT point, asserting the
correct outcome; each failed at 1b7387b74.
"""

import dataclasses
import unittest

import numpy as np

import simsopt_alm as alm
from simsopt_alm import ALMSettings, augmented_inequality_objective, minimize_alm
from simsopt_alm.checkpoint import resume_boundary, transition_snapshot
from simsopt_alm.continuation import ALMInnerPlan, ALMStalledTrialView
from simsopt_alm.core import _constraint_routing_state
from simsopt_alm.policy import DefaultContinuationPolicy


def shifted_multipliers(result):
    """The multipliers ``converged`` certifies: max(0, lambda + rho g)."""
    return np.maximum(0.0, result.multipliers + result.penalty * result.constraint_values)


def pulled_out(x, multipliers, penalty):
    """min (x - 3)^2 s.t. x - 1 <= 0: x* = 1, lambda* = 4. At lambda = 0 and
    rho = 1 the augmented Lagrangian's minimizer 7/3 violates the row by 4/3."""
    return augmented_inequality_objective(
        (x[0] - 3.0) ** 2, [2.0 * (x[0] - 3.0)], [x[0] - 1.0], [[1.0]], multipliers, penalty
    )


class FeasibleStartWithAnInfeasibleSubproblemMinimizerTests(unittest.TestCase):
    """G3 (inner.py): an inner solution beyond the feasibility slack was rolled
    back to the start, so two rollbacks stopped as ``plateau_stall`` at x = 0
    and the penalty never rose."""

    def assert_kkt(self, result):
        self.assertEqual(result.termination_reason, "converged", result.message)
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.x, [1.0], atol=1.0e-6)
        np.testing.assert_allclose(shifted_multipliers(result), [4.0], atol=1.0e-4)

    def test_default_settings_reach_the_kkt_point(self):
        for maxiter in (30, 200):
            with self.subTest(maxiter=maxiter):
                self.assert_kkt(
                    minimize_alm([0.0], ["g"], pulled_out, ALMSettings(), {"maxiter": maxiter})
                )

    def test_a_start_on_the_binding_row_reaches_its_multiplier(self):
        # Grok's G2 example too: at x = 1 with lambda = 0 the old gate (the
        # fitted residual, 0) updated lambda by rho * g = 0 on every outer.
        self.assert_kkt(minimize_alm([1.0], ["g"], pulled_out, ALMSettings(), {"maxiter": 200}))

    def test_the_first_step_raises_the_penalty_instead_of_stalling(self):
        events = []
        minimize_alm(
            [0.0], ["g"], pulled_out, ALMSettings(), {"maxiter": 200},
            on_outer_step=events.append,
        )
        first = events[0]
        self.assertEqual(first.action, "penalty_increase")
        np.testing.assert_allclose(first.inner.x, [7.0 / 3.0], atol=1.0e-6)

    def test_a_box_that_cannot_hold_the_step_within_the_slack_raises_the_penalty(self):
        # One attempt per subproblem: the box around x = 0 contains 7/3, the
        # step cannot shrink, and the rejected step used to be rolled back
        # with the same radius on every continuation.
        result = minimize_alm(
            [0.0], ["g"], pulled_out,
            ALMSettings(trust_radius_init=4.0, max_inner_attempts=1), {"maxiter": 200},
        )
        self.assert_kkt(result)


class DualUpdateGateTests(unittest.TestCase):
    """G2 = B3 (policy.py): the multiplier update was gated on the fitted
    active-set residual instead of the augmented-gradient norm, so it ran at
    points that do not minimize L_A, or skipped a multiplier that must move."""

    def test_no_multiplier_update_away_from_a_subproblem_minimizer(self):
        # min (x - 2)^2 s.t. x - 1 <= 0 at x = 1.2, rho = 1, no inner work:
        # ||grad L_A|| = 1.4 exceeds the update tolerance max(1e-6, 1/rho) = 1.
        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                (x[0] - 2.0) ** 2, [2.0 * (x[0] - 2.0)], [x[0] - 1.0], [[1.0]],
                multipliers, penalty,
            )

        events = []
        minimize_alm(
            [1.2], ["g"], evaluate, ALMSettings(max_outer_iterations=1), {"maxiter": 0},
            on_outer_step=events.append,
        )
        self.assertEqual(len(events), 1)
        self.assertNotEqual(events[0].action, "dual_update")
        np.testing.assert_array_equal(events[0].after.multipliers, [0.0])

    def test_an_excess_multiplier_on_an_inactive_row_is_updated(self):
        # Astra B3: min x0^2/2 - x1 s.t. x0 - 1 <= 0, x1 <= 0 from the exact
        # subproblem minimizer; KKT point x* = (0, 0), lambda* = (0, 1).
        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                x[0] ** 2 / 2.0 - x[1], [x[0], -1.0], [x[0] - 1.0, x[1]],
                [[1.0, 0.0], [0.0, 1.0]], multipliers, penalty,
            )

        result = minimize_alm(
            [0.0, 0.0], ["inactive", "active"], evaluate, ALMSettings(penalty_init=10.0),
            {"maxiter": 100}, initial_multipliers=np.array([20.0, 1.0]),
        )
        self.assertEqual(result.termination_reason, "converged", result.message)
        np.testing.assert_allclose(result.x, [0.0, 0.0], atol=1.0e-6)
        np.testing.assert_allclose(shifted_multipliers(result), [0.0, 1.0], atol=1.0e-6)


def straddling_hybrid(hard, surrogate):
    """min (x - 1)^2 with one constant row whose hard and surrogate values are
    ``hard`` and ``surrogate``, both feasible to ``feasibility_tol``."""

    def evaluate(x, multipliers, penalty):
        physics = alm.ALMPhysics(
            base_value=float((x[0] - 1.0) ** 2),
            base_grad=np.array([2.0 * (x[0] - 1.0)]),
            constraint_values=np.array([surrogate]),
            constraint_grads=(np.array([1.0]),),
            extras={
                "dual_update_values": np.array([hard]),
                "feasibility_values": np.array([max(hard, 0.0)]),
                "hard_signed_constraint_values": np.array([hard]),
                "hard_violation_values": np.array([max(hard, 0.0)]),
                "surrogate_signed_constraint_values": np.array([surrogate]),
                "hard_dual_update_values": np.array([hard]),
            },
        )
        return physics.evaluation(multipliers, penalty)

    return evaluate


class SignalMismatchTests(unittest.TestCase):
    """G1 (core.py): the mismatch flag came from unequal activity masks, so a
    hybrid pair straddling 0 within the tolerance never converged."""

    def test_channels_straddling_zero_within_the_tolerance_converge(self):
        for hard, surrogate in ((-5.0e-7, 5.0e-7), (-1.0e-8, 1.0e-9)):
            with self.subTest(hard=hard, surrogate=surrogate):
                result = minimize_alm(
                    [1.0], ["g"], straddling_hybrid(hard, surrogate),
                    ALMSettings(max_outer_iterations=2), {"maxiter": 5},
                )
                self.assertEqual(result.termination_reason, "converged", result.message)
                self.assertTrue(result.success)

    def test_a_zero_shift_activity_split_is_no_mismatch(self):
        # Hard row on its boundary (active), surrogate 0.05 inside it
        # (inactive, zero shift): no multiplier rides on the row in either
        # channel, so the KKT conditions of both problems agree.
        evaluation = straddling_hybrid(0.0, -0.05)(np.array([1.0]), np.zeros(1), 10.0)
        evaluation["constraint_activity_tolerances"] = np.array([0.02])
        routing = _constraint_routing_state(evaluation, np.zeros(1), 10.0, 1.0e-6)
        self.assertTrue(routing.hard_activity_mask[0])
        self.assertFalse(routing.surrogate_activity_mask[0])
        self.assertFalse(routing.signal_mismatch_active)

    def test_a_live_shift_beyond_the_gate_is_still_a_mismatch(self):
        evaluation = straddling_hybrid(-0.05, 0.001)(np.array([1.0]), np.zeros(1), 10.0)
        routing = _constraint_routing_state(evaluation, np.zeros(1), 10.0, 1.0e-6)
        self.assertTrue(routing.signal_mismatch_active)


def at_most_zero(x, multipliers, penalty):
    """min (x - 1)^2 s.t. x <= 0: x* = 0, lambda* = 2."""
    return augmented_inequality_objective(
        (x[0] - 1.0) ** 2, [2.0 * (x[0] - 1.0)], [x[0]], [[1.0]], multipliers, penalty
    )


class FeasibleStartIncumbentTests(unittest.TestCase):
    """B1 (control.py): a feasible start was never a best-feasible candidate,
    so a run stopped by its limits returned an infeasible later iterate."""

    def test_an_exhausted_run_restores_the_feasible_start(self):
        result = minimize_alm(
            [0.0], ["g"], at_most_zero, ALMSettings(max_outer_iterations=1), {"maxiter": 100}
        )
        self.assertFalse(result.success)
        self.assertTrue(result.restored_best_feasible)
        self.assertEqual(result.restored_best_feasible_reason, "final_iterate_infeasible")
        np.testing.assert_array_equal(result.x, [0.0])
        self.assertEqual(result.max_violation, 0.0)


class FlaggedFiniteTrialTests(unittest.TestCase):
    """B4 (inner.py): a trial flagged ``nonfinite_evaluation`` with finite data
    was not sanitized, so L-BFGS-B took its made-up total and gradient."""

    def test_a_flagged_finite_trial_is_backtracked_like_a_nan_one(self):
        for flagged_total in (0.0, float("nan")):
            with self.subTest(flagged_total=flagged_total):

                def evaluate(x, multipliers, penalty):
                    built = augmented_inequality_objective(
                        (x[0] - 0.4) ** 2, [2.0 * (x[0] - 0.4)], [], [], multipliers, penalty
                    )
                    if x[0] > 0.5:
                        return dict(built, total=flagged_total, grad=np.zeros(1), nonfinite_evaluation=True)
                    return built

                result = minimize_alm([0.0], [], evaluate, ALMSettings(), {"maxiter": 100})
                self.assertEqual(result.termination_reason, "converged", result.message)
                np.testing.assert_allclose(result.x, [0.4], atol=1.0e-6)


class TaylorOwnershipTests(unittest.TestCase):
    """B5 (taylor.py): the test read the evaluator's reused buffers after the
    next evaluation had overwritten them."""

    def test_a_right_gradient_in_a_reused_buffer_passes(self):
        buffer = np.zeros(2)

        def evaluate(x, multipliers, penalty):
            buffer[:] = 2.0 * x
            return {"total": float(x @ x), "grad": buffer}

        result = alm.run_directional_taylor_test(evaluate, [1.0, 2.0], [], 1.0, direction_count=2)
        self.assertEqual(result["status"], "passed", result)

    def test_a_wrong_gradient_in_a_reused_dict_fails(self):
        shared = {"total": 0.0, "grad": np.zeros(1)}

        def evaluate(x, multipliers, penalty):
            shared["total"] = float(x @ x)
            return shared

        result = alm.run_directional_taylor_test(evaluate, [1.0], [], 1.0, direction=[1.0])
        self.assertEqual(result["status"], "failed", result)


class InnerPlanBudgetTests(unittest.TestCase):
    """B6 (inner.py): a custom plan's maxiter ran past the call's budget."""

    def test_a_plan_cannot_raise_maxiter_past_the_call_budget(self):
        class GenerousPlan(DefaultContinuationPolicy):
            def inner_plan(self, view):
                return ALMInnerPlan("custom", dict(super().inner_plan(view).options, maxiter=100))

        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                x[0] ** 2 + 10.0 * x[1] ** 2, [2.0 * x[0], 20.0 * x[1]], [], [], multipliers, penalty
            )

        result = minimize_alm(
            [1.0, 1.0], [], evaluate, ALMSettings(), {"maxiter": 1}, continuation_policy=GenerousPlan()
        )
        self.assertLessEqual(result.nit, 1)

    def test_a_plan_without_maxiter_gets_the_call_budget(self):
        class UncappedPlan(DefaultContinuationPolicy):
            def inner_plan(self, view):
                options = dict(super().inner_plan(view).options)
                options.pop("maxiter", None)
                return ALMInnerPlan("custom", options)

        def evaluate(x, multipliers, penalty):
            return augmented_inequality_objective(
                x[0] ** 2 + 10.0 * x[1] ** 2, [2.0 * x[0], 20.0 * x[1]], [], [], multipliers, penalty
            )

        result = minimize_alm(
            [1.0, 1.0], [], evaluate, ALMSettings(), {"maxiter": 1}, continuation_policy=UncappedPlan()
        )
        self.assertLessEqual(result.nit, 1)


class PhysicsTotalValidationTests(unittest.TestCase):
    """B7 (evaluation.py): a nonfinite ``physics_total`` (the incumbent rank
    and the returned objective) escaped the finiteness check."""

    def test_a_nonfinite_physics_total_at_an_iterate_raises(self):
        def evaluate(x, multipliers, penalty):
            return dict(
                augmented_inequality_objective(x[0] ** 2, [2.0 * x[0]], [], [], multipliers, penalty),
                physics_total=float("nan"),
            )

        with self.assertRaisesRegex(ValueError, "non-finite ALM data: physics_total"):
            minimize_alm([0.0], [], evaluate, ALMSettings(), {"maxiter": 100})


class ScalarArrayDiagnosticCheckpointTests(unittest.TestCase):
    """B8 (checkpoint.py): a 0-d ndarray diagnostic crashed transition_snapshot."""

    def test_a_zero_dimensional_diagnostic_round_trips_as_a_scalar(self):
        def evaluate(x, multipliers, penalty):
            return dict(
                augmented_inequality_objective(x[0] ** 2, [2.0 * x[0]], [], [], multipliers, penalty),
                diagnostic=np.array(1.5),
            )

        boundaries = []
        minimize_alm([0.0], [], evaluate, ALMSettings(), {"maxiter": 100}, on_outer_boundary=boundaries.append)
        snapshot = transition_snapshot(boundaries[0], {"maxiter": 100})
        self.assertEqual(dict(snapshot.best_feasible.evaluation)["diagnostic"], 1.5)
        restored = resume_boundary(snapshot)
        self.assertEqual(restored.state.best_feasible.evaluation["diagnostic"], 1.5)


class TrustRadiusValidationTests(unittest.TestCase):
    """G4 (core.py, inner.py): a zero trust radius was accepted and treated as
    no box. None is the no-box value; a given radius must be positive."""

    def test_a_nonpositive_initial_radius_is_rejected(self):
        for radius in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(radius=radius):
                with self.assertRaisesRegex(ValueError, "trust_radius_init"):
                    ALMSettings(trust_radius_init=radius)
        self.assertIsNone(ALMSettings(trust_radius_init=None).trust_radius_init)

    def test_a_policy_radius_must_be_positive(self):
        class ZeroRetry(DefaultContinuationPolicy):
            def retry_stalled_trial(self, view: ALMStalledTrialView):
                return 0.0

        def frozen(x, multipliers, penalty):
            # A violated row with a zero gradient everywhere: L-BFGS-B stops
            # where it starts, the trial stalls infeasible and the policy is
            # asked for a retry radius.
            return augmented_inequality_objective(0.0, [0.0], [1.0], [[0.0]], multipliers, penalty)

        with self.assertRaisesRegex(ValueError, "trust radius"):
            minimize_alm(
                [3.0], ["g"], frozen, ALMSettings(), {"maxiter": 10},
                continuation_policy=ZeroRetry(),
            )

    def test_a_resume_boundary_with_a_zero_radius_is_rejected(self):
        boundaries = []
        minimize_alm(
            [0.0], ["g"], pulled_out, ALMSettings(max_outer_iterations=3, trust_radius_init=0.5),
            {"maxiter": 200}, on_outer_boundary=boundaries.append,
        )
        boundary = boundaries[0]
        self.assertIsNone(boundary.termination_reason)
        broken = dataclasses.replace(
            boundary, state=dataclasses.replace(boundary.state, trust_radius=0.0)
        )
        with self.assertRaisesRegex(ValueError, "trust_radius"):
            minimize_alm(
                list(broken.state.x), ["g"], pulled_out,
                ALMSettings(max_outer_iterations=3, trust_radius_init=0.5), {"maxiter": 200},
                resume_from=broken,
            )


if __name__ == "__main__":
    unittest.main()
