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
