"""Who owns ``success``: the continuation policy, not the loop.

The loop executes a policy's ``ALMConverge`` as given; it adds no core
convergence veto. The success guarantees (an approximate KKT point at the
shifted multipliers, within the feasibility, stationarity and
complementarity-gap tolerances; no hybrid signal mismatch, no binding
multiplier cap) are those of
``DefaultContinuationPolicy`` and of policies that keep its vetoes. The test
pins that contract as behavior.
"""

import unittest

import numpy as np

import simsopt_alm as alm
from simsopt_alm.continuation import ALMConverge
from simsopt_alm.policy import DefaultContinuationPolicy


class _ConvergeImmediately:
    """Declares convergence at the first start iterate, feasible or not."""

    def __init__(self):
        self.default = DefaultContinuationPolicy()

    def inner_plan(self, view):
        return self.default.inner_plan(view)

    def retry_stalled_trial(self, view):
        return self.default.retry_stalled_trial(view)

    def before_inner(self, view):
        return ALMConverge(
            action="converged",
            termination_reason="converged",
            message="declared by the policy",
            feasible_stall_count=view.feasible_stall_count,
        )

    def after_inner(self, view):
        return self.default.after_inner(view)


class AlmPolicyAuthorityTests(unittest.TestCase):
    def test_the_loop_executes_a_policy_convergence_as_given(self):
        def evaluate_problem(x, multipliers, penalty):
            return alm.augmented_inequality_objective(
                float(x @ x), 2.0 * x, np.array([1.0 - x[0]]), [np.array([-1.0, 0.0])],
                multipliers, penalty,
            )

        settings = alm.ALMSettings()
        result = alm.minimize_alm(
            np.zeros(2), ["x0_at_least_one"], evaluate_problem, settings, {"maxiter": 50},
            continuation_policy=_ConvergeImmediately(),
        )
        # x0 = 0 violates 1 - x0 <= 0 by 1: only the policy's word makes this a success.
        self.assertTrue(result.success)
        self.assertGreater(result.max_violation, settings.feasibility_tol)
        self.assertEqual(result.message, "declared by the policy")


if __name__ == "__main__":
    unittest.main()
