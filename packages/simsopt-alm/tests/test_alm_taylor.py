"""run_directional_taylor_test passes only on finite evidence that establishes
the ratio test (Codex round 16, R16-04): nonfinite totals or gradients report
``unavailable``, invalid or too few steps raise, and steps that check no ratio
while an error stays above the floor report ``unavailable``."""

import unittest

import numpy as np

from simsopt_alm import run_directional_taylor_test

X0 = np.array([1.0])
DIRECTION = np.array([1.0])


def quadratic(x, multipliers, penalty):
    return {"total": float(x @ x), "grad": 2.0 * x}


def sine(x, multipliers, penalty):
    return {"total": float(np.sum(np.sin(x))), "grad": np.cos(x)}


def zero_gradient_quadratic(x, multipliers, penalty):
    return {"total": float(x @ x), "grad": np.zeros_like(x)}


def nonfinite_trials(x, multipliers, penalty):
    return {"total": float(x @ x) if x[0] == 1.0 else float("nan"), "grad": 2.0 * x}


def nonfinite_gradient(x, multipliers, penalty):
    return {"total": float(x @ x), "grad": np.full_like(x, float("nan"))}


def nonfinite_everything(x, multipliers, penalty):
    return {"total": float("nan"), "grad": np.full_like(x, float("nan"))}


def one_sided_bump(x, multipliers, penalty):
    """x^2 plus 1e-3 on (1, 1.008): exact central differences at step 0.01,
    an error of 0.1 at step 0.005."""
    bump = 1.0e-3 if 0.0 < x[0] - 1.0 < 0.008 else 0.0
    return {"total": float(x @ x) + bump, "grad": 2.0 * x}


def cancelling_wrong_gradient(x, multipliers, penalty):
    """x^2 + 0.1 (x - 1) - 419430.4 (x - 1)^3 with the claimed gradient 2x: the
    true derivative at x = 1 is 2.1, and the cubic term cancels the 0.1 error
    exactly at the default step 2^-11 (error 0), then the error rebounds."""
    delta = x[0] - 1.0
    return {"total": float(x[0] ** 2 + 0.1 * delta - 419430.4 * delta ** 3), "grad": 2.0 * x}


def crossing_right_gradient(x, multipliers, penalty):
    """x^2 + (x - 1)^3 - 2^20 (x - 1)^5 with its exact gradient: the central
    difference error at x = 1, h^2 - 2^20 h^4, is exactly 0 at h = 2^-10 and
    then decreases by about 4 per halving."""
    delta = x[0] - 1.0
    return {
        "total": float(x[0] ** 2 + delta ** 3 - 2.0 ** 20 * delta ** 5),
        "grad": np.array([2.0 * x[0] + 3.0 * delta ** 2 - 5.0 * 2.0 ** 20 * delta ** 4]),
    }


def taylor(evaluate, **options):
    return run_directional_taylor_test(evaluate, X0, np.zeros(0), 1.0, direction=DIRECTION, **options)


class TaylorVerdictTests(unittest.TestCase):
    def assert_status(self, result, status):
        self.assertEqual(result["status"], status, result)
        self.assertEqual(result["passed"], status == "passed", result)

    def test_a_right_gradient_passes(self):
        self.assert_status(taylor(quadratic), "passed")
        self.assert_status(taylor(sine), "passed")
        self.assert_status(taylor(sine, epsilons=[0.01, 0.005]), "passed")

    def test_a_wrong_gradient_fails(self):
        self.assert_status(taylor(zero_gradient_quadratic), "failed")
        self.assert_status(taylor(zero_gradient_quadratic, epsilons=[0.01, 0.005]), "failed")

    def test_nonfinite_totals_along_the_steps_are_unavailable(self):
        self.assert_status(taylor(nonfinite_trials), "unavailable")

    def test_a_nonfinite_gradient_is_unavailable(self):
        self.assert_status(taylor(nonfinite_gradient), "unavailable")

    def test_an_all_nonfinite_evaluator_is_unavailable(self):
        result = run_directional_taylor_test(nonfinite_everything, X0, [], 1.0)
        self.assert_status(result, "unavailable")

    def test_steps_that_check_no_ratio_are_unavailable(self):
        result = taylor(one_sided_bump, epsilons=[0.01, 0.005])
        self.assertEqual(result["ratios"], [])
        self.assertGreater(result["errors"][-1], 0.05)
        self.assert_status(result, "unavailable")

    def test_a_rebound_after_an_exact_cancellation_does_not_pass(self):
        # GEO-01: the errors fall by ~4 per halving, cancel to exactly 0 at
        # the fifth step, and rebound above the floor at the last one, whose
        # ratio cannot be taken against 0. That rebound is unresolved
        # evidence, never a pass.
        result = taylor(cancelling_wrong_gradient)
        self.assertEqual(result["errors"][-2], 0.0)
        self.assertGreater(result["errors"][-1], 0.05)
        self.assert_status(result, "unavailable")

    def test_a_rebound_that_keeps_missing_the_claim_fails(self):
        # More steps after the rebound: the error settles at the 0.1 the
        # claim misses by, and the next ratio test fails.
        steps = [0.5 ** power for power in range(7, 16)]
        self.assert_status(taylor(cancelling_wrong_gradient, epsilons=steps), "failed")

    def test_a_rebound_that_converges_again_passes(self):
        # A right gradient whose error crosses 0 at one step: the step after
        # the rebound checks a ratio (about 1/4) again, so it passes.
        result = taylor(crossing_right_gradient)
        self.assertEqual(result["errors"][3], 0.0)
        self.assertGreater(result["errors"][4], 1.0e-10 * 2.0)
        self.assert_status(result, "passed")

    def test_one_step_cannot_establish_the_ratio_test(self):
        with self.assertRaisesRegex(ValueError, "at least two"):
            taylor(zero_gradient_quadratic, epsilons=[0.01])

    def test_invalid_steps_are_rejected(self):
        for epsilons in ([], [0.01, 0.0], [0.01, -0.005], [0.01, float("nan")],
                         [float("inf"), 0.01], [0.005, 0.01], [0.01, 0.01]):
            with self.subTest(epsilons=epsilons):
                with self.assertRaisesRegex(ValueError, "epsilons"):
                    taylor(quadratic, epsilons=epsilons)


if __name__ == "__main__":
    unittest.main()
