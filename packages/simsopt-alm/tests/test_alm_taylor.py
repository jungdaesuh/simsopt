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
