"""Regression tests for the logic bugs of the 2026-09-30 cross-lab review of
1b7387b74 (Codex Astra B1-B8, Grok G1-G4). Each test is the reviewer's
reproduction on an analytic problem with a known KKT point, asserting the
correct outcome; each failed at 1b7387b74.
"""

import unittest

from simsopt_alm import ALMSettings, augmented_inequality_objective, minimize_alm


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


if __name__ == "__main__":
    unittest.main()
