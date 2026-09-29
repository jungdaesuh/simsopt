import unittest
from functools import partial

import numpy as np

from simsopt.geo.curveobjectives import CurveCurveDistance, CurveLength
from simsopt.geo.curvexyzfourier import CurveXYZFourier

from simsopt_alm import augmented_inequality_objective
from simsopt_alm.problem import (
    alm_problem_physics,
    evaluate_alm_problem,
    signed_lower_bound,
    signed_upper_bound,
)
from simsopt_alm.signed_constraints import smooth_min_curve_curve_signed_constraint


_HYBRID_SIGNAL_KEYS = frozenset(
    (
        "hard_signed_constraint_values",
        "hard_violation_values",
        "surrogate_signed_constraint_values",
        "hard_dual_update_values",
    )
)


def _circle(radius, center_z, order=1):
    curve = CurveXYZFourier(64, order)
    curve.set("xc(1)", radius)
    curve.set("ys(1)", radius)
    curve.set("zc(0)", center_z)
    return curve


class SignedBoundTests(unittest.TestCase):
    def test_upper_and_lower_bounds_sign_the_objective_and_its_gradient(self):
        curve = _circle(1.0, 0.0)
        length = CurveLength(curve)
        length_grad = length.dJ()

        upper_value, upper_grad = signed_upper_bound(length, 5.0, length)
        lower_value, lower_grad = signed_lower_bound(length, 7.0, length)

        # g <= 0 means feasible: 2*pi <= 5 is violated, 2*pi >= 7 is violated.
        self.assertAlmostEqual(upper_value, 2.0 * np.pi - 5.0, places=12)
        self.assertAlmostEqual(lower_value, 7.0 - 2.0 * np.pi, places=12)
        np.testing.assert_allclose(upper_grad, length_grad, rtol=0, atol=1e-15)
        np.testing.assert_allclose(lower_grad, -length_grad, rtol=0, atol=1e-15)

    def test_gradient_is_over_the_base_objective_dofs(self):
        # The constraint touches only one curve; its gradient still spans every
        # free dof of the base objective, zero on the other curve.
        near, far = _circle(1.0, 0.0), _circle(1.0, 0.5)
        base_objective = CurveLength(near) + CurveLength(far)

        _value, grad = signed_upper_bound(CurveLength(far), 5.0, base_objective)

        self.assertEqual(grad.shape, base_objective.x.shape)
        near_size = near.x.size
        np.testing.assert_array_equal(grad[:near_size], np.zeros(near_size))
        np.testing.assert_allclose(grad[near_size:], CurveLength(far).dJ())


class EvaluateAlmProblemTests(unittest.TestCase):
    def test_returns_only_augmented_inequality_objective_keys(self):
        curve = _circle(1.0, 0.0)
        length = CurveLength(curve)
        inequalities = [
            partial(signed_upper_bound, length, 5.0),
            partial(signed_lower_bound, length, 7.0),
        ]
        dofs = 1.1 * length.x
        multipliers = np.array([0.5, 0.0])

        evaluation = evaluate_alm_problem(
            dofs, length, inequalities, multipliers, 10.0
        )

        # The dofs were applied before evaluating: the circle has radius 1.1.
        np.testing.assert_array_equal(length.x, dofs)
        circumference = 2.0 * np.pi * 1.1
        reference = augmented_inequality_objective(
            base_value=circumference,
            base_grad=length.dJ(),
            constraint_values=[circumference - 5.0, 7.0 - circumference],
            constraint_grads=[length.dJ(), -length.dJ()],
            multipliers=multipliers,
            penalty=10.0,
        )
        self.assertEqual(set(evaluation), set(reference))
        self.assertFalse(_HYBRID_SIGNAL_KEYS & set(evaluation))
        for key in (
            "constraint_values",
            "feasibility_values",
            "dual_update_values",
            "grad",
        ):
            with self.subTest(key=key):
                np.testing.assert_allclose(evaluation[key], reference[key])
        self.assertAlmostEqual(evaluation["total"], reference["total"], places=12)

    def test_separated_circles_report_signed_slack_through_the_adapter(self):
        # Stock CurveCurveDistance.J() is a hinge: zero for any separation above
        # the threshold. The signed kernel row keeps the slack, g < 0, and the
        # adapter passes it through as a feasible row.
        curves = [_circle(1.0, 0.0), _circle(1.0, 0.5)]
        base_objective = CurveLength(curves[0]) + CurveLength(curves[1])
        minimum_distance = 0.1
        inequalities = [
            partial(
                smooth_min_curve_curve_signed_constraint,
                curves,
                minimum_distance,
                0.005,
            )
        ]

        evaluation = evaluate_alm_problem(
            base_objective.x, base_objective, inequalities, np.zeros(1), 1.0
        )

        self.assertEqual(CurveCurveDistance(curves, minimum_distance).J(), 0.0)
        self.assertLess(evaluation["constraint_values"][0], 0.0)
        np.testing.assert_array_equal(evaluation["feasibility_values"], [0.0])

    def test_total_gradient_matches_central_difference(self):
        # Active rows (positive multiplier shift) so every term of L contributes.
        curves = [_circle(1.0, 0.0, order=2), _circle(0.9, 0.3, order=2)]
        curves[0].set("xc(2)", 0.05)
        curves[1].set("zs(2)", 0.04)
        lengths = [CurveLength(curve) for curve in curves]
        base_objective = lengths[0] + lengths[1]
        inequalities = [
            partial(signed_upper_bound, lengths[0], 5.0),
            partial(signed_lower_bound, lengths[1], 6.0),
            partial(smooth_min_curve_curve_signed_constraint, curves, 0.4, 0.01),
        ]
        multipliers = np.array([0.3, 0.2, 0.5])
        penalty = 10.0
        x0 = base_objective.x.copy()
        direction = np.random.default_rng(0).standard_normal(x0.size)

        def total_at(dofs):
            return evaluate_alm_problem(
                dofs, base_objective, inequalities, multipliers, penalty
            )["total"]

        analytic = evaluate_alm_problem(
            x0, base_objective, inequalities, multipliers, penalty
        )
        self.assertTrue(np.all(analytic["positive_shift_values"] > 0.0))
        step = 1.0e-6
        finite_difference = (
            total_at(x0 + step * direction) - total_at(x0 - step * direction)
        ) / (2.0 * step)
        np.testing.assert_allclose(
            analytic["grad"] @ direction, finite_difference, rtol=1.0e-6
        )


def _assert_bitwise_equal_evaluations(test, got, want):
    """Same keys in the same order; every array, list and float bit-for-bit."""
    test.assertEqual(list(got), list(want), "keys or key order differ")
    for key in want:
        with test.subTest(key=key):
            got_value, want_value = got[key], want[key]
            test.assertIs(type(got_value), type(want_value))
            if isinstance(want_value, list):
                test.assertEqual(len(got_value), len(want_value))
                for got_item, want_item in zip(got_value, want_value):
                    test.assertEqual(got_item.dtype, want_item.dtype)
                    test.assertEqual(got_item.tobytes(), want_item.tobytes())
            else:
                got_array, want_array = np.asarray(got_value), np.asarray(want_value)
                test.assertEqual(got_array.dtype, want_array.dtype)
                test.assertEqual(got_array.shape, want_array.shape)
                test.assertEqual(got_array.tobytes(), want_array.tobytes())


class EvaluateAlmProblemCompatibilityTests(unittest.TestCase):
    """``evaluate_alm_problem`` is the pre-5.9 evaluator, bit for bit."""

    def _problem(self):
        curves = [_circle(1.0, 0.0, order=2), _circle(0.9, 0.3, order=2)]
        curves[0].set("xc(2)", 0.05)
        curves[1].set("zs(2)", 0.04)
        lengths = [CurveLength(curve) for curve in curves]
        base_objective = lengths[0] + lengths[1]
        inequalities = [
            partial(signed_upper_bound, lengths[0], 5.0),
            partial(signed_lower_bound, lengths[1], 6.0),
            partial(smooth_min_curve_curve_signed_constraint, curves, 0.4, 0.01),
        ]
        return base_objective, inequalities

    def test_evaluate_alm_problem_legacy_bitwise(self):
        base_objective, inequalities = self._problem()
        dofs = base_objective.x * 1.01
        multipliers = np.array([0.3, 0.0, 0.5])
        penalty = np.array([10.0, 2.0, 5.0])

        evaluation = evaluate_alm_problem(
            dofs, base_objective, inequalities, multipliers, penalty
        )

        # The body evaluate_alm_problem had before 5.9, inlined.
        base_objective.x = dofs
        rows = [row(base_objective)[:2] for row in inequalities]
        legacy = augmented_inequality_objective(
            base_value=float(base_objective.J()),
            base_grad=np.asarray(base_objective.dJ(), dtype=float),
            constraint_values=[signed_value for signed_value, _grad in rows],
            constraint_grads=[grad for _signed_value, grad in rows],
            multipliers=multipliers,
            penalty=penalty,
        )
        _assert_bitwise_equal_evaluations(self, evaluation, legacy)
        np.testing.assert_array_equal(base_objective.x, dofs)

    def test_alm_problem_physics_is_evaluate_alm_problem_without_the_terms(self):
        base_objective, inequalities = self._problem()
        dofs = base_objective.x * 0.99
        multipliers = np.array([0.1, 0.2, 0.0])

        physics = alm_problem_physics(dofs, base_objective, inequalities)

        np.testing.assert_array_equal(base_objective.x, dofs)
        for penalty in (1.0, 30.0):
            with self.subTest(penalty=penalty):
                _assert_bitwise_equal_evaluations(
                    self,
                    physics.evaluation(multipliers, penalty),
                    evaluate_alm_problem(
                        dofs, base_objective, inequalities, multipliers, penalty
                    ),
                )


if __name__ == "__main__":
    unittest.main()
