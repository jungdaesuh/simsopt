"""The library returns a lean, frozen ``ALMResult``.

It carries what a caller needs to use and judge the answer; per-step detail is
in the outer-step events. These tests pin its fields and that it is read-only
all the way down.
"""

import dataclasses
import sys
import unittest
from pathlib import Path
from types import MappingProxyType

import numpy as np

import simsopt_alm as alm

GOLDEN_DIR = Path(__file__).resolve().parent / "alm_golden"
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))
import alm_golden_scenarios as golden  # noqa: E402

LEAN_FIELDS = (
    "x",
    "success",
    "termination_reason",
    "message",
    "objective",
    "constraint_names",
    "constraint_values",
    "max_violation",
    "multipliers",
    "penalty",
    "stationarity_norm",
    "kkt_stationarity_norm",
    "nit",
    "outer_iterations",
    "restored_best_feasible",
    "restored_best_feasible_reason",
    "evaluation",
    "inner_result",
    "last_iterate",
)


def _solve_unit_halfspace_problem():
    """min ||x||^2 s.t. 1 - x0 <= 0; the solution is x = (1, 0), lambda = 2."""

    def evaluate_problem(x, multipliers, penalty):
        return alm.augmented_inequality_objective(
            base_value=float(x @ x),
            base_grad=2.0 * x,
            constraint_values=np.array([1.0 - x[0]]),
            constraint_grads=[np.array([-1.0, 0.0])],
            multipliers=multipliers,
            penalty=penalty,
        )

    return alm.minimize_alm(
        np.array([3.0, 2.0]),
        ["x0_at_least_one"],
        evaluate_problem,
        alm.ALMSettings(),
        {"maxiter": 200},
    )


class AlmLibraryResultTests(unittest.TestCase):
    def test_minimize_alm_returns_the_lean_frozen_result(self):
        result = _solve_unit_halfspace_problem()

        self.assertIs(
            type(result),
            alm.ALMResult,
            f"the library returned {type(result).__name__}, not ALMResult",
        )
        self.assertEqual(
            tuple(field.name for field in dataclasses.fields(result)), LEAN_FIELDS
        )
        with self.assertRaises(dataclasses.FrozenInstanceError):
            result.success = False

    def test_lean_result_answers_a_plain_caller(self):
        result = _solve_unit_halfspace_problem()

        self.assertTrue(result.success, result.message)
        self.assertEqual(result.termination_reason, "converged")
        np.testing.assert_allclose(result.x, [1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(result.objective, 1.0, atol=1e-5)
        self.assertEqual(result.constraint_names, ("x0_at_least_one",))
        np.testing.assert_allclose(result.constraint_values, [0.0], atol=1e-5)
        self.assertLessEqual(result.max_violation, alm.ALMSettings().feasibility_tol)
        np.testing.assert_allclose(result.multipliers, [2.0], rtol=1e-4)
        self.assertLessEqual(result.kkt_stationarity_norm, 1e-5)
        self.assertGreater(result.nit, 0)
        self.assertGreater(result.outer_iterations, 0)
        self.assertFalse(result.restored_best_feasible)
        self.assertIsNone(result.restored_best_feasible_reason)

    def test_lean_result_is_read_only(self):
        result = _solve_unit_halfspace_problem()

        for name in ("x", "constraint_values", "multipliers"):
            with self.subTest(field=name):
                self.assertFalse(getattr(result, name).flags.writeable)
        self.assertIsInstance(result.evaluation, MappingProxyType)
        with self.assertRaises(TypeError):
            result.evaluation["total"] = 0.0

    def test_lean_result_evaluation_is_read_only_all_the_way_down(self):
        result = _solve_unit_halfspace_problem()
        before = golden.encoded_digest(result.evaluation)

        grad = result.evaluation["grad"]
        constraint_grads = result.evaluation["constraint_grads"]
        self.assertIsInstance(constraint_grads, tuple)
        writes = {
            "grad[0] = 1": lambda: grad.__setitem__(0, 1.0),
            "constraint_grads[0][0] = 1": lambda: constraint_grads[0].__setitem__(
                0, 1.0
            ),
            "constraint_grads.append": lambda: constraint_grads.append(grad),
            "constraint_grads[0] = grad": lambda: constraint_grads.__setitem__(
                0, grad
            ),
        }
        for label, write in writes.items():
            with self.subTest(write=label):
                with self.assertRaises(
                    (ValueError, TypeError, AttributeError),
                    msg=f"{label} mutated ALMResult.evaluation",
                ):
                    write()
        for key, value in result.evaluation.items():
            if isinstance(value, np.ndarray):
                with self.subTest(key=key):
                    self.assertFalse(value.flags.writeable, f"evaluation[{key!r}]")
        self.assertEqual(golden.encoded_digest(result.evaluation), before)


if __name__ == "__main__":
    unittest.main()
