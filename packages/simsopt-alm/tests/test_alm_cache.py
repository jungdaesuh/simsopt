"""Physics reuse at a revisited x: :class:`ALMPhysics` and :func:`cached_alm_evaluator`.

``minimize_alm`` evaluates the same x more than once: the outer iterate is
evaluated before the inner solve, and L-BFGS-B evaluates it again as its first
call; the outer loop evaluates it again after a dual or penalty update. Only
the augmented terms depend on the multipliers and penalty, so a deterministic
evaluator can reuse its physics (f, grad f, g, grad g) there. These tests pin
that reuse: one physics evaluation per x, and every output bit-for-bit what the
uncached evaluator returns.
"""

import dataclasses
import unittest
from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType

import numpy as np

import simsopt_alm as alm
from simsopt_alm import (
    ALMPhysics,
    ALMSettings,
    augmented_inequality_objective,
    cached_alm_evaluator,
    minimize_alm,
)


def _assert_bitwise_equal(test, got, want, path="value"):
    """Same types, dict key order, dtypes, shapes and bytes, recursively."""
    test.assertIs(type(got), type(want), f"{path}: type differs")
    if isinstance(want, np.ndarray):
        test.assertEqual(got.dtype, want.dtype, f"{path}: dtype differs")
        test.assertEqual(got.shape, want.shape, f"{path}: shape differs")
        test.assertEqual(got.tobytes(), want.tobytes(), f"{path}: bytes differ")
    elif isinstance(want, Mapping):
        test.assertEqual(list(got), list(want), f"{path}: keys or key order differ")
        for key in want:
            _assert_bitwise_equal(test, got[key], want[key], f"{path}[{key!r}]")
    elif isinstance(want, (list, tuple)):
        test.assertEqual(len(got), len(want), f"{path}: length differs")
        for index, (got_item, want_item) in enumerate(zip(got, want)):
            _assert_bitwise_equal(test, got_item, want_item, f"{path}[{index}]")
    elif isinstance(want, float):
        test.assertEqual(
            np.float64(got).tobytes(), np.float64(want).tobytes(), f"{path}: {got!r} != {want!r}"
        )
    elif dataclasses.is_dataclass(want):
        for field in dataclasses.fields(want):
            _assert_bitwise_equal(
                test, getattr(got, field.name), getattr(want, field.name), f"{path}.{field.name}"
            )
    elif hasattr(want, "__dict__"):
        _assert_bitwise_equal(test, vars(got), vars(want), f"{path}.__dict__")
    else:
        test.assertEqual(got, want, f"{path} differs")


class _TwoRowProblem:
    """min (x0 - 2)^2 + (x1 - 1.5)^2 + 0.3 x0 x1
    s.t. x0 + x1 - 2 <= 0 and x0^2 - 1.2 <= 0 (both active at the solution).

    ``physics_x`` lists the x of every physics evaluation.
    """

    def __init__(self):
        self.physics_x = []

    def physics(self, x):
        x = np.asarray(x, dtype=float)
        self.physics_x.append(x.tobytes())
        return ALMPhysics(
            base_value=float((x[0] - 2.0) ** 2 + (x[1] - 1.5) ** 2 + 0.3 * x[0] * x[1]),
            base_grad=np.array(
                [2.0 * (x[0] - 2.0) + 0.3 * x[1], 2.0 * (x[1] - 1.5) + 0.3 * x[0]]
            ),
            constraint_values=np.array([x[0] + x[1] - 2.0, x[0] ** 2 - 1.2]),
            constraint_grads=(np.array([1.0, 1.0]), np.array([2.0 * x[0], 0.0])),
        )

    def legacy_evaluate(self, x, multipliers, penalty):
        """The uncached evaluator every ALM example wrote before 5.9."""
        x = np.asarray(x, dtype=float)
        self.physics_x.append(x.tobytes())
        return augmented_inequality_objective(
            base_value=float((x[0] - 2.0) ** 2 + (x[1] - 1.5) ** 2 + 0.3 * x[0] * x[1]),
            base_grad=np.array(
                [2.0 * (x[0] - 2.0) + 0.3 * x[1], 2.0 * (x[1] - 1.5) + 0.3 * x[0]]
            ),
            constraint_values=np.array([x[0] + x[1] - 2.0, x[0] ** 2 - 1.2]),
            constraint_grads=[np.array([1.0, 1.0]), np.array([2.0 * x[0], 0.0])],
            multipliers=multipliers,
            penalty=penalty,
        )


def _run(evaluate_problem, outer_x_log=None):
    def outer_state_callback(outer_iteration, multipliers, penalty):
        if outer_x_log is not None:
            outer_x_log.append(outer_iteration)

    return minimize_alm(
        np.array([3.0, 2.5]),
        ["sum_at_most_two", "x0_squared_at_most"],
        evaluate_problem,
        ALMSettings(max_outer_iterations=6),
        {"maxiter": 200},
        outer_state_callback=outer_state_callback,
    )


class CachedEvaluatorTests(unittest.TestCase):
    def test_cached_evaluator_physics_once_across_lambda_rho(self):
        problem = _TwoRowProblem()
        evaluate_problem = cached_alm_evaluator(problem.physics)
        x = np.array([1.3, 0.9])
        pairs = [
            (np.zeros(2), 1.0),
            (np.array([0.4, 1.7]), 10.0),
            (np.array([2.0, 0.0]), np.array([3.0, 30.0])),
        ]

        evaluations = [evaluate_problem(x.copy(), m, p) for m, p in pairs]

        self.assertEqual(
            len(problem.physics_x),
            1,
            "physics must run once at one x, whatever the multipliers and penalty",
        )
        reference = _TwoRowProblem()
        for (multipliers, penalty), evaluation in zip(pairs, evaluations):
            with self.subTest(penalty=penalty):
                _assert_bitwise_equal(
                    self,
                    evaluation,
                    reference.legacy_evaluate(x, multipliers, penalty),
                )

        evaluate_problem(x + 1.0e-3, *pairs[0])
        evaluate_problem(x, *pairs[0])
        self.assertEqual(len(problem.physics_x), 3, "a new x must recompute the physics")
        evaluate_problem.cache_clear()
        evaluate_problem(x, *pairs[0])
        self.assertEqual(len(problem.physics_x), 4, "cache_clear must drop the reused physics")

    def test_minimize_alm_cached_matches_legacy_bitwise(self):
        legacy_problem = _TwoRowProblem()
        legacy_calls = []

        def legacy(x, multipliers, penalty):
            legacy_calls.append(np.asarray(x, dtype=float).tobytes())
            return legacy_problem.legacy_evaluate(x, multipliers, penalty)

        cached_problem = _TwoRowProblem()
        cached_calls = []
        cached_evaluate = cached_alm_evaluator(cached_problem.physics)

        def cached(x, multipliers, penalty):
            cached_calls.append(np.asarray(x, dtype=float).tobytes())
            return cached_evaluate(x, multipliers, penalty)

        legacy_result = _run(legacy)
        cached_result = _run(cached)

        self.assertGreaterEqual(legacy_result.outer_iterations, 2, "fixture needs outer updates")
        self.assertEqual(cached_calls, legacy_calls, "minimize_alm took another path")
        for name in ("x", "evaluation", "multipliers", "penalty"):
            with self.subTest(field=name):
                _assert_bitwise_equal(
                    self,
                    getattr(cached_result, name),
                    getattr(legacy_result, name),
                    name,
                )
        _assert_bitwise_equal(self, cached_result, legacy_result, "result")
        distinct_x = len(set(legacy_calls))
        self.assertLess(distinct_x, len(legacy_calls), "the fixture revisits no x")
        self.assertEqual(
            len(cached_problem.physics_x),
            distinct_x,
            f"{len(legacy_calls)} evaluations at {distinct_x} distinct x",
        )

    def test_physics_once_per_x_per_outer_iteration(self):
        problem = _TwoRowProblem()
        outer_iterations = []
        physics_by_outer = []

        def physics(x):
            physics_by_outer.append((len(outer_iterations), np.asarray(x, dtype=float).tobytes()))
            return problem.physics(x)

        evaluate_problem = cached_alm_evaluator(physics)
        result = _run(evaluate_problem, outer_iterations)

        self.assertGreaterEqual(result.outer_iterations, 2, "fixture needs outer updates")
        repeated = {key: count for key, count in Counter(physics_by_outer).items() if count > 1}
        self.assertEqual(
            repeated,
            {},
            "one outer iteration evaluated the physics at one x more than once",
        )

    def test_cached_evaluation_not_aliased(self):
        problem = _TwoRowProblem()
        evaluate_problem = cached_alm_evaluator(problem.physics)
        x = np.array([1.3, 0.9])
        multipliers = np.array([0.4, 1.7])
        first = evaluate_problem(x, multipliers, 10.0)
        expected = _TwoRowProblem().legacy_evaluate(x.copy(), multipliers, 10.0)

        # A caller that edits its result in place, and its x, changes nothing
        # the next call at the same x returns.
        x[0] = 99.0
        for key, value in first.items():
            if isinstance(value, np.ndarray):
                value[...] = -7.0
        first["constraint_grads"][0][...] = -7.0
        first["constraint_grads"].append(np.zeros(2))
        first["total"] = -7.0
        second = evaluate_problem(np.array([1.3, 0.9]), multipliers, 10.0)

        self.assertEqual(len(problem.physics_x), 1)
        _assert_bitwise_equal(self, second, expected)
        third = evaluate_problem(np.array([1.3, 0.9]), multipliers, 10.0)
        self.assertIsNot(third, second)
        self.assertIsNot(third["constraint_grads"], second["constraint_grads"])
        for key, value in third.items():
            if isinstance(value, np.ndarray):
                with self.subTest(key=key):
                    self.assertFalse(np.shares_memory(value, second[key]))
        for got, other in zip(third["constraint_grads"], second["constraint_grads"]):
            self.assertFalse(np.shares_memory(got, other))

    def test_cached_extras_not_aliased(self):
        nested = {"rows": [np.array([1.0, 2.0]), [3.0, 4.0]], "label": "a"}
        names = ["first", "second"]

        def physics(x):
            return ALMPhysics(
                base_value=1.0,
                base_grad=np.asarray(x, dtype=float),
                constraint_values=[0.1],
                constraint_grads=[np.ones(2)],
                extras={"nested": nested, "names": names},
            )

        evaluate_problem = cached_alm_evaluator(physics)
        x = np.array([0.5, -0.5])
        first = evaluate_problem(x, np.zeros(1), 1.0)
        # The physics owns its extras: editing the caller's inputs later changes nothing.
        nested["rows"][0][0] = -1.0
        names.append("third")

        # A caller that edits the returned extras in place, at every depth.
        first["nested"]["rows"][0][0] = 99.0
        first["nested"]["rows"][1].append(5.0)
        first["nested"]["label"] = "edited"
        first["nested"]["added"] = True
        first["names"].append("edited")
        second = evaluate_problem(x.copy(), np.zeros(1), 1.0)

        np.testing.assert_array_equal(second["nested"]["rows"][0], [1.0, 2.0])
        self.assertEqual(second["nested"]["rows"][1], [3.0, 4.0])
        self.assertEqual(second["nested"], {"rows": second["nested"]["rows"], "label": "a"})
        self.assertEqual(second["names"], ["first", "second"])
        self.assertIsNot(second["nested"], first["nested"])
        self.assertIsNot(second["nested"]["rows"], first["nested"]["rows"])
        self.assertIsNot(second["nested"]["rows"][1], first["nested"]["rows"][1])
        self.assertIsNot(second["names"], first["names"])
        self.assertFalse(
            np.shares_memory(second["nested"]["rows"][0], first["nested"]["rows"][0])
        )
        self.assertTrue(second["nested"]["rows"][0].flags.writeable)

    def test_extras_behind_any_mapping_are_owned(self):
        # A read-only mapping does not make its arrays read-only: the physics
        # must own what sits behind any Mapping, not only behind a dict.
        source_row = np.array([1.0, 2.0])
        source_names = ["first"]
        nested = MappingProxyType(
            {"row": source_row, "names": source_names, "deeper": MappingProxyType({"row": source_row})}
        )

        def physics(x):
            return ALMPhysics(
                base_value=1.0,
                base_grad=np.asarray(x, dtype=float),
                constraint_values=[0.1],
                constraint_grads=[np.ones(2)],
                extras={"nested": nested},
            )

        evaluate_problem = cached_alm_evaluator(physics)
        x = np.array([0.5, -0.5])
        first = evaluate_problem(x, np.zeros(1), 1.0)

        # Source -> physics: editing the caller's inputs changes nothing.
        source_row[0] = 7.0
        source_names.append("second")
        # Returned evaluation -> physics: editing what a call returned changes nothing.
        first["nested"]["row"][1] = 7.0
        first["nested"]["deeper"]["row"][1] = 7.0
        first["nested"]["names"].append("edited")
        second = evaluate_problem(x.copy(), np.zeros(1), 1.0)

        np.testing.assert_array_equal(second["nested"]["row"], [1.0, 2.0])
        np.testing.assert_array_equal(second["nested"]["deeper"]["row"], [1.0, 2.0])
        self.assertEqual(second["nested"]["names"], ["first"])
        self.assertIsInstance(second["nested"], dict)
        self.assertTrue(second["nested"]["row"].flags.writeable)
        self.assertFalse(np.shares_memory(second["nested"]["row"], source_row))

    def test_physics_stores_nested_extras_read_only(self):
        physics = ALMPhysics(
            base_value=1.0,
            base_grad=np.zeros(2),
            constraint_values=[0.1],
            constraint_grads=[np.ones(2)],
            extras={"nested": {"row": np.array([1.0, 2.0]), "names": ["first"]}},
        )

        stored = physics.extras["nested"]
        self.assertFalse(stored["row"].flags.writeable)
        with self.assertRaises(TypeError):
            stored["added"] = True
        with self.assertRaises(AttributeError):
            stored["names"].append("edited")
        # The evaluation still hands out the original container types.
        evaluation = physics.evaluation(np.zeros(1), 1.0)
        self.assertEqual(evaluation["nested"]["names"], ["first"])
        self.assertIsInstance(evaluation["nested"]["names"], list)


class ALMPhysicsTests(unittest.TestCase):
    def _physics(self, **extras):
        return ALMPhysics(
            base_value=1.5,
            base_grad=np.array([0.5, -1.0]),
            constraint_values=np.array([0.2, -0.3]),
            constraint_grads=(np.array([1.0, 0.0]), np.array([0.0, 1.0])),
            extras=extras,
        )

    def test_physics_extras_reject_lambda_rho_keys(self):
        for key in (
            "total",
            "grad",
            "positive_shift_values",
            "augmented_term_by_constraint",
            "stationarity_norm",
        ):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, key):
                    self._physics(**{key: 0.0})

    def test_physics_extras_reject_the_physics_fields(self):
        # f, grad f, g and grad g have one source, the constructor: an extra
        # replacing one after L and its gradient were built from the
        # constructor's values would publish a self-contradictory evaluation.
        for key in ("base_value", "base_grad", "constraint_values", "constraint_grads"):
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"{key}.*constructor"):
                    self._physics(**{key: 0.0})

    def test_physics_extras_may_overlay_routing_and_diagnostics(self):
        # What an evaluator legitimately routes differently from g (the
        # Stage-2 hybrid overlay) and any diagnostic key stay allowed.
        overlays = {
            "feasibility_values": np.array([0.1, 0.0]),
            "dual_update_values": np.array([0.2, -0.3]),
            "max_feasibility_violation": 0.1,
            "hard_signed_constraint_values": np.array([0.1, -0.3]),
            "hard_violation_values": np.array([0.1, 0.0]),
            "surrogate_signed_constraint_values": np.array([0.2, -0.3]),
            "hard_dual_update_values": np.array([0.1, -0.3]),
            "constraint_scales": [2.0, 3.0],
            "my_diagnostic": {"note": "kept"},
        }
        evaluation = self._physics(**overlays).evaluation(np.array([0.5, 0.0]), 4.0)
        for key, value in overlays.items():
            with self.subTest(key=key):
                self.assertEqual(repr(evaluation[key]), repr(value))

    def test_evaluation_overlays_x_only_extras(self):
        physics = self._physics(
            feasibility_values=np.array([0.1, 0.0]),
            constraint_scales=[2.0, 3.0],
        )
        evaluation = physics.evaluation(np.array([0.5, 0.0]), 4.0)

        reference = augmented_inequality_objective(
            1.5,
            np.array([0.5, -1.0]),
            np.array([0.2, -0.3]),
            [np.array([1.0, 0.0]), np.array([0.0, 1.0])],
            np.array([0.5, 0.0]),
            4.0,
        )
        reference.update(
            feasibility_values=np.array([0.1, 0.0]), constraint_scales=[2.0, 3.0]
        )
        _assert_bitwise_equal(self, evaluation, reference)

    def test_nonfinite_evaluation_extra_makes_total_nan(self):
        # As the Stage-2 evaluator does for sanitized inputs: the point is
        # unusable, so L is NaN whatever the finite physics says.
        evaluation = self._physics(
            nonfinite_evaluation=True, nonfinite_fields=["base_value"]
        ).evaluation(np.zeros(2), 1.0)

        self.assertTrue(np.isnan(evaluation["total"]))
        self.assertIs(evaluation["nonfinite_evaluation"], True)
        self.assertEqual(evaluation["nonfinite_fields"], ["base_value"])

    def test_nonfinite_physics_passes_through(self):
        # A failed solve reports NaN physics; minimize_alm, not ALMPhysics,
        # rejects the point.
        evaluation = ALMPhysics(
            base_value=np.nan,
            base_grad=np.full(2, np.nan),
            constraint_values=np.full(1, np.nan),
            constraint_grads=(np.full(2, np.nan),),
        ).evaluation(np.zeros(1), 1.0)

        self.assertTrue(np.isnan(evaluation["total"]))
        self.assertTrue(np.all(np.isnan(evaluation["grad"])))

    def test_physics_is_frozen_and_owns_read_only_arrays(self):
        base_grad = np.array([0.5, -1.0])
        physics = ALMPhysics(
            base_value=1.5,
            base_grad=base_grad,
            constraint_values=[0.2],
            constraint_grads=[np.array([1.0, 0.0])],
        )
        base_grad[0] = 99.0

        self.assertEqual(physics.base_grad[0], 0.5)
        self.assertFalse(physics.base_grad.flags.writeable)
        self.assertFalse(physics.constraint_grads[0].flags.writeable)
        self.assertIsInstance(physics.constraint_grads, tuple)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            physics.base_value = 0.0

    def test_public_names(self):
        self.assertIn("ALMPhysics", alm.__all__)
        self.assertIn("cached_alm_evaluator", alm.__all__)
        self.assertIn("cached_alm_evaluator", alm.__doc__)


if __name__ == "__main__":
    unittest.main()
