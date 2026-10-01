"""Nested ALM data must be acyclic; shared subtrees are fine.

The solver copies the mappings, lists and tuples in an evaluation (events,
the result, ``ALMPhysics`` extras, checkpoints), so a container that contains
itself has no finite copy. Each place such data enters rejects a cycle with a
``ValueError`` naming the path that closes it: the loop when it keeps an
evaluation (the outer iterate before any inner solve, an accepted inner
candidate), ``ALMPhysics`` at construction, the checkpoint encoder, and
``minimize_alm(resume_from=...)``. A shared, acyclic subtree stays accepted.
"""

import unittest

import numpy as np

import simsopt_alm as alm
from simsopt_alm.boundary import ALMOuterBoundary
from simsopt_alm.checkpoint import _transition_json_value, transition_snapshot

X0 = np.array([3.0, 2.0])


def _self_mapping():
    metadata = {"label": "m"}
    metadata["self"] = metadata
    return {"metadata": metadata}, "evaluation['metadata']['self'] is evaluation['metadata']"


def _self_list():
    rows = [1.0]
    rows.append(rows)
    return {"rows": rows}, "evaluation['rows'][1] is evaluation['rows']"


def _tuple_list_cycle():
    rows = []
    wrapped = (rows,)
    rows.append(wrapped)
    return (
        {"mixed": wrapped},
        "evaluation['mixed'][0][0] is evaluation['mixed']",
    )


CYCLES = {"self_mapping": _self_mapping, "self_list": _self_list, "tuple_list": _tuple_list_cycle}


def _halfspace_evaluator(extra_keys_at, calls):
    """min ||x||^2 s.t. 1 - x0 <= 0, with ``extra_keys_at(x)`` merged in."""

    def evaluate_problem(x, multipliers, penalty):
        calls.append(np.asarray(x, dtype=float).copy())
        evaluation = alm.augmented_inequality_objective(
            base_value=float(x @ x),
            base_grad=2.0 * x,
            constraint_values=np.array([1.0 - x[0]]),
            constraint_grads=[np.array([-1.0, 0.0])],
            multipliers=multipliers,
            penalty=penalty,
        )
        evaluation.update(extra_keys_at(x))
        return evaluation

    return evaluate_problem


def _solve(evaluate_problem, **kwargs):
    return alm.minimize_alm(
        X0.copy(), ["x0_at_least_one"], evaluate_problem, alm.ALMSettings(),
        {"maxiter": 200}, **kwargs,
    )


class AlmEvaluationCycleTests(unittest.TestCase):
    def test_cyclic_evaluation_is_rejected_at_the_first_evaluation(self):
        for name, build in CYCLES.items():
            for observed in (False, True):
                with self.subTest(cycle=name, observed=observed):
                    extra, path_message = build()
                    calls = []
                    events = []
                    with self.assertRaises(ValueError) as caught:
                        _solve(
                            _halfspace_evaluator(lambda x, extra=extra: extra, calls),
                            on_outer_step=events.append if observed else None,
                        )
                    self.assertIn("ALM outer iterate evaluation is cyclic", str(caught.exception))
                    self.assertIn(path_message, str(caught.exception))
                    self.assertEqual(len(calls), 1, "rejected only after the solve ran")
                    self.assertEqual(events, [])

    def test_cyclic_accepted_inner_candidate_is_rejected(self):
        # The start iterate is acyclic; every other x carries a cycle, so the
        # first accepted inner candidate brings it into the loop.
        extra, path_message = _self_mapping()

        def extra_keys_at(x):
            return {} if np.array_equal(x, X0) else extra

        with self.assertRaises(ValueError) as caught:
            _solve(_halfspace_evaluator(extra_keys_at, []))
        self.assertIn("ALM accepted inner evaluation is cyclic", str(caught.exception))
        self.assertIn(path_message, str(caught.exception))

    def test_shared_subtree_is_accepted_and_stays_shared(self):
        shared = {"row": np.array([1.0, 2.0])}
        extra = {"metadata": {"a": shared, "b": shared, "rows": [shared, shared]}}
        for observed in (False, True):
            with self.subTest(observed=observed):
                events = []
                result = _solve(
                    _halfspace_evaluator(lambda x: extra, []),
                    on_outer_step=events.append if observed else None,
                )
                self.assertTrue(result.success, result.message)
                metadata = result.evaluation["metadata"]
                self.assertIs(metadata["a"], metadata["b"])
                np.testing.assert_array_equal(metadata["rows"][1]["row"], [1.0, 2.0])
                self.assertEqual(bool(events), observed)


class AlmPhysicsCycleTests(unittest.TestCase):
    def test_cyclic_extras_are_rejected_at_construction(self):
        nested = {"label": "n"}
        nested["self"] = nested
        with self.assertRaises(ValueError) as caught:
            alm.ALMPhysics(
                base_value=0.0, base_grad=np.zeros(2), constraint_values=[0.0],
                constraint_grads=[np.zeros(2)], extras={"nested": nested},
            )
        self.assertIn("ALMPhysics extras is cyclic", str(caught.exception))
        self.assertIn("extras['nested']['self'] is extras['nested']", str(caught.exception))

    def test_shared_extras_are_accepted(self):
        shared = [np.array([1.0])]
        physics = alm.ALMPhysics(
            base_value=0.0, base_grad=np.zeros(2), constraint_values=[0.0],
            constraint_grads=[np.zeros(2)], extras={"a": shared, "b": {"again": shared}},
        )
        evaluation = physics.evaluation(np.zeros(1), 1.0)
        np.testing.assert_array_equal(evaluation["b"]["again"][0], [1.0])


def _boundary_with_cyclic_incumbent():
    """A resumable boundary of a real run whose best-feasible incumbent
    carries a cyclic mapping (the loop itself never builds one)."""
    boundaries = []
    _solve(
        _halfspace_evaluator(lambda x: {}, []),
        on_outer_boundary=boundaries.append,
    )
    start = boundaries[0]
    extra, path_message = _self_mapping()
    incumbent = alm.ALMFeasibleIncumbent(
        x=np.array([1.0, 0.0]), evaluation=extra, multipliers=np.array([2.0]),
        penalty=1.0, inner_result=None,
    )
    state = alm.ALMLoopState(**{**vars(start.state), "best_feasible": incumbent})
    boundary = ALMOuterBoundary(
        completed_outer_iterations=0, completed_action="dual_update",
        termination_reason=None, constraint_names=start.constraint_names,
        constraint_blocks=None, accepted_state=None, geometry_identity=None,
        state=state,
    )
    return boundary, path_message


class AlmCheckpointCycleTests(unittest.TestCase):
    def test_cyclic_snapshot_evaluation_is_rejected_when_encoded(self):
        boundary, path_message = _boundary_with_cyclic_incumbent()
        with self.assertRaises(ValueError) as caught:
            transition_snapshot(boundary, {"maxiter": 200})
        self.assertIn("ALM transition value is cyclic", str(caught.exception))
        self.assertIn(path_message, str(caught.exception))

    def test_encoder_rejects_a_cycle_and_accepts_a_shared_subtree(self):
        rows = []
        rows.append(rows)
        with self.assertRaises(ValueError) as caught:
            _transition_json_value(rows)
        self.assertIn("value[0] is value", str(caught.exception))
        shared = [1.0]
        self.assertEqual(
            _transition_json_value({"a": shared, "b": shared}),
            (
                "__alm_mapping__",
                (
                    ("a", ("__alm_sequence__", (1.0,))),
                    ("b", ("__alm_sequence__", (1.0,))),
                ),
            ),
        )


class AlmResumeCycleTests(unittest.TestCase):
    def test_resume_boundary_with_a_cyclic_incumbent_is_rejected_before_evaluating(self):
        resumable, _path_message = _boundary_with_cyclic_incumbent()
        state = resumable.state
        calls = []
        with self.assertRaises(ValueError) as caught:
            alm.minimize_alm(
                np.asarray(state.x, dtype=float), ["x0_at_least_one"],
                _halfspace_evaluator(lambda x: {}, calls), alm.ALMSettings(),
                {"maxiter": 200}, resume_from=resumable,
            )
        self.assertIn("ALM resume boundary best_feasible evaluation is cyclic", str(caught.exception))
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
