"""What a continuation policy is shown: borrowed read-only views, one type per call.

``minimize_alm`` hands each ``ALMContinuationPolicy`` call a view of its own
measurements. The views are borrowed for the synchronous call and read-only
all the way down: arrays are non-writable views of the loop's arrays,
mappings read-only proxies, lists tuples, and the solver's carriers are
rebuilt from those. A policy that tries an ordinary write gets an error, and
the loop's own arrays (and the evaluator's) stay writable and unchanged.
``before_inner`` gets an ``ALMPreInnerView`` (no inner solve yet) and
``after_inner`` an ``ALMPostInnerView``, whose ``inner`` is required.
"""

import dataclasses
import sys
import typing
import unittest
from operator import setitem
from pathlib import Path

import numpy as np

import simsopt.solve.alm as alm
from simsopt.solve.alm import continuation as alm_continuation
from simsopt.solve.alm.policy import ALMContinuationPolicy, DefaultContinuationPolicy

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "test_files" / "alm_golden"
if str(GOLDEN_DIR) not in sys.path:
    sys.path.insert(0, str(GOLDEN_DIR))
import alm_golden_scenarios as golden  # noqa: E402


def _hybrid_run(policy, evaluations):
    """The hybrid band problem (all four hybrid signals, a live routing state)."""
    band = golden.hybrid_band_evaluate(surrogate_bias=0.05, activity_tolerance=0.02)

    def evaluate_problem(x, multipliers, penalty):
        evaluation = band(x, multipliers, penalty)
        evaluations.append(evaluation)
        return evaluation

    return alm.minimize_alm(
        np.array([1.2, 1.0]),
        list(golden.HYBRID_CONSTRAINTS),
        evaluate_problem,
        alm.ALMSettings(max_outer_iterations=4, max_subproblem_continuations=2),
        {"maxiter": 50},
        continuation_policy=policy,
    )


class _WritingPolicy:
    """The default decisions, after trying the writes a careless policy makes."""

    def __init__(self, test):
        self.test = test
        self.default = DefaultContinuationPolicy()
        self.checked = {"inner_plan": 0, "before_inner": 0, "after_inner": 0}

    def _blocked(self, write, error):
        with self.test.assertRaises(error):
            write()

    def _try_measurement_writes(self, measured):
        evaluation = measured.evaluation
        self._blocked(lambda: setitem(measured.multipliers, 0, 99.0), ValueError)
        self._blocked(lambda: setitem(measured.solver_constraint_values, 0, 99.0), ValueError)
        self._blocked(lambda: setitem(evaluation["grad"], 0, 99.0), ValueError)
        self._blocked(lambda: setitem(evaluation["constraint_grads"][0], 0, 99.0), ValueError)
        self._blocked(lambda: evaluation["constraint_grads"].append(None), AttributeError)
        self._blocked(lambda: setitem(evaluation, "added_by_policy", 1.0), TypeError)
        signals = measured.routing_state.signal_state
        self._blocked(lambda: setitem(signals.hard_signed_constraint_values, 0, 99.0), ValueError)
        self._blocked(
            lambda: setitem(measured.routing_state.surrogate_activity_mask, 0, True),
            ValueError,
        )

    def inner_plan(self, view):
        self.checked["inner_plan"] += 1
        self._blocked(lambda: setitem(view.inner_options, "maxiter", 1), TypeError)
        return self.default.inner_plan(view)

    def retry_stalled_trial(self, view):
        return self.default.retry_stalled_trial(view)

    def before_inner(self, view):
        self.checked["before_inner"] += 1
        self._try_measurement_writes(view.start)
        return self.default.before_inner(view)

    def after_inner(self, view):
        self.checked["after_inner"] += 1
        self._try_measurement_writes(view.start)
        self._try_measurement_writes(view.inner.measured)
        self._blocked(lambda: setitem(view.inner.x, 0, 99.0), ValueError)
        return self.default.after_inner(view)


class AlmBorrowedPolicyViewTests(unittest.TestCase):
    def test_policy_writes_are_blocked_and_change_nothing(self):
        written, reference = [], []
        policy = _WritingPolicy(self)
        result = _hybrid_run(policy, written)
        expected = _hybrid_run(DefaultContinuationPolicy(), reference)

        self.assertGreater(policy.checked["after_inner"], 1)
        self.assertGreater(policy.checked["inner_plan"], 1)
        self.assertEqual(result.x.tobytes(), expected.x.tobytes())
        self.assertEqual(result.multipliers.tobytes(), expected.multipliers.tobytes())
        self.assertEqual(result.termination_reason, expected.termination_reason)
        # The evaluator's arrays keep their flags and values: views carry the flags.
        self.assertEqual(len(written), len(reference))
        for got, want in zip(written, reference):
            for key, value in got.items():
                if isinstance(value, np.ndarray):
                    with self.subTest(key=key):
                        self.assertTrue(value.flags.writeable)
                        self.assertEqual(value.tobytes(), want[key].tobytes())


class AlmPolicyViewTypeTests(unittest.TestCase):
    def test_the_after_inner_view_requires_the_inner_solve(self):
        post_inner = alm_continuation.ALMPostInnerView
        pre_inner = alm_continuation.ALMPreInnerView
        self.assertIn("inner", [field.name for field in dataclasses.fields(post_inner)])
        self.assertNotIn("inner", [field.name for field in dataclasses.fields(pre_inner)])
        self.assertFalse(hasattr(alm_continuation, "ALMStepView"))
        with self.assertRaises(TypeError):
            post_inner(
                settings=alm.ALMSettings(), outer_iteration=1, continuation_iteration=0,
                start=None, last_cap_binding_active=False, feasible_stall_count=0,
                trust_radius=None,
            )

    def test_the_protocol_names_the_view_each_call_gets(self):
        before = typing.get_type_hints(ALMContinuationPolicy.before_inner)["view"]
        after = typing.get_type_hints(ALMContinuationPolicy.after_inner)["view"]
        self.assertIs(before, alm_continuation.ALMPreInnerView)
        self.assertIs(after, alm_continuation.ALMPostInnerView)

    def test_the_loop_hands_each_call_its_view_type(self):
        seen = {"before_inner": set(), "after_inner": set()}
        default = DefaultContinuationPolicy()

        class Recording:
            inner_plan = default.inner_plan
            retry_stalled_trial = default.retry_stalled_trial

            def before_inner(self, view):
                seen["before_inner"].add(type(view).__name__)
                return default.before_inner(view)

            def after_inner(self, view):
                seen["after_inner"].add(type(view).__name__)
                return default.after_inner(view)

        _hybrid_run(Recording(), [])
        self.assertEqual(
            seen, {"before_inner": {"ALMPreInnerView"}, "after_inner": {"ALMPostInnerView"}}
        )


if __name__ == "__main__":
    unittest.main()
