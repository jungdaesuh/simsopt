"""Where an evaluation enters the solver.

An evaluator may return the same array objects on every call (a cache that
fills one buffer in place, as simsopt's objectives do). The solver keeps
evaluations across later calls: the outer iterate's while the inner solve
tries other points, the accepted one until the next step. So it snapshots
every array it reads where an evaluation enters, and a run with such an
evaluator is bit for bit the run with one that returns new arrays.
"""

import unittest

import numpy as np

from simsopt_alm import ALMSettings, augmented_inequality_objective, minimize_alm

HYBRID_QUARTET = (
    "hard_signed_constraint_values",
    "hard_violation_values",
    "surrogate_signed_constraint_values",
    "hard_dual_update_values",
)


class _Evaluator:
    """min (x0 - 2)^2 + (x1 - 1)^2 s.t. x0 + x1 - 2 <= 0 and -x0 <= 0, with
    the hybrid quartet (both channels g) and every trial with x0 > 1.7
    rejected (``search_step_success=False``), so inner solves reject trials.

    With ``reuse`` every array, the ``constraint_grads`` list and its rows
    are the same objects on every call, refilled in place."""

    def __init__(self, reuse: bool):
        self.reuse = reuse
        self.buffers: dict = {}
        self.rows: list = []

    def _array(self, key, values) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if not self.reuse:
            return values.copy()
        buffer = self.buffers.setdefault(key, np.empty_like(values))
        buffer[...] = values
        return buffer

    def __call__(self, x, multipliers, penalty):
        x = np.asarray(x, dtype=float)
        g = np.array([x[0] + x[1] - 2.0, -x[0]])
        built = augmented_inequality_objective(
            float((x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2),
            np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
            g,
            [np.array([1.0, 1.0]), np.array([-1.0, 0.0])],
            multipliers,
            penalty,
        )
        evaluation = {
            key: self._array(key, value) if isinstance(value, np.ndarray) else value
            for key, value in built.items()
        }
        rows = [
            self._array(("row", index), row)
            for index, row in enumerate(built["constraint_grads"])
        ]
        if self.reuse:
            self.rows[:] = rows
            rows = self.rows
        evaluation["constraint_grads"] = rows
        channels = {
            "hard_signed_constraint_values": g,
            "hard_violation_values": np.maximum(g, 0.0),
            "surrogate_signed_constraint_values": g,
            "hard_dual_update_values": g,
        }
        for key in HYBRID_QUARTET:
            evaluation[key] = self._array(key, channels[key])
        evaluation["search_step_success"] = bool(x[0] <= 1.7)
        return evaluation


def _event_record(event) -> tuple:
    measurements = [event.start] + ([] if event.inner is None else [event.inner.measured])
    return (
        event.outer_iteration,
        event.continuation_iteration,
        event.action,
        event.after.x.tobytes(),
        event.after.multipliers.tobytes(),
        float(event.after.penalty).hex(),
        tuple(
            (
                np.asarray(measured.evaluation["grad"]).tobytes(),
                np.asarray(measured.evaluation["constraint_values"]).tobytes(),
                np.asarray(measured.evaluation["feasibility_values"]).tobytes(),
                np.asarray(measured.evaluation["dual_update_values"]).tobytes(),
                tuple(np.asarray(measured.evaluation[key]).tobytes() for key in HYBRID_QUARTET),
                tuple(np.asarray(row).tobytes() for row in measured.evaluation["constraint_grads"]),
                float(measured.stationarity_norm).hex(),
            )
            for measured in measurements
        ),
    )


def _run(reuse: bool):
    events: list = []
    result = minimize_alm(
        np.zeros(2),
        ["sum_cap", "x0_floor"],
        _Evaluator(reuse),
        ALMSettings(max_outer_iterations=8, max_subproblem_continuations=2),
        {"maxiter": 60},
        on_outer_step=lambda event: events.append(_event_record(event)),
    )
    return result, events


class ReusedEvaluatorBuffersTests(unittest.TestCase):
    def test_a_rejected_trial_cannot_change_the_kept_gradient(self):
        # Astra's reproduction: every trial is rejected, so the run cannot
        # leave x = 0, where the gradient is 2(x - 1) = -2. A rejected
        # trial refilled the buffer the outer iterate's gradient still
        # aliased, and the run certified x = 0 with stationarity 0.
        buffer = np.zeros(1)

        def evaluate(x, multipliers, penalty):
            buffer[:] = 2.0 * (x - 1.0)
            return {
                "total": float((x[0] - 1.0) ** 2),
                "grad": buffer,
                "constraint_values": np.array([-1.0]),
                "feasibility_values": np.zeros(1),
                "dual_update_values": np.array([-1.0]),
                "constraint_grads": [np.zeros(1)],
                "search_step_success": bool(x[0] == 0.0),
            }

        result = minimize_alm(np.zeros(1), ["g"], evaluate, ALMSettings(), {"maxiter": 10})
        self.assertEqual(result.x.tolist(), [0.0])
        self.assertFalse(result.success, result.termination_reason)
        self.assertEqual(result.stationarity_norm, 2.0)
        self.assertEqual(np.asarray(result.evaluation["grad"]).tolist(), [-2.0])

    def test_a_run_with_reused_buffers_is_the_run_with_new_arrays(self):
        fresh_result, fresh_events = _run(reuse=False)
        reused_result, reused_events = _run(reuse=True)
        self.assertTrue(fresh_result.success, fresh_result.termination_reason)
        self.assertEqual(reused_events, fresh_events)
        for name in ("x", "constraint_values", "multipliers"):
            with self.subTest(field=name):
                self.assertEqual(
                    getattr(reused_result, name).tobytes(),
                    getattr(fresh_result, name).tobytes(),
                )
        self.assertEqual(
            (reused_result.termination_reason, reused_result.nit,
             reused_result.outer_iterations, float(reused_result.stationarity_norm).hex()),
            (fresh_result.termination_reason, fresh_result.nit,
             fresh_result.outer_iterations, float(fresh_result.stationarity_norm).hex()),
        )


if __name__ == "__main__":
    unittest.main()


class FlaggedEvaluationTests(unittest.TestCase):
    """``nonfinite_evaluation=True`` marks a point unusable even when every
    value is finite. A flagged trial point is rejected like a non-finite one;
    a flagged outer evaluation raises ``ValueError`` like a non-finite one,
    so it is never certified and never becomes the incumbent."""

    def test_a_flagged_start_raises(self):
        # Astra's reproduction: finite data flagged unusable at x0 came back
        # ``converged``.
        def evaluate(x, multipliers, penalty):
            return dict(
                augmented_inequality_objective(
                    float(x @ x), 2.0 * x, [-1.0], [np.zeros(1)], multipliers, penalty
                ),
                nonfinite_evaluation=True,
            )

        with self.assertRaisesRegex(
            ValueError,
            "ALM outer iterate evaluation produced non-finite ALM data: "
            "nonfinite_evaluation",
        ):
            minimize_alm(np.zeros(1), ["g"], evaluate, ALMSettings(), {"maxiter": 10})

    def test_a_flagged_outer_reevaluation_raises(self):
        # The start is usable; every evaluation after the first dual update
        # (nonzero multipliers) is flagged, so the next outer evaluation of
        # the accepted iterate is.
        def evaluate(x, multipliers, penalty):
            built = augmented_inequality_objective(
                float((x[0] - 2.0) ** 2), np.array([2.0 * (x[0] - 2.0)]),
                [x[0] - 1.0], [np.ones(1)], multipliers, penalty,
            )
            return dict(built, nonfinite_evaluation=bool(np.any(multipliers > 0.0)))

        with self.assertRaisesRegex(ValueError, "non-finite ALM data: nonfinite_evaluation"):
            minimize_alm(np.zeros(1), ["x_at_most_one"], evaluate, ALMSettings(), {"maxiter": 50})

    def test_a_flagged_trial_is_rejected_and_the_run_goes_on(self):
        # min 10 (x - 0.5)^2 with an inactive row from x = 0: L-BFGS-B's
        # first trial, a unit step to x = 1, is flagged (x > 0.8).
        flagged_trials = []

        def evaluate(x, multipliers, penalty):
            built = augmented_inequality_objective(
                float(10.0 * (x[0] - 0.5) ** 2), np.array([20.0 * (x[0] - 0.5)]),
                [-1.0], [np.zeros(1)], multipliers, penalty,
            )
            flagged = bool(x[0] > 0.8)
            flagged_trials.append(flagged)
            return dict(built, nonfinite_evaluation=flagged)

        result = minimize_alm(np.zeros(1), ["g"], evaluate, ALMSettings(), {"maxiter": 50})
        self.assertTrue(any(flagged_trials), "no trial point was flagged")
        self.assertTrue(result.success, result.termination_reason)
        self.assertLessEqual(float(result.x[0]), 0.8)
        self.assertFalse(result.evaluation.get("nonfinite_evaluation", False))
