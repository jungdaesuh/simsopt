"""An inner solve stopped by the KKT gate reports the iterations it ran.

The inner callback ends an L-BFGS-B solve early once the dual-update gate
holds. SciPy calls the callback once per completed L-BFGS-B iteration, so the
callbacks a caller sees are the optimizer's iterations: ``nit``, each step's
``inner.iterations``, every boundary's ``state.total_inner_iterations`` and the
``inner_options["maxiter"]`` budget of one ``minimize_alm`` call must all count
them, whether an inner solve ended on its own or at the gate.

The problem is SciPy's Rosenbrock with one constraint that is never active, so
every inner solve ends at the gate after several iterations. Continuations are
off in the count tests; with them on, the continuation steps of one outer
iteration must also keep within the call's budget.
"""

import unittest

import numpy as np
from scipy.optimize import rosen, rosen_der

from simsopt_alm import ALMPhysics, ALMSettings, cached_alm_evaluator, minimize_alm

X0 = np.array([-1.2, 1.0])
SETTINGS = ALMSettings(max_outer_iterations=10, max_subproblem_continuations=0)


def inactive_rosenbrock(x) -> ALMPhysics:
    """Rosenbrock with the constraint -1 <= 0 (never active)."""
    x = np.asarray(x, dtype=float)
    return ALMPhysics(
        base_value=float(rosen(x)),
        base_grad=rosen_der(x),
        constraint_values=np.array([-1.0]),
        constraint_grads=(np.zeros_like(x),),
    )


class _CountedRun:
    """One minimize_alm call with its callback count at every step and boundary."""

    def __init__(self, inner_maxiter: int, resume_from=None, settings=SETTINGS):
        self.callbacks = 0
        self.step_iterations: list = []
        self.callbacks_at_boundary: list = []
        self.boundaries: list = []
        x0 = X0 if resume_from is None else resume_from.state.x
        callbacks_before_step = [0]

        def on_inner(_x):
            self.callbacks += 1

        def on_step(event):
            self.step_iterations.append(
                (
                    None if event.inner is None else int(event.inner.iterations),
                    self.callbacks - callbacks_before_step[0],
                )
            )
            callbacks_before_step[0] = self.callbacks

        def on_boundary(boundary):
            self.boundaries.append(boundary)
            self.callbacks_at_boundary.append(self.callbacks)

        self.result = minimize_alm(
            np.array(x0, dtype=float),
            ["inactive"],
            cached_alm_evaluator(inactive_rosenbrock),
            settings,
            {"maxiter": int(inner_maxiter)},
            resume_from=resume_from,
            inner_callback=on_inner,
            on_outer_step=on_step,
            on_outer_boundary=on_boundary,
        )


class GateStoppedInnerSolveCountTests(unittest.TestCase):
    def test_gate_stops_after_several_iterations(self):
        # Guards the premise: an early stop that ran one iteration would not
        # tell a true count from the old fixed nit=1.
        run = _CountedRun(200)
        self.assertGreater(
            max(callbacks for _reported, callbacks in run.step_iterations),
            1,
            "no inner solve ran more than one iteration before the gate stopped it",
        )

    def test_nit_equals_optimizer_callbacks(self):
        for budget in (38, 40, 50, 200):
            with self.subTest(maxiter=budget):
                run = _CountedRun(budget)
                self.assertEqual(
                    run.result.nit,
                    run.callbacks,
                    "result.nit does not count the L-BFGS-B iterations the "
                    "callback saw",
                )

    def test_step_iterations_equal_callbacks_of_that_step(self):
        run = _CountedRun(200)
        for index, (reported, callbacks) in enumerate(run.step_iterations):
            with self.subTest(step=index):
                self.assertEqual(
                    0 if reported is None else reported,
                    callbacks,
                    "a step's inner.iterations differs from the iterations its "
                    "inner solve ran",
                )

    def test_checkpoint_totals_equal_callbacks_so_far(self):
        run = _CountedRun(200)
        self.assertEqual(
            [int(b.state.total_inner_iterations) for b in run.boundaries],
            run.callbacks_at_boundary,
            "a boundary's total_inner_iterations differs from the iterations run "
            "before it, so a resume would budget from a wrong count",
        )

    def test_call_budget_bounds_the_iterations_run(self):
        for budget in (5, 10, 20, 38):
            with self.subTest(maxiter=budget):
                run = _CountedRun(budget)
                self.assertLessEqual(
                    run.callbacks,
                    budget,
                    "the call ran more L-BFGS-B iterations than its maxiter budget",
                )
                self.assertEqual(run.result.nit, run.callbacks)

    def test_resume_budget_continues_the_uninterrupted_count(self):
        budget = 38
        full = _CountedRun(budget)
        resumable = [
            (boundary, callbacks_before)
            for boundary, callbacks_before in zip(
                full.boundaries, full.callbacks_at_boundary
            )
            if boundary.termination_reason is None
        ]
        self.assertTrue(resumable, "the run published no resumable boundary")
        for boundary, callbacks_before in resumable:
            spent = int(boundary.state.total_inner_iterations)
            with self.subTest(completed_outer=boundary.completed_outer_iterations):
                resumed = _CountedRun(max(budget - spent, 0), resume_from=boundary)
                self.assertEqual(spent, callbacks_before)
                self.assertEqual(
                    resumed.callbacks,
                    full.callbacks - callbacks_before,
                    "the resumed call ran a different number of iterations than "
                    "the rest of the uninterrupted run",
                )
                self.assertEqual(resumed.result.nit, full.result.nit)
                self.assertEqual(resumed.result.x.tobytes(), full.result.x.tobytes())


class ContinuationBudgetTests(unittest.TestCase):
    def test_continuation_steps_keep_within_the_call_budget(self):
        # Grok's reproduction: maxiter=8 with 5 continuations ran 41
        # iterations, every continuation step getting the 8 left at the
        # outer's start.
        settings = ALMSettings(
            max_outer_iterations=3, max_subproblem_continuations=5, stationarity_tol=1e-12
        )
        unbounded = _CountedRun(10_000, settings=settings)
        continuations = [reported for reported, _ in unbounded.step_iterations[1:]]
        self.assertGreater(len(continuations), 0, "the fixture needs continuation steps")
        for budget in (1, 8, 20, 35):
            with self.subTest(maxiter=budget):
                self.assertLess(budget, unbounded.callbacks)
                run = _CountedRun(budget, settings=settings)
                self.assertEqual(run.result.nit, run.callbacks)
                self.assertLessEqual(run.callbacks, budget)


if __name__ == "__main__":
    unittest.main()
