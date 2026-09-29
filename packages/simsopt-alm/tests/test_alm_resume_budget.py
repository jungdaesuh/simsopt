"""A resume from an outer boundary continues exactly as the uninterrupted run.

``inner_options["maxiter"]`` budgets one ``minimize_alm`` call's L-BFGS-B
iterations, so a caller resuming from a boundary passes the run's budget minus
the boundary's ``state.total_inner_iterations``, floored at 0 (the continuation
steps of one outer share the budget left at its start, so the spent count can
pass the budget). With that budget the resumed call must make the same
decisions, publish the same boundaries and return the same result as the
uninterrupted run from every non-terminal boundary, including one where the
budget ran out at the boundary (a remaining budget of 0). The only field
allowed to differ is ``inner_result``: a resumed process that runs no inner
solve has no L-BFGS-B result (``None``).
"""

import unittest

import numpy as np

from simsopt_alm import (
    ALMPhysics,
    ALMSettings,
    cached_alm_evaluator,
    minimize_alm,
)

CONSTRAINT_NAMES = ("sum_cap", "x0_floor")


def toy_physics(x) -> ALMPhysics:
    """min (x0-2)^2 + (x1-1)^2 s.t. x0 + x1 <= 2, x0 >= 0; x* = (1.5, 0.5)."""
    x = np.asarray(x, dtype=float)
    return ALMPhysics(
        base_value=(x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2,
        base_grad=np.array([2.0 * (x[0] - 2.0), 2.0 * (x[1] - 1.0)]),
        constraint_values=np.array([x[0] + x[1] - 2.0, -x[0]]),
        constraint_grads=(np.array([1.0, 1.0]), np.array([-1.0, 0.0])),
    )


SETTINGS = ALMSettings(
    max_outer_iterations=12,
    max_subproblem_continuations=2,
    penalty_init=10.0,
    penalty_scale=10.0,
    feasibility_tol=1.0e-6,
    stationarity_tol=1.0e-6,
)


def _step_record(event) -> tuple:
    return (
        event.outer_iteration,
        event.continuation_iteration,
        event.action,
        event.outer_termination,
        event.after.x.tobytes(),
        event.after.multipliers.tobytes(),
        float(event.after.penalty).hex(),
        int(event.after.total_inner_iterations),
    )


def _boundary_record(boundary) -> tuple:
    return (
        boundary.completed_outer_iterations,
        boundary.completed_action,
        boundary.termination_reason,
        np.asarray(boundary.state.x, dtype=float).tobytes(),
        np.asarray(boundary.state.multipliers, dtype=float).tobytes(),
        float(boundary.state.penalty).hex(),
        int(boundary.state.total_inner_iterations),
    )


def _result_record(result) -> dict:
    return {
        "x": result.x.tobytes(),
        "success": result.success,
        "termination_reason": result.termination_reason,
        "message": result.message,
        "objective": float(result.objective).hex(),
        "constraint_values": result.constraint_values.tobytes(),
        "max_violation": float(result.max_violation).hex(),
        "multipliers": result.multipliers.tobytes(),
        "penalty": float(result.penalty).hex(),
        "stationarity_norm": float(result.stationarity_norm).hex(),
        "nit": result.nit,
        "outer_iterations": result.outer_iterations,
        "restored_best_feasible": result.restored_best_feasible,
    }


def _run(inner_maxiter: int, resume_from=None):
    steps: list = []
    boundaries: list = []
    x0 = np.zeros(2) if resume_from is None else resume_from.state.x
    result = minimize_alm(
        np.array(x0, dtype=float),
        CONSTRAINT_NAMES,
        cached_alm_evaluator(toy_physics),
        SETTINGS,
        {"maxiter": int(inner_maxiter)},
        resume_from=resume_from,
        on_outer_step=lambda event: steps.append(_step_record(event)),
        on_outer_boundary=boundaries.append,
    )
    return result, steps, boundaries


class ResumeContinuesTheUninterruptedRunTests(unittest.TestCase):
    def assert_every_boundary_resumes_identically(self, run_maxiter: int) -> int:
        full_result, full_steps, full_boundaries = _run(run_maxiter)
        resumable = [
            boundary
            for boundary in full_boundaries
            if boundary.termination_reason is None
        ]
        self.assertTrue(resumable, "the run published no resumable boundary")
        exhausted_boundaries = 0
        for boundary in resumable:
            completed = boundary.completed_outer_iterations
            spent = int(boundary.state.total_inner_iterations)
            remaining = max(run_maxiter - spent, 0)
            exhausted_boundaries += remaining == 0
            with self.subTest(completed_outer=completed, remaining_maxiter=remaining):
                resumed_result, resumed_steps, resumed_boundaries = _run(
                    remaining, resume_from=boundary
                )
                self.assertEqual(
                    resumed_steps,
                    [step for step in full_steps if step[0] > completed],
                    "the resumed run made different outer decisions",
                )
                self.assertEqual(
                    [_boundary_record(b) for b in resumed_boundaries],
                    [
                        _boundary_record(b)
                        for b in full_boundaries
                        if b.completed_outer_iterations > completed
                        or b.termination_reason is not None
                    ],
                    "the resumed run published different boundaries",
                )
                self.assertEqual(
                    _result_record(resumed_result),
                    _result_record(full_result),
                    "the resumed run returned a different result",
                )
        return exhausted_boundaries

    def test_budget_spent_at_a_boundary(self):
        # Each budget runs out at a boundary (the uninterrupted run stops
        # before the next outer, and so must a resume with 0 remaining) after
        # a different decision: penalty increase (1-5), dual update (6, 8, 9)
        # and subproblem-limit penalty increase (7), some of them ending in a
        # best-feasible restore.
        for run_maxiter in range(1, 10):
            with self.subTest(run_maxiter=run_maxiter):
                self.assertGreaterEqual(
                    self.assert_every_boundary_resumes_identically(run_maxiter), 1
                )

    def test_budget_left_at_every_boundary(self):
        self.assertEqual(self.assert_every_boundary_resumes_identically(200), 0)


if __name__ == "__main__":
    unittest.main()
