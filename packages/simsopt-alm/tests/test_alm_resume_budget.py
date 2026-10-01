"""A resume from an outer boundary continues exactly as the uninterrupted run.

``inner_options["maxiter"]`` budgets one ``minimize_alm`` call's L-BFGS-B
iterations: every step's inner solve gets what the call has left, so the
call never runs more, and a spent budget ends the run before its next step,
with the latest step's action as the termination reason. A budget spent
inside an outer iteration ends the run there, and that outer publishes only
the terminal boundary. A caller resuming from a boundary passes the run's
budget minus the boundary's ``state.total_inner_iterations``. With that
budget the resumed call must make the same decisions, publish the same
boundaries and return the same result as the uninterrupted run from every
non-terminal boundary, including one where the budget ran out at the
boundary (a remaining budget of 0). The only field allowed to differ is
``inner_result``: a resumed process that runs no inner solve has no L-BFGS-B
result (``None``).
"""

import dataclasses
import unittest
from collections.abc import Mapping

import numpy as np

from simsopt_alm import (
    ALMPhysics,
    ALMResult,
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


# Every public ALMResult field but the one the module docstring lets differ.
RESUME_DIFFERING_RESULT_FIELDS = frozenset(("inner_result",))
COMPARED_RESULT_FIELDS = tuple(
    field.name
    for field in dataclasses.fields(ALMResult)
    if field.name not in RESUME_DIFFERING_RESULT_FIELDS
)


def _bits(value) -> object:
    """A bit-exact, comparable record of a result value: arrays by dtype,
    shape and bytes, floats by hex, mappings, sequences and dataclasses item
    by item."""
    if isinstance(value, np.ndarray):
        return ("ndarray", value.dtype.str, value.shape, value.tobytes())
    if isinstance(value, (float, np.floating)):
        return ("float", float(value).hex())
    if isinstance(value, Mapping):
        return ("mapping", tuple((key, _bits(item)) for key, item in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(_bits(item) for item in value))
    if dataclasses.is_dataclass(value):
        return (type(value).__name__, tuple(
            (field.name, _bits(getattr(value, field.name))) for field in dataclasses.fields(value)
        ))
    return (type(value).__name__, value)


def _result_record(result) -> dict:
    return {name: _bits(getattr(result, name)) for name in COMPARED_RESULT_FIELDS}


# An infeasible start (f's unconstrained minimizer, x0 + x1 <= 2 violated): a
# spent budget then ends on the latest step's action, where a feasible start
# would be restored as the best hard-feasible iterate.
X0 = np.array([2.0, 1.0])


def _run(inner_maxiter: int, resume_from=None):
    steps: list = []
    boundaries: list = []
    x0 = X0 if resume_from is None else resume_from.state.x
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
        exhausted_boundaries = 0
        for boundary in resumable:
            completed = boundary.completed_outer_iterations
            spent = int(boundary.state.total_inner_iterations)
            self.assertLessEqual(spent, run_maxiter)
            remaining = run_maxiter - spent
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
                if resumed_result.inner_result is not None:
                    # A resumed process that ran an inner solve returns its
                    # L-BFGS-B result, the uninterrupted run's last one.
                    self.assertEqual(
                        (resumed_result.inner_result.nit, _bits(resumed_result.inner_result.x)),
                        (full_result.inner_result.nit, _bits(full_result.inner_result.x)),
                    )
        return exhausted_boundaries

    def test_budget_spent_at_a_boundary_or_inside_an_outer(self):
        # Budgets 4-10 run out at the end of an outer iteration (after a
        # penalty hold or a dual update), so the uninterrupted run stops
        # before the next outer, and so must a resume with 0 remaining; 1-3
        # run out after a subproblem continuation, inside an outer, which
        # then publishes only the terminal boundary. The run converges after
        # 11 iterations.
        spent_where = {}
        for run_maxiter in range(1, 11):
            with self.subTest(run_maxiter=run_maxiter):
                result, steps, boundaries = _run(run_maxiter)
                self.assertEqual(result.nit, run_maxiter)
                self.assertEqual(result.termination_reason, steps[-1][2])
                exhausted = self.assert_every_boundary_resumes_identically(run_maxiter)
                spent_where[run_maxiter] = (
                    "boundary" if exhausted else "inside_outer",
                    steps[-1][2],
                )
        self.assertEqual(
            spent_where,
            {
                **{budget: ("inside_outer", "subproblem_continue") for budget in (1, 2, 3)},
                **{budget: ("boundary", "sufficient_decrease_hold") for budget in (4, 5)},
                **{budget: ("boundary", "dual_update") for budget in range(6, 11)},
            },
        )

    def test_budget_left_at_every_boundary(self):
        self.assertEqual(self.assert_every_boundary_resumes_identically(200), 0)


class OuterCountTests(unittest.TestCase):
    def test_an_outer_counts_once_it_runs_a_step(self):
        # Astra's reproduction: with maxiter=1 only outer 1 ran a step, yet
        # the result and the terminal boundary counted 2 completed outers.
        for run_maxiter in (1, 2, 5, 9, 200):
            with self.subTest(run_maxiter=run_maxiter):
                result, steps, boundaries = _run(run_maxiter)
                ran = max(step[0] for step in steps)
                self.assertEqual(result.outer_iterations, ran)
                self.assertEqual(boundaries[-1].completed_outer_iterations, ran)
                self.assertEqual(
                    [b.completed_outer_iterations for b in boundaries[:-1]],
                    sorted({step[0] for step in steps})[: len(boundaries) - 1],
                )


if __name__ == "__main__":
    unittest.main()
