#!/usr/bin/env python
r"""
Compose the ALM solver with its opt-in history and checkpoint plugins and with
a stateful evaluator, then resume a run from a checkpoint.

The problem is

.. math::

    \min_x\ (x_0 - 2)^2 + (x_1 - 1)^2 + y(x)^2
    \quad \text{s.t.} \quad x_0 + x_1 - 2 \le 0,

where :math:`y(x)` solves :math:`y^3 - y = 0.1 \sin x_0`. That equation has
three solutions, near -1, 0 and 1, and the evaluator's Newton solve finds the
one nearest its warm start, its last solution. The warm start is therefore
state the evaluator owns and the physics depends on, as a Boozer surface's
does. The run is seeded on the branch near 1. The solver takes a snapshot of
the state through ``snapshot_accepted_state_fn`` and hands one back through
``restore_incumbent_state_fn`` when it restores a best-feasible iterate or
resumes a run.

The script

1. solves the problem once, with ``ALMHistoryRecorder`` recording one entry
   per outer-step decision and ``alm_checkpointing`` taking a checkpoint after
   every outer iteration;
2. resumes from the checkpoint of outer iteration 2 with a fresh, cold
   evaluator, as a new process would: the solver restores the evaluator's warm
   start from the checkpoint before it evaluates anything (a cold evaluator
   would find the branch near 0 instead);
3. checks that the resumed run ends bit for bit where the uninterrupted one did.

Checkpoints stay in memory here. An ``ALMTransitionSnapshot`` is immutable,
finite data (no constraint Jacobians) that a driver can serialize.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from simsopt.solve.alm import ALMSettings, augmented_inequality_objective, minimize_alm
from simsopt.solve.alm.checkpoint import alm_checkpointing
from simsopt.solve.alm.history import ALMHistoryRecorder

CONSTRAINT_NAMES = ["sum_at_most_two"]
SETTINGS = ALMSettings(max_outer_iterations=12, history_max_entries=64)
# ``maxiter`` is this process's L-BFGS-B budget over all outer iterations; it
# is generous here, so both runs take the same inner steps.
INNER_OPTIONS = {"maxiter": 2000}
X0 = np.array([3.0, 2.5])
SEEDED_BRANCH = 1.0
RESUME_AFTER_OUTER = 2


@dataclass(frozen=True)
class WarmStart:
    """The evaluator's state: the solution its next Newton solve starts from."""

    y: float


class WarmStartedEvaluator:
    """``evaluate_problem(x, multipliers, penalty)`` with a warm-started solve.

    ``snapshot`` and ``restore`` are the two callbacks the solver needs for a
    stateful evaluator; ``log`` records the order of restores and evaluations.
    A new evaluator starts cold, at ``y = 0``.
    """

    def __init__(self, y: float = 0.0):
        self.y = float(y)
        self.log = []

    def solve_y(self, x0: float) -> float:
        """The solution of ``y**3 - y = 0.1 sin(x0)`` nearest the warm start."""
        y = self.y
        for _ in range(50):
            residual = y**3 - y - 0.1 * np.sin(x0)
            if abs(residual) <= 1e-14:
                break
            y = y - residual / (3.0 * y**2 - 1.0)
        self.y = float(y)
        return self.y

    def __call__(self, x, multipliers, penalty):
        x = np.asarray(x, dtype=float)
        self.log.append(("evaluate", self.y))
        y = self.solve_y(x[0])
        dy_dx0 = 0.1 * np.cos(x[0]) / (3.0 * y**2 - 1.0)
        return augmented_inequality_objective(
            base_value=(x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2 + y**2,
            base_grad=np.array([2.0 * (x[0] - 2.0) + 2.0 * y * dy_dx0, 2.0 * (x[1] - 1.0)]),
            constraint_values=np.array([x[0] + x[1] - 2.0]),
            constraint_grads=[np.array([1.0, 1.0])],
            multipliers=multipliers,
            penalty=penalty,
        )

    def snapshot(self) -> WarmStart:
        return WarmStart(y=self.y)

    def restore(self, state: WarmStart) -> None:
        self.log.append(("restore", state.y))
        self.y = state.y


def solve(evaluator, recorder, checkpointing, x0):
    """One ``minimize_alm`` call with the history recorder and the checkpoint
    plugin attached (``checkpointing`` holds the resume boundary, if any)."""
    return minimize_alm(
        x0,
        CONSTRAINT_NAMES,
        evaluator,
        SETTINGS,
        checkpointing.inner_options,
        snapshot_accepted_state_fn=evaluator.snapshot,
        restore_incumbent_state_fn=evaluator.restore,
        resume_from=checkpointing.resume_from,
        on_outer_step=recorder.record,
        on_outer_boundary=checkpointing.on_outer_boundary,
    )


def print_history(title, history):
    print(title)
    for entry in history:
        print(
            f"  outer {entry['outer_iteration']} step {entry['continuation_iteration']}: "
            f"{entry['action']:<34} max_violation={entry['max_violation']:.2e} "
            f"penalty={entry['penalty']:.1e}"
        )


def main():
    # 1. The uninterrupted run, seeded on the branch near 1, with a checkpoint
    #    after every outer iteration.
    checkpoints = []
    recorder = ALMHistoryRecorder.from_settings(SETTINGS)
    result = solve(
        WarmStartedEvaluator(SEEDED_BRANCH),
        recorder,
        alm_checkpointing(INNER_OPTIONS, completed_outer_callback=checkpoints.append),
        X0.copy(),
    )
    print_history("uninterrupted run:", recorder.history())
    print(f"  -> {result.termination_reason}: x = {result.x}, f = {result.objective:.6f}")

    # 2. Resume from the checkpoint of outer iteration 2 with a cold evaluator.
    checkpoint = next(
        snapshot
        for snapshot in checkpoints
        if snapshot.completed_outer_iterations == RESUME_AFTER_OUTER
    )
    checkpoint_x = np.asarray(checkpoint.x, dtype=float)
    cold_y = WarmStartedEvaluator().solve_y(checkpoint_x[0])
    print(
        f"at the checkpoint's x a cold evaluator finds y = {cold_y:.4f}; "
        f"the checkpointed warm start is y = {checkpoint.accepted_state.y:.4f}"
    )
    resumed_evaluator = WarmStartedEvaluator()
    resumed_recorder = ALMHistoryRecorder.from_settings(SETTINGS)
    resumed = solve(
        resumed_evaluator,
        resumed_recorder,
        alm_checkpointing(INNER_OPTIONS, resume_state=checkpoint),
        checkpoint_x,
    )
    resumed_history = resumed_recorder.history()
    print_history(f"resumed after outer {RESUME_AFTER_OUTER}:", resumed_history)

    # 3. The solver restored the warm start first; the runs end at the same bits.
    first_action, first_y = resumed_evaluator.log[0]
    state_restored = first_action == "restore" and first_y == checkpoint.accepted_state.y
    matches = (
        resumed.x.tobytes() == result.x.tobytes()
        and resumed.multipliers.tobytes() == result.multipliers.tobytes()
        and float(resumed.penalty).hex() == float(result.penalty).hex()
        and resumed.termination_reason == result.termination_reason
    )
    print(
        f"warm start restored before the first evaluation: {state_restored}; "
        + (
            "resumed run matches the uninterrupted run bit for bit"
            if matches
            else "resumed run DIFFERS from the uninterrupted run"
        )
    )
    return {
        "uninterrupted_success": bool(result.success),
        "termination_reason": result.termination_reason,
        "checkpoints": len(checkpoints),
        "resumed_from_outer": RESUME_AFTER_OUTER,
        "cold_start_finds_another_branch": abs(cold_y - checkpoint.accepted_state.y) > 0.5,
        "resumed_matches_uninterrupted_bitwise": matches,
        "evaluator_state_restored_before_resume": state_restored,
        "resumed_history_entries": len(resumed_history),
        "x": result.x.tolist(),
    }


if __name__ == "__main__":
    main()
