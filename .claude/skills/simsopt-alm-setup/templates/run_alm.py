#!/usr/bin/env python
"""Run the ALM problem of ``alm_problem.py`` (this directory) with
``minimize_alm``; print the outcome.

    python run_alm.py [--smoke] [--history FILE] [--checkpoints DIR] [--resume FILE]

``--smoke`` builds the problem at its smoke size (small resolution, small
``maxiter``) to exercise the whole pipeline in seconds or minutes. The opt-ins:
``--history FILE`` writes one JSON entry per outer-step decision
(``ALMHistoryRecorder``); ``--checkpoints DIR`` pickles an
``ALMTransitionSnapshot`` after every outer iteration (``outer_NNN.pkl``) and
at the end (``final.pkl``); ``--resume FILE`` continues a run from one such
non-final snapshot with the rest of the ``maxiter`` budget (load only
checkpoints you wrote: unpickling runs code). The last line printed is
``ALM_RESULT {json}``: the termination reason, feasibility, multipliers by
row, and the problem's own ``finish`` summary.

An existing script calls ``run(build_problem())`` instead (the skill's
``references/existing-script.md``).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np

from alm_problem import build_problem
from simsopt.solve.alm import minimize_alm
from simsopt.solve.alm.checkpoint import ALMTransitionSnapshot, alm_checkpointing
from simsopt.solve.alm.history import ALMHistoryRecorder

RESULT_PREFIX = "ALM_RESULT "


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--smoke", action="store_true", help="build the problem at its smoke size")
    parser.add_argument("--history", type=Path, help="write the per-step history to this JSON file")
    parser.add_argument("--checkpoints", type=Path, help="pickle a checkpoint per outer iteration here")
    parser.add_argument("--resume", type=Path, help="resume from this non-final checkpoint pickle")
    return parser.parse_args(argv)


def checkpoint_writer(directory: Path):
    """``completed_outer_callback`` that pickles each snapshot into ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)

    def write(snapshot: ALMTransitionSnapshot) -> None:
        name = (
            f"outer_{snapshot.completed_outer_iterations:03d}.pkl"
            if snapshot.resume_eligible
            else "final.pkl"
        )
        with open(directory / name, "wb") as handle:
            pickle.dump(snapshot, handle)

    return write


def load_checkpoint(path: Path) -> ALMTransitionSnapshot:
    with open(path, "rb") as handle:
        return pickle.load(handle)


def run(problem, history: Optional[Path] = None, checkpoints: Optional[Path] = None,
        resume: Optional[Path] = None) -> dict:
    """Solve ``problem`` (from ``build_problem``) with ``minimize_alm``.

    ``history`` (a JSON file), ``checkpoints`` (a directory) and ``resume`` (a
    non-final checkpoint pickle) are the opt-ins of the command line. Returns
    the summary: termination reason, success, message, objective, feasibility,
    signed values and multipliers by row, iteration counts, the restore flag,
    and ``problem.finish(result)`` under ``finish``.
    """
    resume_state = None if resume is None else load_checkpoint(resume)
    x0 = problem.x0
    inner_options = dict(problem.inner_options)
    if resume_state is not None:
        # The run continues at the checkpoint's x with what is left of its
        # L-BFGS-B budget, as the uninterrupted run would have.
        x0 = np.asarray(resume_state.x, dtype=float)
        inner_options["maxiter"] = max(
            int(inner_options["maxiter"]) - int(resume_state.total_inner_iterations), 0
        )
    checkpointing = alm_checkpointing(
        inner_options,
        resume_state=resume_state,
        completed_outer_callback=(
            None if checkpoints is None else checkpoint_writer(checkpoints)
        ),
    )
    recorder = None if history is None else ALMHistoryRecorder.from_settings(problem.settings)

    result = minimize_alm(
        x0,
        problem.constraint_names,
        problem.evaluator,
        problem.settings,
        checkpointing.inner_options,
        resume_from=checkpointing.resume_from,
        on_outer_step=None if recorder is None else recorder.record,
        on_outer_boundary=checkpointing.on_outer_boundary,
        **problem.solver_callbacks(),
    )

    if recorder is not None:
        history.write_text(json.dumps(recorder.history(), indent=1))
    return {
        "problem": problem.name,
        "termination_reason": result.termination_reason,
        "success": bool(result.success),
        "message": result.message,
        "objective": float(result.objective),
        "max_violation": float(result.max_violation),
        "constraint_values": dict(zip(result.constraint_names, result.constraint_values.tolist())),
        "multipliers": dict(zip(result.constraint_names, result.multipliers.tolist())),
        "penalty": float(result.penalty),
        "stationarity_norm": float(result.stationarity_norm),
        "kkt_stationarity_norm": result.kkt_stationarity_norm,
        "outer_iterations": int(result.outer_iterations),
        "inner_iterations": int(result.nit),
        "restored_best_feasible": bool(result.restored_best_feasible),
        "restored_best_feasible_reason": result.restored_best_feasible_reason,
        "finish": problem.finish(result),
    }


def main(argv=None) -> dict:
    args = parse_args(argv)
    summary = run(build_problem(smoke=args.smoke), history=args.history,
                  checkpoints=args.checkpoints, resume=args.resume)
    print(summary["message"])
    print(RESULT_PREFIX + json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
