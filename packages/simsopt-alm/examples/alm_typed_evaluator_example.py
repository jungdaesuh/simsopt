#!/usr/bin/env python
r"""
Type an ALM evaluator against the published schema, ``ALMEvaluation``.

The problem is :math:`\min_x \|x\|^2` s.t. :math:`1 - x_0 \le 0` (solution
:math:`x = (1, 0)`). The evaluator returns the six required keys and optional
keys the solver reads: the objective without penalty terms and its gradient
(``base_value``, ``base_grad``), stationarity norms, the builder's summaries
(``max_violation``, ``positive_shift_values``, ...) and a constraint scale.

An application's own diagnostic keys pass through the solver at run time. To
type them, subclass the schema: ``HalfspaceEvaluation`` below adds
``distance_to_boundary``. A subclass is still an ``ALMEvaluation``, so
``evaluate_halfspace`` satisfies the ``ALMEvaluator`` protocol, which a type
checker verifies at the ``evaluator: ALMEvaluator = ...`` assignment.
"""

from __future__ import annotations

import numpy as np

from simsopt_alm import (
    ALMEvaluation,
    ALMEvaluator,
    ALMResult,
    ALMSettings,
    augmented_inequality_objective,
    minimize_alm,
)


class HalfspaceEvaluation(ALMEvaluation, total=False):
    """``ALMEvaluation`` plus this application's own diagnostic."""

    distance_to_boundary: float


def evaluate_halfspace(
    x: np.ndarray, multipliers: np.ndarray, penalty: float
) -> HalfspaceEvaluation:
    built = augmented_inequality_objective(
        float(x @ x), 2.0 * x, np.array([1.0 - x[0]]), [np.array([-1.0, 0.0])],
        multipliers, penalty,
    )
    evaluation: HalfspaceEvaluation = {
        # Required.
        "total": built["total"],
        "grad": built["grad"],
        "constraint_values": built["constraint_values"],
        "feasibility_values": built["feasibility_values"],
        "dual_update_values": built["dual_update_values"],
        "constraint_grads": built["constraint_grads"],
        # Optional keys the solver reads.
        "base_value": built["base_value"],
        "base_grad": built["base_grad"],
        "stationarity_norm": built["stationarity_norm"],
        "metric_stationarity_norm": float(np.linalg.norm(built["base_grad"])),
        "max_violation": built["max_violation"],
        "max_feasibility_violation": built["max_feasibility_violation"],
        "positive_shift_values": built["positive_shift_values"],
        "augmented_term_by_constraint": built["augmented_term_by_constraint"],
        "constraint_scales": np.ones(1),
        # The application's own.
        "distance_to_boundary": abs(1.0 - float(x[0])),
    }
    return evaluation


evaluator: ALMEvaluator = evaluate_halfspace


def main() -> ALMResult:
    result = minimize_alm(
        np.array([3.0, 2.0]), ["x0_at_least_one"], evaluator, ALMSettings(),
        {"maxiter": 200},
    )
    # The diagnostic came through; ``ALMResult.evaluation`` is typed as a
    # read-only ``Mapping[str, object]``.
    print(f"{result.termination_reason}: x = {result.x}, distance to the "
          f"boundary {result.evaluation['distance_to_boundary']:.1e}")
    return result


if __name__ == "__main__":
    main()
