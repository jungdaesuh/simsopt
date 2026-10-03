"""Phase-neutral lifecycle vocabulary for SciPy objective evaluations."""

from enum import Enum
from typing import Protocol, runtime_checkable


class TargetScipyEvaluationClass(Enum):
    """Role of one provider evaluation in SciPy optimizer control."""

    INITIAL_OPTIMIZER_EVALUATION = "initial_optimizer_evaluation"
    OPTIMIZER_TRIAL = "optimizer_trial"


class TargetScipyEvaluationOutcome(Enum):
    """Final SciPy decision for one optimizer trial evaluation."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"


@runtime_checkable
class TargetScipyEvaluationLifecycle(Protocol):
    """Typed observer for one exact SciPy evaluation identity."""

    def classify_target_scipy_evaluation(
        self,
        evaluation_class: TargetScipyEvaluationClass,
        /,
    ) -> None: ...

    def resolve_target_scipy_evaluation(
        self,
        outcome: TargetScipyEvaluationOutcome,
        /,
    ) -> None: ...
