"""Build a :func:`minimize_alm` evaluator from simsopt objectives.

A constraint row is ``row(base_objective) -> (signed_value, grad, ...)`` at the dofs just
set: ``signed_value <= 0`` is feasible, ``grad`` spans ``base_objective``'s free dofs, and
trailing items are ignored. ``functools.partial(signed_upper_bound, objective, bound)`` and a
:mod:`simsopt_alm.signed_constraints` kernel with its leading arguments bound are rows.

:class:`ALMPhysics` is what an evaluation depends on besides the multipliers and the
penalty; :func:`cached_alm_evaluator` reuses it when ``minimize_alm`` revisits an x.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, Tuple

import numpy as np

from .core import augmented_inequality_objective
from .evaluation import ALMEvaluation, ALMEvaluator
from .events import _frozen_event_value, _require_acyclic_containers, _writable_copy

__all__ = [
    "ALMPhysics",
    "CachedALMEvaluator",
    "alm_problem_physics",
    "cached_alm_evaluator",
    "evaluate_alm_problem",
    "signed_lower_bound",
    "signed_upper_bound",
]

# Evaluation keys ``extras`` may not overlay, and why: the terms that depend
# on the multipliers or the penalty, which ALMPhysics.evaluation computes, and
# the physics that L and its gradient are built from, which has one source.
_RESERVED_EXTRA_KEYS = (
    (
        frozenset(
            (
                "total",
                "grad",
                "positive_shift_values",
                "augmented_term_by_constraint",
                "stationarity_norm",
            )
        ),
        "depend on the multipliers or the penalty; ALMPhysics.evaluation computes them",
    ),
    (
        frozenset(("base_value", "base_grad", "constraint_values", "constraint_grads")),
        "are ALMPhysics fields; pass them to the constructor",
    ),
)


@dataclass(frozen=True, eq=False)
class ALMPhysics:
    """The multiplier- and penalty-independent part of one ALM evaluation at one x.

    ``extras`` are further evaluation keys that depend only on x: routing
    (``feasibility_values``, ``dual_update_values``, the hybrid quartet), scales,
    tolerances and diagnostics, but neither the four physics fields nor a term
    that depends on the multipliers or the penalty (``ValueError``); an extra
    ``nonfinite_evaluation=True`` makes ``total`` NaN. The instance owns
    read-only copies of every array it is given and of every mapping, list and
    tuple in ``extras`` (any ``Mapping``, not only a dict), which must not
    contain themselves; other objects in ``extras`` are kept by reference.
    """

    base_value: float
    base_grad: np.ndarray
    constraint_values: np.ndarray
    constraint_grads: Tuple[np.ndarray, ...]
    extras: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self):
        for reserved_keys, reason in _RESERVED_EXTRA_KEYS:
            rejected = sorted(reserved_keys.intersection(self.extras))
            if rejected:
                raise ValueError(f"ALMPhysics extras {rejected} {reason}")
        _require_acyclic_containers(self.extras, context="ALMPhysics extras", path="extras")
        # The solver's read-only copy rule (events.py): arrays become read-only
        # copies, mappings read-only views, lists and tuples tuples. One memo
        # per field: its keys are ids of objects alive only during that call.
        object.__setattr__(self, "base_value", float(self.base_value))
        object.__setattr__(
            self,
            "base_grad",
            _frozen_event_value(np.asarray(self.base_grad, dtype=float), {}),
        )
        object.__setattr__(
            self,
            "constraint_values",
            _frozen_event_value(np.asarray(self.constraint_values, dtype=float), {}),
        )
        object.__setattr__(
            self,
            "constraint_grads",
            _frozen_event_value(
                tuple(np.asarray(grad, dtype=float) for grad in self.constraint_grads),
                {},
            ),
        )
        object.__setattr__(
            self, "extras", _frozen_event_value(MappingProxyType(dict(self.extras)), {})
        )

    def evaluation(self, multipliers, penalty) -> ALMEvaluation:
        """:func:`augmented_inequality_objective` at ``multipliers`` and ``penalty``,
        with ``extras`` overlaid; a new dict whose arrays and lists the caller owns."""
        evaluation = augmented_inequality_objective(
            self.base_value,
            self.base_grad,
            self.constraint_values,
            [grad.copy() for grad in self.constraint_grads],
            multipliers,
            penalty,
        )
        evaluation.update(
            (key, _writable_copy(value)) for key, value in self.extras.items()
        )
        if self.extras.get("nonfinite_evaluation"):
            evaluation["total"] = float("nan")
        return evaluation


class CachedALMEvaluator(ALMEvaluator, Protocol):
    """An :class:`ALMEvaluator` that reuses its physics at a revisited x."""

    def cache_clear(self) -> None:
        """Forget the reused physics; the next call evaluates it again."""


class _CachedALMEvaluator:
    """``(x, multipliers, penalty) -> dict`` over one reused :class:`ALMPhysics`."""

    __slots__ = ("_evaluate_physics", "_x_key", "_physics")

    def __init__(self, evaluate_physics: Callable[[np.ndarray], ALMPhysics]):
        self._evaluate_physics = evaluate_physics
        self._x_key = None
        self._physics = None

    def __call__(self, x, multipliers, penalty) -> ALMEvaluation:
        x_array = np.asarray(x, dtype=float)
        x_key = (x_array.shape, x_array.tobytes())
        if x_key != self._x_key:
            self._physics = self._evaluate_physics(x)
            self._x_key = x_key
        return self._physics.evaluation(multipliers, penalty)

    def cache_clear(self) -> None:
        """Forget the reused physics; the next call evaluates it again."""
        self._x_key = None
        self._physics = None


def cached_alm_evaluator(
    evaluate_physics: Callable[[np.ndarray], ALMPhysics],
) -> CachedALMEvaluator:
    """The ``evaluate_problem`` of :func:`minimize_alm` for ``evaluate_physics(x) -> ALMPhysics``.

    It calls ``evaluate_physics`` only when x differs bitwise from the last call's
    and otherwise reuses that physics for the new multipliers and penalty. Call its
    ``cache_clear()`` when anything else the physics reads changes. One instance per
    solve, not shared between threads.
    """
    return _CachedALMEvaluator(evaluate_physics)


def signed_upper_bound(objective, upper_bound, base_objective):
    """``(objective.J() - upper_bound, gradient)``; the gradient spans ``base_objective``'s free dofs."""
    signed_value = float(objective.J()) - float(upper_bound)
    grad = np.asarray(objective.dJ(partials=True)(base_objective), dtype=float)
    return signed_value, grad


def signed_lower_bound(objective, lower_bound, base_objective):
    """``(lower_bound - objective.J(), gradient)``; the gradient spans ``base_objective``'s free dofs."""
    signed_value, grad = signed_upper_bound(objective, lower_bound, base_objective)
    return -signed_value, -grad


def alm_problem_physics(dofs, base_objective, inequalities) -> ALMPhysics:
    """Set ``base_objective.x = dofs``; return the :class:`ALMPhysics` of
    ``f = base_objective.J()`` and one row per entry of ``inequalities``.

    ``cached_alm_evaluator(functools.partial(alm_problem_physics,
    base_objective=..., inequalities=...))`` is an ``evaluate_problem``.
    """
    base_objective.x = dofs
    base_value = float(base_objective.J())
    base_grad = np.asarray(base_objective.dJ(), dtype=float)
    rows = [row(base_objective)[:2] for row in inequalities]
    return ALMPhysics(
        base_value=base_value,
        base_grad=base_grad,
        constraint_values=[signed_value for signed_value, _grad in rows],
        constraint_grads=[grad for _signed_value, grad in rows],
    )


def evaluate_alm_problem(
    dofs, base_objective, inequalities, multipliers, penalty
) -> ALMEvaluation:
    """Set ``base_objective.x = dofs``; return :func:`augmented_inequality_objective`
    for ``f = base_objective.J()`` and one row per entry of ``inequalities``.

    Wrap it as ``lambda x, multipliers, penalty: evaluate_alm_problem(x, ...)`` to
    get the ``evaluate_problem`` that :func:`minimize_alm` calls; it evaluates the
    physics on every call (see :func:`alm_problem_physics` to reuse it).
    """
    return alm_problem_physics(dofs, base_objective, inequalities).evaluation(
        multipliers, penalty
    )
