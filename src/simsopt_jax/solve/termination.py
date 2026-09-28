"""Termination labels of a bounded minimization: the single owner.

A label says why a solve stopped and whether its endpoint is stationary, from the
emitter's ORIGINAL status read through that emitter's own table, the projected gradient
against ``TERMINAL_STATIONARITY_ATOL`` and, when a Hessian is supplied, a second-order
screen on the box. Precedence: NONFINITE, then INFEASIBLE (any coordinate outside the
original box, one ulp included), then the emitter's own convergence stop (CONVERGED,
CONVERGED_FIRST_ORDER_ONLY, SADDLE or SECOND_ORDER_INCONCLUSIVE when stationary, else
OWN_STOP_NOT_STATIONARY), else the table's terminal outcome. The emitter's ``success``
flag is recorded and never read; stagnation of the accepted history is reported and
never overrides a label. Pure host functions: numpy only, no solve, no I/O.

Every curvature verdict is computed from the supplied (AD) Hessian at the endpoint. With
a C1 objective neither PASS nor any label proves a local minimum; finite-difference
confirmation and geometry are separate gates.

What CONVERGED certifies, and what it does not (the frozen metric, unchanged):

* the emitter's own convergence test fired (never a budget or failure stop);
* SciPy's clipped projected gradient ``projgr`` is ``<= TERMINAL_STATIONARITY_ATOL``.
  This is finite-tolerance projected stationarity in the optimizer's own (unscaled)
  coordinates, not exact KKT stationarity: each component is clipped by the distance to
  the bound its descent step moves toward, so in a box narrower than the tolerance an
  inward gradient larger than the tolerance can pass. Such coordinates are reported as
  INWARD (``inward_bound_coordinates``); they do not change the label;
* the symmetrized AD Hessian passes the screen on the movable set, which is sufficient
  for second-order necessity on every cone a resolution of the tolerances could give,
  for the computed model only.

It does not certify a local minimum of the C1 objective, the physical acceptability of
the endpoint, or a small raw (unprojected) gradient.

A record (``TerminationReport.to_record()``) is authoritative only through
:func:`verify_termination_evidence`, which recomputes it from the endpoint evidence;
:func:`termination_record_defects` checks the record's internal consistency only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, TypeGuard

import numpy as np
from simsopt_contracts.optimization_endpoint import TERMINAL_STATIONARITY_ATOL

from .contracts import (
    NONFINITE_RESULT_STATUS,
    SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS,
    LbfgsbRestartReason,
)
from .scipy.contracts import ScipyBounds

__all__ = [
    "BOUND_ACTIVITY_RTOL",
    "STAGNATION_RTOL",
    "STAGNATION_WINDOW",
    "CoordinateClasses",
    "Emitter",
    "EmitterStop",
    "NotApplicable",
    "SecondOrderScreen",
    "SecondOrderVerdict",
    "StagnationReport",
    "TerminationLabel",
    "TerminationReport",
    "classify_coordinates",
    "projected_gradient_inf_norm",
    "second_order_screen",
    "stagnation",
    "EvidenceVerification",
    "termination_record_defects",
    "termination_report",
    "verify_termination_evidence",
]

# A coordinate within this relative distance of a bound, but not on it bitwise, is
# AMBIGUOUS (the D4 activity rule, ``diagnostics/D4/RESULT.md:49``).
BOUND_ACTIVITY_RTOL: Final[float] = 1e-12
# The accepted history is stagnant when its value fell by at most STAGNATION_RTOL
# (relative) over the last STAGNATION_WINDOW accepted iterates.
STAGNATION_WINDOW: Final[int] = 50
STAGNATION_RTOL: Final[float] = 1e-9
# tau_H = max(_TAU_EIGEN_RTOL * lambda_max, _TAU_ASYMMETRY_FACTOR * ||H - H^T||_2) (D4
# ``RESULT.md:50-52``).
_TAU_EIGEN_RTOL: Final[float] = 1e-9
_TAU_ASYMMETRY_FACTOR: Final[float] = 10.0


class TerminationLabel(StrEnum):
    CONVERGED = "CONVERGED"
    CONVERGED_FIRST_ORDER_ONLY = "CONVERGED_FIRST_ORDER_ONLY"
    SADDLE = "SADDLE"
    SECOND_ORDER_INCONCLUSIVE = "SECOND_ORDER_INCONCLUSIVE"
    OWN_STOP_NOT_STATIONARY = "OWN_STOP_NOT_STATIONARY"
    UNRESOLVED_STALL = "UNRESOLVED_STALL"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    LINE_SEARCH_FAILED = "LINE_SEARCH_FAILED"
    INFEASIBLE = "INFEASIBLE"
    NONFINITE = "NONFINITE"
    FAILED = "FAILED"


class Emitter(StrEnum):
    """Whose status vocabulary a result's ``status`` integer is in."""

    # scipy.optimize.minimize(method="L-BFGS-B"), and this package's restart route
    # (status 7) and shared result boundary (``NONFINITE_RESULT_STATUS``, 8).
    SCIPY_LBFGSB = "scipy-lbfgsb"
    SCIPY_TRUST_CONSTR = "scipy-trust-constr"
    SCIPY_TRUST_EXACT = "scipy-trust-exact"


_OwnStop = Literal["own_stop"]
_OWN_STOP: Final[_OwnStop] = "own_stop"
_Outcome = TerminationLabel | _OwnStop
_L = TerminationLabel

# Each table transcribes one emitter's vocabulary (SciPy 1.17.1). A status a table does
# not carry is FAILED.
_STATUS_OUTCOMES: Final[Mapping[Emitter, Mapping[int, _Outcome]]] = MappingProxyType(
    {
        # ``_lbfgsb_py.py:487-492``: 0 is the pgtol or ftol stop, 1 the iteration or
        # evaluation limit, 2 anything else (ABNORMAL line search). Status 7 is resolved
        # by the restart reason below. The integer meanings are those of
        # ``simsopt_contracts.optimization_endpoint``'s "scipy-lbfgsb" table.
        # ``NONFINITE_RESULT_STATUS`` is not SciPy's: ``dispatch._public_result`` sets
        # it on a non-finite returned state, SciPy's own status kept as
        # ``OptimizerResult.raw_status``; the finiteness precedence gives the same label.
        Emitter.SCIPY_LBFGSB: MappingProxyType(
            {
                0: _OWN_STOP,
                1: _L.BUDGET_EXHAUSTED,
                2: _L.LINE_SEARCH_FAILED,
                NONFINITE_RESULT_STATUS: _L.NONFINITE,
            }
        ),
        # ``minimize_trustregion_constr.py:17-22, 463-469, 560-566``: 1 gtol, 2 xtol (a
        # radius stop, still the method's own stop), 0 the iteration/evaluation limit, 3
        # a callback, 4 constraint violation above gtol.
        Emitter.SCIPY_TRUST_CONSTR: MappingProxyType(
            {
                1: _OWN_STOP,
                2: _OWN_STOP,
                0: _L.BUDGET_EXHAUSTED,
                3: _L.FAILED,
                4: _L.INFEASIBLE,
            }
        ),
        # ``_trustregion.py:241-321``: 0 ||g||_2 < gtol, 1 maxiter, 2 predicted
        # reduction <= 0, 3 LinAlgError.
        Emitter.SCIPY_TRUST_EXACT: MappingProxyType(
            {0: _OWN_STOP, 1: _L.BUDGET_EXHAUSTED, 2: _L.FAILED, 3: _L.FAILED}
        ),
    }
)
# ``SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS`` is emitted for exactly these two last restart
# reasons (``dispatch._minimize_lbfgsb_with_restarts``); the reason decides before any
# generic mapping.
_UNRESOLVED_STALL_OUTCOMES: Final[Mapping[LbfgsbRestartReason, TerminationLabel]] = (
    MappingProxyType(
        {
            LbfgsbRestartReason.BUDGET_EXHAUSTED: _L.BUDGET_EXHAUSTED,
            LbfgsbRestartReason.FRESH_MEMORY_STALL: _L.UNRESOLVED_STALL,
        }
    )
)


@dataclass(frozen=True, slots=True)
class NotApplicable:
    """A field with no value, for a stated reason: ``{"not_applicable": reason}``."""

    reason: str

    def to_record(self) -> dict[str, str]:
        return {"not_applicable": self.reason}


_NO_MOVABLE_DIRECTIONS: Final = NotApplicable("no_movable_directions")
_SHORT_HISTORY: Final = NotApplicable("history_shorter_than_window")
_NONFINITE_HISTORY: Final = NotApplicable("nonfinite_history")


def _box_sides(bounds: ScipyBounds | None, size: int) -> tuple[np.ndarray, np.ndarray]:
    if bounds is None:
        return np.full(size, -math.inf), np.full(size, math.inf)
    return np.asarray(bounds.lower, dtype=float), np.asarray(bounds.upper, dtype=float)


def projected_gradient_inf_norm(
    x: np.ndarray, gradient: np.ndarray, bounds: ScipyBounds | None
) -> float:
    """SciPy L-BFGS-B's ``projgr`` (``__lbfgsb.c:2764``), the norm its pgtol test reads.

    Each gradient component is clipped by the distance from ``x`` to the bound its
    descent step moves toward; for a feasible ``x`` this equals
    ``|P[l,u](x - g) - x|_inf``. A NaN component propagates.
    """
    if bounds is None:
        return float(np.max(np.abs(gradient)))
    lower, upper = _box_sides(bounds, len(gradient))
    projected = np.where(
        gradient < 0.0,
        np.maximum(x - upper, gradient),
        np.minimum(x - lower, gradient),
    )
    return float(np.max(np.abs(projected)))


@dataclass(frozen=True, slots=True)
class CoordinateClasses:
    """Each coordinate's class at a feasible point, as sorted index tuples.

    Below, ``tol = TERMINAL_STATIONARITY_ATOL``. FIXED:
    ``l == u``. On a bound bitwise: STRONG when descent points out of the box with
    ``|g| > tol``, INWARD when it points into the box with ``|g| > tol``, WEAK when
    ``g == 0`` exactly, AMBIGUOUS when ``0 < |g| <= tol``. Also AMBIGUOUS: within
    ``BOUND_ACTIVITY_RTOL * max(1, |b|)`` of a finite bound without touching it.
    FREE otherwise. ``on_lower`` / ``on_upper`` list the non-fixed coordinates equal
    to that bound bitwise.
    """

    fixed: tuple[int, ...]
    strong: tuple[int, ...]
    weak: tuple[int, ...]
    inward: tuple[int, ...]
    ambiguous: tuple[int, ...]
    free: tuple[int, ...]
    on_lower: tuple[int, ...]
    on_upper: tuple[int, ...]

    @property
    def movable(self) -> tuple[int, ...]:
        return tuple(sorted((*self.free, *self.weak, *self.inward, *self.ambiguous)))

    @property
    def size(self) -> int:
        return len(self.movable) + len(self.fixed) + len(self.strong)

    def to_record(self) -> dict[str, list[int]]:
        return {
            "fixed": list(self.fixed),
            "strong": list(self.strong),
            "weak": list(self.weak),
            "inward": list(self.inward),
            "ambiguous": list(self.ambiguous),
            "free": list(self.free),
        }


def _near_finite_bound(value: float, bound: float) -> bool:
    return math.isfinite(bound) and abs(value - bound) <= BOUND_ACTIVITY_RTOL * max(
        1.0, abs(bound)
    )


def classify_coordinates(
    x: np.ndarray, gradient: np.ndarray, bounds: ScipyBounds | None
) -> CoordinateClasses:
    """Classes of every coordinate of a feasible ``x`` (outside the box: ValueError)."""
    lower, upper = _box_sides(bounds, len(x))
    if np.any(x < lower) or np.any(x > upper):
        raise ValueError(
            "the second-order screen needs a point inside the box; x is outside it"
        )
    classes: dict[str, list[int]] = {
        name: []
        for name in (
            "fixed",
            "strong",
            "weak",
            "inward",
            "ambiguous",
            "free",
            "on_lower",
            "on_upper",
        )
    }
    tol = TERMINAL_STATIONARITY_ATOL
    for i, (value, grad, low, high) in enumerate(zip(x, gradient, lower, upper)):
        if low == high:
            classes["fixed"].append(i)
            continue
        if value == low or value == high:
            classes["on_lower" if value == low else "on_upper"].append(i)
            # Positive ``outward`` means the descent step -g leaves the box through this
            # bound.
            outward = grad if value == low else -grad
            if outward > tol:
                classes["strong"].append(i)
            elif outward < -tol:
                classes["inward"].append(i)
            elif grad == 0.0:
                classes["weak"].append(i)
            else:
                classes["ambiguous"].append(i)
        elif _near_finite_bound(value, low) or _near_finite_bound(value, high):
            classes["ambiguous"].append(i)
        else:
            classes["free"].append(i)
    return CoordinateClasses(
        **{name: tuple(indices) for name, indices in classes.items()}
    )


class SecondOrderVerdict(StrEnum):
    PASS = "PASS"
    SADDLE = "SADDLE"
    INCONCLUSIVE = "INCONCLUSIVE"


ConeKind = Literal["exact", "candidate"]


@dataclass(frozen=True, slots=True)
class SecondOrderScreen:
    """The box second-order screen at one endpoint.

    ``lambda_min`` / ``tau_h`` are of the symmetrized Hessian on the movable set M = F,
    W, I and Z, and are ``NotApplicable("no_movable_directions")`` exactly when M is
    empty (a vacuous PASS). ``witness`` is the full-length SADDLE direction, else None.
    ``cone`` is "exact" only when no coordinate is AMBIGUOUS or INWARD, at most one is
    WEAK and the free gradient is exactly zero; otherwise only PASS or a valid witness
    is decisive.
    """

    verdict: SecondOrderVerdict
    cone: ConeKind
    lambda_min: float | NotApplicable
    tau_h: float | NotApplicable
    witness: tuple[float, ...] | None
    classes: CoordinateClasses
    # Skew-threshold sensitivity: the verdict is the same whether tau_H takes the skew
    # of H, H^T or H_sym = (H + H^T)/2 (one symmetric part; only the skew term of tau_H
    # differs). Not an accuracy check of H: a symmetric but wrong H passes it.
    decision_stable: bool

    def to_record(self) -> dict[str, object]:
        def value(field: float | NotApplicable) -> float | dict[str, str]:
            return field.to_record() if isinstance(field, NotApplicable) else field

        return {
            "verdict": self.verdict.value,
            "cone": self.cone,
            "lambda_min": value(self.lambda_min),
            "tau_H": value(self.tau_h),
            "witness": None if self.witness is None else list(self.witness),
            "decision_stable": self.decision_stable,
            "n": self.classes.size,
            "classes": self.classes.to_record(),
            "on_lower": list(self.classes.on_lower),
            "on_upper": list(self.classes.on_upper),
            "inward_bound_coordinates": {
                "count": len(self.classes.inward),
                "indices": list(self.classes.inward),
            },
        }


def _in_tangent_cone(direction: np.ndarray, classes: CoordinateClasses) -> bool:
    """``T(x)``: >= 0 on a bitwise lower bound, <= 0 on an upper one, 0 where fixed."""
    return bool(
        np.all(direction[list(classes.on_lower)] >= 0.0)
        and np.all(direction[list(classes.on_upper)] <= 0.0)
        and np.all(direction[list(classes.fixed)] == 0.0)
    )


def _witness_accepted(
    direction: np.ndarray,
    symmetric: np.ndarray,
    gradient: np.ndarray,
    classes: CoordinateClasses,
    tau_h: float,
) -> bool:
    """A SADDLE witness: in ``T(x)``, zero on AMBIGUOUS/STRONG/FIXED coordinates,
    first-order compatible (``g.d <= 0``) and ``d.H.d < -tau_H |d|^2``."""
    pinned = [*classes.ambiguous, *classes.strong, *classes.fixed]
    return bool(
        _in_tangent_cone(direction, classes)
        and np.all(direction[pinned] == 0.0)
        and float(gradient @ direction) <= 0.0
        and float(direction @ symmetric @ direction)
        < -tau_h * float(direction @ direction)
    )


def _witness_candidates(
    symmetric: np.ndarray, classes: CoordinateClasses
) -> tuple[np.ndarray, ...]:
    """+-v(F, W, I), then +-v(F), embedded in full coordinates; empty sets skipped."""
    size = symmetric.shape[0]
    candidates: list[np.ndarray] = []
    for index_set in (
        tuple(sorted((*classes.free, *classes.weak, *classes.inward))),
        classes.free,
    ):
        if not index_set:
            continue
        rows = list(index_set)
        _values, vectors = np.linalg.eigh(symmetric[np.ix_(rows, rows)])
        for sign in (1.0, -1.0):
            direction = np.zeros(size)
            direction[rows] = sign * vectors[:, 0]
            candidates.append(direction)
    return tuple(candidates)


def second_order_screen(
    hessian: np.ndarray, x: np.ndarray, gradient: np.ndarray, bounds: ScipyBounds | None
) -> SecondOrderScreen:
    """Screen the symmetrized Hessian on the box at a feasible ``x``.

    PASS if ``lambda_min >= -tau_H`` on the movable set (sufficient: that set spans
    every cone any resolution of the tolerances could produce). SADDLE if a candidate
    witness lies in the box's tangent cone, leaves AMBIGUOUS, STRONG and FIXED
    coordinates at zero, and has ``g.d <= 0`` and ``d.H.d < -tau_H |d|^2``.
    INCONCLUSIVE otherwise, and also when the verdict depends on whether tau_H takes
    the skew of H, H^T or H_sym (a skew-threshold sensitivity check, not evidence that
    H is accurate). Decisions use H_sym; the raw H's skew enters only through tau_H.
    No eigen routine is ever called on an empty matrix.
    """
    classes = classify_coordinates(x, gradient, bounds)
    exact_cone = (
        not classes.ambiguous
        and not classes.inward
        and len(classes.weak) <= 1
        and bool(np.all(gradient[list(classes.free)] == 0.0))
    )
    cone: ConeKind = "exact" if exact_cone else "candidate"
    movable = list(classes.movable)
    if not movable:
        return SecondOrderScreen(
            verdict=SecondOrderVerdict.PASS,
            cone=cone,
            lambda_min=_NO_MOVABLE_DIRECTIONS,
            tau_h=_NO_MOVABLE_DIRECTIONS,
            witness=None,
            classes=classes,
            decision_stable=True,
        )
    # H, H^T and H_sym share this symmetric part bitwise, so one eigen-solve serves
    # all three; they differ only in the skew term of tau_H.
    symmetric = 0.5 * (hessian + hessian.T)
    eigenvalues = np.linalg.eigvalsh(symmetric[np.ix_(movable, movable)])
    lambda_min, lambda_max = float(eigenvalues[0]), float(eigenvalues[-1])
    taus = tuple(
        max(
            _TAU_EIGEN_RTOL * lambda_max,
            _TAU_ASYMMETRY_FACTOR * float(np.linalg.norm(matrix - matrix.T, 2)),
        )
        for matrix in (hessian, hessian.T, symmetric)
    )
    candidates: tuple[np.ndarray, ...] | None = None
    decisions: list[tuple[SecondOrderVerdict, tuple[float, ...] | None]] = []
    for tau in taus:
        if lambda_min >= -tau:
            decisions.append((SecondOrderVerdict.PASS, None))
            continue
        if candidates is None:
            candidates = _witness_candidates(symmetric, classes)
        witness = next(
            (
                tuple(float(component) for component in direction)
                for direction in candidates
                if _witness_accepted(direction, symmetric, gradient, classes, tau)
            ),
            None,
        )
        decisions.append(
            (
                SecondOrderVerdict.INCONCLUSIVE
                if witness is None
                else SecondOrderVerdict.SADDLE,
                witness,
            )
        )
    stable = len({verdict for verdict, _witness in decisions}) == 1
    verdict, witness = (
        decisions[0] if stable else (SecondOrderVerdict.INCONCLUSIVE, None)
    )
    return SecondOrderScreen(
        verdict=verdict,
        cone=cone,
        lambda_min=lambda_min,
        tau_h=taus[0],
        witness=witness,
        classes=classes,
        decision_stable=stable,
    )


@dataclass(frozen=True, slots=True)
class StagnationReport:
    """Whether the accepted history fell by at most ``STAGNATION_RTOL`` (relative).

    The fall is measured over the last ``STAGNATION_WINDOW`` accepted iterates.
    ``window_rel_drop`` is that fall over ``|f_old|`` (at ``f_old == 0``: 0, or +-inf
    by the sign of the fall); it is not applicable for a shorter or non-finite history.
    """

    stagnated: bool
    window_rel_drop: float | NotApplicable


def stagnation(accepted_fun: Sequence[float]) -> StagnationReport:
    if len(accepted_fun) <= STAGNATION_WINDOW:
        return StagnationReport(stagnated=False, window_rel_drop=_SHORT_HISTORY)
    old, new = float(accepted_fun[-1 - STAGNATION_WINDOW]), float(accepted_fun[-1])
    if not (math.isfinite(old) and math.isfinite(new)):
        return StagnationReport(stagnated=False, window_rel_drop=_NONFINITE_HISTORY)
    drop = old - new
    if old != 0.0:
        rel_drop = drop / abs(old)
    else:
        rel_drop = 0.0 if drop == 0.0 else math.copysign(math.inf, drop)
    return StagnationReport(
        stagnated=drop <= STAGNATION_RTOL * abs(old), window_rel_drop=rel_drop
    )


def _boundary_status_emitter_defect(emitter: str) -> str:
    """``NONFINITE_RESULT_STATUS`` is set only by ``dispatch._public_result``, and SciPy
    L-BFGS-B is the only emitter here routed through it."""
    return (
        f"status NONFINITE_RESULT_STATUS is the SciPy L-BFGS-B result boundary's, not "
        f"emitter {emitter!r}'s"
    )


@dataclass(frozen=True, slots=True)
class EmitterStop:
    """A solver's termination exactly as it reported it."""

    emitter: Emitter
    status: int
    message: str
    success: bool
    # SciPy L-BFGS-B restart route only: ``OptimizerResult.restart_log[-1].reason``,
    # else None.
    last_restart_reason: LbfgsbRestartReason | None = None

    def __post_init__(self) -> None:
        if (
            self.last_restart_reason is not None
            and self.emitter is not Emitter.SCIPY_LBFGSB
        ):
            raise ValueError(
                f"a restart reason is SciPy L-BFGS-B restart-route metadata, not "
                f"{self.emitter.value!r}'s"
            )
        if (
            self.status == NONFINITE_RESULT_STATUS
            and self.emitter is not Emitter.SCIPY_LBFGSB
        ):
            raise ValueError(_boundary_status_emitter_defect(self.emitter.value))


def _emitter_outcome(
    emitter: Emitter, status: int, restart: LbfgsbRestartReason | None
) -> _Outcome:
    if (
        emitter is Emitter.SCIPY_LBFGSB
        and status == SCIPY_LBFGSB_UNRESOLVED_STALL_STATUS
    ):
        if restart is None:
            return _L.FAILED
        return _UNRESOLVED_STALL_OUTCOMES.get(restart, _L.FAILED)
    return _STATUS_OUTCOMES[emitter].get(status, _L.FAILED)


_VERDICT_LABELS: Final[Mapping[SecondOrderVerdict, TerminationLabel]] = (
    MappingProxyType(
        {
            SecondOrderVerdict.PASS: _L.CONVERGED,
            SecondOrderVerdict.SADDLE: _L.SADDLE,
            SecondOrderVerdict.INCONCLUSIVE: _L.SECOND_ORDER_INCONCLUSIVE,
        }
    )
)


def _label(
    *,
    emitter: Emitter,
    status: int,
    restart: LbfgsbRestartReason | None,
    nonfinite: bool,
    infeasible: bool,
    projected_grad_norm_inf: float | None,
    verdict: SecondOrderVerdict | None,
) -> TerminationLabel:
    """The precedence of the module docstring; the only place a label is decided."""
    if nonfinite:
        return _L.NONFINITE
    if infeasible:
        return _L.INFEASIBLE
    outcome = _emitter_outcome(emitter, status, restart)
    if outcome != _OWN_STOP:
        return TerminationLabel(outcome)
    # None only with a non-finite x or gradient, which returned NONFINITE above.
    if (
        projected_grad_norm_inf is None
        or projected_grad_norm_inf > TERMINAL_STATIONARITY_ATOL
    ):
        return _L.OWN_STOP_NOT_STATIONARY
    return (
        _L.CONVERGED_FIRST_ORDER_ONLY if verdict is None else _VERDICT_LABELS[verdict]
    )


@dataclass(frozen=True, slots=True)
class TerminationReport:
    """One labelled endpoint.

    ``projected_grad_norm_inf`` and ``max_bound_excursion`` are None only when the
    values they need are non-finite (listed in ``nonfinite_fields``). ``second_order``
    is None when no Hessian was supplied, the post-solve Hessian raised
    (``hessian_exception`` names its class), an input is non-finite, or ``x`` lies
    outside the box. ``to_record`` is the JSON form :func:`termination_record_defects`
    validates.
    """

    emitter_stop: EmitterStop
    label: TerminationLabel
    projected_grad_norm_inf: float | None
    nonfinite_fields: tuple[str, ...]
    max_bound_excursion: float | None
    hessian_supplied: bool
    hessian_exception: str | None
    second_order: SecondOrderScreen | None
    stagnation: StagnationReport

    def to_record(self) -> dict[str, object]:
        drop = self.stagnation.window_rel_drop
        restart = self.emitter_stop.last_restart_reason
        return {
            "emitter": self.emitter_stop.emitter.value,
            "status": self.emitter_stop.status,
            "message": self.emitter_stop.message,
            "success": self.emitter_stop.success,
            "last_restart_reason": None if restart is None else restart.value,
            "label": self.label.value,
            "projected_grad_norm_inf": self.projected_grad_norm_inf,
            "stationarity_tol": TERMINAL_STATIONARITY_ATOL,
            "nonfinite_fields": list(self.nonfinite_fields),
            "max_bound_excursion": self.max_bound_excursion,
            "hessian_supplied": self.hessian_supplied,
            "hessian_exception": self.hessian_exception,
            "second_order": None
            if self.second_order is None
            else self.second_order.to_record(),
            "stagnated": self.stagnation.stagnated,
            "stagnation_window_rel_drop": drop.to_record()
            if isinstance(drop, NotApplicable)
            else drop,
        }


def termination_report(
    stop: EmitterStop,
    *,
    x: np.ndarray,
    fun: float,
    gradient: np.ndarray,
    bounds: ScipyBounds | None,
    accepted_fun: Sequence[float],
    hessian: np.ndarray | None = None,
    hessian_exception: str | None = None,
) -> TerminationReport:
    """Label one endpoint against the ORIGINAL ``bounds``.

    ``accepted_fun`` is the accepted value history. ``hessian`` is the post-solve (AD)
    Hessian at ``x``; ``hessian_exception`` instead names the exception class that
    computing it raised. Passing both is a ``ValueError``.
    """
    if hessian is not None and hessian_exception is not None:
        raise ValueError(
            "a post-solve Hessian and a Hessian exception are mutually exclusive"
        )
    finite = {
        "x": bool(np.all(np.isfinite(x))),
        "fun": math.isfinite(fun),
        "gradient": bool(np.all(np.isfinite(gradient))),
        "hessian": hessian is None or bool(np.all(np.isfinite(hessian))),
    }
    nonfinite_fields = tuple(name for name, ok in finite.items() if not ok)
    lower, upper = _box_sides(bounds, len(x))
    excursion = (
        float(max(np.max(lower - x, initial=0.0), np.max(x - upper, initial=0.0), 0.0))
        if finite["x"]
        else None
    )
    projected = (
        projected_gradient_inf_norm(x, gradient, bounds)
        if finite["x"] and finite["gradient"]
        else None
    )
    infeasible = excursion is not None and excursion > 0.0
    screen = (
        second_order_screen(hessian, x, gradient, bounds)
        if hessian is not None and not nonfinite_fields and not infeasible
        else None
    )
    label = _label(
        emitter=stop.emitter,
        status=stop.status,
        restart=stop.last_restart_reason,
        nonfinite=bool(nonfinite_fields),
        infeasible=infeasible,
        projected_grad_norm_inf=projected,
        verdict=None if screen is None else screen.verdict,
    )
    return TerminationReport(
        emitter_stop=stop,
        label=label,
        projected_grad_norm_inf=projected,
        nonfinite_fields=nonfinite_fields,
        max_bound_excursion=excursion,
        hessian_supplied=hessian is not None,
        hessian_exception=hessian_exception,
        second_order=screen,
        stagnation=stagnation(accepted_fun),
    )


# ---------------------------------------------------------------------------
# The record validator.

_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {
        "emitter",
        "status",
        "message",
        "success",
        "last_restart_reason",
        "label",
        "projected_grad_norm_inf",
        "stationarity_tol",
        "nonfinite_fields",
        "max_bound_excursion",
        "hessian_supplied",
        "hessian_exception",
        "second_order",
        "stagnated",
        "stagnation_window_rel_drop",
    }
)
_SCREEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "verdict",
        "cone",
        "lambda_min",
        "tau_H",
        "witness",
        "decision_stable",
        "n",
        "classes",
        "on_lower",
        "on_upper",
        "inward_bound_coordinates",
    }
)
_CLASS_NAMES: Final[tuple[str, ...]] = (
    "fixed",
    "strong",
    "weak",
    "inward",
    "ambiguous",
    "free",
)
_EMITTERS: Final[frozenset[str]] = frozenset(member.value for member in Emitter)
_LABELS: Final[frozenset[str]] = frozenset(member.value for member in TerminationLabel)
_VERDICTS: Final[frozenset[str]] = frozenset(
    member.value for member in SecondOrderVerdict
)
_RESTART_REASONS: Final[frozenset[str]] = frozenset(
    member.value for member in LbfgsbRestartReason
)
_NONFINITE_FIELD_NAMES: Final[frozenset[str]] = frozenset(
    {"x", "fun", "gradient", "hessian"}
)


def _integer(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: object) -> TypeGuard[int | float]:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _index_list(value: object) -> TypeGuard[list[int]]:
    return isinstance(value, list) and all(_integer(i) for i in value)


def _screen_defects(screen: Mapping[str, object]) -> list[str]:
    if set(screen) != _SCREEN_KEYS:
        return [
            f"second_order keys {sorted(set(screen) ^ _SCREEN_KEYS)} missing or extra"
        ]
    classes, size = screen["classes"], screen["n"]
    if not _integer(size) or size < 1:
        return ["second_order.n is not a positive integer"]
    if (
        not isinstance(classes, dict)
        or set(classes) != set(_CLASS_NAMES)
        or not all(_index_list(classes[name]) for name in _CLASS_NAMES)
    ):
        return ["second_order.classes is not six index lists"]
    on_lower, on_upper = screen["on_lower"], screen["on_upper"]
    if not _index_list(on_lower) or not _index_list(on_upper):
        return ["second_order.on_lower/on_upper are not index lists"]
    members = [i for name in _CLASS_NAMES for i in classes[name]]
    defects: list[str] = []
    if sorted(members) != list(range(size)):
        defects.append("second_order.classes do not partition range(n)")
    on_bound = set(on_lower) | set(on_upper)
    if (
        set(on_lower) & set(on_upper)
        or not {*classes["strong"], *classes["weak"], *classes["inward"]} <= on_bound
    ):
        defects.append("second_order bound sides are inconsistent with the classes")
    if not on_bound <= {
        *classes["strong"],
        *classes["weak"],
        *classes["inward"],
        *classes["ambiguous"],
    }:
        defects.append("second_order bound sides include a fixed or free coordinate")
    inward = screen["inward_bound_coordinates"]
    if inward != {"count": len(classes["inward"]), "indices": classes["inward"]}:
        defects.append(
            "second_order.inward_bound_coordinates disagrees with the INWARD class"
        )
    movable = [
        i for name in ("free", "weak", "inward", "ambiguous") for i in classes[name]
    ]
    verdict, cone = screen["verdict"], screen["cone"]
    if verdict not in _VERDICTS or cone not in ("exact", "candidate"):
        return [*defects, "second_order verdict or cone is not in the vocabulary"]
    stable = screen["decision_stable"]
    if not isinstance(stable, bool):
        return [*defects, "second_order.decision_stable is not a boolean"]
    if cone == "exact" and (
        classes["ambiguous"] or classes["inward"] or len(classes["weak"]) > 1
    ):
        defects.append(
            "second_order.cone is exact with ambiguous, inward or several weak entries"
        )
    lambda_min, tau_h, witness = (
        screen["lambda_min"],
        screen["tau_H"],
        screen["witness"],
    )
    tag = _NO_MOVABLE_DIRECTIONS.to_record()
    if lambda_min == tag or tau_h == tag:
        if (
            lambda_min != tag
            or tau_h != tag
            or movable
            or verdict != SecondOrderVerdict.PASS
            or witness is not None
        ):
            defects.append(
                "not_applicable eigen fields need a vacuous PASS, empty movable set"
            )
        return defects
    if not movable:
        defects.append(
            "an empty movable set must carry the not_applicable eigen fields"
        )
    if (
        not _finite_number(lambda_min)
        or not _finite_number(tau_h)
        or float(tau_h) < 0.0
    ):
        return [
            *defects,
            "second_order lambda_min/tau_H are not finite numbers (tau_H >= 0)",
        ]
    lambda_min, tau_h = float(lambda_min), float(tau_h)
    if not stable:
        if verdict != SecondOrderVerdict.INCONCLUSIVE or witness is not None:
            defects.append("an unstable decision is INCONCLUSIVE with no witness")
        return defects
    if verdict == SecondOrderVerdict.PASS:
        if lambda_min < -tau_h or witness is not None:
            defects.append("PASS needs lambda_min >= -tau_H and no witness")
        return defects
    if lambda_min >= -tau_h:
        defects.append(f"{verdict} needs lambda_min < -tau_H")
    if verdict == SecondOrderVerdict.INCONCLUSIVE:
        if witness is not None:
            defects.append("INCONCLUSIVE carries no witness")
        return defects
    if (
        not isinstance(witness, list)
        or len(witness) != size
        or not all(_finite_number(c) for c in witness)
    ):
        return [*defects, "SADDLE needs a finite full-length witness"]
    direction = np.asarray(witness, dtype=float)
    pinned = [*classes["ambiguous"], *classes["strong"], *classes["fixed"]]
    if (
        np.any(direction[on_lower] < 0.0)
        or np.any(direction[on_upper] > 0.0)
        or np.any(direction[pinned] != 0.0)
        or not np.any(direction != 0.0)
    ):
        defects.append(
            "SADDLE witness is outside the tangent cone or moves a pinned coordinate"
        )
    return defects


def termination_record_defects(record: Mapping[str, object]) -> tuple[str, ...]:
    """Every way ``record`` breaks the ``TerminationReport.to_record()`` schema.

    Internal consistency only: empty means the record is well formed, not that its
    label holds for any endpoint (that is :func:`verify_termination_evidence`). A
    field may be null only under the failure schema: non-finite inputs listed in
    ``nonfinite_fields`` (label NONFINITE), a named ``hessian_exception``, no Hessian
    supplied, or (for the screen) ``x`` outside the box. The not-applicable eigen tags
    are accepted only with a vacuous PASS on an empty movable set.
    ``NONFINITE_RESULT_STATUS`` is accepted only on the SciPy L-BFGS-B emitter and
    only with a non-finite x, fun or gradient listed. The
    label must be the one the recorded fields give under the module's precedence.
    """
    if set(record) != _RECORD_KEYS:
        return (
            f"termination keys {sorted(set(record) ^ _RECORD_KEYS)} missing or extra",
        )
    defects: list[str] = []
    emitter, status, label = record["emitter"], record["status"], record["label"]
    restart = record["last_restart_reason"]
    if emitter not in _EMITTERS:
        defects.append(f"unknown emitter {emitter!r}")
    if not _integer(status):
        defects.append("status is not an integer")
    if not isinstance(record["message"], str) or not isinstance(
        record["success"], bool
    ):
        defects.append("message/success are not a string and a boolean")
    if restart is not None and restart not in _RESTART_REASONS:
        defects.append(f"unknown restart reason {restart!r}")
    if restart is not None and emitter != Emitter.SCIPY_LBFGSB.value:
        defects.append(f"a restart reason on emitter {emitter!r}, not SciPy L-BFGS-B")
    if label not in _LABELS:
        defects.append(f"unknown label {label!r}")
    if record["stationarity_tol"] != TERMINAL_STATIONARITY_ATOL:
        defects.append("stationarity_tol is not TERMINAL_STATIONARITY_ATOL")
    nonfinite = record["nonfinite_fields"]
    if not isinstance(nonfinite, list) or not set(nonfinite) <= _NONFINITE_FIELD_NAMES:
        return (*defects, "nonfinite_fields is not a list of x/fun/gradient/hessian")
    # The boundary sets this status only for SciPy L-BFGS-B here, and only on a
    # non-finite x, fun or gradient (a post-solve Hessian is not a returned field).
    # Whether the listed fields are non-finite in the endpoint is
    # ``verify_termination_evidence``'s recomputation.
    if _integer(status) and status == NONFINITE_RESULT_STATUS:
        if emitter != Emitter.SCIPY_LBFGSB.value:
            defects.append(_boundary_status_emitter_defect(str(emitter)))
        elif not {"x", "fun", "gradient"} & set(nonfinite):
            defects.append(
                "status is NONFINITE_RESULT_STATUS without a non-finite x, fun or "
                "gradient"
            )
    projected, excursion = (
        record["projected_grad_norm_inf"],
        record["max_bound_excursion"],
    )
    if projected is None:
        if not {"x", "gradient"} & set(nonfinite):
            defects.append(
                "projected_grad_norm_inf is null without a non-finite x or gradient"
            )
    elif not _finite_number(projected) or float(projected) < 0.0:
        defects.append("projected_grad_norm_inf is not a finite nonnegative number")
    if excursion is None:
        if "x" not in nonfinite:
            defects.append("max_bound_excursion is null without a non-finite x")
    elif not _finite_number(excursion) or float(excursion) < 0.0:
        defects.append("max_bound_excursion is not a finite nonnegative number")
    supplied, exception = record["hessian_supplied"], record["hessian_exception"]
    if not isinstance(supplied, bool):
        defects.append("hessian_supplied is not a boolean")
    if exception is not None and (
        not isinstance(exception, str) or not exception or supplied
    ):
        defects.append(
            "hessian_exception must be a class name, and only without a Hessian"
        )
    screen = record["second_order"]
    infeasible = _finite_number(excursion) and float(excursion) > 0.0
    screen_expected = supplied is True and not nonfinite and not infeasible
    if screen is None:
        if screen_expected:
            defects.append(
                "second_order is null although a finite Hessian was supplied"
            )
    elif not screen_expected or not isinstance(screen, dict):
        defects.append("second_order is present although no screen could run")
    else:
        defects.extend(_screen_defects(screen))
    if not isinstance(record["stagnated"], bool):
        defects.append("stagnated is not a boolean")
    drop = record["stagnation_window_rel_drop"]
    if drop in (_SHORT_HISTORY.to_record(), _NONFINITE_HISTORY.to_record()):
        if record["stagnated"] is not False:
            defects.append("a short or non-finite history cannot be stagnant")
    elif (
        not isinstance(drop, int | float) or isinstance(drop, bool) or math.isnan(drop)
    ):
        defects.append(
            "stagnation_window_rel_drop is not a number or its not_applicable tag"
        )
    if defects or not _integer(status) or not isinstance(screen, dict | None):
        return tuple(defects)
    verdict = None if screen is None else SecondOrderVerdict(screen["verdict"])
    expected = _label(
        emitter=Emitter(emitter),
        status=status,
        restart=None if restart is None else LbfgsbRestartReason(restart),
        nonfinite=bool(nonfinite),
        infeasible=infeasible,
        projected_grad_norm_inf=projected if _finite_number(projected) else None,
        verdict=verdict,
    )
    if expected != label:
        return (
            f"label {label!r} is not the {expected.value!r} its recorded fields give",
        )
    return ()


# ---------------------------------------------------------------------------
# Binding a record to its endpoint evidence.

# Screen fields not compared by equality. An eigen-solve repeated in another process
# (another BLAS path) may differ in its last bits, so ``lambda_min`` and ``tau_H`` are
# compared under the backward-stability rule of :func:`_eigen_descriptor_defects`;
# an eigenvector's sign and basis are not unique, so the witness is re-tested instead.
_NUMERICAL_SCREEN_FIELDS: Final[frozenset[str]] = frozenset(
    {"lambda_min", "tau_H", "witness"}
)
# An engineering consistency rule for recomputed descriptors, motivated by (not derived
# from) backward-stability theory: a stable symmetric eigen-solver returns the exact
# eigenvalues of H + E with ||E||_2 <= p(n) eps ||H||_2 for a modest polynomial p
# (Golub & Van Loan, 4th ed., 8.3 and 8.4), and by Weyl each eigenvalue moves by at
# most that much. p(n) = 10 n is chosen, not proven for every backend; it bounds
# recomputation differences only and says nothing about the AD Hessian's own error.
_EIGEN_BACKWARD_ERROR_FACTOR: Final[float] = 10.0


def _eigen_descriptor_defects(
    recorded: Mapping[str, object], screen: SecondOrderScreen, symmetric: np.ndarray
) -> list[str]:
    """|d lambda_min| <= 10 n eps ||H_sym,MM||_2 and
    |d tau_H| <= 10 n eps max(tau_H, 1e-9 ||H_sym,MM||_2), n = |M|."""
    lambda_min, tau_h = screen.lambda_min, screen.tau_h
    if isinstance(lambda_min, NotApplicable) or isinstance(tau_h, NotApplicable):
        return [
            f"second_order.{field} is recorded as {recorded[field]!r}, but the "
            f"endpoint evidence gives {expected.to_record()!r}"
            for field, expected in (("lambda_min", lambda_min), ("tau_H", tau_h))
            if isinstance(expected, NotApplicable)
            and recorded[field] != expected.to_record()
        ]
    movable = list(screen.classes.movable)
    norm = float(np.linalg.norm(symmetric[np.ix_(movable, movable)], 2))
    unit = _EIGEN_BACKWARD_ERROR_FACTOR * len(movable) * float(np.finfo(float).eps)
    tolerances = {
        "lambda_min": (lambda_min, unit * norm),
        "tau_H": (tau_h, unit * max(tau_h, _TAU_EIGEN_RTOL * norm)),
    }
    defects: list[str] = []
    for field, (expected, tolerance) in tolerances.items():
        value = recorded[field]
        if not _finite_number(value) or abs(float(value) - expected) > tolerance:
            defects.append(
                f"second_order.{field} is recorded as {value!r}, but the endpoint "
                f"evidence gives {expected!r} (consistency bound {tolerance:.3e})"
            )
    return defects


@dataclass(frozen=True, slots=True)
class EvidenceVerification:
    """``report`` recomputed from the endpoint evidence (None when the evidence has the
    wrong shape); ``defects`` lists every way the record disagrees with it."""

    report: TerminationReport | None
    defects: tuple[str, ...]


def _evidence_shape_defects(
    x: np.ndarray,
    gradient: np.ndarray,
    bounds: ScipyBounds | None,
    hessian: np.ndarray | None,
) -> list[str]:
    size = x.size
    if x.ndim != 1 or size < 1:
        return [f"x dimension {x.shape} is not a vector of at least one coordinate"]
    defects: list[str] = []
    if gradient.shape != (size,):
        defects.append(f"gradient dimension {gradient.shape} is not x's ({size},)")
    if bounds is not None and len(bounds.lower) != size:
        defects.append(f"bounds dimension {len(bounds.lower)} is not x's {size}")
    if hessian is not None and hessian.shape != (size, size):
        defects.append(f"Hessian dimension {hessian.shape} is not ({size}, {size})")
    return defects


def verify_termination_evidence(
    record: Mapping[str, object],
    *,
    x: np.ndarray,
    fun: float,
    gradient: np.ndarray,
    bounds: ScipyBounds | None,
    accepted_fun: Sequence[float],
    hessian: np.ndarray | None = None,
    hessian_exception: str | None = None,
) -> EvidenceVerification:
    """Recompute the report from the endpoint evidence and compare the record to it.

    The emitter's stop (emitter, status, message, success, restart reason) is taken from
    the record; everything the evidence decides is recomputed with the ORIGINAL
    ``bounds``: finiteness, feasibility, projected gradient, coordinate classes, the
    screen's verdict and cone, stagnation and the label. A recorded SADDLE witness must
    itself pass the witness test against the evidence. Only a record with no defects
    may be counted, and the recomputed report is the authority.
    """
    x, gradient = np.asarray(x, dtype=float), np.asarray(gradient, dtype=float)
    matrix = None if hessian is None else np.asarray(hessian, dtype=float)
    shape_defects = _evidence_shape_defects(x, gradient, bounds, matrix)
    if shape_defects:
        return EvidenceVerification(report=None, defects=tuple(shape_defects))
    schema_defects = termination_record_defects(record)
    if schema_defects:
        return EvidenceVerification(report=None, defects=schema_defects)
    restart = record["last_restart_reason"]
    status = record["status"]
    assert _integer(status)  # narrowing: the schema check rejected any other status
    report = termination_report(
        EmitterStop(
            emitter=Emitter(record["emitter"]),
            status=status,
            message=str(record["message"]),
            success=record["success"] is True,
            last_restart_reason=None
            if restart is None
            else LbfgsbRestartReason(restart),
        ),
        x=x,
        fun=float(fun),
        gradient=gradient,
        bounds=bounds,
        accepted_fun=accepted_fun,
        hessian=matrix,
        hessian_exception=hessian_exception,
    )
    expected = report.to_record()
    defects: list[str] = []
    for key, value in expected.items():
        recorded = record[key]
        if (
            key == "second_order"
            and isinstance(value, dict)
            and isinstance(recorded, dict)
        ):
            for field, field_value in value.items():
                if (
                    field not in _NUMERICAL_SCREEN_FIELDS
                    and recorded[field] != field_value
                ):
                    defects.append(
                        f"second_order.{field} is recorded as {recorded[field]!r}, "
                        f"but the endpoint evidence gives {field_value!r}"
                    )
        elif not _same_value(recorded, value):
            defects.append(
                f"{key} is recorded as {recorded!r}, but the endpoint evidence gives "
                f"{value!r}"
            )
    screen = report.second_order
    recorded_screen = record["second_order"]
    if (
        defects
        or matrix is None
        or screen is None
        or not isinstance(recorded_screen, dict)
    ):
        return EvidenceVerification(report=report, defects=tuple(defects))
    symmetric = 0.5 * (matrix + matrix.T)
    defects.extend(_eigen_descriptor_defects(recorded_screen, screen, symmetric))
    if screen.verdict is SecondOrderVerdict.SADDLE and isinstance(screen.tau_h, float):
        witness = np.asarray(recorded_screen["witness"], dtype=float)
        if not _witness_accepted(
            witness, symmetric, gradient, screen.classes, screen.tau_h
        ):
            defects.append(
                "second_order.witness fails the witness test at the endpoint evidence"
            )
    return EvidenceVerification(report=report, defects=tuple(defects))


def _same_value(recorded: object, expected: object) -> bool:
    """Equality that treats NaN as equal to NaN (a recorded non-finite value)."""
    if isinstance(recorded, float) and isinstance(expected, float):
        return recorded == expected or (math.isnan(recorded) and math.isnan(expected))
    return recorded == expected
