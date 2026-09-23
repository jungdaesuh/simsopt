"""Shipped-scale contract of the three tracing mirrors, derived from upstream's own one-ulp scatter.

Rule (pre-registered in ``.artifacts/official-mirror-closure-20260919/d1-diagnostic/NOTES.md``, section of
2026-09-20 06:54 EDT, fixed before any post-fix lane value was seen; it may not be re-tuned here).

At shipped scale a long trace amplifies rounding differences, so upstream itself does not reproduce its own final
states or every hit when the start data of the traced objects move by one unit in the last place.  The tracked record
``examples/jax/parity/official_reference/9e027eac3/tracing/<case_id>.json`` holds what upstream's own nine runs did.
A lane is therefore judged against UPSTREAM's canonical record, never against another lane, and per quantity:

* an item whose tracked maximum over ``k = 1..8`` is exactly ``0`` is EXACT: upstream reproduces it, so the lane must
  reproduce it too (terminal status on all three cases; the qa hit counts and the qa final time);
* every other item gets a GROSS-FAILURE ceiling ``CEILING_FACTOR * maximum``.  The factor is stated engineering
  headroom over a maximum of eight samples, not a probability statement (a lane is not an exchangeable draw from the
  seeded runs' distribution), and it is stated, not fitted to any lane value.

Nothing else is judged at this scale: hit positions row by row and the final states lane against lane are not compared
at all.  A route the manifest marks ``applicable: false`` is SKIPPED by the arbiter (``examples/jax/parity/arbiter.py``,
``arbitrate``): both lanes still publish those values, and the per-line distances to upstream's record below are
published beside them, but NO lane-against-lane number is computed for that key and none appears in the verdict
matrix.  The bounded scale keeps its tight lane-versus-lane routes.

Which lane-against-lane routes stay judged at this scale is ONE rule, owned here (:func:`lane_route_is_judged`) and
bound to ``examples/jax/parity_manifest.json`` by a test: a route of a final-state quantity is judged if and only if
upstream reproduces that quantity EXACTLY under one-ulp perturbations.  A bar upstream cannot meet against itself
cannot separate a faithful lane from an unfaithful one: measured on the guiding-centre case, upstream's own final time
moves by up to 1.9e-03 s and its final parallel-speed fraction by up to 7.6e-02 in every one of its eight samples,
against the bounded-scale bars ``2e-06 s`` and ``2e-03``.  On the ncsx field-line case the SAME rule switches off
``final:times``, and there the reason is consistency of the one rule, NOT a bar upstream cannot meet: upstream meets
that route's ``atol 2e-02 s`` against itself with a margin of about 1e8 (its own final-time scatter is
1.81e-10 s).  That case loses no coverage, because the far tighter vs-upstream ceiling of
``final_time_difference_max`` (3.6e-10 s) keeps judging the same quantity.

Every quantity of the final state has an item here, the guiding-centre parallel-speed fraction included
(``final_parallel_speed_fraction_difference_max``).  That item was added in fix wave 8, AFTER the first shipped-scale
lane values had been seen, because until then the quantity was judged by nothing at this scale; its FORM is the one
registered above and unchanged (``CEILING_FACTOR`` times upstream's own maximum over its eight one-ulp runs), so it
can only tighten what is judged, never loosen it.

Two rules of the three mirrors are NOT scale-dependent and live here because the three cases share them:

* :func:`non_finite_observables` — a tracing lane that publishes one non-finite number anywhere has not traced what
  upstream traced, whatever its provider reported, at EVERY scale.  ``nan > ceiling`` is ``False`` in IEEE
  arithmetic, so a ceiling alone certifies such a lane instead of failing it; :meth:`TracingItem.exceeded_by`
  therefore treats a non-finite value as breaking every item, exact or not;
* :func:`lane_status_reasons` — the lane reports EVERY reason it failed, not only the first one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

import numpy as np

from examples.jax.parity.official_reference import (
    MissingObservableError,
    load_official_reference,
    load_official_tracing_scatter,
)

#: Headroom over upstream's own maximum of eight samples. Stated by the contract, never fitted, never re-tuned.
CEILING_FACTOR: Final[int] = 2

#: The quantity of the guiding-centre final state that no position can stand for: the pitch, as the dimensionless
#: fraction ``v_parallel / v``.  A case is judged on it exactly when its tracked record holds the item, which is what
#: the loader's :data:`TRACING_SCATTER_CASE_QUANTITIES` means: the record, not a table here, says which case has it.
PARALLEL_SPEED_FRACTION_QUANTITY: Final[str] = (
    "final_parallel_speed_fraction_difference_max"
)

#: The observable holding that fraction, on a lane and in upstream's canonical record alike: ONE name for both.
FINAL_PARALLEL_SPEED_FRACTION_KEY: Final[str] = "final:parallel_speed_fraction"

#: The quantities the contract judges, and nothing else.  A case is judged on the members its tracked record holds
#: (:attr:`TracingContract.judged_quantities`), so this is one list for the three cases, not one list per case.
JUDGED_QUANTITIES: Final[tuple[str, ...]] = (
    "status_changes",
    "final_position_distance_max",
    "final_time_difference_max",
    "hit_count_difference_per_line_max_abs",
    PARALLEL_SPEED_FRACTION_QUANTITY,
)

#: The canonical observable holding upstream's final POSITION of every line: the guiding-centre case records the
#: parallel speed in ``final:states`` as well, and the contract measures the geometric distance here and the pitch
#: in its own item (:data:`PARALLEL_SPEED_FRACTION_QUANTITY`).
FINAL_POSITION_KEYS: Final[dict[str, str]] = {
    "native-tracing-fieldlines-ncsx": "final:states",
    "native-tracing-fieldlines-qa": "final:states",
    "native-tracing-particle": "final:positions",
}

#: Observables a lane publishes at ``native_default``: its distance to upstream's canonical record, line by line.
#: Keyed by quantity, never by case: a lane publishes the entry of every item its case's record holds.
PUBLISHED_DISTANCE_KEYS: Final[dict[str, str]] = {
    "final_position_distance_max": "final:upstream_position_distance",
    "final_time_difference_max": "final:upstream_time_difference",
    "status_changes": "final:upstream_status_difference",
    "hit_count_difference_per_line_max_abs": "poincare:upstream_count_difference",
    PARALLEL_SPEED_FRACTION_QUANTITY: "final:upstream_parallel_speed_fraction_difference",
}

#: Lane-against-lane route keys of the final state and of the Poincare hit counts at ``native_default``, and the
#: contract item that decides each.  Every key is decided by the item that MEASURES it: the guiding-centre case's
#: parallel-speed fraction has its own item and no longer follows the final position, whose metres could never
#: decide a pitch.  Hit positions row by row are reported on every case and are named by no key here.
LANE_ROUTE_DECIDING_ITEMS: Final[dict[str, dict[str, str]]] = {
    "native-tracing-fieldlines-ncsx": {
        "final:states": "final_position_distance_max",
        "final:times": "final_time_difference_max",
        "poincare:counts": "hit_count_difference_per_line_max_abs",
    },
    "native-tracing-fieldlines-qa": {
        "final:states": "final_position_distance_max",
        "final:times": "final_time_difference_max",
        "poincare:counts": "hit_count_difference_per_line_max_abs",
    },
    "native-tracing-particle": {
        "final:positions": "final_position_distance_max",
        FINAL_PARALLEL_SPEED_FRACTION_KEY: PARALLEL_SPEED_FRACTION_QUANTITY,
        "final:times": "final_time_difference_max",
        "poincare:counts": "hit_count_difference_per_line_max_abs",
    },
}


class TracingContractError(ValueError):
    """Raised when a lane cannot be compared with upstream's record at all."""


@dataclass(frozen=True)
class TracingItem:
    """One quantity of the contract, derived from the tracked record on every import."""

    quantity: str
    upstream_maximum: float
    exact: bool
    ceiling: float
    judged: bool
    derivation: str

    def exceeded_by(self, value: float) -> bool:
        """True when ``value`` breaks this item.

        A NON-FINITE value breaks every item, exact or not, and it is tested first: ``nan > ceiling`` is ``False``
        in IEEE arithmetic, so the plain comparison reports a lane whose distance to upstream cannot even be
        measured as being INSIDE the ceiling.  A gate that accepts ``nan`` is worse than no gate, because it
        certifies.  Then the tracked rule: any difference when upstream is exact, above the ceiling otherwise.
        """
        if not bool(np.isfinite(value)):
            return True
        if self.exact:
            return bool(value != 0)
        return bool(value > self.ceiling)


@dataclass(frozen=True)
class TracingContract:
    """The contract of one tracing case: every tracked quantity, with the judged ones marked."""

    case_id: str
    upstream_commit: str
    samples: int
    items: Mapping[str, TracingItem]

    def item(self, quantity: str) -> TracingItem:
        if quantity not in self.items:
            raise MissingObservableError(
                f"case {self.case_id!r} has no tracing contract item {quantity!r}; "
                f"available: {list(self.items)}"
            )
        return self.items[quantity]

    @property
    def judged_quantities(self) -> tuple[str, ...]:
        """The members of :data:`JUDGED_QUANTITIES` this case's tracked record holds, in that order.

        A quantity only some cases measure (the guiding-centre parallel-speed fraction) is judged on the cases whose
        record holds it and is absent elsewhere, so the judged set follows the RECORD and there is no second,
        per-case copy of the list.
        """
        return tuple(
            quantity for quantity in JUDGED_QUANTITIES if quantity in self.items
        )

    @property
    def judged_items(self) -> tuple[TracingItem, ...]:
        return tuple(self.items[quantity] for quantity in self.judged_quantities)

    @property
    def exact_quantities(self) -> tuple[str, ...]:
        return tuple(
            quantity
            for quantity in self.judged_quantities
            if self.items[quantity].exact
        )


def tracing_contract(case_id: str) -> TracingContract:
    """The ``native_default`` contract of one tracing case, computed from the tracked record."""
    scatter = load_official_tracing_scatter(case_id)
    maxima = scatter.maxima.as_mapping()
    perturbed = len(scatter.perturbed_runs)
    items = {}
    for quantity, maximum in maxima.items():
        exact = maximum == 0
        items[quantity] = TracingItem(
            quantity=quantity,
            upstream_maximum=maximum,
            exact=exact,
            ceiling=CEILING_FACTOR * maximum,
            judged=quantity in JUDGED_QUANTITIES,
            derivation=(
                f"upstream reproduces {quantity} exactly over its {perturbed} one-ulp runs "
                f"(tracked maximum {maximum!r})"
                if exact
                else f"{CEILING_FACTOR} x upstream's own maximum {maximum!r} of {quantity} over its {perturbed} "
                f"one-ulp runs (tracked tracing record of {case_id})"
            ),
        )
    return TracingContract(
        case_id=case_id,
        upstream_commit=scatter.upstream_commit,
        samples=perturbed,
        items=items,
    )


def lane_route_is_judged(case_id: str, route_key: str) -> bool:
    """Whether the ``native_default`` lane-against-lane route ``route_key`` is judged: only where upstream is exact."""
    return (
        tracing_contract(case_id)
        .item(LANE_ROUTE_DECIDING_ITEMS[case_id][route_key])
        .exact
    )


def judges_parallel_speed_fraction(case_id: str) -> bool:
    """Whether ``case_id`` is judged on the parallel-speed fraction: exactly when its tracked record measures it."""
    return PARALLEL_SPEED_FRACTION_QUANTITY in tracing_contract(case_id).items


@dataclass(frozen=True)
class UpstreamTrace:
    """Upstream's canonical shipped-scale end state of one tracing case, read from the tracked fixture."""

    case_id: str
    initial_states: np.ndarray
    final_positions: np.ndarray
    final_times: np.ndarray
    final_statuses: np.ndarray
    poincare_counts: np.ndarray
    final_parallel_speed_fractions: np.ndarray | None

    @property
    def lines(self) -> int:
        return int(self.final_times.size)


def upstream_trace(case_id: str) -> UpstreamTrace:
    """The canonical final states, times, terminal statuses and hit counts upstream recorded for ``case_id``.

    The parallel-speed fraction is read for the cases judged on it and is ``None`` for the field-line cases, whose
    traced object has no parallel speed; which cases those are is decided by the tracked scatter record alone.
    """
    reference = load_official_reference(case_id)
    return UpstreamTrace(
        case_id=case_id,
        initial_states=np.asarray(reference.array("initial:states"), dtype=np.float64),
        final_positions=np.asarray(
            reference.array(FINAL_POSITION_KEYS[case_id]), dtype=np.float64
        ),
        final_times=np.asarray(reference.array("final:times"), dtype=np.float64),
        final_statuses=np.rint(reference.array("final:status")).astype(np.int64),
        poincare_counts=np.rint(reference.array("poincare:counts")).astype(np.int64),
        final_parallel_speed_fractions=(
            np.asarray(
                reference.array(FINAL_PARALLEL_SPEED_FRACTION_KEY), dtype=np.float64
            )
            if judges_parallel_speed_fraction(case_id)
            else None
        ),
    )


@dataclass(frozen=True)
class LaneDistances:
    """One lane's distance to upstream's canonical record, line by line."""

    case_id: str
    position_distance: np.ndarray
    time_difference: np.ndarray
    status_difference: np.ndarray
    count_difference: np.ndarray
    parallel_speed_fraction_difference: np.ndarray | None

    def _per_line(self) -> dict[str, np.ndarray]:
        """Every distance this lane measured, keyed by the contract quantity it feeds."""
        distances = {
            "final_position_distance_max": self.position_distance,
            "final_time_difference_max": self.time_difference,
            "status_changes": self.status_difference,
            "hit_count_difference_per_line_max_abs": self.count_difference,
        }
        if self.parallel_speed_fraction_difference is not None:
            distances[PARALLEL_SPEED_FRACTION_QUANTITY] = (
                self.parallel_speed_fraction_difference
            )
        return distances

    def maxima(self) -> dict[str, float]:
        """The judged quantity of each distance this lane measured, in :data:`JUDGED_QUANTITIES` order."""
        maxima: dict[str, float] = {
            "status_changes": int(np.count_nonzero(self.status_difference)),
            "final_position_distance_max": float(np.max(self.position_distance)),
            "final_time_difference_max": float(np.max(self.time_difference)),
            "hit_count_difference_per_line_max_abs": int(
                np.max(np.abs(self.count_difference))
            ),
        }
        if self.parallel_speed_fraction_difference is not None:
            maxima[PARALLEL_SPEED_FRACTION_QUANTITY] = float(
                np.max(self.parallel_speed_fraction_difference)
            )
        return maxima

    def published_values(self) -> dict[str, np.ndarray]:
        """The per-line arrays a lane publishes at ``native_default``, keyed by observable."""
        return {
            PUBLISHED_DISTANCE_KEYS[quantity]: values
            for quantity, values in self._per_line().items()
        }


def _require_shape(
    case_id: str, name: str, value: np.ndarray, expected: tuple[int, ...]
) -> None:
    if value.shape != expected:
        raise TracingContractError(
            f"{case_id}: lane {name} has shape {value.shape}, upstream's canonical record has {expected}; "
            f"the two runs did not trace the same objects"
        )


def _require_line_identity(
    case_id: str, lane_initial: np.ndarray, upstream_initial: np.ndarray
) -> None:
    """The ORDERING INVARIANT of :func:`lane_distances`, asserted on the traced objects' start states.

    Row ``i`` of the lane is compared with row ``i`` of upstream's canonical record, so the two runs must return
    their objects in the SAME order; a shape check alone cannot see a reordering, and two lines with nearly equal
    end states would swap silently.  The start states are the identity of a traced object -- both runs are handed
    the same ones -- so the invariant is: **row i of the lane is strictly closer to upstream's row i than to any
    other upstream row**.  Upstream's own start states are separated by 4.8e-3 (ncsx), 9.2e-3 (qa) and 43.8
    (particle), so a faithful lane satisfies it with an enormous margin and only a reordering breaks it.

    A lane whose start states are not finite cannot be placed at all; it is already failed by
    :func:`non_finite_observables` with its own named reason and must keep publishing its distance arrays for the
    arbiter's route matrix, so the pairing is asserted only where it can be decided.
    """
    if not bool(np.all(np.isfinite(lane_initial))):
        return
    deltas = np.linalg.norm(
        lane_initial[:, None, :] - upstream_initial[None, :, :], axis=2
    )
    paired = np.diagonal(deltas)
    nearest_other = np.min(deltas + np.diag(np.full(deltas.shape[0], np.inf)), axis=1)
    mismatched = np.flatnonzero(paired >= nearest_other)
    if mismatched.size:
        line = int(mismatched[0])
        raise TracingContractError(
            f"{case_id}: lane line {line} starts closer to a DIFFERENT line of upstream's canonical record "
            f"({float(nearest_other[line])!r}) than to the line it is paired with "
            f"({float(paired[line])!r}); the two runs did not trace the same objects in the same order"
        )


def _parallel_speed_fraction_difference(
    case_id: str,
    upstream: UpstreamTrace,
    parallel_speed_fractions: np.ndarray | None,
) -> np.ndarray | None:
    """The per-line pitch distance to upstream, or ``None`` for a case that has no parallel speed.

    A lane must supply the fraction exactly when its case is judged on it: a guiding-centre lane that stopped
    publishing the quantity would otherwise drop a judged item silently, and a field-line lane that supplied one
    would be measuring something its record cannot bound.
    """
    if parallel_speed_fractions is None:
        if upstream.final_parallel_speed_fractions is not None:
            raise TracingContractError(
                f"{case_id}: the tracked record judges {PARALLEL_SPEED_FRACTION_QUANTITY}, but the lane published "
                f"no {FINAL_PARALLEL_SPEED_FRACTION_KEY!r}"
            )
        return None
    if upstream.final_parallel_speed_fractions is None:
        raise TracingContractError(
            f"{case_id}: the lane published {FINAL_PARALLEL_SPEED_FRACTION_KEY!r}, but upstream's record for this "
            f"case holds no such quantity to compare it with"
        )
    fractions = np.asarray(parallel_speed_fractions, dtype=np.float64)
    _require_shape(
        case_id,
        "final parallel-speed fractions",
        fractions,
        upstream.final_parallel_speed_fractions.shape,
    )
    return np.abs(fractions - upstream.final_parallel_speed_fractions)


def lane_distances(
    case_id: str,
    *,
    initial_states: np.ndarray,
    final_positions: np.ndarray,
    final_times: np.ndarray,
    final_statuses: np.ndarray,
    poincare_counts: np.ndarray,
    parallel_speed_fractions: np.ndarray | None = None,
) -> LaneDistances:
    """Distance of one lane's shipped-scale end state to upstream's canonical record, line by line."""
    upstream = upstream_trace(case_id)
    initial = np.asarray(initial_states, dtype=np.float64)
    positions = np.asarray(final_positions, dtype=np.float64)
    times = np.asarray(final_times, dtype=np.float64)
    statuses = np.rint(np.asarray(final_statuses)).astype(np.int64)
    counts = np.rint(np.asarray(poincare_counts)).astype(np.int64)
    fraction_difference = _parallel_speed_fraction_difference(
        case_id, upstream, parallel_speed_fractions
    )
    _require_shape(case_id, "initial states", initial, upstream.initial_states.shape)
    _require_shape(
        case_id, "final positions", positions, upstream.final_positions.shape
    )
    _require_shape(case_id, "final times", times, upstream.final_times.shape)
    _require_shape(
        case_id, "terminal statuses", statuses, upstream.final_statuses.shape
    )
    _require_shape(case_id, "hit counts", counts, upstream.poincare_counts.shape)
    _require_line_identity(case_id, initial, upstream.initial_states)
    return LaneDistances(
        case_id=case_id,
        position_distance=np.linalg.norm(
            positions - upstream.final_positions, axis=1
        ).astype(np.float64),
        time_difference=np.abs(times - upstream.final_times),
        status_difference=statuses - upstream.final_statuses,
        count_difference=counts - upstream.poincare_counts,
        parallel_speed_fraction_difference=fraction_difference,
    )


def _violation_text(item: TracingItem, value: float) -> str:
    """How ``value`` broke ``item``: a non-finite value is named as such, never as "above a bound"."""
    if not bool(np.isfinite(value)):
        return f"{item.quantity}={value!r} is not finite"
    if item.exact:
        return f"{item.quantity}={value!r}>0 (upstream is exact)"
    return f"{item.quantity}={value!r}>{item.ceiling!r}"


def contract_violations(distances: LaneDistances) -> tuple[str, ...]:
    """Every judged quantity of ``distances`` that breaks the contract, named with its value and its bound."""
    contract = tracing_contract(distances.case_id)
    measured = distances.maxima()
    return tuple(
        _violation_text(contract.item(quantity), measured[quantity])
        for quantity in contract.judged_quantities
        if contract.item(quantity).exceeded_by(measured[quantity])
    )


def non_finite_observables(values: Mapping[str, np.ndarray]) -> tuple[str, ...]:
    """Every observable of a tracing lane that holds a non-finite entry, named, in a stable order.

    Checked on the FULL published array of EVERY key the lane publishes, at every scale, before any selection that
    could drop the bad row.  A tracing lane publishes positions, times, terminal statuses and hit counts; a
    non-finite entry in any of them means the integration did not produce what upstream produced, whatever the
    provider reported, so such a lane is never ``converged``.  This is the every-scale half of the contract: the
    ceilings above judge a lane that DID produce numbers.
    """
    return tuple(
        key
        for key, value in sorted(values.items())
        if not bool(np.all(np.isfinite(np.asarray(value, dtype=np.float64))))
    )


def lane_status_reasons(
    non_finite: tuple[str, ...],
    violations: tuple[str, ...],
    terminal_statuses: np.ndarray,
) -> str:
    """The ``raw_status`` of a tracing lane: EVERY reason it has, in one string.

    The three cases used to report the first reason only, so a lane that both broke the contract and reported a
    failed integration published the contract violation and dropped the provider's own failure.  ``normalized_status``
    is ``failed`` either way, but the diagnostic that says WHICH things went wrong is the point of the raw status.
    """
    reasons = tuple(f"non_finite_observable:{key}" for key in non_finite) + tuple(
        f"upstream_contract_violation:{violation}" for violation in violations
    )
    if not bool(np.all(np.asarray(terminal_statuses) <= 0)):
        reasons += ("integration_incomplete_or_failed",)
    if reasons:
        return ";".join(reasons)
    return "integration_complete_or_levelset_stop"


TRACING_CONTRACT_CASE_IDS: Final[tuple[str, ...]] = tuple(sorted(FINAL_POSITION_KEYS))
