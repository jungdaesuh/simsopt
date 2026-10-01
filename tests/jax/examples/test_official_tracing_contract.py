"""The shipped-scale tracing contract is exactly the one fixed before any lane value was seen, and it can fail.

The expected ceilings below are the numbers that rule fixed (see ``examples/jax/parity/official_tracing_contract.py``),
written here as literals on purpose: the module must derive them from the tracked record, so a changed rule, a changed
factor or an edited record fails here instead of silently widening the shipped-scale verdict.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json
from pathlib import Path

import numpy as np
import pytest

from examples.jax.parity.official_reference import (
    TRACING_SCATTER_CASE_QUANTITIES,
    TRACING_SCATTER_QUANTITIES,
    MissingObservableError,
    load_official_reference,
    load_official_tracing_scatter,
)
from examples.jax.parity.official_tracing_contract import (
    CEILING_FACTOR,
    FINAL_POSITION_KEYS,
    JUDGED_QUANTITIES,
    LANE_ROUTE_DECIDING_ITEMS,
    PUBLISHED_DISTANCE_KEYS,
    TRACING_CONTRACT_CASE_IDS,
    LaneDistances,
    TracingContractError,
    contract_violations,
    lane_distances,
    lane_route_is_judged,
    lane_status_reasons,
    non_finite_observables,
    tracing_contract,
    upstream_trace,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_MANIFEST = REPO_ROOT / "examples" / "jax" / "parity_manifest.json"
LANE_PAIRS = frozenset({"native-cpu:jax-cpu", "native-cpu:jax-gpu", "jax-cpu:jax-gpu"})

#: case_id -> quantity -> the ceiling the notes state, or None where the notes say upstream is exact.
#: The guiding-centre parallel-speed fraction was added in fix wave 8 in the registered form, from upstream's own
#: maximum over its eight one-ulp runs (7.635197511514785e-02, the tracked record's ``maxima_over_k``).
EXPECTED = {
    "native-tracing-fieldlines-ncsx": {
        "status_changes": None,
        "final_position_distance_max": 2 * 0.26748080419886355,
        "final_time_difference_max": 2 * 1.80810921790453e-10,
        "hit_count_difference_per_line_max_abs": 2,
    },
    "native-tracing-fieldlines-qa": {
        "status_changes": None,
        "final_position_distance_max": 2 * 6.6268598728074056e-09,
        "final_time_difference_max": None,
        "hit_count_difference_per_line_max_abs": None,
    },
    "native-tracing-particle": {
        "status_changes": None,
        "final_position_distance_max": 2 * 2.92717370827105,
        "final_time_difference_max": 2 * 0.0019276353087928984,
        "hit_count_difference_per_line_max_abs": 238,
        "final_parallel_speed_fraction_difference_max": 2 * 0.07635197511514785,
    },
}

#: Which case measures a quantity the others do not. Literal on purpose: a field line has no parallel speed, and the
#: test must SAY which case holds the item instead of reading the answer back from the loader.
CASE_SPECIFIC_QUANTITIES = {
    "native-tracing-fieldlines-ncsx": (),
    "native-tracing-fieldlines-qa": (),
    "native-tracing-particle": ("final_parallel_speed_fraction_difference_max",),
}

#: Which contract item must decide each lane-against-lane route key. Literal, so a route handed to the WRONG item
#: fails here even when the wrong item happens to give the same applicability -- the defect review wave 6-7 found
#: (`w7-vpar-proxy`: the dimensionless pitch was decided by a distance in metres).
EXPECTED_DECIDING_ITEMS = {
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
        "final:parallel_speed_fraction": "final_parallel_speed_fraction_difference_max",
        "final:times": "final_time_difference_max",
        "poincare:counts": "hit_count_difference_per_line_max_abs",
    },
}

#: The only case whose ``native_default`` routes carry the level-set diagnostic.
CASES_WITH_LEVELSET_DISTANCE = {"native-tracing-fieldlines-qa"}

#: Every (case, judged quantity) pair, for the legs that must cover each item of each case exactly once.
CASE_QUANTITIES = tuple(
    (case_id, quantity)
    for case_id in sorted(EXPECTED)
    for quantity in EXPECTED[case_id]
)


def test_the_contract_covers_the_three_tracing_cases_and_judges_five_quantities() -> (
    None
):
    assert TRACING_CONTRACT_CASE_IDS == tuple(sorted(EXPECTED))
    assert tuple(sorted(FINAL_POSITION_KEYS)) == TRACING_CONTRACT_CASE_IDS
    assert JUDGED_QUANTITIES == (
        "status_changes",
        "final_position_distance_max",
        "final_time_difference_max",
        "hit_count_difference_per_line_max_abs",
        "final_parallel_speed_fraction_difference_max",
    )
    assert set(JUDGED_QUANTITIES) <= set(TRACING_SCATTER_QUANTITIES) | set(
        TRACING_SCATTER_CASE_QUANTITIES
    )
    assert set(PUBLISHED_DISTANCE_KEYS) == set(JUDGED_QUANTITIES)
    assert CEILING_FACTOR == 2
    # One list of judged quantities for the three cases: a case judges the members its RECORD holds.
    assert tuple(sorted(CASE_SPECIFIC_QUANTITIES)) == TRACING_CONTRACT_CASE_IDS
    for case_id, extra in CASE_SPECIFIC_QUANTITIES.items():
        contract = tracing_contract(case_id)
        assert contract.judged_quantities == tuple(EXPECTED[case_id]), case_id
        assert set(extra) <= set(TRACING_SCATTER_CASE_QUANTITIES), case_id
        assert set(contract.items) == set(TRACING_SCATTER_QUANTITIES) | set(extra), (
            case_id
        )


def test_every_ceiling_is_twice_upstreams_own_maximum_from_the_tracked_record() -> None:
    for case_id, expected in EXPECTED.items():
        contract = tracing_contract(case_id)
        scatter = load_official_tracing_scatter(case_id)
        assert contract.samples == 8
        assert contract.upstream_commit == scatter.upstream_commit
        assert set(contract.items) == set(TRACING_SCATTER_QUANTITIES) | set(
            CASE_SPECIFIC_QUANTITIES[case_id]
        )
        for quantity, ceiling in expected.items():
            item = contract.item(quantity)
            assert item.judged is True
            assert item.upstream_maximum == scatter.maxima.quantity(quantity)
            if ceiling is None:
                assert item.exact is True
                assert item.upstream_maximum == 0
                assert item.ceiling == 0
                assert item.exceeded_by(0) is False
                assert item.exceeded_by(1) is True
                assert "exactly" in item.derivation
            else:
                assert item.exact is False
                assert item.ceiling == ceiling
                assert item.ceiling == CEILING_FACTOR * item.upstream_maximum
                assert item.exceeded_by(ceiling) is False
                assert item.exceeded_by(np.nextafter(ceiling, np.inf)) is True
                assert repr(item.upstream_maximum) in item.derivation
        assert contract.exact_quantities == tuple(
            quantity for quantity, ceiling in expected.items() if ceiling is None
        )


def test_the_unjudged_quantities_are_tracked_but_not_judged() -> None:
    for case_id in TRACING_CONTRACT_CASE_IDS:
        contract = tracing_contract(case_id)
        unjudged = set(contract.items) - set(JUDGED_QUANTITIES)
        assert unjudged == {
            "final_state_distance_max",
            "hit_count_difference_per_plane_max_abs",
            "geometry_max_over_groups",
            "geometry_max_of_group_medians",
        }
        for quantity in unjudged:
            assert contract.item(quantity).judged is False


def test_an_unknown_contract_item_is_named_with_the_available_ones() -> None:
    contract = tracing_contract(TRACING_CONTRACT_CASE_IDS[0])
    with pytest.raises(MissingObservableError) as raised:
        contract.item("final_position_distance_p95")
    assert "final_position_distance_p95" in str(raised.value)


def test_upstream_measured_against_itself_is_inside_the_contract() -> None:
    for case_id in TRACING_CONTRACT_CASE_IDS:
        upstream = upstream_trace(case_id)
        distances = lane_distances(
            case_id,
            initial_states=upstream.initial_states,
            final_positions=upstream.final_positions,
            final_times=upstream.final_times,
            final_statuses=upstream.final_statuses,
            poincare_counts=upstream.poincare_counts,
            parallel_speed_fractions=upstream.final_parallel_speed_fractions,
        )
        # Upstream against itself is zero in EVERY quantity the case measures -- including the parallel-speed
        # fraction, which the guiding-centre case has and the field-line cases have not.
        assert distances.maxima() == {quantity: 0 for quantity in EXPECTED[case_id]}
        assert contract_violations(distances) == ()
        published = distances.published_values()
        assert set(published) == {
            PUBLISHED_DISTANCE_KEYS[quantity] for quantity in EXPECTED[case_id]
        }
        for values in published.values():
            assert values.shape == (upstream.lines,)


def _lane_outside(case_id: str, quantity: str) -> LaneDistances:
    """Upstream's own end state with ONE line moved past the bound of ``quantity``, and nothing else."""
    upstream = upstream_trace(case_id)
    item = tracing_contract(case_id).item(quantity)
    positions = upstream.final_positions.copy()
    times = upstream.final_times.copy()
    statuses = upstream.final_statuses.copy()
    counts = upstream.poincare_counts.copy()
    fractions = (
        None
        if upstream.final_parallel_speed_fractions is None
        else upstream.final_parallel_speed_fractions.copy()
    )
    if quantity == "final_position_distance_max":
        positions[0, 0] += 2.0 * item.ceiling
    elif quantity == "final_time_difference_max":
        times[0] = (
            np.nextafter(times[0], np.inf)
            if item.exact
            else times[0] + 2.0 * item.ceiling
        )
    elif quantity == "hit_count_difference_per_line_max_abs":
        counts[0] += int(item.ceiling) + 1
    elif quantity == "final_parallel_speed_fraction_difference_max":
        fractions[0] = (
            np.nextafter(fractions[0], np.inf)
            if item.exact
            else fractions[0] + 2.0 * item.ceiling
        )
    else:
        statuses[0] = statuses[0] - 1
    return lane_distances(
        case_id,
        initial_states=upstream.initial_states,
        final_positions=positions,
        final_times=times,
        final_statuses=statuses,
        poincare_counts=counts,
        parallel_speed_fractions=fractions,
    )


@pytest.mark.parametrize(("case_id", "quantity"), CASE_QUANTITIES)
def test_a_lane_outside_one_bound_is_a_named_violation(
    case_id: str, quantity: str
) -> None:
    item = tracing_contract(case_id).item(quantity)
    distances = _lane_outside(case_id, quantity)
    measured = distances.maxima()[quantity]
    assert item.exceeded_by(measured), (quantity, measured, item.ceiling)
    violations = contract_violations(distances)
    assert len(violations) == 1, violations
    assert violations[0].startswith(f"{quantity}=")
    assert repr(measured) in violations[0]


@pytest.mark.parametrize("case_id", TRACING_CONTRACT_CASE_IDS)
def test_a_lane_inside_every_bound_passes(case_id: str) -> None:
    """Every perturbation here must be REAL, or the leg proves nothing.

    Scaling a departure by the ceiling is a no-op wherever upstream is exact (the ceiling is then ``0``), which
    would assert ``converged`` on an unperturbed copy of upstream's own record and could not tell a working gate
    from one that never fires. So a non-exact item is moved by half its ceiling and the move is asserted to have
    changed the array; an exact item has no room inside by construction, and that is asserted instead -- its
    smallest representable departure is already a violation.
    """
    upstream = upstream_trace(case_id)
    contract = tracing_contract(case_id)
    positions = upstream.final_positions.copy()
    times = upstream.final_times.copy()
    counts = upstream.poincare_counts.copy()
    perturbed: list[str] = []
    if contract.item("final_position_distance_max").exact:
        assert contract.item("final_position_distance_max").ceiling == 0
    else:
        positions[0, 0] += 0.5 * contract.item("final_position_distance_max").ceiling
        assert not np.array_equal(positions, upstream.final_positions)
        perturbed.append("final_position_distance_max")
    if contract.item("final_time_difference_max").exact:
        assert contract.item("final_time_difference_max").ceiling == 0
        assert contract.item("final_time_difference_max").exceeded_by(
            float(np.nextafter(0.0, 1.0))
        )
    else:
        times[0] += 0.5 * contract.item("final_time_difference_max").ceiling
        assert not np.array_equal(times, upstream.final_times)
        perturbed.append("final_time_difference_max")
    counts_item = contract.item("hit_count_difference_per_line_max_abs")
    if counts_item.exact:
        assert counts_item.ceiling == 0
        assert counts_item.exceeded_by(1)
    else:
        counts[0] += int(counts_item.ceiling) // 2
        assert not np.array_equal(counts, upstream.poincare_counts)
        perturbed.append("hit_count_difference_per_line_max_abs")
    fractions = (
        None
        if upstream.final_parallel_speed_fractions is None
        else upstream.final_parallel_speed_fractions.copy()
    )
    if fractions is not None:
        fraction_item = contract.item("final_parallel_speed_fraction_difference_max")
        assert fraction_item.exact is False
        fractions[0] += 0.5 * fraction_item.ceiling
        assert not np.array_equal(fractions, upstream.final_parallel_speed_fractions)
        perturbed.append("final_parallel_speed_fraction_difference_max")
    # Upstream is never exact in every judged quantity, so this leg always moves something.
    assert perturbed
    distances = lane_distances(
        case_id,
        initial_states=upstream.initial_states,
        final_positions=positions,
        final_times=times,
        final_statuses=upstream.final_statuses,
        poincare_counts=counts,
        parallel_speed_fractions=fractions,
    )
    measured = distances.maxima()
    for quantity in perturbed:
        assert measured[quantity] > 0, quantity
    for quantity in contract.judged_quantities:
        assert not contract.item(quantity).exceeded_by(measured[quantity]), quantity
    assert contract_violations(distances) == ()


def test_a_lane_that_traced_a_different_number_of_lines_cannot_be_compared() -> None:
    case_id = TRACING_CONTRACT_CASE_IDS[0]
    upstream = upstream_trace(case_id)
    with pytest.raises(TracingContractError) as raised:
        lane_distances(
            case_id,
            initial_states=upstream.initial_states[:-1],
            final_positions=upstream.final_positions[:-1],
            final_times=upstream.final_times[:-1],
            final_statuses=upstream.final_statuses[:-1],
            poincare_counts=upstream.poincare_counts[:-1],
        )
    assert "did not trace the same objects" in str(raised.value)


def test_the_parallel_speed_fraction_is_recorded_exactly_where_upstream_publishes_one() -> (
    None
):
    """The tracked record decides which case is judged on the pitch; the canonical record must agree with it.

    The scatter record and the canonical record are separate files, so a case could gain the scatter quantity
    without the canonical array its lanes are measured against, or the reverse. That would surface as a broken
    shipped-scale run hours later instead of here.
    """
    for case_id in TRACING_CONTRACT_CASE_IDS:
        expected = bool(CASE_SPECIFIC_QUANTITIES[case_id])
        contract = tracing_contract(case_id)
        assert ("final_parallel_speed_fraction_difference_max" in contract.items) is (
            expected
        ), case_id
        assert (
            "final:parallel_speed_fraction" in load_official_reference(case_id).keys()
        ) is expected, case_id
        assert (
            upstream_trace(case_id).final_parallel_speed_fractions is not None
        ) is expected, case_id


def test_a_lane_publishes_the_parallel_speed_fraction_exactly_where_it_is_judged_on_it() -> (
    None
):
    """Neither half may drift: the quantity is judged if and only if the lane hands it over.

    A guiding-centre lane that stopped publishing the fraction would otherwise drop a judged item in silence, and
    a field-line lane that published one would be measuring something its record cannot bound.
    """
    particle = "native-tracing-particle"
    upstream = upstream_trace(particle)
    with pytest.raises(TracingContractError) as raised:
        lane_distances(
            particle,
            initial_states=upstream.initial_states,
            final_positions=upstream.final_positions,
            final_times=upstream.final_times,
            final_statuses=upstream.final_statuses,
            poincare_counts=upstream.poincare_counts,
        )
    assert "final_parallel_speed_fraction_difference_max" in str(raised.value)

    field_line = "native-tracing-fieldlines-ncsx"
    other = upstream_trace(field_line)
    with pytest.raises(TracingContractError) as raised:
        lane_distances(
            field_line,
            initial_states=other.initial_states,
            final_positions=other.final_positions,
            final_times=other.final_times,
            final_statuses=other.final_statuses,
            poincare_counts=other.poincare_counts,
            parallel_speed_fractions=np.zeros(other.lines),
        )
    assert "no such quantity" in str(raised.value)


def _native_default_routes(case_id: str) -> list[dict[str, object]]:
    manifest = json.loads(PARITY_MANIFEST.read_text(encoding="utf-8"))
    for relationship in manifest["relationships"]:
        if relationship["case_id"] == case_id:
            return list(
                relationship["scale_contracts"]["native_default"]["comparison_routes"]
            )
    raise AssertionError(f"{case_id} is not in the parity manifest")


def _grouped(routes: list[dict[str, object]]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for route in routes:
        grouped.setdefault((str(route["phase"]), str(route["observable"])), []).append(
            route
        )
    return grouped


@pytest.mark.parametrize("case_id", TRACING_CONTRACT_CASE_IDS)
def test_every_published_distance_key_has_a_complete_native_default_route(
    case_id: str,
) -> None:
    """A lane publishes one distance array per item its record holds, and the arbiter needs a route group for each.

    The group is never applicable -- a lane's distance to UPSTREAM is not compared with another lane's -- but
    ``_validate_route_matrix`` requires it to exist and to be complete for every published observable, so a missing
    group fails the whole case at the runner.
    """
    grouped = _grouped(_native_default_routes(case_id))
    expected_keys = {
        PUBLISHED_DISTANCE_KEYS[quantity] for quantity in EXPECTED[case_id]
    }
    assert expected_keys <= set(PUBLISHED_DISTANCE_KEYS.values())
    for key in expected_keys:
        phase, observable = key.split(":", maxsplit=1)
        routes = grouped[(phase, observable)]
        assert {str(route["lane_pair"]) for route in routes} == LANE_PAIRS, key
        assert all(route["applicable"] is False for route in routes), key
    # A case must NOT carry a distance route for an item its record does not hold.
    absent = set(PUBLISHED_DISTANCE_KEYS.values()) - expected_keys
    for key in absent:
        phase, observable = key.split(":", maxsplit=1)
        assert (phase, observable) not in grouped, (case_id, key)


@pytest.mark.parametrize("case_id", TRACING_CONTRACT_CASE_IDS)
def test_scatter_dominated_observables_are_no_longer_judged_lane_against_lane(
    case_id: str,
) -> None:
    """ONE rule for every final-state route: judged lane against lane iff upstream reproduces the quantity exactly.

    The rule's owner is ``lane_route_is_judged``; the manifest must agree route group by route group. Before this
    test covered every final-state key, the guiding-centre case kept ``final:times`` and
    ``final:parallel_speed_fraction`` judged with their bounded-scale bars, which upstream fails against itself in
    eight of eight one-ulp samples.
    """
    grouped = _grouped(_native_default_routes(case_id))
    assert FINAL_POSITION_KEYS[case_id] in LANE_ROUTE_DECIDING_ITEMS[case_id]
    # The owner's map itself, against this file's literal table: a route handed to the wrong contract item fails
    # here even when that item happens to give the same applicability.
    assert LANE_ROUTE_DECIDING_ITEMS[case_id] == EXPECTED_DECIDING_ITEMS[case_id]
    for route_key, deciding_item in EXPECTED_DECIDING_ITEMS[case_id].items():
        phase, observable = route_key.split(":", maxsplit=1)
        routes = grouped[(phase, observable)]
        # ``EXPECTED`` holds the notes' ceiling, and ``None`` exactly where upstream reproduces the quantity, so
        # this expectation comes from the tracked NUMBERS and not from the module under test.
        expected_applicable = EXPECTED[case_id][deciding_item] is None
        assert {str(route["lane_pair"]) for route in routes} == LANE_PAIRS, route_key
        assert {bool(route["applicable"]) for route in routes} == {
            expected_applicable
        }, (case_id, route_key)
        # and the owner must agree with the same table.
        assert lane_route_is_judged(case_id, route_key) is expected_applicable, (
            case_id,
            route_key,
        )
    # Hit positions row by row are reported on every case, whatever upstream reproduces.
    assert all(
        route["applicable"] is False for route in grouped[("poincare", "positions")]
    ), case_id
    assert all(
        route["applicable"] is True and route["comparator"] == "exact"
        for route in grouped[("final", "status")]
    ), case_id
    # qa's level-set distance is a diagnostic that is judged at NO scale, and it is the only case that has one.
    has_levelset = ("final", "levelset_distance") in grouped
    assert has_levelset is (case_id in CASES_WITH_LEVELSET_DISTANCE), case_id
    if has_levelset:
        levelset_routes = grouped[("final", "levelset_distance")]
        assert {str(route["lane_pair"]) for route in levelset_routes} == LANE_PAIRS, (
            case_id
        )
        assert all(route["applicable"] is False for route in levelset_routes), case_id
    # No final-state key of the case escapes the rule: every judged-or-reported final route is named by the owner.
    # (``final:status`` is exact by item 1 of the contract; qa's ``final:levelset_distance`` is asserted just above;
    # the ``upstream_*`` keys are the lane's own distances to upstream's record.)
    final_keys = {
        f"{phase}:{observable}"
        for phase, observable in grouped
        if phase == "final"
        and observable not in {"status", "levelset_distance"}
        and f"{phase}:{observable}" not in PUBLISHED_DISTANCE_KEYS.values()
    }
    assert final_keys == {
        key for key in LANE_ROUTE_DECIDING_ITEMS[case_id] if key.startswith("final:")
    }, case_id


@pytest.mark.parametrize("case_id", TRACING_CONTRACT_CASE_IDS)
def test_the_bounded_routes_do_not_mention_the_shipped_scale_keys(case_id: str) -> None:
    manifest = json.loads(PARITY_MANIFEST.read_text(encoding="utf-8"))
    relationship = next(
        record for record in manifest["relationships"] if record["case_id"] == case_id
    )
    keys = {
        f"{route['phase']}:{route['observable']}"
        for route in relationship["comparison_routes"]
    }
    assert keys & set(PUBLISHED_DISTANCE_KEYS.values()) == set()


# --- the finiteness half of the contract ---------------------------------------------------------------------------


@pytest.mark.parametrize(("case_id", "quantity"), CASE_QUANTITIES)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_a_non_finite_value_breaks_every_item_exact_or_not(
    case_id: str, quantity: str, value: float
) -> None:
    """``nan > ceiling`` is ``False``, so a bare comparison certifies a lane that produced no usable number at all.

    Every item of every case, exact or not, and every non-finite value: this is the rule, not a special case of the
    ceiling.
    """
    item = tracing_contract(case_id).item(quantity)
    assert item.exceeded_by(value) is True
    # The bound itself is untouched: a finite value is still judged by the tracked rule.
    assert item.exceeded_by(item.ceiling) is False


@pytest.mark.parametrize("case_id", TRACING_CONTRACT_CASE_IDS)
def test_a_lane_with_a_non_finite_end_state_is_a_named_violation(case_id: str) -> None:
    """The measured case: one NaN final position and one NaN final time on an otherwise perfect lane."""
    upstream = upstream_trace(case_id)
    positions = upstream.final_positions.astype(np.float64).copy()
    times = upstream.final_times.copy()
    positions[0, 0] = np.nan
    times[0] = np.nan
    distances = lane_distances(
        case_id,
        initial_states=upstream.initial_states,
        final_positions=positions,
        final_times=times,
        final_statuses=upstream.final_statuses,
        poincare_counts=upstream.poincare_counts,
        parallel_speed_fractions=upstream.final_parallel_speed_fractions,
    )
    violations = contract_violations(distances)
    assert len(violations) == 2, violations
    assert violations[0] == "final_position_distance_max=nan is not finite"
    assert violations[1] == "final_time_difference_max=nan is not finite"


def test_non_finite_observables_names_every_bad_array_on_the_full_array() -> None:
    """Checked on the WHOLE array, so a non-finite entry anywhere is found, and every bad key is named."""
    values = {
        "final:times": np.array([1.0, 2.0, 3.0]),
        "final:status": np.array([0, 0, 0]),
        "poincare:positions": np.zeros((4, 3)),
        "poincare:counts": np.array([2, 3, 4]),
    }
    assert non_finite_observables(values) == ()
    values["poincare:positions"][3, 2] = np.nan
    values["final:times"][1] = np.inf
    assert non_finite_observables(values) == ("final:times", "poincare:positions")


def test_the_lane_status_reports_every_reason_not_only_the_first() -> None:
    """A lane that broke the contract AND reported a failed integration used to publish only the contract line."""
    assert (
        lane_status_reasons((), (), np.array([0, -1, 0]))
        == "integration_complete_or_levelset_stop"
    )
    assert (
        lane_status_reasons((), (), np.array([0, 2, 0]))
        == "integration_incomplete_or_failed"
    )
    assert lane_status_reasons(
        ("final:times",), ("status_changes=3>0 (upstream is exact)",), np.array([0, 2])
    ) == (
        "non_finite_observable:final:times;"
        "upstream_contract_violation:status_changes=3>0 (upstream is exact);"
        "integration_incomplete_or_failed"
    )


@pytest.mark.parametrize("case_id", TRACING_CONTRACT_CASE_IDS)
def test_a_lane_whose_lines_are_reordered_cannot_be_compared(case_id: str) -> None:
    """The pairing is positional, so the ordering invariant is asserted on the traced objects' start states.

    Swapping two lines leaves every shape intact and every maximum unchanged (the arrays are upstream's own, just
    permuted), so nothing else in the module can see it: the per-line published arrays would simply be attributed
    to the wrong lines.
    """
    upstream = upstream_trace(case_id)
    order = np.arange(upstream.lines)
    order[[0, 1]] = order[[1, 0]]
    with pytest.raises(TracingContractError) as raised:
        lane_distances(
            case_id,
            initial_states=upstream.initial_states[order],
            final_positions=upstream.final_positions[order],
            final_times=upstream.final_times[order],
            final_statuses=upstream.final_statuses[order],
            poincare_counts=upstream.poincare_counts[order],
            parallel_speed_fractions=(
                None
                if upstream.final_parallel_speed_fractions is None
                else upstream.final_parallel_speed_fractions[order]
            ),
        )
    assert "the same objects in the same order" in str(raised.value)
    # The unpermuted lane with the same arrays is accepted, so the check is about the ORDER and nothing else.
    assert (
        contract_violations(
            lane_distances(
                case_id,
                initial_states=upstream.initial_states,
                final_positions=upstream.final_positions,
                final_times=upstream.final_times,
                final_statuses=upstream.final_statuses,
                poincare_counts=upstream.poincare_counts,
                parallel_speed_fractions=upstream.final_parallel_speed_fractions,
            )
        )
        == ()
    )
