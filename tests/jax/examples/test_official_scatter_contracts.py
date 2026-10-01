"""Contracts derived from upstream's own end-state scatter at the harness's scales."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import dataclasses
from pathlib import Path

import numpy as np
import pytest
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.parity._manifest import ComparisonRoute
from examples.jax.parity.arbiter import (
    upstream_branch_representatives,
    upstream_end_state_matches,
)
from examples.jax.parity.cases import get_case, implemented_case_ids
from examples.jax.parity.cases.native_boozer import (
    END_STATE_OBSERVABLES,
    REPLAY_DERIVED_OBSERVABLES,
    REPLAY_EXACT_OBSERVABLES,
    REPLAY_RULE_OBSERVABLES,
    REPLAY_SOLUTION_OBSERVABLES,
    REPLAY_STARTS,
)
from examples.jax.parity.cases.native_boozer import create_input as create_boozer_input
from examples.jax.parity.input_bundle import load_input_bundle
from examples.jax.parity.official_reference import (
    load_upstream_scatter,
)
from examples.jax.parity.official_scatter_contracts import (
    PRE_REGISTERED_DRAWS,
    pre_registered_runs,
    upstream_end_states,
)
from examples.jax.parity.work_budget import WorkBudgetContract

PLANAR = "native-stage-two-optimization-planar-coils"
SCALES = ("bounded", "native_default")


def test_only_the_pre_registered_cases_declare_scatter_contracts() -> None:
    bounded_bands = {
        case_id
        for case_id in implemented_case_ids()
        if get_case(case_id).quality_band("bounded") is not None
    }
    end_state_cases = {
        case_id
        for case_id in implemented_case_ids()
        if any(get_case(case_id).end_states(scale) is not None for scale in SCALES)
    }
    stage_wise_cases = {
        (case_id, scale)
        for case_id in implemented_case_ids()
        for scale in SCALES
        if get_case(case_id).stage_wise(scale) is not None
    }
    # PLAN.md amendment 5: the planar bounded band is removed; native-boozer's
    # end-state set stays declared and is recorded informationally.
    assert bounded_bands == set()
    assert end_state_cases == {"native-boozer"}
    assert stage_wise_cases == {
        ("native-boozer", "bounded"),
        ("native-boozer", "native_default"),
        (PLANAR, "bounded"),
    }


def test_planar_bounded_is_judged_stage_wise_with_its_work_budget() -> None:
    """Amendment 5, P1: no band; final:objective informational; two tracked tests decide."""
    case = get_case(PLANAR)
    contract = case.stage_wise("bounded")
    assert contract is not None

    assert case.quality_band("bounded") is None
    assert case.quality_band("native_default") is None
    assert case.stage_wise("native_default") is None
    assert contract.informational_observables == ("final:objective",)
    assert contract.deciding_observables == (
        "initial:objective",
        "initial:objective_gradient",
    )
    assert contract.same_state_tests == (
        "tests/integration/test_jax_mirror_planar_coils_bounded_upstream_states.py",
        "tests/integration/test_jax_mirror_planar_coils_bounded_trajectory_twins.py",
    )
    # Both lanes run each bounded stage to MAXITER, as before the band.
    budget = case.work_budget_contract
    assert budget is not None
    assert budget.scales == ("bounded", "native_default")


def _boozer_routes(scale: str) -> tuple[ComparisonRoute, ...]:
    """native-boozer's own manifest routes at ``scale``: the comparators branch identity uses."""
    root = Path(__file__).resolve().parents[3]
    pair = load_runtime_contract_pair(
        root / "examples/jax/manifest.json",
        root / "examples/jax/parity_manifest.json",
        repo_root=root,
    )
    relationship = next(
        item
        for item in pair.parity.relationships
        if item.case_id == "native-boozer"
    )
    return relationship.resolve_scale(scale).comparison_routes


@pytest.mark.parametrize("scale", SCALES)
def test_boozer_end_states_are_upstreams_nine_successful_draws(scale: str) -> None:
    end_states = get_case("native-boozer").end_states(scale)
    assert end_states is not None
    scatter = load_upstream_scatter("native-boozer", scale)

    assert end_states.case_id == "native-boozer"
    assert end_states.scale == scale
    assert end_states.observables == END_STATE_OBSERVABLES
    assert get_case("native-boozer").quality_band(scale) is None
    assert scatter.success_keys == ("area:solver_success", "flux:solver_success")
    successful = [run for run in pre_registered_runs(scatter) if run.workflow_success]
    assert [state.k for state in end_states.states] == [run.k for run in successful]
    for state, run in zip(end_states.states, successful, strict=True):
        for key in END_STATE_OBSERVABLES:
            np.testing.assert_array_equal(state.values[key], run.value(key))
    assert "same-state proof" in end_states.derivation
    assert "lowest-k draw" in end_states.derivation


#: PLAN.md C3's branch set B of each scale, read off by the pre-registration's own
#: clustering of the nine (area iota): bounded -0.19382 (k = 0, 1, 3, 4, 8),
#: -0.19650 (k = 2), -0.41398 (k = 5), -9.8e-5 (k = 6), +4.3e-4 (k = 7);
#: native_default -0.19897 (k = 0, 4) and -0.40213 (k = 1-3, 5-8).
BOOZER_BRANCH_MEMBERS = {
    "bounded": {0: (0, 1, 3, 4, 8), 2: (2,), 5: (5,), 6: (6,), 7: (7,)},
    "native_default": {0: (0, 4), 1: (1, 2, 3, 5, 6, 7, 8)},
}


@pytest.mark.parametrize("scale", SCALES)
def test_boozer_branches_are_represented_by_their_lowest_k_draw(scale: str) -> None:
    """C3 (ii): a lane is matched against ONE representative per branch, its lowest-k draw."""
    end_states = get_case("native-boozer").end_states(scale)
    assert end_states is not None
    routes = _boozer_routes(scale)
    representatives = upstream_branch_representatives(end_states, routes)

    assert tuple(state.k for state in representatives) == tuple(
        BOOZER_BRANCH_MEMBERS[scale]
    )
    for representative_k, members in BOOZER_BRANCH_MEMBERS[scale].items():
        assert representative_k == min(members)
        for state in end_states.states:
            if state.k in members:
                # Every member of a branch is matched to its representative alone.
                assert upstream_end_state_matches(end_states, routes, state.values) == (
                    representative_k,
                ), (scale, state.k)


def test_a_state_matching_a_branch_member_but_not_its_representative_is_rejected() -> (
    None
):
    """Codex's boundary witness: the comparator balls are not transitive.

    A finite ten-key end state built from bounded draw k = 1, with area iota
    moved to -0.19401097835235442, still matches k = 1 under the case's own
    area-iota route (rtol 1e-3 against k = 1's -0.19381715120215226) but not
    k = 0, the representative of k = 1's branch. Matching any member would
    admit it; C3 (ii) admits only a match to a representative.
    """
    end_states = get_case("native-boozer").end_states("bounded")
    assert end_states is not None
    routes = _boozer_routes("bounded")
    (member,) = [state for state in end_states.states if state.k == 1]
    witness = {
        **member.values,
        "area:iota": np.asarray([-0.19401097835235442], dtype=np.float64),
    }
    (other_branch,) = [state for state in end_states.states if state.k == 2]

    def matches_with(first_k: int) -> tuple[int, ...]:
        """The witness's matches when draw ``first_k`` alone stands for its branch."""
        (first,) = [state for state in end_states.states if state.k == first_k]
        two_branches = dataclasses.replace(end_states, states=(first, other_branch))
        return upstream_end_state_matches(two_branches, routes, witness)

    assert all(np.all(np.isfinite(value)) for value in witness.values())
    assert matches_with(1) == (1,)
    assert matches_with(0) == ()
    assert upstream_end_state_matches(end_states, routes, witness) == ()


def test_an_end_state_set_needs_upstreams_own_success_flags() -> None:
    """Planar coils' record carries no stage success flag, so it cannot declare an end-state set."""
    with pytest.raises(ValueError, match="needs upstream's own stage success flags"):
        upstream_end_states(
            load_upstream_scatter(PLANAR, "bounded"),
            ("final:objective",),
            same_state_proof="test",
        )


def test_a_receipt_scalar_matches_the_upstream_draw_it_equals() -> None:
    """A lane receipt stores a scalar as a one-element array; the match must not care.

    ``artifacts.write_array`` publishes every scalar with shape ``(1,)`` while an
    in-process observation holds it 0-d, so the same end state reaches the
    matcher in both forms and must match the same draws in both.
    """
    end_states = get_case("native-boozer").end_states("bounded")
    assert end_states is not None
    routes = _boozer_routes("bounded")
    state = end_states.states[0]
    scalar_form = {
        key: value.reshape(()) if value.size == 1 else value
        for key, value in state.values.items()
    }
    receipt_form = {key: np.atleast_1d(value) for key, value in state.values.items()}

    assert state.k in upstream_end_state_matches(end_states, routes, scalar_form)
    assert upstream_end_state_matches(
        end_states, routes, scalar_form
    ) == upstream_end_state_matches(end_states, routes, receipt_form)


def test_an_end_state_set_refuses_a_work_budget_at_its_scale() -> None:
    with pytest.raises(ValueError, match="end-state set and work budget"):
        dataclasses.replace(
            get_case("native-boozer"),
            work_budget_contract=WorkBudgetContract(
                scales=("bounded",), derivation="Upstream fixed optimizer budget"
            ),
        )


# ------------------------------------------------ stage-wise judgment (PLAN.md amendment 5)


def test_every_stage_wise_contract_names_tracked_same_state_tests() -> None:
    root = Path(__file__).resolve().parents[3]
    declared = [
        contract
        for case_id in implemented_case_ids()
        for scale in SCALES
        if (contract := get_case(case_id).stage_wise(scale)) is not None
    ]
    assert declared
    for contract in declared:
        for test in contract.same_state_tests:
            assert (root / test).is_file(), (contract.case_id, contract.scale, test)


@pytest.mark.parametrize("scale", SCALES)
def test_boozer_is_judged_stage_wise_with_its_end_state_set_informational(
    scale: str,
) -> None:
    case = get_case("native-boozer")
    contract = case.stage_wise(scale)
    assert contract is not None

    assert contract.informational_observables == (
        *END_STATE_OBSERVABLES,
        *REPLAY_DERIVED_OBSERVABLES,
        *REPLAY_SOLUTION_OBSERVABLES,
        *REPLAY_RULE_OBSERVABLES,
    )
    # The cross-check decides through each lane's success gate; across lanes
    # only exactly shared quantities decide (PLAN.md amendment 9).
    assert contract.deciding_observables == REPLAY_EXACT_OBSERVABLES
    assert contract.same_state_tests == (
        "tests/integration/test_jax_mirror_boozer_official_end_states.py",
        "tests/integration/test_jax_mirror_boozer_first_stage_same_state.py",
    )
    # The end-state set stays declared, and is recorded informationally.
    assert case.end_states(scale) is not None
    assert case.quality_band(scale) is None
    routes = {
        f"{route.phase}:{route.observable}": route for route in _boozer_routes(scale)
    }
    for key in REPLAY_EXACT_OBSERVABLES:
        assert routes[key].applicable and routes[key].comparator == "exact", key
    for key in (*REPLAY_SOLUTION_OBSERVABLES, *REPLAY_RULE_OBSERVABLES):
        assert routes[key].applicable, key
    for key in REPLAY_DERIVED_OBSERVABLES:
        assert routes[key].applicable, key
    for key in END_STATE_OBSERVABLES:
        assert routes[key].applicable, key


@pytest.mark.parametrize("scale", SCALES)
def test_boozer_replay_starts_are_upstreams_first_stage_ends(
    scale: str, tmp_path: Path
) -> None:
    """The bundle freezes upstream's nine first-stage ends, bit for bit, after the native start."""
    bundle = create_boozer_input(tmp_path / "inputs", scale)
    _, arrays = load_input_bundle(tmp_path / "inputs", bundle)
    runs = pre_registered_runs(load_upstream_scatter("native-boozer", scale))

    assert REPLAY_STARTS == ("native", *(f"k{k}" for k in PRE_REGISTERED_DRAWS))
    assert [run.k for run in runs] == list(PRE_REGISTERED_DRAWS)
    for index, run in enumerate(runs):
        np.testing.assert_array_equal(
            arrays["replay_upstream_surface_dofs"][index],
            run.value("first:surface_dofs"),
        )
        assert arrays["replay_upstream_iota"][index] == float(run.value("first:iota"))
        assert arrays["replay_upstream_G"][index] == float(run.value("first:G"))
