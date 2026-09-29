"""Contracts derived from upstream's own end-state scatter at the harness's scales."""

from __future__ import annotations

from pathlib import Path

import dataclasses

import numpy as np
import pytest
from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.parity.arbiter import upstream_end_state_matches
from examples.jax.parity.cases import get_case, implemented_case_ids
from examples.jax.parity.work_budget import WorkBudgetContract
from examples.jax.parity.cases.native_boozer import END_STATE_OBSERVABLES
from examples.jax.parity.official_reference import load_upstream_scatter
from examples.jax.parity.official_scatter_contracts import (
    PRE_REGISTERED_DRAWS,
    pre_registered_runs,
)

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
    assert bounded_bands == {PLANAR}
    assert end_state_cases == {"native-boozer"}


@pytest.mark.parametrize("scale", SCALES)
def test_planar_band_is_rule_v2_over_upstreams_nine_draws(scale: str) -> None:
    band = get_case(PLANAR).quality_band(scale)
    assert band is not None
    samples = [
        float(load_upstream_scatter(PLANAR, scale).run(k).value("final:objective"))
        for k in PRE_REGISTERED_DRAWS
    ]
    low, high = min(samples), max(samples)

    assert band.scale == scale
    assert band.observable == "final:objective"
    assert band.max_value == high * (1.0 + (high - low) / low)
    assert "same-state proof" in band.derivation
    assert "stale" in band.derivation
    # A band admits a stop at the cap, so no work budget may cover a banded scale.
    assert get_case(PLANAR).work_budget_contract is None


@pytest.mark.parametrize("scale", SCALES)
def test_boozer_end_states_are_upstreams_nine_successful_draws(scale: str) -> None:
    end_states = get_case("native-boozer").end_states(scale)
    assert end_states is not None
    scatter = load_upstream_scatter("native-boozer", scale)

    assert end_states.case_id == "native-boozer"
    assert end_states.scale == scale
    assert end_states.observables == END_STATE_OBSERVABLES
    assert get_case("native-boozer").quality_band(scale) is None
    successful = [run for run in pre_registered_runs(scatter) if run.workflow_success]
    assert [state.k for state in end_states.states] == [run.k for run in successful]
    for state, run in zip(end_states.states, successful, strict=True):
        for key in END_STATE_OBSERVABLES:
            np.testing.assert_array_equal(state.values[key], run.value(key))
    assert "same-state proof" in end_states.derivation


def test_a_receipt_scalar_matches_the_upstream_draw_it_equals() -> None:
    """A lane receipt stores a scalar as a one-element array; the match must not care.

    ``artifacts.write_array`` publishes every scalar with shape ``(1,)`` while an
    in-process observation holds it 0-d, so the same end state reaches the
    matcher in both forms and must match the same draws in both.
    """
    end_states = get_case("native-boozer").end_states("bounded")
    assert end_states is not None
    root = Path(__file__).resolve().parents[3]
    pair = load_runtime_contract_pair(
        root / "examples/jax/manifest.json",
        root / "examples/jax/parity_manifest.json",
        repo_root=root,
    )
    relationship = next(
        item
        for item in pair.parity.all_relationships
        if item.case_id == "native-boozer"
    )
    routes = relationship.resolve_scale("bounded").comparison_routes
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
