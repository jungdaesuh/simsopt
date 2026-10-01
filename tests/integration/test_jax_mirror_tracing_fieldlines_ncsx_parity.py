"""Exact parity for the ``1_Simple/tracing_fieldlines_NCSX.py`` mirror."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from pathlib import Path

import numpy as np
import pytest
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases.native_tracing_fieldlines_ncsx import (
    CASE_ID,
    _observation,
    _values,
)
from examples.jax.parity.input_bundle import InputBundle, load_input_bundle
from examples.jax.parity.official_tracing_contract import (
    PUBLISHED_DISTANCE_KEYS,
    tracing_contract,
    upstream_trace,
)
from simsopt_jax.examples import ExecutionScale


def test_exact_tracing_fieldlines_ncsx_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-tracing-fieldlines-ncsx")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    native = case.execute("native-cpu", bundle, arrays)

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")
    jax = case.execute("jax-cpu", bundle, arrays)

    assert native.success is True
    assert jax.success is True
    assert native.input_fingerprint == jax.input_fingerprint
    assert native.configuration_fingerprint == jax.configuration_fingerprint
    assert native.effective_construction_fingerprint == (
        jax.effective_construction_fingerprint
    )
    assert native.completed_workflow_stages == jax.completed_workflow_stages
    assert set(native.values) == set(jax.values)

    for observable in (
        "construction:axis_dofs",
        "construction:field_dofs",
        "initial:states",
        "interpolation:axis_field",
        "interpolation:relative_error",
        "final:status",
        "poincare:counts",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-13,
            atol=1.0e-14,
        )

    np.testing.assert_allclose(
        jax.values["final:times"],
        native.values["final:times"],
        rtol=0.0,
        atol=2.0e-2,
    )
    np.testing.assert_allclose(
        jax.values["final:states"],
        native.values["final:states"],
        rtol=0.0,
        atol=3.0e-2,
    )
    np.testing.assert_allclose(
        jax.values["poincare:positions"],
        native.values["poincare:positions"],
        rtol=0.0,
        atol=3.0e-2,
    )

    assert native.values["final:status"].tolist() == [0, 0, -1]
    assert native.values["poincare:counts"].tolist() == [31, 31, 4]


@pytest.mark.parametrize("event_index", [-1, -2])
def test_ncsx_short_trace_without_event_is_incomplete(event_index: int) -> None:
    trajectories = [
        np.asarray([[0.0, 1.0, 0.0, 0.0], [time, 1.0, 0.0, 0.0]])
        for time in (1.0, 2.0, 10.0)
    ]
    hits = [
        np.empty((0, 5)),
        np.asarray([[2.0, float(event_index), 1.0, 0.0, 0.0]]),
        np.empty((0, 5)),
    ]
    values = _values(
        axis_dofs=np.zeros(1),
        field_dofs=np.zeros(1),
        initial_states=np.zeros((3, 3)),
        axis_field=np.zeros((1, 3)),
        interpolation_error=0.0,
        trajectories=trajectories,
        phi_hits=hits,
        tmax=10.0,
    )
    assert values["final:status"].tolist() == [1, event_index, 0]


# --- shipped-scale contract gate (`native_default`) -------------------------------------------------------------
# These exercise the gate itself, not a shipped-scale trace: they hand `_observation` upstream's own canonical end
# state (and deliberate departures from it) and check the lane verdict. See
# `examples/jax/parity/official_tracing_contract.py` for the rule and its tracked evidence.


def _contract_bundle(scale: ExecutionScale) -> InputBundle:
    return InputBundle(
        schema_version=1,
        case_id=CASE_ID,
        scale=scale,
        random_seed=0,
        configuration={"synthetic_gate_input": True},
        configuration_fingerprint="0" * 64,
        arrays={},
        input_fingerprint="1" * 64,
    )


def _upstream_shaped_values() -> dict[str, np.ndarray]:
    """A lane value dictionary whose traced observables ARE upstream's canonical record."""
    upstream = upstream_trace(CASE_ID)
    return {
        "construction:axis_dofs": np.zeros(1),
        "construction:field_dofs": np.zeros(1),
        "initial:states": upstream.initial_states.copy(),
        "interpolation:axis_field": np.zeros((1, 3)),
        "interpolation:relative_error": np.asarray(0.0),
        "final:states": upstream.final_positions.copy(),
        "final:times": upstream.final_times.copy(),
        "final:status": upstream.final_statuses.copy(),
        "poincare:counts": upstream.poincare_counts.copy(),
        "poincare:positions": np.zeros((1, 3)),
    }


def _gate_observation(
    values: dict[str, np.ndarray], scale: ExecutionScale
) -> LaneObservation:
    return _observation(
        "native-cpu",
        _contract_bundle(scale),
        values,
        platform="cpu",
        precision="fp64",
        driver="contract_gate",
    )


#: The distance arrays a FIELD-LINE lane publishes at ``native_default``, written out here: the contract also owns
#: the guiding-centre parallel-speed fraction, which this case's traced object does not have.
PUBLISHED_KEYS = (
    "final:upstream_position_distance",
    "final:upstream_time_difference",
    "final:upstream_status_difference",
    "poincare:upstream_count_difference",
)


def test_native_default_lane_equal_to_upstream_is_converged() -> None:
    observation = _gate_observation(_upstream_shaped_values(), "native_default")
    assert observation.normalized_status == "converged"
    assert observation.success is True
    assert set(observation.values) >= set(PUBLISHED_KEYS)
    # and NOTHING else the contract can measure: a field line has no parallel speed, so this lane must not
    # publish the guiding-centre pitch distance, and the case must not carry a route for one.
    assert set(observation.values) & set(PUBLISHED_DISTANCE_KEYS.values()) == set(
        PUBLISHED_KEYS
    )
    for key in PUBLISHED_KEYS:
        assert float(np.max(np.abs(observation.values[key]))) == 0.0


@pytest.mark.parametrize(
    "quantity", ["final_position_distance_max", "final_time_difference_max"]
)
def test_native_default_lane_outside_a_ceiling_fails_with_the_reason_named(
    quantity: str,
) -> None:
    item = tracing_contract(CASE_ID).item(quantity)
    values = _upstream_shaped_values()
    if quantity == "final_position_distance_max":
        values["final:states"] = values["final:states"].copy()
        values["final:states"][0, 0] += 2.0 * item.ceiling
    else:
        values["final:times"] = values["final:times"].copy()
        values["final:times"][0] = (
            np.nextafter(values["final:times"][0], np.inf)
            if item.exact
            else values["final:times"][0] + 2.0 * item.ceiling
        )
    observation = _gate_observation(values, "native_default")
    assert observation.normalized_status == "failed"
    assert observation.success is False
    assert observation.raw_status.startswith(f"upstream_contract_violation:{quantity}=")


def test_native_default_lane_with_a_terminal_status_mismatch_fails() -> None:
    values = _upstream_shaped_values()
    values["final:status"] = values["final:status"].copy()
    values["final:status"][0] = values["final:status"][0] - 1
    observation = _gate_observation(values, "native_default")
    assert observation.normalized_status == "failed"
    assert observation.raw_status.startswith(
        "upstream_contract_violation:status_changes="
    )


def test_native_default_lane_with_a_hit_count_departure_is_judged_by_the_record() -> (
    None
):
    item = tracing_contract(CASE_ID).item("hit_count_difference_per_line_max_abs")
    values = _upstream_shaped_values()
    values["poincare:counts"] = values["poincare:counts"].copy()
    values["poincare:counts"][0] += int(item.ceiling) + 1
    observation = _gate_observation(values, "native_default")
    assert observation.normalized_status == "failed"
    assert observation.raw_status.startswith(
        "upstream_contract_violation:hit_count_difference_per_line_max_abs="
    )
    # The "inside the bound" leg must PERTURB something, or it asserts ``converged`` on an unperturbed copy of
    # upstream's record and cannot tell a working gate from one that never fires (the ceiling of an exact item is
    # ``0``, so scaling a departure by it is a no-op). Where upstream is exact there is no room inside, and the
    # fact asserted is that one: the smallest possible departure is already a violation.
    if item.exact:
        assert item.ceiling == 0
        assert item.exceeded_by(1)
    else:
        inside = _upstream_shaped_values()
        inside["poincare:counts"] = inside["poincare:counts"].copy()
        inside["poincare:counts"][0] += int(item.ceiling) // 2
        assert not np.array_equal(
            inside["poincare:counts"], _upstream_shaped_values()["poincare:counts"]
        )
        observation = _gate_observation(inside, "native_default")
        assert observation.normalized_status == "converged"


def test_the_bounded_scale_publishes_no_distance_to_upstream() -> None:
    observation = _gate_observation(_upstream_shaped_values(), "bounded")
    assert observation.normalized_status == "converged"
    assert set(observation.values) & set(PUBLISHED_DISTANCE_KEYS.values()) == set()


@pytest.mark.parametrize("scale", ["native_default", "bounded"])
@pytest.mark.parametrize("key", ["final:times", "poincare:positions"])
def test_a_lane_with_a_non_finite_published_array_fails_with_the_key_named(
    scale: ExecutionScale, key: str
) -> None:
    """A non-finite number ANYWHERE in a published array fails the lane, at EVERY scale.

    ``final:times`` is in no health predicate and the contract's ceiling used to accept ``nan`` outright
    (``nan > ceiling`` is ``False``), so a lane whose times were all ``nan`` was published as ``converged``. The
    whole array is checked, so it does not matter which entry went bad, and the bounded scale -- which has no
    upstream record to be judged against -- is covered by the same rule.
    """
    values = _upstream_shaped_values()
    values[key] = values[key].astype(np.float64).copy()
    values[key][tuple(0 for _ in values[key].shape)] = np.nan

    observation = _gate_observation(values, scale)

    assert observation.normalized_status == "failed"
    assert observation.success is False
    assert f"non_finite_observable:{key}" in observation.raw_status
    # Unperturbed, the same lane at the same scale is converged, so the test cannot pass by the gate always firing.
    assert _gate_observation(_upstream_shaped_values(), scale).normalized_status == (
        "converged"
    )


@pytest.mark.parametrize("scale", ["native_default", "bounded"])
def test_an_all_non_finite_lane_is_never_converged(scale: ExecutionScale) -> None:
    """The measured shipped-scale case: every final time and every hit position ``nan``."""
    values = _upstream_shaped_values()
    values["final:times"] = np.full_like(
        values["final:times"].astype(np.float64), np.nan
    )
    values["poincare:positions"] = np.full_like(
        values["poincare:positions"].astype(np.float64), np.nan
    )

    observation = _gate_observation(values, scale)

    assert observation.normalized_status == "failed"
    assert observation.success is False
    assert "non_finite_observable:final:times" in observation.raw_status
    assert "non_finite_observable:poincare:positions" in observation.raw_status


def test_a_lane_that_also_reports_a_failed_integration_names_both_reasons() -> None:
    """``raw_status`` used to report the contract violation only and drop the provider's own failure."""
    values = _upstream_shaped_values()
    values["final:status"] = values["final:status"].copy()
    values["final:status"][0] = 2

    observation = _gate_observation(values, "native_default")

    assert observation.normalized_status == "failed"
    assert "upstream_contract_violation:status_changes=" in observation.raw_status
    assert "integration_incomplete_or_failed" in observation.raw_status
