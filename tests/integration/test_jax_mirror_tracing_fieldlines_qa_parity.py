"""Exact parity for the ``1_Simple/tracing_fieldlines_QA.py`` mirror."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.cases import get_case
from examples.jax.parity.cases import native_tracing_fieldlines_qa as qa_case
from examples.jax.parity.cases.native_tracing_fieldlines_qa import (
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
from simsopt_jax_adapters.field import tracing as tracing_adapter
from simsopt_jax_adapters.field.toroidal_field import ToroidalFieldJAX

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_exact_tracing_fieldlines_qa_matches_native_and_jax_cpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = get_case("native-tracing-fieldlines-qa")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)

    # ``_values`` is the single funnel every lane's trajectories pass through.
    # Recording them here lets the level-set assertion below compute its bound
    # from the run itself instead of hard-coding one, without publishing a new
    # observable (which would need a parity-manifest route) or tracing a third
    # time.
    trajectories: list[list[np.ndarray]] = []
    event_rows: list[list[np.ndarray]] = []
    original_values = qa_case._values

    def recording_values(**keywords):
        trajectories.append([np.asarray(row) for row in keywords["trajectories"]])
        event_rows.append([np.asarray(row) for row in keywords["phi_hits"]])
        return original_values(**keywords)

    monkeypatch.setattr(qa_case, "_values", recording_values)

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
        "construction:surface_dofs",
        "construction:field_dofs",
        "initial:states",
        "interpolation:surface_field",
        "interpolation:relative_error",
        "final:status",
        "poincare:counts",
    ):
        np.testing.assert_allclose(
            jax.values[observable],
            native.values[observable],
            rtol=1.0e-12,
            atol=1.0e-14,
        )

    np.testing.assert_allclose(
        jax.values["final:times"],
        native.values["final:times"],
        rtol=0.0,
        atol=6.0e-3,
    )
    np.testing.assert_allclose(
        jax.values["final:states"],
        native.values["final:states"],
        rtol=0.0,
        atol=2.0e-3,
    )
    np.testing.assert_allclose(
        jax.values["poincare:positions"],
        native.values["poincare:positions"],
        rtol=0.0,
        atol=7.0e-3,
    )

    assert native.values["final:status"].tolist() == [0, 0, -1]
    assert native.values["poincare:counts"].tolist() == [9, 9, 5]

    # Lines 0 and 1 run to ``tmax``: their end point is a property of the ODE,
    # not of the step grid, so it is held tight.
    for line in (0, 1):
        assert native.values["final:times"][line] == 50.0
        np.testing.assert_allclose(
            jax.values["final:states"][line],
            native.values["final:states"][line],
            rtol=0.0,
            atol=1.0e-11,
        )

    # Line 2 stops on the level set. Upstream does NOT root-find a
    # stopping-criterion crossing: ``simsoptpp/tracing.cpp`` evaluates every
    # criterion on the accepted post-step state and, on the first firing, records
    # ``{t, -1 - i, y}`` and breaks without pushing that state into the
    # trajectory. The published end point is therefore the last accepted step
    # still INSIDE the surface, quantised to the accepted-step grid, and the two
    # lanes' grids cannot coincide at ``tol = 1e-16`` (below double epsilon) where
    # the local error estimate sits on the floating-point noise floor of the two
    # interpolated-field implementations. What the rule does pin is asserted here.
    assert len(trajectories) == 2
    _levelset_stop_obeys_the_upstream_rule(
        native.values,
        jax.values,
        bundle=bundle,
        native_trajectories=trajectories[0],
        jax_trajectories=trajectories[1],
        native_event_rows=event_rows[0],
        jax_event_rows=event_rows[1],
        line=2,
    )


def _levelset_stop_obeys_the_upstream_rule(
    native_values: dict[str, np.ndarray],
    jax_values: dict[str, np.ndarray],
    *,
    bundle,
    native_trajectories: list[np.ndarray],
    jax_trajectories: list[np.ndarray],
    native_event_rows: list[np.ndarray],
    jax_event_rows: list[np.ndarray],
    line: int,
) -> None:
    assert int(native_values["final:status"][line]) == -1
    assert int(jax_values["final:status"][line]) == -1
    # The kept row is the last accepted step still inside (signed distance > 0);
    # a lane that published the post-crossing state would be negative here.
    assert float(native_values["final:levelset_distance"][line]) > 0.0
    assert float(jax_values["final:levelset_distance"][line]) > 0.0

    stop_time_gap = abs(
        float(jax_values["final:times"][line])
        - float(native_values["final:times"][line])
    )
    state_gap = float(
        np.max(
            np.abs(
                jax_values["final:states"][line] - native_values["final:states"][line]
            )
        )
    )
    # Both factors of the bound are COMPUTED from this run, not written down.
    # The step that matters is the CROSSING step, the one the level set was
    # crossed inside -- not the last recorded one, which is the step BEFORE it
    # and which the controller may grow by up to ``_SAFETY * 5 = 4.5`` before
    # taking the crossing step. The crossing step is in the run: upstream pushes
    # the pre-step row at the top of the loop and records the criterion event at
    # the POST-step time (``simsoptpp/tracing.cpp``
    # ``res_phi_hits.push_back(join<2, RHS::Size>({t, -1-double(i)}, y))``), so
    # for each lane it is ``t_criterion_hit - t_final``.
    crossing_step = max(
        _criterion_crossing_step(native_trajectories[line], native_event_rows[line]),
        _criterion_crossing_step(jax_trajectories[line], jax_event_rows[line]),
    )
    # The path speed of a field line is |B|, evaluated with the bundle's own
    # coils at the two published end points. The tracer integrates the
    # interpolant of this field, which the same test pins to a relative error of
    # order 1e-7 through ``interpolation:relative_error`` -- far below the scale
    # of this bound.
    _surface, coil_field = qa_case._qa_objects(bundle)
    end_points = np.ascontiguousarray(
        np.stack(
            (
                np.asarray(native_values["final:states"][line], dtype=np.float64),
                np.asarray(jax_values["final:states"][line], dtype=np.float64),
            )
        )
    )
    coil_field.set_points(end_points)
    max_field_strength = float(
        np.max(np.linalg.norm(np.asarray(coil_field.B(), dtype=np.float64), axis=1))
    )
    bound = max_field_strength * crossing_step

    # Upstream does not root-find a criterion stop: it fires on the accepted
    # post-step state and keeps the pre-crossing row, so a stop state is only
    # defined to within ONE accepted step. That is the whole content of the rule,
    # and it is what is asserted -- with the guarantee that the bound is not
    # vacuous, i.e. strictly tighter than the blanket ``final:states`` bound the
    # same test already applies to all three lines.
    assert bound < 2.0e-3, f"level-set bound {bound:.3e} is not tighter than 2.0e-3"
    assert stop_time_gap <= crossing_step
    assert state_gap <= bound


def _criterion_crossing_step(trajectory: np.ndarray, event_rows: np.ndarray) -> float:
    """The accepted step the stopping criterion fired inside, for one lane.

    ``trajectory[-1, 0]`` is the last PRE-step row upstream keeps; the criterion
    event row carries the POST-step time of the step that fired
    (``simsoptpp/tracing.cpp``, the ``stopping_criteria`` loop). Their difference
    is that step, measured.
    """
    criterion_rows = event_rows[event_rows[:, 1] < 0]
    assert criterion_rows.shape[0] == 1
    crossing_step = float(criterion_rows[-1, 0] - trajectory[-1, 0])
    # Upstream's order: the pre-step row is pushed, the step is taken, the
    # criterion is evaluated on the POST-step state. So the event time is
    # strictly after the published end time, by exactly one accepted step.
    assert crossing_step > 0.0
    return crossing_step


@pytest.mark.parametrize("positive_status", [1, 2])
def test_qa_producer_rejects_positive_core_status_as_incomplete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    positive_status: int,
) -> None:
    case = get_case("native-tracing-fieldlines-qa")
    input_root = tmp_path / "inputs"
    bundle = case.create_input(input_root, "bounded")
    _, arrays = load_input_bundle(input_root, bundle)
    initial = arrays["initial_states"]
    trajectory = np.stack(
        [
            np.stack((np.r_[0.0, point], np.r_[time, point]))
            for point, time in zip(initial, (1.0, 1.0, 50.0), strict=True)
        ]
    )
    hits = np.zeros((3, 1, 5), dtype=np.float64)
    hits[1, 0, 1] = -1.0
    core_result = SimpleNamespace(
        trajectory=trajectory,
        mask=np.ones((3, 2), dtype=bool),
        phi_hits=hits,
        phi_hits_count=np.asarray([0, 1, 0]),
        status=np.asarray([positive_status, -1, 0]),
        t_final=np.asarray([1.0, 1.0, 50.0]),
        steps_taken=np.ones(3, dtype=np.int32),
    )
    monkeypatch.setattr(
        tracing_adapter,
        "_trace_cartesian_chunks",
        lambda *_args, **_kwargs: (
            list(trajectory),
            [hits[0, :0], hits[1, :1], hits[2, :0]],
            core_result,
        ),
    )
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_parity")
    monkeypatch.setenv("SIMSOPT_PRECISION", "fp64")
    monkeypatch.setenv("JAX_ENABLE_X64", "1")

    observation = case.execute("jax-cpu", bundle, arrays)

    assert observation.values["final:status"].tolist() == [positive_status, -1, 0]
    assert observation.values["final:times"].tolist() == [1.0, 1.0, 50.0]
    assert observation.success is False
    assert observation.normalized_status == "failed"
    assert observation.raw_status == "integration_incomplete_or_failed"

    paths, crossings = tracing_adapter.compute_fieldlines(
        ToroidalFieldJAX(1.3, 0.8),
        arrays["initial_states"][:, 0],
        arrays["initial_states"][:, 2],
        tmax=50.0,
        tol=1.0e-9,
        phis=(),
        stopping_criteria=(),
    )
    assert len(paths) == len(crossings) == 3


@pytest.mark.parametrize("event_index", [-1, -2])
def test_qa_native_short_trace_requires_recorded_stop_event(event_index: int) -> None:
    trajectories = [
        np.asarray([[0.0, 1.0, 0.0, 0.0], [time, 1.0, 0.0, 0.0]])
        for time in (1.0, 2.0, 10.0)
    ]
    hits = [
        np.empty((0, 5)),
        np.asarray([[2.0, float(event_index), 1.0, 0.0, 0.0]]),
        np.empty((0, 5)),
    ]
    classifier = SimpleNamespace(evaluate_xyz=lambda xyz: np.ones(xyz.shape[0]))

    values = _values(
        surface_dofs=np.zeros(1),
        field_dofs=np.zeros(1),
        initial_states=np.zeros((3, 3)),
        surface_field=np.zeros((1, 3)),
        interpolation_error=0.0,
        trajectories=trajectories,
        phi_hits=hits,
        tmax=10.0,
        classifier=classifier,
    )

    assert values["final:status"].tolist() == [1, event_index, 0]


def test_qa_public_and_parity_jax_use_official_integrator_tolerance(
    tmp_path: Path,
) -> None:
    """The public and parity JAX routes retain the official `1e-16` policy."""
    case = get_case("native-tracing-fieldlines-qa")
    bundle = case.create_input(tmp_path / "inputs", "native_default")
    assert bundle.configuration["integrator_tolerance"] == 1.0e-16
    assert "native_integrator_tolerance" not in bundle.configuration
    assert "jax_integrator_tolerance" not in bundle.configuration

    source = (REPO_ROOT / "examples/jax/1_Simple/tracing_fieldlines_QA.py").read_text()
    module = ast.parse(source)
    solve = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "solve"
    )
    trace_call = next(
        node
        for node in ast.walk(solve)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "compute_fieldlines_with_status"
    )
    tolerance = next(
        keyword.value for keyword in trace_call.keywords if keyword.arg == "tol"
    )
    assert isinstance(tolerance, ast.Constant)
    assert tolerance.value == 1.0e-16


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
        "construction:surface_dofs": np.zeros(1),
        "construction:field_dofs": np.zeros(1),
        "initial:states": upstream.initial_states.copy(),
        "interpolation:surface_field": np.zeros((1, 3)),
        "interpolation:relative_error": np.asarray(0.0),
        "final:states": upstream.final_positions.copy(),
        "final:times": upstream.final_times.copy(),
        "final:status": upstream.final_statuses.copy(),
        "final:levelset_distance": np.zeros(upstream.lines),
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
