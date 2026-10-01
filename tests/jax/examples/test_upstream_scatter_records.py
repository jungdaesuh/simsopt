"""The tracked upstream scatter records: upstream's own end states at the parity harness's scales.

These tests import neither simsopt nor jax. They check what a record claims about itself against the rest of the
fixture: the protocol, the draws, the runner's provenance, and -- at ``native_default``, where the runner is the
official script verbatim -- that the unperturbed draw reproduces the canonical official capture bitwise.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import copy
import json

import numpy as np
import pytest
from examples.jax.parity.official_reference import (
    UPSTREAM_SCATTER_SUCCESS_KEYS,
    OfficialUpstreamScatter,
    load_official_reference,
    load_official_sensitivity,
    load_upstream_scatter,
    upstream_scatter_path,
    upstream_scatter_records,
)

#: Every tracked record, fixed by the 2026-09-29 pre-registration (native-boozer: C3; planar coils: C4).
RECORDS = (
    ("native-boozer", "bounded"),
    ("native-boozer", "native_default"),
    ("native-stage-two-optimization-planar-coils", "bounded"),
    ("native-stage-two-optimization-planar-coils", "native_default"),
)
#: The only lines of the official body a bounded runner may change: the harness's bounded scale.
BOUNDED_SCALE_LINES = {
    "native-boozer": ("mpol = ", "ntor = "),
    "native-stage-two-optimization-planar-coils": (
        "order = ",
        "MAXITER = ",
        "nphi = ",
        "ntheta = ",
        "base_curves = create_equally_spaced_planar_curves(",
    ),
}


def test_the_tracked_records_are_exactly_the_pre_registered_set() -> None:
    assert upstream_scatter_records() == tuple(sorted(RECORDS))


@pytest.mark.parametrize(("case_id", "scale"), RECORDS)
def test_record_follows_the_one_ulp_protocol_at_one_thread(
    case_id: str, scale: str
) -> None:
    scatter = load_upstream_scatter(case_id, scale)
    sensitivity_protocol = load_official_sensitivity(
        "native-stage-two-optimization-minimal"
    ).protocol

    assert scatter.case_id == case_id and scatter.scale == scale
    assert scatter.protocol.perturbation_rule == sensitivity_protocol.perturbation_rule
    assert scatter.protocol.seed_rule == sensitivity_protocol.seed_rule
    assert scatter.protocol.perturbed_call_index == 0
    assert scatter.protocol.unperturbed_k == 0
    assert set(scatter.protocol.threads.values()) == {"1"}
    assert [run.k for run in scatter.runs] == list(range(9))
    assert len({run.perturbation_sha256 for run in scatter.runs}) == len(scatter.runs)
    assert len({run.capture_sha256 for run in scatter.runs}) == len(scatter.runs)
    for run in scatter.runs:
        assert set(run.keys()) == set(scatter.capture_keys)
        for key in scatter.capture_keys:
            value = run.value(key)
            assert value.dtype == np.float64
            assert np.all(np.isfinite(value)), (run.k, key)


@pytest.mark.parametrize(("case_id", "scale"), RECORDS)
def test_runner_is_the_official_script_or_its_scale_derivative(
    case_id: str, scale: str
) -> None:
    scatter = load_upstream_scatter(case_id, scale)
    official = load_official_reference(case_id)

    assert scatter.official_script == official.official_script
    assert scatter.official_script_sha256 == official.official_script_sha256
    if scale == "native_default":
        assert scatter.runner.kind == "verbatim"
        assert scatter.runner.body_sha256 == official.official_script_sha256
        return
    assert scatter.runner.kind == "derived"
    assert scatter.runner.body_sha256 != official.official_script_sha256
    changed = [
        line[1:]
        for line in scatter.runner.body_diff.splitlines()
        if line[:1] in "+-" and not line.startswith(("+++", "---"))
    ]
    assert changed
    assert all(line.startswith(BOUNDED_SCALE_LINES[case_id]) for line in changed), (
        changed
    )


@pytest.mark.parametrize("case_id", sorted({case_id for case_id, _scale in RECORDS}))
def test_unperturbed_native_default_draw_is_the_canonical_capture_bitwise(
    case_id: str,
) -> None:
    scatter = load_upstream_scatter(case_id, "native_default")
    canonical = load_official_reference(case_id)
    unperturbed = scatter.run(0)

    for lane_key, capture_key in scatter.capture_keys.items():
        expected = (
            np.asarray(canonical.array(capture_key), dtype=np.float64)
            if canonical.kind(capture_key) == "array"
            else np.asarray(canonical.scalar(capture_key), dtype=np.float64)
        )
        np.testing.assert_array_equal(unperturbed.value(lane_key), expected, lane_key)
    assert [
        (call.status, call.nit, call.nfev) for call in unperturbed.provider_calls
    ] == [
        (call.result["status"], call.result["nit"], call.result["nfev"])
        for call in canonical.provider_calls
    ]


@pytest.mark.parametrize(("case_id", "scale"), RECORDS)
def test_record_carries_the_cases_own_stage_success_flags(
    case_id: str, scale: str
) -> None:
    """Every run reports upstream's own flag of each stage the case names, as captured."""
    scatter = load_upstream_scatter(case_id, scale)

    assert scatter.success_keys == UPSTREAM_SCATTER_SUCCESS_KEYS[case_id]
    for run in scatter.runs:
        assert set(run.success_flags) == set(scatter.success_keys)
        assert run.workflow_success is all(run.success_flags.values())
    if case_id == "native-boozer":
        # The pre-registered nine all solved both surfaces at both scales.
        assert all(run.workflow_success for run in scatter.runs)


def _tracked_payload(case_id: str, scale: str) -> dict[str, object]:
    return json.loads(upstream_scatter_path(case_id, scale).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("schema_1", "schema"),
        ("no_success_keys", "success keys"),
        ("one_success_key", "success keys"),
        ("run_lacks_a_flag", "reports success flags"),
        ("flag_not_bool", "must be booleans"),
    ),
)
def test_the_loader_refuses_a_record_with_other_success_flags(
    mutation: str, message: str
) -> None:
    """A record regenerated without a stage's success flag can never be read."""
    payload = copy.deepcopy(_tracked_payload("native-boozer", "bounded"))
    runs = payload["runs"]
    assert isinstance(runs, list)
    if mutation == "schema_1":
        payload["schema_version"] = 1
    elif mutation == "no_success_keys":
        payload["success_keys"] = []
    elif mutation == "one_success_key":
        payload["success_keys"] = ["area:solver_success"]
    elif mutation == "run_lacks_a_flag":
        del runs[4]["success_flags"]["flux:solver_success"]
    else:
        runs[4]["success_flags"]["flux:solver_success"] = 1

    with pytest.raises(ValueError, match=message):
        OfficialUpstreamScatter.from_payload(payload)


def test_a_failed_flag_in_a_record_marks_that_run_unsuccessful() -> None:
    payload = copy.deepcopy(_tracked_payload("native-boozer", "bounded"))
    runs = payload["runs"]
    assert isinstance(runs, list)
    runs[4]["success_flags"]["area:solver_success"] = False
    scatter = OfficialUpstreamScatter.from_payload(payload)

    assert [run.k for run in scatter.runs if not run.workflow_success] == [4]
