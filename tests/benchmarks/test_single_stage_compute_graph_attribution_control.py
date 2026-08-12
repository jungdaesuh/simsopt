from __future__ import annotations

import copy
from dataclasses import replace
from typing import Literal

import pytest
from benchmarks.single_stage_compute_graph_attribution_control import (
    COMMAND_BUFFER_DISABLE_FLAG,
    NUMERICAL_PARITY_TOLERANCE_SOURCE,
    AttributionAttempt,
    AttributionBinding,
    AttributionControlError,
    build_attribution_evidence,
    require_promoting_attribution_evidence,
)
from benchmarks.single_stage_compute_graph_phase0_receipt import canonical_json_bytes


def _binding() -> AttributionBinding:
    return AttributionBinding(
        candidate_sha256="a" * 64,
        specimen_sha256="b" * 64,
        input_bundle_sha256="c" * 64,
        source_sha256="d" * 64,
        production_runtime_identity_sha256="1" * 64,
        lane_id="rtx5090",
        gpu_uuid="GPU-test",
        gate_checkpoint_sha256="e" * 64,
        warm_checkpoint_sha256="f" * 64,
        warm_p50_ns=1_000.0,
    )


def _attempt(
    mode: Literal["default_control", "command_buffer_disabled"], index: int
) -> AttributionAttempt:
    disabled = mode == "command_buffer_disabled"
    device_active_ns = 800 if disabled else 500
    phase_device_ns = (
        (("newton.residual_jvp", 480), ("adjoint.lu_solve", 280))
        if disabled
        else (("newton.residual_jvp", 300), ("adjoint.lu_solve", 180))
    )
    return AttributionAttempt(
        mode=mode,
        attempt_index=index,
        binding=_binding(),
        runtime_identity_sha256=("2" if disabled else "1") * 64,
        xla_flag_tokens=(COMMAND_BUFFER_DISABLE_FLAG,) if disabled else (),
        compilation_cache_root=f"/campaign/cache/{mode}/{index}",
        artifact_root=f"/campaign/artifacts/{mode}/{index}",
        raw_trace_path=f"post-gate/{mode}/{index}/trace.gz",
        raw_trace_sha256="3" * 64,
        child_observation_path=f"post-gate/{mode}/{index}/child.json",
        child_observation_sha256="4" * 64,
        hlo_anchor_path=f"post-gate/{mode}/{index}/anchor.json",
        hlo_anchor_sha256="5" * 64,
        profile_derivation_version="compute-graph-profile-attribution-v1",
        objective=12.5 + (5.0e-10 if disabled and index == 1 else 0.0),
        gradient=(
            1.0,
            -2.0,
            3.0 + (1.0e-9 if disabled and index == 2 else 0.0),
        ),
        solve_certificate={
            "inner_newton_success": True,
            "adjoint_success": True,
            "residual_certificates": {
                "adjoint_residual_l2": (
                    3.0e-13 if disabled and index == 0 else 1.0e-13
                ),
                "adjoint_residual_relative": 2.0e-13,
            },
        },
        module_topology_identity_sha256="9" * 64,
        evaluation_envelope_ns=1_000,
        device_active_ns=device_active_ns,
        phase_device_ns=phase_device_ns,
    )


def _attempts(
    mode: Literal["default_control", "command_buffer_disabled"],
) -> tuple[AttributionAttempt, ...]:
    return tuple(_attempt(mode, index) for index in range(3))


def test_direct_default_route_ignores_low_disabled_coverage() -> None:
    disabled = tuple(
        replace(
            attempt,
            phase_device_ns=(
                ("newton.residual_jvp", 200),
                ("adjoint.lu_solve", 80),
            ),
        )
        for attempt in _attempts("command_buffer_disabled")
    )
    document = build_attribution_evidence(
        _attempts("default_control"),
        disabled,
    )

    require_promoting_attribution_evidence(document)
    assert document["state"] == "PRODUCED"
    assert document["promotion_eligible"] is True
    assert document["blockers"] == []
    equivalence = document["equivalence"]
    assert equivalence["tolerance_source"] == NUMERICAL_PARITY_TOLERANCE_SOURCE
    assert equivalence["quantitative_result_and_residual_parity"] is True
    assert equivalence["exact_solve_status_and_residual_names"] is True
    assert equivalence["passing_solve_status"] is True
    assert equivalence["objective"]["observed_max_absolute_difference"] > 0.0
    assert equivalence["gradient"]["observed_max_absolute_difference"] > 0.0
    assert (
        equivalence["residual_certificates"]["observed_max_absolute_difference"] > 0.0
    )
    assert equivalence["objective"]["observed_max_tolerance_ratio"] <= 1.0
    assert equivalence["gradient"]["observed_max_tolerance_ratio"] <= 1.0
    assert equivalence["residual_certificates"]["observed_max_tolerance_ratio"] <= 1.0
    first_row = document["direct_default_measurement"]["attempts"][0]
    assert first_row["objective"] == 12.5
    assert first_row["gradient"] == [1.0, -2.0, 3.0]
    assert first_row["solve_certificate"]["inner_newton_success"] is True
    assert document["direct_default_measurement"]["authoritative_for_timing"] is True
    assert document["attribution_replay"]["authoritative_for_timing"] is False
    selection = document["selected_attribution"]
    assert selection["route"] == "direct_default"
    assert selection["method"] == "direct-default-median-phase-envelope-share"
    assert selection["phase_shares"] == [
        {
            "phase_id": "adjoint.lu_solve",
            "selected_default_envelope_share": 0.18,
        },
        {
            "phase_id": "newton.residual_jvp",
            "selected_default_envelope_share": 0.3,
        },
    ]
    assert selection["unattributed_default_envelope_share"] == pytest.approx(0.52)
    fallback = document["stability"]["disabled_transfer_fallback"]
    assert fallback["eligible"] is False
    assert "disabled_attribution_coverage_below_threshold" in fallback["blockers"]


@pytest.mark.parametrize(
    ("mutation", "blocker"),
    [
        (
            lambda attempt: replace(attempt, objective=13.0),
            "result_or_solve_certificate_mismatch",
        ),
        (
            lambda attempt: replace(attempt, gradient=(1.0, -2.0, 3.1)),
            "result_or_solve_certificate_mismatch",
        ),
        (
            lambda attempt: replace(
                attempt,
                solve_certificate={
                    **attempt.solve_certificate,
                    "residual_certificates": {
                        "adjoint_residual_l2": 1.0e-5,
                        "adjoint_residual_relative": 2.0e-13,
                    },
                },
            ),
            "result_or_solve_certificate_mismatch",
        ),
        (
            lambda attempt: replace(
                attempt,
                solve_certificate={
                    **attempt.solve_certificate,
                    "adjoint_success": False,
                },
            ),
            "solve_status_or_residual_names_mismatch",
        ),
        (
            lambda attempt: replace(attempt, module_topology_identity_sha256="8" * 64),
            "module_topology_identity_mismatch",
        ),
        (
            lambda attempt: replace(
                attempt,
                binding=replace(attempt.binding, gate_checkpoint_sha256="7" * 64),
            ),
            "production_binding_mismatch",
        ),
    ],
)
def test_equivalence_mismatch_is_explicitly_non_promoting(mutation, blocker) -> None:
    disabled = list(_attempts("command_buffer_disabled"))
    disabled[1] = mutation(disabled[1])

    document = build_attribution_evidence(_attempts("default_control"), disabled)

    assert document["state"] == "NON_PROMOTING"
    assert document["promotion_eligible"] is False
    assert blocker in document["blockers"]
    assert document["selected_attribution"] is None
    with pytest.raises(AttributionControlError, match="explicitly non-promoting"):
        require_promoting_attribution_evidence(document)


def test_requires_three_attempts_per_mode_before_transfer() -> None:
    document = build_attribution_evidence(
        _attempts("default_control")[:2],
        _attempts("command_buffer_disabled"),
    )

    assert document["state"] == "NON_PROMOTING"
    assert "default_attempt_count_mismatch" in document["blockers"]
    assert document["selected_attribution"] is None
    canonical_json_bytes(document)


def test_validator_recomputes_equivalence_and_rejects_summary_tampering() -> None:
    document = build_attribution_evidence(
        _attempts("default_control"),
        _attempts("command_buffer_disabled"),
    )
    tampered = copy.deepcopy(document)
    tampered["equivalence"]["gradient"]["observed_max_absolute_difference"] = 0.0

    with pytest.raises(AttributionControlError, match="differs from attempt evidence"):
        require_promoting_attribution_evidence(tampered)


def test_validator_rejects_numerical_attempt_tampering() -> None:
    document = build_attribution_evidence(
        _attempts("default_control"),
        _attempts("command_buffer_disabled"),
    )
    tampered = copy.deepcopy(document)
    tampered["attribution_replay"]["attempts"][0]["objective"] = 20.0

    with pytest.raises(AttributionControlError, match="differs from attempt evidence"):
        require_promoting_attribution_evidence(tampered)


def test_validator_rejects_categorical_digest_tampering() -> None:
    document = build_attribution_evidence(
        _attempts("default_control"),
        _attempts("command_buffer_disabled"),
    )
    tampered = copy.deepcopy(document)
    tampered["attribution_replay"]["attempts"][0][
        "solve_certificate_categorical_identity_sha256"
    ] = "0" * 64

    with pytest.raises(
        AttributionControlError, match="categorical identity differs from evidence"
    ):
        require_promoting_attribution_evidence(tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method", "disabled-device-fraction-times-default-device-active-share"),
        ("share", 0.25),
    ],
)
def test_validator_rejects_selected_route_tampering(field: str, value: object) -> None:
    document = build_attribution_evidence(
        _attempts("default_control"), _attempts("command_buffer_disabled")
    )
    tampered = copy.deepcopy(document)
    if field == "method":
        tampered["selected_attribution"]["method"] = value
    else:
        tampered["selected_attribution"]["phase_shares"][0][
            "selected_default_envelope_share"
        ] = value

    with pytest.raises(AttributionControlError, match="differs from attempt evidence"):
        require_promoting_attribution_evidence(tampered)


def test_rebuilding_from_identical_attempts_is_exact() -> None:
    defaults = _attempts("default_control")
    disabled = _attempts("command_buffer_disabled")

    first = build_attribution_evidence(defaults, disabled)
    rebuilt = build_attribution_evidence(defaults, disabled)

    assert rebuilt == first


def test_consistently_failed_solve_status_cannot_promote() -> None:
    defaults = tuple(
        replace(
            attempt,
            solve_certificate={
                **attempt.solve_certificate,
                "adjoint_success": False,
            },
        )
        for attempt in _attempts("default_control")
    )
    disabled = tuple(
        replace(
            attempt,
            solve_certificate={
                **attempt.solve_certificate,
                "adjoint_success": False,
            },
        )
        for attempt in _attempts("command_buffer_disabled")
    )

    document = build_attribution_evidence(defaults, disabled)

    assert "solve_status_not_passing" in document["blockers"]
    assert document["promotion_eligible"] is False


def test_runtime_and_cache_identity_cannot_be_reused_across_modes() -> None:
    defaults = _attempts("default_control")
    disabled = list(_attempts("command_buffer_disabled"))
    disabled[0] = replace(
        disabled[0],
        runtime_identity_sha256=defaults[0].runtime_identity_sha256,
        compilation_cache_root=defaults[0].compilation_cache_root,
    )

    document = build_attribution_evidence(defaults, disabled)

    assert document["state"] == "NON_PROMOTING"
    assert "disabled_runtime_identity_instability" in document["blockers"]
    assert "disabled_runtime_identity_not_distinct" in document["blockers"]
    assert "compilation_cache_root_reused" in document["blockers"]


def test_default_runtime_must_equal_bound_production_runtime() -> None:
    defaults = tuple(
        replace(attempt, runtime_identity_sha256="6" * 64)
        for attempt in _attempts("default_control")
    )

    document = build_attribution_evidence(
        defaults, _attempts("command_buffer_disabled")
    )

    assert document["state"] == "NON_PROMOTING"
    assert "default_runtime_identity_not_production" in document["blockers"]


def test_disabled_tokens_must_be_exact_ordered_extension_of_default() -> None:
    defaults = tuple(
        replace(attempt, xla_flag_tokens=("--xla_gpu_triton_gemm_any=true",))
        for attempt in _attempts("default_control")
    )
    disabled = tuple(
        replace(
            attempt,
            xla_flag_tokens=(
                "--xla_gpu_enable_command_buffer=",
                "--xla_gpu_triton_gemm_any=true",
            ),
        )
        for attempt in _attempts("command_buffer_disabled")
    )

    document = build_attribution_evidence(defaults, disabled)

    assert document["state"] == "NON_PROMOTING"
    assert "disabled_xla_token_sequence_not_exact_extension" in document["blockers"]


def test_disabled_configuration_must_have_exactly_one_disable_flag() -> None:
    disabled = list(_attempts("command_buffer_disabled"))
    disabled[2] = replace(disabled[2], xla_flag_tokens=())

    document = build_attribution_evidence(_attempts("default_control"), disabled)

    assert document["state"] == "NON_PROMOTING"
    assert "disabled_xla_token_sequence_not_exact_extension" in document["blockers"]


def test_contradictory_command_buffer_flag_is_not_accepted_as_disabled() -> None:
    disabled = list(_attempts("command_buffer_disabled"))
    disabled[2] = replace(
        disabled[2], xla_flag_tokens=("--xla_gpu_enable_command_buffer=true",)
    )

    document = build_attribution_evidence(_attempts("default_control"), disabled)

    assert document["state"] == "NON_PROMOTING"
    assert "disabled_xla_token_sequence_not_exact_extension" in document["blockers"]


def test_disabled_transfer_is_selected_when_direct_coverage_is_insufficient() -> None:
    defaults = tuple(
        replace(
            attempt,
            phase_device_ns=(
                ("newton.residual_jvp", 200),
                ("adjoint.lu_solve", 100),
            ),
        )
        for attempt in _attempts("default_control")
    )

    document = build_attribution_evidence(
        defaults, _attempts("command_buffer_disabled")
    )

    require_promoting_attribution_evidence(document)
    selection = document["selected_attribution"]
    assert selection["route"] == "disabled_transfer_fallback"
    assert selection["default_device_active_share"] == pytest.approx(0.5)
    assert selection["phase_shares"] == [
        {
            "phase_id": "adjoint.lu_solve",
            "disabled_device_fraction": 0.35,
            "selected_default_envelope_share": 0.175,
        },
        {
            "phase_id": "newton.residual_jvp",
            "disabled_device_fraction": 0.6,
            "selected_default_envelope_share": 0.3,
        },
    ]
    assert selection["unattributed_default_envelope_share"] == pytest.approx(0.525)


def test_direct_instability_blocks_when_fallback_is_not_eligible() -> None:
    defaults = list(_attempts("default_control"))
    defaults[2] = replace(
        defaults[2],
        phase_device_ns=(
            ("newton.residual_jvp", 280),
            ("adjoint.lu_solve", 200),
        ),
    )
    disabled = tuple(
        replace(
            attempt,
            phase_device_ns=(
                ("newton.residual_jvp", 200),
                ("adjoint.lu_solve", 80),
            ),
        )
        for attempt in _attempts("command_buffer_disabled")
    )

    document = build_attribution_evidence(defaults, disabled)

    assert document["state"] == "NON_PROMOTING"
    assert "default_phase_fraction_instability" in document["blockers"]
    assert "disabled_attribution_coverage_below_threshold" in document["blockers"]
    assert document["selected_attribution"] is None


def test_unattributed_fraction_participates_in_total_variation_gate() -> None:
    disabled = list(_attempts("command_buffer_disabled"))
    disabled[2] = replace(
        disabled[2],
        phase_device_ns=(
            ("newton.residual_jvp", 480),
            ("adjoint.lu_solve", 256),
        ),
    )

    defaults = tuple(
        replace(
            attempt,
            phase_device_ns=(
                ("newton.residual_jvp", 200),
                ("adjoint.lu_solve", 100),
            ),
        )
        for attempt in _attempts("default_control")
    )
    document = build_attribution_evidence(defaults, disabled)

    assert document["state"] == "NON_PROMOTING"
    assert document["stability"][
        "observed_disabled_phase_total_variation_distance"
    ] == pytest.approx(0.03)
    assert "disabled_phase_fraction_instability" in document["blockers"]


def test_neither_route_can_promote_when_both_coverages_are_insufficient() -> None:
    defaults = tuple(
        replace(
            attempt,
            phase_device_ns=(
                ("newton.residual_jvp", 200),
                ("adjoint.lu_solve", 100),
            ),
        )
        for attempt in _attempts("default_control")
    )
    disabled = tuple(
        replace(
            attempt,
            phase_device_ns=(
                ("newton.residual_jvp", 200),
                ("adjoint.lu_solve", 80),
            ),
        )
        for attempt in _attempts("command_buffer_disabled")
    )

    document = build_attribution_evidence(defaults, disabled)

    assert document["state"] == "NON_PROMOTING"
    assert "default_attribution_coverage_below_threshold" in document["blockers"]
    assert "disabled_attribution_coverage_below_threshold" in document["blockers"]


def test_malformed_attempt_fails_closed_before_evidence_is_written() -> None:
    malformed = replace(
        _attempt("command_buffer_disabled", 0),
        artifact_root="relative/path",
    )

    with pytest.raises(AttributionControlError, match="normalized absolute path"):
        build_attribution_evidence(_attempts("default_control"), (malformed,))
