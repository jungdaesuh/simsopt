"""Tests for typed JAX runtime and host boundaries."""

from __future__ import annotations

import os
import subprocess
import sys
from contextvars import Context
from dataclasses import FrozenInstanceError
from pathlib import Path

import jax
import numpy as np
import pytest
from simsopt_jax import numerical_policy
from simsopt_jax.numerical_policy import (
    MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
    MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL,
    MIXED_DENSE_IR_CERTIFICATE_PROBABILITY_MODEL,
    MIXED_DENSE_IR_CERTIFICATE_PROBE_SAMPLING_MODEL,
    CertificateProbeAuthority,
    CertificateProbeEvidence,
    CertificateProbeKeyData,
    fresh_certificate_probe_key_data,
    resolve_certificate_probe_authority,
)
from simsopt_jax.runtime.host_boundary import (
    host_array,
    host_transfer_audit,
    host_transfer_evaluation,
    host_transfer_phase,
    runtime_certificate_probe_key,
)
from simsopt_jax.runtime.trace_annotations import (
    EvaluationDisposition,
    EvaluationKind,
    HostEvent,
    PhaseId,
    ProfilerBoundaryOperation,
    accepted_iteration_span,
    annotations_enabled,
    current_device_phase_stack,
    device_scope,
    evaluation_context,
    host_span,
    record_evaluation_disposition,
    record_host_event,
    segmented_profiler_boundaries,
    trace_session,
)


def test_trace_session_records_nested_immutable_host_intervals() -> None:
    assert not annotations_enabled()

    with trace_session() as audit:  # noqa: SIM117 - nested depth is under test
        with host_span(PhaseId.OPTIMIZER_LIFECYCLE, attributes={"label": "outer"}):
            with host_span(PhaseId.HOST_H2D_SUBMIT):
                pass

    records = audit.records()
    assert not annotations_enabled()
    assert tuple(record.phase for record in records) == (
        PhaseId.HOST_H2D_SUBMIT,
        PhaseId.OPTIMIZER_LIFECYCLE,
    )
    assert tuple(record.depth for record in records) == (1, 0)
    assert records[1].attributes == (("label", "outer"),)
    assert all(record.end_ns >= record.start_ns for record in records)
    assert isinstance(records, tuple)


def test_trace_context_cleanup_is_exception_safe_and_context_local() -> None:
    with pytest.raises(RuntimeError, match="sentinel"), trace_session():
        assert annotations_enabled()
        with device_scope(PhaseId.NEWTON_RESIDUAL_JVP):
            raise RuntimeError("sentinel")

    assert not annotations_enabled()
    assert current_device_phase_stack() == ()
    with device_scope(PhaseId.NEWTON_RESIDUAL_JVP):
        pass

    with trace_session() as outer:
        with host_span(PhaseId.OPTIMIZER_LIFECYCLE):
            pass
        with trace_session() as inner, host_span(PhaseId.HOST_H2D_SUBMIT):
            pass
        with host_span(PhaseId.HOST_D2H_MATERIALIZE):
            pass

    assert tuple(record.phase for record in outer.records()) == (
        PhaseId.OPTIMIZER_LIFECYCLE,
        PhaseId.HOST_D2H_MATERIALIZE,
    )
    assert tuple(record.phase for record in inner.records()) == (
        PhaseId.HOST_H2D_SUBMIT,
    )


def test_device_phase_stack_is_trace_only_nested_and_session_local() -> None:
    assert current_device_phase_stack() == ()
    with device_scope(PhaseId.NEWTON_RESIDUAL_JVP):
        assert current_device_phase_stack() == ()

    with trace_session(), device_scope(PhaseId.ADJOINT_OUTER_VJP_RHS):
        assert current_device_phase_stack() == (PhaseId.ADJOINT_OUTER_VJP_RHS,)
        with device_scope(PhaseId.BIOTSAVART_FORWARD):
            assert current_device_phase_stack() == (
                PhaseId.ADJOINT_OUTER_VJP_RHS,
                PhaseId.BIOTSAVART_FORWARD,
            )
        with trace_session():
            assert current_device_phase_stack() == ()
        assert current_device_phase_stack() == (PhaseId.ADJOINT_OUTER_VJP_RHS,)

    assert current_device_phase_stack() == ()


def test_segmented_profiler_hooks_exactly_bracket_iterations_one_through_seven(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []

    class FakeStepTraceAnnotation:
        def __init__(self, _name: str, *, step_num: int) -> None:
            self._iteration_id = step_num

        def __enter__(self) -> None:
            calls.append(("step_enter", self._iteration_id))

        def __exit__(self, *_args: object) -> None:
            calls.append(("step_exit", self._iteration_id))

    monkeypatch.setattr(jax.profiler, "StepTraceAnnotation", FakeStepTraceAnnotation)

    with segmented_profiler_boundaries(
        lambda iteration_id: calls.append(("start", iteration_id)),
        lambda iteration_id: calls.append(("stop", iteration_id)),
    ) as boundary_audit, trace_session():
        for iteration_id in range(1, 8):
            with accepted_iteration_span(iteration_id):
                calls.append(("body", iteration_id))

    assert calls == [
        (operation, iteration_id)
        for iteration_id in range(1, 8)
        for operation in ("start", "step_enter", "body", "step_exit", "stop")
    ]
    assert tuple(
        (record.iteration_id, record.operation) for record in boundary_audit.records()
    ) == tuple(
        (iteration_id, operation)
        for iteration_id in range(1, 8)
        for operation in (
            ProfilerBoundaryOperation.START,
            ProfilerBoundaryOperation.STOP,
        )
    )
    assert all(record.end_ns >= record.start_ns for record in boundary_audit.records())


def test_segmented_profiler_controller_is_context_local_and_default_inert() -> None:
    calls: list[tuple[str, int]] = []

    with accepted_iteration_span(1):
        pass
    with segmented_profiler_boundaries(
        lambda iteration_id: calls.append(("start", iteration_id)),
        lambda iteration_id: calls.append(("stop", iteration_id)),
    ):

        def isolated_context() -> None:
            with trace_session(), accepted_iteration_span(2):
                pass

        Context().run(isolated_context)
        with trace_session(), accepted_iteration_span(2):
            pass

    with trace_session(), accepted_iteration_span(3):
        pass

    assert calls == [("start", 2), ("stop", 2)]


def test_segmented_profiler_stop_hook_runs_when_iteration_body_raises() -> None:
    calls: list[tuple[str, int]] = []

    with pytest.raises(  # noqa: SIM117 - Python 3.8 cannot parenthesize with
        RuntimeError, match="iteration failed"
    ):
        with segmented_profiler_boundaries(
            lambda iteration_id: calls.append(("start", iteration_id)),
            lambda iteration_id: calls.append(("stop", iteration_id)),
        ) as boundary_audit, trace_session(), accepted_iteration_span(1):
            raise RuntimeError("iteration failed")

    assert calls == [("start", 1), ("stop", 1)]
    assert tuple(record.operation for record in boundary_audit.records()) == (
        ProfilerBoundaryOperation.START,
        ProfilerBoundaryOperation.STOP,
    )


@pytest.mark.parametrize(
    ("failing_operation", "expected_operations"),
    (
        (ProfilerBoundaryOperation.START, (ProfilerBoundaryOperation.START,)),
        (
            ProfilerBoundaryOperation.STOP,
            (
                ProfilerBoundaryOperation.START,
                ProfilerBoundaryOperation.STOP,
            ),
        ),
    ),
)
def test_segmented_profiler_hook_exceptions_are_recorded_and_cleanup_context(
    failing_operation: ProfilerBoundaryOperation,
    expected_operations: tuple[ProfilerBoundaryOperation, ...],
) -> None:
    def hook(operation: ProfilerBoundaryOperation, _iteration_id: int) -> None:
        if operation is failing_operation:
            raise RuntimeError(f"{operation.value} failed")

    with pytest.raises(  # noqa: SIM117 - Python 3.8 cannot parenthesize with
        RuntimeError, match=f"{failing_operation.value} failed"
    ):
        with segmented_profiler_boundaries(
            lambda iteration_id: hook(ProfilerBoundaryOperation.START, iteration_id),
            lambda iteration_id: hook(ProfilerBoundaryOperation.STOP, iteration_id),
        ) as boundary_audit, trace_session(), accepted_iteration_span(1):
            pass

    assert tuple(record.operation for record in boundary_audit.records()) == (
        expected_operations
    )
    with segmented_profiler_boundaries(
        lambda _iteration_id: None,
        lambda _iteration_id: None,
    ), trace_session(), accepted_iteration_span(1):
        pass


def test_segmented_profiler_pause_records_are_immutable() -> None:
    with segmented_profiler_boundaries(
        lambda _iteration_id: None,
        lambda _iteration_id: None,
    ) as boundary_audit, trace_session(), accepted_iteration_span(1):
        pass

    records = boundary_audit.records()
    assert isinstance(records, tuple)
    with pytest.raises(FrozenInstanceError):
        records[0].iteration_id = 2  # type: ignore[misc]


def test_segmented_profiler_rejects_nested_controllers_spans_and_retries() -> None:
    calls: list[tuple[str, int]] = []

    with segmented_profiler_boundaries(
        lambda iteration_id: calls.append(("start", iteration_id)),
        lambda iteration_id: calls.append(("stop", iteration_id)),
    ):
        with pytest.raises(  # noqa: SIM117 - Python 3.8 compatibility
            RuntimeError, match="controllers cannot nest"
        ):
            with segmented_profiler_boundaries(
                lambda _iteration_id: None,
                lambda _iteration_id: None,
            ):
                pass
        with trace_session(), accepted_iteration_span(1):  # noqa: SIM117
            with pytest.raises(
                RuntimeError, match="boundaries cannot nest"
            ), accepted_iteration_span(2):
                pass
        with trace_session():  # noqa: SIM117 - retry needs a fresh trace audit
            with pytest.raises(
                RuntimeError, match="cannot retry iteration 1"
            ), accepted_iteration_span(1):
                pass

    assert calls == [("start", 1), ("stop", 1)]


def test_evaluation_events_and_disposition_share_canonical_identity() -> None:
    with trace_session() as audit:
        with accepted_iteration_span(2), evaluation_context(
            "evaluation-3", "a" * 64, EvaluationKind.TRIAL
        ) as evaluation:
            with host_span(PhaseId.HOST_H2D_SUBMIT):
                pass
            record_host_event(HostEvent.EVALUATOR_ENTRY)
            record_host_event(HostEvent.DEVICE_READY)
            record_host_event(HostEvent.EVALUATOR_RETURN)
        record_evaluation_disposition(
            evaluation,
            EvaluationDisposition.ACCEPTED,
            accepted_iteration_id=2,
        )

    assert tuple(event.event for event in audit.events()) == tuple(HostEvent)
    assert all(event.evaluation == evaluation for event in audit.events())
    assert all(event.evaluation.outer_iteration_id == 2 for event in audit.events())
    assert dict(audit.records()[0].attributes) == {
        "evaluation_id": "evaluation-3",
        "evaluation_kind": "trial",
        "outer_iteration_id": 2,
        "parameter_sha256": "a" * 64,
    }
    assert audit.dispositions()[0].evaluation == evaluation
    assert audit.dispositions()[0].accepted_iteration_id == 2


def test_trial_entries_record_exclusive_host_control_gaps() -> None:
    def emit_lifecycle() -> None:
        record_host_event(HostEvent.EVALUATOR_ENTRY)
        record_host_event(HostEvent.DEVICE_READY)
        record_host_event(HostEvent.EVALUATOR_RETURN)

    with trace_session() as audit:
        with evaluation_context("initial", "1" * 64, EvaluationKind.INITIAL):
            emit_lifecycle()
        with accepted_iteration_span(1):
            with evaluation_context(
                "trial-1", "2" * 64, EvaluationKind.TRIAL
            ) as rejected:
                emit_lifecycle()
            with evaluation_context(
                "trial-2", "3" * 64, EvaluationKind.TRIAL
            ) as accepted:
                emit_lifecycle()
        record_evaluation_disposition(
            rejected,
            EvaluationDisposition.REJECTED,
            accepted_iteration_id=None,
        )
        record_evaluation_disposition(
            accepted,
            EvaluationDisposition.ACCEPTED,
            accepted_iteration_id=1,
        )
        with accepted_iteration_span(2), evaluation_context(
            "trial-3", "4" * 64, EvaluationKind.TRIAL
        ):
            emit_lifecycle()
        with evaluation_context("final", "5" * 64, EvaluationKind.FINAL_REPORTING):
            emit_lifecycle()

    gaps = tuple(
        record
        for record in audit.records()
        if record.phase is PhaseId.HOST_LINE_SEARCH_CONTROL
    )
    assert tuple(
        (
            dict(record.attributes)["previous_evaluation_id"],
            dict(record.attributes)["next_evaluation_id"],
            dict(record.attributes)["outer_iteration_id"],
        )
        for record in gaps
    ) == (
        ("initial", "trial-1", 1),
        ("trial-1", "trial-2", 1),
        ("trial-2", "trial-3", 2),
    )
    events = audit.events()
    returns = {
        event.evaluation.evaluation_id: event.timestamp_ns
        for event in events
        if event.event is HostEvent.EVALUATOR_RETURN
    }
    entries = {
        event.evaluation.evaluation_id: event.timestamp_ns
        for event in events
        if event.event is HostEvent.EVALUATOR_ENTRY
    }
    assert all(
        record.start_ns == returns[dict(record.attributes)["previous_evaluation_id"]]
        and record.end_ns == entries[dict(record.attributes)["next_evaluation_id"]]
        for record in gaps
    )


def test_trace_annotations_have_no_numerical_outputs() -> None:
    value = jax.numpy.asarray([1.0, 2.0], dtype=jax.numpy.float64)
    with trace_session():
        with device_scope(PhaseId.BIOTSAVART_FORWARD) as device_output:
            result = value + 1.0
        with host_span(PhaseId.HOST_D2H_MATERIALIZE) as host_output:
            result = result * 2.0

    assert device_output is None
    assert host_output is None
    np.testing.assert_array_equal(np.asarray(result), np.asarray([4.0, 6.0]))


def test_unprofiled_iteration_context_still_correlates_control_observations() -> None:
    assert not annotations_enabled()


@pytest.mark.parametrize(
    ("evaluation_id", "parameter_sha256", "message"),
    (
        ("", "a" * 64, "evaluation_id"),
        ("evaluation-1", "A" * 64, "parameter_sha256"),
        ("evaluation-1", "a" * 63, "parameter_sha256"),
    ),
)
def test_evaluation_context_rejects_noncanonical_identity(
    evaluation_id: str,
    parameter_sha256: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message), evaluation_context(
        evaluation_id,
        parameter_sha256,
        EvaluationKind.TRIAL,
    ):
        pass


def test_trace_audit_rejects_duplicate_iterations_events_and_dispositions() -> None:
    with trace_session() as audit:
        with accepted_iteration_span(1), evaluation_context(
            "evaluation-1", "d" * 64, EvaluationKind.TRIAL
        ) as evaluation:
            record_host_event(HostEvent.EVALUATOR_ENTRY)
            with pytest.raises(RuntimeError, match="ENTRY, READY, RETURN"):
                record_host_event(HostEvent.EVALUATOR_ENTRY)
            record_host_event(HostEvent.DEVICE_READY)
            record_host_event(HostEvent.EVALUATOR_RETURN)
        record_evaluation_disposition(
            evaluation,
            EvaluationDisposition.ACCEPTED,
            accepted_iteration_id=1,
        )
        with pytest.raises(RuntimeError, match="only one disposition"):
            record_evaluation_disposition(
                evaluation,
                EvaluationDisposition.ACCEPTED,
                accepted_iteration_id=1,
            )
        with pytest.raises(
            RuntimeError, match="duplicate accepted iteration"
        ), accepted_iteration_span(1):
            pass

    assert len(audit.events()) == 3


def test_disposition_requires_trial_and_matching_iteration() -> None:
    with trace_session():
        with evaluation_context("initial", "e" * 64, EvaluationKind.INITIAL) as initial:
            pass
        with pytest.raises(ValueError, match="only trial"):
            record_evaluation_disposition(
                initial,
                EvaluationDisposition.REJECTED,
                accepted_iteration_id=None,
            )
        with accepted_iteration_span(2), evaluation_context(
            "trial", "f" * 64, EvaluationKind.TRIAL
        ) as trial:
            pass
        with pytest.raises(ValueError, match="must match"):
            record_evaluation_disposition(
                trial,
                EvaluationDisposition.ACCEPTED,
                accepted_iteration_id=1,
            )

    with accepted_iteration_span(4), evaluation_context(
        "control-evaluation", "c" * 64, EvaluationKind.TRIAL
    ) as evaluation:
        pass

    assert evaluation.outer_iteration_id == 4
    assert not annotations_enabled()


def test_transfer_audit_correlates_evaluation_ids_without_changing_defaults() -> None:
    with host_transfer_audit() as audit:
        with host_transfer_evaluation("evaluation-1"):  # noqa: SIM117
            with host_transfer_phase("objective"):
                host_array(jax.numpy.asarray([1.0, 2.0], dtype=jax.numpy.float64))
        with host_transfer_phase("objective"):
            host_array(jax.numpy.asarray([3.0], dtype=jax.numpy.float64))

    summaries = audit.summary()
    assert tuple((item.phase, item.evaluation_id) for item in summaries) == (
        ("objective", "evaluation-1"),
        ("objective", None),
    )
    assert tuple(item.bytes for item in summaries) == (16, 8)


@pytest.mark.parametrize(
    "key_data",
    (
        CertificateProbeKeyData(0, 0),
        CertificateProbeKeyData(17, 29),
        CertificateProbeKeyData(
            MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
            MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
        ),
    ),
)
def test_certificate_probe_key_survives_strict_transfer_guard(
    key_data: CertificateProbeKeyData,
):
    with jax.transfer_guard("disallow"):
        key = runtime_certificate_probe_key(key_data)

    assert str(jax.random.key_impl(key)) == MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL
    np.testing.assert_array_equal(
        np.asarray(jax.device_get(jax.random.key_data(key))),
        np.asarray(key_data.words, dtype=np.uint32),
    )


def test_fresh_certificate_probe_key_data_mints_both_words(monkeypatch):
    words = iter((11, 23))

    def next_word(bit_count: int) -> int:
        assert bit_count == 32
        return next(words)

    monkeypatch.setattr(numerical_policy.secrets, "randbits", next_word)

    assert fresh_certificate_probe_key_data() == CertificateProbeKeyData(11, 23)


def test_certificate_probe_authority_round_trips_both_uint32_words():
    authority = CertificateProbeAuthority(
        source="supplied_replay",
        key_data=CertificateProbeKeyData(
            MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX,
            MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX - 1,
        ),
    )

    payload = authority.as_json()

    assert payload["prng_impl"] == MIXED_DENSE_IR_CERTIFICATE_PRNG_IMPL
    assert payload["sampling_model"] == MIXED_DENSE_IR_CERTIFICATE_PROBE_SAMPLING_MODEL
    assert payload["probability_model"] == MIXED_DENSE_IR_CERTIFICATE_PROBABILITY_MODEL
    assert CertificateProbeAuthority.from_json(payload) == authority


def test_certificate_probe_authority_distinguishes_fresh_from_replay(monkeypatch):
    monkeypatch.setattr(
        numerical_policy,
        "fresh_certificate_probe_key_data",
        lambda: CertificateProbeKeyData(11, 23),
    )
    replay_key_data = CertificateProbeKeyData(29, 31)

    fresh = resolve_certificate_probe_authority(None)
    replay = resolve_certificate_probe_authority(replay_key_data)

    assert fresh == CertificateProbeAuthority(
        source="fresh_runtime",
        key_data=CertificateProbeKeyData(11, 23),
    )
    assert replay == CertificateProbeAuthority(
        source="supplied_replay",
        key_data=replay_key_data,
    )


def test_certificate_probe_evidence_round_trip_binds_trust_to_fresh_challenge():
    key_data = CertificateProbeKeyData(11, 23)
    evidence = CertificateProbeEvidence(
        authority=CertificateProbeAuthority(
            source="fresh_runtime",
            key_data=key_data,
        ),
        observed_key_data=key_data,
        active=True,
        proposal_trusted=True,
        fp64_rebuild_count=0,
        fallback_attempted=False,
        fallback_success=False,
    )

    restored = CertificateProbeEvidence.from_json(
        evidence.as_json(),
        require_claim_eligible=True,
    )

    assert restored == evidence


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"observed_key_data": CertificateProbeKeyData(29, 31)}, "observed key"),
        ({"proposal_trusted": False}, "rebuild decisions"),
        (
            {
                "proposal_trusted": False,
                "fp64_rebuild_count": 1,
                "fallback_attempted": False,
            },
            "fallback evidence",
        ),
        (
            {
                "proposal_trusted": False,
                "fp64_rebuild_count": 1,
                "fallback_attempted": True,
                "fallback_success": False,
            },
            "did not certify",
        ),
        ({"fallback_success": True}, "success without an attempt"),
    ),
)
def test_certificate_probe_evidence_rejects_inconsistent_decisions(overrides, message):
    key_data = CertificateProbeKeyData(11, 23)
    fields = {
        "authority": CertificateProbeAuthority(
            source="fresh_runtime",
            key_data=key_data,
        ),
        "observed_key_data": key_data,
        "active": True,
        "proposal_trusted": True,
        "fp64_rebuild_count": 0,
        "fallback_attempted": False,
        "fallback_success": False,
        **overrides,
    }
    evidence = CertificateProbeEvidence(**fields)

    with pytest.raises(ValueError, match=message):
        evidence.require_valid_for_mixed()


def test_replay_certificate_probe_evidence_cannot_authorize_fresh_claim():
    key_data = CertificateProbeKeyData(11, 23)
    evidence = CertificateProbeEvidence(
        authority=CertificateProbeAuthority(
            source="supplied_replay",
            key_data=key_data,
        ),
        observed_key_data=key_data,
        active=True,
        proposal_trusted=True,
        fp64_rebuild_count=0,
        fallback_attempted=False,
        fallback_success=False,
    )

    with pytest.raises(ValueError, match="cannot authorize"):
        CertificateProbeEvidence.from_json(
            evidence.as_json(),
            require_claim_eligible=True,
        )


def test_certificate_probe_authority_preserves_both_words_with_x64_disabled():
    repo_root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root / "src")
    environment["JAX_ENABLE_X64"] = "False"
    environment["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        (
            sys.executable,
            str(repo_root / "tests" / "subprocess" / "jax_runtime_cases.py"),
            "certificate-probe-key-x64-disabled",
        ),
        check=False,
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    "serialized_key_data",
    (
        17,
        [17],
        [17, 23, 29],
        [True, 23],
        [17, -1],
        [17, MIXED_DENSE_IR_CERTIFICATE_KEY_WORD_MAX + 1],
    ),
)
def test_certificate_probe_key_data_rejects_noncanonical_json(
    serialized_key_data: object,
):
    with pytest.raises(ValueError, match="Certificate probe key"):
        CertificateProbeKeyData.from_json(serialized_key_data)
