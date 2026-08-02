"""Fail-closed contracts for bounded Boozer trial evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from benchmarks.boozer_trial_diagnostic import (
    JoinedBoozerTrialRecord,
    LineSearchTrialEvidence,
    ObjectiveTrialEvidence,
    TrialKey,
    parameter_sha256,
    run_boozer_host_diagnostic,
    validate_boozer_trial_trace,
    write_boozer_trial_trace,
)
from benchmarks.fixtures.custom_quasi_newton import Fixture, ObjectiveTrialEvaluation


def _objective(parameter_hash: str) -> ObjectiveTrialEvidence:
    return ObjectiveTrialEvidence(
        raw_objective=1.0,
        raw_objective_certified=True,
        filtered_objective=1.0,
        gradient_inf_norm=0.5,
        gradient_finite=True,
        gradient_source="candidate",
        gradient_source_parameter_sha256=parameter_hash,
        predictor_kind="baseline",
        predictor_success=True,
        primal_success=True,
        adjoint_success=True,
        newton_success=True,
        newton_stop_reason_code=0,
        newton_accepted_iterations=1,
        newton_attempted_iterations=1,
        newton_last_linear_solve_success=True,
        inner_penalty_residual_l2=1.0e-12,
        inner_final_gradient_inf_norm=1.0e-12,
    )


def _records(
    parameters: np.ndarray, count: int = 3
) -> tuple[tuple[JoinedBoozerTrialRecord, ...], dict[str, np.ndarray]]:
    parameter_hash = parameter_sha256(parameters)
    records: list[JoinedBoozerTrialRecord] = []
    for index in range(count):
        phase = (
            "initial"
            if index == 0
            else "final_refresh"
            if index == count - 1
            else "line_search"
        )
        line_search = (
            LineSearchTrialEvidence(None, None, None, None, None)
            if phase != "line_search"
            else LineSearchTrialEvidence(index, 0.5, -0.25, -0.1, -0.2)
        )
        records.append(
            JoinedBoozerTrialRecord(
                key=TrialKey(index, parameter_hash),
                phase=phase,
                objective=_objective(parameter_hash),
                line_search=line_search,
                parameter_archive_key=parameter_hash,
                parameter_shape=(parameters.size,),
            )
        )
    return tuple(records), {parameter_hash: parameters}


def test_trial_trace_round_trip_binds_records_parameters_and_bounds(
    tmp_path: Path,
) -> None:
    records, parameters = _records(np.asarray([1.0, 2.0], dtype=np.float64))
    manifest = write_boozer_trial_trace(
        tmp_path / "trial.json",
        provider="custom",
        production_route="custom_bfgs_stepwise",
        maxiter=1000,
        maxls=20,
        records=records,
        parameters_by_sha256=parameters,
    )

    summary = validate_boozer_trial_trace(
        manifest,
        expected_provider="custom",
        expected_production_route="custom_bfgs_stepwise",
        expected_maxiter=1000,
        expected_evaluations=2,
    )

    assert summary.record_count == 3
    assert summary.max_records == 20_002
    assert summary.parameter_bytes == 16


def test_trial_trace_rejects_empty_placeholder(tmp_path: Path) -> None:
    placeholder = tmp_path / "trial.json"
    placeholder.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        validate_boozer_trial_trace(
            placeholder,
            expected_provider="custom",
            expected_production_route="custom_bfgs_stepwise",
            expected_maxiter=1000,
            expected_evaluations=2,
        )


def test_trial_trace_rejects_parameter_hash_tampering(tmp_path: Path) -> None:
    records, parameters = _records(np.asarray([1.0, 2.0], dtype=np.float64))
    manifest = write_boozer_trial_trace(
        tmp_path / "trial.json",
        provider="custom",
        production_route="custom_bfgs_stepwise",
        maxiter=1000,
        maxls=20,
        records=records,
        parameters_by_sha256=parameters,
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["parameters_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="parameter archive checksum"):
        validate_boozer_trial_trace(
            manifest,
            expected_provider="custom",
            expected_production_route="custom_bfgs_stepwise",
            expected_maxiter=1000,
            expected_evaluations=2,
        )


def test_writer_rejects_unbounded_record_count(tmp_path: Path) -> None:
    records, parameters = _records(np.asarray([1.0], dtype=np.float64), count=4)

    with pytest.raises(ValueError, match="record count"):
        write_boozer_trial_trace(
            tmp_path / "trial.json",
            provider="custom",
            production_route="custom_bfgs_stepwise",
            maxiter=1,
            maxls=1,
            records=records,
            parameters_by_sha256=parameters,
        )


def test_host_diagnostic_correlates_objective_and_line_search_trials(
    tmp_path: Path,
) -> None:
    def evaluate(parameters: np.ndarray) -> ObjectiveTrialEvaluation:
        gradient = np.asarray(parameters, dtype=np.float64)
        value = float(0.5 * np.dot(gradient, gradient))
        return ObjectiveTrialEvaluation(
            raw_objective=value,
            raw_objective_certified=True,
            filtered_objective=value,
            gradient=gradient,
            gradient_source="candidate",
            predictor_kind=None,
            predictor_success=None,
            primal_success=True,
            adjoint_success=True,
            newton_success=True,
            newton_stop_reason_code=0,
            newton_accepted_iterations=1,
            newton_attempted_iterations=1,
            newton_last_linear_solve_success=True,
            inner_penalty_residual_l2=0.0,
            inner_final_gradient_inf_norm=0.0,
        )

    fixture_case = Fixture(
        name="boozer",
        objective=lambda x: 0.5 * np.dot(x, x),
        initial=np.asarray([2.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_boozer_diagnostic_contract",
        certificate="synthetic diagnostic join contract",
        method="bfgs",
        trial_evaluator=evaluate,
        native_trial_evaluator=evaluate,
    )

    result = run_boozer_host_diagnostic(
        fixture_case,
        provider="custom",
        manifest_path=tmp_path / "trial.json",
        maxiter=5,
        maxls=20,
    )

    assert result.converged
    assert result.status == 0
    assert result.evaluations >= 2
    summary = validate_boozer_trial_trace(
        result.trial_trace,
        expected_provider="custom",
        expected_production_route="custom_bfgs_stepwise",
        expected_maxiter=5,
        expected_evaluations=result.evaluations,
    )
    assert summary.record_count == result.evaluations + 1
