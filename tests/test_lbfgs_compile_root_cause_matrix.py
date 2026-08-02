"""Contracts for the bounded L-BFGS compile root-cause matrix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from benchmarks.lbfgs_compile_root_cause_matrix import (
    JsonObject,
    _progress_path,
    default_cases,
    payload_summary,
    watchdog_verdict,
)


def test_progress_path_is_a_sibling_of_case_output() -> None:
    output = Path(".artifacts/root-cause.case.json")

    assert _progress_path(output) == Path(".artifacts/root-cause.case.progress.json")


def test_default_matrix_covers_declared_axes() -> None:
    cases = default_cases()

    assert {case.objective for case in cases} == {"quadratic", "coil47"}
    assert {case.maxcor for case in cases} == {10, 300}
    assert {case.compile_only for case in cases} == {False, True}
    assert {case.maxiter for case in cases} == {2, 20}
    assert len({case.name for case in cases}) == len(cases)
    assert all(case.skip_lowering == (not case.compile_only) for case in cases)


def test_payload_summary_keeps_compile_and_result_fields() -> None:
    result_summary: JsonObject = {
        "converged": False,
        "status": 1,
        "iterations": 2,
        "evaluations": 3,
        "objective": 1.25,
        "finite": True,
    }
    payload = cast(
        JsonObject,
        {
            "case": {"objective_kind": "quadratic"},
            "runtime_compile": {
                "compiled_executable_count": 5,
                "peak_host_rss_bytes": 1234,
            },
            "repeated_call_compile": {
                "compile_log_count": 5,
                "recompiled_on_repeated_calls": False,
                "result_summary": {
                    **result_summary,
                    "run_seconds": 0.25,
                    "iteration_progress": [{"iteration": 1, "step_s": 0.1}],
                },
            },
            "objective_timing": {
                "cold_s": 0.5,
                "warm_s": [0.01, 0.01],
            },
            "summaries": [
                {"label": "step", "text_bytes": 10, "lower_s": 0.5},
            ],
        },
    )

    assert payload_summary(payload) == {
        "objective_kind": "quadratic",
        "compiled_executable_count": 5,
        "compile_log_count": 5,
        "recompiled_on_repeated_calls": False,
        "peak_host_rss_bytes": 1234,
        "result_summary": {
            **result_summary,
            "run_seconds": 0.25,
            "iteration_progress": [{"iteration": 1, "step_s": 0.1}],
        },
        "lowered_text_bytes": 10,
        "lower_s": 0.5,
        "objective_timing": {
            "cold_s": 0.5,
            "warm_s": [0.01, 0.01],
        },
        "iteration_progress": [{"iteration": 1, "step_s": 0.1}],
        "solver_run_seconds": 0.25,
    }


def test_watchdog_verdict_is_fail_closed() -> None:
    assert watchdog_verdict(returncode=0, timed_out=False, rss_exceeded=False) == (
        "completed"
    )
    assert watchdog_verdict(returncode=-15, timed_out=True, rss_exceeded=False) == (
        "timeout"
    )
    assert watchdog_verdict(returncode=-9, timed_out=False, rss_exceeded=True) == (
        "rss_limit"
    )


def test_compile_design_ab_receipt_is_self_consistent() -> None:
    receipt = (
        Path(__file__).resolve().parents[1]
        / "docs/receipts/custom-quasi-newton/compile-design-ab"
    )
    manifest = json.loads((receipt / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((receipt / "metrics.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["verdict"] == "diagnostic-pass-not-promotion"
    assert manifest["exit_codes"] == {"design_a": 124, "design_b": 0}
    assert metrics["comparison"]["design_b_completed_within_watchdog"] is True
    assert metrics["comparison"]["design_a_completed_within_watchdog"] is False
    for artifact in manifest["artifacts"]:
        path = receipt / artifact["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
