"""Unit coverage for L-BFGS-B compile diagnostic classification."""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest
from benchmarks.lbfgs_ondevice_compile_shape import (
    _LBFGS_MONOLITHIC_COMPILE_FRAGMENTS,
    _LBFGS_PRIVATE_COMPILE_FRAGMENTS,
    _LBFGS_STEPWISE_COMPILE_FRAGMENTS,
    _build_kernel_cases,
    _comparison,
    _CompileCounter,
    _ProgressRecorder,
    _sum_fragment_counts,
    _summarize_bounded_result,
)


def _log_record(message: str) -> logging.LogRecord:
    return logging.LogRecord("jax", logging.WARNING, __file__, 1, message, (), None)


def test_progress_recorder_persists_atomic_phase_checkpoints(tmp_path) -> None:
    progress_path = tmp_path / "compile.progress.json"
    recorder = _ProgressRecorder(progress_path)

    recorder.record(
        "lower_complete",
        {"label": "step", "text_bytes": 123, "lower_s": 0.5},
    )

    payload = json.loads(progress_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["events"][-1] == {
        "event": "lower_complete",
        "elapsed_s": payload["events"][-1]["elapsed_s"],
        "label": "step",
        "lower_s": 0.5,
        "text_bytes": 123,
    }


def test_compile_counter_classifies_stepwise_and_monolithic_lanes() -> None:
    counter = _CompileCounter(_LBFGS_PRIVATE_COMPILE_FRAGMENTS)

    counter.emit(_log_record("Compiling jit(lbfgs_private_initial_state_solver)"))
    counter.emit(_log_record("Compiling jit(lbfgs_private_macro_step_solver)"))
    counter.emit(_log_record("Compiling jit(lbfgs_private_macro_step_solver)"))
    counter.emit(_log_record("Compiling jit(lbfgs_private_result_payload_solver)"))
    counter.emit(_log_record("Compiling jit(lbfgs_private_monolithic_mainlb_solver)"))
    counter.emit(_log_record("Finished tracing jit(lbfgs_private_macro_step_solver)"))

    assert counter.count == 5
    assert (
        _sum_fragment_counts(
            counter.counts_by_fragment,
            _LBFGS_STEPWISE_COMPILE_FRAGMENTS,
        )
        == 4
    )
    assert (
        _sum_fragment_counts(
            counter.counts_by_fragment,
            _LBFGS_MONOLITHIC_COMPILE_FRAGMENTS,
        )
        == 1
    )


def test_compile_shape_comparison_reports_specialized_step_reduction() -> None:
    summaries = [
        {
            "label": "old_monolithic_full_solve",
            "text_bytes": 5000,
            "jaxpr_text_bytes": 4500,
        },
        {
            "label": "old_generic_step_to_next_observable",
            "text_bytes": 3000,
            "jaxpr_text_bytes": 2500,
        },
        {
            "label": "step_from_start_to_next_observable",
            "text_bytes": 1200,
            "jaxpr_text_bytes": 1000,
        },
        {
            "label": "step_from_search_to_next_observable",
            "text_bytes": 1500,
            "jaxpr_text_bytes": 1300,
        },
        {
            "label": "reenter_from_new_x",
            "text_bytes": 1800,
            "jaxpr_text_bytes": 1600,
        },
        {
            "label": "result_payload",
            "text_bytes": 400,
            "jaxpr_text_bytes": 350,
        },
    ]

    comparison = _comparison(summaries)

    assert comparison["specialized_step_text_bytes_reduced_vs_generic"] == 1
    assert comparison["specialized_step_jaxpr_text_bytes_reduced_vs_generic"] == 1


def test_compile_shape_default_excludes_memory_heavy_legacy_kernels() -> None:
    cases = _build_kernel_cases(
        dimension=2,
        maxcor=3,
        maxiter=2,
        maxfun=8,
        maxls=20,
        ftol=0.0,
        gtol=1.0e-8,
    )

    labels = {case.label for case in cases}
    assert "old_generic_step_to_next_observable" not in labels
    assert "old_monolithic_full_solve" not in labels

    legacy_cases = _build_kernel_cases(
        dimension=2,
        maxcor=3,
        maxiter=2,
        maxfun=8,
        maxls=20,
        ftol=0.0,
        gtol=1.0e-8,
        include_legacy_kernels=True,
    )
    legacy_labels = {case.label for case in legacy_cases}
    assert "old_generic_step_to_next_observable" in legacy_labels
    assert "old_monolithic_full_solve" in legacy_labels


def test_compile_shape_objective_contract_rejects_mismatched_inputs() -> None:
    with pytest.raises(ValueError, match="coil47.*dimension=47"):
        _build_kernel_cases(
            dimension=2,
            maxcor=3,
            maxiter=2,
            maxfun=8,
            maxls=20,
            ftol=0.0,
            gtol=1.0e-8,
            objective="coil47",
        )

    with pytest.raises(ValueError, match="unknown compile-shape objective"):
        _build_kernel_cases(
            dimension=2,
            maxcor=3,
            maxiter=2,
            maxfun=8,
            maxls=20,
            ftol=0.0,
            gtol=1.0e-8,
            objective="unknown",
        )


def test_bounded_result_summary_accepts_finite_capped_result() -> None:
    summary = _summarize_bounded_result(
        SimpleNamespace(
            converged=False,
            status=1,
            k=2,
            nfev=5,
            f_k=1.25,
            g_k=[0.5, -0.25],
        )
    )

    assert summary == {
        "converged": False,
        "status": 1,
        "iterations": 2,
        "evaluations": 5,
        "objective": 1.25,
        "finite": True,
    }
