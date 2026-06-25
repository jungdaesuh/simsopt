"""Unit coverage for L-BFGS-B compile diagnostic classification."""

from __future__ import annotations

import logging

from benchmarks.lbfgs_ondevice_compile_shape import (
    _CompileCounter,
    _LBFGS_MONOLITHIC_COMPILE_FRAGMENTS,
    _LBFGS_PRIVATE_COMPILE_FRAGMENTS,
    _LBFGS_STEPWISE_COMPILE_FRAGMENTS,
    _comparison,
    _sum_fragment_counts,
)


def _log_record(message: str) -> logging.LogRecord:
    return logging.LogRecord("jax", logging.WARNING, __file__, 1, message, (), None)


def test_compile_counter_classifies_stepwise_and_monolithic_lanes() -> None:
    counter = _CompileCounter(_LBFGS_PRIVATE_COMPILE_FRAGMENTS)

    counter.emit(_log_record("Compiling jit(lbfgs_private_initial_state_solver)"))
    counter.emit(_log_record("Compiling jit(lbfgs_private_macro_step_solver)"))
    counter.emit(_log_record("Compiling jit(lbfgs_private_macro_step_solver)"))
    counter.emit(_log_record("Compiling jit(lbfgs_private_result_payload_solver)"))
    counter.emit(_log_record("Compiling jit(lbfgs_private_monolithic_mainlb_solver)"))
    counter.emit(_log_record("Finished tracing jit(lbfgs_private_macro_step_solver)"))

    assert counter.count == 5
    assert _sum_fragment_counts(
        counter.counts_by_fragment,
        _LBFGS_STEPWISE_COMPILE_FRAGMENTS,
    ) == 4
    assert _sum_fragment_counts(
        counter.counts_by_fragment,
        _LBFGS_MONOLITHIC_COMPILE_FRAGMENTS,
    ) == 1


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
