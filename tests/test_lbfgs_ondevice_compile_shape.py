"""Unit coverage for L-BFGS-B compile diagnostic classification."""

from __future__ import annotations

import logging

from benchmarks.lbfgs_ondevice_compile_shape import (
    _CompileCounter,
    _LBFGS_MONOLITHIC_COMPILE_FRAGMENTS,
    _LBFGS_PRIVATE_COMPILE_FRAGMENTS,
    _LBFGS_STEPWISE_COMPILE_FRAGMENTS,
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
