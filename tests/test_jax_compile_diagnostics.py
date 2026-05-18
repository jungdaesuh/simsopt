"""Parser invariants for JAX compile-diagnostic accounting.

This file verifies the bookkeeping/parsing invariants of the
``JAX_COMPILE_DIAGNOSTICS`` recorder threaded through the single-stage
example. It does not test physics parity and intentionally avoids running
the full single-stage subprocess.
"""

from __future__ import annotations

import logging
from pathlib import Path

from examples.single_stage_optimization.SINGLE_STAGE.single_stage_banana_example import (
    _JaxCompileDiagnosticsRecorder,
    maybe_record_jax_compile_diagnostics,
)


def _record(message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="jax",
        level=logging.WARNING,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_compile_diagnostic_recorder_parser_accounting_invariants():
    recorder = _JaxCompileDiagnosticsRecorder(sample_limit=1)

    recorder.emit(_record("Compiling target_lane_value with global shapes and types"))
    recorder.emit(_record("Compiling target_lane_value with global shapes and types"))
    recorder.emit(_record("Compiling  with global shapes and types"))
    recorder.emit(_record("TRACING CACHE MISS at foo.py:12 (tracing key mismatch)"))
    recorder.emit(_record("TRACING CACHE MISS at foo.py:12 (tracing key mismatch)"))
    recorder.emit(_record("TRACING CACHE MISS at  (tracing key mismatch)"))

    diagnostics = recorder.summary()
    compile_event_count = diagnostics["compile_event_count"]
    cache_miss_count = diagnostics["cache_miss_count"]
    compile_target_parse_miss_count = diagnostics["compile_target_parse_miss_count"]
    cache_miss_site_parse_miss_count = diagnostics["cache_miss_site_parse_miss_count"]

    assert diagnostics["compile_targets"] == {"target_lane_value": 2}
    assert diagnostics["cache_miss_sites"] == {"foo.py:12": 2}
    assert diagnostics["compile_messages"] == [
        "Compiling target_lane_value with global shapes and types"
    ]
    assert diagnostics["cache_miss_messages"] == [
        "TRACING CACHE MISS at foo.py:12 (tracing key mismatch)"
    ]
    assert sum(diagnostics["compile_targets"].values()) == (
        compile_event_count - compile_target_parse_miss_count
    )
    assert sum(diagnostics["cache_miss_sites"].values()) == (
        cache_miss_count - cache_miss_site_parse_miss_count
    )


def test_compile_diagnostic_recorder_ignores_unrelated_warnings():
    recorder = _JaxCompileDiagnosticsRecorder()

    recorder.emit(_record("unrelated warning"))

    assert recorder.summary() == {
        "compile_event_count": 0,
        "cache_miss_count": 0,
        "compile_target_parse_miss_count": 0,
        "cache_miss_site_parse_miss_count": 0,
        "compile_targets": {},
        "cache_miss_sites": {},
        "compile_messages": [],
        "cache_miss_messages": [],
    }


def test_compile_diagnostic_context_manager_captures_jax_logger_messages():
    with maybe_record_jax_compile_diagnostics(True) as recorder:
        logging.getLogger("jax").warning(
            "Compiling smoke_target with global shapes and types"
        )

    assert recorder is not None
    assert recorder.summary()["compile_targets"] == {"smoke_target": 1}


def test_single_stage_runtime_wires_compile_diagnostic_results():
    repo_root = Path(__file__).resolve().parents[1]
    source = (
        repo_root
        / "examples/single_stage_optimization/SINGLE_STAGE"
        / "single_stage_banana_example.py"
    ).read_text(encoding="utf-8")

    assert "maybe_record_jax_compile_diagnostics(" in source
    assert "args.record_jax_compile_diagnostics and use_target_lane" in source
    assert "jax_compile_diagnostics_recorder.summary()" in source
    assert 'results["JAX_COMPILE_DIAGNOSTICS"] = jax_compile_diagnostics' in source
    assert '"jax_compile_diagnostics.json"' in source
