"""Unit coverage for L-BFGS-B compile diagnostic classification."""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from benchmarks import custom_quasi_newton_runtime as runtime
from benchmarks import lbfgs_ondevice_compile_shape as compile_shape
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


def _cpu_device_identity() -> runtime.DeviceIdentity:
    return runtime.DeviceIdentity(
        requested_device="cpu",
        backend="cpu",
        platform="cpu",
        jax_device="TFRT_CPU_0",
        device_kind="cpu",
        device_id=0,
        process_index=0,
        gpu_uuid=None,
        gpu_model=None,
        compute_capability=None,
        total_memory_bytes=None,
        driver_version=None,
        cuda_version=None,
        visible_devices=None,
        hostname="test-host",
        scheduler_job_id=None,
    )


def _prepared_provider_compiles(
    provider: compile_shape._CompileProvider,
) -> tuple[object, tuple[compile_shape._CapturedProviderCompile, ...]]:
    if provider == "custom":
        executables = tuple(object() for _ in range(6))
        prepared = SimpleNamespace(
            program=SimpleNamespace(
                initial_state=executables[0],
                value_and_grad=executables[1],
                advance_from_start=executables[2],
                advance_from_search=executables[3],
                reenter_new_x=executables[4],
                result_payload=executables[5],
            )
        )
    else:
        executables = tuple(object() for _ in range(2))
        prepared = SimpleNamespace(
            step=executables[0],
            final_value_and_grad=executables[1],
        )
    captured = tuple(
        compile_shape._CapturedProviderCompile(
            executable=executable,
            stablehlo_module=f"jit_program_{index}",
            stablehlo_bytes=(index + 1) * 100,
            compile_s=(index + 1) * 0.25,
        )
        for index, executable in enumerate(executables)
    )
    return prepared, captured


@pytest.mark.parametrize(
    ("provider", "expected_labels"),
    [
        (
            "custom",
            [
                "initial_state",
                "value_and_grad",
                "advance_from_start",
                "advance_from_search",
                "reenter_new_x",
                "result_payload",
            ],
        ),
        ("optax", ["step", "final_value_and_grad"]),
    ],
)
def test_provider_compile_summary_labels_returned_executables_and_aggregates(
    provider: compile_shape._CompileProvider,
    expected_labels: list[str],
) -> None:
    prepared, captured = _prepared_provider_compiles(provider)

    programs, aggregate = compile_shape._summarize_provider_compiles(
        provider,
        prepared,
        captured,
    )

    assert [program["label"] for program in programs] == expected_labels
    assert cast(int, aggregate["stablehlo_bytes"]) == sum(
        record.stablehlo_bytes for record in captured
    )
    assert cast(float, aggregate["compile_s"]) == pytest.approx(
        sum(record.compile_s for record in captured)
    )
    assert cast(int, aggregate["compiled_executable_count"]) == len(expected_labels)
    assert (
        aggregate["compiled_executable_count_source"]
        == "observed_lowered_compile_calls"
    )


def test_provider_compile_summary_rejects_an_unreturned_executable() -> None:
    prepared, captured = _prepared_provider_compiles("optax")
    unmatched = replace(captured[1], executable=object())

    with pytest.raises(RuntimeError, match="unreturned executable"):
        compile_shape._summarize_provider_compiles(
            "optax",
            prepared,
            (captured[0], unmatched),
        )


@pytest.mark.parametrize(
    ("provider", "expected_route", "expected_count"),
    [("custom", "stepwise", 6), ("optax", "optax_lbfgs", 2)],
)
def test_payload_binds_provider_route_fixture_options_and_cpu_unavailability(
    monkeypatch: pytest.MonkeyPatch,
    provider: compile_shape._CompileProvider,
    expected_route: str,
    expected_count: int,
) -> None:
    prepared, captured = _prepared_provider_compiles(provider)
    monkeypatch.setattr(
        compile_shape,
        "_capture_provider_preparation",
        lambda *_args, **_kwargs: (prepared, captured, 3.5),
    )
    monkeypatch.setattr(
        compile_shape.runtime,
        "_checkout_provenance",
        lambda: ("candidate-head", True),
    )
    monkeypatch.setattr(
        compile_shape.runtime,
        "_device_identity",
        lambda _device: _cpu_device_identity(),
    )
    monkeypatch.setattr(compile_shape, "_git_text", lambda *_args: "")

    payload = compile_shape._provider_compile_payload(
        provider=provider,
        fixture_name="rosenbrock",
        device="cpu",
        intent="parity",
        maxiter=2,
        maxcor=4,
    )

    options = cast(dict[str, object], payload["options"])
    aggregate = cast(dict[str, object], payload["aggregate"])
    device_identity = cast(dict[str, object], payload["device_identity"])
    factory_options = cast(dict[str, object], payload["provider_factory_options"])
    assert payload["provider"] == provider
    assert payload["solver_route"] == expected_route
    assert payload["candidate_sha"] == "candidate-head"
    assert payload["candidate_sha_availability"] == "available"
    assert payload["fixture"] == "rosenbrock"
    assert payload["dtype"] == "float64"
    assert payload["parameter_shape"] == [2]
    assert payload["provider_preparation_s"] == 3.5
    assert options == {
        "device": "cpu",
        "intent": "parity",
        "maxcor": 4,
        "maxfun": None,
        "maxiter": 2,
        "ftol": runtime._SOLVER_FTOL,
        "gtol": runtime._SOLVER_GTOL,
        "maxls": runtime._SOLVER_MAXLS,
        "method": "lbfgs",
    }
    assert aggregate["compiled_executable_count"] == expected_count
    assert device_identity["gpu_uuid"] is None
    assert device_identity["cuda_version"] is None
    assert payload["gpu_identity_availability"] == "unavailable"
    assert payload["gpu_identity_unavailable_reason"] == "cpu_device"
    if provider == "custom":
        assert factory_options == {
            "maxcor": 4,
            "ftol": runtime._SOLVER_FTOL,
            "gtol": runtime._SOLVER_GTOL,
            "maxls": runtime._SOLVER_MAXLS,
            "x_dtype": "float64",
        }
    else:
        assert factory_options == {"memory_size": 4}


def test_dirty_checkout_marks_candidate_sha_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared, captured = _prepared_provider_compiles("optax")
    monkeypatch.setattr(
        compile_shape,
        "_capture_provider_preparation",
        lambda *_args, **_kwargs: (prepared, captured, 1.0),
    )
    monkeypatch.setattr(
        compile_shape.runtime,
        "_checkout_provenance",
        lambda: ("dirty-head", False),
    )
    monkeypatch.setattr(
        compile_shape.runtime,
        "_device_identity",
        lambda _device: _cpu_device_identity(),
    )
    monkeypatch.setattr(compile_shape, "_git_text", lambda *_args: " M benchmark.py")

    payload = compile_shape._provider_compile_payload(
        provider="optax",
        fixture_name="rosenbrock",
        device="cpu",
        intent="parity",
        maxiter=1,
        maxcor=2,
    )

    assert payload["git_commit"] == "dirty-head"
    assert payload["git_clean"] is False
    assert payload["candidate_sha"] is None
    assert payload["candidate_sha_availability"] == "unavailable"
    assert payload["candidate_sha_unavailable_reason"] == "dirty_worktree"


@pytest.mark.parametrize(
    ("provider", "expected_count"),
    [("custom", 6), ("optax", 2)],
)
def test_exact_runtime_factory_compiles_are_observed(
    provider: compile_shape._CompileProvider,
    expected_count: int,
) -> None:
    fixture_case = runtime.fixture("rosenbrock")
    x0 = np.asarray(fixture_case.initial, dtype=np.float64)

    prepared, captured, preparation_s = compile_shape._capture_provider_preparation(
        provider,
        fixture_case,
        x0,
        maxcor=2,
    )
    programs, aggregate = compile_shape._summarize_provider_compiles(
        provider,
        prepared,
        captured,
    )

    stablehlo_bytes = [cast(int, program["stablehlo_bytes"]) for program in programs]
    compile_seconds = [cast(float, program["compile_s"]) for program in programs]
    assert preparation_s > 0.0
    assert len(programs) == expected_count
    assert all(byte_count > 0 for byte_count in stablehlo_bytes)
    assert all(seconds >= 0.0 for seconds in compile_seconds)
    assert cast(int, aggregate["stablehlo_bytes"]) == sum(stablehlo_bytes)


def test_unknown_provider_fails_before_writing_an_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "unknown.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "lbfgs_ondevice_compile_shape.py",
            "--provider",
            "unknown",
            "--output-json",
            str(output),
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        compile_shape.main()

    assert not output.exists()
