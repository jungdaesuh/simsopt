"""Contract for wiring ``--record-jax-compile-diagnostics`` into the parity harness.

``benchmarks/single_stage_init_parity.py`` builds the JAX-vs-cpp/CPU single-stage
comparison. It already shipped the *relay* machinery to forward compile/cache-miss
diagnostics to the child (``resolve_target_lane_compile_diagnostics`` +
``_append_optional_single_stage_flags``), but no CLI flag fed it, so
``enable_compile_diagnostics`` was permanently ``False`` and the fair launcher could
not separate one-time compile cost from steady-state throughput (nor surface GPU
XLA recompile counts).

These tests pin the now-complete contract:

- the harness exposes ``--record-jax-compile-diagnostics`` (default off);
- the pure relay appends the child flag iff enabled, and host-callback diagnostic
  modes suppress it (they are mutually exclusive with compile diagnostics in the
  child);
- end to end, the parsed flag reaches the JAX target-lane child command, gated to
  ``backend == "jax"`` so the cpp/CPU reference lane never carries a JAX-only flag.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

from benchmarks.single_stage_init_parity import (
    _append_optional_single_stage_flags,
    _reference_case_backend,
    _run_single_stage_case,
    parse_args,
    resolve_target_lane_compile_diagnostics,
)

FLAG = "--record-jax-compile-diagnostics"


def _parse(extra: list[str]):
    argv = [
        "single_stage_init_parity.py",
        "--output-json",
        "/tmp/ss_out.json",
        "--platform",
        "cpu",
        *extra,
    ]
    with mock.patch.object(sys, "argv", argv):
        return parse_args()


def _relay(enable: bool, **over) -> list[str]:
    command: list[str] = []
    kwargs = dict(
        benchmark_mode=False,
        profile_target_lane=False,
        profile_target_lane_only=False,
        diagnose_target_lane_scaled_phase1=False,
        record_target_lane_invalid_state_events=False,
        profile_target_lane_batch_size=None,
        enable_compile_diagnostics=enable,
        jax_profile_dir=None,
        experimental_target_lane_value_and_grad=False,
        disable_target_lane_success_filter=False,
        record_objective_evaluation_trace=False,
        record_target_optimizer_state_trace=False,
    )
    kwargs.update(over)
    _append_optional_single_stage_flags(command, **kwargs)
    return command


def _capture_child_command(monkeypatch, tmp_path, args, backend, platform) -> list[str]:
    captured: dict[str, list[str]] = {}

    def fake_run(script_path, command, **kwargs):
        captured["command"] = list(command)

    def fake_payload(output_root):
        return {}, Path(output_root) / "results.json"

    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity.run_python_script", fake_run
    )
    monkeypatch.setattr(
        "benchmarks.single_stage_init_parity._load_single_stage_final_payload",
        fake_payload,
    )
    _run_single_stage_case(
        args,
        backend,
        platform=platform,
        load_surface_gamma=False,
        output_root=tmp_path,
    )
    return captured["command"]


def test_cli_exposes_compile_diagnostics_flag_default_off():
    assert _parse([]).record_jax_compile_diagnostics is False
    assert _parse([FLAG]).record_jax_compile_diagnostics is True


def test_warm_start_auto_reference_backend_stays_jax_cpu():
    args = _parse(["--warm-start-run-dir", "/tmp/seed"])
    assert _reference_case_backend(args) == "jax"


def test_warm_start_can_force_native_cpu_reference_backend():
    args = _parse(
        [
            "--warm-start-run-dir",
            "/tmp/seed",
            "--reference-backend",
            "cpu",
        ]
    )
    assert _reference_case_backend(args) == "cpu"


def test_reference_backend_default_fixture_stays_native_cpu():
    args = _parse([])
    assert _reference_case_backend(args) == "cpu"


def test_relay_appends_child_flag_only_when_enabled():
    assert FLAG in _relay(True)
    assert FLAG not in _relay(False)


def test_host_callback_modes_suppress_compile_diagnostics():
    # Compile diagnostics and host-callback diagnostics cannot run together in the
    # child, so the resolver must suppress the flag instead of emitting a conflict.
    assert resolve_target_lane_compile_diagnostics(
        enable_compile_diagnostics=True,
        diagnose_target_lane_scaled_phase1=False,
        record_target_lane_invalid_state_events=False,
    ) == (True, None)
    assert FLAG not in _relay(True, diagnose_target_lane_scaled_phase1=True)
    assert FLAG not in _relay(True, record_target_lane_invalid_state_events=True)


def test_flag_reaches_jax_target_child_command(monkeypatch, tmp_path):
    args = _parse([FLAG])
    command = _capture_child_command(monkeypatch, tmp_path, args, "jax", "cpu")
    assert FLAG in command


def test_cuda_host_jax_child_uses_host_jax_boozer_backend(monkeypatch, tmp_path):
    args = _parse(["--optimizer-backend", "host-jax"])
    command = _capture_child_command(monkeypatch, tmp_path, args, "jax", "cuda")
    index = command.index("--boozer-optimizer-backend")
    assert command[index + 1] == "host-jax"


def test_cuda_scipy_jax_child_keeps_ondevice_boozer_backend(monkeypatch, tmp_path):
    args = _parse(["--optimizer-backend", "scipy-jax"])
    command = _capture_child_command(monkeypatch, tmp_path, args, "jax", "cuda")
    index = command.index("--boozer-optimizer-backend")
    assert command[index + 1] == "ondevice"


def test_child_command_forwards_boozer_least_squares_algorithm(monkeypatch, tmp_path):
    args = _parse([])
    args.boozer_least_squares_algorithm = "lm"
    command = _capture_child_command(monkeypatch, tmp_path, args, "jax", "cuda")
    index = command.index("--boozer-least-squares-algorithm")
    assert command[index + 1] == "lm"


def test_flag_absent_from_jax_child_when_not_requested(monkeypatch, tmp_path):
    args = _parse([])
    command = _capture_child_command(monkeypatch, tmp_path, args, "jax", "cpu")
    assert FLAG not in command


def test_flag_never_reaches_cpp_reference_child(monkeypatch, tmp_path):
    # Even with the CLI flag set, the cpp/CPU reference lane must not carry the
    # JAX-only diagnostics flag (the ``backend == "jax"`` gate).
    args = _parse([FLAG])
    command = _capture_child_command(monkeypatch, tmp_path, args, "cpu", "cpu")
    assert FLAG not in command


def test_explicit_donor_outer_run_reuses_resolved_startup_solve(
    monkeypatch,
    tmp_path,
):
    args = _parse(["--warm-start-run-dir", "/tmp/seed", "--maxiter", "1500"])

    command = _capture_child_command(monkeypatch, tmp_path, args, "cpu", "cpu")

    assert "--reuse-resolved-warm-start-solve" in command
    assert "--init-only" not in command


def test_init_only_donor_run_replays_setup_solve(monkeypatch, tmp_path):
    args = _parse(["--warm-start-run-dir", "/tmp/seed", "--maxiter", "0"])

    command = _capture_child_command(monkeypatch, tmp_path, args, "cpu", "cpu")

    assert "--reuse-resolved-warm-start-solve" not in command
    assert "--init-only" in command
