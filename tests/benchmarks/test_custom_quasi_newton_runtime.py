"""Runner contract tests for custom quasi-Newton measurements."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from benchmarks import custom_quasi_newton_runtime as runtime
from benchmarks.fixtures.custom_quasi_newton import Fixture, fixture


class _Child:
    pid = 1234
    returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def communicate(self, timeout: float | None = None) -> tuple[str, str]:
        del timeout
        return "", ""


def test_provider_child_timeout_is_fail_closed(monkeypatch) -> None:
    child = _Child()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(runtime, "_PROVIDER_CHILD_TIMEOUT_SECONDS", 0)
    monkeypatch.setattr(runtime, "_PROVIDER_CHILD_POLL_SECONDS", 0)

    with pytest.raises(RuntimeError, match="second watchdog"):
        runtime._run_provider_child_process(["provider"])


def test_provider_child_rss_limit_is_fail_closed(monkeypatch) -> None:
    child = _Child()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *_a, **_k: child)
    monkeypatch.setattr(runtime, "_PROVIDER_CHILD_TIMEOUT_SECONDS", 120)
    monkeypatch.setattr(
        runtime, "_child_rss_kib", lambda _pid: runtime._PROVIDER_CHILD_RSS_LIMIT_KIB
    )

    with pytest.raises(RuntimeError, match="8-GiB RSS watchdog"):
        runtime._run_provider_child_process(["provider"])


def test_provider_child_discards_unbounded_stdout(monkeypatch) -> None:
    child = _Child()
    child.returncode = 0
    calls: dict[str, object] = {}

    def fake_popen(*_args, **kwargs):
        calls.update(kwargs)
        return child

    monkeypatch.setattr(runtime.subprocess, "Popen", fake_popen)

    runtime._run_provider_child_process(["provider"])

    assert calls["stdout"] is runtime.subprocess.DEVNULL


@pytest.mark.parametrize(
    ("iterations", "maxiter", "status", "success", "expected"),
    [
        (4, 20, 0, True, "converged"),
        (20, 20, 1, False, "iteration-limit"),
        (3, 20, 2, False, "line-search-failed"),
        (3, 20, 6, False, "nonfinite"),
        (3, 20, 99, False, "callback-stopped"),
        (3, 20, None, False, "failed"),
    ],
)
def test_stopping_reason_labels_terminal_state(
    iterations: int,
    maxiter: int,
    status: int | None,
    success: bool,
    expected: str,
) -> None:
    assert (
        runtime._stopping_reason(
            iterations=iterations,
            maxiter=maxiter,
            status=status,
            success=success,
        )
        == expected
    )


def test_runtime_environment_payload_records_device_allocator_contract(
    monkeypatch,
) -> None:
    monkeypatch.setenv("JAX_PLATFORMS", "cuda")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    payload = runtime._runtime_environment_payload()

    assert payload["JAX_PLATFORMS"] == "cuda"
    assert payload["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert payload["XLA_PYTHON_CLIENT_ALLOCATOR"] == "platform"
    assert "SIMSOPT_BACKEND_MODE" in payload


@pytest.mark.parametrize(
    ("device", "intent", "mode"),
    (
        ("cpu", "fast", "jax_cpu_fast"),
        ("cpu", "parity", "jax_cpu_parity"),
        ("gpu", "fast", "jax_gpu_fast"),
        ("gpu", "parity", "jax_gpu_parity"),
    ),
)
def test_intent_environment_requires_canonical_profile(
    monkeypatch,
    device: str,
    intent: str,
    mode: str,
) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", mode)

    assert runtime._validate_intent_environment(device, intent) == mode


def test_intent_environment_rejects_missing_or_mismatched_profile(monkeypatch) -> None:
    monkeypatch.delenv("SIMSOPT_BACKEND_MODE", raising=False)
    with pytest.raises(RuntimeError, match="expected 'jax_cpu_parity'"):
        runtime._validate_intent_environment("cpu", "parity")


def test_main_rejects_intent_before_fixture_construction(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_fast")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "custom_quasi_newton_runtime.py",
            "--device",
            "cpu",
            "--intent",
            "parity",
            "--output",
            str(tmp_path),
        ],
    )

    def fixture_must_not_run(_name: str) -> Fixture:
        pytest.fail("profile validation must run before fixture construction")

    monkeypatch.setattr(runtime, "fixture", fixture_must_not_run)
    with pytest.raises(RuntimeError, match="expected 'jax_cpu_parity'"):
        runtime.main()

    monkeypatch.setenv("SIMSOPT_BACKEND_MODE", "jax_cpu_fast")
    with pytest.raises(RuntimeError, match="expected 'jax_cpu_parity'"):
        runtime._validate_intent_environment("cpu", "parity")


@pytest.mark.parametrize(
    "receipt_name",
    (
        "rosenbrock-pre-refactor-trajectory",
        "bfgs-pre-refactor-trajectory",
    ),
)
def test_tracked_pre_refactor_trajectory_receipt_is_self_consistent(
    receipt_name: str,
) -> None:
    receipt = (
        Path(__file__).resolve().parents[2]
        / "docs/receipts/custom-quasi-newton"
        / receipt_name
    )
    manifest = json.loads((receipt / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["verdict"] == "diagnostic-pass-not-promotion"
    for artifact in manifest["artifacts"]:
        path = receipt / artifact["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == artifact["sha256"]

    pre_refactor = (receipt / "raw/pre_refactor.json").read_bytes()
    candidate = (receipt / "raw/candidate_worktree.json").read_bytes()
    assert pre_refactor == candidate


def test_measurement_records_fixture_build_costs() -> None:
    measurement = runtime._measurement(
        fixture("rosenbrock"),
        "custom",
        "cpu",
        "parity",
        np.asarray([-1.2, 1.0], dtype=np.float64),
        maxiter=1,
        maxcor=10,
        method="lbfgs",
        fixture_build_seconds=1.25,
        fixture_build_peak_rss_kib=123,
    )

    assert measurement.fixture_build_seconds == 1.25
    assert measurement.fixture_build_peak_rss_kib == 123
    assert measurement.solver_start_rss_kib >= 0
    assert measurement.solver_peak_rss_kib >= measurement.solver_start_rss_kib
    assert measurement.solver_peak_rss_delta_kib >= 0
    assert measurement.initial_parameters == (-1.2, 1.0)
    assert isinstance(measurement.final_parameters, tuple)
    assert jnp.isfinite(measurement.final_objective)
    assert measurement.warm_transfer_audit
    assert {entry.phase for entry in measurement.warm_transfer_audit} >= {
        "advance",
        "final_result",
    }
    assert measurement.fixture_contract["final_certificate_fields"]
    assert measurement.fixture_contract["generator_sha256"]
    assert {entry.phase for entry in measurement.phase_rss} == {
        "preparation",
        "cold_solver",
        "warm_solver",
    }
    assert all(entry.sample_count >= 2 for entry in measurement.phase_rss)
    assert all(
        entry.peak_rss_kib >= entry.start_rss_kib
        and entry.peak_rss_kib >= entry.end_rss_kib
        for entry in measurement.phase_rss
    )


def test_rss_phase_records_named_scope() -> None:
    with runtime._RSSPhase("test") as phase:
        np.zeros(1024, dtype=np.float64)

    measurement = phase.measurement()
    assert measurement.phase == "test"
    assert measurement.scope == "self_proc_status_poll_10ms"
    assert measurement.sample_count >= 2


def test_bfgs_memory_analysis_has_a_separate_rss_phase() -> None:
    fixture_case = fixture("bfgs_quadratic")
    measurement = runtime._measurement(
        fixture_case,
        "custom",
        "cpu",
        "parity",
        np.asarray(fixture_case.initial, dtype=np.float64),
        maxiter=1,
        maxcor=10,
        method="bfgs",
        fixture_build_seconds=0.0,
        fixture_build_peak_rss_kib=0,
    )

    assert {entry.phase for entry in measurement.phase_rss} == {
        "algorithm_memory_analysis",
        "cold_solver",
        "warm_solver",
    }


def test_measurement_passes_prepared_custom_program_to_both_runs(monkeypatch) -> None:
    original_run_custom = runtime._run_custom
    prepared_runs: list[object | None] = []

    def recording_run_custom(*args, **kwargs):
        prepared_runs.append(kwargs.get("prepared"))
        return original_run_custom(*args, **kwargs)

    monkeypatch.setattr(runtime, "_run_custom", recording_run_custom)
    fixture_case = fixture("rosenbrock")
    runtime._measurement(
        fixture_case,
        "custom",
        "cpu",
        "parity",
        np.asarray(fixture_case.initial, dtype=np.float64),
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        fixture_build_seconds=0.0,
        fixture_build_peak_rss_kib=0,
    )

    assert len(prepared_runs) == 2
    assert all(prepared is not None for prepared in prepared_runs)


def test_measurement_native_provider_has_no_prepared_argument() -> None:
    fixture_case = fixture("rosenbrock")
    measurement = runtime._measurement(
        fixture_case,
        "native",
        "cpu",
        "parity",
        np.asarray(fixture_case.initial, dtype=np.float64),
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        fixture_build_seconds=0.0,
        fixture_build_peak_rss_kib=0,
    )

    assert measurement.provider == "native"


def test_native_measurement_resets_mutable_provider_between_runs() -> None:
    state = {
        "evaluations": 0,
        "resets": 0,
        "first_evaluation_after_reset": [],
        "awaiting_first_evaluation": False,
    }

    def native_value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        if state["awaiting_first_evaluation"]:
            state["first_evaluation_after_reset"].append(state["evaluations"])
            state["awaiting_first_evaluation"] = False
        state["evaluations"] += 1
        value = float(np.sum(np.square(x)) + state["evaluations"])
        return value, 2.0 * np.asarray(x, dtype=np.float64)

    def reset_native() -> None:
        state["evaluations"] = 0
        state["resets"] += 1
        state["awaiting_first_evaluation"] = True

    fixture_case = Fixture(
        name="stateful_native",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([1.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_stateful_native",
        certificate="synthetic native reset contract",
        method="lbfgs",
        native_value_and_grad=native_value_and_grad,
        native_reset=reset_native,
    )

    measurement = runtime._measurement(
        fixture_case,
        "native",
        "cpu",
        "parity",
        fixture_case.initial.copy(),
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        fixture_build_seconds=0.0,
        fixture_build_peak_rss_kib=0,
    )

    assert state["resets"] == 2
    assert state["first_evaluation_after_reset"] == [0, 0]
    assert np.isfinite(measurement.final_objective)


def test_optax_comparator_uses_a_jitted_step(monkeypatch) -> None:
    jit_calls = 0
    original_jit = runtime.jax.jit

    def counted_jit(fun, *args, **kwargs):
        nonlocal jit_calls
        jit_calls += 1
        return original_jit(fun, *args, **kwargs)

    monkeypatch.setattr(runtime.jax, "jit", counted_jit)
    result = runtime._run_optax(
        fixture("rosenbrock"),
        np.asarray([-1.2, 1.0], dtype=np.float64),
        maxiter=1,
        maxcor=3,
    )

    assert jit_calls == 2
    assert result[0] is not None


def test_optax_prepared_program_reuses_compiled_step(monkeypatch) -> None:
    jit_calls = 0
    original_jit = runtime.jax.jit

    def counted_jit(fun, *args, **kwargs):
        nonlocal jit_calls
        jit_calls += 1
        return original_jit(fun, *args, **kwargs)

    monkeypatch.setattr(runtime.jax, "jit", counted_jit)
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    prepared = runtime._prepare_optax(fixture_case, initial, maxcor=3)

    first = runtime._run_optax(
        fixture_case,
        initial,
        maxiter=1,
        maxcor=3,
        prepared=prepared,
    )
    second = runtime._run_optax(
        fixture_case,
        initial,
        maxiter=1,
        maxcor=3,
        prepared=prepared,
    )

    assert jit_calls == 2
    np.testing.assert_array_equal(np.asarray(first[0][0]), np.asarray(second[0][0]))
    assert first[0][3] == second[0][3]


def test_optax_prepared_program_rejects_mismatched_inputs() -> None:
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    prepared = runtime._prepare_optax(fixture_case, initial, maxcor=3)

    with pytest.raises(ValueError, match="does not match"):
        runtime._run_optax(
            fixture_case,
            initial + np.asarray([0.1, 0.0], dtype=np.float64),
            maxiter=1,
            maxcor=3,
            prepared=prepared,
        )


def test_custom_prepared_program_rejects_mismatched_inputs() -> None:
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    prepared = runtime._prepare_custom(fixture_case, initial, maxcor=3)

    with pytest.raises(ValueError, match="does not match"):
        runtime._run_custom(
            fixture_case,
            initial + np.asarray([0.1, 0.0], dtype=np.float64),
            maxiter=1,
            maxcor=3,
            method="lbfgs",
            prepared=prepared,
        )


def test_custom_prepared_program_reuses_compiled_transitions(monkeypatch) -> None:
    prepare_calls = 0
    original_prepare = runtime.prepare_lbfgs_private

    def counted_prepare(*args, **kwargs):
        nonlocal prepare_calls
        prepare_calls += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(runtime, "prepare_lbfgs_private", counted_prepare)
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    prepared = runtime._prepare_custom(fixture_case, initial, maxcor=3)
    first = runtime._run_custom(
        fixture_case,
        initial,
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        prepared=prepared,
    )
    second = runtime._run_custom(
        fixture_case,
        initial,
        maxiter=1,
        maxcor=3,
        method="lbfgs",
        prepared=prepared,
    )

    assert prepare_calls == 1
    np.testing.assert_array_equal(np.asarray(first[0].x_k), np.asarray(second[0].x_k))
    assert first[0].k == second[0].k


def test_optax_prepared_nan_input_reaches_nonfinite_status() -> None:
    fixture_case = Fixture(
        name="nan_input",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([np.nan], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_nonfinite",
        certificate="solver-runtime-only",
        method="lbfgs",
    )
    prepared = runtime._prepare_optax(
        fixture_case,
        fixture_case.initial,
        maxcor=3,
    )

    result = runtime._run_optax(
        fixture_case,
        fixture_case.initial.copy(),
        maxiter=1,
        maxcor=3,
        prepared=prepared,
    )

    assert result[2] == 6
    assert result[3] is False


def test_optax_prepared_signed_zero_input_is_bound_exactly() -> None:
    objective = lambda x: jnp.where(jnp.signbit(x[0]), 1.0, 0.0)
    fixture_case = Fixture(
        name="signed_zero_input",
        objective=objective,
        initial=np.asarray([0.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_signed_zero",
        certificate="solver-runtime-only",
        method="lbfgs",
    )
    prepared = runtime._prepare_optax(
        fixture_case,
        fixture_case.initial,
        maxcor=3,
    )

    with pytest.raises(ValueError, match="does not match"):
        runtime._run_optax(
            fixture_case,
            np.asarray([-0.0], dtype=np.float64),
            maxiter=1,
            maxcor=3,
            prepared=prepared,
        )


def test_optax_stop_check_uses_gradient_after_update() -> None:
    fixture_case = Fixture(
        name="one_dimensional_quadratic",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([1.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_quadratic",
        certificate="solver-runtime-only",
        method="lbfgs",
    )

    result = runtime._run_optax(
        fixture_case,
        fixture_case.initial,
        maxiter=2,
        maxcor=3,
    )

    assert result[0][3] == 1
    assert result[3] is True
    np.testing.assert_array_equal(np.asarray(result[0][0]), np.asarray([0.0]))


def test_optax_initial_terminal_state_takes_no_step() -> None:
    fixture_case = Fixture(
        name="already_converged_quadratic",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([0.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_quadratic",
        certificate="solver-runtime-only",
        method="lbfgs",
    )

    result = runtime._run_optax(
        fixture_case,
        fixture_case.initial,
        maxiter=5,
        maxcor=3,
    )

    assert result[0][3] == 0
    assert result[2] == 0
    assert result[3] is True


def test_optax_nonfinite_zero_gradient_is_not_success() -> None:
    fixture_case = Fixture(
        name="nonfinite_constant",
        objective=lambda _x: jnp.asarray(jnp.inf, dtype=jnp.float64),
        initial=np.asarray([0.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_nonfinite",
        certificate="solver-runtime-only",
        method="lbfgs",
    )

    result = runtime._run_optax(
        fixture_case,
        fixture_case.initial,
        maxiter=5,
        maxcor=3,
    )

    assert result[0][3] == 0
    assert result[2] == 6
    assert result[3] is False


def test_optax_line_search_failure_is_labeled() -> None:
    fixture_case = Fixture(
        name="linear_objective",
        objective=lambda x: x[0],
        initial=np.asarray([1.0], dtype=np.float64),
        expected_dimension=1,
        source="synthetic_contract_linear",
        certificate="solver-runtime-only",
        method="lbfgs",
    )

    result = runtime._run_optax(
        fixture_case,
        fixture_case.initial,
        maxiter=3,
        maxcor=3,
    )

    assert result[0][3] == 1
    assert result[2] == 2
    assert result[3] is False


def test_fixture_contract_records_provenance_observables_and_tolerances() -> None:
    fixture_case = fixture("rosenbrock")
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    initial_objective, initial_gradient = runtime._initial_value_and_grad(
        fixture_case,
        initial,
    )

    contract = runtime._fixture_contract_payload(
        fixture_case,
        initial,
        initial_objective=initial_objective,
        initial_gradient_inf_norm=float(np.max(np.abs(initial_gradient))),
        method="lbfgs",
        maxiter=20,
        maxcor=10,
        device="cpu",
        intent="parity",
    )

    assert contract["generator_sha256"]
    assert contract["source_sha256"]
    assert contract["initial_parameters"] == [-1.2, 1.0]
    assert contract["expected_initial_observables"]["objective"] == initial_objective
    assert contract["solver_options"] == {
        "device": "cpu",
        "ftol": 0.0,
        "gtol": 1.0e-10,
        "intent": "parity",
        "maxcor": 10,
        "maxfun": None,
        "maxiter": 20,
        "maxls": 20,
        "method": "lbfgs",
    }
    assert contract["tolerances"]


def test_dense_bfgs_memory_contract_reports_no_donation_upper_bound() -> None:
    contract = runtime._bfgs_memory_contract(47, np.float64)

    assert contract["inverse_hessian_bytes"] == 47 * 47 * 8
    assert contract["simultaneous_old_new_hessian_bytes"] == 2 * 47 * 47 * 8
    assert (
        contract["derived_peak_live_upper_bound_bytes"]
        > contract["simultaneous_old_new_hessian_bytes"]
    )
    assert contract["buffer_donation"] is False


def test_dense_bfgs_memory_analysis_is_update_only_and_bounded() -> None:
    report = runtime._dense_bfgs_update_memory_analysis(5, np.dtype(np.float64))
    contract = runtime._bfgs_memory_contract(5, np.float64)

    assert report["dense_update_compiled_memory_is_update_only"] is True
    assert report["dense_update_peak_live_bytes"] >= report["dense_update_output_bytes"]
    assert report["dense_update_temp_bytes"] >= 0
    if runtime.jax.default_backend() == "cpu":
        assert (
            report["dense_update_peak_live_bytes"]
            <= contract["derived_peak_live_upper_bound_bytes"]
        )
    else:
        # Device compiler temporaries are physical backend accounting, not the
        # logical no-donation bound derived from the algorithm's live arrays.
        assert report["dense_update_peak_live_bytes"] > 0


@pytest.mark.slow
def test_coil47_fixture_exposes_native_objective_callback() -> None:
    fixture_case = fixture("coil47")

    assert fixture_case.native_value_and_grad is not None
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    native_value, native_gradient = fixture_case.native_value_and_grad(initial)
    jax_value, jax_gradient = runtime._initial_value_and_grad(fixture_case, initial)

    assert native_value == pytest.approx(jax_value, abs=1.0e-12, rel=1.0e-12)
    np.testing.assert_allclose(native_gradient, jax_gradient, atol=1.0e-8, rtol=1.0e-10)


@pytest.mark.slow
def test_boozer_fixture_exposes_matched_native_objective_callback() -> None:
    fixture_case = fixture("boozer")

    assert fixture_case.native_value_and_grad is not None
    initial = np.asarray(fixture_case.initial, dtype=np.float64)
    native_value, native_gradient = fixture_case.native_value_and_grad(initial)
    jax_value, jax_gradient = runtime._initial_value_and_grad(fixture_case, initial)

    assert native_value == pytest.approx(jax_value, abs=1.0e-15, rel=1.0e-12)
    np.testing.assert_allclose(native_gradient, jax_gradient, atol=2.0e-12, rtol=2.0e-9)


def test_native_provider_rejects_unmatched_source_fixture() -> None:
    fixture_case = Fixture(
        name="unmatched",
        objective=lambda x: jnp.sum(x * x),
        initial=np.asarray([1.0], dtype=np.float64),
        expected_dimension=1,
        source="source_owned_unmatched",
        certificate="unmatched",
        method="bfgs",
    )

    with pytest.raises(ValueError, match="unmatched source-owned fixture"):
        runtime._initial_value_and_grad(
            fixture_case,
            fixture_case.initial,
            provider="native",
        )
