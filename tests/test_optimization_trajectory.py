"""Tests for the native optimization trajectory recorder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Protocol

import examples.jax.parity.cases.native_boozerqa as single_stage_case
import numpy as np
import pytest
import simsopt.optimization_trajectory as trajectory_module
from examples.jax.parity.measurement import MeasurementExecution
from simsopt.optimization_trajectory import (
    OptimizationMeasurementWindow,
    OptimizationTrajectoryRecorder,
    read_optimization_window_timing,
)


class _OptimizeIntermediateResult(Protocol):
    fun: float


_LaneTimingStrategy = Literal["native_cpu", "jax_gpu_custom", "jax_gpu_optax"]


@pytest.mark.parametrize(
    "lane_strategy",
    ("native_cpu", "jax_gpu_custom", "jax_gpu_optax"),
)
def test_single_stage_lane_strategies_share_the_recorded_timing_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lane_strategy: _LaneTimingStrategy,
) -> None:
    clock = [100.0]
    initial_evaluation_seconds = 0.375
    endpoint_reporting_seconds = 8.0
    monkeypatch.setattr(trajectory_module, "perf_counter", lambda: clock[0])
    phase_events: list[str] = []
    x0 = np.asarray([4.0, -5.0], dtype=np.float64)
    target = np.asarray([1.0, -2.0], dtype=np.float64)

    def numpy_value_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        residual = x - target
        return float(np.dot(residual, residual)), 2.0 * residual

    def evaluate_initial() -> tuple[float, np.ndarray]:
        phase_events.append("initial_value_and_grad")
        value, gradient = numpy_value_and_grad(x0)
        clock[0] += initial_evaluation_seconds
        return value, gradient

    trajectory_path = tmp_path / f"{lane_strategy}.jsonl"
    timing_path = tmp_path / f"{lane_strategy}-timing.json"
    measurement = MeasurementExecution(
        trajectory_path=trajectory_path,
        optimization_timing_path=timing_path,
    )
    with single_stage_case._measurement_optimization_window(
        measurement,
        evaluate_initial,
    ) as (initial_value_and_grad, recorder):
        phase_events.append("optimizer_started")
        assert recorder is not None
        if lane_strategy == "native_cpu":
            scipy = pytest.importorskip("scipy", minversion="1.11")
            accepted_iterations = 0

            def record_native(intermediate_result: _OptimizeIntermediateResult) -> None:
                nonlocal accepted_iterations
                accepted_iterations += 1
                recorder.record(accepted_iterations, float(intermediate_result.fun))

            optimizer_result = scipy.optimize.minimize(
                numpy_value_and_grad,
                x0,
                jac=True,
                method="BFGS",
                callback=record_native,
                options={"gtol": 1.0e-12, "maxiter": 4},
            )
            endpoint_parameters = np.asarray(optimizer_result.x, dtype=np.float64)
        elif lane_strategy == "jax_gpu_custom":
            from simsopt_jax.geo.optimizer_host_lbfgs import (
                minimize_lbfgs_host_core,
            )

            optimizer_result = minimize_lbfgs_host_core(
                numpy_value_and_grad,
                x0,
                maxiter=4,
                maxcor=4,
                gtol=1.0e-12,
                maxls=10,
                initial_value_and_grad=initial_value_and_grad,
                progress_callback=lambda iteration, objective, gradient_norm: (
                    recorder.record(iteration, objective)
                ),
            )
            endpoint_parameters = np.asarray(
                optimizer_result.x_k,
                dtype=np.float64,
            )
        else:
            jnp = pytest.importorskip("jax.numpy")
            pytest.importorskip("optax")
            from simsopt_jax.geo.optimizers.optimizer import target_minimize

            target_device = jnp.asarray(target)

            def jax_value_and_grad(x):
                residual = x - target_device
                return jnp.vdot(residual, residual), 2.0 * residual

            optimizer_result = target_minimize(
                jax_value_and_grad,
                jnp.asarray(x0),
                method="optax-lbfgs-ondevice",
                tol=1.0e-12,
                maxiter=4,
                options={"maxcor": 4, "maxls": 10},
                value_and_grad=True,
                progress_callback=lambda iteration, objective, gradient_norm: (
                    recorder.record(iteration, objective)
                ),
            )
            endpoint_parameters = np.asarray(optimizer_result.x, dtype=np.float64)
        phase_events.append("optimizer_finished")

    clock[0] += endpoint_reporting_seconds
    endpoint_objective, _endpoint_gradient = numpy_value_and_grad(endpoint_parameters)
    phase_events.append("endpoint_reporting")

    records = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records
    assert read_optimization_window_timing(timing_path).wall_seconds == (
        initial_evaluation_seconds
    )
    assert records[0]["wall_seconds_from_start"] == initial_evaluation_seconds
    assert {record["wall_seconds_from_start"] for record in records} == {
        initial_evaluation_seconds
    }
    assert phase_events == [
        "initial_value_and_grad",
        "optimizer_started",
        "optimizer_finished",
        "endpoint_reporting",
    ]
    assert np.isfinite(endpoint_objective)


def test_measurement_execution_allows_backend_selection_without_recording() -> None:
    request = MeasurementExecution(optimizer_backend="optax-lbfgs")

    assert request.trajectory_path is None
    assert request.optimizer_backend == "optax-lbfgs"
    with pytest.raises(ValueError, match="instrumentation path or optimizer backend"):
        MeasurementExecution()


def test_optimizer_window_writes_exact_exclusive_timing_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((20.0, 20.75, 30.0, 30.5))
    monkeypatch.setattr(trajectory_module, "perf_counter", lambda: next(clock_values))
    timing_path = tmp_path / "optimization-timing.json"

    with OptimizationMeasurementWindow(
        trajectory_path=None,
        timing_path=timing_path,
    ):
        pass

    assert timing_path.read_text(encoding="utf-8") == (
        '{"schema_version":1,"wall_seconds":0.75}\n'
    )
    assert read_optimization_window_timing(timing_path).wall_seconds == 0.75

    with pytest.raises(FileExistsError), OptimizationMeasurementWindow(
        trajectory_path=None,
        timing_path=timing_path,
    ):
        pass

    assert timing_path.read_text(encoding="utf-8") == (
        '{"schema_version":1,"wall_seconds":0.75}\n'
    )


@pytest.mark.parametrize(
    "document",
    (
        '{"schema_version":1,"wall_seconds":false}\n',
        '{"schema_version":1,"wall_seconds":-1.0}\n',
        '{"schema_version":2,"wall_seconds":1.0}\n',
        '{"extra":0,"schema_version":1,"wall_seconds":1.0}\n',
    ),
)
def test_timing_sidecar_reader_rejects_noncanonical_documents(
    tmp_path: Path,
    document: str,
) -> None:
    timing_path = tmp_path / "optimization-timing.json"
    timing_path.write_text(document, encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match="optimization"):
        read_optimization_window_timing(timing_path)


def test_recorder_writes_exact_canonical_records(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"

    with OptimizationTrajectoryRecorder.create(path) as recorder:
        recorder.record(1, 4, wall_seconds_from_start=0)
        recorder.record(2, 2.5, wall_seconds_from_start=1.25)

    assert path.read_text(encoding="utf-8") == (
        '{"iteration":1,"objective":4.0,"wall_seconds_from_start":0.0}\n'
        '{"iteration":2,"objective":2.5,"wall_seconds_from_start":1.25}\n'
    )
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(
        set(record) == {"iteration", "objective", "wall_seconds_from_start"}
        for record in records
    )


def test_default_elapsed_time_starts_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock_values = iter((100.0, 100.375))
    monkeypatch.setattr(trajectory_module, "perf_counter", lambda: next(clock_values))

    path = tmp_path / "trajectory.jsonl"
    with OptimizationTrajectoryRecorder(path) as recorder:
        recorder.record(1, 3.0)

    assert json.loads(path.read_text()) == {
        "iteration": 1,
        "objective": 3.0,
        "wall_seconds_from_start": 0.375,
    }


def test_supplied_elapsed_time_does_not_read_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clock_values = iter((50.0,))
    monkeypatch.setattr(trajectory_module, "perf_counter", lambda: next(clock_values))

    path = tmp_path / "trajectory.jsonl"
    with OptimizationTrajectoryRecorder(path) as recorder:
        recorder.record(1, 3.0, wall_seconds_from_start=0.0)

    assert json.loads(path.read_text())["wall_seconds_from_start"] == 0.0


def test_existing_path_is_not_reopened_or_appended(tmp_path: Path) -> None:
    path = tmp_path / "trajectory.jsonl"
    path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        OptimizationTrajectoryRecorder(path)

    assert path.read_text(encoding="utf-8") == "existing\n"


@pytest.mark.parametrize(
    ("iteration", "objective", "elapsed", "message"),
    (
        (0, 1.0, 0.0, "exact successor"),
        (1, float("nan"), 0.0, "objective"),
        (1, 1.0, -0.1, "wall_seconds_from_start"),
        (1, 1.0, float("inf"), "wall_seconds_from_start"),
    ),
)
def test_record_rejects_invalid_first_record(
    tmp_path: Path,
    iteration: int,
    objective: float,
    elapsed: float,
    message: str,
) -> None:
    path = tmp_path / "trajectory.jsonl"
    with OptimizationTrajectoryRecorder(path) as recorder:  # noqa: SIM117
        with pytest.raises(ValueError, match=message):
            recorder.record(
                iteration,
                objective,
                wall_seconds_from_start=elapsed,
            )

    assert path.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("iteration", "objective", "elapsed", "message"),
    (
        (1, 2.0, 0.5, "exact successor"),
        (3, 2.0, 0.5, "exact successor"),
        (2, 2.0, 0.4, "nondecreasing"),
        (2, float("-inf"), 0.6, "objective"),
    ),
)
def test_record_rejects_invalid_followup_without_advancing_state(
    tmp_path: Path,
    iteration: int,
    objective: float,
    elapsed: float,
    message: str,
) -> None:
    path = tmp_path / "trajectory.jsonl"
    with OptimizationTrajectoryRecorder(path) as recorder:
        recorder.record(1, 2.0, wall_seconds_from_start=0.5)
        with pytest.raises(ValueError, match=message):
            recorder.record(
                iteration,
                objective,
                wall_seconds_from_start=elapsed,
            )
        recorder.record(2, 1.0, wall_seconds_from_start=0.5)

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [record["iteration"] for record in records] == [1, 2]


def _rosenbrock_value_and_gradient(x: np.ndarray) -> tuple[float, np.ndarray]:
    value = 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2
    gradient = np.asarray(
        [
            -400.0 * x[0] * (x[1] - x[0] ** 2) - 2.0 * (1.0 - x[0]),
            200.0 * (x[1] - x[0] ** 2),
        ],
        dtype=np.float64,
    )
    return float(value), gradient


def test_scipy_bfgs_recording_does_not_change_solver_result(
    tmp_path: Path,
) -> None:
    scipy = pytest.importorskip("scipy", minversion="1.11")
    optimize = scipy.optimize
    x0 = np.asarray([-1.2, 1.0], dtype=np.float64)
    plain_calls = 0
    recorded_calls = 0

    def plain_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal plain_calls
        plain_calls += 1
        return _rosenbrock_value_and_gradient(x)

    def recorded_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal recorded_calls
        recorded_calls += 1
        return _rosenbrock_value_and_gradient(x)

    plain = optimize.minimize(
        plain_objective,
        x0,
        jac=True,
        method="BFGS",
        options={"gtol": 1.0e-8, "maxiter": 100},
    )

    path = tmp_path / "trajectory.jsonl"
    with OptimizationTrajectoryRecorder(path) as recorder:
        accepted_iterations = 0

        def callback(intermediate_result: _OptimizeIntermediateResult) -> None:
            nonlocal accepted_iterations
            accepted_iterations += 1
            recorder.record(
                accepted_iterations,
                float(intermediate_result.fun),
            )

        recorded = optimize.minimize(
            recorded_objective,
            x0,
            jac=True,
            method="BFGS",
            callback=callback,
            options={"gtol": 1.0e-8, "maxiter": 100},
        )

    np.testing.assert_array_equal(recorded.x, plain.x)
    assert recorded.fun == plain.fun
    assert recorded.nfev == plain.nfev
    assert recorded.njev == plain.njev
    assert recorded_calls == plain_calls

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records
    assert len(records) == recorded.nit
    assert [record["iteration"] for record in records] == list(
        range(1, len(records) + 1)
    )
    assert records[-1]["objective"] == recorded.fun


def test_jax_host_bfgs_recording_does_not_change_solver_result(
    tmp_path: Path,
) -> None:
    pytest.importorskip("jax")
    from simsopt_jax.geo.optimizer_host_lbfgs import (
        line_search_value_and_grad_more_thuente_host,
        minimize_bfgs_host_core,
    )

    x0 = np.asarray([4.0, -5.0], dtype=np.float64)
    target = np.asarray([1.0, -2.0], dtype=np.float64)
    plain_calls = 0
    recorded_calls = 0

    def plain_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal plain_calls
        plain_calls += 1
        residual = x - target
        return float(np.dot(residual, residual)), 2.0 * residual

    def recorded_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal recorded_calls
        recorded_calls += 1
        residual = x - target
        return float(np.dot(residual, residual)), 2.0 * residual

    plain = minimize_bfgs_host_core(
        plain_objective,
        x0,
        maxiter=4,
        gtol=1.0e-12,
        maxls=10,
        line_search_value_and_grad=line_search_value_and_grad_more_thuente_host,
    )

    path = tmp_path / "trajectory.jsonl"
    with OptimizationTrajectoryRecorder(path) as recorder:
        recorded = minimize_bfgs_host_core(
            recorded_objective,
            x0,
            maxiter=4,
            gtol=1.0e-12,
            maxls=10,
            line_search_value_and_grad=line_search_value_and_grad_more_thuente_host,
            progress_callback=lambda iteration, objective, gradient_norm: (
                recorder.record(
                    iteration,
                    objective,
                )
            ),
        )

    np.testing.assert_array_equal(recorded.x_k, plain.x_k)
    assert recorded.f_k == plain.f_k
    assert recorded.nfev == plain.nfev
    assert recorded.ngev == plain.ngev
    assert recorded_calls == plain_calls

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == recorded.k
    assert [record["iteration"] for record in records] == list(range(1, recorded.k + 1))
    assert records[-1]["objective"] == recorded.f_k


def test_jax_host_lbfgs_recording_does_not_change_solver_result(
    tmp_path: Path,
) -> None:
    pytest.importorskip("jax")
    from simsopt_jax.geo.optimizer_host_lbfgs import minimize_lbfgs_host_core

    x0 = np.asarray([4.0, -5.0], dtype=np.float64)
    target = np.asarray([1.0, -2.0], dtype=np.float64)
    plain_calls = 0
    recorded_calls = 0

    def plain_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal plain_calls
        plain_calls += 1
        residual = x - target
        return float(np.dot(residual, residual)), 2.0 * residual

    def recorded_objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal recorded_calls
        recorded_calls += 1
        residual = x - target
        return float(np.dot(residual, residual)), 2.0 * residual

    plain = minimize_lbfgs_host_core(
        plain_objective,
        x0,
        maxiter=4,
        maxcor=4,
        gtol=1.0e-12,
        maxls=10,
    )

    path = tmp_path / "trajectory.jsonl"
    with OptimizationTrajectoryRecorder(path) as recorder:
        recorded = minimize_lbfgs_host_core(
            recorded_objective,
            x0,
            maxiter=4,
            maxcor=4,
            gtol=1.0e-12,
            maxls=10,
            progress_callback=lambda iteration, objective, gradient_norm: (
                recorder.record(iteration, objective)
            ),
        )

    np.testing.assert_array_equal(recorded.x_k, plain.x_k)
    assert recorded.f_k == plain.f_k
    assert recorded.nfev == plain.nfev
    assert recorded.ngev == plain.ngev
    assert recorded_calls == plain_calls

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == recorded.k
    assert [record["iteration"] for record in records] == list(range(1, recorded.k + 1))
    assert records[-1]["objective"] == recorded.f_k


def test_optax_lbfgs_recording_does_not_change_solver_result(
    tmp_path: Path,
) -> None:
    pytest.importorskip("jax")
    pytest.importorskip("optax")
    jnp = pytest.importorskip("jax.numpy")
    from simsopt_jax.geo.optimizers.optimizer import target_minimize

    x0 = jnp.asarray([4.0, -5.0], dtype=jnp.float64)
    target = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    def value_and_grad(x):
        residual = x - target
        return jnp.vdot(residual, residual), 2.0 * residual

    options = {"maxcor": 4, "maxls": 10}
    plain = target_minimize(
        value_and_grad,
        x0,
        method="optax-lbfgs-ondevice",
        tol=1.0e-12,
        maxiter=4,
        options=options,
        value_and_grad=True,
    )

    path = tmp_path / "trajectory.jsonl"
    with OptimizationTrajectoryRecorder(path) as recorder:
        recorded = target_minimize(
            value_and_grad,
            x0,
            method="optax-lbfgs-ondevice",
            tol=1.0e-12,
            maxiter=4,
            options=options,
            value_and_grad=True,
            progress_callback=lambda iteration, objective, gradient_norm: (
                recorder.record(
                    iteration,
                    objective,
                )
            ),
        )

    np.testing.assert_array_equal(recorded.x, plain.x)
    assert recorded.fun == plain.fun
    assert recorded.nfev == plain.nfev
    assert recorded.njev == plain.njev

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == recorded.nit
    assert [record["iteration"] for record in records] == list(
        range(1, recorded.nit + 1)
    )
    assert records[-1]["objective"] == recorded.fun
