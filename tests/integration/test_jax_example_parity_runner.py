from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from benchmarks.validation_ladder_contract import parity_ladder_tolerances
from examples.jax.parity._manifest import ComparisonRoute
from examples.jax.parity.arbiter import (
    ArbitrationError,
    LaneObservation,
    arbitrate,
)
from examples.jax.parity.artifacts import canonical_json_bytes, write_array
from examples.jax.parity.audit import audit_published_run
from examples.jax.parity.cases import get_case
from examples.jax.parity.provenance import (
    DeviceMetadata,
    ExecutedSource,
    LaneProvenance,
    collect_repository_state,
    generated_version_matches_checkout,
)
from examples.jax.parity.receipts import write_lane_observation
from examples.jax.parity.runner import (
    ChildProcessResult,
    RunnerError,
    build_child_command,
    execute_case_lanes,
    execute_child_process,
)


def _routes() -> tuple[ComparisonRoute, ...]:
    return tuple(
        ComparisonRoute(
            phase="initial",
            observable="objective_sum_squares",
            lane_pair=lane_pair,
            applicable=True,
            comparator="allclose",
            tolerance_bucket=(
                "gpu_runtime" if lane_pair == "jax-cpu:jax-gpu" else "native_workflow"
            ),
        )
        for lane_pair in (
            "native-cpu:jax-cpu",
            "native-cpu:jax-gpu",
            "jax-cpu:jax-gpu",
        )
    )


def _provenance(backend_mode: str) -> LaneProvenance:
    is_jax = backend_mode != "native_cpu"
    platform = "gpu" if backend_mode == "jax_gpu_parity" else "cpu"
    policy = {"SIMSOPT_BACKEND_MODE": backend_mode}
    if backend_mode == "jax_gpu_parity":
        policy.update(
            {
                "SIMSOPT_JAX_TRANSFER_GUARD": "disallow",
                "JAX_TRANSFER_GUARD": "disallow",
            }
        )
    return LaneProvenance(
        repository_commit="d" * 40,
        repository_dirty=False,
        tracked_diff_sha256="e" * 64,
        untracked_files=(),
        executed_sources=(
            ExecutedSource("examples/jax/run_parity.py", "f" * 64, "a" * 40),
        ),
        python_version="3.11.0",
        jax_version="0.10.0" if is_jax else None,
        simsopt_version="1.10.0",
        simsopt_version_commit="g" + "d" * 9,
        simsopt_version_checkout_compatible=True,
        lane_environment_policy=policy,
        jax_effective_transfer_guards=(
            {
                "device_to_device": "disallow",
                "device_to_host": "disallow",
                "host_to_device": "disallow",
            }
            if backend_mode == "jax_gpu_parity"
            else {}
        ),
        devices=(DeviceMetadata(0, platform, "test", 0),) if is_jax else (),
        host_peak_rss_bytes=1024,
        host_peak_rss_method="test fixture",
        device_memory_peak_bytes=None,
        device_memory_status="unavailable: test fixture",
        memory_measurement_scope=(
            "combined import, compile/warmup, and one bounded execution"
        ),
        steady_state_memory_measured=False,
        measurement_synchronization=(
            "jax.block_until_ready over published observation values"
            if is_jax
            else "native synchronous execution"
        ),
        simsoptpp_path=None,
        simsoptpp_sha256=None,
        simsoptpp_version=None,
        simsoptpp_build_commit=None,
        simsoptpp_checkout_compatible=None,
        authoritative=True,
    )


def _observations() -> dict[str, LaneObservation]:
    return {
        "native-cpu": LaneObservation(
            lane="native-cpu",
            backend_mode="native_cpu",
            platform="cpu",
            precision="fp64",
            input_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
            effective_construction_fingerprint="c" * 64,
            driver="scipy_least_squares",
            normalized_status="converged",
            raw_status="1",
            success=True,
            nit=None,
            nfev=3,
            njev=3,
            completed_workflow_stages=("construct", "evaluate"),
            provenance=_provenance("native_cpu"),
            values={"initial:objective_sum_squares": np.array(1.0)},
        ),
        "jax-cpu": LaneObservation(
            lane="jax-cpu",
            backend_mode="jax_cpu_parity",
            platform="cpu",
            precision="fp64",
            input_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
            effective_construction_fingerprint="c" * 64,
            driver="simsopt_lm_gmres",
            normalized_status="converged",
            raw_status="0",
            success=True,
            nit=2,
            nfev=3,
            njev=3,
            completed_workflow_stages=("construct", "evaluate"),
            provenance=_provenance("jax_cpu_parity"),
            values={"initial:objective_sum_squares": np.array(1.0)},
        ),
        "jax-gpu": LaneObservation(
            lane="jax-gpu",
            backend_mode="jax_gpu_parity",
            platform="gpu",
            precision="fp64",
            input_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
            effective_construction_fingerprint="c" * 64,
            driver="simsopt_lm_gmres",
            normalized_status="converged",
            raw_status="0",
            success=True,
            nit=2,
            nfev=3,
            njev=3,
            completed_workflow_stages=("construct", "evaluate"),
            provenance=_provenance("jax_gpu_parity"),
            values={"initial:objective_sum_squares": np.array(1.0)},
        ),
    }


def test_native_workflow_tolerance_is_centrally_owned_and_adversarial() -> None:
    tolerance = parity_ladder_tolerances("native_workflow")

    assert tolerance["requires_native_workflow_oracle"] is True
    assert tolerance["requires_direct_cpp_oracle"] is False
    assert float(tolerance["same_state_value_rtol"]) < 1.0e-8
    assert float(tolerance["terminal_relative_reduction"]) == 1.0e-12
    assert float(tolerance["terminal_constraint_norm_atol"]) == 1.0e-10
    assert float(tolerance["terminal_orthonormality_atol"]) == 1.0e-12


def test_qfm_terminal_success_rejects_retained_infeasible_state() -> None:
    from examples.jax.parity.cases.qfm_surface import _terminal_success

    initial_state = {
        "initial:penalty_objective": np.asarray([4.41288576329861e-2]),
    }
    final_state = {
        "final:penalty_objective": np.asarray([7.268155221443783e-4]),
        "final:penalty_gradient": np.zeros(9, dtype=np.float64),
        "final:qfm_objective": np.asarray([7.268032062695834e-4]),
        "final:constraint_value": np.asarray([1.2315874794985946e-8]),
    }

    assert not _terminal_success(initial_state, final_state)


def test_qfm_normalized_status_preserves_driver_failure() -> None:
    from examples.jax.parity.cases.qfm_surface import _normalized_driver_status

    assert _normalized_driver_status(driver_success=True) == "converged"
    assert _normalized_driver_status(driver_success=False) == "failed"


def test_generated_version_source_must_name_the_clean_checkout() -> None:
    repository_commit = "123456789abcdef" + "0" * 25

    assert generated_version_matches_checkout(repository_commit, "g123456789")
    assert not generated_version_matches_checkout(repository_commit, "gabcdef123")
    assert not generated_version_matches_checkout(repository_commit, None)


def test_arbiter_requires_direct_all_pairs_and_passes_matching_receipts() -> None:
    result = arbitrate(_routes(), _observations())

    assert result.verdict == "pass"
    assert len(result.comparisons) == 3
    assert all(comparison.passed for comparison in result.comparisons)


def test_arbiter_rejects_applicable_observable_without_routes() -> None:
    observations = {
        lane: dataclasses.replace(
            observation,
            values={
                **observation.values,
                "final:residual_jacobian": np.eye(1, dtype=np.float64),
            },
            applicability={},
        )
        for lane, observation in _observations().items()
    }

    with pytest.raises(ArbitrationError, match="complete direct lane-pair matrix"):
        arbitrate(_routes(), observations)


def test_lane_receipts_expose_explicit_observable_applicability() -> None:
    observation = _observations()["native-cpu"]

    assert observation.applicability == {
        "initial:objective_sum_squares": True,
        "optimizer_outcome": True,
    }


def test_memory_receipt_distinguishes_compile_and_steady_state_scope() -> None:
    provenance = _provenance("jax_gpu_parity")

    assert provenance.memory_measurement_scope == (
        "combined import, compile/warmup, and one bounded execution"
    )
    assert provenance.steady_state_memory_measured is False
    assert provenance.measurement_synchronization.startswith("jax.block_until_ready")


def test_gpu_receipt_records_effective_jax_transfer_guards() -> None:
    provenance = _provenance("jax_gpu_parity")

    assert provenance.jax_effective_transfer_guards == {
        "device_to_device": "disallow",
        "device_to_host": "disallow",
        "host_to_device": "disallow",
    }


def test_arbiter_accepts_unrelated_dirty_worktree_drift() -> None:
    observations = _observations()
    jax_cpu_provenance = observations["jax-cpu"].provenance
    assert jax_cpu_provenance is not None
    observations["jax-cpu"] = dataclasses.replace(
        observations["jax-cpu"],
        provenance=dataclasses.replace(
            jax_cpu_provenance,
            repository_dirty=True,
            tracked_diff_sha256="1" * 64,
            untracked_files=("unrelated.txt",),
            authoritative=False,
        ),
    )

    result = arbitrate(_routes(), observations)

    assert result.verdict == "pass"


def test_arbiter_rejects_shared_executed_source_hash_mismatch() -> None:
    observations = _observations()
    jax_cpu_provenance = observations["jax-cpu"].provenance
    assert jax_cpu_provenance is not None
    observations["jax-cpu"] = dataclasses.replace(
        observations["jax-cpu"],
        provenance=dataclasses.replace(
            jax_cpu_provenance,
            executed_sources=(
                ExecutedSource("examples/jax/run_parity.py", "1" * 64, "a" * 40),
            ),
        ),
    )

    with pytest.raises(ArbitrationError, match="executed source mismatch"):
        arbitrate(_routes(), observations)


def test_arbiter_rejects_workflow_stage_mismatch() -> None:
    observations = _observations()

    with pytest.raises(ArbitrationError, match="workflow stage mismatch"):
        arbitrate(
            _routes(),
            observations,
            expected_workflow_stages=("construct", "evaluate", "solve"),
        )


@pytest.mark.parametrize(
    ("mutation", "expected_message"),
    [
        ("missing_lane", "missing required lane"),
        ("wrong_gpu", "jax-gpu platform must be gpu"),
        ("input_mismatch", "input fingerprint mismatch"),
        ("effective_mismatch", "effective construction fingerprint mismatch"),
        ("nonfinite", "non-finite"),
        ("wrong_float_dtype", "must be FP64"),
        ("missing_direct_pair", "complete direct lane-pair matrix"),
        ("hidden_host_solver", "forbidden parity driver"),
        ("scientific_failure", "scientific success"),
    ],
)
def test_arbiter_fails_closed_on_invalid_receipts(
    mutation: str, expected_message: str
) -> None:
    observations = _observations()
    routes = _routes()
    if mutation == "missing_lane":
        observations.pop("jax-gpu")
    elif mutation == "wrong_gpu":
        observations["jax-gpu"] = dataclasses.replace(
            observations["jax-gpu"], platform="cpu"
        )
    elif mutation == "input_mismatch":
        observations["jax-cpu"] = dataclasses.replace(
            observations["jax-cpu"], input_fingerprint="d" * 64
        )
    elif mutation == "effective_mismatch":
        observations["jax-cpu"] = dataclasses.replace(
            observations["jax-cpu"], effective_construction_fingerprint="d" * 64
        )
    elif mutation == "nonfinite":
        observations["jax-gpu"] = dataclasses.replace(
            observations["jax-gpu"],
            values={"initial:objective_sum_squares": np.array(np.nan)},
        )
    elif mutation == "wrong_float_dtype":
        observations["jax-gpu"] = dataclasses.replace(
            observations["jax-gpu"],
            values={"initial:objective_sum_squares": np.array(1.0, dtype=np.float32)},
        )
    elif mutation == "missing_direct_pair":
        routes = tuple(
            route for route in routes if route.lane_pair != "native-cpu:jax-gpu"
        )
    elif mutation == "hidden_host_solver":
        observations["jax-gpu"] = dataclasses.replace(
            observations["jax-gpu"], driver="scipy_host_callback"
        )
    elif mutation == "scientific_failure":
        observations["jax-gpu"] = dataclasses.replace(
            observations["jax-gpu"], success=False, normalized_status="failed"
        )
    else:
        raise AssertionError(f"unhandled mutation: {mutation}")

    with pytest.raises(ArbitrationError, match=expected_message):
        arbitrate(routes, observations)


def test_direct_native_gpu_gate_catches_transitive_tolerance_drift() -> None:
    tolerance = parity_ladder_tolerances("native_workflow")
    absolute_tolerance = float(tolerance["same_state_value_atol"])
    observations = _observations()
    observations["jax-cpu"] = dataclasses.replace(
        observations["jax-cpu"],
        values={"initial:objective_sum_squares": np.array(0.75 * absolute_tolerance)},
    )
    observations["jax-gpu"] = dataclasses.replace(
        observations["jax-gpu"],
        values={"initial:objective_sum_squares": np.array(1.5 * absolute_tolerance)},
    )
    observations["native-cpu"] = dataclasses.replace(
        observations["native-cpu"],
        values={"initial:objective_sum_squares": np.array(0.0)},
    )

    result = arbitrate(_routes(), observations)

    comparisons = {
        comparison.lane_pair: comparison for comparison in result.comparisons
    }
    assert comparisons["native-cpu:jax-cpu"].passed
    assert comparisons["jax-cpu:jax-gpu"].passed
    assert not comparisons["native-cpu:jax-gpu"].passed
    assert result.verdict == "fail"


def test_child_command_is_exact_and_bounded(tmp_path: Path) -> None:
    command = build_child_command(
        python_executable="/venv/bin/python",
        case_id="quadratic",
        lane="jax-gpu",
        input_bundle_path=tmp_path / "input_bundle.json",
        result_directory=tmp_path / "result",
        smoke=True,
    )
    assert command == (
        "/venv/bin/python",
        "-m",
        "examples.jax.parity.child",
        "--case",
        "quadratic",
        "--lane",
        "jax-gpu",
        "--input-bundle",
        str(tmp_path / "input_bundle.json"),
        "--result-directory",
        str(tmp_path / "result"),
        "--smoke",
    )


def test_parent_observes_child_peak_rss(tmp_path: Path) -> None:
    completed: ChildProcessResult = execute_child_process(
        (
            sys.executable,
            "-c",
            "payload = bytearray(32_000_000); print(len(payload))",
        ),
        tmp_path,
        {},
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "32000000"
    assert completed.parent_peak_rss_bytes is not None
    assert completed.parent_peak_rss_bytes >= 32_000_000


def test_runner_executes_isolated_lanes_and_loads_hash_bound_receipts(
    tmp_path: Path,
) -> None:
    observations = _observations()
    seen: list[tuple[tuple[str, ...], str]] = []

    def executor(
        command: tuple[str, ...], cwd: Path, environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        lane = command[command.index("--lane") + 1]
        result_directory = Path(command[command.index("--result-directory") + 1])
        write_lane_observation(result_directory, observations[lane])
        seen.append((command, environment["SIMSOPT_BACKEND_MODE"]))
        return subprocess.CompletedProcess(command, 0, f"{lane} stdout", "")

    executions, loaded = execute_case_lanes(
        case_id="quadratic",
        lanes=("native-cpu", "jax-cpu", "jax-gpu"),
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        run_directory=tmp_path / "run.partial",
        repo_root=Path(__file__).resolve().parents[2],
        base_environment={"PRESERVED": "yes"},
        python_executable=sys.executable,
        smoke=True,
        executor=executor,
    )

    assert tuple(item.lane for item in executions) == (
        "native-cpu",
        "jax-cpu",
        "jax-gpu",
    )
    assert tuple(item.returncode for item in executions) == (0, 0, 0)
    assert loaded == observations
    assert [backend for _, backend in seen] == [
        "native_cpu",
        "jax_cpu_parity",
        "jax_gpu_parity",
    ]


@pytest.mark.parametrize("failure", ["nonzero", "missing_receipt", "wrong_lane"])
def test_runner_fails_closed_on_child_failure(tmp_path: Path, failure: str) -> None:
    observation = _observations()["native-cpu"]

    def executor(
        command: tuple[str, ...], cwd: Path, environment: dict[str, str]
    ) -> subprocess.CompletedProcess[str]:
        del cwd, environment
        result_directory = Path(command[command.index("--result-directory") + 1])
        if failure != "missing_receipt":
            receipt = (
                dataclasses.replace(observation, lane="jax-cpu")
                if failure == "wrong_lane"
                else observation
            )
            write_lane_observation(result_directory, receipt)
        return subprocess.CompletedProcess(
            command,
            7 if failure == "nonzero" else 0,
            "child stdout",
            "child stderr",
        )

    with pytest.raises(RunnerError, match="native-cpu"):
        execute_case_lanes(
            case_id="quadratic",
            lanes=("native-cpu",),
            input_bundle_path=tmp_path / "input_bundle.json",
            run_directory=tmp_path / "run.partial",
            repo_root=Path(__file__).resolve().parents[2],
            base_environment={},
            python_executable=sys.executable,
            smoke=True,
            executor=executor,
        )


def test_traceable_least_squares_case_runs_native_and_jax_cpu_end_to_end(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    case = get_case("traceable-least-squares")
    bundle = case.create_input(tmp_path / "inputs", True)

    executions, observations = execute_case_lanes(
        case_id=case.case_id,
        lanes=("native-cpu", "jax-cpu"),
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        run_directory=tmp_path / "run.partial",
        repo_root=repo_root,
        base_environment={},
        python_executable=sys.executable,
        smoke=True,
    )

    assert bundle.case_id == case.case_id
    assert len(executions) == 2
    native = observations["native-cpu"]
    jax_cpu = observations["jax-cpu"]
    for observable in (
        "initial:residual",
        "initial:residual_jacobian",
        "initial:objective_sum_squares",
        "initial:solver_cost",
        "initial:objective_gradient",
    ):
        np.testing.assert_allclose(
            native.values[observable],
            jax_cpu.values[observable],
            rtol=1.0e-8,
            atol=1.0e-10,
        )
    final_tolerance = parity_ladder_tolerances("native_workflow")
    for observable in (
        "final:parameters",
        "final:residual",
        "final:residual_jacobian",
        "final:objective_sum_squares",
        "final:solver_cost",
        "final:objective_gradient",
    ):
        np.testing.assert_allclose(
            native.values[observable],
            jax_cpu.values[observable],
            rtol=float(final_tolerance["whole_solve_value_rtol"]),
            atol=float(final_tolerance["whole_solve_value_atol"]),
        )
    np.testing.assert_allclose(
        native.values["initial:objective_gradient"],
        2.0
        * native.values["initial:residual_jacobian"].T
        @ native.values["initial:residual"],
    )
    for observation in (native, jax_cpu):
        for phase in ("initial", "final"):
            np.testing.assert_allclose(
                observation.values[f"{phase}:solver_cost"],
                0.5 * observation.values[f"{phase}:objective_sum_squares"],
            )
    assert native.success and jax_cpu.success
    assert native.input_fingerprint == jax_cpu.input_fingerprint
    assert native.effective_construction_fingerprint == (
        jax_cpu.effective_construction_fingerprint
    )


def test_run_parity_cli_publishes_complete_wave_a_cpu_artifact(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    initial_repository_state = collect_repository_state(repo_root)
    completed = subprocess.run(
        (
            sys.executable,
            str(repo_root / "examples" / "jax" / "run_parity.py"),
            "--case",
            "traceable-least-squares",
            "--lanes",
            "native-cpu,jax-cpu",
            "--smoke",
            "--artifact-root",
            str(tmp_path),
        ),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    published = [path for path in tmp_path.iterdir() if path.is_dir()]
    assert len(published) == 1
    assert not published[0].name.endswith(".partial")
    summary = json.loads((published[0] / "summary.json").read_text(encoding="utf-8"))
    assert summary["verdict"] == "pass"
    assert summary["lanes"] == ["native-cpu", "jax-cpu"]
    assert summary["authoritative"] is False
    assert len(summary["repository_commit"]) == 40
    assert summary["repository_dirty"] is initial_repository_state.repository_dirty
    assert len(summary["tracked_diff_sha256"]) == 64
    assert summary["untracked_files"] == sorted(summary["untracked_files"])
    assert len(summary["cases"]) == 1
    assert summary["cases"][0]["jax_example_id"] == "traceable-least-squares"
    assert summary["cases"][0]["native_source"] == "1_Simple/just_a_quadratic.py"
    assert summary["cases"][0]["classification"] == "full"
    assert summary["cases"][0]["scale_tier"] == "bounded"
    assert summary["cases"][0]["oracle_kind"] == "native_python_scipy"
    assert len(summary["cases"][0]["comparisons"]) == 11
    for lane in ("native-cpu", "jax-cpu"):
        receipt = json.loads(
            (
                published[0] / "traceable-least-squares" / lane / "lane_result.json"
            ).read_text(encoding="utf-8")
        )
        provenance = receipt["provenance"]
        assert provenance["repository_commit"] == summary["repository_commit"]
        assert (
            provenance["repository_dirty"] is initial_repository_state.repository_dirty
        )
        assert provenance["executed_sources"]
        assert provenance["python_version"]
        assert provenance["lane_environment_policy"]["SIMSOPT_BACKEND_MODE"]

    for mutation in ("input-json", "input-sidecar"):
        copied_run = tmp_path / mutation / published[0].name
        shutil.copytree(published[0], copied_run)
        input_root = copied_run / "traceable-least-squares" / "inputs"
        bundle_path = input_root / "input_bundle.json"
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        if mutation == "input-json":
            bundle["configuration"]["rtol"] = 1.0e-4
            bundle_path.write_bytes(canonical_json_bytes(bundle))
        else:
            targets = bundle["arrays"]["targets"]
            write_array(
                input_root,
                targets["path"],
                np.asarray([9.0, 8.0, 7.0], dtype=np.float64),
            )
        with pytest.raises(ValueError, match="fingerprint|SHA-256"):
            audit_published_run(copied_run, repo_root=repo_root)

    jax_receipt_path = (
        published[0] / "traceable-least-squares" / "jax-cpu" / "lane_result.json"
    )
    jax_receipt = json.loads(jax_receipt_path.read_text(encoding="utf-8"))
    changed_reference = write_array(
        jax_receipt_path.parent,
        jax_receipt["values"]["initial:objective_sum_squares"]["path"],
        np.asarray(123.0, dtype=np.float64),
    )
    jax_receipt["values"]["initial:objective_sum_squares"] = dataclasses.asdict(
        changed_reference
    )
    jax_receipt_path.write_bytes(canonical_json_bytes(jax_receipt))

    with pytest.raises(ValueError, match="recomputed comparison verdict"):
        audit_published_run(published[0], repo_root=repo_root)


def test_run_parity_cli_resolves_relative_artifact_root(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        (
            sys.executable,
            str(repo_root / "examples" / "jax" / "run_parity.py"),
            "--case",
            "traceable-least-squares",
            "--lanes",
            "native-cpu,jax-cpu",
            "--smoke",
            "--artifact-root",
            "parity-artifacts",
        ),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    published = tuple((tmp_path / "parity-artifacts").glob("*/summary.json"))
    assert len(published) == 1


def test_curve_length_case_runs_native_and_jax_cpu_end_to_end(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    case = get_case("curve-length-optimization")
    case.create_input(tmp_path / "inputs", True)

    _, observations = execute_case_lanes(
        case_id=case.case_id,
        lanes=("native-cpu", "jax-cpu"),
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        run_directory=tmp_path / "run.partial",
        repo_root=repo_root,
        base_environment={},
        python_executable=sys.executable,
        smoke=True,
    )

    native = observations["native-cpu"]
    jax_cpu = observations["jax-cpu"]
    np.testing.assert_allclose(
        native.values["initial:objective"],
        jax_cpu.values["initial:objective"],
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        native.values["initial:objective_gradient"],
        jax_cpu.values["initial:objective_gradient"],
        rtol=1.0e-8,
        atol=1.0e-10,
    )
    tolerance = parity_ladder_tolerances("native_workflow")
    for observable in ("parameters", "objective", "objective_gradient"):
        np.testing.assert_allclose(
            native.values[f"final:{observable}"],
            jax_cpu.values[f"final:{observable}"],
            rtol=float(tolerance["whole_solve_value_rtol"]),
            atol=float(tolerance["whole_solve_value_atol"]),
        )
    circle_length = 4.0 * np.pi
    np.testing.assert_allclose(
        native.values["final:objective"], circle_length, rtol=1.0e-9
    )
    assert native.success and jax_cpu.success


def test_surface_geometry_case_runs_native_and_jax_cpu_end_to_end(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    case = get_case("surface-geometry-optimization")
    case.create_input(tmp_path / "inputs", True)
    _, observations = execute_case_lanes(
        case_id=case.case_id,
        lanes=("native-cpu", "jax-cpu"),
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        run_directory=tmp_path / "run.partial",
        repo_root=repo_root,
        base_environment={},
        python_executable=sys.executable,
        smoke=True,
    )
    native = observations["native-cpu"]
    jax_cpu = observations["jax-cpu"]
    for observable in (
        "residual",
        "residual_jacobian",
        "objective_sum_squares",
        "objective_gradient",
        "area",
        "volume",
    ):
        np.testing.assert_allclose(
            native.values[f"initial:{observable}"],
            jax_cpu.values[f"initial:{observable}"],
            rtol=1.0e-8,
            atol=1.0e-10,
        )
    tolerance = parity_ladder_tolerances("native_workflow")
    for observable in (
        "parameter_invariants",
        "residual",
        "objective_sum_squares",
        "objective_gradient",
        "area",
        "volume",
    ):
        np.testing.assert_allclose(
            native.values[f"final:{observable}"],
            jax_cpu.values[f"final:{observable}"],
            rtol=float(tolerance["whole_solve_value_rtol"]),
            atol=float(tolerance["whole_solve_value_atol"]),
        )
    for observation in (native, jax_cpu):
        parameters = observation.values["final:parameters"]
        residual_jacobian = observation.values["final:residual_jacobian"]
        assert parameters.shape == (2,)
        assert residual_jacobian.shape == (2, 2)
        assert np.all(np.isfinite(parameters))
        assert np.all(np.isfinite(residual_jacobian))
        assert observation.applicability["final:parameters"] is False
        for phase in ("initial", "final"):
            np.testing.assert_allclose(
                observation.values[f"{phase}:solver_cost"],
                0.5 * observation.values[f"{phase}:objective_sum_squares"],
            )
    assert native.success and jax_cpu.success


def test_coil_flux_case_runs_native_and_jax_cpu_end_to_end(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    case = get_case("coil-flux-optimization")
    case.create_input(tmp_path / "inputs", True)

    _, observations = execute_case_lanes(
        case_id=case.case_id,
        lanes=("native-cpu", "jax-cpu"),
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        run_directory=tmp_path / "run.partial",
        repo_root=repo_root,
        base_environment={},
        python_executable=sys.executable,
        smoke=True,
    )

    native = observations["native-cpu"]
    jax_cpu = observations["jax-cpu"]
    for phase in ("initial", "final"):
        for observable in ("parameters", "flux", "flux_gradient", "coil_length"):
            np.testing.assert_allclose(
                native.values[f"{phase}:{observable}"],
                jax_cpu.values[f"{phase}:{observable}"],
                rtol=1.0e-8,
                atol=1.0e-12,
            )
    assert native.values["final:flux"].item() <= (
        1.0e-12 * native.values["initial:flux"].item()
    )
    assert native.success and jax_cpu.success


def test_permanent_magnet_case_matches_cpp_and_jax_cpu_end_to_end(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    case = get_case("permanent-magnet-optimization")
    case.create_input(tmp_path / "inputs", True)

    _, observations = execute_case_lanes(
        case_id=case.case_id,
        lanes=("native-cpu", "jax-cpu"),
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        run_directory=tmp_path / "run.partial",
        repo_root=repo_root,
        base_environment={},
        python_executable=sys.executable,
        smoke=True,
    )

    native = observations["native-cpu"]
    jax_cpu = observations["jax-cpu"]
    for observable in ("moments", "residual", "objective_sum_squares"):
        np.testing.assert_allclose(
            native.values[f"initial:{observable}"],
            jax_cpu.values[f"initial:{observable}"],
        )
        np.testing.assert_allclose(
            native.values[f"final:{observable}"],
            jax_cpu.values[f"final:{observable}"],
        )
    np.testing.assert_array_equal(
        native.values["final:selected_dipoles"],
        jax_cpu.values["final:selected_dipoles"],
    )
    assert native.values["final:objective_sum_squares"].item() < (
        native.values["initial:objective_sum_squares"].item()
    )
    assert native.success and jax_cpu.success


def test_wireframe_rcls_case_matches_native_and_jax_cpu_end_to_end(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    case = get_case("wireframe-optimization")
    case.create_input(tmp_path / "inputs", True)

    _, observations = execute_case_lanes(
        case_id=case.case_id,
        lanes=("native-cpu", "jax-cpu"),
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        run_directory=tmp_path / "run.partial",
        repo_root=repo_root,
        base_environment={},
        python_executable=sys.executable,
        smoke=True,
    )

    native = observations["native-cpu"]
    jax_cpu = observations["jax-cpu"]
    for phase in ("initial", "final"):
        for observable in (
            "currents",
            "normal_field_residual",
            "objective",
            "objective_gradient",
            "constraint_residual",
        ):
            np.testing.assert_allclose(
                native.values[f"{phase}:{observable}"],
                jax_cpu.values[f"{phase}:{observable}"],
                rtol=1.0e-9,
                atol=1.0e-10,
            )
    assert np.linalg.norm(native.values["final:constraint_residual"]) < 1.0e-10
    assert native.success and jax_cpu.success


def test_coil_force_fixed_state_matches_native_and_jax_cpu_end_to_end(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    case = get_case("coil-force-and-finite-build")
    case.create_input(tmp_path / "inputs", True)

    _, observations = execute_case_lanes(
        case_id=case.case_id,
        lanes=("native-cpu", "jax-cpu"),
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        run_directory=tmp_path / "run.partial",
        repo_root=repo_root,
        base_environment={},
        python_executable=sys.executable,
        smoke=True,
    )

    native = observations["native-cpu"]
    jax_cpu = observations["jax-cpu"]
    for phase in ("initial", "final"):
        for observable in (
            "parameters",
            "force_objective",
            "force_gradient",
            "frame",
            "frame_orthonormality_residual",
            "torsion",
        ):
            np.testing.assert_allclose(
                native.values[f"{phase}:{observable}"],
                jax_cpu.values[f"{phase}:{observable}"],
                rtol=1.0e-8,
                atol=1.0e-9,
            )
    assert (
        np.max(np.abs(native.values["final:frame_orthonormality_residual"])) < 1.0e-12
    )
    assert native.success and jax_cpu.success


def test_qfm_case_matches_native_and_jax_cpu_original_residuals(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    case = get_case("qfm-surface-optimization")
    case.create_input(tmp_path / "inputs", True)

    _, observations = execute_case_lanes(
        case_id=case.case_id,
        lanes=("native-cpu", "jax-cpu"),
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        run_directory=tmp_path / "run.partial",
        repo_root=repo_root,
        base_environment={},
        python_executable=sys.executable,
        smoke=True,
    )

    native = observations["native-cpu"]
    jax_cpu = observations["jax-cpu"]
    for phase in ("initial", "final"):
        for observable in (
            "parameters",
            "qfm_objective",
            "qfm_gradient",
            "constraint_value",
            "constraint_gradient",
            "penalty_objective",
            "penalty_gradient",
        ):
            np.testing.assert_allclose(
                native.values[f"{phase}:{observable}"],
                jax_cpu.values[f"{phase}:{observable}"],
                rtol=1.0e-6 if phase == "final" else 1.0e-8,
                atol=1.0e-7 if phase == "final" else 1.0e-10,
            )
    assert native.values["final:penalty_objective"].item() < (
        native.values["initial:penalty_objective"].item()
    )
    terminal_constraint_atol = float(
        parity_ladder_tolerances("native_workflow")["terminal_constraint_norm_atol"]
    )
    for observation in (native, jax_cpu):
        assert (
            np.max(np.abs(observation.values["final:constraint_value"]))
            <= terminal_constraint_atol
        )
        assert observation.normalized_status == "converged"
        assert all(
            isinstance(counter, int) and counter >= 0
            for counter in (observation.nit, observation.nfev, observation.njev)
        )
    assert native.success and jax_cpu.success
