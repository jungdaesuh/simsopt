"""CPU-only contracts for the single-stage campaign measurement profile."""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass, field, replace
from pathlib import Path

import benchmarks.run_jax_native_example_measurements as measurements
import numpy as np
import pytest
from benchmarks.run_jax_native_example_measurements import (
    SINGLE_STAGE_SPEED_PROFILE_IDS,
    MeasurementRunnerError,
    MonitoredCommandResult,
    _single_stage_endpoint_audit,
    _validate_observation,
    _validate_single_stage_campaign_identity_pair,
    _validate_single_stage_campaign_observation,
    _validate_single_stage_trajectory_count,
    build_measurement_environment,
    build_profile_command,
    build_single_stage_speed_collection_plan,
    collect_single_stage_speed_campaign,
)
from benchmarks.single_stage_speed_campaign_receipt import (
    DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE,
)
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.provenance import (
    DeviceMetadata,
    ExecutedSource,
    LaneProvenance,
)
from simsopt.single_stage_boozer_vacuum import JAX_FAST_DRIVER_ID, JAX_OPTAX_DRIVER_ID
from simsopt_jax.parity_tolerances import parity_ladder_tolerances


def test_optax_profile_uses_fast_gpu_runtime_with_explicit_optimizer_route(
    tmp_path: Path,
) -> None:
    trajectory_path = tmp_path / "trajectory-warm-0.jsonl"
    timing_path = tmp_path / "optimization-timing-warm-0.json"
    command = build_profile_command(
        python_executable="/python",
        case_id="native-single-stage-boozer-vacuum-optimization",
        profile_id="jax_gpu_optax",
        input_bundle_path=tmp_path / "inputs" / "input_bundle.json",
        result_directory=tmp_path / "result",
        scale="native_default",
        trajectory_path=trajectory_path,
        optimization_timing_path=timing_path,
    )
    environment = build_measurement_environment(
        "jax_gpu_optax",
        allocation_sensitive=False,
        base_environment={"SIMSOPT_BACKEND_MODE": "stale"},
        repo_root=tmp_path,
    )

    assert command[command.index("--lane") + 1] == "jax-gpu"
    assert command[command.index("--trajectory-path") + 1] == str(trajectory_path)
    assert command[command.index("--optimization-timing-path") + 1] == str(timing_path)
    assert command[command.index("--optimizer-backend") + 1] == "optax-lbfgs"
    assert environment["SIMSOPT_BACKEND_MODE"] == "jax_gpu_fast"
    assert environment["JAX_PLATFORMS"] == "cuda"
    assert environment["SIMSOPT_PRECISION"] == "fp64"


def test_campaign_plan_is_balanced_and_isolates_every_sample() -> None:
    plan = build_single_stage_speed_collection_plan(mirror_index=0)
    timed = tuple(run for run in plan if not run.allocation_sensitive)
    allocations = tuple(run for run in plan if run.allocation_sensitive)

    assert len(timed) == 4 * (1 + 1 + 7)
    assert not allocations
    assert tuple(run.profile_id for run in timed[:4]) == SINGLE_STAGE_SPEED_PROFILE_IDS
    for sample_index in range(7):
        warm = tuple(
            run
            for run in timed
            if run.phase == "warm" and run.sample_index == sample_index
        )
        assert {run.profile_id for run in warm} == set(SINGLE_STAGE_SPEED_PROFILE_IDS)
        assert {run.order_position for run in warm} == set(range(4))


@pytest.mark.parametrize(
    ("process_wall_seconds", "optimization_wall_seconds", "rejects"),
    (
        (8.4, 0.4, False),
        (0.4, 0.5, True),
    ),
)
def test_speed_run_validates_optimizer_timing_and_retains_process_diagnostic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    process_wall_seconds: float,
    optimization_wall_seconds: float,
    rejects: bool,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "logs").mkdir()

    def fake_monitored_command(**kwargs) -> MonitoredCommandResult:
        command = kwargs["command"]
        timing_path = Path(command[command.index("--optimization-timing-path") + 1])
        timing_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "wall_seconds": optimization_wall_seconds,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return MonitoredCommandResult(
            returncode=0,
            termination="normal",
            wall_seconds=process_wall_seconds,
            peak_process_tree_rss_bytes=1,
            peak_gpu_process_bytes=None,
            gpu_counter_status="not_applicable",
            stdout_sha256="stdout",
            stderr_sha256="stderr",
        )

    observation = _campaign_observation(
        "native_cpu",
        driver="simsopt_scipy_bfgs_with_boozer_newton",
    )
    monkeypatch.setattr(
        measurements,
        "execute_monitored_command",
        fake_monitored_command,
    )
    monkeypatch.setattr(
        measurements,
        "load_lane_observation",
        lambda result_directory: observation,
    )
    run = measurements.CollectionRun(
        profile_id="native_cpu",
        phase="cold",
        sample_index=0,
        order_position=0,
        measured=True,
        allocation_sensitive=False,
    )

    def execute_speed_run():
        return measurements._execute_single_stage_speed_run(
            bundle_path=tmp_path / "input_bundle.json",
            workspace=workspace,
            run=run,
            sequence_index=0,
            environment={},
            python_executable="/python",
            isolated_site=False,
            repo_root=tmp_path,
            gpu_index=0,
            poll_interval_seconds=0.01,
            timeout_seconds=1.0,
        )

    if rejects:
        with pytest.raises(
            MeasurementRunnerError,
            match="optimizer timing exceeds subprocess wall",
        ):
            execute_speed_run()
        assert not tuple((workspace / "logs").glob("*.timing.json"))
        return

    _, loaded_observation, _, measured_wall_seconds = execute_speed_run()

    assert loaded_observation is observation
    assert measured_wall_seconds == optimization_wall_seconds
    diagnostic_path = next((workspace / "logs").glob("*.timing.json"))
    assert json.loads(diagnostic_path.read_text(encoding="utf-8")) == {
        "optimization_wall_seconds": optimization_wall_seconds,
        "subprocess_wall_seconds": process_wall_seconds,
    }


def test_optax_profile_rejects_forged_driver() -> None:
    observation = LaneObservation(
        lane="jax-gpu",
        backend_mode="jax_gpu_fast",
        platform="gpu",
        precision="fp64",
        scale="native_default",
        input_fingerprint="input",
        configuration_fingerprint="configuration",
        effective_construction_fingerprint="construction",
        driver="simsopt_jax_host_lbfgsb_with_traceable_boozer_newton",
        normalized_status="budget_exhausted",
        raw_status="iteration-limit",
        success=True,
        nit=2,
        nfev=3,
        njev=3,
        completed_workflow_stages=("optimize",),
        provenance=None,
        values={"final:objective": np.asarray(1.0, dtype=np.float64)},
    )

    with pytest.raises(MeasurementRunnerError, match="jax_gpu_optax driver must be"):
        _validate_observation(
            observation,
            profile_id="jax_gpu_optax",
            scale="native_default",
            input_fingerprint="input",
        )


def test_optax_driver_id_names_the_actual_provider() -> None:
    assert "optax_lbfgs" in JAX_OPTAX_DRIVER_ID


@dataclass(frozen=True)
class _CampaignBundle:
    input_fingerprint: str = "campaign-input"
    configuration: dict[str, int] = field(default_factory=lambda: {"outer_maxiter": 2})


class _CampaignCase:
    def create_input(self, root: Path, scale: str) -> _CampaignBundle:
        assert scale == "native_default"
        root.mkdir(parents=True)
        (root / "input_bundle.json").write_text("{}\n", encoding="utf-8")
        return _CampaignBundle()


def _campaign_provenance(profile_id: str, synchronization: str) -> LaneProvenance:
    if profile_id == "native_cpu":
        lane_environment_policy: dict[str, str] = {}
        devices = (DeviceMetadata(0, "cpu", "test", 0),)
    elif profile_id.startswith("jax_gpu"):
        lane_environment_policy = {
            "SIMSOPT_BACKEND_MODE": "jax_gpu_fast",
            "SIMSOPT_EXACT_ADJOINT_DENSE_LU": "1",
        }
        devices = (DeviceMetadata(0, "cuda", "test", 0),)
    else:
        lane_environment_policy = {
            "SIMSOPT_BACKEND_MODE": "jax_cpu_fast",
            "SIMSOPT_EXACT_ADJOINT_DENSE_LU": "1",
        }
        devices = (DeviceMetadata(0, "cpu", "test", 0),)
    return LaneProvenance(
        repository_commit="campaign-commit",
        repository_dirty=False,
        tracked_diff_sha256="campaign-diff",
        untracked_files=(),
        executed_sources=(ExecutedSource("campaign.py", "source-hash", None),),
        python_version="3.12",
        jax_version="0.0",
        simsopt_version="0.0",
        simsopt_version_commit=None,
        simsopt_version_checkout_compatible=None,
        lane_environment_policy=lane_environment_policy,
        jax_effective_transfer_guards={},
        devices=devices,
        host_peak_rss_bytes=1,
        host_peak_rss_method="test",
        device_memory_peak_bytes=None,
        device_memory_status="not_applicable",
        memory_measurement_scope="test",
        steady_state_memory_measured=False,
        measurement_synchronization=synchronization,
        simsoptpp_path=None,
        simsoptpp_sha256=None,
        simsoptpp_version=None,
        simsoptpp_build_commit=None,
        simsoptpp_checkout_compatible=None,
        authoritative=False,
    )


def _campaign_observation(profile_id: str, *, driver: str) -> LaneObservation:
    is_native = profile_id == "native_cpu"
    is_gpu = profile_id.startswith("jax_gpu")
    return LaneObservation(
        lane="native-cpu" if is_native else "jax-gpu" if is_gpu else "jax-cpu",
        backend_mode="native_cpu"
        if is_native
        else "jax_gpu_fast"
        if is_gpu
        else "jax_cpu_fast",
        platform="cpu" if not is_gpu else "gpu",
        precision="fp64",
        scale="native_default",
        input_fingerprint="campaign-input",
        configuration_fingerprint="campaign-configuration",
        effective_construction_fingerprint="campaign-construction",
        driver=driver,
        normalized_status="budget_exhausted",
        raw_status="iteration-limit",
        success=False,
        nit=2,
        nfev=3,
        njev=3,
        completed_workflow_stages=("campaign-stage",),
        provenance=_campaign_provenance(
            profile_id,
            "native synchronous execution"
            if is_native
            else "jax.block_until_ready over published observation values",
        ),
        values={
            "initial:parameters": np.asarray([1.0, 2.0], dtype=np.float64),
            "final:objective": np.asarray(1.0, dtype=np.float64),
            "final:iota": np.asarray(0.4, dtype=np.float64),
            "final:volume": np.asarray(0.1, dtype=np.float64),
            "final:non_qs_ratio": np.asarray(0.2, dtype=np.float64),
            "final:boozer_residual": np.asarray(0.01, dtype=np.float64),
            "final:inner_solver_success": np.asarray(True, dtype=np.bool_),
            "final:gradient": np.asarray([0.1, 0.2], dtype=np.float64),
            "final:parameters": np.asarray([1.0, 2.0], dtype=np.float64),
            "final:endpoint_certificate_success": np.asarray(False, dtype=np.bool_),
            "final:endpoint_initial_stationary": np.asarray(False, dtype=np.bool_),
            "final:endpoint_terminal_stationary": np.asarray(False, dtype=np.bool_),
            "final:endpoint_constraints_satisfied": np.asarray(True, dtype=np.bool_),
            "final:outer_solver_status": np.asarray(1, dtype=np.int64),
        },
    )


def test_campaign_rejects_nonfinite_terminal_gradient() -> None:
    observation = _campaign_observation(
        "native_cpu", driver="simsopt_scipy_bfgs_with_boozer_newton"
    )
    observation = replace(
        observation,
        values={
            **observation.values,
            "final:gradient": np.asarray([0.1, np.nan], dtype=np.float64),
        },
    )

    with pytest.raises(MeasurementRunnerError, match="final:gradient must be finite"):
        _validate_single_stage_campaign_observation(
            observation,
            profile_id="native_cpu",
            input_fingerprint="campaign-input",
        )


def test_campaign_rejects_mismatched_child_runtime_versions() -> None:
    reference = _campaign_observation(
        "native_cpu", driver="simsopt_scipy_bfgs_with_boozer_newton"
    )
    lane = _campaign_observation("jax_cpu_fast", driver="jax-driver")
    assert lane.provenance is not None
    lane = replace(
        lane,
        provenance=replace(lane.provenance, python_version="3.13"),
    )

    with pytest.raises(MeasurementRunnerError, match="python_version"):
        _validate_single_stage_campaign_identity_pair(
            reference,
            "jax_cpu_fast",
            lane,
        )


def test_campaign_rejects_mismatched_initial_parameters() -> None:
    reference = _campaign_observation(
        "native_cpu", driver="simsopt_scipy_bfgs_with_boozer_newton"
    )
    lane = _campaign_observation("jax_cpu_fast", driver="jax-driver")
    lane = replace(
        lane,
        values={
            **lane.values,
            "initial:parameters": np.asarray([1.0, 2.5], dtype=np.float64),
        },
    )

    with pytest.raises(MeasurementRunnerError, match="initial parameters"):
        _validate_single_stage_campaign_identity_pair(
            reference,
            "jax_cpu_fast",
            lane,
        )


def test_campaign_rejects_incomplete_iteration_trajectory() -> None:
    observation = _campaign_observation(
        "native_cpu", driver="simsopt_scipy_bfgs_with_boozer_newton"
    )

    with pytest.raises(MeasurementRunnerError, match="1 records for reported nit=2"):
        _validate_single_stage_trajectory_count(
            profile_id="native_cpu",
            phase="warm",
            trajectory=(
                measurements.TrajectoryPoint(
                    iteration=1,
                    objective=1.0,
                    wall_seconds_from_start=0.01,
                ),
            ),
            observation=observation,
            iteration_budget=2,
        )


def test_campaign_rejects_a_contiguous_trajectory_short_of_the_budget() -> None:
    observation = replace(
        _campaign_observation(
            "native_cpu",
            driver="simsopt_scipy_bfgs_with_boozer_newton",
        ),
        nit=1,
    )

    with pytest.raises(MeasurementRunnerError, match="end exactly at iteration budget"):
        _validate_single_stage_trajectory_count(
            profile_id="native_cpu",
            phase="warm",
            trajectory=(
                measurements.TrajectoryPoint(
                    iteration=1,
                    objective=1.0,
                    wall_seconds_from_start=0.01,
                ),
            ),
            observation=observation,
            iteration_budget=2,
        )


def test_campaign_endpoint_audit_binds_provider_and_parameter_hashes() -> None:
    observation = _campaign_observation("jax_gpu_optax", driver=JAX_OPTAX_DRIVER_ID)
    observation = replace(
        observation,
        input_fingerprint="1" * 64,
        configuration_fingerprint="2" * 64,
        effective_construction_fingerprint="3" * 64,
    )

    audit = _single_stage_endpoint_audit("jax_gpu_optax", observation)

    assert audit.backend_mode == "jax_gpu_fast"
    assert audit.driver == JAX_OPTAX_DRIVER_ID
    assert audit.adjoint_route == DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE
    assert len(audit.initial_parameters_sha256) == 64
    assert len(audit.final_parameters_sha256) == 64
    assert audit.final_gradient_inf_norm == pytest.approx(0.2)
    assert audit.certificate.outer_status == 1


def test_campaign_accepts_persisted_singleton_scalars() -> None:
    observation = _campaign_observation(
        "native_cpu", driver="simsopt_scipy_bfgs_with_boozer_newton"
    )
    observation = replace(
        observation,
        values={
            **observation.values,
            "final:objective": np.asarray([1.0], dtype=np.float64),
            "final:iota": np.asarray([0.4], dtype=np.float64),
            "final:volume": np.asarray([0.1], dtype=np.float64),
            "final:non_qs_ratio": np.asarray([0.2], dtype=np.float64),
            "final:boozer_residual": np.asarray([0.01], dtype=np.float64),
            "final:inner_solver_success": np.asarray([True], dtype=np.bool_),
            "final:endpoint_certificate_success": np.asarray([False], dtype=np.bool_),
            "final:endpoint_initial_stationary": np.asarray([False], dtype=np.bool_),
            "final:endpoint_terminal_stationary": np.asarray([False], dtype=np.bool_),
            "final:endpoint_constraints_satisfied": np.asarray([True], dtype=np.bool_),
            "final:outer_solver_status": np.asarray([1], dtype=np.int64),
        },
    )

    _validate_single_stage_campaign_observation(
        observation,
        profile_id="native_cpu",
        input_fingerprint="campaign-input",
    )
    audit = _single_stage_endpoint_audit("native_cpu", observation)

    assert audit.certificate.outer_status == 1
    assert audit.adjoint_route is None


def test_campaign_rejects_multielement_persisted_scalar() -> None:
    observation = _campaign_observation(
        "native_cpu", driver="simsopt_scipy_bfgs_with_boozer_newton"
    )
    observation = replace(
        observation,
        values={
            **observation.values,
            "final:objective": np.asarray([1.0, 1.0], dtype=np.float64),
        },
    )

    with pytest.raises(MeasurementRunnerError, match="must be exactly one FP64 scalar"):
        _validate_single_stage_campaign_observation(
            observation,
            profile_id="native_cpu",
            input_fingerprint="campaign-input",
        )


def test_campaign_rejects_noncanonical_singleton_scalar_shape() -> None:
    observation = _campaign_observation(
        "native_cpu", driver="simsopt_scipy_bfgs_with_boozer_newton"
    )
    observation = replace(
        observation,
        values={
            **observation.values,
            "final:objective": np.asarray([[1.0]], dtype=np.float64),
        },
    )

    with pytest.raises(MeasurementRunnerError, match="must be exactly one FP64 scalar"):
        _validate_single_stage_campaign_observation(
            observation,
            profile_id="native_cpu",
            input_fingerprint="campaign-input",
        )


def test_public_campaign_uses_four_lane_plan_and_rejects_forged_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_plan = measurements.build_single_stage_speed_collection_plan
    plan_calls: list[int] = []
    executed_profiles: list[str] = []

    def tracked_plan(mirror_index: int):
        plan_calls.append(mirror_index)
        return original_plan(mirror_index)

    def fake_execute(**kwargs):
        run = kwargs["run"]
        executed_profiles.append(run.profile_id)
        trajectory_path = kwargs["workspace"] / f"{len(executed_profiles)}.jsonl"
        trajectory_path.write_text(
            '{"iteration":1,"objective":1.5,"wall_seconds_from_start":0.01}\n'
            '{"iteration":2,"objective":1.0,"wall_seconds_from_start":0.02}\n',
            encoding="utf-8",
        )
        driver = (
            "simsopt_scipy_bfgs_with_boozer_newton"
            if run.profile_id == "native_cpu"
            else "forged-provider"
        )
        return (
            MonitoredCommandResult(
                returncode=0,
                termination="normal",
                wall_seconds=0.01,
                peak_process_tree_rss_bytes=1,
                peak_gpu_process_bytes=None,
                gpu_counter_status="not_applicable",
                stdout_sha256="stdout",
                stderr_sha256="stderr",
            ),
            _campaign_observation(run.profile_id, driver=driver),
            trajectory_path,
            0.02,
        )

    monkeypatch.setattr(measurements, "get_case", lambda case_id: _CampaignCase())
    monkeypatch.setattr(
        measurements, "build_single_stage_speed_collection_plan", tracked_plan
    )
    monkeypatch.setattr(measurements, "_execute_single_stage_speed_run", fake_execute)
    monkeypatch.setattr(
        measurements, "_gpu_concurrent_use_preflight", lambda gpu_index: "pass"
    )

    artifact_root = Path.cwd() / f".{tmp_path.name}-campaign"
    try:
        with pytest.raises(MeasurementRunnerError, match="jax_gpu_fast driver must be"):
            collect_single_stage_speed_campaign(
                artifact_root=artifact_root,
                python_executable="/python",
                repo_root=tmp_path,
            )
    finally:
        for workspace in artifact_root.parent.glob(f".{artifact_root.name}.partial-*"):
            shutil.rmtree(workspace)

    assert plan_calls == [0]
    assert executed_profiles == ["native_cpu", "jax_gpu_fast"]
    assert not artifact_root.exists()


def test_public_campaign_publishes_optimizer_window_not_process_duration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    input_fingerprint = "1" * 64
    configuration_fingerprint = "2" * 64
    construction_fingerprint = "3" * 64
    observed_route_selectors: dict[str, str | None] = {}

    class ValidCampaignCase:
        def create_input(self, root: Path, scale: str) -> _CampaignBundle:
            assert scale == "native_default"
            root.mkdir(parents=True)
            (root / "input_bundle.json").write_text("{}\n", encoding="utf-8")
            return _CampaignBundle(input_fingerprint=input_fingerprint)

    def fake_execute(**kwargs):
        run = kwargs["run"]
        observed_route_selectors[run.profile_id] = kwargs["environment"].get(
            "SIMSOPT_EXACT_ADJOINT_DENSE_LU"
        )
        trajectory_path = kwargs["workspace"] / (
            f"{run.phase}-{run.sample_index}-{run.profile_id}.jsonl"
        )
        trajectory_path.write_text(
            '{"iteration":1,"objective":1.5,"wall_seconds_from_start":0.01}\n'
            '{"iteration":2,"objective":1.0,"wall_seconds_from_start":0.02}\n',
            encoding="utf-8",
        )
        driver = {
            "native_cpu": "simsopt_scipy_bfgs_with_boozer_newton",
            "jax_gpu_fast": JAX_FAST_DRIVER_ID,
            "jax_gpu_optax": JAX_OPTAX_DRIVER_ID,
            "jax_cpu_fast": JAX_FAST_DRIVER_ID,
        }[run.profile_id]
        observation = replace(
            _campaign_observation(run.profile_id, driver=driver),
            input_fingerprint=input_fingerprint,
            configuration_fingerprint=configuration_fingerprint,
            effective_construction_fingerprint=construction_fingerprint,
        )
        return (
            MonitoredCommandResult(
                returncode=0,
                termination="normal",
                wall_seconds=8.02,
                peak_process_tree_rss_bytes=1,
                peak_gpu_process_bytes=None,
                gpu_counter_status="not_applicable",
                stdout_sha256="stdout",
                stderr_sha256="stderr",
            ),
            observation,
            trajectory_path,
            0.02,
        )

    monkeypatch.setattr(
        measurements,
        "get_case",
        lambda case_id: ValidCampaignCase(),
    )
    monkeypatch.setattr(measurements, "_execute_single_stage_speed_run", fake_execute)
    monkeypatch.setattr(
        measurements, "_gpu_concurrent_use_preflight", lambda gpu_index: "pass"
    )
    monkeypatch.setattr(
        measurements,
        "_gpu_identity",
        lambda gpu_index: ("synthetic-gpu", "uuid", "driver", "cuda"),
    )

    artifact_root = Path.cwd() / f".{tmp_path.name}-valid-campaign"
    try:
        published_root = collect_single_stage_speed_campaign(
            artifact_root=artifact_root,
            python_executable="/python",
            repo_root=Path.cwd(),
        )

        assert published_root == artifact_root
        assert observed_route_selectors == {
            "native_cpu": None,
            "jax_gpu_fast": "1",
            "jax_gpu_optax": "1",
            "jax_cpu_fast": "1",
        }
        native_endpoint = json.loads(
            (artifact_root / "lanes" / "native_cpu" / "endpoint.json").read_text()
        )
        native_measurement = json.loads(
            (artifact_root / "lanes" / "native_cpu" / "measurement.json").read_text()
        )
        assert {sample["wall_seconds"] for sample in native_measurement["samples"]} == {
            0.02
        }
        assert native_endpoint["audit"]["adjoint_route"] is None
        for lane_id in (
            "jax_gpu_custom",
            "jax_gpu_optax",
            "jax_cpu_custom",
        ):
            endpoint = json.loads(
                (artifact_root / "lanes" / lane_id / "endpoint.json").read_text()
            )
            assert (
                endpoint["audit"]["adjoint_route"] == DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE
            )
            assert tuple(
                row["observable"] for row in endpoint["parity"]["rows"]
            ) == tuple(
                name for name, _ in measurements._SINGLE_STAGE_PARITY_OBSERVABLES
            )
    finally:
        if artifact_root.exists():
            shutil.rmtree(artifact_root)
        for workspace in artifact_root.parent.glob(f".{artifact_root.name}.partial-*"):
            shutil.rmtree(workspace)


def test_campaign_rejects_a_claim_artifact_root_under_tmp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        measurements,
        "_gpu_concurrent_use_preflight",
        lambda gpu_index: pytest.fail("preflight must not run for a forbidden root"),
    )

    with pytest.raises(MeasurementRunnerError, match="must not be written under /tmp"):
        collect_single_stage_speed_campaign(
            artifact_root=tmp_path / "campaign",
            python_executable="/python",
        )


def test_campaign_direct_parity_emits_exactly_five_ssot_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        measurements,
        "arbitrate",
        lambda *args, **kwargs: pytest.fail("campaign parity must not invoke arbiter"),
    )
    rows = measurements._single_stage_campaign_parity_rows(
        profile_id="jax_gpu_optax",
        observations={
            "native_cpu": _campaign_observation(
                "native_cpu", driver="simsopt_scipy_bfgs_with_boozer_newton"
            ),
            "jax_gpu_optax": _campaign_observation(
                "jax_gpu_optax", driver=JAX_OPTAX_DRIVER_ID
            ),
        },
    )

    assert tuple(row.observable for row in rows) == (
        "final_objective",
        "final_iota",
        "final_volume",
        "final_non_qs_ratio",
        "final_boozer_residual",
    )
    assert all(
        row.tolerance
        == measurements.single_stage_speed_parity_tolerance(row.native_value)
        for row in rows
    )


def test_campaign_direct_parity_rejects_a_row_outside_the_ssot_tolerance() -> None:
    native = _campaign_observation(
        "native_cpu", driver="simsopt_scipy_bfgs_with_boozer_newton"
    )
    optax = _campaign_observation("jax_gpu_optax", driver=JAX_OPTAX_DRIVER_ID)
    optax = replace(
        optax,
        values={
            **optax.values,
            "final:objective": np.asarray(1.1, dtype=np.float64),
        },
    )

    with pytest.raises(MeasurementRunnerError, match="direct parity failed"):
        measurements._single_stage_campaign_parity_rows(
            profile_id="jax_gpu_optax",
            observations={"native_cpu": native, "jax_gpu_optax": optax},
        )


def test_campaign_direct_parity_rejects_lane_owned_asymmetric_bound() -> None:
    native = _campaign_observation(
        "native_cpu",
        driver="simsopt_scipy_bfgs_with_boozer_newton",
    )
    native_value = float(native.values["final:objective"])
    native_tolerance = measurements.single_stage_speed_parity_tolerance(native_value)
    lane_value = native_value + native_tolerance
    while abs(lane_value - native_value) <= native_tolerance:
        lane_value = math.nextafter(lane_value, math.inf)
    contract = parity_ladder_tolerances("mirror_single_stage_final_value")
    rtol = contract["rtol"]
    atol = contract["atol"]
    assert isinstance(rtol, float)
    assert isinstance(atol, float)
    assert abs(lane_value - native_value) <= atol + rtol * abs(lane_value)
    optax = _campaign_observation("jax_gpu_optax", driver=JAX_OPTAX_DRIVER_ID)
    optax = replace(
        optax,
        values={
            **optax.values,
            "final:objective": np.asarray(lane_value, dtype=np.float64),
        },
    )

    with pytest.raises(MeasurementRunnerError, match="direct parity failed"):
        measurements._single_stage_campaign_parity_rows(
            profile_id="jax_gpu_optax",
            observations={"native_cpu": native, "jax_gpu_optax": optax},
        )


def test_last_warm_order_is_rotated_while_campaign_lanes_remain_complete() -> None:
    plan = build_single_stage_speed_collection_plan(mirror_index=0)
    last_warm = tuple(
        run.profile_id for run in plan if run.phase == "warm" and run.sample_index == 6
    )

    assert last_warm == (
        "jax_gpu_optax",
        "jax_cpu_fast",
        "native_cpu",
        "jax_gpu_fast",
    )
    assert set(last_warm) == set(SINGLE_STAGE_SPEED_PROFILE_IDS)
