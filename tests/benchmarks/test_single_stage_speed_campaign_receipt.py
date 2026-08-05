"""Behavioral tests for the single-stage speed campaign receipt writer."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import benchmarks.single_stage_speed_campaign_receipt as receipt_writer
import pytest
from benchmarks.single_stage_speed_campaign_receipt import (
    DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE,
    CampaignMetadata,
    CampaignReceipt,
    EndpointAudit,
    EndpointCertificateAudit,
    EndpointObservables,
    LaneEndpoint,
    LaneReceipt,
    ParityRow,
    SampleMeasurement,
    SamplePhase,
    TrajectoryPoint,
    write_campaign_receipt,
)
from simsopt.single_stage_boozer_vacuum import (
    JAX_FAST_DRIVER_ID,
    JAX_OPTAX_DRIVER_ID,
)


def _parity_rows() -> tuple[ParityRow, ...]:
    return (
        ParityRow("final_objective", 1.0, 1.0, 0.0),
        ParityRow("final_iota", 0.2, 0.2, 0.0),
        ParityRow("final_volume", 0.1, 0.1, 0.0),
        ParityRow("final_non_qs_ratio", 0.01, 0.01, 0.0),
        ParityRow("final_boozer_residual", 0.001, 0.001, 0.0),
    )


def _endpoint_audit(backend_mode: str, driver: str) -> EndpointAudit:
    return EndpointAudit(
        backend_mode=backend_mode,
        driver=driver,
        input_fingerprint="1" * 64,
        configuration_fingerprint="2" * 64,
        effective_construction_fingerprint="3" * 64,
        initial_parameters_sha256="4" * 64,
        final_parameters_sha256="5" * 64,
        final_gradient_inf_norm=0.01,
        normalized_status="budget_exhausted",
        raw_status="iteration-limit",
        nit=2,
        nfev=3,
        njev=3,
        adjoint_route=(
            None if backend_mode == "native_cpu" else DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE
        ),
        certificate=EndpointCertificateAudit(
            success=False,
            initial_stationary=False,
            terminal_stationary=False,
            constraints_satisfied=True,
            outer_status=1,
        ),
    )


def _sample(
    phase: SamplePhase, sample_index: int, *, wall_seconds: float
) -> SampleMeasurement:
    return SampleMeasurement(
        phase=phase,
        sample_index=sample_index,
        wall_seconds=wall_seconds,
        trajectory=(
            TrajectoryPoint(1, 2.0, wall_seconds / 2.0),
            TrajectoryPoint(2, 1.0, wall_seconds),
        ),
    )


def _receipt() -> CampaignReceipt:
    samples = (
        _sample("cold", 0, wall_seconds=0.01),
        _sample("warmup", 0, wall_seconds=0.01),
        *(_sample("warm", index, wall_seconds=0.01) for index in range(7)),
    )
    native_endpoint = LaneEndpoint(
        observables=EndpointObservables(1.0, 0.2, 0.1, 0.01, 0.001, True),
        precision="fp64",
        audit=_endpoint_audit("native_cpu", "simsopt_scipy_bfgs_with_boozer_newton"),
    )
    return CampaignReceipt(
        metadata=CampaignMetadata(
            campaign_id="synthetic-campaign",
            git_describe="synthetic",
            hostname="test-host",
            device_name="test-device",
            python_version="3.12",
            jax_version="0.0",
            iteration_budget=2,
            scale="native_default",
            created_utc="2026-08-04T00:00:00Z",
        ),
        lanes=(
            LaneReceipt("native_cpu", samples, native_endpoint),
            LaneReceipt(
                "jax_gpu_custom",
                samples,
                LaneEndpoint(
                    EndpointObservables(1.0, 0.2, 0.1, 0.01, 0.001, True),
                    "fp64",
                    _endpoint_audit("jax_gpu_fast", JAX_FAST_DRIVER_ID),
                    _parity_rows(),
                ),
            ),
            LaneReceipt(
                "jax_gpu_optax",
                samples,
                LaneEndpoint(
                    EndpointObservables(1.0, 0.2, 0.1, 0.01, 0.001, True),
                    "fp64",
                    _endpoint_audit("jax_gpu_fast", JAX_OPTAX_DRIVER_ID),
                    _parity_rows(),
                ),
            ),
            LaneReceipt(
                "jax_cpu_custom",
                samples,
                LaneEndpoint(
                    EndpointObservables(1.0, 0.2, 0.1, 0.01, 0.001, True),
                    "fp64",
                    _endpoint_audit("jax_cpu_fast", JAX_FAST_DRIVER_ID),
                    _parity_rows(),
                ),
            ),
        ),
    )


def _synthetic_frozen_repo(repo_root: Path) -> None:
    frozen_paths = (
        "benchmarks/validate_single_stage_speed_claim.py",
        "docs/single_stage_speed_campaign_protocol.md",
        "src/simsopt_jax/parity_tolerances.py",
        "src/simsopt/optimization_endpoint.py",
        "src/simsopt_jax/solve/endpoint_certificate.py",
        "examples/jax/parity/arbiter.py",
    )
    for relative_path in frozen_paths:
        path = repo_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen content\n", encoding="utf-8")
    for command in (
        ("git", "init", "--quiet"),
        ("git", "add", "."),
        (
            "git",
            "-c",
            "user.name=pytest",
            "-c",
            "user.email=pytest@example.test",
            "commit",
            "--quiet",
            "-m",
            "baseline",
        ),
        ("git", "tag", "synthetic-baseline"),
    ):
        subprocess.run(command, cwd=repo_root, check=True)


def test_writer_creates_only_protocol_paths_and_validator_reaches_speed_verdict(
    tmp_path: Path,
) -> None:
    artifact_root = write_campaign_receipt(tmp_path / "campaign", _receipt())
    expected_paths = {
        Path("campaign.json"),
        *(
            Path("lanes") / lane_id / filename
            for lane_id in (
                "native_cpu",
                "jax_gpu_custom",
                "jax_gpu_optax",
                "jax_cpu_custom",
            )
            for filename in (
                "measurement.json",
                "endpoint.json",
                "trajectory-cold-0.jsonl",
                "trajectory-warmup-0.jsonl",
                *(f"trajectory-warm-{index}.jsonl" for index in range(7)),
            )
        ),
    }
    actual_paths = {
        path.relative_to(artifact_root)
        for path in artifact_root.rglob("*")
        if path.is_file()
    }
    assert actual_paths == expected_paths
    assert not tuple(tmp_path.glob(".campaign.staging-*"))
    campaign = json.loads((artifact_root / "campaign.json").read_text())
    assert campaign["lanes"] == [
        "native_cpu",
        "jax_gpu_custom",
        "jax_gpu_optax",
        "jax_cpu_custom",
    ]
    endpoint = json.loads(
        (artifact_root / "lanes" / "jax_gpu_custom" / "endpoint.json").read_text()
    )
    assert endpoint["audit"] == {
        "backend_mode": "jax_gpu_fast",
        "driver": JAX_FAST_DRIVER_ID,
        "adjoint_route": DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE,
        "input_fingerprint": "1" * 64,
        "configuration_fingerprint": "2" * 64,
        "effective_construction_fingerprint": "3" * 64,
        "initial_parameters_sha256": "4" * 64,
        "final_parameters_sha256": "5" * 64,
        "final_gradient_inf_norm": 0.01,
        "normalized_status": "budget_exhausted",
        "raw_status": "iteration-limit",
        "nit": 2,
        "nfev": 3,
        "njev": 3,
        "certificate": {
            "success": False,
            "initial_stationary": False,
            "terminal_stationary": False,
            "constraints_satisfied": True,
            "outer_status": 1,
        },
    }
    assert {row["observable"] for row in endpoint["parity"]["rows"]} == {
        "final_objective",
        "final_iota",
        "final_volume",
        "final_non_qs_ratio",
        "final_boozer_residual",
    }

    synthetic_repo = tmp_path / "frozen-repo"
    synthetic_repo.mkdir()
    _synthetic_frozen_repo(synthetic_repo)
    validator = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "validate_single_stage_speed_claim.py"
    )
    completed = subprocess.run(
        (
            sys.executable,
            str(validator),
            "--artifact-root",
            str(artifact_root),
            "--repo-root",
            str(synthetic_repo),
            "--baseline-tag",
            "synthetic-baseline",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "VERDICT: LOSS" in completed.stdout
    assert "INTEGRITY ERROR" not in completed.stdout


def test_writer_rejects_an_incomplete_schedule_before_creating_artifacts(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    incomplete_native = LaneReceipt(
        lane_id="native_cpu",
        samples=receipt.lanes[0].samples[:-1],
        endpoint=receipt.lanes[0].endpoint,
    )
    incomplete_receipt = CampaignReceipt(
        metadata=receipt.metadata,
        lanes=(incomplete_native, *receipt.lanes[1:]),
    )
    artifact_root = tmp_path / "incomplete"

    with pytest.raises(ValueError, match="ordered 1 cold, 1 warmup, and 7 warm"):
        write_campaign_receipt(artifact_root, incomplete_receipt)

    assert not artifact_root.exists()


def test_writer_rejects_a_trajectory_that_skips_an_optimizer_iteration(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    first_sample = replace(
        receipt.lanes[0].samples[0],
        trajectory=(
            TrajectoryPoint(1, 2.0, 0.005),
            TrajectoryPoint(3, 1.0, 0.01),
        ),
    )
    native_lane = replace(
        receipt.lanes[0], samples=(first_sample, *receipt.lanes[0].samples[1:])
    )
    noncontiguous_receipt = replace(receipt, lanes=(native_lane, *receipt.lanes[1:]))
    artifact_root = tmp_path / "noncontiguous"

    with pytest.raises(ValueError, match="iterations must be contiguous from 1"):
        write_campaign_receipt(artifact_root, noncontiguous_receipt)

    assert not artifact_root.exists()


def test_writer_rejects_a_non_native_parity_row_outside_its_tolerance(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    out_of_tolerance_endpoint = replace(
        receipt.lanes[1].endpoint,
        parity_rows=(
            ParityRow("final_objective", 1.0, 1.01, 0.001),
            *_parity_rows()[1:],
        ),
    )
    custom_lane = replace(receipt.lanes[1], endpoint=out_of_tolerance_endpoint)
    parity_failure_receipt = replace(
        receipt, lanes=(receipt.lanes[0], custom_lane, *receipt.lanes[2:])
    )
    artifact_root = tmp_path / "parity-failure"

    with pytest.raises(
        ValueError, match="parity row final_objective exceeds tolerance"
    ):
        write_campaign_receipt(artifact_root, parity_failure_receipt)

    assert not artifact_root.exists()


def test_writer_rejects_a_reordered_measurement_schedule(tmp_path: Path) -> None:
    receipt = _receipt()
    reordered_native = replace(
        receipt.lanes[0],
        samples=(
            receipt.lanes[0].samples[1],
            receipt.lanes[0].samples[0],
            *receipt.lanes[0].samples[2:],
        ),
    )
    reordered_receipt = replace(receipt, lanes=(reordered_native, *receipt.lanes[1:]))

    with pytest.raises(ValueError, match="ordered 1 cold, 1 warmup, and 7 warm"):
        write_campaign_receipt(tmp_path / "reordered", reordered_receipt)


def test_writer_rejects_missing_required_nonnative_parity_observable(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    incomplete_endpoint = replace(
        receipt.lanes[1].endpoint, parity_rows=_parity_rows()[:-1]
    )
    incomplete_lane = replace(receipt.lanes[1], endpoint=incomplete_endpoint)
    incomplete_receipt = replace(
        receipt, lanes=(receipt.lanes[0], incomplete_lane, *receipt.lanes[2:])
    )

    with pytest.raises(ValueError, match="parity rows must contain exactly"):
        write_campaign_receipt(tmp_path / "missing-parity", incomplete_receipt)


def test_writer_rejects_whitespace_metadata_and_non_utc_timestamps(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    whitespace_metadata = replace(receipt.metadata, hostname="   ")
    with pytest.raises(ValueError, match="hostname must be non-empty"):
        write_campaign_receipt(
            tmp_path / "whitespace", replace(receipt, metadata=whitespace_metadata)
        )

    non_utc_metadata = replace(
        receipt.metadata, created_utc="2026-08-04T00:00:00+01:00"
    )
    with pytest.raises(ValueError, match="created_utc must be a UTC ISO timestamp"):
        write_campaign_receipt(
            tmp_path / "non-utc", replace(receipt, metadata=non_utc_metadata)
        )


def test_writer_rejects_claim_bearing_receipts_under_tmp(tmp_path: Path) -> None:
    claim_receipt = replace(
        _receipt(),
        metadata=replace(
            _receipt().metadata, campaign_id="single-stage-speed-20260804"
        ),
    )
    artifact_root = Path("/tmp") / f"single-stage-speed-writer-{tmp_path.name}"

    with pytest.raises(
        ValueError,
        match="claim-bearing campaign receipts must not be written under /tmp",
    ):
        write_campaign_receipt(artifact_root, claim_receipt)

    assert not artifact_root.exists()


def test_writer_rejects_a_trajectory_that_exceeds_sample_wall_time(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    cold_sample = replace(
        receipt.lanes[0].samples[0],
        trajectory=(
            TrajectoryPoint(1, 2.0, 0.005),
            TrajectoryPoint(2, 1.0, 0.02),
        ),
    )
    native_lane = replace(
        receipt.lanes[0], samples=(cold_sample, *receipt.lanes[0].samples[1:])
    )
    elapsed_receipt = replace(receipt, lanes=(native_lane, *receipt.lanes[1:]))

    with pytest.raises(ValueError, match="trajectory exceeds sample wall_seconds"):
        write_campaign_receipt(tmp_path / "elapsed", elapsed_receipt)


def test_writer_rejects_invalid_endpoint_audit_hashes_and_counters(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    invalid_hash_endpoint = replace(
        receipt.lanes[0].endpoint,
        audit=replace(receipt.lanes[0].endpoint.audit, final_parameters_sha256="bad"),
    )
    invalid_hash_lane = replace(receipt.lanes[0], endpoint=invalid_hash_endpoint)
    invalid_hash_receipt = replace(
        receipt, lanes=(invalid_hash_lane, *receipt.lanes[1:])
    )
    with pytest.raises(ValueError, match="final_parameters_sha256"):
        write_campaign_receipt(tmp_path / "invalid-hash", invalid_hash_receipt)

    invalid_counter_endpoint = replace(
        receipt.lanes[0].endpoint,
        audit=replace(receipt.lanes[0].endpoint.audit, nfev=-1),
    )
    invalid_counter_lane = replace(receipt.lanes[0], endpoint=invalid_counter_endpoint)
    invalid_counter_receipt = replace(
        receipt, lanes=(invalid_counter_lane, *receipt.lanes[1:])
    )
    with pytest.raises(ValueError, match="nfev must be a nonnegative integer"):
        write_campaign_receipt(tmp_path / "invalid-counter", invalid_counter_receipt)


def test_writer_rejects_forged_endpoint_driver_and_mismatched_initial_state(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    forged_endpoint = replace(
        receipt.lanes[2].endpoint,
        audit=replace(receipt.lanes[2].endpoint.audit, driver=JAX_FAST_DRIVER_ID),
    )
    forged_lane = replace(receipt.lanes[2], endpoint=forged_endpoint)
    forged_receipt = replace(
        receipt,
        lanes=(receipt.lanes[0], receipt.lanes[1], forged_lane, receipt.lanes[3]),
    )
    with pytest.raises(ValueError, match="backend/driver identity"):
        write_campaign_receipt(tmp_path / "forged-driver", forged_receipt)

    missing_route_endpoint = replace(
        receipt.lanes[2].endpoint,
        audit=replace(receipt.lanes[2].endpoint.audit, adjoint_route=None),
    )
    missing_route_lane = replace(receipt.lanes[2], endpoint=missing_route_endpoint)
    missing_route_receipt = replace(
        receipt,
        lanes=(
            receipt.lanes[0],
            receipt.lanes[1],
            missing_route_lane,
            receipt.lanes[3],
        ),
    )
    with pytest.raises(ValueError, match="endpoint adjoint route"):
        write_campaign_receipt(
            tmp_path / "missing-adjoint-route", missing_route_receipt
        )

    contaminated_native_endpoint = replace(
        receipt.lanes[0].endpoint,
        audit=replace(
            receipt.lanes[0].endpoint.audit,
            adjoint_route=DIRECT_FP64_LU_EXACT_ADJOINT_ROUTE,
        ),
    )
    contaminated_native_lane = replace(
        receipt.lanes[0], endpoint=contaminated_native_endpoint
    )
    contaminated_native_receipt = replace(
        receipt,
        lanes=(contaminated_native_lane, *receipt.lanes[1:]),
    )
    with pytest.raises(ValueError, match="endpoint adjoint route"):
        write_campaign_receipt(
            tmp_path / "contaminated-native-route", contaminated_native_receipt
        )

    mismatched_endpoint = replace(
        receipt.lanes[1].endpoint,
        audit=replace(
            receipt.lanes[1].endpoint.audit,
            initial_parameters_sha256="6" * 64,
        ),
    )
    mismatched_lane = replace(receipt.lanes[1], endpoint=mismatched_endpoint)
    mismatched_receipt = replace(
        receipt,
        lanes=(receipt.lanes[0], mismatched_lane, *receipt.lanes[2:]),
    )
    with pytest.raises(ValueError, match="initial_parameters_sha256"):
        write_campaign_receipt(tmp_path / "mismatched-initial", mismatched_receipt)


def test_writer_rejects_unbound_parity_values_and_contradictory_certificate(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    unbound_endpoint = replace(
        receipt.lanes[1].endpoint,
        parity_rows=(
            receipt.lanes[1].endpoint.parity_rows[0],
            ParityRow("final_iota", 0.25, 0.25, 0.0),
            *receipt.lanes[1].endpoint.parity_rows[2:],
        ),
    )
    unbound_lane = replace(receipt.lanes[1], endpoint=unbound_endpoint)
    unbound_receipt = replace(
        receipt, lanes=(receipt.lanes[0], unbound_lane, *receipt.lanes[2:])
    )
    with pytest.raises(ValueError, match="final_iota is not endpoint-bound"):
        write_campaign_receipt(tmp_path / "unbound-parity", unbound_receipt)

    contradictory_endpoint = replace(
        receipt.lanes[0].endpoint,
        audit=replace(
            receipt.lanes[0].endpoint.audit,
            certificate=replace(
                receipt.lanes[0].endpoint.audit.certificate,
                success=True,
            ),
        ),
    )
    contradictory_lane = replace(receipt.lanes[0], endpoint=contradictory_endpoint)
    contradictory_receipt = replace(
        receipt, lanes=(contradictory_lane, *receipt.lanes[1:])
    )
    with pytest.raises(ValueError, match="contradicts normalized_status"):
        write_campaign_receipt(
            tmp_path / "contradictory-certificate", contradictory_receipt
        )


def test_writer_never_publishes_a_partial_final_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_write_json = receipt_writer._write_json

    def fail_endpoint_write(path: Path, document: object) -> None:
        if path.name == "endpoint.json":
            raise OSError("forced endpoint write failure")
        original_write_json(path, document)

    monkeypatch.setattr(receipt_writer, "_write_json", fail_endpoint_write)
    artifact_root = tmp_path / "atomic"

    with pytest.raises(OSError, match="forced endpoint write failure"):
        write_campaign_receipt(artifact_root, _receipt())

    assert not artifact_root.exists()
    assert not tuple(tmp_path.glob(".atomic.staging-*"))
