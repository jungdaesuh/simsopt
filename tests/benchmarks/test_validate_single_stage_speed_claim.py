"""Verdict-path and tamper-class tests for the frozen campaign validator."""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / "benchmarks" / "validate_single_stage_speed_claim.py"
BUDGET = 50

NATIVE_FINAL = 4.5e-8
OBSERVABLES = {
    "final_objective": NATIVE_FINAL,
    "final_iota": -0.406,
    "final_volume": -0.290,
    "final_non_qs_ratio": 4.4e-8,
    "final_boozer_residual": 2.0e-29,
}
IDENTITIES = {
    "native_cpu": ("native_cpu", "simsopt_scipy_bfgs_with_boozer_newton"),
    "jax_gpu_custom": (
        "jax_gpu_fast",
        "simsopt_jax_host_lbfgsb_with_traceable_boozer_newton",
    ),
    "jax_gpu_optax": (
        "jax_gpu_fast",
        "simsopt_jax_optax_lbfgs_with_traceable_boozer_newton",
    ),
}
LANE_SPECS = {
    "native_cpu": {"final": NATIVE_FINAL, "reach_iteration": 40, "wall_total": 100.0},
    "jax_gpu_custom": {
        "final": NATIVE_FINAL + 1e-12,
        "reach_iteration": 20,
        "wall_total": 50.0,
    },
    "jax_gpu_optax": {
        "final": NATIVE_FINAL + 1.5e-12,
        "reach_iteration": 35,
        "wall_total": 80.0,
    },
}


def _trajectory_lines(final, reach_iteration, wall_total):
    lines = []
    for iteration in range(1, BUDGET + 1):
        if iteration < reach_iteration:
            fraction = iteration / reach_iteration
            objective = 1e-4 * (1.0 - fraction) + final * fraction * 1.01
        else:
            objective = final
        wall = wall_total * iteration / BUDGET
        lines.append(
            json.dumps(
                {
                    "iteration": iteration,
                    "objective": objective,
                    "wall_seconds_from_start": wall,
                }
            )
        )
    return "\n".join(lines) + "\n"


def build_tree(root: Path, custom_wall_total: float | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "campaign.json").write_text(
        json.dumps(
            {
                "campaign_id": "validator-test",
                "git_describe": "test-0",
                "hostname": "testhost",
                "device_name": "synthetic",
                "python_version": "3.11",
                "jax_version": "0.10",
                "iteration_budget": BUDGET,
                "scale": "native_default",
                "created_utc": "2026-08-05T00:00:00Z",
            }
        )
    )
    (root / "lanes" / "jax_cpu_custom").mkdir(parents=True, exist_ok=True)
    for lane_id, spec in LANE_SPECS.items():
        wall_total = spec["wall_total"]
        if lane_id == "jax_gpu_custom" and custom_wall_total is not None:
            wall_total = custom_wall_total
        lane_dir = root / "lanes" / lane_id
        lane_dir.mkdir(parents=True, exist_ok=True)
        samples = [
            {"phase": "cold", "sample_index": 0, "wall_seconds": wall_total * 3},
            {"phase": "warmup", "sample_index": 0, "wall_seconds": wall_total * 1.5},
        ]
        trajectory = _trajectory_lines(
            spec["final"], spec["reach_iteration"], wall_total
        )
        for index in range(7):
            samples.append(
                {"phase": "warm", "sample_index": index, "wall_seconds": wall_total}
            )
            (lane_dir / f"trajectory-warm-{index}.jsonl").write_text(trajectory)
        (lane_dir / "measurement.json").write_text(json.dumps({"samples": samples}))
        backend_mode, driver = IDENTITIES[lane_id]
        audit = {
            "backend_mode": backend_mode,
            "driver": driver,
            "initial_parameters_sha256": "a" * 64,
            "input_fingerprint": "b" * 64,
            "configuration_fingerprint": "c" * 64,
            "effective_construction_fingerprint": "d" * 64,
        }
        if lane_id != "native_cpu":
            audit["adjoint_route"] = "direct-fp64-lu"
        endpoint = {
            "backend_mode": backend_mode,
            "precision": "fp64",
            "observables": {
                **{
                    name: (value if lane_id == "native_cpu" else value)
                    for name, value in OBSERVABLES.items()
                },
                "final_objective": spec["final"],
                "inner_solver_success": True,
            },
            "audit": audit,
        }
        if lane_id != "native_cpu":
            rows = []
            for name, native_value in OBSERVABLES.items():
                native_row_value = (
                    NATIVE_FINAL if name == "final_objective" else native_value
                )
                lane_value = (
                    spec["final"] if name == "final_objective" else native_value
                )
                rows.append(
                    {
                        "observable": name,
                        "native_value": native_row_value,
                        "lane_value": lane_value,
                        "tolerance": 2e-12 + 2e-8 * abs(native_row_value),
                    }
                )
            endpoint["parity"] = {"rows": rows}
        else:
            endpoint["observables"]["final_objective"] = NATIVE_FINAL
        (lane_dir / "endpoint.json").write_text(json.dumps(endpoint))


def run_validator(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), "--artifact-root", str(root)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def test_win_exits_zero(tmp_path):
    root = tmp_path / "receipts"
    build_tree(root)
    result = run_validator(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VERDICT: WIN" in result.stdout


def test_loss_without_profile_evidence(tmp_path):
    root = tmp_path / "receipts"
    build_tree(root, custom_wall_total=99.0)
    result = run_validator(root)
    assert result.returncode == 1, result.stdout
    assert "VERDICT: LOSS" in result.stdout


def test_tie_with_sibling_profile_evidence(tmp_path):
    root = tmp_path / "receipts"
    build_tree(root, custom_wall_total=99.0)
    profile_dir = tmp_path / "receipts.profile"
    profile_dir.mkdir()
    (profile_dir / "nsys-summary.txt").write_text("dominant cost: inner newton\n")
    result = run_validator(root)
    assert result.returncode == 3, result.stdout
    assert "VERDICT: TIE" in result.stdout


def test_faster_than_native_inside_margin_is_tie_eligible(tmp_path):
    """Ratio in (0.90, 1.0) — faster but short of margin — must not grade
    worse than a dead-even ratio (the r5 dead-zone repair)."""
    root = tmp_path / "receipts"
    build_tree(root, custom_wall_total=93.0)
    profile_dir = tmp_path / "receipts.profile"
    profile_dir.mkdir()
    (profile_dir / "profile.txt").write_text("evidence\n")
    result = run_validator(root)
    assert result.returncode == 3, result.stdout


def test_tie_denied_above_band(tmp_path):
    root = tmp_path / "receipts"
    build_tree(root, custom_wall_total=200.0)
    profile_dir = tmp_path / "receipts.profile"
    profile_dir.mkdir()
    (profile_dir / "profile.txt").write_text("evidence\n")
    result = run_validator(root)
    assert result.returncode == 1, result.stdout


def test_endpoint_trajectory_tamper_is_integrity_error(tmp_path):
    root = tmp_path / "receipts"
    build_tree(root)
    endpoint_path = root / "lanes" / "jax_gpu_custom" / "endpoint.json"
    payload = json.loads(endpoint_path.read_text())
    payload["observables"]["final_objective"] = 9.9e-9
    endpoint_path.write_text(json.dumps(payload))
    result = run_validator(root)
    assert result.returncode == 2, result.stdout


def test_forged_driver_is_integrity_error(tmp_path):
    root = tmp_path / "receipts"
    build_tree(root)
    endpoint_path = root / "lanes" / "jax_gpu_optax" / "endpoint.json"
    payload = json.loads(endpoint_path.read_text())
    payload["audit"]["driver"] = IDENTITIES["jax_gpu_custom"][1]
    endpoint_path.write_text(json.dumps(payload))
    result = run_validator(root)
    assert result.returncode == 2, result.stdout


def test_mismatched_initial_point_is_integrity_error(tmp_path):
    root = tmp_path / "receipts"
    build_tree(root)
    endpoint_path = root / "lanes" / "jax_gpu_custom" / "endpoint.json"
    payload = json.loads(endpoint_path.read_text())
    payload["audit"]["initial_parameters_sha256"] = "e" * 64
    endpoint_path.write_text(json.dumps(payload))
    result = run_validator(root)
    assert result.returncode == 2, result.stdout


def test_inflated_receipt_tolerance_cannot_hide_drift(tmp_path):
    """A drifted lane value with a widened receipt tolerance must still fail:
    the bound is recomputed from frozen constants, not read from the row."""
    root = tmp_path / "receipts"
    build_tree(root)
    endpoint_path = root / "lanes" / "jax_gpu_custom" / "endpoint.json"
    payload = json.loads(endpoint_path.read_text())
    for row in payload["parity"]["rows"]:
        if row["observable"] == "final_iota":
            row["lane_value"] = row["native_value"] + 1e-3
            row["tolerance"] = 1.0
    endpoint_path.write_text(json.dumps(payload))
    result = run_validator(root)
    assert result.returncode == 2, result.stdout
    assert "recomputed from the frozen tolerance constants" in result.stdout


def test_malformed_json_is_integrity_error_not_loss(tmp_path):
    root = tmp_path / "receipts"
    build_tree(root)
    (root / "lanes" / "native_cpu" / "measurement.json").write_text("{not json")
    result = run_validator(root)
    assert result.returncode == 2, result.stdout


def test_short_trajectory_is_integrity_error(tmp_path):
    root = tmp_path / "receipts"
    build_tree(root)
    trajectory_path = root / "lanes" / "jax_gpu_custom" / "trajectory-warm-3.jsonl"
    lines = trajectory_path.read_text().splitlines()
    trajectory_path.write_text("\n".join(lines[: BUDGET - 5]) + "\n")
    result = run_validator(root)
    assert result.returncode == 2, result.stdout


def test_missing_reference_lane_directory_is_integrity_error(tmp_path):
    root = tmp_path / "receipts"
    build_tree(root)
    (root / "lanes" / "jax_cpu_custom").rmdir()
    result = run_validator(root)
    assert result.returncode == 2, result.stdout
