from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks import (
    run_single_stage_fullspace_gauss_newton_trust_region as runner,
)
from benchmarks.single_stage_fullspace_gntr_receipt import (
    EQUALITY_SIZE,
    GPU_UUID,
    OBJECTIVE_RESIDUAL_SIZE,
    ROUTE,
    STATE_SIZE,
)
from simsopt_jax.geo.optimizers.projected_gauss_newton_trust_region import (
    ProjectedGaussNewtonAttemptOutcome,
)
from simsopt_jax.geo.optimizers.projected_hvp_trust_region import (
    ProjectedSteihaugTermination,
)


def _history() -> SimpleNamespace:
    floating_fields = (
        "current_objective",
        "current_feasibility_inf",
        "current_stationarity_inf",
        "candidate_objective",
        "candidate_feasibility_inf",
        "actual_reduction",
        "predicted_reduction",
        "reduction_ratio",
        "trust_radius",
        "next_trust_radius",
        "tangent_step_norm",
        "correction_norm",
        "applied_step_norm",
        "correction_step_ratio",
        "corrected_radius_ratio",
        "terminal_normalized_curvature",
        "residual_value_defect",
        "residual_gradient_defect",
        "hvp_symmetry_defect",
        "probe_normalized_curvature",
        "direction_rotation",
        "correction_relative_residual",
        "correction_forward_error_bound",
        "trial_gram_factorization_relative_residual",
        "trial_gram_solve_relative_residual",
        "current_projection_tangency_relative_residual",
        "current_projection_solve_relative_residual",
        "current_projection_forward_error_bound",
        "steihaug_tangency_relative_residual",
        "steihaug_final_projected_residual_norm",
        "steihaug_projected_residual_target",
        "steihaug_residual_projection_tangency_relative_residual",
        "steihaug_residual_projection_solve_relative_residual",
        "steihaug_residual_projection_forward_error_bound",
    )
    values: dict[str, object] = {
        "outcome": np.asarray(
            [
                int(ProjectedGaussNewtonAttemptOutcome.ACCEPTED),
                int(ProjectedGaussNewtonAttemptOutcome.ACCEPTED),
            ]
        ),
        "accepted_step_number": np.asarray([1, 2]),
        "steihaug_iterations": np.asarray([1, 2]),
        "steihaug_hvp_evaluations": np.asarray([1, 2]),
        "steihaug_termination": np.asarray(
            [
                int(ProjectedSteihaugTermination.TRUST_BOUNDARY),
                int(ProjectedSteihaugTermination.INTERIOR_CONVERGED),
            ]
        ),
        "steihaug_hit_boundary": np.asarray([True, False]),
    }
    values.update(
        {field: np.asarray([1.0, 0.5], dtype=np.float64) for field in floating_fields}
    )
    values["candidate_objective"] = np.asarray([0.8, 0.6])
    values["candidate_feasibility_inf"] = np.asarray([1.0e-12, 2.0e-12])
    values["current_stationarity_inf"] = np.asarray([1.0, 0.7])
    return SimpleNamespace(**values)


def test_frozen_identity_and_source_scope() -> None:
    assert ROUTE == "CFS-GNTR1"
    assert GPU_UUID == "GPU-7951f78e-c05d-e01c-303f-d644f4341fe1"
    assert (STATE_SIZE, EQUALITY_SIZE, OBJECTIVE_RESIDUAL_SIZE) == (716, 255, 2110)
    assert (
        Path("src/simsopt_jax/geo/optimizers/projected_gauss_newton_trust_region.py")
        in runner.SOURCE_PATHS
    )
    assert (
        Path("src/simsopt_jax/solve/fullspace_gauss_newton_trust_region.py")
        in runner.SOURCE_PATHS
    )
    assert Path("benchmarks/single_stage_fullspace_gntr_receipt.py") in (
        runner.SOURCE_PATHS
    )


def test_source_manifest_covers_untracked_route_and_excludes_dotenv() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths = runner._tracked_paths(repo_root)

    assert set(runner.SOURCE_PATHS).issubset(paths)
    assert all(
        path.name == ".env.example"
        or (path.name != ".env" and not path.name.startswith(".env."))
        for path in paths
    )


def test_attempt_and_accepted_state_serialization_preserves_scaled_trend() -> None:
    optimizer = SimpleNamespace(
        attempts=np.asarray(2),
        history=_history(),
        scaled_stationarity_inf=np.asarray(0.4),
    )
    result = SimpleNamespace(
        optimizer_result=optimizer,
        initial_endpoint=SimpleNamespace(
            physical_objective=np.asarray(9.0),
            scaled_constraint_infinity_norm=np.asarray(9.0),
        ),
    )

    attempts = runner._attempt_payloads(result)
    states = runner._accepted_states_payload(result, attempts)

    assert [item["outcome"] for item in attempts] == ["ACCEPTED", "ACCEPTED"]
    assert [item["accepted_step"] for item in states] == [0, 1, 2]
    assert states[0]["physical_objective"] == attempts[0]["current_objective"]
    assert states[0]["scaled_feasibility_inf"] == attempts[0]["current_feasibility_inf"]
    assert [item["scaled_stationarity_inf"] for item in states] == [1.0, 0.7, 0.4]


def test_canonical_worker_protocol_rejects_nonfinite_json() -> None:
    payload = {"schema_version": runner.WORKER_SCHEMA_VERSION, "value": None}
    encoded = runner.canonical_json_bytes(payload)

    assert json.loads(encoded) == payload
    with pytest.raises(ValueError, match="Out of range float values"):
        runner.canonical_json_bytes({"value": float("nan")})


def test_main_dispatches_supervisor_without_gpu_in_unit_test(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    expected = {"terminal_status": "CANARY_NOT_USABLE"}
    monkeypatch.setattr(runner, "run", lambda output: expected)

    assert runner.main(["--output", str(tmp_path / "artifact")]) == 0
    assert json.loads(capsysbinary.readouterr().out) == expected


def test_worker_launch_uses_importable_benchmarks_module_from_repo_root() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    argv = runner._worker_argv()

    assert argv == (
        sys.executable,
        "-m",
        "benchmarks.run_single_stage_fullspace_gauss_newton_trust_region",
        "--worker",
    )
    completed = subprocess.run(
        (*argv, "--help"),
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--worker" in completed.stdout
