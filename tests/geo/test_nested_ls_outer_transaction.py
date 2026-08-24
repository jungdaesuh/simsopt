"""Transactional outer-state regressions for the nested Boozer-LS lanes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import cast

import numpy as np
import pytest
from benchmarks import nested_ls_outer_jax_child as jax_child
from benchmarks import nested_ls_outer_native_child as native_child
from benchmarks.nested_ls_outer_claim import (
    CLAIM_SCHEMA,
    SWEEP_REPEATS,
    SWEEP_SCHEMA,
    _launch_rejudge,
    _parse_j_parity_rtol,
    _require_b3_green,
    _require_child_schema,
    _require_omp_evidence,
    _rejudge_endpoint_mismatch,
)
from scipy.optimize import minimize
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_OUTER_FTOL_STALL_MESSAGE,
    NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
    NESTED_LS_OUTER_REJUDGE_SCHEMA,
    NestedLsOuterCandidateStore,
    nested_ls_outer_endpoint_success,
    nested_ls_outer_ftol_zero_stop,
    nested_ls_outer_rejection_barrier,
    nested_ls_outer_restart_reason,
)


@dataclass(frozen=True, slots=True)
class _Candidate:
    name: str


@pytest.mark.parametrize(
    ("script", "source_probe"),
    (
        (
            "benchmarks/nested_ls_outer_claim.py",
            "from benchmarks import nested_ls_outer_claim as entrypoint",
        ),
        (
            "benchmarks/nested_ls_outer_jax_child.py",
            "from benchmarks import nested_ls_outer_jax_child as entrypoint",
        ),
        (
            "benchmarks/nested_ls_outer_native_child.py",
            "from benchmarks import nested_ls_outer_native_child as entrypoint",
        ),
    ),
)
def test_outer_entrypoint_bootstraps_this_worktree_before_local_imports(
    script: str,
    source_probe: str,
):
    repo = Path(__file__).resolve().parents[2]
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    completed = subprocess.run(
        [sys.executable, script, "--help"],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
    identity = subprocess.run(
        [
            sys.executable,
            "-c",
            source_probe
            + "; from simsopt_jax_adapters.geo import nested_ls_contract"
            + "; from simsopt_jax_adapters.geo import nested_ls_reduced_scale"
            + "; print(nested_ls_contract.__file__)"
            + "; print(nested_ls_reduced_scale.__file__)",
        ],
        cwd=repo,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert identity.returncode == 0, identity.stderr
    source_paths = [Path(line).resolve() for line in identity.stdout.splitlines()]
    assert len(source_paths) == 2
    assert all(path.is_relative_to(repo / "src") for path in source_paths)


def test_transaction_schema_refuses_historical_b3_as_a_b37_parent():
    assert CLAIM_SCHEMA == "nested-ls-outer-claim.v2"
    historical_b3 = (
        Path(__file__).resolve().parents[2]
        / "docs/receipts/evidence/nested_ls_outer_b3_20260823.json"
    )
    with pytest.raises(SystemExit, match="expected 'nested-ls-outer-claim.v2'"):
        _require_b3_green(
            b3_receipt=historical_b3,
            omp_num_threads=14,
            expected_git_head="current-head",
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )


def test_b3_interlock_refuses_same_schema_from_another_source(tmp_path: Path):
    receipt = tmp_path / "b3.json"
    receipt.write_text(
        json.dumps({"schema": CLAIM_SCHEMA, "git_head": "unrelated-head"}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="git_head"):
        _require_b3_green(
            b3_receipt=receipt,
            omp_num_threads=14,
            expected_git_head="current-head",
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )


def _omp_payload(git_head: str) -> dict[str, object]:
    omp_set = (4, 8)
    rows = [
        {
            "omp_num_threads": omp,
            "observed_omp_num_threads": omp,
            "omp_pinned": True,
            "repeat": repeat,
            "success": True,
            "child_schema": "nested-ls-outer-native-child.v3",
            "process_wall_seconds": float(20 - omp + repeat),
        }
        for repeat in range(SWEEP_REPEATS)
        for omp in omp_set
    ]
    return {
        "schema": SWEEP_SCHEMA,
        "git_head": git_head,
        "budget": 3,
        "maxcor": 10,
        "aggregation": "min",
        "repeats": SWEEP_REPEATS,
        "omp_set": list(omp_set),
        "rows": rows,
        "best_omp_num_threads": 8,
        "per_omp_min_process_wall_seconds": {
            str(omp): min(
                float(row["process_wall_seconds"])
                for row in rows
                if int(row["omp_num_threads"]) == omp
            )
            for omp in omp_set
        },
    }


def test_omp_interlock_recomputes_complete_sweep_and_binds_source(tmp_path: Path):
    expected_head = "transactional-source"
    artifact = tmp_path / "omp.json"
    artifact.write_text(json.dumps(_omp_payload(expected_head)), encoding="utf-8")

    accepted = _require_omp_evidence(
        omp_evidence=artifact,
        omp_num_threads=8,
        expected_git_head=expected_head,
        expected_maxcor=10,
        expected_omp_set=(4, 8),
    )
    assert accepted["rows"] == 2 * SWEEP_REPEATS

    with pytest.raises(SystemExit, match="git_head"):
        _require_omp_evidence(
            omp_evidence=artifact,
            omp_num_threads=8,
            expected_git_head="other-source",
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )

    duplicate = _omp_payload(expected_head)
    duplicate_rows = duplicate["rows"]
    assert isinstance(duplicate_rows, list)
    duplicate["rows"] = [duplicate_rows[0], duplicate_rows[0]]
    artifact.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(SystemExit, match="repeats row"):
        _require_omp_evidence(
            omp_evidence=artifact,
            omp_num_threads=8,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )

    wrong_best = _omp_payload(expected_head)
    wrong_best["best_omp_num_threads"] = 4
    artifact.write_text(json.dumps(wrong_best), encoding="utf-8")
    with pytest.raises(SystemExit, match="rows recompute"):
        _require_omp_evidence(
            omp_evidence=artifact,
            omp_num_threads=4,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )

    for field, malformed, message in (
        ("success", "false", "did not succeed"),
        ("omp_pinned", 1, "was not OMP-pinned"),
    ):
        malformed_boolean = _omp_payload(expected_head)
        malformed_rows = cast(
            list[dict[str, object]],
            malformed_boolean["rows"],
        )
        malformed_rows[0][field] = malformed
        artifact.write_text(json.dumps(malformed_boolean), encoding="utf-8")
        with pytest.raises(SystemExit, match=message):
            _require_omp_evidence(
                omp_evidence=artifact,
                omp_num_threads=8,
                expected_git_head=expected_head,
                expected_maxcor=10,
                expected_omp_set=(4, 8),
            )


@pytest.mark.parametrize(
    "invalid_wall",
    [float("nan"), float("inf"), -float("inf"), 0.0, -1.0, True, "1.0"],
)
def test_omp_interlock_refuses_nonfinite_or_nonpositive_walls(
    tmp_path: Path,
    invalid_wall: object,
):
    expected_head = "transactional-source"
    artifact = tmp_path / "omp.json"
    payload = _omp_payload(expected_head)
    rows = cast(list[dict[str, object]], payload["rows"])
    rows[0]["process_wall_seconds"] = invalid_wall
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        SystemExit,
        match="strict finite JSON|must be positive|must be a JSON number",
    ):
        _require_omp_evidence(
            omp_evidence=artifact,
            omp_num_threads=8,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )


def _claim_pair(repeat: int) -> dict[str, object]:
    policy = {"method": "L-BFGS-B", "maxiter": 3, "maxcor": 10}
    native_payload: dict[str, object] = {
        "schema": "nested-ls-outer-native-child.v3",
        "budget": 3,
        "maxcor": 10,
        "omp_num_threads": 8,
        "omp_pinned": True,
        "threading": {"OMP_PROC_BIND": "spread", "OMP_PLACES": "cores"},
        "success": True,
        "nit": 3,
        "nfev": 4,
        "restart_count": 0,
        "endpoint_is_optimizer_x": True,
        "outer_policy": policy,
        "start": {"coil_dofs": [0.0]},
        "endpoint": {
            "objective": 1.0,
            "iota": 0.14,
            "G": 2.0,
            "gradient_l2": 1.0e-14,
            "coil_sha256": f"native-coils-{repeat}",
            "surface_sha256": f"native-surface-{repeat}",
            "coil_dofs": [1.0],
        },
    }
    native_raw = json.dumps(native_payload, allow_nan=False, indent=2) + "\n"
    native_hash = hashlib.sha256(native_raw.encode("utf-8")).hexdigest()
    native: dict[str, object] = {
        "role": "measure",
        "repeat": repeat,
        "timed": True,
        "success": True,
        "omp_pinned": True,
        "observed_omp_num_threads": 8,
        "omp_proc_bind": "spread",
        "omp_places": "cores",
        "nit": 3,
        "nfev": 4,
        "restart_count": 0,
        "endpoint_is_optimizer_endpoint": True,
        "outer_policy": policy,
        "start_coil_dofs": [0.0],
        "endpoint_coil_dofs": [1.0],
        "endpoint_coil_sha256": f"native-coils-{repeat}",
        "endpoint_surface_sha256": f"native-surface-{repeat}",
        "endpoint_j": 1.0,
        "endpoint_iota": 0.14,
        "endpoint_g": 2.0,
        "endpoint_gradient_l2": 1.0e-14,
        "process_wall_seconds": float(20 + repeat),
        "claim_wall_seconds": float(20 + repeat),
        "child_payload": native_payload,
        "child_payload_raw": native_raw,
        "child_payload_sha256": native_hash,
    }
    jax_payload: dict[str, object] = {
        "schema": NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
        "budget": 3,
        "maxcor": 10,
        "success": True,
        "nit": 3,
        "nfev": 4,
        "restart_count": 0,
        "endpoint_is_optimizer_x": True,
        "outer_policy": policy,
        "start_policy": "symmetric",
        "iota_branch_guard": 0.02,
        "feasible_evaluations": 4,
        "rejected_evaluations": 0,
        "endpoint_j": 1.05,
        "endpoint_grad_l2": 1.0e-14,
        "endpoint_grad_inf": 1.0e-14,
        "endpoint_iota": 0.14,
        "endpoint_g": 2.0,
        "endpoint_adjoint_live_eta": 1.0e-14,
        "endpoint_coil_sha256": f"jax-coils-{repeat}",
        "endpoint_surface_sha256": f"jax-surface-{repeat}",
        "start_coil_dofs": [0.0],
        "endpoint_coil_dofs": [1.0],
    }
    jax_raw = json.dumps(jax_payload, allow_nan=False) + "\n"
    jax_hash = hashlib.sha256(jax_raw.encode("utf-8")).hexdigest()
    jax_row: dict[str, object] = {
        "role": "measure",
        "repeat": repeat,
        "timed": True,
        "success": True,
        "nit": 3,
        "nfev": 4,
        "restart_count": 0,
        "endpoint_is_optimizer_endpoint": True,
        "outer_policy": policy,
        "start_policy": "symmetric",
        "iota_branch_guard": 0.02,
        "feasible_evaluations": 4,
        "rejected_evaluations": 0,
        "start_coil_dofs": [0.0],
        "endpoint_coil_dofs": [1.0],
        "endpoint_coil_sha256": f"jax-coils-{repeat}",
        "endpoint_surface_sha256": f"jax-surface-{repeat}",
        "endpoint_j": 1.05,
        "endpoint_grad_l2": 1.0e-14,
        "endpoint_grad_inf": 1.0e-14,
        "endpoint_iota": 0.14,
        "endpoint_g": 2.0,
        "endpoint_adjoint_live_eta": 1.0e-14,
        "process_wall_seconds": float(10 + repeat),
        "claim_wall_seconds": float(10 + repeat),
        "child_payload": jax_payload,
        "child_payload_raw": jax_raw,
        "child_payload_sha256": jax_hash,
    }

    def rejudge(
        lane: str,
        row: dict[str, object],
        source_hash: str,
    ) -> dict[str, object]:
        child_payload = row["child_payload"]
        assert isinstance(child_payload, dict)
        payload = {
            "schema": NESTED_LS_OUTER_REJUDGE_SCHEMA,
            "judged_lane": lane,
            "source_child_schema": child_payload["schema"],
            "source_child_payload_sha256": source_hash,
            "budget": 3,
            "maxcor": 10,
            "endpoint_coil_sha256": row["endpoint_coil_sha256"],
            "endpoint_surface_sha256": row["endpoint_surface_sha256"],
            "endpoint_j": row["endpoint_j"],
            "endpoint_iota": row["endpoint_iota"],
            "endpoint_g": row["endpoint_g"],
            "fail_closed_reason": None,
        }
        reencoded = json.dumps(payload, allow_nan=False, indent=2) + "\n"
        return {
            "payload": payload,
            "payload_sha256": hashlib.sha256(reencoded.encode("utf-8")).hexdigest(),
            "process_wall_seconds": 1.0,
            "timed": False,
            "repeat": repeat,
        }

    gap = (1.05 - 1.0) / 1.0
    return {
        "repeat": repeat,
        "native": native,
        "jax": jax_row,
        "native_rejudge": rejudge("native", native, native_hash),
        "jax_rejudge": rejudge("jax", jax_row, jax_hash),
        "endpoint_j_native": 1.0,
        "endpoint_j_jax": 1.05,
        "endpoint_j_rel_gap": gap,
        "physics_ok": True,
        "fail_closed_reason": None,
        "endpoint_j_rel_gap_worse_direction": max(0.0, gap),
        "endpoint_j_within_frozen_band": None,
    }


def _claim_payload(
    *,
    git_head: str,
    omp_evidence: dict[str, object],
) -> dict[str, object]:
    pairs = [_claim_pair(repeat) for repeat in range(3)]
    native_rows = [cast(dict[str, object], pair["native"]) for pair in pairs]
    jax_rows = [cast(dict[str, object], pair["jax"]) for pair in pairs]
    native_walls = [
        float(cast(float, row["claim_wall_seconds"])) for row in native_rows
    ]
    jax_walls = [float(cast(float, row["claim_wall_seconds"])) for row in jax_rows]
    native_min = min(native_walls)
    jax_min = min(jax_walls)
    measured_gap = max(
        cast(float, pair["endpoint_j_rel_gap_worse_direction"]) for pair in pairs
    )
    return {
        "schema": CLAIM_SCHEMA,
        "git_head": git_head,
        "fail_closed_reason": None,
        "pairs": pairs,
        "claim_boundary": {
            "budget": 3,
            "maxcor": 10,
            "repeats": 3,
            "native_omp_num_threads": 8,
            "omp_provenance": "swept_artifact",
            "omp_evidence": omp_evidence,
            "measured_j_rel_gap_max": measured_gap,
            "nested_speed_claim": True,
        },
        "native_min_process_wall_seconds": native_min,
        "native_median_process_wall_seconds": 21.0,
        "native_max_process_wall_seconds": 22.0,
        "jax_min_process_wall_seconds": jax_min,
        "jax_median_process_wall_seconds": 11.0,
        "jax_max_process_wall_seconds": 12.0,
        "speedup_min_over_min": native_min / jax_min,
    }


def test_b3_interlock_recomputes_pair_physics_and_aggregates(tmp_path: Path):
    expected_head = "transactional-source"
    omp_path = tmp_path / "omp.json"
    omp_path.write_text(json.dumps(_omp_payload(expected_head)), encoding="utf-8")
    omp_summary = _require_omp_evidence(
        omp_evidence=omp_path,
        omp_num_threads=8,
        expected_git_head=expected_head,
        expected_maxcor=10,
        expected_omp_set=(4, 8),
    )
    assert omp_summary["sha256"] == hashlib.sha256(omp_path.read_bytes()).hexdigest()
    receipt_path = tmp_path / "b3.json"
    payload = _claim_payload(git_head=expected_head, omp_evidence=omp_summary)
    receipt_path.write_text(json.dumps(payload), encoding="utf-8")
    accepted = _require_b3_green(
        b3_receipt=receipt_path,
        omp_num_threads=8,
        expected_git_head=expected_head,
        expected_maxcor=10,
        expected_omp_set=(4, 8),
    )
    assert accepted["pairs"] == 3

    failed_child = _claim_payload(git_head=expected_head, omp_evidence=omp_summary)
    failed_pairs = cast(list[dict[str, object]], failed_child["pairs"])
    first_pair = failed_pairs[0]
    first_native = cast(dict[str, object], first_pair["native"])
    first_native["success"] = False
    receipt_path.write_text(json.dumps(failed_child), encoding="utf-8")
    with pytest.raises(SystemExit, match="native_row_success_mismatch"):
        _require_b3_green(
            b3_receipt=receipt_path,
            omp_num_threads=8,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )

    stale_aggregate = _claim_payload(git_head=expected_head, omp_evidence=omp_summary)
    stale_aggregate["speedup_min_over_min"] = 999.0
    receipt_path.write_text(json.dumps(stale_aggregate), encoding="utf-8")
    with pytest.raises(SystemExit, match="aggregate speedup_min_over_min"):
        _require_b3_green(
            b3_receipt=receipt_path,
            omp_num_threads=8,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )

    missing_pair_field = _claim_payload(
        git_head=expected_head,
        omp_evidence=omp_summary,
    )
    missing_pairs = cast(list[dict[str, object]], missing_pair_field["pairs"])
    missing_pairs[0].pop("endpoint_j_native")
    receipt_path.write_text(json.dumps(missing_pair_field), encoding="utf-8")
    with pytest.raises(SystemExit, match="endpoint_j_native.*JSON number"):
        _require_b3_green(
            b3_receipt=receipt_path,
            omp_num_threads=8,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )

    stale_pair_field = _claim_payload(
        git_head=expected_head,
        omp_evidence=omp_summary,
    )
    stale_pairs = cast(list[dict[str, object]], stale_pair_field["pairs"])
    stale_pairs[0]["endpoint_j_rel_gap"] = 999.0
    receipt_path.write_text(json.dumps(stale_pair_field), encoding="utf-8")
    with pytest.raises(SystemExit, match="stale endpoint-J fields"):
        _require_b3_green(
            b3_receipt=receipt_path,
            omp_num_threads=8,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )

    malformed_verdict = _claim_payload(
        git_head=expected_head,
        omp_evidence=omp_summary,
    )
    malformed_boundary = cast(dict[str, object], malformed_verdict["claim_boundary"])
    malformed_boundary["nested_speed_claim"] = "false"
    receipt_path.write_text(json.dumps(malformed_verdict), encoding="utf-8")
    with pytest.raises(SystemExit, match="nested_speed_claim is stale"):
        _require_b3_green(
            b3_receipt=receipt_path,
            omp_num_threads=8,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )


@pytest.mark.parametrize(
    "invalid_wall",
    [float("nan"), float("inf"), -float("inf"), 0.0, -1.0, True, "1.0"],
)
def test_b3_interlock_refuses_nonfinite_or_nonpositive_walls(
    tmp_path: Path,
    invalid_wall: object,
):
    expected_head = "transactional-source"
    omp_path = tmp_path / "omp.json"
    omp_path.write_text(json.dumps(_omp_payload(expected_head)), encoding="utf-8")
    omp_summary = _require_omp_evidence(
        omp_evidence=omp_path,
        omp_num_threads=8,
        expected_git_head=expected_head,
        expected_maxcor=10,
        expected_omp_set=(4, 8),
    )
    payload = _claim_payload(git_head=expected_head, omp_evidence=omp_summary)
    pairs = cast(list[dict[str, object]], payload["pairs"])
    native = cast(dict[str, object], pairs[0]["native"])
    native["claim_wall_seconds"] = invalid_wall
    receipt = tmp_path / "b3.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        SystemExit,
        match="strict finite JSON|must be positive|must be a JSON number",
    ):
        _require_b3_green(
            b3_receipt=receipt,
            omp_num_threads=8,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )


@pytest.mark.parametrize(
    "invalid_rtol", [float("nan"), float("inf"), -float("inf"), -1.0]
)
def test_j_parity_preflight_refuses_invalid_tolerances(invalid_rtol: float):
    with pytest.raises(SystemExit, match="finite|nonnegative"):
        _parse_j_parity_rtol(invalid_rtol)


def test_j_parity_preflight_accepts_zero_tolerance():
    assert _parse_j_parity_rtol(0.0) == 0.0


def test_b3_interlock_refuses_row_endpoint_tampering(tmp_path: Path):
    expected_head = "transactional-source"
    omp_path = tmp_path / "omp.json"
    omp_path.write_text(json.dumps(_omp_payload(expected_head)), encoding="utf-8")
    omp_summary = _require_omp_evidence(
        omp_evidence=omp_path,
        omp_num_threads=8,
        expected_git_head=expected_head,
        expected_maxcor=10,
        expected_omp_set=(4, 8),
    )
    payload = _claim_payload(git_head=expected_head, omp_evidence=omp_summary)
    pairs = cast(list[dict[str, object]], payload["pairs"])
    jax_row = cast(dict[str, object], pairs[0]["jax"])
    jax_row["endpoint_j"] = 0.5
    receipt = tmp_path / "b3.json"
    receipt.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="jax_row_endpoint_j_mismatch"):
        _require_b3_green(
            b3_receipt=receipt,
            omp_num_threads=8,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )

    payload = _claim_payload(git_head=expected_head, omp_evidence=omp_summary)
    pairs = cast(list[dict[str, object]], payload["pairs"])
    jax_row = cast(dict[str, object], pairs[0]["jax"])
    child_payload = cast(dict[str, object], jax_row["child_payload"])
    child_payload["endpoint_j"] = 0.5
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit, match="jax_embedded_child_payload_mismatch"):
        _require_b3_green(
            b3_receipt=receipt,
            omp_num_threads=8,
            expected_git_head=expected_head,
            expected_maxcor=10,
            expected_omp_set=(4, 8),
        )


def test_claim_parent_rejects_a_child_from_another_schema():
    _require_child_schema(
        {"schema": "nested-ls-outer-jax-child.v4"},
        label="jax",
        expected_schema="nested-ls-outer-jax-child.v4",
    )
    with pytest.raises(RuntimeError, match="outer jax child schema"):
        _require_child_schema(
            {"schema": "nested-ls-outer-jax-child.v3"},
            label="jax",
            expected_schema="nested-ls-outer-jax-child.v4",
        )


def test_claim_parent_accepts_only_the_distinct_rejudge_payload_schema():
    rejudge_payload: dict[str, object] = {
        "schema": NESTED_LS_OUTER_REJUDGE_SCHEMA,
        "mode": "rejudge",
        "judged_lane": "jax",
        "judged_schema": "nested-ls-outer-jax-child.v4",
    }
    _require_child_schema(
        rejudge_payload,
        label="rejudge",
        expected_schema=NESTED_LS_OUTER_REJUDGE_SCHEMA,
    )
    with pytest.raises(RuntimeError, match="outer rejudge child schema"):
        _require_child_schema(
            rejudge_payload,
            label="rejudge",
            expected_schema="nested-ls-outer-jax-child.v4",
        )


def test_rejudge_identity_is_bound_to_the_exact_child_payload_and_endpoint():
    row: dict[str, object] = {
        "child_payload": {"schema": NESTED_LS_OUTER_JAX_CHILD_SCHEMA},
        "child_payload_sha256": "child-bytes",
        "endpoint_coil_sha256": "coil-bytes",
        "endpoint_surface_sha256": "surface-bytes",
        "endpoint_j": 0.25,
        "endpoint_iota": 0.14,
        "endpoint_g": 2.0,
    }
    rejudge_payload: dict[str, object] = {
        "schema": NESTED_LS_OUTER_REJUDGE_SCHEMA,
        "judged_lane": "jax",
        "source_child_schema": NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
        "source_child_payload_sha256": "child-bytes",
        "budget": 3,
        "maxcor": 10,
        "endpoint_coil_sha256": "coil-bytes",
        "endpoint_surface_sha256": "surface-bytes",
        "endpoint_j": 0.25,
        "endpoint_iota": 0.14,
        "endpoint_g": 2.0,
    }
    reencoded = json.dumps(rejudge_payload, allow_nan=False, indent=2) + "\n"
    envelope: dict[str, object] = {
        "payload": rejudge_payload,
        "payload_sha256": hashlib.sha256(reencoded.encode("utf-8")).hexdigest(),
    }

    assert (
        _rejudge_endpoint_mismatch(
            lane="jax",
            row=row,
            rejudge_envelope=envelope,
            budget=3,
            maxcor=10,
        )
        is None
    )
    envelope["payload_sha256"] = "stale-rejudge-payload"
    assert (
        _rejudge_endpoint_mismatch(
            lane="jax",
            row=row,
            rejudge_envelope=envelope,
            budget=3,
            maxcor=10,
        )
        == "jax_rejudge_payload_sha256_mismatch"
    )
    envelope["payload_sha256"] = hashlib.sha256(reencoded.encode("utf-8")).hexdigest()
    rejudge_payload["source_child_payload_sha256"] = "other-child"
    reencoded = json.dumps(rejudge_payload, allow_nan=False, indent=2) + "\n"
    envelope["payload_sha256"] = hashlib.sha256(reencoded.encode("utf-8")).hexdigest()
    assert (
        _rejudge_endpoint_mismatch(
            lane="jax",
            row=row,
            rejudge_envelope=envelope,
            budget=3,
            maxcor=10,
        )
        == "jax_rejudge_source_child_payload_sha256_mismatch"
    )
    rejudge_payload["source_child_payload_sha256"] = "child-bytes"
    rejudge_payload["endpoint_coil_sha256"] = "other-coils"
    reencoded = json.dumps(rejudge_payload, allow_nan=False, indent=2) + "\n"
    envelope["payload_sha256"] = hashlib.sha256(reencoded.encode("utf-8")).hexdigest()
    assert (
        _rejudge_endpoint_mismatch(
            lane="jax",
            row=row,
            rejudge_envelope=envelope,
            budget=3,
            maxcor=10,
        )
        == "jax_rejudge_endpoint_coil_sha256_mismatch"
    )


def test_rejudge_refuses_changed_child_payload_before_launch(tmp_path: Path):
    endpoint = tmp_path / "child.json"
    endpoint.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after the timed child"):
        _launch_rejudge(
            lane="jax",
            endpoint_path=endpoint,
            budget=3,
            maxcor=10,
            expected_child_payload_sha256="not-the-file-hash",
            log=lambda _message: None,
        )


def test_endpoint_loader_preserves_declared_coil_hash_for_rejudge(tmp_path: Path):
    endpoint = tmp_path / "jax-child.json"
    endpoint.write_text(
        json.dumps(
            {
                "schema": NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
                "endpoint_coil_dofs": [1.0, 2.0],
                "endpoint_coil_sha256": "declared-coils",
                "endpoint_surface_dofs": [3.0],
                "endpoint_surface_sha256": "declared-surface",
                "endpoint_iota": 0.14,
                "endpoint_g": 2.0,
                "endpoint_j": 0.25,
            }
        ),
        encoding="utf-8",
    )
    record = jax_child._load_endpoint_record(endpoint)
    assert record.declared_coil_sha256 == "declared-coils"
    assert (
        jax_child._endpoint_declaration_mismatch(
            record,
            expected_coil_sha256="actual-coils",
            expected_surface_sha256="declared-surface",
        )
        == "endpoint_coil_declaration_mismatch"
    )


def test_candidate_store_primes_x0_and_commits_only_callback_bytes():
    start = np.array([1.0, 2.0], dtype=np.float64)
    trial = np.array([1.25, 2.0], dtype=np.float64)
    store = NestedLsOuterCandidateStore[_Candidate](start)

    with pytest.raises(RuntimeError, match="does not match x0"):
        store.record(trial, _Candidate("wrong-first-point"))

    initial = _Candidate("x0")
    pending = _Candidate("trial")
    assert store.record(start, initial) is True
    assert store.committed is initial
    assert store.record(trial, pending) is False
    assert store.committed is initial

    assert store.accept(trial.copy()) is pending
    assert store.committed is pending
    assert store.committed_matches(trial)
    with pytest.raises(RuntimeError, match="neither the incumbent nor a pending"):
        store.accept(start)


def test_rejection_barrier_has_no_offset_and_returns_its_exact_gradient():
    anchor = np.array([1.0, -2.0], dtype=np.float64)
    trial = np.array([1.5, -1.0], dtype=np.float64)
    value, gradient = nested_ls_outer_rejection_barrier(
        anchor_value=3.0,
        anchor_parameters=anchor,
        trial_parameters=trial,
        scale=2.0,
    )

    assert value == pytest.approx(4.25)
    np.testing.assert_array_equal(gradient, np.array([1.0, 2.0]))
    anchor_value, anchor_gradient = nested_ls_outer_rejection_barrier(
        anchor_value=3.0,
        anchor_parameters=anchor,
        trial_parameters=anchor,
        scale=2.0,
    )
    assert anchor_value == 3.0
    np.testing.assert_array_equal(anchor_gradient, np.zeros(2))


def test_no_offset_barrier_cannot_manufacture_ftol_zero_convergence():
    anchor = np.array([0.0], dtype=np.float64)
    evaluations = 0

    def blocked_objective(point: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        if evaluations == 1:
            return 0.0, np.array([-1.0], dtype=np.float64)
        return nested_ls_outer_rejection_barrier(
            anchor_value=0.0,
            anchor_parameters=anchor,
            trial_parameters=np.asarray(point, dtype=np.float64),
            scale=1.0,
        )

    result = minimize(
        blocked_objective,
        anchor,
        jac=True,
        method="L-BFGS-B",
        options={"ftol": 0.0, "gtol": 1.0e-12, "maxiter": 1, "maxls": 8},
    )

    assert result.status == 2
    assert str(result.message).startswith("ABNORMAL")
    assert not result.success


def test_ftol_zero_message_is_fail_closed_without_sentinel_preconditions():
    assert nested_ls_outer_ftol_zero_stop(
        ftol=0.0,
        message=NESTED_LS_OUTER_FTOL_STALL_MESSAGE,
    )
    assert (
        nested_ls_outer_restart_reason(
            ftol=0.0,
            status=0,
            message=NESTED_LS_OUTER_FTOL_STALL_MESSAGE,
        )
        == "false_ftol_stall"
    )
    assert not nested_ls_outer_ftol_zero_stop(
        ftol=1.0e-12,
        message=NESTED_LS_OUTER_FTOL_STALL_MESSAGE,
    )
    assert nested_ls_outer_ftol_zero_stop(
        ftol=0.0,
        message="warning: relative reduction of f was reported",
    )
    assert not nested_ls_outer_endpoint_success(
        endpoint_matches=True,
        ftol=0.0,
        status=0,
        message=NESTED_LS_OUTER_FTOL_STALL_MESSAGE,
    )
    assert nested_ls_outer_endpoint_success(
        endpoint_matches=True,
        ftol=0.0,
        status=1,
        message="STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT",
    )
    assert not nested_ls_outer_endpoint_success(
        endpoint_matches=True,
        ftol=0.0,
        status=2,
        message="ABNORMAL: LINE SEARCH FAILED",
    )
    assert (
        nested_ls_outer_restart_reason(
            ftol=0.0,
            status=2,
            message="ABNORMAL: LINE SEARCH FAILED",
        )
        == "abnormal_line_search"
    )


class _FakeSurface:
    def __init__(self, dofs: np.ndarray) -> None:
        self._dofs = np.array(dofs, dtype=np.float64, copy=True)

    def get_dofs(self) -> np.ndarray:
        return np.array(self._dofs, copy=True)

    def set_dofs(self, dofs: np.ndarray) -> None:
        self._dofs = np.array(dofs, dtype=np.float64, copy=True)


class _FakeBiotSavart:
    def __init__(self, coil_dofs: np.ndarray) -> None:
        self.x = np.array(coil_dofs, dtype=np.float64, copy=True)


class _FakeBoozer:
    def __init__(self, coil_dofs: np.ndarray, surface_dofs: np.ndarray) -> None:
        self.biotsavart = _FakeBiotSavart(coil_dofs)
        self.surface = _FakeSurface(surface_dofs)
        self.need_to_run_code = False


class _FakeJaxBoozer(_FakeBoozer):
    def __init__(self, coil_dofs: np.ndarray, surface_dofs: np.ndarray) -> None:
        super().__init__(coil_dofs, surface_dofs)
        self.refresh_count = 0

    def _refresh_coil_data(self) -> None:
        self.refresh_count += 1


class _FakeJaxOuterState:
    def __init__(self) -> None:
        self.anchor_surface_dofs = np.array([-1.0], dtype=np.float64)
        self.anchor_iota = -1.0
        self.anchor_G = -1.0
        self.inner_iterations = -1
        self.inner_grad_l2 = -1.0
        self.adjoint_live_eta = -1.0

    def set_anchor(self, surface_dofs: object, iota: float, G: float) -> None:
        self.anchor_surface_dofs = np.array(surface_dofs, dtype=np.float64, copy=True)
        self.anchor_iota = float(iota)
        self.anchor_G = float(G)


class _FakeObjective:
    def __init__(self, boozer: _FakeBoozer) -> None:
        self.boozer = boozer

    def evaluate(self) -> tuple[float, dict[str, float], np.ndarray]:
        coil = np.asarray(self.boozer.biotsavart.x, dtype=np.float64)
        value = float(np.dot(coil, coil))
        terms = {key: value for key in native_child.FLAT675_OBJECTIVE_TERM_KEYS}
        return value, terms, 2.0 * coil


def test_jax_candidate_restore_reinstates_all_transaction_state():
    state = _FakeJaxOuterState()
    boozer = _FakeJaxBoozer(
        np.array([9.0, 9.0], dtype=np.float64),
        np.array([9.0], dtype=np.float64),
    )
    candidate = jax_child._NestedCandidate(
        eval_index=3,
        coil_dofs=np.array([0.25, -0.5], dtype=np.float64),
        j=0.25,
        gradient=np.array([1.0, 2.0], dtype=np.float64),
        grad_l2=3.0,
        grad_inf=2.0,
        surface_dofs=np.array([-0.25], dtype=np.float64),
        iota=0.14,
        G=2.0,
        inner_iterations=4,
        inner_grad_l2=1.0e-14,
        adjoint_live_eta=2.0e-14,
    )

    jax_child._restore_nested_candidate(
        state,  # type: ignore[arg-type]
        boozer,  # type: ignore[arg-type]
        candidate,
    )

    np.testing.assert_array_equal(state.anchor_surface_dofs, candidate.surface_dofs)
    assert state.anchor_iota == candidate.iota
    assert state.anchor_G == candidate.G
    assert state.inner_iterations == candidate.inner_iterations
    assert state.inner_grad_l2 == candidate.inner_grad_l2
    assert state.adjoint_live_eta == candidate.adjoint_live_eta
    np.testing.assert_array_equal(boozer.surface.get_dofs(), candidate.surface_dofs)
    np.testing.assert_array_equal(boozer.biotsavart.x, candidate.coil_dofs)
    assert boozer.refresh_count == 1


def test_native_outer_trial_order_cannot_change_the_committed_objective(monkeypatch):
    start = np.array([0.25, -0.5], dtype=np.float64)
    bad_trial = np.array([1.25, 0.5], dtype=np.float64)
    failed_trial = np.array([9.0, 9.0], dtype=np.float64)
    boozer = _FakeBoozer(start, np.array([99.0], dtype=np.float64))
    objective = _FakeObjective(boozer)

    def fake_inner(fake_boozer, *, iota: float, G: float):
        coil = np.asarray(fake_boozer.biotsavart.x, dtype=np.float64)
        fake_boozer.surface.set_dofs(np.array([float(np.sum(coil))]))
        success = not np.array_equal(coil, failed_trial)
        return {
            "success": success,
            "bfgs_iter": 1,
            "newton_iter": 1,
            "bfgs_seconds": 0.0,
            "newton_seconds": 0.0,
            "seconds": 0.0,
            "coil_delta_inf": 0.0,
            "iota": iota,
            "G": G,
        }

    monkeypatch.setattr(
        native_child,
        "_run_native_banana_bfgs_then_newton",
        fake_inner,
    )
    run = native_child.NativeOuterRun(
        objective,  # type: ignore[arg-type]
        seed=native_child.InnerWarmStart(
            surface_dofs=np.array([99.0], dtype=np.float64),
            iota=0.14,
            G=2.0,
        ),
        rejection_distance_scale=1.0,
        start_coil_dofs=start,
    )

    first_value, first_gradient = run(start)
    anchor = run.anchor
    assert anchor is not None
    np.testing.assert_array_equal(anchor.coil_dofs, start)
    np.testing.assert_array_equal(boozer.surface.get_dofs(), np.array([-0.25]))

    bad_value, _bad_gradient = run(bad_trial)
    assert bad_value > first_value
    anchor = run.anchor
    assert anchor is not None
    np.testing.assert_array_equal(anchor.coil_dofs, start)
    np.testing.assert_array_equal(boozer.biotsavart.x, start)
    np.testing.assert_array_equal(boozer.surface.get_dofs(), np.array([-0.25]))

    replay_value, replay_gradient = run(start)
    assert replay_value == first_value
    np.testing.assert_array_equal(replay_gradient, first_gradient)
    np.testing.assert_array_equal(boozer.surface.get_dofs(), np.array([-0.25]))

    run.accept(bad_trial.copy())
    anchor = run.anchor
    assert anchor is not None
    np.testing.assert_array_equal(anchor.coil_dofs, bad_trial)
    np.testing.assert_array_equal(boozer.biotsavart.x, bad_trial)
    np.testing.assert_array_equal(boozer.surface.get_dofs(), np.array([1.75]))

    barrier_value, barrier_gradient = run(failed_trial)
    displacement = failed_trial - bad_trial
    assert barrier_value == pytest.approx(
        bad_value + 0.5 * float(np.dot(displacement, displacement))
    )
    np.testing.assert_array_equal(barrier_gradient, displacement)
    anchor = run.anchor
    assert anchor is not None
    np.testing.assert_array_equal(anchor.coil_dofs, bad_trial)
    np.testing.assert_array_equal(boozer.biotsavart.x, bad_trial)
    np.testing.assert_array_equal(boozer.surface.get_dofs(), np.array([1.75]))

    failed_trial[:] = bad_trial
    with pytest.raises(RuntimeError, match="exact committed outer point"):
        run(bad_trial)
    np.testing.assert_array_equal(boozer.biotsavart.x, bad_trial)
    np.testing.assert_array_equal(boozer.surface.get_dofs(), np.array([1.75]))
