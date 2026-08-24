"""Transactional outer-state regressions for the nested Boozer-LS lanes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Final, cast

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
import scipy
from scipy.optimize import fmin_l_bfgs_b, minimize
from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_OUTER_FTOL_STALL_MESSAGE,
    NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
    NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
    NESTED_LS_OUTER_PUBLISHABLE_STOP_STATUSES,
    NESTED_LS_OUTER_REJUDGE_SCHEMA,
    NestedLsOuterCandidateStore,
    nested_ls_outer_endpoint_success,
    nested_ls_outer_ftol_zero_stop,
    nested_ls_outer_parameter_bytes,
    nested_ls_outer_rejection_barrier,
    nested_ls_outer_restart_reason,
)


def _production_child_encoding(payload: dict[str, object]) -> str:
    """Serialize exactly as the outer children write their payload files.

    Compact separators, insertion order, one trailing newline — see
    ``benchmarks/nested_ls_outer_jax_child.py``'s ``main``. Every fixture that
    stands in for a child's bytes goes through here so no test can quietly
    adopt the parent's encoding and mask a producer/consumer disagreement.
    """

    return json.dumps(payload, allow_nan=False) + "\n"


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
            "child_schema": NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
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
        "schema": NESTED_LS_OUTER_NATIVE_CHILD_SCHEMA,
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
        # Encode the way the PRODUCTION child writes its payload — compact,
        # insertion-ordered — not the way the parent happens to re-encode.
        # A fixture that adopts the consumer's convention can never catch a
        # producer/consumer disagreement, which is how the rejudge binding
        # stayed broken end to end without a red test.
        payload_raw = _production_child_encoding(payload)
        return {
            "payload": payload,
            "payload_raw": payload_raw,
            "payload_sha256": hashlib.sha256(payload_raw.encode("utf-8")).hexdigest(),
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
    """The parent admits this campaign's child contract and refuses any other.

    Both sides come from the live constant rather than a literal. A frozen
    literal here would keep passing after a schema bump while every producer
    had moved, which is the drift the schema string exists to catch. The
    refusal case is built by mutating the live constant, so it cannot
    accidentally become the accepted one.
    """

    foreign_schema = f"{NESTED_LS_OUTER_JAX_CHILD_SCHEMA}-not-this-contract"
    assert foreign_schema != NESTED_LS_OUTER_JAX_CHILD_SCHEMA

    _require_child_schema(
        {"schema": NESTED_LS_OUTER_JAX_CHILD_SCHEMA},
        label="jax",
        expected_schema=NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
    )
    with pytest.raises(RuntimeError, match="outer jax child schema"):
        _require_child_schema(
            {"schema": foreign_schema},
            label="jax",
            expected_schema=NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
        )


def test_claim_parent_accepts_only_the_distinct_rejudge_payload_schema():
    rejudge_payload: dict[str, object] = {
        "schema": NESTED_LS_OUTER_REJUDGE_SCHEMA,
        "mode": "rejudge",
        "judged_lane": "jax",
        "judged_schema": NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
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
            expected_schema=NESTED_LS_OUTER_JAX_CHILD_SCHEMA,
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
    payload_raw = _production_child_encoding(rejudge_payload)
    envelope: dict[str, object] = {
        "payload": rejudge_payload,
        "payload_raw": payload_raw,
        "payload_sha256": hashlib.sha256(payload_raw.encode("utf-8")).hexdigest(),
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
    envelope["payload_sha256"] = hashlib.sha256(payload_raw.encode("utf-8")).hexdigest()
    rejudge_payload["source_child_payload_sha256"] = "other-child"
    payload_raw = _production_child_encoding(rejudge_payload)
    envelope["payload_raw"] = payload_raw
    envelope["payload_sha256"] = hashlib.sha256(payload_raw.encode("utf-8")).hexdigest()
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
    payload_raw = _production_child_encoding(rejudge_payload)
    envelope["payload_raw"] = payload_raw
    envelope["payload_sha256"] = hashlib.sha256(payload_raw.encode("utf-8")).hexdigest()
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


def test_rejudge_binding_survives_a_receipt_reload_that_reorders_keys():
    """The rejudge digest still matches after the receipt round-trips to disk.

    The parent writes receipts with ``dump_strict_json``, i.e. ``sort_keys=True``,
    and ``json.loads`` preserves document order — so the parsed rejudge payload
    a reload hands back is key-reordered relative to the compact,
    insertion-ordered bytes the child wrote. The binding must be immune to
    that, because the same check runs twice: live, on the envelope
    ``_launch_rejudge`` just built, and again on the reloaded receipt inside
    ``_require_b3_green``. A check that re-serializes the parsed payload passes
    in one context and fails in the other; one that re-hashes the child's
    stored bytes passes in both.

    Fails against the predecessor, which re-encoded the parsed payload with
    ``indent=2``: that could not match the child's compact bytes in EITHER
    context, so every claim.v2 pair failed closed on
    ``{lane}_rejudge_payload_sha256_mismatch``.
    """

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
    row: dict[str, object] = {
        "child_payload": {"schema": NESTED_LS_OUTER_JAX_CHILD_SCHEMA},
        "child_payload_sha256": "child-bytes",
        "endpoint_coil_sha256": "coil-bytes",
        "endpoint_surface_sha256": "surface-bytes",
        "endpoint_j": 0.25,
        "endpoint_iota": 0.14,
        "endpoint_g": 2.0,
    }
    payload_raw = _production_child_encoding(rejudge_payload)
    live_envelope: dict[str, object] = {
        "payload": rejudge_payload,
        "payload_raw": payload_raw,
        "payload_sha256": hashlib.sha256(payload_raw.encode("utf-8")).hexdigest(),
    }
    reloaded_envelope = json.loads(
        json.dumps(live_envelope, allow_nan=False, sort_keys=True, indent=2)
    )
    reloaded_keys = list(reloaded_envelope["payload"])
    assert reloaded_keys == sorted(reloaded_keys), (
        "this test is pointless unless the reload actually reorders the "
        f"payload's keys; got {reloaded_keys}"
    )
    assert reloaded_keys != list(rejudge_payload), (
        "the fixture must not already be in sorted order, or the reordering "
        "the reload performs is invisible"
    )

    for label, envelope in (("live", live_envelope), ("reloaded", reloaded_envelope)):
        assert (
            _rejudge_endpoint_mismatch(
                lane="jax",
                row=row,
                rejudge_envelope=envelope,
                budget=3,
                maxcor=10,
            )
            is None
        ), (
            f"the {label} rejudge envelope was refused; the binding must hold "
            "both when the parent has just built it and after the receipt has "
            "round-tripped through disk"
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
                # Required by the child schema, and read by the loader: a
                # receipt a child wrote while failing closed is not a
                # rejudgeable endpoint. None is what a completed run carries.
                "child_fault_reason": None,
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


# --------------------------------------------------------------------------
# Tier-0 hardening. The sealed outer policy's own line-search contract, the
# all-rejected containment endpoint, and the shared stop classifiers.
# --------------------------------------------------------------------------

# Everything from here down was READ OFF a real run on the installed scipy --
# the dcsrch contraction ratios, the abnormal-stop message strings, the stop
# decision table. None of it is derived from documentation, so a scipy bump
# invalidates the observation rather than the code, and the gate below says so
# in one place instead of leaving the bump to surface as an opaque numeric
# failure somewhere in the parametrised tables.
_PINNED_SCIPY_VERSION: Final[str] = "1.17.1"


def test_installed_scipy_is_the_version_these_pins_were_observed_on() -> None:
    assert scipy.__version__ == _PINNED_SCIPY_VERSION, (
        "scipy version drifted: every scipy-behaviour pin below this line was "
        f"observed on {_PINNED_SCIPY_VERSION} and must be RE-OBSERVED on "
        f"{scipy.__version__}, not edited to match. Editing a pin to make a "
        "test pass again would relabel a changed line search, a changed stop "
        "message or a changed status code as unchanged, and both outer lanes "
        "route their restart and publish decisions through exactly those."
    )


# The F3 B37 native lane (``pair2-l2/lane.json``, read at runtime through
# ``load_outer_optimizer_policy``) seals these outer knobs. They are repeated
# as literals because that lane JSON lives in the campaign artifact tree, not
# in this repository, so a unit test cannot read it.
_SEALED_OUTER_FTOL: Final[float] = 0.0
_SEALED_OUTER_GTOL: Final[float] = 1.0e-3
_SEALED_OUTER_MAXLS: Final[int] = 8
_SEALED_OUTER_MAXCOR: Final[int] = 10
_SEALED_REJECTION_DISTANCE_SCALE: Final[float] = 1.0
_B37_OUTER_BUDGET: Final[int] = 37

# Ceiling on dcsrch's trial-to-trial step ratio when every trial is answered
# by the sealed quadratic barrier. Measured live against scipy 1.17.1 by
# ``_drive_fully_rejecting_line_search`` below, not modelled here. It is also
# the exact limit of MINPACK dcstep's "the new point is worse" branch: with
# the incumbent pinned at the left end of the bracket, the cubic minimiser
# sits at one third of the bracket as the barrier's rise vanishes and closer
# to the incumbent as the rise grows, so 1/3 is approached and never passed.
_DCSRCH_BARRIER_CONTRACTION_CEILING: Final[float] = 1.0 / 3.0

# The ceiling is ATTAINED IN THE LIMIT, not merely bounded: dcstep's bound
# ``r = (g - v) / (2g + u + 2v)`` with ``g = sqrt(u**2 + 4uv + v**2)`` reduces
# to ``0 <= 6uv + 24v**2``, an equality as ``v -> 0``. Asserting ``<= 1/3``
# exactly therefore tests floating-point rounding in scipy's cubic
# interpolation as much as it tests contraction. Measured relative headroom
# ``(1/3 - worst) / (1/3)`` over the three parametrised cells on scipy 1.17.1:
# 5.25e-4, 1.13e-3, and 2.57e-8 -- the last (the 675-DOF cell, whose barrier
# rise is largest and so sits closest to the limit) is within a few hundred
# million ULP of the bare ceiling, close enough that one differently-rounded
# interpolation would fail the test for no behavioural reason.
#
# So the bound carries an explicit relative margin. 1e-6 is ~39x the tightest
# measured headroom, which is ample for rounding, while a line search that
# genuinely STOPPED contracting geometrically would show ratios of order 0.5
# to 1.0 -- five orders of magnitude coarser than the margin. The bound still
# discriminates "the search contracts geometrically" from "it does not"; it
# just no longer discriminates the last ULP of a limit it cannot exceed.
_DCSRCH_CONTRACTION_ROUNDING_MARGIN: Final[float] = 1.0e-6
_DCSRCH_BARRIER_CONTRACTION_BOUND: Final[float] = (
    _DCSRCH_BARRIER_CONTRACTION_CEILING * (1.0 + _DCSRCH_CONTRACTION_ROUNDING_MARGIN)
)


@dataclass(frozen=True, slots=True)
class _RejectedLineSearch:
    """What one fully-rejected scipy line search actually did."""

    trial_steps: tuple[float, ...]
    trial_barriers: tuple[float, ...]
    status: int
    message: str


def _drive_fully_rejecting_line_search(
    *,
    anchor_value: float,
    gradient_scale: float,
    dimension: int,
    seed: int,
) -> _RejectedLineSearch:
    """Run the real L-BFGS-B line search against the real barrier.

    Evaluation 0 returns a declared incumbent value and gradient; every later
    evaluation is answered by :func:`nested_ls_outer_rejection_barrier`
    anchored at that incumbent, which is exactly what both outer children do
    when an inner solve refuses a trial. Nothing about dcsrch is modelled
    here -- the returned step lengths are the distances scipy itself asked
    for.
    """

    generator = np.random.default_rng(seed)
    anchor_point = generator.standard_normal(dimension) * 0.1
    anchor_gradient = generator.standard_normal(dimension) * gradient_scale
    trial_steps: list[float] = []
    trial_barriers: list[float] = []
    evaluations = 0

    def rejecting_objective(point: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal evaluations
        evaluations += 1
        trial = np.asarray(point, dtype=np.float64)
        if evaluations == 1:
            return anchor_value, np.array(anchor_gradient, copy=True)
        value, gradient = nested_ls_outer_rejection_barrier(
            anchor_value=anchor_value,
            anchor_parameters=anchor_point,
            trial_parameters=trial,
            scale=_SEALED_REJECTION_DISTANCE_SCALE,
        )
        trial_steps.append(float(np.linalg.norm(trial - anchor_point)))
        trial_barriers.append(value)
        return value, gradient

    result = minimize(
        rejecting_objective,
        np.array(anchor_point, copy=True),
        jac=True,
        method="L-BFGS-B",
        options={
            "ftol": _SEALED_OUTER_FTOL,
            "gtol": _SEALED_OUTER_GTOL,
            "maxls": _SEALED_OUTER_MAXLS,
            "maxiter": _B37_OUTER_BUDGET,
            "maxcor": _SEALED_OUTER_MAXCOR,
        },
    )
    return _RejectedLineSearch(
        trial_steps=tuple(trial_steps),
        trial_barriers=tuple(trial_barriers),
        status=int(result.status),
        message=str(result.message),
    )


@pytest.mark.parametrize(
    ("anchor_value", "gradient_scale", "dimension", "seed"),
    (
        (1.0, 1.0, 2, 0),
        (12.5, 1.0e-2, 8, 3),
        (1.0e4, 1.0e3, 675, 1),
    ),
)
def test_rejected_barrier_trials_contract_by_at_least_three_per_trial(
    anchor_value: float,
    gradient_scale: float,
    dimension: int,
    seed: int,
):
    """Each rejected trial step is at most a third of the previous one.

    "At most a third" to within an explicit rounding margin -- see
    ``_DCSRCH_CONTRACTION_ROUNDING_MARGIN`` for why the bare 1/3 is a
    knife-edge and why the margin does not blunt the test.

    Operationally: a rejecting line search that does not contract burns its
    whole ``maxls`` budget at a near-constant step, never returns to the
    incumbent's neighbourhood, and hands scipy a bracket it cannot close --
    one of the mechanisms behind the measured B37 stall. Geometric
    contraction is what makes the sealed barrier a containment device rather
    than a way to spend evaluations.

    This pins measured scipy 1.17.1 behaviour rather than flipping a repo
    behaviour, so there is no "old behaviour" build to fail it against. It
    fails if the barrier stops being one differentiable function of the
    distance to the incumbent, or if the scipy pin moves to a line search
    with a different interpolation rule -- which is what it is here to catch.
    """

    measured = _drive_fully_rejecting_line_search(
        anchor_value=anchor_value,
        gradient_scale=gradient_scale,
        dimension=dimension,
        seed=seed,
    )

    assert len(measured.trial_steps) >= 2, (
        "the line search did not take enough rejected trials to measure a "
        f"contraction: steps {measured.trial_steps!r}, stop "
        f"{measured.status}/{measured.message!r}"
    )
    # The one-third bound is the cubic-interpolation limit for a bracket
    # whose best point is the incumbent and whose barrier strictly exceeds
    # the incumbent's value. Once the rise underflows into the incumbent's
    # own ULP the interpolation data is degenerate and the bound no longer
    # models anything, so the regime is asserted, not assumed.
    assert all(value > anchor_value for value in measured.trial_barriers), (
        "the barrier collapsed into the incumbent value's ULP, so this cell "
        "no longer measures dcsrch interpolation: barriers "
        f"{measured.trial_barriers!r} against incumbent {anchor_value!r}"
    )

    ratios = tuple(
        measured.trial_steps[index + 1] / measured.trial_steps[index]
        for index in range(len(measured.trial_steps) - 1)
    )
    worst_ratio = max(ratios)
    headroom = (
        _DCSRCH_BARRIER_CONTRACTION_CEILING - worst_ratio
    ) / _DCSRCH_BARRIER_CONTRACTION_CEILING
    assert worst_ratio <= _DCSRCH_BARRIER_CONTRACTION_BOUND, (
        "dcsrch stopped contracting on the sealed rejection barrier: the "
        f"worst measured trial-to-trial step ratio was {worst_ratio!r}, above "
        f"the {_DCSRCH_BARRIER_CONTRACTION_BOUND!r} bound (the analytic "
        f"dcstep ceiling {_DCSRCH_BARRIER_CONTRACTION_CEILING!r} plus a "
        f"{_DCSRCH_CONTRACTION_ROUNDING_MARGIN!r} relative rounding margin). "
        f"Relative headroom against the bare ceiling was {headroom!r}; the "
        "margin exists because that ceiling is attained in the limit as the "
        "barrier's rise vanishes, and the tightest cell measured on scipy "
        "1.17.1 cleared it by only 2.57e-08. A negative headroom of this "
        "order is rounding; one of order 0.1 or more means the line search "
        "is no longer contracting geometrically, which is the finding this "
        f"guards. Measured step sequence {measured.trial_steps!r}; measured "
        f"ratios {ratios!r}."
    )


def test_rejected_line_search_spends_its_whole_budget_collapsing_to_the_anchor():
    """A fully-rejected line search ends its ``maxls`` budget at the anchor.

    The observable is the end of the budget, not the trend: after the sealed
    ``maxls=8`` trials the last point scipy asked for sits within
    ``(1/3)**7`` of the first trial's distance from the incumbent, and scipy
    reports the abnormal stop that the shared classifier restarts on. A
    non-contracting sentinel would instead leave the final trial a
    significant fraction of a full step away, which is what turns a rejected
    outer step into wall clock spent far from the incumbent.

    Like the contraction test this pins measured scipy 1.17.1 behaviour, so
    there is no prior in-tree behaviour to falsify it against.
    """

    measured = _drive_fully_rejecting_line_search(
        anchor_value=12.5,
        gradient_scale=1.0e-2,
        dimension=8,
        seed=3,
    )

    assert len(measured.trial_steps) == _SEALED_OUTER_MAXLS, (
        f"the rejected line search took {len(measured.trial_steps)} trials, "
        f"not the sealed maxls={_SEALED_OUTER_MAXLS}: steps "
        f"{measured.trial_steps!r}"
    )
    assert measured.status == 2 and measured.message.startswith("ABNORMAL"), (
        "a fully-rejected line search must end in scipy's abnormal stop, the "
        "one the shared classifier restarts on; got "
        f"{measured.status}/{measured.message!r}"
    )
    collapse = measured.trial_steps[-1] / measured.trial_steps[0]
    budget_collapse = _DCSRCH_BARRIER_CONTRACTION_CEILING ** (_SEALED_OUTER_MAXLS - 1)
    assert collapse <= budget_collapse, (
        "the rejected line search ended its budget far from the incumbent: "
        f"final/first step ratio {collapse!r} exceeds the geometric budget "
        f"collapse {budget_collapse!r}. Measured step sequence "
        f"{measured.trial_steps!r}."
    )


def _float64_ulp_gap(actual: object, expected: object) -> int:
    """Largest IEEE-754 float64 ULP distance between two equal-shaped blocks."""

    left = np.ascontiguousarray(actual, dtype=np.float64).reshape(-1)
    right = np.ascontiguousarray(expected, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise AssertionError(
            f"cannot measure a ULP gap between shapes {left.shape} and {right.shape}"
        )
    ordered = [
        [
            bits if bits >= 0 else -(1 << 63) - bits
            for bits in (int(word) for word in block.view(np.int64))
        ]
        for block in (left, right)
    ]
    return max(
        (abs(a - b) for a, b in zip(ordered[0], ordered[1], strict=True)),
        default=0,
    )


def _assert_block_is_bitwise(block: str, actual: object, expected: object) -> None:
    """Assert one float64 block is bit-for-bit its declared value."""

    left = np.ascontiguousarray(actual, dtype=np.float64).reshape(-1)
    right = np.ascontiguousarray(expected, dtype=np.float64).reshape(-1)
    # ``array_equal`` alone would accept +0.0 against -0.0, and the byte
    # comparison alone would accept nothing useful about shape, so the
    # endpoint contract asserts both.
    identical = left.shape == right.shape and bool(np.array_equal(left, right))
    if identical:
        identical = nested_ls_outer_parameter_bytes(
            left
        ) == nested_ls_outer_parameter_bytes(right)
    gap = _float64_ulp_gap(left, right) if left.shape == right.shape else -1
    drift = {
        -1: f"a shape change, {left.shape} against {right.shape}",
        0: "0 ULP -- a signed-zero flip that only the byte comparison sees",
    }.get(gap, f"{gap} ULP")
    assert identical, (
        f"the {block} block drifted from the declared start state by {drift}: "
        f"returned {left.tolist()!r}, declared {right.tolist()!r}"
    )


class _NoisedGradientObjective(_FakeObjective):
    """Objective whose reported gradient is deliberately not its own gradient.

    DESC's ``test_overstepping`` perturbs the reported derivative so the
    optimizer proposes steps the true model would never take. The same
    perturbation here guarantees the outer line search leaves the start point
    on every trial instead of converging on it.
    """

    def __init__(self, boozer: _FakeBoozer, *, gradient_noise: np.ndarray) -> None:
        super().__init__(boozer)
        self._gradient_noise = np.array(gradient_noise, dtype=np.float64, copy=True)

    def evaluate(self) -> tuple[float, dict[str, float], np.ndarray]:
        value, terms, gradient = super().evaluate()
        return value, terms, gradient + self._gradient_noise


def test_outer_run_rejecting_every_step_returns_the_start_state_bitwise(monkeypatch):
    """An all-rejected outer run returns exactly the bytes it started from.

    The inner solve is crippled so that only the declared start point is ever
    feasible, and the reported gradient is noised so scipy keeps proposing
    steps. Every trial therefore lands on the rejection barrier, the
    transaction restores the incumbent, and the run's endpoint -- coil DOFs,
    surface DOFs, iota and G -- must be the start state bit for bit, not
    merely close to it. This is the containment promise the whole
    commit-on-accept design exists to make.

    Falsification, run against a subclass whose ``_restore_anchor`` is a
    no-op -- the pre-transaction behaviour: the immutable anchor record still
    reads back correctly, but the live Boozer object is left holding the last
    rejected trial (coil DOFs off the start point, surface ``-0.12490983...``
    against the declared ``-0.125``), so the two live-state assertions below
    fail. Those are the assertions that make this a containment test rather
    than a test that a frozen dataclass stayed frozen.
    """

    start_coil_dofs = np.array([0.25, -0.5, 0.125], dtype=np.float64)
    # The declared start state, derived here rather than read back out of the
    # run: the crippled inner solve writes ``sum(coils)`` into the surface,
    # and holds iota and G at the seed the run was warm started from.
    declared_surface_dofs = np.array([float(np.sum(start_coil_dofs))])
    declared_iota = 0.14
    declared_g = 2.0
    start_bytes = nested_ls_outer_parameter_bytes(start_coil_dofs)

    boozer = _FakeBoozer(start_coil_dofs, np.array([99.0], dtype=np.float64))
    objective = _NoisedGradientObjective(
        boozer,
        gradient_noise=np.array([0.75, 0.5, -1.5], dtype=np.float64),
    )

    def crippled_inner(fake_boozer, *, iota: float, G: float):
        coil = np.asarray(fake_boozer.biotsavart.x, dtype=np.float64)
        # A failed inner solve still leaves its partial state behind; only
        # the transaction's restore puts the surface back.
        fake_boozer.surface.set_dofs(np.array([float(np.sum(coil))]))
        return {
            "success": nested_ls_outer_parameter_bytes(coil) == start_bytes,
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
        crippled_inner,
    )
    run = native_child.NativeOuterRun(
        objective,  # type: ignore[arg-type]
        seed=native_child.InnerWarmStart(
            surface_dofs=np.array([99.0], dtype=np.float64),
            iota=declared_iota,
            G=declared_g,
        ),
        rejection_distance_scale=_SEALED_REJECTION_DISTANCE_SCALE,
        start_coil_dofs=start_coil_dofs,
    )

    result = minimize(
        run,
        np.array(start_coil_dofs, copy=True),
        jac=True,
        method="L-BFGS-B",
        callback=run.accept,
        options={
            "ftol": _SEALED_OUTER_FTOL,
            "gtol": _SEALED_OUTER_GTOL,
            "maxls": _SEALED_OUTER_MAXLS,
            "maxiter": _B37_OUTER_BUDGET,
            "maxcor": _SEALED_OUTER_MAXCOR,
        },
    )

    assert run.records[0]["rejection_reason"] is None
    assert len(run.feasible) == 1, (
        "the crippled inner solve was supposed to accept the start point and "
        f"nothing else; feasible evaluations: {len(run.feasible)}"
    )
    trial_reasons = [record["rejection_reason"] for record in run.records[1:]]
    assert trial_reasons and set(trial_reasons) == {"inner_solve_failed"}, (
        "this regression only means something when every trial after the "
        f"start point rejected; observed reasons {trial_reasons!r}"
    )

    endpoint = run.endpoint_at(np.asarray(result.x, dtype=np.float64))
    assert endpoint is not None, (
        "the optimizer's final iterate is not the committed transaction "
        f"point: result.x={np.asarray(result.x).tolist()!r} against committed "
        f"{run.candidates.committed.coil_dofs.tolist()!r}"
    )

    _assert_block_is_bitwise("endpoint coil DOF", endpoint.coil_dofs, start_coil_dofs)
    _assert_block_is_bitwise(
        "endpoint surface DOF",
        endpoint.warm_start.surface_dofs,
        declared_surface_dofs,
    )
    _assert_block_is_bitwise("endpoint iota", endpoint.warm_start.iota, declared_iota)
    _assert_block_is_bitwise("endpoint G", endpoint.warm_start.G, declared_g)
    # Containment is a property of the live objects too, not only of the
    # record the run hands back.
    _assert_block_is_bitwise("live coil DOF", boozer.biotsavart.x, start_coil_dofs)
    _assert_block_is_bitwise(
        "live surface DOF",
        boozer.surface.get_dofs(),
        declared_surface_dofs,
    )

    assert not nested_ls_outer_endpoint_success(
        endpoint_matches=True,
        ftol=_SEALED_OUTER_FTOL,
        status=int(result.status),
        message=str(result.message),
    ), (
        "an outer run that never left its start point must not be published "
        f"as a successful endpoint; scipy stopped with {result.status}/"
        f"{str(result.message)!r}"
    )


# The exact strings scipy 1.17.1 composes for L-BFGS-B, read out of the
# installed interpreter's own source. ``result.message`` is
# ``status_messages[task[0]] + ": " + task_messages[task[1]]``
# (.venv-qn-cpu/lib/python3.11/site-packages/scipy/optimize/_lbfgsb_py.py:505)
# over the two tables at _lbfgsb_py.py:51-61 and _lbfgsb_py.py:64-92;
# ``result.status`` is the warnflag derived at _lbfgsb_py.py:487-492, so the
# only codes L-BFGS-B itself can return are 0, 1 and 2. ``minimize`` then
# overwrites status and message with 99 / "`callback` raised `StopIteration`."
# at _minimize.py:823-826, which is a fourth code the classifiers must judge.
# Every literal below is reproduced from a live run by
# ``test_stop_table_quotes_only_the_stops_scipy_actually_emits``.
_MSG_PGTOL: Final[str] = "CONVERGENCE: NORM OF PROJECTED GRADIENT <= PGTOL"
_MSG_MAXITER: Final[str] = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
_MSG_MAXFUN: Final[str] = "STOP: TOTAL NO. OF F,G EVALUATIONS EXCEEDS LIMIT"
# task_messages[0] is the empty string, so the abnormal stop carries no
# detail at all -- the trailing space is scipy's, not a typo.
_MSG_ABNORMAL: Final[str] = "ABNORMAL: "
# Composed by _lbfgsb_py.py but unreachable through ``minimize``, which
# rewrites it to the status-99 pair; ``fmin_l_bfgs_b`` still returns it.
_MSG_CALLBACK_HALT: Final[str] = "STOP: CALLBACK REQUESTED HALT"
_MSG_STOP_ITERATION: Final[str] = "`callback` raised `StopIteration`."

_POSITIVE_FTOL: Final[float] = 1.0e-12


@dataclass(frozen=True, slots=True)
class _StopCase:
    """One cell of the lane-agnostic stop decision table."""

    label: str
    ftol: float
    status: int
    message: str
    endpoint_matches: bool
    restart_reason: str | None
    endpoint_success: bool


# The decision table both outer children share. Written out cell by cell on
# purpose: a generator that derived these from the same predicates the
# classifiers use would agree with any drift instead of catching it.
_LBFGSB_STOP_CASES: Final[tuple[_StopCase, ...]] = (
    # scipy status 0, projected-gradient convergence. Never an FTOL stop, so
    # the sealed ftol=0 policy has nothing to object to.
    _StopCase("pgtol|ftol=0|matched", 0.0, 0, _MSG_PGTOL, True, None, True),
    _StopCase("pgtol|ftol=0|unmatched", 0.0, 0, _MSG_PGTOL, False, None, False),
    _StopCase("pgtol|ftol>0|matched", _POSITIVE_FTOL, 0, _MSG_PGTOL, True, None, True),
    _StopCase(
        "pgtol|ftol>0|unmatched", _POSITIVE_FTOL, 0, _MSG_PGTOL, False, None, False
    ),
    # scipy status 0, relative-reduction convergence. Under ftol=0 this stop
    # is impossible by construction, so seeing it is a false stall to restart
    # from and never a publishable endpoint. Under a positive ftol it is an
    # ordinary convergence.
    _StopCase(
        "factr|ftol=0|matched",
        0.0,
        0,
        NESTED_LS_OUTER_FTOL_STALL_MESSAGE,
        True,
        "false_ftol_stall",
        False,
    ),
    _StopCase(
        "factr|ftol=0|unmatched",
        0.0,
        0,
        NESTED_LS_OUTER_FTOL_STALL_MESSAGE,
        False,
        "false_ftol_stall",
        False,
    ),
    _StopCase(
        "factr|ftol>0|matched",
        _POSITIVE_FTOL,
        0,
        NESTED_LS_OUTER_FTOL_STALL_MESSAGE,
        True,
        None,
        True,
    ),
    _StopCase(
        "factr|ftol>0|unmatched",
        _POSITIVE_FTOL,
        0,
        NESTED_LS_OUTER_FTOL_STALL_MESSAGE,
        False,
        None,
        False,
    ),
    # scipy status 1, budget exhausted. This is the expected stop of a
    # fixed-budget rung: terminal, and a usable endpoint whenever the
    # optimizer's final iterate is the committed transaction point.
    _StopCase("maxiter|ftol=0|matched", 0.0, 1, _MSG_MAXITER, True, None, True),
    _StopCase("maxiter|ftol=0|unmatched", 0.0, 1, _MSG_MAXITER, False, None, False),
    _StopCase(
        "maxiter|ftol>0|matched", _POSITIVE_FTOL, 1, _MSG_MAXITER, True, None, True
    ),
    _StopCase(
        "maxiter|ftol>0|unmatched", _POSITIVE_FTOL, 1, _MSG_MAXITER, False, None, False
    ),
    _StopCase("maxfun|ftol=0|matched", 0.0, 1, _MSG_MAXFUN, True, None, True),
    _StopCase("maxfun|ftol=0|unmatched", 0.0, 1, _MSG_MAXFUN, False, None, False),
    _StopCase(
        "maxfun|ftol>0|matched", _POSITIVE_FTOL, 1, _MSG_MAXFUN, True, None, True
    ),
    _StopCase(
        "maxfun|ftol>0|unmatched", _POSITIVE_FTOL, 1, _MSG_MAXFUN, False, None, False
    ),
    # scipy status 2 with the ABNORMAL message: the line search gave up.
    # Restartable from the committed incumbent, never a usable endpoint.
    _StopCase(
        "abnormal|ftol=0|matched",
        0.0,
        2,
        _MSG_ABNORMAL,
        True,
        "abnormal_line_search",
        False,
    ),
    _StopCase(
        "abnormal|ftol=0|unmatched",
        0.0,
        2,
        _MSG_ABNORMAL,
        False,
        "abnormal_line_search",
        False,
    ),
    _StopCase(
        "abnormal|ftol>0|matched",
        _POSITIVE_FTOL,
        2,
        _MSG_ABNORMAL,
        True,
        "abnormal_line_search",
        False,
    ),
    _StopCase(
        "abnormal|ftol>0|unmatched",
        _POSITIVE_FTOL,
        2,
        _MSG_ABNORMAL,
        False,
        "abnormal_line_search",
        False,
    ),
    # scipy status 2 WITHOUT the ABNORMAL message: a halting callback. The
    # endpoint judge refuses it on the status alone, but restarting would
    # just re-run whatever asked to stop, so it is terminal.
    _StopCase(
        "callback_halt|ftol=0|matched", 0.0, 2, _MSG_CALLBACK_HALT, True, None, False
    ),
    _StopCase(
        "callback_halt|ftol=0|unmatched", 0.0, 2, _MSG_CALLBACK_HALT, False, None, False
    ),
    _StopCase(
        "callback_halt|ftol>0|matched",
        _POSITIVE_FTOL,
        2,
        _MSG_CALLBACK_HALT,
        True,
        None,
        False,
    ),
    _StopCase(
        "callback_halt|ftol>0|unmatched",
        _POSITIVE_FTOL,
        2,
        _MSG_CALLBACK_HALT,
        False,
        None,
        False,
    ),
    # scipy status 99: ``minimize``'s StopIteration rewrite. A callback that
    # raised did not let the optimizer finish, so the endpoint judge refuses
    # it on the status alone -- 99 is outside
    # ``NESTED_LS_OUTER_PUBLISHABLE_STOP_STATUSES``. Restarting would just
    # re-run whatever asked to stop, so it is terminal. Pinned here because
    # the lanes' callback is the transaction's own ``accept``, and the
    # upgrade plan intends to raise ``StopIteration`` from it.
    _StopCase(
        "stop_iteration|ftol=0|matched", 0.0, 99, _MSG_STOP_ITERATION, True, None, False
    ),
    _StopCase(
        "stop_iteration|ftol=0|unmatched",
        0.0,
        99,
        _MSG_STOP_ITERATION,
        False,
        None,
        False,
    ),
    _StopCase(
        "stop_iteration|ftol>0|matched",
        _POSITIVE_FTOL,
        99,
        _MSG_STOP_ITERATION,
        True,
        None,
        False,
    ),
    _StopCase(
        "stop_iteration|ftol>0|unmatched",
        _POSITIVE_FTOL,
        99,
        _MSG_STOP_ITERATION,
        False,
        None,
        False,
    ),
)


@pytest.mark.parametrize(
    "case",
    _LBFGSB_STOP_CASES,
    ids=[case.label for case in _LBFGSB_STOP_CASES],
)
def test_restart_reason_classifies_every_real_lbfgsb_stop(case: _StopCase):
    """``nested_ls_outer_restart_reason`` returns the table's verdict.

    Both outer children route their restart decision through this one pure
    function, so any drift here forks the lanes silently: one lane would burn
    budget restarting a stop the other treats as terminal, and the two
    endpoints would stop being comparable. The table is exhaustive over the
    ``(status, message)`` pairs scipy 1.17.1 can produce crossed with both
    FTOL regimes, including the cells that must return ``None``.

    This pins current behaviour rather than changing it; it fails against any
    edit that widens the abnormal predicate past ``ABNORMAL``-prefixed
    status-2 stops (the callback-halt cells) or that lets a positive FTOL
    still report a false stall (the ``factr|ftol>0`` cells).
    """

    assert (
        nested_ls_outer_restart_reason(
            ftol=case.ftol,
            status=case.status,
            message=case.message,
        )
        == case.restart_reason
    ), (
        f"stop {case.label} (status={case.status}, message={case.message!r}, "
        f"ftol={case.ftol!r}) must classify as {case.restart_reason!r}"
    )


def test_endpoint_judge_refuses_a_stop_status_it_has_never_seen():
    """An unrecognised scipy stop code is unpublishable, not publishable.

    The stop table above pins the codes scipy 1.17.1 emits today. The point of
    this test is the codes it does not: the judge admits from
    ``NESTED_LS_OUTER_PUBLISHABLE_STOP_STATUSES`` rather than excluding the
    codes we happened to think of, so a scipy upgrade that introduces a new
    terminal status cannot make a truncated run's endpoint publishable by
    default. That is the failure the predecessor had — it excluded status 2 by
    name and so silently admitted status 99 when ``minimize`` grew its
    ``StopIteration`` rewrite.

    Fails against any edit that turns the allow-list back into a deny-list.
    """

    # Every status this contract has ever met: the ones the table pins plus
    # the ones the judge admits. One above the largest of them is novel by
    # construction -- it cannot be in either set.
    seen_statuses = {case.status for case in _LBFGSB_STOP_CASES} | set(
        NESTED_LS_OUTER_PUBLISHABLE_STOP_STATUSES
    )
    novel = max(seen_statuses) + 1

    assert (
        nested_ls_outer_endpoint_success(
            endpoint_matches=True,
            ftol=0.0,
            status=novel,
            message="STOP: A REASON THIS CONTRACT HAS NEVER SEEN",
        )
        is False
    ), (
        f"an endpoint stopped on unrecognised scipy status {novel} was judged "
        "publishable; the judge must admit only "
        f"{sorted(NESTED_LS_OUTER_PUBLISHABLE_STOP_STATUSES)} and refuse "
        "everything else, so that a new scipy status fails closed"
    )


@pytest.mark.parametrize(
    "case",
    _LBFGSB_STOP_CASES,
    ids=[case.label for case in _LBFGSB_STOP_CASES],
)
def test_endpoint_success_classifies_every_real_lbfgsb_stop(case: _StopCase):
    """``nested_ls_outer_endpoint_success`` returns the table's verdict.

    This is the flag the claim parent reads to decide whether a child's
    endpoint may be published at all, so a drift here would let one lane
    publish an endpoint the other refuses. The table is exhaustive over the
    real scipy stops crossed with both FTOL regimes and both values of
    ``endpoint_matches``, including the cells that must return ``True``.

    It pins current behaviour; it fails against any edit that publishes an
    endpoint whose bytes are not the optimizer's final iterate (every
    ``unmatched`` cell), that publishes a status-2 or status-99 stop, or that
    publishes an FTOL stall the sealed ``ftol=0`` policy made impossible. The
    four ``stop_iteration`` cells are the falsification of the predecessor
    implementation, which excluded status 2 by name (``int(status) != 2``)
    and therefore published a callback-halted run's endpoint as a good one.
    """

    assert (
        nested_ls_outer_endpoint_success(
            endpoint_matches=case.endpoint_matches,
            ftol=case.ftol,
            status=case.status,
            message=case.message,
        )
        is case.endpoint_success
    ), (
        f"stop {case.label} (status={case.status}, message={case.message!r}, "
        f"ftol={case.ftol!r}, endpoint_matches={case.endpoint_matches}) must "
        f"judge as endpoint_success={case.endpoint_success}"
    )


def _quadratic(point: np.ndarray) -> tuple[float, np.ndarray]:
    values = np.asarray(point, dtype=np.float64)
    return float(values @ values), 2.0 * values


def _rosenbrock(point: np.ndarray) -> tuple[float, np.ndarray]:
    values = np.asarray(point, dtype=np.float64)
    residual = values[1] - values[0] ** 2
    return (
        float(100.0 * residual**2 + (1.0 - values[0]) ** 2),
        np.array(
            [
                -400.0 * values[0] * residual - 2.0 * (1.0 - values[0]),
                200.0 * residual,
            ],
            dtype=np.float64,
        ),
    )


def _halt_on_second_iteration():
    seen = 0

    def callback(intermediate_result):
        nonlocal seen
        seen += 1
        if seen >= 2:
            raise StopIteration

    return callback


def test_stop_table_quotes_only_the_stops_scipy_actually_emits():
    """Every table row's stop is one scipy 1.17.1 really produces.

    A decision table over invented message strings classifies nothing. This
    drives the installed scipy six ways -- projected-gradient convergence,
    relative-reduction convergence, the iteration and evaluation budgets, the
    fully-rejected line search, and a halting callback on both entry points
    -- and requires the produced ``(status, message)`` pairs and the table's
    pairs to be the same set.

    It fails today against the previously hand-written literal
    ``"ABNORMAL: LINE SEARCH FAILED"``: scipy composes ``"ABNORMAL: "`` with
    an empty task message, so that string is never emitted.
    """

    _halted_x, _halted_f, halted_info = fmin_l_bfgs_b(
        _rosenbrock,
        np.array([-1.2, 1.0]),
        factr=0.0,
        pgtol=0.0,
        callback=_halt_on_second_iteration(),
    )
    rejected = _drive_fully_rejecting_line_search(
        anchor_value=12.5,
        gradient_scale=1.0e-2,
        dimension=8,
        seed=3,
    )
    produced = {
        (rejected.status, rejected.message),
        (int(halted_info["warnflag"]), str(halted_info["task"])),
    }
    for label, result in (
        (
            "pgtol",
            minimize(_quadratic, np.array([1.0, 2.0]), jac=True, method="L-BFGS-B"),
        ),
        (
            "factr",
            minimize(
                _rosenbrock,
                np.array([-1.2, 1.0]),
                jac=True,
                method="L-BFGS-B",
                options={"ftol": 1.0e-1, "gtol": 0.0},
            ),
        ),
        (
            "maxiter",
            minimize(
                _quadratic,
                np.array([1.0e6, 2.0e6]),
                jac=True,
                method="L-BFGS-B",
                options={"maxiter": 1, "gtol": 0.0, "ftol": 0.0},
            ),
        ),
        (
            "maxfun",
            minimize(
                _quadratic,
                np.array([1.0e6, 2.0e6]),
                jac=True,
                method="L-BFGS-B",
                options={"maxfun": 2, "gtol": 0.0, "ftol": 0.0},
            ),
        ),
        (
            "stop_iteration",
            minimize(
                _rosenbrock,
                np.array([-1.2, 1.0]),
                jac=True,
                method="L-BFGS-B",
                callback=_halt_on_second_iteration(),
                options={"ftol": 0.0, "gtol": 0.0},
            ),
        ),
    ):
        assert result is not None, label
        produced.add((int(result.status), str(result.message)))

    tabled = {(case.status, case.message) for case in _LBFGSB_STOP_CASES}
    assert produced == tabled, (
        "the stop decision table has drifted from what scipy emits. Produced "
        f"but not tabled: {sorted(produced - tabled)!r}; tabled but never "
        f"produced: {sorted(tabled - produced)!r}"
    )
