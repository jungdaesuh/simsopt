from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from simsopt_jax.geo.optimizers.filter_trust_region_sqp import (
    FilterTrustRegionSQPStatus,
)
from simsopt_jax.solve.fullspace import (
    CFS_FTR1_POLICY,
    LEGACY_V1_ROUTE_CONTRACT_SHA256,
    LEGACY_V1_ROUTE_CONTRACT_SIZE_BYTES,
    ROUTE_V2_CONTRACT_SHA256,
    ROUTE_V2_CONTRACT_SIZE_BYTES,
    ROUTE_V3_CONTRACT_SHA256,
    ROUTE_V3_CONTRACT_SIZE_BYTES,
    frozen_route_contract_payload_v3,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLAN = _REPO_ROOT / (
    "docs/single_stage_jax_gpu_coupled_fullspace_filter_trust_region_"
    "implementation_plan.md"
)
_BUDGET = _REPO_ROOT / (
    "docs/single_stage_jax_gpu_coupled_fullspace_filter_trust_region_phase0_budget.json"
)
_PRIOR_EVIDENCE = _REPO_ROOT / (
    "docs/single_stage_jax_gpu_coupled_fullspace_filter_trust_region_"
    "prior_evidence_manifest.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def _load_canonical_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert path.read_bytes() == _canonical_json_bytes(payload)
    return payload


def test_ftr_budget_binds_current_plan_route_contract_and_parent_evidence() -> None:
    budget = _load_canonical_json(_BUDGET)
    identity = budget["identity"]
    assert isinstance(identity, dict)

    assert identity == {
        "plan_sha256": _sha256(_PLAN),
        "prior_evidence_manifest_sha256": _sha256(_PRIOR_EVIDENCE),
        "route": "CFS-FTR1",
        "route_contract_sha256": ROUTE_V3_CONTRACT_SHA256,
        "route_contract_size_bytes": ROUTE_V3_CONTRACT_SIZE_BYTES,
        "schema_version": "single-stage-fullspace-cfs-ftr1-budget-v1",
    }
    route_contract = _canonical_json_bytes(frozen_route_contract_payload_v3())
    assert len(route_contract) == ROUTE_V3_CONTRACT_SIZE_BYTES == 8179
    assert hashlib.sha256(route_contract).hexdigest() == ROUTE_V3_CONTRACT_SHA256


def test_ftr_budget_preserves_legacy_route_contract_digests() -> None:
    budget = _load_canonical_json(_BUDGET)

    assert budget["legacy"] == {
        "route_v1_contract_sha256": LEGACY_V1_ROUTE_CONTRACT_SHA256,
        "route_v1_contract_size_bytes": LEGACY_V1_ROUTE_CONTRACT_SIZE_BYTES,
        "route_v2_contract_sha256": ROUTE_V2_CONTRACT_SHA256,
        "route_v2_contract_size_bytes": ROUTE_V2_CONTRACT_SIZE_BYTES,
    }


def test_ftr_budget_matches_the_frozen_policy_and_terminal_statuses() -> None:
    budget = _load_canonical_json(_BUDGET)
    algorithm = budget["algorithm"]
    gates = budget["gates"]
    assert isinstance(algorithm, dict)
    assert isinstance(gates, dict)

    assert algorithm["initial_multipliers"] == 0.0
    assert algorithm["linear_solve_forward_error_tolerance"] == (
        CFS_FTR1_POLICY.linear_solve_forward_error_tolerance
    )
    assert algorithm["linear_solve_relative_residual_tolerance"] == (
        CFS_FTR1_POLICY.linear_solve_relative_residual_tolerance
    )
    assert algorithm["normal_step"] == {
        "normal_radius_fraction": CFS_FTR1_POLICY.normal_radius_fraction,
        "solve": "Cholesky of A A^T",
        "zero_constraint_step": "exact zero",
    }
    assert algorithm["tangential_step"] == {
        "boundary_fraction": CFS_FTR1_POLICY.boundary_fraction,
        "maximum_projected_cg_iterations": (
            CFS_FTR1_POLICY.maximum_tangential_cg_iterations
        ),
        "projector": "I-A^T(A A^T)^-1 A",
        "tangency_relative_residual_tolerance": (
            CFS_FTR1_POLICY.tangency_relative_residual_tolerance
        ),
    }
    assert gates["maximum_joint_evaluations"] == (
        CFS_FTR1_POLICY.maximum_joint_evaluations
    )
    assert gates["maximum_optimizer_iterations"] == CFS_FTR1_POLICY.maximum_iterations
    assert gates["ten_step_process_timeout_s"] == (
        CFS_FTR1_POLICY.ten_step_process_timeout_s
    )
    assert gates["complete_process_timeout_s"] == (
        CFS_FTR1_POLICY.complete_process_timeout_s
    )
    assert budget["termination_codes"] == [
        status.name
        for status in FilterTrustRegionSQPStatus
        if status is not FilterTrustRegionSQPStatus.RUNNING
    ]


def test_ftr_prior_evidence_manifest_revalidates_every_frozen_byte() -> None:
    manifest = _load_canonical_json(_PRIOR_EVIDENCE)
    assert manifest["schema_version"] == (
        "single-stage-fullspace-cfs-ftr1-prior-evidence-manifest-v1"
    )
    assert manifest["parent_route"] == "CFS-SQP1"
    assert manifest["parent_disposition"] == ("CLOSED_BOUNDED_NEGATIVE / NON_PROMOTING")

    entries = manifest["entries"]
    assert isinstance(entries, list)
    assert len(entries) == 8
    for entry in entries:
        assert isinstance(entry, dict)
        path = Path(entry["root"]) / entry["relative_path"]
        assert path.is_file()
        assert path.stat().st_size == entry["size_bytes"]
        assert _sha256(path) == entry["sha256"]
        assert f"{stat.S_IMODE(path.stat().st_mode):04o}" == entry["mode"]


def test_ftr_budget_parent_evidence_matches_the_manifest_entries() -> None:
    budget = _load_canonical_json(_BUDGET)
    manifest = _load_canonical_json(_PRIOR_EVIDENCE)
    entries = manifest["entries"]
    assert isinstance(entries, list)
    digests = {
        entry["relative_path"]: entry["sha256"]
        for entry in entries
        if isinstance(entry, dict)
    }

    assert budget["parent_evidence"] == {
        "disposition": manifest["parent_disposition"],
        "gate_receipt_sha256": digests["gates/canary-10/gate-receipt.json"],
        "gpu_memory_sha256": digests["gates/canary-10/gpu-memory.json"],
        "raw_result_sha256": digests["gates/canary-10/raw-result.json"],
        "results_document_sha256": digests[
            "docs/single_stage_jax_gpu_sqp_primal_dual_results.md"
        ],
        "revision3_plan_sha256": digests[
            "docs/single_stage_jax_gpu_sqp_primal_dual_implementation_plan_r3.md"
        ],
        "runtime_evidence_sha256": digests[
            "gates/canary-10/evidence/runtime-evidence.json"
        ],
    }
