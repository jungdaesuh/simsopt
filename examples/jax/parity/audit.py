"""Independent fail-closed audit for a published parity run."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.parity.arbiter import LaneObservation, arbitrate
from examples.jax.parity.artifacts import read_bytes
from examples.jax.parity.input_bundle import read_input_bundle
from examples.jax.parity.publication import require_published_run
from examples.jax.parity.provenance import (
    ExecutedSource,
    collect_repository_state,
    validate_sources_current,
)
from examples.jax.parity.receipts import load_lane_observation

_LANES = frozenset({"native-cpu", "jax-cpu", "jax-gpu"})


@dataclass(frozen=True)
class AuditResult:
    run_id: str
    case_count: int
    lane_receipt_count: int
    comparison_count: int
    authoritative: bool
    verdict: str


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _string(record: dict[str, object], field: str, context: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{field} must be a non-empty string")
    return value


def _boolean(record: dict[str, object], field: str, context: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise TypeError(f"{context}.{field} must be boolean")
    return value


def _canonical_relative_path(value: str, context: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
    ):
        raise ValueError(f"{context} must be a canonical relative path")
    return path


def _explicit_sources(value: object) -> tuple[ExecutedSource, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("summary explicit_sources must be a non-empty array")
    sources = []
    for item in value:
        record = _mapping(item, "explicit source")
        if set(record) != {"path", "sha256", "git_blob_id"}:
            raise ValueError("explicit source has invalid fields")
        blob = record["git_blob_id"]
        if blob is not None and (not isinstance(blob, str) or not blob):
            raise ValueError("explicit source git_blob_id must be a string or null")
        sources.append(
            ExecutedSource(
                path=_string(record, "path", "explicit source"),
                sha256=_string(record, "sha256", "explicit source"),
                git_blob_id=blob,
            )
        )
    return tuple(sources)


def audit_published_run(
    run_directory: Path,
    *,
    repo_root: Path,
    require_authoritative: bool = False,
) -> AuditResult:
    """Re-read all receipts and independently validate an aggregate verdict."""
    run_directory = require_published_run(run_directory.parent, run_directory.name)
    summary = _mapping(json.loads(read_bytes(run_directory, "summary.json")), "summary")
    summary_schema = summary.get("schema_version")
    if summary_schema not in (1, 2):
        raise ValueError("unsupported aggregate summary schema")
    run_id = _string(summary, "run_id", "summary")
    if run_id != run_directory.name:
        raise ValueError("summary run_id does not match its directory")
    if _string(summary, "verdict", "summary") != "pass":
        raise ValueError("only passing published runs can be audited")
    authoritative = _boolean(summary, "authoritative", "summary")
    if require_authoritative and not authoritative:
        raise ValueError("run is exploratory, not authoritative")
    repository_commit = _string(summary, "repository_commit", "summary")
    repository_state = collect_repository_state(repo_root)
    if authoritative and (
        repository_state.repository_dirty
        or repository_state.repository_commit != repository_commit
    ):
        raise ValueError(
            "authoritative audit requires the clean recorded repository checkout"
        )
    contract_pair = load_runtime_contract_pair(
        repo_root / "examples" / "jax" / "manifest.json",
        repo_root / "examples" / "jax" / "parity_manifest.json",
        repo_root=repo_root,
    )
    if summary.get("manifest_schema_version") != contract_pair.version_pair[0]:
        raise ValueError("aggregate manifest schema version mismatch")
    if summary_schema == 2 and (
        summary.get("parity_manifest_schema_version") != contract_pair.version_pair[1]
    ):
        raise ValueError("aggregate parity manifest schema version mismatch")
    expected_legacy_adapter = (
        contract_pair.used_legacy_adapter
        if summary_schema == 2
        else contract_pair.version_pair[0] == 1
    )
    if summary.get("used_legacy_manifest_adapter") is not expected_legacy_adapter:
        raise ValueError("aggregate legacy manifest adapter mismatch")
    scale_value = summary.get(
        "scale",
        "bounded" if summary.get("smoke") is True else "native_default",
    )
    if scale_value not in ("bounded", "native_default"):
        raise ValueError("aggregate execution scale is invalid")
    parity_manifest = contract_pair.parity
    relationships_by_case = {
        relationship.case_id: relationship
        for relationship in parity_manifest.relationships
        if relationship.case_id is not None
    }
    lanes_value = summary.get("lanes")
    if (
        not isinstance(lanes_value, list)
        or not lanes_value
        or not all(isinstance(lane, str) and lane in _LANES for lane in lanes_value)
        or len(lanes_value) != len(set(lanes_value))
    ):
        raise ValueError("summary lanes are invalid")
    lanes = tuple(lanes_value)
    validate_sources_current(
        repo_root, _explicit_sources(summary.get("explicit_sources"))
    )
    cases_value = summary.get("cases")
    if not isinstance(cases_value, list) or not cases_value:
        raise ValueError("summary cases must be a non-empty array")
    case_ids: set[str] = set()
    lane_receipt_count = 0
    comparison_count = 0
    for case_value in cases_value:
        case = _mapping(case_value, "case")
        case_id = _string(case, "case_id", "case")
        if case_id in case_ids:
            raise ValueError(f"duplicate case summary: {case_id}")
        case_ids.add(case_id)
        if _string(case, "verdict", case_id) != "pass":
            raise ValueError(f"case verdict is not pass: {case_id}")
        expected_input = _string(case, "input_fingerprint", case_id)
        expected_configuration = _string(case, "configuration_fingerprint", case_id)
        input_bundle, _input_arrays = read_input_bundle(
            run_directory / case_id / "inputs"
        )
        if input_bundle.case_id != case_id:
            raise ValueError(f"input bundle case mismatch: {case_id}")
        if input_bundle.scale != scale_value:
            raise ValueError(f"input bundle scale mismatch: {case_id}")
        if input_bundle.input_fingerprint != expected_input:
            raise ValueError(f"input bundle fingerprint mismatch: {case_id}")
        if input_bundle.configuration_fingerprint != expected_configuration:
            raise ValueError(f"input bundle configuration mismatch: {case_id}")
        stages_value = case.get("completed_workflow_stages")
        if not isinstance(stages_value, list) or not all(
            isinstance(stage, str) and stage for stage in stages_value
        ):
            raise ValueError(f"invalid workflow stages: {case_id}")
        stages = tuple(stages_value)
        executions_value = case.get("executions")
        if not isinstance(executions_value, list) or len(executions_value) != len(
            lanes
        ):
            raise ValueError(f"execution count mismatch: {case_id}")
        execution_lanes: set[str] = set()
        observations: dict[str, LaneObservation] = {}
        for execution_value in executions_value:
            execution = _mapping(execution_value, "execution")
            lane = _string(execution, "lane", "execution")
            if lane not in lanes or lane in execution_lanes:
                raise ValueError(f"invalid execution lane: {case_id}:{lane}")
            execution_lanes.add(lane)
            result_path = _canonical_relative_path(
                _string(execution, "result_directory", "execution"),
                "execution.result_directory",
            )
            if result_path != PurePosixPath(case_id, lane):
                raise ValueError(f"unexpected result directory: {result_path}")
            if execution.get("returncode") != 0:
                raise ValueError(f"nonzero child receipt: {case_id}:{lane}")
            parent_peak = execution.get("parent_peak_rss_bytes")
            if (
                isinstance(parent_peak, bool)
                or not isinstance(parent_peak, int)
                or parent_peak <= 0
            ):
                raise ValueError(f"missing parent-observed RSS: {case_id}:{lane}")
            observation = load_lane_observation(run_directory / case_id / lane)
            if observation.lane != lane:
                raise ValueError(f"lane identity mismatch: {case_id}:{lane}")
            if observation.scale != scale_value:
                raise ValueError(f"lane scale mismatch: {case_id}:{lane}")
            if observation.input_fingerprint != expected_input:
                raise ValueError(f"input fingerprint mismatch: {case_id}:{lane}")
            if observation.configuration_fingerprint != expected_configuration:
                raise ValueError(
                    f"configuration fingerprint mismatch: {case_id}:{lane}"
                )
            if observation.completed_workflow_stages != stages:
                raise ValueError(f"workflow stage mismatch: {case_id}:{lane}")
            provenance = observation.provenance
            if provenance is None:
                raise ValueError(f"missing provenance: {case_id}:{lane}")
            if provenance.repository_commit != repository_commit:
                raise ValueError(f"repository commit mismatch: {case_id}:{lane}")
            validate_sources_current(repo_root, provenance.executed_sources)
            if authoritative and not provenance.authoritative:
                raise ValueError(
                    f"non-authoritative lane in authoritative run: {case_id}:{lane}"
                )
            if (
                provenance.steady_state_memory_measured
                or provenance.memory_measurement_scope
                != "combined import, compile/warmup, and one bounded execution"
            ):
                raise ValueError(f"invalid memory measurement scope: {case_id}:{lane}")
            if lane.startswith("jax-") and not (
                provenance.measurement_synchronization.startswith(
                    "jax.block_until_ready"
                )
            ):
                raise ValueError(
                    f"missing JAX synchronization receipt: {case_id}:{lane}"
                )
            if lane == "jax-gpu":
                policy = provenance.lane_environment_policy
                if (
                    observation.backend_mode != "jax_gpu_parity"
                    or observation.platform != "gpu"
                    or observation.precision != "fp64"
                    or policy.get("SIMSOPT_JAX_TRANSFER_GUARD") != "disallow"
                    or policy.get("JAX_TRANSFER_GUARD") != "disallow"
                    or set(provenance.jax_effective_transfer_guards.values())
                    != {"disallow"}
                    or set(provenance.jax_effective_transfer_guards)
                    != {"device_to_device", "device_to_host", "host_to_device"}
                    or not provenance.devices
                    or any(device.platform != "gpu" for device in provenance.devices)
                ):
                    raise ValueError(f"invalid strict GPU receipt: {case_id}")
            observations[lane] = observation
            lane_receipt_count += 1
        comparisons_value = case.get("comparisons")
        if not isinstance(comparisons_value, list) or not comparisons_value:
            raise ValueError(f"case has no comparisons: {case_id}")
        for comparison_value in comparisons_value:
            comparison = _mapping(comparison_value, "comparison")
            if not _boolean(comparison, "passed", "comparison"):
                raise ValueError(f"failed comparison in passing case: {case_id}")
            comparison_count += 1
        relationship = relationships_by_case.get(case_id)
        if relationship is None:
            raise ValueError(f"summary names unknown executable case: {case_id}")
        recomputed = arbitrate(
            relationship.comparison_routes,
            observations,
            required_lanes=frozenset(lanes),
            expected_workflow_stages=relationship.workflow_stages,
        )
        if recomputed.verdict != "pass":
            raise ValueError(f"recomputed comparison verdict is not pass: {case_id}")
        recomputed_payload = [
            {
                "phase": comparison.phase,
                "observable": comparison.observable,
                "lane_pair": comparison.lane_pair,
                "passed": comparison.passed,
                "tolerance_bucket": comparison.tolerance_bucket,
                "diagnostic": comparison.diagnostic,
            }
            for comparison in recomputed.comparisons
        ]
        if comparisons_value != recomputed_payload:
            raise ValueError(f"stored comparisons differ from recomputation: {case_id}")
    return AuditResult(
        run_id=run_id,
        case_count=len(case_ids),
        lane_receipt_count=lane_receipt_count,
        comparison_count=comparison_count,
        authoritative=authoritative,
        verdict="pass",
    )


def main(argv: list[str] | None = None) -> int:
    """Audit one published run and print its compact validated receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--require-authoritative", action="store_true")
    args = parser.parse_args(argv)
    result = audit_published_run(
        args.run,
        repo_root=args.repo_root,
        require_authoritative=args.require_authoritative,
    )
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
