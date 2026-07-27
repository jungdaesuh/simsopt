"""Run typed native/JAX example parity cases in isolated subprocesses."""

from __future__ import annotations

import argparse
import dataclasses
import os
import secrets
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SOURCE_ROOT = _REPO_ROOT / "src"
for import_root in (str(_SOURCE_ROOT), str(_REPO_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from examples.jax._manifest import load_manifest
from examples.jax.parity._manifest import load_parity_manifest
from examples.jax.parity.arbiter import arbitrate
from examples.jax.parity.artifacts import canonical_json_bytes, write_bytes_exclusive
from examples.jax.parity.cases import get_case, implemented_case_ids
from examples.jax.parity.provenance import (
    REQUIRED_PROVENANCE_SOURCE_PATHS,
    collect_explicit_sources,
    collect_repository_state,
    validate_sources_current,
)
from examples.jax.parity.publication import (
    begin_run,
    mark_run_failed,
    publish_run,
)
from examples.jax.parity.runner import RunnerError, execute_case_lanes

_LANES = frozenset({"native-cpu", "jax-cpu", "jax-gpu"})


def _lane_tuple(value: str) -> tuple[str, ...]:
    lanes = tuple(value.split(","))
    if not lanes or len(lanes) != len(set(lanes)) or set(lanes) - _LANES:
        raise argparse.ArgumentTypeError(
            "lanes must be a unique comma-separated subset of "
            "native-cpu,jax-cpu,jax-gpu"
        )
    return lanes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", required=True)
    parser.add_argument("--lanes", type=_lane_tuple, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _selected_cases(
    requested: list[str], applicable_case_ids: tuple[str, ...]
) -> tuple[str, ...]:
    if requested == ["all-applicable"]:
        missing = set(applicable_case_ids) - set(implemented_case_ids())
        if missing:
            raise RunnerError(
                "all-applicable requested before cases are implemented: "
                f"{sorted(missing)}"
            )
        return applicable_case_ids
    if "all-applicable" in requested or len(requested) != len(set(requested)):
        raise RunnerError("case selection must be unique or exactly all-applicable")
    for case_id in requested:
        get_case(case_id)
    return tuple(requested)


def main(argv: list[str] | None = None) -> int:
    """Execute requested cases and publish only a complete passing run."""
    args = _parser().parse_args(argv)
    repo_root = _REPO_ROOT
    examples_manifest = load_manifest(
        repo_root / "examples" / "jax" / "manifest.json", repo_root=repo_root
    )
    parity_manifest = load_parity_manifest(
        repo_root / "examples" / "jax" / "parity_manifest.json",
        examples_manifest=examples_manifest,
        repo_root=repo_root,
    )
    repository_state = collect_repository_state(repo_root)
    explicit_sources = collect_explicit_sources(
        repo_root,
        REQUIRED_PROVENANCE_SOURCE_PATHS,
    )
    applicable_case_ids = tuple(
        dict.fromkeys(
            relationship.case_id
            for relationship in parity_manifest.relationships
            if relationship.case_id is not None
        )
    )
    case_ids = _selected_cases(args.case, applicable_case_ids)
    paths = begin_run(args.artifact_root.resolve(), _run_id())
    summaries: list[dict[str, object]] = []
    repository_changed_during_run = False
    try:
        for case_id in case_ids:
            case = get_case(case_id)
            input_root = paths.partial / case_id / "inputs"
            bundle = case.create_input(input_root, args.smoke)
            executions, observations = execute_case_lanes(
                case_id=case_id,
                lanes=args.lanes,
                input_bundle_path=input_root / "input_bundle.json",
                run_directory=paths.partial,
                repo_root=repo_root,
                base_environment=os.environ,
                python_executable=sys.executable,
                smoke=args.smoke,
            )
            relationships = tuple(
                relationship
                for relationship in parity_manifest.relationships
                if relationship.case_id == case_id
            )
            if len(relationships) != 1:
                raise RunnerError(
                    f"case {case_id} must own exactly one executable relationship"
                )
            relationship = relationships[0]
            arbitration = arbitrate(
                relationship.comparison_routes,
                observations,
                required_lanes=frozenset(args.lanes),
                expected_workflow_stages=relationship.workflow_stages,
            )
            for observation in observations.values():
                provenance = observation.provenance
                if provenance is None:
                    raise RunnerError(f"{case_id} lane omitted provenance")
                if provenance.repository_commit != repository_state.repository_commit:
                    raise RunnerError(
                        f"{case_id} repository commit changed during parity execution"
                    )
                try:
                    validate_sources_current(repo_root, provenance.executed_sources)
                except ValueError as error:
                    raise RunnerError(
                        f"{case_id} invalid source provenance: {error}"
                    ) from error
                lane_repository_changed = (
                    provenance.repository_dirty != repository_state.repository_dirty
                    or provenance.tracked_diff_sha256
                    != repository_state.tracked_diff_sha256
                    or provenance.untracked_files != repository_state.untracked_files
                )
                repository_changed_during_run = (
                    repository_changed_during_run or lane_repository_changed
                )
                if not repository_state.repository_dirty and lane_repository_changed:
                    raise RunnerError(
                        f"{case_id} clean repository changed during parity execution"
                    )
            case_authoritative = all(
                observation.provenance is not None
                and observation.provenance.authoritative
                for observation in observations.values()
            )
            summaries.append(
                {
                    "case_id": case_id,
                    "jax_example_id": relationship.jax_example_id,
                    "native_source": relationship.native_source,
                    "classification": relationship.classification,
                    "classification_reason": relationship.classification_reason,
                    "scale_tier": relationship.scale_tier,
                    "oracle_kind": relationship.oracle_kind,
                    "cost_tier": relationship.cost_tier,
                    "omitted_scientific_stages": list(
                        relationship.omitted_scientific_stages
                    ),
                    "excluded_teaching_stages": list(
                        relationship.excluded_teaching_stages
                    ),
                    "authoritative": case_authoritative,
                    "repository_changed_during_run": any(
                        observation.provenance is not None
                        and (
                            observation.provenance.repository_dirty
                            != repository_state.repository_dirty
                            or observation.provenance.tracked_diff_sha256
                            != repository_state.tracked_diff_sha256
                            or observation.provenance.untracked_files
                            != repository_state.untracked_files
                        )
                        for observation in observations.values()
                    ),
                    "input_fingerprint": bundle.input_fingerprint,
                    "configuration_fingerprint": bundle.configuration_fingerprint,
                    "completed_workflow_stages": list(relationship.workflow_stages),
                    "verdict": arbitration.verdict,
                    "comparisons": [
                        {
                            "phase": comparison.phase,
                            "observable": comparison.observable,
                            "lane_pair": comparison.lane_pair,
                            "passed": comparison.passed,
                            "tolerance_bucket": comparison.tolerance_bucket,
                            "diagnostic": comparison.diagnostic,
                        }
                        for comparison in arbitration.comparisons
                    ],
                    "executions": [
                        {
                            "lane": execution.lane,
                            "command": list(execution.command),
                            "stdout": execution.stdout,
                            "stderr": execution.stderr,
                            "returncode": execution.returncode,
                            "elapsed_seconds": execution.elapsed_seconds,
                            "parent_peak_rss_bytes": execution.parent_peak_rss_bytes,
                            "result_directory": str(
                                execution.result_directory.relative_to(paths.partial)
                            ),
                        }
                        for execution in executions
                    ],
                }
            )
        validate_sources_current(repo_root, explicit_sources)
        final_repository_state = collect_repository_state(repo_root)
        final_repository_changed = final_repository_state != repository_state
        repository_changed_during_run = (
            repository_changed_during_run or final_repository_changed
        )
        if not repository_state.repository_dirty and final_repository_changed:
            raise RunnerError("clean repository changed during parity execution")
        verdict = (
            "pass"
            if summaries and all(item["verdict"] == "pass" for item in summaries)
            else "fail"
        )
        authoritative = not repository_state.repository_dirty and all(
            source.git_blob_id is not None for source in explicit_sources
        )
        authoritative = authoritative and all(
            case_summary["authoritative"] for case_summary in summaries
        )
        summary_path = paths.partial / "summary.json"
        write_bytes_exclusive(
            paths.partial,
            summary_path.name,
            canonical_json_bytes(
                {
                    "schema_version": 1,
                    "run_id": paths.run_id,
                    "lanes": list(args.lanes),
                    "smoke": args.smoke,
                    "authoritative": authoritative,
                    "repository_commit": repository_state.repository_commit,
                    "repository_dirty": repository_state.repository_dirty,
                    "repository_changed_during_run": repository_changed_during_run,
                    "tracked_diff_sha256": repository_state.tracked_diff_sha256,
                    "untracked_files": list(repository_state.untracked_files),
                    "explicit_sources": [
                        dataclasses.asdict(source) for source in explicit_sources
                    ],
                    "verdict": verdict,
                    "cases": summaries,
                }
            ),
        )
        if verdict != "pass":
            mark_run_failed(paths, "one or more parity comparisons failed")
            return 1
        published = publish_run(paths)
        print(published)
        return 0
    except Exception as error:  # noqa: BLE001 - preserve every failed-run receipt
        mark_run_failed(paths, str(error))
        print(f"parity run failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
