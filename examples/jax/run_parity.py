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
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from repo_bootstrap import bootstrap_local_simsopt

bootstrap_local_simsopt(_REPO_ROOT / "src")

from examples.jax.manifest_runtime import load_runtime_contract_pair
from examples.jax.outer_optimizer_policy import (
    policy_owns_parity_case,
    validate_ready_example_policy,
)
from examples.jax.parity.arbiter import (
    QUALITY_BAND_VERDICT,
    LaneObservation,
    LaneOutcomeRejection,
    arbitrate,
)
from examples.jax.parity.artifacts import canonical_json_bytes, write_bytes_exclusive
from examples.jax.parity.cases import get_case, implemented_case_ids
from examples.jax.parity.contracts import EndStateResult
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
from examples.jax.parity.runner import (
    ChildExecution,
    RunnerError,
    execute_case_lanes,
)
from simsopt_jax.examples import EXECUTION_SCALES, ExecutionScale

import jax
import jax.numpy as jnp

_LANES = frozenset({"native-cpu", "jax-cpu", "jax-gpu"})

#: How to launch the runner so its own process builds bundles in float64.
#: Names only what the check below reads: JAX's own x64 flag in THIS process.
_FLOAT64_LAUNCH_HINT = (
    "set JAX_ENABLE_X64=1 in the runner process's own environment before it "
    "imports JAX (for example by launching it through "
    ".artifacts/official-scope-cleanup-20260919/run-python.sh); "
    "examples/jax/_lane_environment.py sets that flag for the lane "
    "subprocesses only, and it is the only setting this check reads"
)


def require_input_construction_float64() -> None:
    """Fail closed unless this process builds input bundles in float64.

    ``case.create_input`` runs in the runner process, not in a lane subprocess,
    so it never sees ``examples/jax/_lane_environment.py``'s
    ``JAX_ENABLE_X64=1``. With x64 off, the JAX-jitted ``simsopt.geo`` terms a
    case's construction may call run in float32, and every lane then consumes
    the one wrong bundle and agrees with the others: the
    ``native-permanent-magnet-qa`` TF-coil pre-optimization froze after two
    L-BFGS-B iterations on a float32-quantized objective, and all three lanes
    solved a permanent-magnet problem built from the wrong coil currents
    (``.artifacts/official-mirror-closure-20260919/investigations/pm-qa-coils/REPORT.md``).
    This is a loud check at the boundary, never a silent ``config.update``: the
    x64 policy of ``simsopt.geo`` is owned elsewhere and float32 modes depend
    on it.
    """

    dtype = jnp.zeros(1).dtype
    if not bool(jax.config.read("jax_enable_x64")) or dtype != jnp.float64:
        raise RunnerError(
            "parity input construction requires JAX float64 in the runner "
            f"process: jax_enable_x64={bool(jax.config.read('jax_enable_x64'))}, "
            f"default array dtype={dtype}; {_FLOAT64_LAUNCH_HINT}"
        )


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
    parser.add_argument("--scale", choices=EXECUTION_SCALES)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _parse_arguments(argv: list[str] | None) -> argparse.Namespace:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.smoke and args.scale == "native_default":
        parser.error("--smoke cannot be combined with --scale native_default")
    if args.scale is None:
        args.scale = "bounded"
    return args


def _run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{secrets.token_hex(4)}"


def _lane_completed_workflow_stages(
    observations: dict[str, LaneObservation],
) -> dict[str, list[str]]:
    """Report what each lane itself completed, disagreement included."""
    return {
        lane: list(observations[lane].completed_workflow_stages)
        for lane in sorted(observations)
    }


def _agreed_completed_workflow_stages(
    observations: dict[str, LaneObservation],
) -> list[str]:
    """Report the one stage sequence every lane completed, else nothing.

    ``[]`` means the lanes did not agree OR no lane completed a stage; the
    per-lane record written beside it (``lane_completed_workflow_stages``)
    is what distinguishes the two, so neither state is erased.
    """
    stages = {
        tuple(observation.completed_workflow_stages)
        for observation in observations.values()
    }
    if len(stages) != 1:
        return []
    return list(next(iter(stages)))


def _validate_case_lane_provenance(
    *,
    case_id: str,
    observations: dict[str, LaneObservation],
    repository_state,
    repo_root: Path,
) -> bool:
    repository_changed_during_run = False
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
            or provenance.tracked_diff_sha256 != repository_state.tracked_diff_sha256
            or provenance.untracked_files != repository_state.untracked_files
        )
        repository_changed_during_run = (
            repository_changed_during_run or lane_repository_changed
        )
        if not repository_state.repository_dirty and lane_repository_changed:
            raise RunnerError(
                f"{case_id} clean repository changed during parity execution"
            )
    return repository_changed_during_run


def _case_summary_record(
    *,
    case_id: str,
    relationship,
    scale: ExecutionScale,
    bundle,
    observations: dict[str, LaneObservation],
    executions: tuple[ChildExecution, ...],
    paths,
    repository_state,
    verdict: str,
    comparisons: tuple[object, ...],
    arbitration_rejection: str | None = None,
    work_budget_admitted: bool = False,
    quality_band_results: tuple[object, ...] = (),
    admitted_terminal_lanes: tuple[tuple[str, str], ...] = (),
    end_state_results: tuple[EndStateResult, ...] = (),
) -> dict[str, object]:
    case_authoritative = all(
        observation.provenance is not None and observation.provenance.authoritative
        for observation in observations.values()
    )
    case_record: dict[str, object] = {
        "case_id": case_id,
        "jax_example_id": relationship.jax_example_id,
        "native_source": relationship.native_source,
        "classification": relationship.classification,
        "classification_reason": relationship.classification_reason,
        "scale_tier": scale,
        "oracle_kind": relationship.oracle_kind,
        "cost_tier": relationship.cost_tier,
        "omitted_scientific_stages": list(relationship.omitted_scientific_stages),
        "excluded_teaching_stages": list(relationship.excluded_teaching_stages),
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
        # One source on every path: what the lanes OBSERVABLY completed. On a
        # certifying path the arbiter has already required each lane to equal
        # the relationship's declared stages (arbiter.py `_validate_lanes`), so
        # this is the declared contract too, without a second writer for it.
        "completed_workflow_stages": _agreed_completed_workflow_stages(observations),
        "lane_completed_workflow_stages": _lane_completed_workflow_stages(observations),
        "verdict": verdict,
        "comparisons": [
            {
                "phase": comparison.phase,
                "observable": comparison.observable,
                "lane_pair": comparison.lane_pair,
                "passed": comparison.passed,
                "tolerance_bucket": comparison.tolerance_bucket,
                "diagnostic": comparison.diagnostic,
            }
            for comparison in comparisons
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
    if arbitration_rejection is not None:
        case_record["arbitration_rejection"] = arbitration_rejection
    if work_budget_admitted:
        case_record["terminal_contract"] = "work-budget"
    if admitted_terminal_lanes:
        case_record["admitted_terminal_lanes"] = [
            {"lane": lane, "raw_status": raw_status}
            for lane, raw_status in admitted_terminal_lanes
        ]
    if quality_band_results:
        case_record["quality_band"] = [
            {
                "lane": band_result.lane,
                "observable": band_result.observable,
                "max_value": band_result.max_value,
                "observed_value": band_result.observed_value,
                "passed": band_result.passed,
            }
            for band_result in quality_band_results
        ]
    if end_state_results:
        case_record["upstream_end_states"] = [
            {
                "lane": end_state_result.lane,
                "matched_draws": list(end_state_result.matched_draws),
                "passed": end_state_result.passed,
            }
            for end_state_result in end_state_results
        ]
    return case_record


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
    args = _parse_arguments(argv)
    require_input_construction_float64()
    scale: ExecutionScale = args.scale
    repo_root = _REPO_ROOT
    contract_pair = load_runtime_contract_pair(
        repo_root / "examples" / "jax" / "manifest.json",
        repo_root / "examples" / "jax" / "parity_manifest.json",
        repo_root=repo_root,
    )
    examples_by_id = {example.id: example for example in contract_pair.examples}
    parity_manifest = contract_pair.parity
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
            relationships = tuple(
                relationship
                for relationship in parity_manifest.all_relationships
                if relationship.case_id == case_id
            )
            if len(relationships) != 1:
                raise RunnerError(
                    f"case {case_id} must own exactly one executable relationship"
                )
            relationship = relationships[0]
            example = examples_by_id[relationship.jax_example_id]
            validate_ready_example_policy(
                example.outer_optimizer_policy,
                example_id=example.id,
                example_path=example.path,
            )
            if (
                example.outer_optimizer_policy is not None
                and not policy_owns_parity_case(
                    example.outer_optimizer_policy,
                    case_id=case_id,
                    example_id=example.id,
                )
            ):
                raise RunnerError(
                    "outer optimizer policy belongs to a different parity case"
                )
            if scale not in relationship.supported_scales:
                raise RunnerError(
                    f"case {case_id} declares scale_tier "
                    f"{relationship.scale_tier!r}; requested scale {scale!r} "
                    "is unsupported"
                )
            relationship = relationship.resolve_scale(scale)
            input_root = paths.partial / case_id / "inputs"
            bundle = case.create_input(input_root, scale)
            executions, observations = execute_case_lanes(
                case_id=case_id,
                lanes=args.lanes,
                input_bundle_path=input_root / "input_bundle.json",
                run_directory=paths.partial,
                repo_root=repo_root,
                base_environment=os.environ,
                python_executable=sys.executable,
                scale=scale,
            )
            try:
                arbitration = arbitrate(
                    relationship.comparison_routes,
                    observations,
                    required_lanes=frozenset(args.lanes),
                    expected_workflow_stages=relationship.workflow_stages,
                    case_id=case_id,
                    example_id=relationship.jax_example_id,
                    outer_optimizer_policy=example.outer_optimizer_policy,
                    quality_band=case.quality_band(scale),
                    work_budget_contract=case.work_budget_contract,
                    admitted_terminal_outcomes=(
                        case.native_default_admitted_terminal_outcomes
                        if scale == "native_default"
                        else ()
                    ),
                    upstream_end_states=case.end_states(scale),
                )
            except LaneOutcomeRejection as error:
                rejection = str(error).strip()
                if not rejection:
                    raise
                repository_changed_during_run = (
                    repository_changed_during_run
                    or _validate_case_lane_provenance(
                        case_id=case_id,
                        observations=observations,
                        repository_state=repository_state,
                        repo_root=repo_root,
                    )
                )
                summaries.append(
                    _case_summary_record(
                        case_id=case_id,
                        relationship=relationship,
                        scale=scale,
                        bundle=bundle,
                        observations=observations,
                        executions=executions,
                        paths=paths,
                        repository_state=repository_state,
                        verdict="fail",
                        comparisons=(),
                        arbitration_rejection=rejection,
                    )
                )
                continue
            repository_changed_during_run = (
                repository_changed_during_run
                or _validate_case_lane_provenance(
                    case_id=case_id,
                    observations=observations,
                    repository_state=repository_state,
                    repo_root=repo_root,
                )
            )
            summaries.append(
                _case_summary_record(
                    case_id=case_id,
                    relationship=relationship,
                    scale=scale,
                    bundle=bundle,
                    observations=observations,
                    executions=executions,
                    paths=paths,
                    repository_state=repository_state,
                    verdict=arbitration.verdict,
                    comparisons=arbitration.comparisons,
                    work_budget_admitted=arbitration.work_budget_admitted,
                    quality_band_results=arbitration.quality_band_results,
                    admitted_terminal_lanes=arbitration.admitted_terminal_lanes,
                    end_state_results=arbitration.end_state_results,
                )
            )
        validate_sources_current(repo_root, explicit_sources)
        final_repository_state = collect_repository_state(repo_root)
        final_repository_changed = final_repository_state != repository_state
        repository_changed_during_run = (
            repository_changed_during_run or final_repository_changed
        )
        if not repository_state.repository_dirty and final_repository_changed:
            raise RunnerError("clean repository changed during parity execution")
        case_verdicts = tuple(str(item["verdict"]) for item in summaries)
        verdict = (
            "fail"
            if not case_verdicts
            or any(
                value not in ("pass", QUALITY_BAND_VERDICT) for value in case_verdicts
            )
            else QUALITY_BAND_VERDICT
            if QUALITY_BAND_VERDICT in case_verdicts
            else "pass"
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
                    "schema_version": 2,
                    "manifest_schema_version": contract_pair.version_pair[0],
                    "parity_manifest_schema_version": contract_pair.version_pair[1],
                    # Retired flag kept for summary schema 2; no legacy reader exists.
                    "used_legacy_manifest_adapter": False,
                    "run_id": paths.run_id,
                    "lanes": list(args.lanes),
                    "scale": scale,
                    "smoke": scale == "bounded",
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
        if verdict == "fail":
            mark_run_failed(paths, "one or more parity cases failed")
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
