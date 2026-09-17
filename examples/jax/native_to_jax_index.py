"""Generate the source-owned native-to-JAX example index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from examples.jax.manifest_contracts_v3 import (
    JaxExamplesManifestV3,
    load_manifest_contract_pair_documents,
)
from examples.jax.parity.audit import audit_published_run
from examples.jax.parity.cases import get_case
from examples.jax.parity.provenance import validate_authoritative_provenance
from examples.jax.parity.receipts import load_lane_observation

JAX_EXAMPLES_DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = JAX_EXAMPLES_DIRECTORY.parents[1]
INDEX_PATH = JAX_EXAMPLES_DIRECTORY / "NATIVE_TO_JAX_INDEX.md"
MANIFEST_PATH = JAX_EXAMPLES_DIRECTORY / "manifest.json"
PARITY_MANIFEST_PATH = JAX_EXAMPLES_DIRECTORY / "parity_manifest.json"
AUTHORITY_EVIDENCE_PATH = JAX_EXAMPLES_DIRECTORY / "authority_evidence.json"


@dataclass(frozen=True, slots=True)
class AuthorityEvidence:
    """Compact, tracked pointer to one authoritative local run."""

    run_id: str
    repository_commit: str
    summary_sha256: str
    scale: str
    verdict: str
    case_count: int
    lane_receipt_count: int
    comparison_count: int
    case_ids: frozenset[str]
    native_default_status: str | None
    evidence_scope: str
    qualification: str = "raw summary and lane receipts require local verification"


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


_RUN_FIELDS = {
    "run_id",
    "repository_commit",
    "summary_sha256",
    "scale",
    "verdict",
    "case_count",
    "lane_receipt_count",
    "comparison_count",
    "case_ids",
    "evidence_scope",
}


def _parse_authority_run(document: object, *, legacy: bool) -> AuthorityEvidence:
    if not isinstance(document, dict):
        raise TypeError("authority evidence run must be a JSON object")
    expected_fields = _RUN_FIELDS | (
        {"schema_version", "native_default_status"} if legacy else set()
    )
    if set(document) not in (expected_fields, expected_fields | {"qualification"}) or (
        legacy and document["schema_version"] != 1
    ):
        raise ValueError("authority evidence run has invalid fields")
    string_fields = (
        "run_id",
        "repository_commit",
        "summary_sha256",
        "scale",
        "verdict",
        "evidence_scope",
    )
    if not all(isinstance(document[field], str) for field in string_fields) or (
        not legacy and not all(document[field] for field in string_fields)
    ):
        raise ValueError("authority evidence identity fields must be strings")
    if legacy and document["scale"] != "bounded":
        raise ValueError("authority evidence schema v1 requires bounded scale")
    if not legacy and document["scale"] not in ("bounded", "native_default"):
        raise ValueError("authority evidence scale is invalid")
    if legacy and document["verdict"] != "pass":
        raise ValueError("authority evidence schema v1 requires pass verdict")
    if not legacy and document["verdict"] not in ("pass", "quality-band"):
        raise ValueError("authority evidence verdict is invalid")
    if (
        not legacy
        and document["verdict"] == "quality-band"
        and document["scale"] != "native_default"
    ):
        raise ValueError("quality-band evidence requires native_default scale")
    if legacy and document["native_default_status"] != "not_run":
        raise ValueError("authority evidence schema v1 requires native_default not_run")
    if document["evidence_scope"] != "local_only":
        raise ValueError("authority evidence requires local_only scope")
    count_fields = ("case_count", "lane_receipt_count", "comparison_count")
    if not all(
        isinstance(document[field], int) and not isinstance(document[field], bool)
        for field in count_fields
    ):
        raise ValueError("authority evidence counts must be integers")
    if not all(document[field] > 0 for field in count_fields):
        raise ValueError("authority evidence counts must be positive")
    case_ids_value = document["case_ids"]
    if not isinstance(case_ids_value, list) or not all(
        isinstance(case_id, str) and bool(case_id) for case_id in case_ids_value
    ):
        raise ValueError("authority evidence case_ids must be strings")
    case_ids = frozenset(case_ids_value)
    if len(case_ids) != len(case_ids_value) or len(case_ids) != document["case_count"]:
        raise ValueError("authority evidence case_ids do not match case_count")
    qualification = document.get(
        "qualification", "raw summary and lane receipts require local verification"
    )
    if not isinstance(qualification, str) or not qualification:
        raise ValueError("authority evidence qualification must be a non-empty string")
    return AuthorityEvidence(
        run_id=document["run_id"],
        repository_commit=document["repository_commit"],
        summary_sha256=document["summary_sha256"],
        scale=document["scale"],
        verdict=document["verdict"],
        case_count=document["case_count"],
        lane_receipt_count=document["lane_receipt_count"],
        comparison_count=document["comparison_count"],
        case_ids=case_ids,
        native_default_status=document["native_default_status"] if legacy else None,
        evidence_scope=document["evidence_scope"],
        qualification=qualification,
    )


def _load_authority_evidence(path: Path) -> AuthorityEvidence:
    """Load the historical version 1 authority record unchanged."""
    return _parse_authority_run(_load_json(path), legacy=True)


def _load_authority_evidence_runs(
    path: Path,
) -> tuple[int, tuple[AuthorityEvidence, ...]]:
    """Load one historical record or a version 2 list of scoped run receipts."""
    document = _load_json(path)
    if not isinstance(document, dict):
        raise TypeError("authority evidence must be a JSON object")
    if document.get("schema_version") == 1:
        return 1, (_parse_authority_run(document, legacy=True),)
    if set(document) != {"schema_version", "runs"} or document["schema_version"] != 2:
        raise ValueError("unsupported authority evidence schema")
    runs_value = document["runs"]
    if not isinstance(runs_value, list) or not runs_value:
        raise ValueError("authority evidence runs must be a non-empty array")
    runs = tuple(_parse_authority_run(run, legacy=False) for run in runs_value)
    if len({run.run_id for run in runs}) != len(runs):
        raise ValueError("authority evidence run IDs must be unique")
    return 2, runs


def verify_authority_summary(
    evidence: AuthorityEvidence,
    summary_path: Path,
) -> None:
    """Recompute the compact authority record from a retained run summary."""
    summary_bytes = summary_path.read_bytes()
    if hashlib.sha256(summary_bytes).hexdigest() != evidence.summary_sha256:
        raise RuntimeError("authority summary SHA-256 does not match its record")
    summary = json.loads(summary_bytes)
    if not isinstance(summary, dict):
        raise TypeError("authority summary must be a JSON object")
    identity = (
        summary.get("run_id"),
        summary.get("repository_commit"),
        summary.get("scale"),
        summary.get("verdict"),
    )
    if identity != (
        evidence.run_id,
        evidence.repository_commit,
        evidence.scale,
        evidence.verdict,
    ):
        raise RuntimeError("authority summary identity does not match its record")
    if summary.get("authoritative") is not True:
        raise RuntimeError("authority summary is not authoritative")
    cases = summary.get("cases")
    lanes = summary.get("lanes")
    if not isinstance(cases, list) or not isinstance(lanes, list):
        raise TypeError("authority summary cases and lanes must be arrays")
    if not lanes or not all(isinstance(lane, str) for lane in lanes):
        raise RuntimeError("authority summary lanes must be non-empty strings")
    expected_lanes = set(lanes)
    if len(expected_lanes) != len(lanes):
        raise RuntimeError("authority summary lanes must be unique")
    case_ids: set[str] = set()
    comparison_count = 0
    lane_receipt_count = 0
    quality_band_case_count = 0
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("authority summary case must be an object")
        case_id = case.get("case_id")
        comparisons = case.get("comparisons")
        executions = case.get("executions")
        if (
            not isinstance(case_id, str)
            or not isinstance(comparisons, list)
            or not comparisons
            or not isinstance(executions, list)
            or case.get("authoritative") is not True
            or case.get("verdict") not in ("pass", "quality-band")
            or case.get("scale_tier") != evidence.scale
        ):
            raise RuntimeError("authority summary case is incomplete")
        if not all(isinstance(comparison, dict) for comparison in comparisons):
            raise RuntimeError("authority summary contains a malformed comparison")
        if case["verdict"] == "pass" and not all(
            comparison.get("passed") is True for comparison in comparisons
        ):
            raise RuntimeError("authority summary contains a failed comparison")
        if case["verdict"] == "quality-band":
            quality_band_case_count += 1
            quality_band = case.get("quality_band")
            declared_band = get_case(case_id).native_default_quality_band
            if (
                evidence.scale != "native_default"
                or declared_band is None
                or not isinstance(quality_band, list)
                or expected_lanes != {"native-cpu", "jax-cpu", "jax-gpu"}
                or len(quality_band) != len(expected_lanes)
            ):
                raise RuntimeError("authority summary quality band is incomplete")
            band_lanes: set[str] = set()
            for result in quality_band:
                if not isinstance(result, dict) or set(result) != {
                    "lane",
                    "observable",
                    "max_value",
                    "observed_value",
                    "passed",
                }:
                    raise RuntimeError("authority summary quality band is incomplete")
                lane = result["lane"]
                observable = result["observable"]
                limit = result["max_value"]
                observed = result["observed_value"]
                if (
                    not isinstance(lane, str)
                    or lane not in expected_lanes
                    or lane in band_lanes
                    or not isinstance(observable, str)
                    or observable != declared_band.observable
                    or not isinstance(limit, float)
                    or not math.isfinite(limit)
                    or limit != declared_band.max_value
                    or not isinstance(observed, float)
                    or not math.isfinite(observed)
                    or result["passed"] is not True
                    or observed > limit
                ):
                    raise RuntimeError("authority summary quality band is incomplete")
                band_lanes.add(lane)
            if band_lanes != expected_lanes:
                raise RuntimeError("authority summary quality band is incomplete")
        execution_lanes = {
            execution.get("lane")
            for execution in executions
            if isinstance(execution, dict) and execution.get("returncode") == 0
        }
        if execution_lanes != expected_lanes or len(executions) != len(lanes):
            raise RuntimeError("authority summary case has incomplete lane executions")
        case_ids.add(case_id)
        comparison_count += len(comparisons)
        lane_receipt_count += len(executions)
    if (evidence.verdict == "quality-band") != (quality_band_case_count > 0):
        raise RuntimeError("authority summary verdict mislabels quality-band evidence")
    if (
        case_ids != evidence.case_ids
        or len(cases) != evidence.case_count
        or lane_receipt_count != evidence.lane_receipt_count
        or comparison_count != evidence.comparison_count
    ):
        raise RuntimeError("authority summary counts do not match its record")

    references = subprocess.run(
        (
            "git",
            "for-each-ref",
            "--format=%(refname)",
            f"--contains={evidence.repository_commit}",
        ),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if references.returncode != 0 or not references.stdout.strip():
        raise RuntimeError("authority source commit is not reachable from a named ref")
    for case in cases:
        for execution in case["executions"]:
            relative = execution.get("result_directory")
            if (
                not isinstance(relative, str)
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
            ):
                raise RuntimeError(
                    "authority summary lacks retained lane receipt paths"
                )
            observation = load_lane_observation(summary_path.parent / relative)
            provenance = observation.provenance
            if (
                provenance is None
                or not provenance.authoritative
                or provenance.repository_commit != evidence.repository_commit
            ):
                raise RuntimeError("authority lane provenance is not source-bound")
            validate_authoritative_provenance(REPO_ROOT, provenance)
            for source in provenance.executed_sources:
                if source.git_blob_id is None:
                    if source.path not in provenance.generated_source_bindings:
                        raise RuntimeError(
                            "authority lane has an unbound generated source"
                        )
                    continue
                committed = subprocess.run(
                    ("git", "show", f"{evidence.repository_commit}:{source.path}"),
                    cwd=REPO_ROOT,
                    capture_output=True,
                    check=True,
                ).stdout
                if hashlib.sha256(committed).hexdigest() != source.sha256:
                    raise RuntimeError(
                        "authority executed source differs from its reachable commit"
                    )

    audit_published_run(
        summary_path.parent, repo_root=REPO_ROOT, require_authoritative=True
    )


def authority_record_from_summary(summary_path: Path) -> dict[str, object]:
    """Build a version 2 local run record from one retained passing summary."""
    summary_bytes = summary_path.read_bytes()
    summary = json.loads(summary_bytes)
    if not isinstance(summary, dict):
        raise TypeError("authority summary must be a JSON object")
    cases = summary.get("cases")
    if not isinstance(cases, list):
        raise TypeError("authority summary cases must be an array")
    case_ids: list[str] = []
    lane_receipt_count = 0
    comparison_count = 0
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("authority summary case must be an object")
        case_id = case.get("case_id")
        executions = case.get("executions")
        comparisons = case.get("comparisons")
        if (
            not isinstance(case_id, str)
            or not isinstance(executions, list)
            or not isinstance(comparisons, list)
        ):
            raise TypeError("authority summary case has invalid receipt fields")
        case_ids.append(case_id)
        lane_receipt_count += len(executions)
        comparison_count += len(comparisons)
    record: dict[str, object] = {
        "run_id": summary.get("run_id"),
        "repository_commit": summary.get("repository_commit"),
        "summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
        "scale": summary.get("scale"),
        "verdict": summary.get("verdict"),
        "case_count": len(cases),
        "lane_receipt_count": lane_receipt_count,
        "comparison_count": comparison_count,
        "case_ids": case_ids,
        "evidence_scope": "local_only",
    }
    evidence = _parse_authority_run(record, legacy=False)
    verify_authority_summary(evidence, summary_path)
    if summary.get("verdict") == "quality-band":
        failed = sum(
            comparison["passed"] is False
            for case in cases
            for comparison in case["comparisons"]
        )
        observations = [
            load_lane_observation(summary_path.parent / execution["result_directory"])
            for case in cases
            for execution in case["executions"]
        ]
        exhausted = sum(
            observation.normalized_status == "budget_exhausted"
            for observation in observations
        )
        bands = "; ".join(
            f"{result['lane']} {result['observable']} {result['observed_value']:.17g} <= {result['max_value']:.17g}"
            for case in cases
            for result in case.get("quality_band", [])
        )
        record["qualification"] = (
            f"{failed}/{comparison_count} comparisons failed; {exhausted}/{lane_receipt_count} lanes budget-exhausted; "
            f"{bands}; endpoint quality only, no convergence or final-value equivalence"
        )
    return record


def _markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def render_native_to_jax_index(*, repo_root: Path = REPO_ROOT) -> str:
    """Render the complete index from validated manifest contracts."""
    jax_examples_directory = repo_root / "examples" / "jax"
    pair = load_manifest_contract_pair_documents(
        _load_json(jax_examples_directory / MANIFEST_PATH.name),
        _load_json(jax_examples_directory / PARITY_MANIFEST_PATH.name),
        repo_root=repo_root,
    )
    if pair.version_pair != (3, 2) or not isinstance(
        pair.examples, JaxExamplesManifestV3
    ):
        raise ValueError("native-to-JAX index requires manifest v3 and parity v2")
    authority_version, authority_runs = _load_authority_evidence_runs(
        jax_examples_directory / AUTHORITY_EVIDENCE_PATH.name
    )
    evidence = authority_runs[0]
    examples_by_id = {example.id: example for example in pair.examples.jax_examples}
    parity_by_source = {
        relationship.native_source: relationship
        for relationship in pair.parity.relationships
    }
    lines = [
        "# Native-to-JAX example index",
        "",
        (
            "Generated from `manifest.json`, `parity_manifest.json`, and "
            "`authority_evidence.json`. Do not edit this table by hand."
        ),
        "",
        "Historical local evidence (unverified):"
        if authority_version == 1
        else "Historical local evidence pointers (unverified):",
        "",
    ]
    if authority_version == 1:
        assert evidence.native_default_status is not None
        lines.extend(
            (
                (
                    f"- `{evidence.scale}`: historical {evidence.verdict} (unverified); run `{evidence.run_id}`; "
                    f"{evidence.case_count} cases / {evidence.lane_receipt_count} lanes / "
                    f"{evidence.comparison_count:,} comparisons."
                ),
                (
                    f"- Evidence revision: `{evidence.repository_commit}`; "
                    f"summary SHA-256: `{evidence.summary_sha256}`; scope: "
                    f"`{evidence.evidence_scope}`."
                ),
                f"- `native_default`: {evidence.native_default_status.replace('_', ' ')}.",
            )
        )
    else:
        for run in authority_runs:
            lines.append(
                f"- `{run.scale}`: historical {run.verdict} (unverified); run `{run.run_id}`; "
                f"{run.case_count} cases / {run.lane_receipt_count} lanes / "
                f"{run.comparison_count:,} comparisons; revision `{run.repository_commit}`; "
                f"summary SHA-256 `{run.summary_sha256}`; scope `{run.evidence_scope}`. "
                f"{run.qualification}."
            )
    lines.extend(
        (
            "",
            (
                "| Native example | JAX mirror | Classification | "
                "Runtime dependencies | Device scope | Scale | "
                + ("Latest evidence |" if authority_version == 1 else "Evidence |")
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
        )
    )
    for source in pair.examples.source_catalog:
        example = (
            examples_by_id[source.mirror_example_id]
            if source.mirror_example_id is not None
            else None
        )
        relationship = parity_by_source.get(source.source)
        mirror_path = "—" if example is None else f"`examples/jax/{example.path}`"
        classification = source.disposition
        if example is not None:
            classification = f"{source.disposition} / {example.classification}"
        dependencies = (
            ", ".join(source.dependencies.external_runtimes)
            if source.dependencies.external_runtimes
            else "none"
        )
        device_scope = "—"
        if example is not None:
            device_scope = ", ".join(
                f"{device}: {scope}"
                for device, scope in example.supported_device_scopes
            )
            if example.outer_optimizer_policy is not None:
                device_scope = "outer: CPU SciPy; " + device_scope
        scale = "—" if relationship is None else relationship.scale_tier
        latest = "not run"
        if authority_version == 1:
            if (
                relationship is not None
                and relationship.scale_tier == evidence.scale
                and relationship.case_id in evidence.case_ids
            ):
                latest = (
                    f"historical {evidence.verdict}, unverified (`{evidence.run_id}`)"
                )
            elif (
                relationship is not None
                and relationship.classification == "unsupported"
            ):
                latest = "unsupported"
        else:
            matched_runs = (
                tuple(
                    run
                    for run in authority_runs
                    if relationship.case_id in run.case_ids
                    and relationship.scale_tier == run.scale
                )
                if relationship is not None and relationship.case_id is not None
                else ()
            )
            if matched_runs:
                latest = "; ".join(
                    f"{run.scale}: historical {run.verdict}, unverified (`{run.run_id}`, {run.evidence_scope}); {run.qualification}"
                    for run in matched_runs
                )
            elif (
                relationship is not None
                and relationship.classification == "unsupported"
            ):
                latest = "unsupported"
        cells = (
            f"`examples/{source.source}`",
            mirror_path,
            classification,
            dependencies,
            device_scope,
            scale,
            latest,
        )
        lines.append("| " + " | ".join(_markdown_cell(cell) for cell in cells) + " |")
    lines.extend(
        (
            "",
            "Regenerate with:",
            "",
            "```bash",
            "python -m examples.jax.native_to_jax_index --write",
            "```",
            "",
            "`--check` verifies index consistency. Local pointers remain unverified until their",
            "retained summary, lane receipts, build bindings, and reachable source are checked.",
            "Use `--check --authority-summary PATH` to verify a selected retained run.",
            "",
        )
    )
    return "\n".join(lines)


def main(arguments: list[str] | None = None) -> int:
    """Write or verify the generated index."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true")
    action.add_argument("--check", action="store_true")
    action.add_argument("--print-authority-record", action="store_true")
    parser.add_argument(
        "--authority-summary",
        type=Path,
        help="require and verify the retained authority summary",
    )
    options = parser.parse_args(arguments)
    if options.print_authority_record:
        if options.authority_summary is None:
            parser.error("--print-authority-record requires --authority-summary")
        print(
            json.dumps(
                authority_record_from_summary(options.authority_summary),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    rendered = render_native_to_jax_index()
    if options.authority_summary is not None:
        _, runs = _load_authority_evidence_runs(AUTHORITY_EVIDENCE_PATH)
        summary_sha256 = hashlib.sha256(
            options.authority_summary.read_bytes()
        ).hexdigest()
        matches = tuple(run for run in runs if run.summary_sha256 == summary_sha256)
        if len(matches) != 1:
            raise RuntimeError(
                "authority summary does not match exactly one recorded run"
            )
        verify_authority_summary(
            matches[0],
            options.authority_summary,
        )
    if options.write:
        INDEX_PATH.write_text(rendered, encoding="utf-8")
        return 0
    if not INDEX_PATH.is_file() or INDEX_PATH.read_text(encoding="utf-8") != rendered:
        raise SystemExit("native-to-JAX index is stale; regenerate it with --write")
    if options.authority_summary is None:
        print(
            "Index consistent; historical local evidence is UNVERIFIED (no authority summary supplied)."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
