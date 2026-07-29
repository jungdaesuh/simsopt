"""Generate the source-owned native-to-JAX example index."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from examples.jax.manifest_contracts_v3 import (
    JaxExamplesManifestV3,
    load_manifest_contract_pair_documents,
)

JAX_EXAMPLES_DIRECTORY = Path(__file__).resolve().parent
REPO_ROOT = JAX_EXAMPLES_DIRECTORY.parents[1]
INDEX_PATH = JAX_EXAMPLES_DIRECTORY / "NATIVE_TO_JAX_INDEX.md"
MANIFEST_PATH = JAX_EXAMPLES_DIRECTORY / "manifest.json"
PARITY_MANIFEST_PATH = JAX_EXAMPLES_DIRECTORY / "parity_manifest.json"
AUTHORITY_EVIDENCE_PATH = JAX_EXAMPLES_DIRECTORY / "authority_evidence.json"


@dataclass(frozen=True, slots=True)
class AuthorityEvidence:
    """Compact, tracked pointer to the latest authoritative campaign."""

    run_id: str
    repository_commit: str
    summary_sha256: str
    scale: str
    verdict: str
    case_count: int
    lane_receipt_count: int
    comparison_count: int
    case_ids: frozenset[str]
    native_default_status: str
    evidence_scope: str


def _load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_authority_evidence(path: Path) -> AuthorityEvidence:
    document = _load_json(path)
    if not isinstance(document, dict):
        raise TypeError("authority evidence must be a JSON object")
    expected_fields = {
        "schema_version",
        "run_id",
        "repository_commit",
        "summary_sha256",
        "scale",
        "verdict",
        "case_count",
        "lane_receipt_count",
        "comparison_count",
        "case_ids",
        "native_default_status",
        "evidence_scope",
    }
    if set(document) != expected_fields or document["schema_version"] != 1:
        raise ValueError("authority evidence schema does not match version 1")
    string_fields = (
        "run_id",
        "repository_commit",
        "summary_sha256",
        "scale",
        "verdict",
        "native_default_status",
        "evidence_scope",
    )
    if not all(isinstance(document[field], str) for field in string_fields):
        raise ValueError("authority evidence identity fields must be strings")
    if document["scale"] != "bounded":
        raise ValueError("authority evidence schema v1 requires bounded scale")
    if document["verdict"] != "pass":
        raise ValueError("authority evidence schema v1 requires pass verdict")
    if document["native_default_status"] != "not_run":
        raise ValueError("authority evidence schema v1 requires native_default not_run")
    if document["evidence_scope"] != "local_only":
        raise ValueError("authority evidence schema v1 requires local_only scope")
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
        isinstance(case_id, str) for case_id in case_ids_value
    ):
        raise ValueError("authority evidence case_ids must be strings")
    case_ids = frozenset(case_ids_value)
    if len(case_ids) != len(case_ids_value) or len(case_ids) != document["case_count"]:
        raise ValueError("authority evidence case_ids do not match case_count")
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
        native_default_status=document["native_default_status"],
        evidence_scope=document["evidence_scope"],
    )


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
            or case.get("verdict") != "pass"
            or case.get("scale_tier") != evidence.scale
        ):
            raise RuntimeError("authority summary case is incomplete")
        if not all(
            isinstance(comparison, dict) and comparison.get("passed") is True
            for comparison in comparisons
        ):
            raise RuntimeError("authority summary contains a failed comparison")
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
    if (
        case_ids != evidence.case_ids
        or len(cases) != evidence.case_count
        or lane_receipt_count != evidence.lane_receipt_count
        or comparison_count != evidence.comparison_count
    ):
        raise RuntimeError("authority summary counts do not match its record")


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
    evidence = _load_authority_evidence(
        jax_examples_directory / AUTHORITY_EVIDENCE_PATH.name
    )
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
        "Latest authority evidence:",
        "",
        (
            f"- `{evidence.scale}`: {evidence.verdict}; run `{evidence.run_id}`; "
            f"{evidence.case_count} cases / {evidence.lane_receipt_count} lanes / "
            f"{evidence.comparison_count:,} comparisons."
        ),
        (
            f"- Evidence revision: `{evidence.repository_commit}`; "
            f"summary SHA-256: `{evidence.summary_sha256}`; scope: "
            f"`{evidence.evidence_scope}`."
        ),
        f"- `native_default`: {evidence.native_default_status.replace('_', ' ')}.",
        "",
        (
            "| Native example | JAX mirror | Classification | "
            "Runtime dependencies | Device scope | Scale | Latest evidence |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
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
        scale = "—" if relationship is None else relationship.scale_tier
        latest = "not run"
        if (
            relationship is not None
            and relationship.scale_tier == evidence.scale
            and relationship.case_id in evidence.case_ids
        ):
            latest = f"{evidence.verdict} (`{evidence.run_id}`)"
        elif relationship is not None and relationship.classification == "unsupported":
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
            "Verify it is current with `--check`.",
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
    parser.add_argument(
        "--authority-summary",
        type=Path,
        help="require and verify the retained authority summary",
    )
    options = parser.parse_args(arguments)
    rendered = render_native_to_jax_index()
    if options.authority_summary is not None:
        verify_authority_summary(
            _load_authority_evidence(AUTHORITY_EVIDENCE_PATH),
            options.authority_summary,
        )
    if options.write:
        INDEX_PATH.write_text(rendered, encoding="utf-8")
        return 0
    if not INDEX_PATH.is_file() or INDEX_PATH.read_text(encoding="utf-8") != rendered:
        raise SystemExit("native-to-JAX index is stale; regenerate it with --write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
