"""Generate the human-readable parity report from aggregate JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from examples.jax.parity.artifacts import read_bytes
from examples.jax.parity.audit import audit_published_run


def _mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{context} must be a JSON object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _string(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"report field {field} must be a non-empty string")
    return value


def _boolean(record: dict[str, object], field: str) -> bool:
    value = record.get(field)
    if not isinstance(value, bool):
        raise TypeError(f"report field {field} must be boolean")
    return value


def render_results_document(summary: object, *, artifact_reference: str) -> str:
    """Render a deterministic Markdown report from one aggregate summary."""
    document = _mapping(summary, "summary")
    if document.get("schema_version") not in (1, 2):
        raise ValueError("unsupported aggregate summary schema")
    cases_value = document.get("cases")
    lanes_value = document.get("lanes")
    if not isinstance(cases_value, list) or not isinstance(lanes_value, list):
        raise TypeError("aggregate summary requires cases and lanes arrays")
    if not all(isinstance(lane, str) and lane for lane in lanes_value):
        raise ValueError("aggregate summary lanes must be non-empty strings")
    cases = [_mapping(case, "case") for case in cases_value]
    rows = []
    for case in cases:
        comparisons = case.get("comparisons")
        if not isinstance(comparisons, list):
            raise TypeError("case comparisons must be an array")
        passed = sum(
            1
            for comparison in comparisons
            if _boolean(_mapping(comparison, "comparison"), "passed")
        )
        rows.append(
            "| {example} | `{native}` | {classification} | {scale} | {oracle} | "
            "{verdict} | {passed}/{total} |".format(
                example=_string(case, "jax_example_id"),
                native=_string(case, "native_source"),
                classification=_string(case, "classification"),
                scale=_string(case, "scale_tier"),
                oracle=_string(case, "oracle_kind"),
                verdict=_string(case, "verdict"),
                passed=passed,
                total=len(comparisons),
            )
        )
    authority = (
        "authoritative" if _boolean(document, "authoritative") else "exploratory"
    )
    scale = _string(document, "scale") if "scale" in document else "bounded"
    evidence_note = (
        "This authoritative run is source-bound to the named clean committed "
        f"checkout and may promote only the classifications and {scale} scale "
        "reported below."
        if authority == "authoritative"
        else (
            "A passing exploratory run proves the recorded numerical comparisons "
            "on the recorded device, but it cannot promote an authoritative parity "
            "claim until the identical run is source-bound to a clean committed "
            "checkout."
        )
    )
    repository_state = "dirty" if _boolean(document, "repository_dirty") else "clean"
    return "\n".join(
        (
            "# Native SIMSOPT vs JAX Example Parity Results",
            "",
            (
                "This file is generated from one aggregate parity `summary.json`; "
                "do not edit the numerical table manually."
            ),
            "",
            f"- Run ID: `{_string(document, 'run_id')}`",
            f"- Verdict: **{_string(document, 'verdict')}**",
            f"- Evidence class: **{authority}** ({repository_state} checkout)",
            f"- Repository commit: `{_string(document, 'repository_commit')}`",
            f"- Lanes: {', '.join(f'`{lane}`' for lane in lanes_value)}",
            f"- Scale: `{scale}`",
            f"- Aggregate artifact: `{artifact_reference}`",
            "",
            evidence_note,
            "",
            "| JAX example | Native source | Classification | Scale | Oracle | Verdict | Comparisons |",
            "|---|---|---|---|---|---:|---:|",
            *rows,
            "",
            (
                "`full` describes declared scientific workflow-stage coverage, while "
                "`bounded` describes execution scale. `reduced` evidence must not be "
                "reported as full native-example equivalence. Oracle kinds distinguish "
                "native Python/SciPy, loaded `simsoptpp`, analytic, and external references."
            ),
            "",
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Generate a Markdown report from one aggregate summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    audit_published_run(
        args.summary.parent,
        repo_root=args.repo_root,
    )
    summary = json.loads(read_bytes(args.summary.parent, args.summary.name))
    rendered = render_results_document(
        summary,
        artifact_reference=str(args.summary.parent),
    )
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
