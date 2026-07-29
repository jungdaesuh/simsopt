"""Generated native-to-JAX index integrity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from examples.jax.native_to_jax_index import (
    INDEX_PATH,
    AuthorityEvidence,
    _load_authority_evidence,
    render_native_to_jax_index,
    verify_authority_summary,
)


def test_native_to_jax_index_matches_validated_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rendered = render_native_to_jax_index(repo_root=repo_root)

    assert INDEX_PATH.read_text(encoding="utf-8") == rendered
    assert rendered.count("\n| `examples/") == 52
    assert "20260729T005942Z-5ade9aee" in rendered
    assert "26 cases / 78 lanes / 1,248 comparisons" in rendered
    assert "`bounded`" in rendered
    assert "`native_default`: not run" in rendered


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("scale", "native_default", "requires bounded scale"),
        ("verdict", "", "requires pass verdict"),
        ("native_default_status", "pass", "requires native_default not_run"),
        ("evidence_scope", "", "requires local_only scope"),
        ("comparison_count", -1, "counts must be positive"),
    ),
)
def test_authority_record_rejects_malformed_status_or_count(
    tmp_path: Path,
    field: str,
    value: str | int,
    message: str,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    document = json.loads(
        (repo_root / "examples/jax/authority_evidence.json").read_text(encoding="utf-8")
    )
    document[field] = value
    record = tmp_path / "authority_evidence.json"
    record.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        _load_authority_evidence(record)


@pytest.mark.parametrize(
    ("authoritative", "comparisons", "message"),
    (
        (False, [{"passed": True}], "not authoritative"),
        (True, [{"passed": False}], "failed comparison"),
        (True, [], "case is incomplete"),
    ),
)
def test_authority_summary_verification_rejects_non_authority_or_failed_gate(
    tmp_path: Path,
    authoritative: bool,
    comparisons: list[dict[str, bool]],
    message: str,
) -> None:
    lanes = ["native-cpu", "jax-cpu", "jax-gpu"]
    summary = {
        "run_id": "run",
        "repository_commit": "a" * 40,
        "scale": "bounded",
        "verdict": "pass",
        "authoritative": authoritative,
        "lanes": lanes,
        "cases": [
            {
                "case_id": "case",
                "authoritative": True,
                "verdict": "pass",
                "scale_tier": "bounded",
                "comparisons": comparisons,
                "executions": [{"lane": lane, "returncode": 0} for lane in lanes],
            }
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_bytes = (json.dumps(summary, sort_keys=True) + "\n").encode()
    summary_path.write_bytes(summary_bytes)
    evidence = AuthorityEvidence(
        run_id="run",
        repository_commit="a" * 40,
        summary_sha256=hashlib.sha256(summary_bytes).hexdigest(),
        scale="bounded",
        verdict="pass",
        case_count=1,
        lane_receipt_count=3,
        comparison_count=len(comparisons),
        case_ids=frozenset({"case"}),
        native_default_status="not_run",
        evidence_scope="local_only",
    )

    with pytest.raises(RuntimeError, match=message):
        verify_authority_summary(evidence, summary_path)
