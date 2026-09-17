"""Generated native-to-JAX index integrity."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import TypedDict

import examples.jax.native_to_jax_index as index_module
import pytest
from examples.jax.native_to_jax_index import (
    INDEX_PATH,
    AuthorityEvidence,
    _load_authority_evidence,
    _load_authority_evidence_runs,
    authority_record_from_summary,
    main,
    render_native_to_jax_index,
    verify_authority_summary,
)


def test_native_to_jax_index_matches_validated_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rendered = render_native_to_jax_index(repo_root=repo_root)
    version, runs = _load_authority_evidence_runs(
        repo_root / "examples/jax/authority_evidence.json"
    )

    assert INDEX_PATH.read_text(encoding="utf-8") == rendered
    # The native reference helper lives outside the public source catalog.
    assert rendered.count("\n| `examples/") == 53
    assert version == 2
    runs_by_id = {run.run_id: run for run in runs}
    bounded = runs_by_id["20260729T005942Z-5ade9aee"]
    exact = runs_by_id["20260917T025023Z-8c8a5461"]
    assert bounded.scale == "bounded"
    assert bounded.case_count == 26
    assert exact.scale == "native_default"
    assert exact.verdict == "quality-band"
    assert exact.case_ids == frozenset(
        {"native-single-stage-boozer-vacuum-optimization"}
    )
    assert (exact.lane_receipt_count, exact.comparison_count) == (3, 57)
    assert bounded.evidence_scope == exact.evidence_scope == "local_only"
    assert "12/57 comparisons failed" in rendered
    assert "3/3 lanes budget-exhausted" in rendered
    assert "final:objective" in rendered
    assert "<= 9.9999999999999995e-08" in rendered
    assert "Superseded provenance" not in rendered
    assert "23fa387a1" not in rendered
    assert "20260729T005942Z-5ade9aee" in rendered
    assert "26 cases / 78 lanes / 1,248 comparisons" in rendered
    assert "`bounded`" in rendered
    exact_row = next(
        line
        for line in rendered.splitlines()
        if line.startswith(
            "| `examples/3_Advanced/single_stage_boozer_vacuum_optimization.py`"
        )
    )
    assert "20260729T005942Z-5ade9aee" not in exact_row
    assert "outer: CPU SciPy" in exact_row
    assert "historical quality-band, unverified" in exact_row
    assert "12/57 comparisons failed" in exact_row
    assert "3/3 lanes budget-exhausted" in exact_row
    assert "20260917T025023Z-8c8a5461" in exact_row
    assert "`native_default`: not run" not in rendered


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
    document = _historical_v1_record()
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


def _historical_v1_record() -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    registry = json.loads(
        (repo_root / "examples/jax/authority_evidence.json").read_text(encoding="utf-8")
    )
    assert registry["schema_version"] == 2
    baseline = next(
        run for run in registry["runs"] if run["run_id"] == "20260729T005942Z-5ade9aee"
    ).copy()
    baseline["schema_version"] = 1
    baseline["native_default_status"] = "not_run"
    return baseline


def _v2_record_from_v1() -> dict[str, object]:
    baseline = _historical_v1_record()
    del baseline["schema_version"]
    del baseline["native_default_status"]
    return baseline


def test_historical_v1_registry_keeps_legacy_rendering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "authority_evidence.json"
    path.write_text(json.dumps(_historical_v1_record()), encoding="utf-8")
    evidence = _load_authority_evidence(path)
    monkeypatch.setattr(
        index_module, "_load_authority_evidence_runs", lambda _path: (1, (evidence,))
    )

    rendered = render_native_to_jax_index()
    assert "Historical local evidence (unverified):\n\n" in rendered
    assert "- `native_default`: not run." in rendered
    assert "| Latest evidence |" in rendered
    assert "quality-band" not in rendered


def test_v2_registry_keeps_bounded_history_and_scopes_exact_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _v2_record_from_v1()
    exact = {
        **baseline,
        "run_id": "exact-one-case",
        "scale": "native_default",
        "verdict": "quality-band",
        "case_count": 1,
        "case_ids": ["native-single-stage-boozer-vacuum-optimization"],
        "lane_receipt_count": 3,
        "comparison_count": 2,
    }
    record_path = tmp_path / "authority_evidence.json"
    record_path.write_text(
        json.dumps({"schema_version": 2, "runs": [baseline, exact]}), encoding="utf-8"
    )
    version, runs = _load_authority_evidence_runs(record_path)
    assert version == 2
    monkeypatch.setattr(
        index_module, "_load_authority_evidence_runs", lambda _path: (version, runs)
    )

    rendered = render_native_to_jax_index()
    exact_row = next(
        line
        for line in rendered.splitlines()
        if line.startswith(
            "| `examples/3_Advanced/single_stage_boozer_vacuum_optimization.py`"
        )
    )
    other_row = next(
        line
        for line in rendered.splitlines()
        if line.startswith("| `examples/1_Simple/just_a_quadratic.py`")
    )
    assert "26 cases / 78 lanes / 1,248 comparisons" in rendered
    assert "scope `local_only`" in rendered
    assert "outer: CPU SciPy" in exact_row
    assert "historical quality-band, unverified" in exact_row
    assert "exact-one-case" in exact_row
    assert "20260729T005942Z-5ade9aee" not in exact_row
    assert "exact-one-case" not in other_row
    assert "`native_default`: not run" not in rendered


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"runs": []}, "non-empty array"),
        ({"duplicate": True}, "run IDs must be unique"),
        ({"scale": "not_applicable"}, "scale is invalid"),
        ({"verdict": "fail"}, "verdict is invalid"),
        ({"case_count": 2}, "case_ids do not match"),
    ),
)
def test_v2_registry_rejects_invalid_run_records(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    baseline = _v2_record_from_v1()
    runs = [baseline, baseline.copy()] if change.get("duplicate") else [baseline]
    if "runs" in change:
        runs = []
    else:
        runs[-1].update(
            {key: value for key, value in change.items() if key != "duplicate"}
        )
    path = tmp_path / "authority_evidence.json"
    path.write_text(json.dumps({"schema_version": 2, "runs": runs}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load_authority_evidence_runs(path)


class _QualityBandMeasurement(TypedDict):
    lane: str
    observable: str
    max_value: float
    observed_value: float
    passed: bool


class _QualityBandCase(TypedDict):
    case_id: str
    authoritative: bool
    verdict: str
    scale_tier: str
    comparisons: list[dict[str, bool]]
    quality_band: list[_QualityBandMeasurement]
    executions: list[dict[str, str | int]]


class _QualityBandSummary(TypedDict):
    run_id: str
    repository_commit: str
    scale: str
    verdict: str
    authoritative: bool
    lanes: list[str]
    cases: list[_QualityBandCase]


def _quality_band_summary() -> _QualityBandSummary:
    lanes = ["native-cpu", "jax-cpu", "jax-gpu"]
    return {
        "run_id": "exact-one-case",
        "repository_commit": "a" * 40,
        "scale": "native_default",
        "verdict": "quality-band",
        "authoritative": True,
        "lanes": lanes,
        "cases": [
            {
                "case_id": "native-single-stage-boozer-vacuum-optimization",
                "authoritative": True,
                "verdict": "quality-band",
                "scale_tier": "native_default",
                "comparisons": [{"passed": False}],
                "quality_band": [
                    {
                        "lane": lane,
                        "observable": "final:objective",
                        "max_value": 1.0e-7,
                        "observed_value": 4.0e-8,
                        "passed": True,
                    }
                    for lane in lanes
                ],
                "executions": [{"lane": lane, "returncode": 0} for lane in lanes],
            }
        ],
    }


def test_summary_only_quality_band_cannot_establish_authority(tmp_path: Path) -> None:
    summary = _quality_band_summary()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not reachable from a named ref"):
        authority_record_from_summary(path)
    with pytest.raises(RuntimeError, match="not reachable from a named ref"):
        main(["--print-authority-record", "--authority-summary", str(path)])

    summary["cases"][0]["quality_band"][0]["passed"] = False
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="quality band is incomplete"):
        authority_record_from_summary(path)

    summary["cases"][0]["quality_band"][0]["passed"] = True
    summary["verdict"] = "pass"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="mislabels quality-band"):
        authority_record_from_summary(path)


@pytest.mark.parametrize(
    "defect",
    (
        "missing_lane",
        "duplicate_lane",
        "over_limit",
        "two_summary_lanes",
        "nonfinite",
        "different_observable",
        "different_limit",
        "common_wrong_observable",
        "common_looser_limit",
    ),
)
def test_authority_record_rejects_incomplete_quality_band(
    tmp_path: Path, defect: str
) -> None:
    summary = _quality_band_summary()
    band = summary["cases"][0]["quality_band"]
    if defect == "missing_lane":
        band.pop()
    elif defect == "duplicate_lane":
        band[1]["lane"] = "native-cpu"
    elif defect == "over_limit":
        band[2]["observed_value"] = 2.0e-7
    elif defect == "two_summary_lanes":
        summary["lanes"] = ["native-cpu", "jax-cpu"]
        summary["cases"][0]["executions"].pop()
        band.pop()
    elif defect == "nonfinite":
        band[0]["observed_value"] = float("nan")
    elif defect == "different_observable":
        band[0]["observable"] = "final:other"
    elif defect == "common_wrong_observable":
        for result in band:
            result["observable"] = "final:other"
    elif defect == "common_looser_limit":
        for result in band:
            result["max_value"] = 2.0e-7
    else:
        band[0]["max_value"] = 2.0e-7
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="quality band is incomplete"):
        authority_record_from_summary(path)


def test_reachable_source_without_lane_receipts_cannot_establish_authority(
    tmp_path: Path,
) -> None:
    summary = _quality_band_summary()
    summary["repository_commit"] = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=index_module.REPO_ROOT, text=True
    ).strip()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lacks retained lane receipt paths"):
        authority_record_from_summary(path)


def test_check_explicitly_reports_unverified_evidence_without_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index = tmp_path / "index.md"
    index.write_text("historical evidence: unverified")
    monkeypatch.setattr(index_module, "INDEX_PATH", index)
    monkeypatch.setattr(
        index_module, "render_native_to_jax_index", lambda: index.read_text()
    )
    assert main(["--check"]) == 0
    assert "UNVERIFIED (no authority summary supplied)" in capsys.readouterr().out
