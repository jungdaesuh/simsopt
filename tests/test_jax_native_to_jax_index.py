"""Generated native-to-JAX index integrity."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import TypedDict

import numpy as np
import examples.jax.manifest_contracts_v3 as manifest_contracts
import examples.jax.native_to_jax_index as index_module
import pytest
from examples.jax.parity.arbiter import LaneObservation
from examples.jax.parity.provenance import (
    REQUIRED_PROVENANCE_SOURCE_PATHS,
    DeviceMetadata,
    ExecutedSource,
    LaneProvenance,
    collect_explicit_sources,
    collect_repository_state,
)
from examples.jax.parity.official_quality_bands import OFFICIAL_BAND_CASE_IDS
from examples.jax.parity.official_reference import sensitivity_path
from examples.jax.parity.receipts import write_lane_observation
from examples.jax.native_to_jax_index import (
    INDEX_PATH,
    AuthorityEvidence,
    _load_authority_evidence,
    _load_authority_evidence_runs,
    authority_record_from_summary,
    main,
    render_native_to_jax_index,
    verify_authority_summary,
    verify_derived_summaries,
)
from examples.jax.parity._manifest import ScaleContract


def test_native_to_jax_index_matches_validated_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rendered = render_native_to_jax_index(repo_root=repo_root)
    version, runs = _load_authority_evidence_runs(
        repo_root / "examples/jax/authority_evidence.json"
    )

    assert INDEX_PATH.read_text(encoding="utf-8") == rendered
    # One row per official source (53) and per experimental registration (1).
    assert rendered.count("\n| `examples/") == 54
    assert "## Official upstream catalog" in rendered
    assert "## Experimental local registrations" in rendered
    assert "examples/1_Simple/periodicfieldline_QA.py" in rendered
    assert "examples/1_Simple/periodicfieldline_QH.py" in rendered
    assert version == 2
    runs_by_id = {run.run_id: run for run in runs}
    bounded = runs_by_id["20260729T005942Z-5ade9aee"]
    exact = runs_by_id["20260917T035857Z-6a1f0ea9"]
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
        if "`examples/3_Advanced/single_stage_boozer_vacuum_optimization.py`" in line
    )
    assert "20260729T005942Z-5ade9aee" not in exact_row
    assert "outer: CPU SciPy" in exact_row
    assert "historical quality-band, unverified" in exact_row
    assert "12/57 comparisons failed" in exact_row
    assert "3/3 lanes budget-exhausted" in exact_row
    assert "20260917T035857Z-6a1f0ea9" in exact_row
    assert "`native_default`: not run" not in rendered


def test_dual_scale_index_keeps_historical_evidence_at_its_recorded_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    examples = json.loads((repo_root / "examples/jax/manifest.json").read_text())
    parity = json.loads((repo_root / "examples/jax/parity_manifest.json").read_text())
    pair = manifest_contracts.load_manifest_contract_pair_documents(
        examples, parity, repo_root=repo_root
    )
    relationships = tuple(
        replace(
            relationship,
            scale_contracts=(
                ScaleContract(
                    "native_default", relationship.comparison_routes, "scheduled"
                ),
            ),
        )
        if relationship.case_id == "native-just-a-quadratic"
        else relationship
        for relationship in pair.parity.relationships
    )
    dual_scale_pair = replace(
        pair, parity=replace(pair.parity, relationships=relationships)
    )
    monkeypatch.setattr(
        index_module,
        "load_manifest_contract_pair_documents",
        lambda *_args, **_kwargs: dual_scale_pair,
    )

    rendered = render_native_to_jax_index(repo_root=repo_root)
    row = next(
        line
        for line in rendered.splitlines()
        if "`examples/1_Simple/just_a_quadratic.py`" in line
    )
    assert "| bounded, native_default |" in row
    assert "bounded: historical pass" in row
    assert "20260729T005942Z-5ade9aee" in row
    assert "20260917T035857Z-6a1f0ea9" not in row
    assert "native_default: historical" not in row


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
    assert "bounded: historical pass" in rendered
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
        if "`examples/3_Advanced/single_stage_boozer_vacuum_optimization.py`" in line
    )
    other_row = next(
        line
        for line in rendered.splitlines()
        if "`examples/1_Simple/just_a_quadratic.py`" in line
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
        ({"verdict": "success"}, "verdict is invalid"),
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


def test_work_budget_record_qualifies_behavior_without_convergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = {
        "run_id": "bounded-work-test",
        "repository_commit": "a" * 40,
        "scale": "bounded",
        "verdict": "pass",
        "cases": [
            {
                "case_id": "native-stage-two-optimization",
                "terminal_contract": "work-budget",
                "comparisons": [{"passed": True}],
                "executions": [
                    {"lane": lane} for lane in ("native-cpu", "jax-cpu", "jax-gpu")
                ],
            }
        ],
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setattr(index_module, "verify_authority_summary", lambda *_: None)

    record = authority_record_from_summary(summary_path)

    assert record["qualification"] == (
        "1/1 cases work-budget admitted; 3/3 admitted lanes budget-exhausted; "
        "numerical behavior agreement at declared fixed budgets only, no convergence claim"
    )


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


def _stub_rendered_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate derived-evidence assertions from full manifest rendering."""
    index = tmp_path / "index.md"
    index.write_text("historical evidence: unverified")
    monkeypatch.setattr(index_module, "INDEX_PATH", index)
    monkeypatch.setattr(
        index_module, "render_native_to_jax_index", lambda: index.read_text()
    )


def test_shipped_derived_review_summary_is_bound_and_reported_as_derived(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, runs = _load_authority_evidence_runs(
        repo_root / "examples/jax/authority_evidence.json"
    )
    exact = next(run for run in runs if run.run_id == "20260917T035857Z-6a1f0ea9")
    derived = index_module.derived_summary_path(exact.run_id)

    assert exact.derived_summary_sha256 is not None
    assert (
        hashlib.sha256(derived.read_bytes()).hexdigest() == exact.derived_summary_sha256
    )
    verify_derived_summaries(runs)

    _stub_rendered_index(tmp_path, monkeypatch)
    assert main(["--check", "--authority-summary", str(derived)]) == 0
    reported = capsys.readouterr().out
    assert "DERIVED numerical review only" in reported
    assert exact.run_id in reported
    assert "never authority" in reported


def test_check_rejects_a_tampered_derived_review_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, runs = _load_authority_evidence_runs(
        repo_root / "examples/jax/authority_evidence.json"
    )
    exact = next(run for run in runs if run.run_id == "20260917T035857Z-6a1f0ea9")
    shipped = index_module.derived_summary_path(exact.run_id).read_bytes()
    tampered_directory = tmp_path / "evidence" / exact.run_id
    tampered_directory.mkdir(parents=True)
    (tampered_directory / "review-summary.json").write_bytes(
        shipped[:-2] + b" " + shipped[-1:]
    )
    monkeypatch.setattr(
        index_module, "DERIVED_EVIDENCE_DIRECTORY", tmp_path / "evidence"
    )
    _stub_rendered_index(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="do not match their bound digest"):
        main(["--check"])


def test_check_rejects_a_missing_or_unbound_derived_review_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    _, runs = _load_authority_evidence_runs(
        repo_root / "examples/jax/authority_evidence.json"
    )
    exact = next(run for run in runs if run.run_id == "20260917T035857Z-6a1f0ea9")
    shipped = index_module.derived_summary_path(exact.run_id).read_bytes()
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setattr(index_module, "DERIVED_EVIDENCE_DIRECTORY", empty)
    _stub_rendered_index(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="bound derived review summary is missing"):
        main(["--check"])

    unbound_directory = empty / exact.run_id
    unbound_directory.mkdir()
    (unbound_directory / "review-summary.json").write_bytes(shipped)
    unbound_runs = tuple(
        AuthorityEvidence(
            **{
                field: getattr(run, field)
                for field in AuthorityEvidence.__dataclass_fields__
                if field != "derived_summary_sha256"
            }
        )
        for run in runs
    )
    monkeypatch.setattr(
        index_module, "_load_authority_evidence_runs", lambda _path: (2, unbound_runs)
    )
    with pytest.raises(RuntimeError, match="has no derived_summary_sha256 binding"):
        main(["--check"])


def _diagnostic_run(
    baseline: dict[str, object],
    *,
    run_id: str,
    verdict: str,
    scale: str = "native_default",
    termination: str | None = "incomplete",
    case_ids: list[str] | None = None,
) -> dict[str, object]:
    ids = case_ids if case_ids is not None else ["native-tracing-fieldlines-qa"]
    run = {
        **baseline,
        "run_id": run_id,
        "scale": scale,
        "verdict": verdict,
        "case_count": len(ids),
        "case_ids": ids,
        "lane_receipt_count": 3,
        "comparison_count": 1,
        "qualification": "shipped attempt retained; not a pass",
    }
    if termination is not None:
        run["termination"] = termination
    return run


def test_v2_fail_and_incomplete_runs_are_representable_next_to_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = _v2_record_from_v1()
    failed = _diagnostic_run(
        baseline,
        run_id="shipped-fail-qa",
        verdict="fail",
        termination="incomplete",
    )
    incomplete = _diagnostic_run(
        baseline,
        run_id="shipped-incomplete-ncsx",
        verdict="incomplete",
        termination="incomplete",
        case_ids=["native-tracing-fieldlines-ncsx"],
    )
    budget = _diagnostic_run(
        baseline,
        run_id="shipped-fail-minimal",
        verdict="fail",
        termination="budget_exhausted",
        case_ids=["native-stage-two-optimization-minimal"],
    )
    record_path = tmp_path / "authority_evidence.json"
    record_path.write_text(
        json.dumps(
            {"schema_version": 2, "runs": [baseline, failed, incomplete, budget]}
        ),
        encoding="utf-8",
    )
    version, runs = _load_authority_evidence_runs(record_path)
    assert version == 2
    assert {run.verdict for run in runs} == {"pass", "fail", "incomplete"}
    assert {run.termination for run in runs} == {
        None,
        "incomplete",
        "budget_exhausted",
    }
    monkeypatch.setattr(
        index_module, "_load_authority_evidence_runs", lambda _path: (version, runs)
    )

    rendered = render_native_to_jax_index()
    qa_row = next(
        line
        for line in rendered.splitlines()
        if "`examples/1_Simple/tracing_fieldlines_QA.py`" in line
    )
    ncsx_row = next(
        line
        for line in rendered.splitlines()
        if "`examples/1_Simple/tracing_fieldlines_NCSX.py`" in line
    )
    quadratic_row = next(
        line
        for line in rendered.splitlines()
        if "`examples/1_Simple/just_a_quadratic.py`" in line
    )
    minimal_row = next(
        line
        for line in rendered.splitlines()
        if "`examples/1_Simple/stage_two_optimization_minimal.py`" in line
    )
    assert "bounded: historical pass, unverified" in qa_row
    assert "native_default: attempt fail, unverified" in qa_row
    assert "termination incomplete" in qa_row
    assert "shipped-fail-qa" in qa_row
    assert "historical fail" not in qa_row
    assert "latest fail" not in rendered
    assert "native_default: attempt incomplete, unverified" in ncsx_row
    assert "termination incomplete" in ncsx_row
    assert "termination budget_exhausted" in minimal_row
    assert "attempt fail" in minimal_row
    assert "attempt fail" not in quadratic_row
    assert "- `native_default`: attempt fail (unverified); run `shipped-fail-qa`;" in (
        rendered
    )
    assert rendered.count("attempt fail") >= 2
    assert "`native_default`: not run" not in rendered


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"verdict": "fail", "termination": "converged"}, "cannot claim converged"),
        (
            {"verdict": "incomplete", "termination": "failed"},
            "termination must be incomplete",
        ),
        (
            {"verdict": "pass", "termination": "failed"},
            "certifying evidence termination",
        ),
        ({"termination": "done"}, "termination is invalid"),
    ),
)
def test_v2_diagnostic_termination_cannot_impersonate_convergence(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    baseline = _v2_record_from_v1()
    run = _diagnostic_run(baseline, run_id="bad-termination", verdict="fail")
    run.update(change)
    path = tmp_path / "authority_evidence.json"
    path.write_text(json.dumps({"schema_version": 2, "runs": [run]}), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load_authority_evidence_runs(path)


def _fail_summary(
    *,
    comparisons_passed: bool,
    case_verdict: str,
    comparisons: list[dict[str, object]] | None = None,
    arbitration_rejection: str | None = None,
) -> dict[str, object]:
    lanes = ["native-cpu", "jax-cpu", "jax-gpu"]
    case: dict[str, object] = {
        "case_id": "native-tracing-fieldlines-qa",
        "authoritative": False,
        "verdict": case_verdict,
        "scale_tier": "native_default",
        "comparisons": (
            comparisons if comparisons is not None else [{"passed": comparisons_passed}]
        ),
        "executions": [{"lane": lane, "returncode": 0} for lane in lanes],
    }
    if arbitration_rejection is not None:
        case["arbitration_rejection"] = arbitration_rejection
    return {
        "run_id": "shipped-fail-qa",
        "repository_commit": "a" * 40,
        "scale": "native_default",
        "verdict": "fail",
        "authoritative": False,
        "lanes": lanes,
        "cases": [case],
    }


def test_fail_summary_is_not_a_pass_and_still_requires_bound_receipts(
    tmp_path: Path,
) -> None:
    summary = _fail_summary(comparisons_passed=False, case_verdict="fail")
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not reachable from a named ref"):
        authority_record_from_summary(path)

    summary["verdict"] = "pass"
    summary["authoritative"] = True
    summary["cases"][0]["verdict"] = "pass"
    summary["cases"][0]["authoritative"] = True
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="failed comparison"):
        authority_record_from_summary(path)


def test_fail_summary_with_reachable_commit_still_requires_receipt_paths(
    tmp_path: Path,
) -> None:
    summary = _fail_summary(comparisons_passed=False, case_verdict="fail")
    summary["repository_commit"] = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=index_module.REPO_ROOT, text=True
    ).strip()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lacks retained lane receipt paths"):
        authority_record_from_summary(path)


def test_fail_summary_rejects_a_replaced_lane_receipt(tmp_path: Path) -> None:
    summary = _fail_summary(comparisons_passed=False, case_verdict="fail")
    summary["repository_commit"] = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=index_module.REPO_ROOT, text=True
    ).strip()
    for execution in summary["cases"][0]["executions"]:
        relative = f"lanes/{execution['lane']}"
        execution["result_directory"] = relative
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        (directory / "lane_result.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lane receipt"):
        authority_record_from_summary(path)


def test_check_labels_diagnostic_binding_as_not_numerical_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Bind one real retained diagnostic run end to end through ``--check``.

    ``verify_authority_summary`` stays LIVE: the summary carries real lane
    receipts, a really rejected lane and a rejection the replay reproduces, so
    this fails when the diagnostic binding path breaks instead of only pinning
    the printed wording. Only other runs' derived evidence and the rendered
    index are stubbed out.
    """
    _simulate_clean_checkout(monkeypatch)
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=False,
        rejection="jax-cpu did not report scientific success",
    )
    record = {**authority_record_from_summary(path), "termination": "failed"}
    record_path = tmp_path / "authority.json"
    record_path.write_text(
        json.dumps({"schema_version": 2, "runs": [record]}), encoding="utf-8"
    )
    _, runs = _load_authority_evidence_runs(record_path)
    monkeypatch.setattr(
        index_module, "_load_authority_evidence_runs", lambda _path: (2, runs)
    )
    monkeypatch.setattr(index_module, "verify_derived_summaries", lambda _runs: None)
    _stub_rendered_index(tmp_path, monkeypatch)

    assert main(["--check", "--authority-summary", str(path)]) == 0
    reported = capsys.readouterr().out
    assert "Source/receipt binding of fail attempt" in reported
    assert "termination `failed`" in reported
    assert (
        "recorded rejections replayed against the recorded commit's contracts"
        in reported
    )
    assert "recorded comparisons not recomputed (pass-only auditor skipped)" in reported
    assert "never pass" in reported
    assert "certified" not in reported.lower()


def test_fail_summary_without_a_failure_signal_is_rejected(tmp_path: Path) -> None:
    summary = _fail_summary(comparisons_passed=True, case_verdict="pass")
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no failed or incomplete result"):
        authority_record_from_summary(path)


def test_incomplete_summary_without_incomplete_termination_is_rejected(
    tmp_path: Path,
) -> None:
    summary = _fail_summary(comparisons_passed=False, case_verdict="fail")
    summary["verdict"] = "incomplete"
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="no incomplete termination"):
        authority_record_from_summary(path)


def test_zero_comparison_pass_summary_is_still_incomplete(tmp_path: Path) -> None:
    summary = _fail_summary(comparisons_passed=True, case_verdict="pass")
    summary["verdict"] = "pass"
    summary["authoritative"] = True
    summary["cases"][0]["authoritative"] = True
    summary["cases"][0]["comparisons"] = []
    summary["cases"][0]["arbitration_rejection"] = "operator stop"
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="counts must be positive"):
        authority_record_from_summary(path)


def test_zero_comparison_fail_without_rejection_is_incomplete(tmp_path: Path) -> None:
    summary = _fail_summary(
        comparisons_passed=False, case_verdict="fail", comparisons=[]
    )
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="case is incomplete"):
        authority_record_from_summary(path)


def test_zero_comparison_fail_with_rejection_still_requires_receipts(
    tmp_path: Path,
) -> None:
    summary = _fail_summary(
        comparisons_passed=False,
        case_verdict="fail",
        comparisons=[],
        arbitration_rejection="jax-cpu did not report scientific success",
    )
    summary["repository_commit"] = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=index_module.REPO_ROOT, text=True
    ).strip()
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="lacks retained lane receipt paths"):
        authority_record_from_summary(path)


def test_zero_comparison_fail_rejects_a_replaced_lane_receipt(tmp_path: Path) -> None:
    summary = _fail_summary(
        comparisons_passed=False,
        case_verdict="fail",
        comparisons=[],
        arbitration_rejection="jax-cpu did not report scientific success",
    )
    summary["repository_commit"] = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=index_module.REPO_ROOT, text=True
    ).strip()
    for execution in summary["cases"][0]["executions"]:
        relative = f"lanes/{execution['lane']}"
        execution["result_directory"] = relative
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        (directory / "lane_result.json").write_text("{}", encoding="utf-8")
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lane receipt"):
        authority_record_from_summary(path)


def test_fail_aggregate_may_include_a_quality_band_case(tmp_path: Path) -> None:
    lanes = ["native-cpu", "jax-cpu", "jax-gpu"]
    summary = {
        "run_id": "mixed-fail",
        "repository_commit": "a" * 40,
        "scale": "native_default",
        "verdict": "fail",
        "authoritative": False,
        "lanes": lanes,
        "cases": [
            {
                "case_id": "native-single-stage-boozer-vacuum-optimization",
                "authoritative": False,
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
            },
            {
                "case_id": "native-tracing-fieldlines-qa",
                "authoritative": False,
                "verdict": "fail",
                "scale_tier": "native_default",
                "comparisons": [{"passed": False}],
                "executions": [{"lane": lane, "returncode": 0} for lane in lanes],
            },
        ],
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="not reachable from a named ref"):
        authority_record_from_summary(path)


def test_v2_fail_record_allows_zero_comparisons(tmp_path: Path) -> None:
    baseline = _v2_record_from_v1()
    run = {
        **baseline,
        "run_id": "rejected-lane-fail",
        "scale": "native_default",
        "verdict": "fail",
        "case_count": 1,
        "case_ids": ["native-boozer"],
        "lane_receipt_count": 3,
        "comparison_count": 0,
        "qualification": "source/receipt binding only; pass-only auditor skipped; never pass",
        "termination": "failed",
    }
    path = tmp_path / "authority_evidence.json"
    path.write_text(json.dumps({"schema_version": 2, "runs": [run]}), encoding="utf-8")
    _version, runs = _load_authority_evidence_runs(path)
    assert runs[0].comparison_count == 0
    assert runs[0].verdict == "fail"


def test_authority_record_rejects_a_malformed_derived_digest(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    document = json.loads(
        (repo_root / "examples/jax/authority_evidence.json").read_text(encoding="utf-8")
    )
    document["runs"][1]["derived_summary_sha256"] = "NOT-A-DIGEST"
    path = tmp_path / "authority_evidence.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a lowercase SHA-256"):
        _load_authority_evidence_runs(path)


_BOOZER_CASE_ID = "native-single-stage-boozer-vacuum-optimization"
#: The driver `native-single-stage-boozer-vacuum-optimization` declares through
#: its outer optimizer policy. It contains "scipy", so a replay that forgets to
#: pass the policy rejects it as a forbidden parity driver instead.
_BOOZER_POLICY_DRIVER = "simsopt_jax_scipy_bfgs_with_exact_analytic_boozer_newton"
#: The one case-package module an arbitration depends on: it declares every
#: case's quality band and work budget contract.
_CASE_CONTRACT_SOURCE = index_module._CASE_REGISTRY_SOURCE
#: A case-package module the replay never consults, used to prove the binding
#: is scoped to the contract registry instead of the whole case package.
_UNRELATED_CASE_SOURCE = "examples/jax/parity/cases/traceable_least_squares.py"
#: The registry only CALLS for a band; this module holds the rule that decides
#: the ceiling a `native_default` band case is arbitrated against.
_BAND_MODULE_SOURCE = "examples/jax/parity/official_quality_bands.py"
#: The tracked upstream sensitivity records the ceiling is computed from. They
#: are data: no receipt can ever record them, which is why the replay binds
#: them against the recorded commit instead.
_SENSITIVITY_SOURCES = tuple(
    source
    for source in index_module._BAND_CONTRACT_SOURCES
    if source.startswith(f"{index_module._SENSITIVITY_SOURCE_DIRECTORY}/")
)


def _simulate_clean_checkout(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Serve this checkout's bytes as the recorded commit's, and record asks.

    ``verify_authority_summary`` only ever accepts a run whose executed sources
    equal their bytes at the recorded commit, so a retained run is verifiable
    only from a checkout that still holds them. This tree is mid-campaign
    dirty, so the fixture stands in for that clean checkout instead of pinning
    the tests to whatever happens to be committed. The returned list records
    every ``(commit, path)`` the verifier asked the commit for.
    """
    requested: list[tuple[str, str]] = []

    def committed(repository_commit: str, relative_path: str) -> bytes:
        requested.append((repository_commit, relative_path))
        return (index_module.REPO_ROOT / relative_path).read_bytes()

    def entries(repository_commit: str, relative_directory: str) -> frozenset[str]:
        requested.append((repository_commit, relative_directory))
        return frozenset(
            path.name
            for path in (index_module.REPO_ROOT / relative_directory).iterdir()
        )

    monkeypatch.setattr(index_module, "_committed_bytes", committed)
    monkeypatch.setattr(index_module, "_committed_directory_entries", entries)
    return requested


def _boozer_contract(lanes: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Read the published keys and workflow stages the case's routes declare."""
    repo_root = Path(__file__).resolve().parents[1]
    parity = json.loads(
        (repo_root / "examples/jax/parity_manifest.json").read_text(encoding="utf-8")
    )
    relationship = next(
        item
        for item in parity["relationships"] + parity["experimental_relationships"]
        if item.get("case_id") == _BOOZER_CASE_ID
    )
    value_keys = sorted(
        {
            f"{route['phase']}:{route['observable']}"
            for route in relationship["comparison_routes"]
            if set(route["lane_pair"].split(":")) <= set(lanes)
        }
    )
    return tuple(value_keys), tuple(relationship["workflow_stages"])


#: An observable no relationship declares a route for. Publishing it on every
#: lane trips the arbiter's route-matrix gate, which is an integrity violation
#: and never a lane outcome.
_UNROUTED_OBSERVABLE = "final:unrouted_observable"


def _zero_comparison_fail_summary(
    tmp_path: Path,
    *,
    jax_cpu_success: bool,
    rejection: str,
    final_objective: float = 1.0e-9,
    checkout: Path | None = None,
    route_break: bool = False,
    forge_generated_source: str | None = None,
) -> Path:
    """Write a realistic zero-comparison fail run with real lane receipts.

    Both lanes publish exactly the observables the case's routes require for
    the ``native-cpu:jax-cpu`` pair, so ``arbitrate`` really runs the run's own
    gates instead of stopping at an incomplete route matrix. ``checkout`` binds
    the run to another git checkout holding the same source bytes;
    ``route_break`` publishes an unrouted observable so the replay raises an
    integrity violation instead of a lane outcome. ``forge_generated_source``
    declares one tracked source ``git_blob_id: null`` plus a binding string,
    the shape a hand-written diagnostic receipt uses to claim a source is
    generated and so skip the commit-bytes comparison.
    """
    repo_root = Path(__file__).resolve().parents[1] if checkout is None else checkout
    repository_state = collect_repository_state(repo_root)
    commit = repository_state.repository_commit
    sources = collect_explicit_sources(
        repo_root, REQUIRED_PROVENANCE_SOURCE_PATHS + (_CASE_CONTRACT_SOURCE,)
    )
    generated_bindings: dict[str, str] = {}
    if forge_generated_source is not None:
        sources = tuple(
            ExecutedSource(source.path, source.sha256, None)
            if source.path == forge_generated_source
            else source
            for source in sources
        )
        generated_bindings = {forge_generated_source: "generated"}
    lanes = ("native-cpu", "jax-cpu")
    value_keys, workflow_stages = _boozer_contract(lanes)
    values = {key: np.asarray([1.0], dtype=np.float64) for key in value_keys}
    values["final:objective"] = np.asarray([final_objective], dtype=np.float64)
    if route_break:
        values[_UNROUTED_OBSERVABLE] = np.asarray([1.0], dtype=np.float64)
    executions: list[dict[str, object]] = []
    for lane, backend, driver in (
        ("native-cpu", "native_cpu", "simsopt_scipy_bfgs_exact_boozer"),
        ("jax-cpu", "jax_cpu_parity", _BOOZER_POLICY_DRIVER),
    ):
        success = jax_cpu_success or lane != "jax-cpu"
        relative = f"lanes/{lane}"
        observation = LaneObservation(
            lane=lane,
            backend_mode=backend,
            platform="cpu",
            precision="fp64",
            scale="native_default",
            input_fingerprint="a" * 64,
            configuration_fingerprint="b" * 64,
            effective_construction_fingerprint="c" * 64,
            driver=driver,
            normalized_status="converged" if success else "failed",
            raw_status="1" if success else "0",
            success=success,
            nit=4,
            nfev=5,
            njev=5,
            completed_workflow_stages=workflow_stages,
            provenance=LaneProvenance(
                repository_commit=commit,
                repository_dirty=repository_state.repository_dirty,
                tracked_diff_sha256=repository_state.tracked_diff_sha256,
                untracked_files=repository_state.untracked_files,
                executed_sources=sources,
                generated_source_bindings=generated_bindings,
                python_version="3.11.0",
                jax_version="0.10.0" if lane.startswith("jax-") else None,
                simsopt_version="1.10.0",
                simsopt_version_commit="g" + "d" * 9,
                simsopt_version_checkout_compatible=True,
                lane_environment_policy={"SIMSOPT_BACKEND_MODE": backend},
                jax_effective_transfer_guards=(
                    {
                        "device_to_device": "disallow",
                        "device_to_host": "disallow",
                        "host_to_device": "disallow",
                    }
                    if lane.startswith("jax-")
                    else {}
                ),
                devices=(DeviceMetadata(0, "cpu", "cpu", 0),),
                host_peak_rss_bytes=1,
                host_peak_rss_method="test",
                device_memory_peak_bytes=None,
                device_memory_status="unavailable",
                memory_measurement_scope="test",
                steady_state_memory_measured=False,
                measurement_synchronization="test",
                simsoptpp_path=None,
                simsoptpp_sha256=None,
                simsoptpp_version=None,
                simsoptpp_build_commit=None,
                simsoptpp_checkout_compatible=None,
                authoritative=False,
            ),
            values=values,
        )
        directory = tmp_path / relative
        directory.mkdir(parents=True)
        write_lane_observation(directory, observation)
        executions.append({"lane": lane, "returncode": 0, "result_directory": relative})
    summary = {
        "run_id": "zero-comparison-attempt",
        "repository_commit": commit,
        "scale": "native_default",
        "verdict": "fail",
        "authoritative": False,
        "lanes": list(lanes),
        "cases": [
            {
                "case_id": _BOOZER_CASE_ID,
                "authoritative": False,
                "verdict": "fail",
                "scale_tier": "native_default",
                "comparisons": [],
                "arbitration_rejection": rejection,
                "executions": executions,
            }
        ],
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summary, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _commit_checkout(checkout: Path, message: str) -> None:
    """Commit the whole working tree of a temporary checkout."""
    subprocess.run(("git", "add", "-A"), cwd=checkout, check=True)
    subprocess.run(
        (
            "git",
            "-c",
            "user.name=parity",
            "-c",
            "user.email=parity@invalid.example",
            "commit",
            "--quiet",
            "-m",
            message,
        ),
        cwd=checkout,
        check=True,
    )


def _temporary_authority_checkout(tmp_path: Path) -> Path:
    """Commit the sources one receipt binds into a real temporary git repo.

    Only ``git init``/``git commit`` are used, so the tests built on this drive
    the real ``git show`` in ``_committed_bytes`` instead of a stub. The
    committed ``manifest.json`` declares the case's outer optimizer policy;
    ``_drop_declared_policy`` then removes it from the working tree, so a
    replay that reads the checkout rather than the commit behaves differently.
    """
    repo_root = Path(__file__).resolve().parents[1]
    checkout = tmp_path / "checkout"
    parity = json.loads(
        (repo_root / "examples/jax/parity_manifest.json").read_text(encoding="utf-8")
    )
    examples = json.loads(
        (repo_root / "examples/jax/manifest.json").read_text(encoding="utf-8")
    )
    relationship = next(
        item
        for item in parity["relationships"] + parity["experimental_relationships"]
        if item.get("case_id") == _BOOZER_CASE_ID
    )
    example = next(
        item
        for item in examples["jax_examples"]
        if item["id"] == relationship["jax_example_id"]
    )
    assert example["outer_optimizer_policy"] == "scipy-bfgs-over-jax-exact-boozer"
    documents = (
        (
            "examples/jax/parity_manifest.json",
            json.dumps({"schema_version": 2, "relationships": [relationship]}).encode(
                "utf-8"
            ),
        ),
        (
            "examples/jax/manifest.json",
            json.dumps({"schema_version": 3, "jax_examples": [example]}).encode(
                "utf-8"
            ),
        ),
        (
            "examples/jax/run_parity.py",
            (repo_root / "examples/jax/run_parity.py").read_bytes(),
        ),
        (
            "examples/jax/parity/child.py",
            (repo_root / "examples/jax/parity/child.py").read_bytes(),
        ),
        (_CASE_CONTRACT_SOURCE, (repo_root / _CASE_CONTRACT_SOURCE).read_bytes()),
        # The band module, the official-reference loader and the tracked
        # sensitivity records: the bytes that decide a band case's ceiling.
        *(
            (relative, (repo_root / relative).read_bytes())
            for relative in index_module._BAND_CONTRACT_SOURCES
        ),
    )
    for relative, payload in documents:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    for test_path in relationship["correctness_tests"]:
        target = checkout / test_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"")
    subprocess.run(("git", "init", "--quiet"), cwd=checkout, check=True)
    _commit_checkout(checkout, "authority checkout")
    return checkout


def _drop_declared_policy(checkout: Path) -> None:
    """Remove the declared policy from the checkout, never from the commit."""
    path = checkout / "examples/jax/manifest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    for example in document["jax_examples"]:
        example.pop("outer_optimizer_policy", None)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_recorded_rejection_is_confirmed_from_the_commit_by_real_git_show(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accept path over a real commit, and it reads the commit not the tree.

    Wave 2 parsed the commit's manifests with this checkout's cross-file
    validators, so `_confirm_zero_comparison_fail` raised
    `OuterOptimizerPolicyError` for every real commit and could accept no
    genuine record at all. Here `git show` really runs: the commit declares the
    case's outer optimizer policy and the checkout no longer does, so a replay
    that read the checkout would refuse this record as a forbidden parity
    driver instead of confirming it.
    """
    checkout = _temporary_authority_checkout(tmp_path)
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=False,
        rejection="jax-cpu did not report scientific success",
        checkout=checkout,
    )
    _drop_declared_policy(checkout)
    monkeypatch.setattr(index_module, "REPO_ROOT", checkout)
    committed = subprocess.run(
        ("git", "show", "HEAD:examples/jax/manifest.json"),
        cwd=checkout,
        capture_output=True,
        check=True,
    ).stdout
    assert committed != (checkout / "examples/jax/manifest.json").read_bytes()

    record = authority_record_from_summary(path)

    assert record["verdict"] == "fail"
    assert record["comparison_count"] == 0
    qualification = record["qualification"]
    assert isinstance(qualification, str)
    assert "replayed against the recorded commit's contracts" in qualification
    assert qualification.endswith("jax-cpu did not report scientific success")


def test_replay_over_a_commit_that_lost_the_policy_refuses_the_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The discriminator of the test above, run the other way round.

    The commit itself declares no policy, so the run's own gates reject the
    case's driver before any lane outcome; the recorded lane-outcome string is
    then not what the replay produces and the record is refused.
    """
    checkout = _temporary_authority_checkout(tmp_path)
    _drop_declared_policy(checkout)
    _commit_checkout(checkout, "drop policy")
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=False,
        rejection="jax-cpu did not report scientific success",
        checkout=checkout,
    )
    monkeypatch.setattr(index_module, "REPO_ROOT", checkout)

    with pytest.raises(RuntimeError, match="forbidden parity driver"):
        authority_record_from_summary(path)


def test_committed_bytes_diagnoses_a_missing_commit_and_a_missing_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real `git show` failures are reported, not raised as CalledProcessError."""
    checkout = _temporary_authority_checkout(tmp_path)
    monkeypatch.setattr(index_module, "REPO_ROOT", checkout)
    commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert index_module._committed_bytes(commit, "examples/jax/manifest.json")
    with pytest.raises(RuntimeError, match="does not hold examples/jax/manifest.json"):
        index_module._committed_bytes("0" * 40, "examples/jax/manifest.json")
    with pytest.raises(RuntimeError, match="does not hold examples/jax/absent.json"):
        index_module._committed_bytes(commit, "examples/jax/absent.json")


def test_replayed_contract_resolves_at_every_recorded_authority_commit() -> None:
    """The contract must load from the commits the index actually records.

    Wave 2's reader raised `OuterOptimizerPolicyError` here for HEAD and for
    both commits named in `authority_evidence.json`.
    """
    repo_root = Path(__file__).resolve().parents[1]
    runs = json.loads(
        (repo_root / "examples/jax/authority_evidence.json").read_text(encoding="utf-8")
    )["runs"]
    recorded = tuple(
        (run["repository_commit"], run["scale"], run["case_ids"][0]) for run in runs
    )
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert recorded
    for commit, scale, case_id in recorded + (
        (head, "bounded", "native-just-a-quadratic"),
    ):
        contract = index_module._replayed_contract(commit, case_id=case_id, scale=scale)
        assert contract.relationship.case_id == case_id
        assert contract.relationship.scale_tier == scale
        assert contract.relationship.comparison_routes
        assert contract.relationship.workflow_stages


def test_recorded_lane_outcome_rejection_is_confirmed_by_replaying_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The accept path: the replay must raise exactly the recorded rejection.

    This is the branch every genuine rejected-lane record takes, which the
    pre-wave early return (`if any(not observation.success ...): return`) never
    reached.
    """
    _simulate_clean_checkout(monkeypatch)
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=False,
        rejection="jax-cpu did not report scientific success",
    )
    record = authority_record_from_summary(path)

    assert record["verdict"] == "fail"
    assert record["comparison_count"] == 0
    qualification = record["qualification"]
    assert isinstance(qualification, str)
    assert qualification.endswith("jax-cpu did not report scientific success")


def test_forged_rejection_over_rejected_lane_receipts_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator string over a genuinely rejected lane is still not evidence.

    The pre-wave guard returned early for exactly this receipt shape, so this
    record was published verbatim into the index's qualification.
    """
    _simulate_clean_checkout(monkeypatch)
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=False,
        rejection="operator stopped the run",
    )

    with pytest.raises(RuntimeError, match="not the replayed lane outcome"):
        authority_record_from_summary(path)


def test_rejection_matching_the_policy_less_replay_over_certifying_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The run's outer optimizer policy decides the driver gate, not its absence.

    Without `outer_optimizer_policy=`, the replay rejects this case's own
    policy driver as a forbidden parity driver; an operator could therefore file
    a run that certifies as a failed attempt by copying that message. With the
    policy passed, the replay reaches a verdict and the record is refused.
    """
    _simulate_clean_checkout(monkeypatch)
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=True,
        rejection=(f"jax-cpu uses forbidden parity driver {_BOOZER_POLICY_DRIVER}"),
    )

    with pytest.raises(
        RuntimeError, match="contradicted by a replay that arbitrated to quality-band"
    ):
        authority_record_from_summary(path)


def test_legacy_integrity_rejection_is_refused_as_a_diagnosed_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record whose rejection was an integrity violation is diagnosed, not raised.

    The pre-wave runner caught the base `ArbitrationError`, so a harness
    violation could be written into `summary.json` as a zero-comparison case
    fail. Such a record must still be refused -- it is not a lane outcome --
    but as the module's own `RuntimeError`, so `--check` reports it instead of
    propagating the arbiter's `ArbitrationError` out of `main`.
    """
    _simulate_clean_checkout(monkeypatch)
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=True,
        rejection="applicable observables require a complete direct lane-pair matrix",
        route_break=True,
    )

    with pytest.raises(RuntimeError, match="is not replayable: the run's own gates"):
        authority_record_from_summary(path)


def _boozer_lane_observation(sources: tuple[ExecutedSource, ...]) -> LaneObservation:
    """One jax-cpu receipt carrying exactly the executed sources given."""
    repository_state = collect_repository_state(Path(__file__).resolve().parents[1])
    return LaneObservation(
        lane="jax-cpu",
        backend_mode="jax_cpu_parity",
        platform="cpu",
        precision="fp64",
        scale="native_default",
        input_fingerprint="a" * 64,
        configuration_fingerprint="b" * 64,
        effective_construction_fingerprint="c" * 64,
        driver=_BOOZER_POLICY_DRIVER,
        normalized_status="failed",
        raw_status="0",
        success=False,
        nit=1,
        nfev=1,
        njev=1,
        completed_workflow_stages=("construct",),
        provenance=LaneProvenance(
            repository_commit=repository_state.repository_commit,
            repository_dirty=repository_state.repository_dirty,
            tracked_diff_sha256=repository_state.tracked_diff_sha256,
            untracked_files=repository_state.untracked_files,
            executed_sources=sources,
            python_version="3.11.0",
            jax_version="0.10.0",
            simsopt_version="1.10.0",
            simsopt_version_commit="g" + "d" * 9,
            simsopt_version_checkout_compatible=True,
            lane_environment_policy={"SIMSOPT_BACKEND_MODE": "jax_cpu_parity"},
            jax_effective_transfer_guards={},
            devices=(DeviceMetadata(0, "cpu", "cpu", 0),),
            host_peak_rss_bytes=1,
            host_peak_rss_method="test",
            device_memory_peak_bytes=None,
            device_memory_status="unavailable",
            memory_measurement_scope="test",
            steady_state_memory_measured=False,
            measurement_synchronization="test",
            simsoptpp_path=None,
            simsoptpp_sha256=None,
            simsoptpp_version=None,
            simsoptpp_build_commit=None,
            simsoptpp_checkout_compatible=None,
            authoritative=False,
        ),
        values={"final:objective": np.asarray([1.0], dtype=np.float64)},
    )


def _confirm_boozer_rejection(observation: LaneObservation) -> None:
    provenance = observation.provenance
    assert provenance is not None
    index_module._confirm_zero_comparison_fail(
        {
            "case_id": _BOOZER_CASE_ID,
            "arbitration_rejection": "jax-cpu did not report scientific success",
        },
        {"jax-cpu": observation},
        repository_commit=provenance.repository_commit,
        scale="native_default",
        required_lanes={"jax-cpu"},
    )


def test_replay_refuses_a_case_contract_this_checkout_no_longer_holds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`get_case` is this checkout's Python; it must be the run's own bytes."""
    _simulate_clean_checkout(monkeypatch)
    repo_root = Path(__file__).resolve().parents[1]
    sources = collect_explicit_sources(
        repo_root, REQUIRED_PROVENANCE_SOURCE_PATHS + (_CASE_CONTRACT_SOURCE,)
    )
    stale = tuple(
        ExecutedSource(source.path, "0" * 64, source.git_blob_id)
        if source.path == _CASE_CONTRACT_SOURCE
        else source
        for source in sources
    )

    with pytest.raises(RuntimeError, match="is not the source lane jax-cpu executed"):
        _confirm_boozer_rejection(_boozer_lane_observation(stale))


def test_replay_refuses_receipts_that_record_no_case_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The binding must be satisfied by presence, never by absence.

    A receipt set that names no case registry bound nothing at all, and the
    replay then used this checkout's quality band and work budget unchecked.
    """
    _simulate_clean_checkout(monkeypatch)
    repo_root = Path(__file__).resolve().parents[1]
    sources = tuple(
        source
        for source in collect_explicit_sources(
            repo_root, REQUIRED_PROVENANCE_SOURCE_PATHS + (_CASE_CONTRACT_SOURCE,)
        )
        if source.path != _CASE_CONTRACT_SOURCE
    )

    assert all(source.path != _CASE_CONTRACT_SOURCE for source in sources)
    with pytest.raises(RuntimeError, match="is not the source lane jax-cpu executed"):
        _confirm_boozer_rejection(_boozer_lane_observation(sources))


def test_replay_is_not_blocked_by_an_unrelated_case_module_edit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only the module that declares the contract is bound.

    `cases/__init__.py` imports every case module, so a real receipt records
    dozens of them; binding all of them made any unrelated case edit block
    every older record for good. The replay consumes the quality band and work
    budget, which only the registry declares.
    """
    _simulate_clean_checkout(monkeypatch)
    repo_root = Path(__file__).resolve().parents[1]
    sources = collect_explicit_sources(
        repo_root,
        REQUIRED_PROVENANCE_SOURCE_PATHS
        + (_CASE_CONTRACT_SOURCE, _UNRELATED_CASE_SOURCE),
    )
    edited = tuple(
        ExecutedSource(source.path, "0" * 64, source.git_blob_id)
        if source.path == _UNRELATED_CASE_SOURCE
        else source
        for source in sources
    )

    assert any(source.path == _UNRELATED_CASE_SOURCE for source in edited)
    # The pre-wave guard bound every recorded source under the case package,
    # so this edit refused the replay; the recorded rejection is confirmed.
    _confirm_boozer_rejection(_boozer_lane_observation(edited))


def test_replay_contract_sources_bind_the_band_module_its_loader_and_its_data() -> None:
    """Every file the replayed verdict reads, not only the registry that calls for it.

    `cases/__init__.py` declares the work budget but CALLS `official_quality_band`, so a
    band case's ceiling is decided by the band module, the official-reference loader and
    that case's tracked sensitivity record. Binding the registry alone left the deciding
    number free to move with the registry digest still matching.
    """
    bound = index_module._REPLAY_CONTRACT_SOURCES

    assert _CASE_CONTRACT_SOURCE in bound
    assert _BAND_MODULE_SOURCE in bound
    assert "examples/jax/parity/official_reference/__init__.py" in bound
    assert set(_SENSITIVITY_SOURCES) == {
        index_module._repository_relative(sensitivity_path(case_id))
        for case_id in OFFICIAL_BAND_CASE_IDS
    }
    assert _SENSITIVITY_SOURCES


def test_replay_refuses_a_band_ceiling_this_checkout_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The deciding number must be the recorded commit's, over real git objects."""
    checkout = _temporary_authority_checkout(tmp_path)
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=False,
        rejection="jax-cpu did not report scientific success",
        checkout=checkout,
    )
    band_module = checkout / _BAND_MODULE_SOURCE
    committed = band_module.read_bytes()
    moved = committed.replace(
        b"(1.0 + (high - low) / low)", b"(2.0 + (high - low) / low)"
    )
    assert moved != committed
    band_module.write_bytes(moved)
    monkeypatch.setattr(index_module, "REPO_ROOT", checkout)

    with pytest.raises(
        RuntimeError, match="official_quality_bands.py is not the bytes"
    ):
        authority_record_from_summary(path)


def test_replay_refuses_a_sensitivity_record_this_checkout_no_longer_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A band's data file is never imported, so only this binding can hold it."""
    checkout = _temporary_authority_checkout(tmp_path)
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=False,
        rejection="jax-cpu did not report scientific success",
        checkout=checkout,
    )
    (checkout / _SENSITIVITY_SOURCES[0]).unlink()
    monkeypatch.setattr(index_module, "REPO_ROOT", checkout)

    with pytest.raises(RuntimeError, match="is not held by this checkout"):
        authority_record_from_summary(path)


def test_replay_refuses_a_sensitivity_set_the_recorded_commit_does_not_share(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WHICH cases carry a band is a glob of that directory, so the SET is bound.

    This record is recorded at a commit holding a fourth sensitivity record while the
    checkout holds three, so the replayed arbitration bands fewer cases than the run
    did. Per-file equality cannot see that; the committed directory listing can.
    """
    checkout = _temporary_authority_checkout(tmp_path)
    extra = checkout / index_module._SENSITIVITY_SOURCE_DIRECTORY / "native-extra.json"
    extra.write_bytes(b"{}\n")
    _commit_checkout(checkout, "extra sensitivity record")
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=False,
        rejection="jax-cpu did not report scientific success",
        checkout=checkout,
    )
    extra.unlink()
    monkeypatch.setattr(index_module, "REPO_ROOT", checkout)

    with pytest.raises(RuntimeError, match="sensitivity records"):
        authority_record_from_summary(path)


def test_generated_binding_cannot_hide_a_source_the_recorded_commit_tracks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostic receipt may not call a committed file generated.

    `validate_authoritative_provenance` runs only for certifying records, so for the
    diagnostic records this replay accepts the generated-source bindings were never
    validated: declaring the case registry `git_blob_id: null` plus any binding string
    skipped the commit-bytes comparison and left the record bound to the working tree.
    """
    checkout = _temporary_authority_checkout(tmp_path)
    path = _zero_comparison_fail_summary(
        tmp_path,
        jax_cpu_success=False,
        rejection="jax-cpu did not report scientific success",
        checkout=checkout,
        forge_generated_source=_CASE_CONTRACT_SOURCE,
    )
    monkeypatch.setattr(index_module, "REPO_ROOT", checkout)

    with pytest.raises(RuntimeError, match="the recorded commit tracks"):
        authority_record_from_summary(path)
