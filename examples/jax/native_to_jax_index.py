"""Generate the source-owned native-to-JAX example index."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

from examples.jax.manifest_contracts_v3 import load_manifest_contract_pair_documents
from examples.jax.official_source_catalog import (
    OFFICIAL_UPSTREAM_COMMIT,
    OFFICIAL_UPSTREAM_DEFAULT_BRANCH,
    OFFICIAL_UPSTREAM_REPOSITORY,
)
from examples.jax.outer_optimizer_policy import (
    OuterOptimizerPolicy,
    OuterOptimizerPolicyError,
    parse_outer_optimizer_policy,
)
from examples.jax.parity import (
    cases as parity_cases,
    official_quality_bands,
    official_reference,
)
from examples.jax.parity._manifest import (
    ParityManifestValidationError,
    ParityRelationship,
    parse_v2_parity_relationship_groups_document,
)
from examples.jax.parity.arbiter import (
    ArbitrationError,
    LaneObservation,
    LaneOutcomeRejection,
    arbitrate,
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
DERIVED_EVIDENCE_DIRECTORY = JAX_EXAMPLES_DIRECTORY / "parity" / "evidence"
DERIVED_SUMMARY_NAME = "review-summary.json"


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
    # SHA-256 of the tracked derived review package for this run, when one is
    # retained at parity/evidence/<run_id>/review-summary.json. Derived bytes
    # are inspectable evidence only; they never carry authority.
    derived_summary_sha256: str | None = None
    # Optional provider termination for one retained attempt. Absent on
    # historical certifying rows; never upgrades a diagnostic verdict to pass.
    termination: str | None = None


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
_OPTIONAL_RUN_FIELDS = {"qualification", "derived_summary_sha256", "termination"}
_CERTIFYING_VERDICTS = frozenset({"pass", "quality-band"})
_DIAGNOSTIC_VERDICTS = frozenset({"fail", "incomplete"})
_V2_VERDICTS = _CERTIFYING_VERDICTS | _DIAGNOSTIC_VERDICTS
_CERTIFYING_TERMINATIONS = frozenset({"converged", "budget_exhausted"})
_TERMINATIONS = frozenset({"converged", "budget_exhausted", "failed", "incomplete"})


#: What a diagnostic record's verification did and did not establish. Recorded
#: rejections ARE replayed against the recorded commit's contracts
#: (``_confirm_zero_comparison_fail``); recorded comparisons are never
#: recomputed, and the pass-only auditor never runs.
_DIAGNOSTIC_SCOPE = (
    "recorded rejections replayed against the recorded commit's contracts; "
    "recorded comparisons not recomputed (pass-only auditor skipped)"
)


def _diagnostic_qualification(replayed_rejections: tuple[str, ...]) -> str:
    """State what a diagnostic record was verified by; never imply a pass."""
    if not replayed_rejections:
        return "source/receipt binding only; pass-only auditor skipped; never pass"
    return f"source/receipt binding; {_DIAGNOSTIC_SCOPE}; never pass; " + "; ".join(
        replayed_rejections
    )


def _work_budget_qualification(
    admitted_cases: int, total_cases: int, admitted_lanes: int
) -> str:
    """Describe admitted fixed-budget agreement without implying convergence."""
    return (
        f"{admitted_cases}/{total_cases} cases work-budget admitted; "
        f"{admitted_lanes}/{admitted_lanes} admitted lanes budget-exhausted; "
        "numerical behavior agreement at declared fixed budgets only, no convergence claim"
    )


def _parse_authority_run(document: object, *, legacy: bool) -> AuthorityEvidence:
    if not isinstance(document, dict):
        raise TypeError("authority evidence run must be a JSON object")
    expected_fields = _RUN_FIELDS | (
        {"schema_version", "native_default_status"} if legacy else set()
    )
    optional_fields = {"qualification"} if legacy else _OPTIONAL_RUN_FIELDS
    present_fields = set(document)
    if (
        not expected_fields <= present_fields
        or not present_fields - expected_fields <= optional_fields
        or (legacy and document["schema_version"] != 1)
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
    if not legacy and document["verdict"] not in _V2_VERDICTS:
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
    if document["case_count"] <= 0 or document["lane_receipt_count"] <= 0:
        raise ValueError("authority evidence counts must be positive")
    if document["comparison_count"] < 0:
        raise ValueError("authority evidence counts must be positive")
    if document["comparison_count"] == 0 and (legacy or document["verdict"] != "fail"):
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
    derived_summary_sha256 = document.get("derived_summary_sha256")
    if derived_summary_sha256 is not None and (
        not isinstance(derived_summary_sha256, str)
        or len(derived_summary_sha256) != 64
        or not all(
            character in "0123456789abcdef" for character in derived_summary_sha256
        )
    ):
        raise ValueError(
            "authority evidence derived_summary_sha256 must be a lowercase SHA-256"
        )
    termination = document.get("termination")
    if termination is not None and (
        not isinstance(termination, str) or termination not in _TERMINATIONS
    ):
        raise ValueError("authority evidence termination is invalid")
    if not legacy:
        if document["verdict"] in _DIAGNOSTIC_VERDICTS and termination == "converged":
            raise ValueError("diagnostic evidence cannot claim converged termination")
        if document["verdict"] == "incomplete" and termination not in (
            None,
            "incomplete",
        ):
            raise ValueError("incomplete evidence termination must be incomplete")
        if (
            document["verdict"] in _CERTIFYING_VERDICTS
            and termination is not None
            and termination not in _CERTIFYING_TERMINATIONS
        ):
            raise ValueError("certifying evidence termination is invalid")
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
        derived_summary_sha256=derived_summary_sha256,
        termination=termination,
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


def _repository_relative(path: Path) -> str:
    """Spell one in-checkout path the way a commit and a receipt spell it."""
    return path.resolve().relative_to(REPO_ROOT).as_posix()


#: The case contract (``get_case``) is Python, so unlike the two manifests it
#: cannot be read out of a commit; it is instead required to still be the bytes
#: the lanes executed. The replay consumes exactly three of its facts -- the
#: case's ``native_default_quality_band``, its ``work_budget_contract`` and its
#: ``native_default_admitted_terminal_outcomes``.
_CASE_REGISTRY_SOURCE = _repository_relative(Path(parity_cases.__file__))
#: The registry DECLARES the work budget but only CALLS for the band: its
#: ``native_default_quality_band`` is ``official_quality_band(case_id)``, whose
#: number is computed from the band module's rule, the official-reference
#: loader, and the case's tracked sensitivity record. Those bytes decide a band
#: case's verdict, so the replay binds them too. They cannot be bound through
#: the receipts -- ``collect_executed_sources`` hashes loaded Python modules and
#: a sensitivity record is data, never imported -- so they are bound against the
#: recorded commit instead, which is stricter: this checkout's band is then the
#: recorded commit's rule over the recorded commit's data. Everything else the
#: replay consults (routes, buckets, policies) already comes from that commit.
_BAND_CONTRACT_SOURCES = (
    _repository_relative(Path(official_quality_bands.__file__)),
    _repository_relative(Path(official_reference.__file__)),
    *sorted(
        _repository_relative(official_reference.sensitivity_path(case_id))
        for case_id in official_reference.official_sensitivity_case_ids()
    ),
)
#: Every repository file whose bytes decide a replayed arbitration.
_REPLAY_CONTRACT_SOURCES = (_CASE_REGISTRY_SOURCE, *_BAND_CONTRACT_SOURCES)
#: WHICH cases carry a band is the ``*.json`` glob of this directory, so the
#: recorded commit must hold the same set, not merely the same files.
_SENSITIVITY_SOURCE_DIRECTORY = _repository_relative(
    official_reference.SENSITIVITY_ROOT
)
#: Repository-relative spellings of the two contract documents, so the commit
#: they are read out of never has to be this checkout.
_MANIFEST_SOURCE = MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()
_PARITY_MANIFEST_SOURCE = PARITY_MANIFEST_PATH.relative_to(REPO_ROOT).as_posix()


def _committed_bytes(repository_commit: str, relative_path: str) -> bytes:
    """Read one path out of a commit, diagnosing an unreadable request."""
    completed = subprocess.run(
        ("git", "show", f"{repository_commit}:{relative_path}"),
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"recorded commit {repository_commit} does not hold "
            f"{relative_path}: {detail}"
        )
    return completed.stdout


def _committed_path_exists(repository_commit: str, relative_path: str) -> bool:
    """Whether the recorded commit tracks ``relative_path`` at all."""
    return (
        subprocess.run(
            ("git", "cat-file", "-e", f"{repository_commit}:{relative_path}"),
            cwd=REPO_ROOT,
            capture_output=True,
            check=False,
        ).returncode
        == 0
    )


def _committed_directory_entries(
    repository_commit: str, relative_directory: str
) -> frozenset[str]:
    """Names one directory holds in a commit, diagnosing an unreadable request."""
    completed = subprocess.run(
        ("git", "ls-tree", "--name-only", f"{repository_commit}:{relative_directory}"),
        cwd=REPO_ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"recorded commit {repository_commit} does not hold the directory "
            f"{relative_directory}: {detail}"
        )
    return frozenset(
        line for line in completed.stdout.decode("utf-8").splitlines() if line
    )


def _require_replayable_case_contract(
    observations: dict[str, LaneObservation], *, repository_commit: str
) -> None:
    """Refuse to replay unless every byte the verdict depends on is the run's.

    Three bindings, each fail-closed. (1) Every contract source must be held by
    this checkout with the bytes the recorded commit holds, so the band this
    replay arbitrates against is the band the run was judged by; a commit that
    does not hold one refuses through ``_committed_bytes``. (2) The recorded
    commit's sensitivity directory must hold exactly the records this checkout
    globbed, because that glob decides WHICH cases carry a band -- an added
    record is already refused by (1), a removed one only by this. (3) The case
    registry, the one contract source a lane imports, must still appear exactly
    once in every lane's executed sources with those bytes; presence is
    required, not just agreement, since a receipt set that records no case
    registry at all binds nothing. The band module and the loader need no such
    per-lane check: a lane that executed other bytes recorded them, and
    ``verify_authority_summary`` compares every executed source with the same
    commit.
    """
    digests: dict[str, str] = {}
    for relative in _REPLAY_CONTRACT_SOURCES:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise RuntimeError(
                "cannot replay the recorded arbitration: the contract source "
                f"{relative} is not held by this checkout"
            )
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        committed = hashlib.sha256(
            _committed_bytes(repository_commit, relative)
        ).hexdigest()
        if digest != committed:
            raise RuntimeError(
                "cannot replay the recorded arbitration: this checkout's "
                f"{relative} is not the bytes recorded commit "
                f"{repository_commit} holds"
            )
        digests[relative] = digest
    records = frozenset(
        relative.rsplit("/", maxsplit=1)[1]
        for relative in _BAND_CONTRACT_SOURCES
        if relative.startswith(f"{_SENSITIVITY_SOURCE_DIRECTORY}/")
    )
    committed_records = frozenset(
        name
        for name in _committed_directory_entries(
            repository_commit, _SENSITIVITY_SOURCE_DIRECTORY
        )
        if name.endswith(".json")
    )
    if records != committed_records:
        raise RuntimeError(
            "cannot replay the recorded arbitration: this checkout's tracked "
            f"sensitivity records {sorted(records)} are not the records "
            f"recorded commit {repository_commit} holds "
            f"{sorted(committed_records)}"
        )
    for lane in sorted(observations):
        provenance = observations[lane].provenance
        assert provenance is not None
        recorded = tuple(
            source
            for source in provenance.executed_sources
            if source.path == _CASE_REGISTRY_SOURCE
        )
        if len(recorded) != 1 or recorded[0].sha256 != digests[_CASE_REGISTRY_SOURCE]:
            raise RuntimeError(
                "cannot replay the recorded arbitration: this checkout's "
                f"{_CASE_REGISTRY_SOURCE} is not the source lane {lane} executed"
            )


@dataclass(frozen=True, slots=True)
class _ReplayedContract:
    """Exactly the contract facts one replayed arbitration consumes."""

    relationship: ParityRelationship
    outer_optimizer_policy: OuterOptimizerPolicy | None


def _committed_document(
    repository_commit: str, relative_path: str
) -> dict[str, object]:
    document = json.loads(_committed_bytes(repository_commit, relative_path))
    if not isinstance(document, dict):
        raise RuntimeError(
            f"recorded commit contract is not an object: {relative_path}"
        )
    return document


def _document_records(
    document: dict[str, object], field: str
) -> tuple[dict[str, object], ...]:
    value = document.get(field, [])
    if not isinstance(value, list) or not all(
        isinstance(record, dict) for record in value
    ):
        raise RuntimeError(f"recorded commit contract has a malformed {field}")
    return tuple(record for record in value if isinstance(record, dict))


def _replayed_contract(
    repository_commit: str, *, case_id: str, scale: str
) -> _ReplayedContract:
    """Read one case's contract out of the commit that recorded the run.

    Only the replayed case's own relationship and its example's declared outer
    optimizer policy are parsed. Validating the commit's documents with today's
    CROSS-FILE pins instead -- the pinned official catalog, relationship
    ownership order, the ready-example policy table -- refuses every commit
    whose pins have since moved, which is every commit of this campaign.
    """
    parity_document = _committed_document(repository_commit, _PARITY_MANIFEST_SOURCE)
    examples_document = _committed_document(repository_commit, _MANIFEST_SOURCE)
    if (
        examples_document.get("schema_version") != 3
        or parity_document.get("schema_version") != 2
    ):
        raise RuntimeError(
            "recorded arbitration rejection requires manifest v3 and parity v2"
        )
    records = tuple(
        record
        for field in ("relationships", "experimental_relationships")
        for record in _document_records(parity_document, field)
        if record.get("case_id") == case_id
    )
    if len(records) != 1:
        raise RuntimeError(
            f"recorded arbitration rejection has no single relationship: {case_id}"
        )
    try:
        # The selected record is handed to the shared parser as the sole
        # official relationship; scope is the ownership validator's business,
        # and the ownership validator is exactly what cannot run on a commit.
        official_group, _experimental = parse_v2_parity_relationship_groups_document(
            {"schema_version": 2, "relationships": list(records)},
            repo_root=REPO_ROOT,
        )
        relationship = official_group[0].resolve_scale(scale)
        examples = tuple(
            record
            for record in _document_records(examples_document, "jax_examples")
            if record.get("id") == relationship.jax_example_id
        )
        if len(examples) != 1:
            raise RuntimeError(
                "recorded arbitration rejection has no single JAX example: "
                f"{relationship.jax_example_id}"
            )
        example_path = examples[0].get("path")
        if not isinstance(example_path, str):
            raise RuntimeError(
                "recorded commit contract has a malformed example path: "
                f"{relationship.jax_example_id}"
            )
        policy = parse_outer_optimizer_policy(
            examples[0].get("outer_optimizer_policy"),
            example_id=relationship.jax_example_id,
            example_path=example_path,
        )
    except (ParityManifestValidationError, OuterOptimizerPolicyError) as error:
        raise RuntimeError(
            "recorded arbitration rejection has no replayable contract at "
            f"{repository_commit} ({error}): {case_id}"
        ) from error
    return _ReplayedContract(relationship, policy)


def _confirm_zero_comparison_fail(
    case: dict[str, object],
    observations: dict[str, LaneObservation],
    *,
    repository_commit: str,
    scale: str,
    required_lanes: set[str],
) -> None:
    """Require a recorded rejection to be the run's own lane-outcome rejection.

    The string is evidence only if replaying this case's arbitration -- the
    run's own gates, over the recorded commit's contracts -- raises exactly
    that ``LaneOutcomeRejection``. An arbitration that completes at all proves
    no lane outcome was rejected, whatever verdict it reaches.
    """
    case_id = case["case_id"]
    recorded = case["arbitration_rejection"]
    if not isinstance(case_id, str) or not isinstance(recorded, str):
        raise RuntimeError("recorded arbitration rejection is malformed")
    _require_replayable_case_contract(observations, repository_commit=repository_commit)
    declared = get_case(case_id)
    contract = _replayed_contract(repository_commit, case_id=case_id, scale=scale)
    relationship = contract.relationship
    try:
        result = arbitrate(
            relationship.comparison_routes,
            observations,
            required_lanes=frozenset(required_lanes),
            expected_workflow_stages=relationship.workflow_stages,
            case_id=case_id,
            example_id=relationship.jax_example_id,
            outer_optimizer_policy=contract.outer_optimizer_policy,
            quality_band=(
                declared.native_default_quality_band
                if scale == "native_default"
                else None
            ),
            work_budget_contract=declared.work_budget_contract,
            admitted_terminal_outcomes=(
                declared.native_default_admitted_terminal_outcomes
                if scale == "native_default"
                else ()
            ),
        )
    except LaneOutcomeRejection as error:
        if str(error).strip() != recorded.strip():
            raise RuntimeError(
                "recorded arbitration rejection is not the replayed lane "
                f"outcome: {case_id}"
            ) from error
        return
    except ArbitrationError as error:
        raise RuntimeError(
            "recorded arbitration rejection is not replayable: the run's own "
            f"gates refuse this record's receipts ({error}): {case_id}"
        ) from error
    raise RuntimeError(
        "recorded arbitration rejection is contradicted by a replay that "
        f"arbitrated to {result.verdict}: {case_id}"
    )


def derived_summary_path(run_id: str) -> Path:
    """Locate the tracked derived review package retained for one run."""
    return DERIVED_EVIDENCE_DIRECTORY / run_id / DERIVED_SUMMARY_NAME


def verify_derived_summaries(runs: tuple[AuthorityEvidence, ...]) -> None:
    """Fail closed unless every tracked derived package matches its bound digest."""
    bound = {
        run.run_id: run.derived_summary_sha256
        for run in runs
        if run.derived_summary_sha256 is not None
    }
    retained = {
        directory.name
        for directory in DERIVED_EVIDENCE_DIRECTORY.glob("*")
        if (directory / DERIVED_SUMMARY_NAME).is_file()
    }
    unbound = retained - set(bound)
    if unbound:
        raise RuntimeError(
            "retained derived review summary has no derived_summary_sha256 binding: "
            + ", ".join(sorted(unbound))
        )
    for run_id, digest in sorted(bound.items()):
        path = derived_summary_path(run_id)
        if not path.is_file():
            raise RuntimeError(f"bound derived review summary is missing: {run_id}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(
                "derived review summary bytes do not match their bound digest: "
                f"{run_id}"
            )


def verify_authority_summary(
    evidence: AuthorityEvidence,
    summary_path: Path,
) -> None:
    """Bind a retained summary to its compact record.

    Certifying verdicts also re-audit a published pass. Diagnostic verdicts
    check summary SHA-256, identity, reachable commit, lane receipt paths and
    executed-source blob digests, and replay each recorded arbitration
    rejection against the recorded commit's contracts. No recorded comparison
    is recomputed.
    """
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
    certifying = evidence.verdict in _CERTIFYING_VERDICTS
    if certifying and summary.get("authoritative") is not True:
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
    work_budget_case_count = 0
    work_budget_lane_count = 0
    allowed_case_verdicts = _CERTIFYING_VERDICTS if certifying else _V2_VERDICTS
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("authority summary case must be an object")
        case_id = case.get("case_id")
        comparisons = case.get("comparisons")
        executions = case.get("executions")
        rejection = case.get("arbitration_rejection")
        zero_comparison_fail = (
            not certifying
            and evidence.verdict == "fail"
            and case.get("verdict") == "fail"
            and isinstance(comparisons, list)
            and not comparisons
            and isinstance(rejection, str)
            and bool(rejection.strip())
        )
        if (
            not isinstance(case_id, str)
            or not isinstance(comparisons, list)
            or (not comparisons and not zero_comparison_fail)
            or not isinstance(executions, list)
            or (certifying and case.get("authoritative") is not True)
            or case.get("verdict") not in allowed_case_verdicts
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
        if case.get("terminal_contract") == "work-budget":
            work_budget_case_count += 1
            work_budget_lane_count += len(executions)
        case_ids.add(case_id)
        comparison_count += len(comparisons)
        lane_receipt_count += len(executions)
    if not certifying:
        failure_signal = evidence.termination in {
            "failed",
            "incomplete",
            "budget_exhausted",
        }
        incomplete_signal = evidence.termination == "incomplete"
        for case in cases:
            if case.get("verdict") in _DIAGNOSTIC_VERDICTS:
                failure_signal = True
            if case.get("verdict") == "incomplete":
                incomplete_signal = True
            if any(
                comparison.get("passed") is not True
                for comparison in case["comparisons"]
            ):
                failure_signal = True
        if evidence.verdict == "fail" and not failure_signal:
            raise RuntimeError("fail evidence has no failed or incomplete result")
        if evidence.verdict == "incomplete" and not incomplete_signal:
            raise RuntimeError("incomplete evidence has no incomplete termination")
    if evidence.verdict == "quality-band" and quality_band_case_count == 0:
        raise RuntimeError("authority summary verdict mislabels quality-band evidence")
    if evidence.verdict == "pass" and quality_band_case_count > 0:
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
        loaded_observations: dict[str, LaneObservation] = {}
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
            if certifying:
                if (
                    provenance is None
                    or not provenance.authoritative
                    or provenance.repository_commit != evidence.repository_commit
                ):
                    raise RuntimeError("authority lane provenance is not source-bound")
                validate_authoritative_provenance(REPO_ROOT, provenance)
            elif (
                provenance is None
                or provenance.repository_commit != evidence.repository_commit
            ):
                raise RuntimeError(
                    "diagnostic lane provenance does not match its recorded commit"
                )
            for source in provenance.executed_sources:
                if source.git_blob_id is None:
                    if source.path not in provenance.generated_source_bindings:
                        raise RuntimeError(
                            "authority lane has an unbound generated source"
                        )
                    # A generated source is one the commit does not hold. Without
                    # this, a record could declare any TRACKED source generated
                    # (the bindings themselves are validated only for certifying
                    # records) and so skip the comparison below entirely.
                    if _committed_path_exists(evidence.repository_commit, source.path):
                        raise RuntimeError(
                            "authority lane declares a source the recorded "
                            f"commit tracks as generated: {source.path}"
                        )
                    continue
                committed = _committed_bytes(evidence.repository_commit, source.path)
                if hashlib.sha256(committed).hexdigest() != source.sha256:
                    raise RuntimeError(
                        "authority executed source differs from its reachable commit"
                    )
            loaded_observations[observation.lane] = observation
        if not case["comparisons"]:
            _confirm_zero_comparison_fail(
                case,
                loaded_observations,
                repository_commit=evidence.repository_commit,
                scale=evidence.scale,
                required_lanes=expected_lanes,
            )

    if certifying:
        audit_published_run(
            summary_path.parent, repo_root=REPO_ROOT, require_authoritative=True
        )
    if (
        certifying
        and work_budget_case_count
        and not evidence.qualification.endswith(
            _work_budget_qualification(
                work_budget_case_count, len(cases), work_budget_lane_count
            )
        )
    ):
        raise RuntimeError("authority qualification omits work-budget limitation")


def authority_record_from_summary(summary_path: Path) -> dict[str, object]:
    """Build a version 2 local run record from one retained summary.

    Fail and incomplete summaries are compactified after source/receipt binding
    and a replay of every recorded arbitration rejection. They are not a
    published pass, and no recorded comparison is recomputed.
    """
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
    work_budget_case_count = 0
    work_budget_lane_count = 0
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
        if case.get("terminal_contract") == "work-budget":
            work_budget_case_count += 1
            work_budget_lane_count += len(executions)
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
    if work_budget_case_count and summary.get("verdict") in _CERTIFYING_VERDICTS:
        record["qualification"] = _work_budget_qualification(
            work_budget_case_count, len(cases), work_budget_lane_count
        )
    elif summary.get("verdict") in _DIAGNOSTIC_VERDICTS:
        # Only a zero-comparison case's rejection is replayed, so only those
        # strings may be published: the qualification claims each one was.
        record["qualification"] = _diagnostic_qualification(
            tuple(
                case["arbitration_rejection"]
                for case in cases
                if not case["comparisons"]
                and isinstance(case.get("arbitration_rejection"), str)
                and case["arbitration_rejection"].strip()
            )
        )
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
        # A declared terminal outcome is the only reason a banded lane can be
        # neither converged nor budget-exhausted, so it is published beside the
        # count it explains rather than left to the summary alone.
        admitted = "; ".join(
            f"{admission['lane']} terminal outcome {admission['raw_status']} admitted"
            for case in cases
            for admission in case.get("admitted_terminal_lanes", [])
        )
        band_qualification = (
            f"{failed}/{comparison_count} comparisons failed; {exhausted}/{lane_receipt_count} lanes budget-exhausted; "
            + (f"{admitted}; " if admitted else "")
            + f"{bands}; endpoint quality only, no convergence or final-value equivalence"
        )
        if work_budget_case_count:
            record["qualification"] = f"{band_qualification}; {record['qualification']}"
        else:
            record["qualification"] = band_qualification
    return record


def _markdown_cell(value: str) -> str:
    return value.replace("|", r"\|").replace("\n", " ")


def _v2_run_header_line(run: AuthorityEvidence) -> str:
    """One compact pointer line for a schema v2 run.

    Certifying wording is the historical/unverified clause already on disk.
    Diagnostic wording is ``attempt`` plus optional termination; recency is not
    inferred because several attempts can share a case and scale.
    """

    kind = "historical" if run.verdict in _CERTIFYING_VERDICTS else "attempt"
    termination = (
        f" termination `{run.termination}`;" if run.termination is not None else ""
    )
    return (
        f"- `{run.scale}`: {kind} {run.verdict} (unverified); run `{run.run_id}`; "
        f"{run.case_count} cases / {run.lane_receipt_count} lanes / "
        f"{run.comparison_count:,} comparisons; revision `{run.repository_commit}`; "
        f"summary SHA-256 `{run.summary_sha256}`; scope `{run.evidence_scope}`."
        f"{termination} {run.qualification}."
    )


def _v2_run_evidence_clause(run: AuthorityEvidence) -> str:
    """Per-row evidence clause for one schema v2 run.

    Certifying clauses stay byte-identical to the dirty baseline renderer.
    """

    if run.verdict in _CERTIFYING_VERDICTS:
        return (
            f"{run.scale}: historical {run.verdict}, unverified "
            f"(`{run.run_id}`, {run.evidence_scope}); {run.qualification}"
        )
    termination = (
        f"; termination {run.termination}" if run.termination is not None else ""
    )
    return (
        f"{run.scale}: attempt {run.verdict}, unverified "
        f"(`{run.run_id}`, {run.evidence_scope}){termination}; {run.qualification}"
    )


def render_native_to_jax_index(*, repo_root: Path = REPO_ROOT) -> str:
    """Render the complete index from validated manifest contracts."""
    jax_examples_directory = repo_root / "examples" / "jax"
    pair = load_manifest_contract_pair_documents(
        _load_json(jax_examples_directory / MANIFEST_PATH.name),
        _load_json(jax_examples_directory / PARITY_MANIFEST_PATH.name),
        repo_root=repo_root,
    )
    authority_version, authority_runs = _load_authority_evidence_runs(
        jax_examples_directory / AUTHORITY_EVIDENCE_PATH.name
    )
    evidence = authority_runs[0]
    examples_by_id = {example.id: example for example in pair.examples.jax_examples}
    parity_by_source = {
        relationship.native_source: relationship
        for relationship in pair.parity.all_relationships
    }
    lines = [
        "# Native-to-JAX example index",
        "",
        (
            "Generated from `manifest.json`, `parity_manifest.json`, and "
            "`authority_evidence.json`. Do not edit this table by hand."
        ),
        (
            f"Official upstream scope: `{OFFICIAL_UPSTREAM_REPOSITORY}` "
            f"`{OFFICIAL_UPSTREAM_DEFAULT_BRANCH}` at "
            f"`{OFFICIAL_UPSTREAM_COMMIT}` "
            f"({len(pair.examples.source_catalog)} source files)."
        ),
        (
            f"Local experimental registrations: "
            f"{len(pair.examples.experimental_sources)} source files."
        ),
        (
            "Historical local evidence may describe an older roster and does not "
            "establish current official coverage."
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
            lines.append(_v2_run_header_line(run))
    lines.extend(
        (
            "",
            "## Official upstream catalog",
            "",
            (
                "| Native example | JAX mirror | Classification | "
                "Runtime dependencies | Device scope | Scale | "
                + ("Latest evidence |" if authority_version == 1 else "Evidence |")
            ),
            "| --- | --- | --- | --- | --- | --- | --- |",
        )
    )
    for source_index, source in enumerate(pair.examples.all_sources):
        if source_index == len(pair.examples.source_catalog):
            lines.extend(
                (
                    "",
                    "## Experimental local registrations",
                    "",
                    (
                        "| Native example | JAX mirror | Classification | "
                        "Runtime dependencies | Device scope | Scale | "
                        + (
                            "Latest evidence |"
                            if authority_version == 1
                            else "Evidence |"
                        )
                    ),
                    "| --- | --- | --- | --- | --- | --- | --- |",
                )
            )
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
        scale = (
            "—" if relationship is None else ", ".join(relationship.supported_scales)
        )
        latest = "not run"
        if authority_version == 1:
            if (
                relationship is not None
                and evidence.scale in relationship.supported_scales
                and relationship.case_id in evidence.case_ids
            ):
                latest = f"{evidence.scale}: historical {evidence.verdict}, unverified (`{evidence.run_id}`)"
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
                    and run.scale in relationship.supported_scales
                )
                if relationship is not None and relationship.case_id is not None
                else ()
            )
            if matched_runs:
                latest = "; ".join(_v2_run_evidence_clause(run) for run in matched_runs)
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
    _, runs = _load_authority_evidence_runs(AUTHORITY_EVIDENCE_PATH)
    verify_derived_summaries(runs)
    if options.authority_summary is not None:
        summary_sha256 = hashlib.sha256(
            options.authority_summary.read_bytes()
        ).hexdigest()
        derived = tuple(
            run for run in runs if run.derived_summary_sha256 == summary_sha256
        )
        if derived:
            if len(derived) != 1:
                raise RuntimeError(
                    "derived review summary does not match exactly one recorded run"
                )
            print(
                f"Derived review summary of run `{derived[0].run_id}`: bytes match "
                "its recorded derived_summary_sha256; DERIVED numerical review "
                f"only, scope `{derived[0].evidence_scope}`, never authority."
            )
        else:
            matches = tuple(run for run in runs if run.summary_sha256 == summary_sha256)
            if len(matches) != 1:
                raise RuntimeError(
                    "authority summary does not match exactly one recorded run"
                )
            verify_authority_summary(
                matches[0],
                options.authority_summary,
            )
            if matches[0].verdict in _DIAGNOSTIC_VERDICTS:
                termination = (
                    f"termination `{matches[0].termination}`; "
                    if matches[0].termination is not None
                    else ""
                )
                print(
                    f"Source/receipt binding of {matches[0].verdict} attempt "
                    f"`{matches[0].run_id}`: summary SHA-256, reachable commit, "
                    "and lane receipt digests match; "
                    f"scale `{matches[0].scale}`; {termination}"
                    f"{_DIAGNOSTIC_SCOPE}; never pass."
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
