"""Validate and replay immutable RED-GREEN-REFACTOR evidence."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


class ReceiptValidationError(ValueError):
    """A TDD receipt cannot prove the behavior it claims."""


@dataclass(frozen=True)
class ReceiptAudit:
    """Immutable summary of one complete receipt-document audit."""

    receipt_count: int
    behavior_ids: tuple[str, ...]
    replayed: bool


@dataclass(frozen=True)
class _Phase:
    revision: str
    command: tuple[str, ...]
    cwd: str
    expected_exit: int
    failure_excerpt: str | None
    combined_output_sha256: str
    source_sha256: tuple[tuple[str, str], ...]
    timestamp_utc: datetime


@dataclass(frozen=True)
class _Receipt:
    behavior_id: str
    native_source_id: str
    mirror_id: str
    intended_red_failure: str
    test_sha256: tuple[tuple[str, str], ...]
    red: _Phase
    green: _Phase
    refactor: _Phase


_HEX_DIGITS = frozenset("0123456789abcdef")
_PHASE_NAMES = ("red", "green", "refactor")


def _mapping(value: object, context: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ReceiptValidationError(f"{context} must be a JSON object")
    return value


def _sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReceiptValidationError(f"{context} must be a JSON array")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: frozenset[str], context: str
) -> None:
    actual = frozenset(value)
    if actual != expected:
        raise ReceiptValidationError(
            f"{context} keys do not match schema; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReceiptValidationError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReceiptValidationError(f"{context} must be an integer")
    return value


def _sha256(value: object, context: str) -> str:
    digest = _string(value, context)
    if len(digest) != 64 or any(character not in _HEX_DIGITS for character in digest):
        raise ReceiptValidationError(
            f"{context} must be a lowercase hexadecimal SHA-256"
        )
    return digest


def _relative_path(value: object, context: str, *, allow_dot: bool = False) -> str:
    raw = _string(value, context)
    if raw == "." and allow_dot:
        return raw
    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != raw:
        raise ReceiptValidationError(f"{context} must be a canonical relative path")
    return raw


def _timestamp(value: object, context: str) -> datetime:
    raw = _string(value, context)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as error:
        raise ReceiptValidationError(
            f"{context} must be an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ReceiptValidationError(f"{context} must use UTC")
    return parsed


def _hash_mapping(value: object, context: str) -> tuple[tuple[str, str], ...]:
    document = _mapping(value, context)
    if not document:
        raise ReceiptValidationError(f"{context} must not be empty")
    entries = tuple(
        sorted(
            (
                _relative_path(path, f"{context} path"),
                _sha256(digest, f"{context}.{path}"),
            )
            for path, digest in document.items()
        )
    )
    if len({path for path, _ in entries}) != len(entries):
        raise ReceiptValidationError(f"{context} contains duplicate paths")
    return entries


def _phase(value: object, context: str) -> _Phase:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "revision",
                "command",
                "cwd",
                "expected_exit",
                "failure_excerpt",
                "combined_output_sha256",
                "source_sha256",
                "timestamp_utc",
            }
        ),
        context,
    )
    command = tuple(
        _string(argument, f"{context}.command[{index}]")
        for index, argument in enumerate(
            _sequence(document["command"], f"{context}.command")
        )
    )
    if not command:
        raise ReceiptValidationError(f"{context}.command must not be empty")
    failure_value = document["failure_excerpt"]
    if failure_value is not None and not isinstance(failure_value, str):
        raise ReceiptValidationError(
            f"{context}.failure_excerpt must be string or null"
        )
    return _Phase(
        revision=_string(document["revision"], f"{context}.revision"),
        command=command,
        cwd=_relative_path(document["cwd"], f"{context}.cwd", allow_dot=True),
        expected_exit=_integer(document["expected_exit"], f"{context}.expected_exit"),
        failure_excerpt=failure_value,
        combined_output_sha256=_sha256(
            document["combined_output_sha256"],
            f"{context}.combined_output_sha256",
        ),
        source_sha256=_hash_mapping(
            document["source_sha256"], f"{context}.source_sha256"
        ),
        timestamp_utc=_timestamp(document["timestamp_utc"], f"{context}.timestamp_utc"),
    )


def _receipt(value: object, context: str) -> _Receipt:
    document = _mapping(value, context)
    _exact_keys(
        document,
        frozenset(
            {
                "behavior_id",
                "native_source_id",
                "mirror_id",
                "intended_red_failure",
                "test_sha256",
                "phases",
            }
        ),
        context,
    )
    phases = _mapping(document["phases"], f"{context}.phases")
    _exact_keys(phases, frozenset(_PHASE_NAMES), f"{context}.phases")
    return _Receipt(
        behavior_id=_string(document["behavior_id"], f"{context}.behavior_id"),
        native_source_id=_relative_path(
            document["native_source_id"], f"{context}.native_source_id"
        ),
        mirror_id=_string(document["mirror_id"], f"{context}.mirror_id"),
        intended_red_failure=_string(
            document["intended_red_failure"], f"{context}.intended_red_failure"
        ),
        test_sha256=_hash_mapping(document["test_sha256"], f"{context}.test_sha256"),
        red=_phase(phases["red"], f"{context}.phases.red"),
        green=_phase(phases["green"], f"{context}.phases.green"),
        refactor=_phase(phases["refactor"], f"{context}.phases.refactor"),
    )


def _git(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReceiptValidationError(
            f"git {' '.join(arguments)} failed: {detail or completed.returncode}"
        )
    return completed.stdout


def _full_revision(repo_root: Path, revision: str, context: str) -> str:
    if len(revision) != 40 or any(
        character not in _HEX_DIGITS for character in revision
    ):
        raise ReceiptValidationError(f"{context} must be a full lowercase commit ID")
    resolved = _git(repo_root, "rev-parse", "--verify", f"{revision}^{{commit}}")
    full = resolved.decode("ascii").strip()
    if full != revision:
        raise ReceiptValidationError(f"{context} does not resolve exactly")
    return full


def _is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    completed = subprocess.run(
        ("git", "merge-base", "--is-ancestor", ancestor, descendant),
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode not in (0, 1):
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReceiptValidationError(f"git ancestry check failed: {detail}")
    return completed.returncode == 0


def _blob(repo_root: Path, revision: str, relative_path: str) -> bytes:
    return _git(repo_root, "show", f"{revision}:{relative_path}")


def _validate_phase_sequence(repo_root: Path, receipt: _Receipt) -> tuple[str, ...]:
    phases = (receipt.red, receipt.green, receipt.refactor)
    revisions = tuple(
        _full_revision(repo_root, phase.revision, f"{receipt.behavior_id} revision")
        for phase in phases
    )
    if len(set(revisions)) != len(revisions):
        raise ReceiptValidationError("RED, GREEN, and REFACTOR revisions must differ")
    if not _is_ancestor(repo_root, revisions[0], revisions[1]) or not _is_ancestor(
        repo_root, revisions[1], revisions[2]
    ):
        raise ReceiptValidationError(
            "RED, GREEN, and REFACTOR ancestry is not monotonic"
        )
    if not (
        receipt.red.timestamp_utc
        <= receipt.green.timestamp_utc
        <= receipt.refactor.timestamp_utc
    ):
        raise ReceiptValidationError("phase timestamps are not monotonic")
    if not (receipt.red.command == receipt.green.command == receipt.refactor.command):
        raise ReceiptValidationError(
            "GREEN and REFACTOR must run the unchanged RED command"
        )
    if not (receipt.red.cwd == receipt.green.cwd == receipt.refactor.cwd):
        raise ReceiptValidationError(
            "GREEN and REFACTOR must use the unchanged RED cwd"
        )
    if receipt.red.expected_exit == 0 or receipt.red.failure_excerpt is None:
        raise ReceiptValidationError(
            "RED must retain a nonzero exit and failure excerpt"
        )
    if receipt.intended_red_failure not in receipt.red.failure_excerpt:
        raise ReceiptValidationError(
            "failure excerpt does not identify the intended RED failure"
        )
    if receipt.green.expected_exit != 0 or receipt.refactor.expected_exit != 0:
        raise ReceiptValidationError("GREEN and REFACTOR must expect exit zero")
    if (
        receipt.green.failure_excerpt is not None
        or receipt.refactor.failure_excerpt is not None
    ):
        raise ReceiptValidationError("GREEN and REFACTOR failure excerpts must be null")
    return revisions


def _validate_test_blobs(
    repo_root: Path,
    receipt: _Receipt,
    revisions: tuple[str, ...],
) -> None:
    for path, expected_digest in receipt.test_sha256:
        digests = tuple(
            hashlib.sha256(_blob(repo_root, revision, path)).hexdigest()
            for revision in revisions
        )
        if len(set(digests)) != 1:
            raise ReceiptValidationError(
                "test bytes differ between RED, GREEN, and REFACTOR"
            )
        if digests[0] != expected_digest:
            raise ReceiptValidationError(f"test SHA-256 mismatch: {path}")


def _validate_source_blobs(repo_root: Path, receipt: _Receipt) -> None:
    phases = (receipt.red, receipt.green, receipt.refactor)
    for phase in phases:
        for path, expected_digest in phase.source_sha256:
            actual_digest = hashlib.sha256(
                _blob(repo_root, phase.revision, path)
            ).hexdigest()
            if actual_digest != expected_digest:
                raise ReceiptValidationError(
                    f"source SHA-256 mismatch in {phase.revision}: {path}"
                )


def _validate_revision_evidence(repo_root: Path, receipt: _Receipt) -> None:
    revisions = _validate_phase_sequence(repo_root, receipt)
    _validate_test_blobs(repo_root, receipt, revisions)
    _validate_source_blobs(repo_root, receipt)


def _replay_phase(
    repo_root: Path, checkout: Path, phase: _Phase, phase_name: str
) -> None:
    _git(
        repo_root,
        "worktree",
        "add",
        "--detach",
        "--quiet",
        str(checkout),
        phase.revision,
    )
    try:
        execution_root = checkout if phase.cwd == "." else checkout / phase.cwd
        if not execution_root.is_dir():
            raise ReceiptValidationError(
                f"{phase_name} cwd does not exist: {phase.cwd}"
            )
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = subprocess.run(
            phase.command,
            cwd=execution_root,
            env=environment,
            check=False,
            capture_output=True,
        )
        combined = completed.stdout + b"\0" + completed.stderr
        if completed.returncode != phase.expected_exit:
            raise ReceiptValidationError(
                f"{phase_name} replay exit {completed.returncode} != {phase.expected_exit}"
            )
        if hashlib.sha256(combined).hexdigest() != phase.combined_output_sha256:
            raise ReceiptValidationError(f"{phase_name} replay output SHA-256 mismatch")
        if phase_name == "RED":
            decoded = combined.decode("utf-8", errors="replace")
            if phase.failure_excerpt is None or phase.failure_excerpt not in decoded:
                raise ReceiptValidationError(
                    "replay did not reproduce the intended RED failure"
                )
        tracked_diff = _git(checkout, "status", "--porcelain", "--untracked-files=no")
        if tracked_diff:
            raise ReceiptValidationError(f"{phase_name} replay modified tracked files")
    finally:
        _git(repo_root, "worktree", "remove", "--force", str(checkout))


def _replay_receipt(repo_root: Path, receipt: _Receipt) -> None:
    with tempfile.TemporaryDirectory(prefix="simsopt-tdd-replay-") as temporary:
        replay_root = Path(temporary)
        _replay_phase(repo_root, replay_root / "red", receipt.red, "RED")
        _replay_phase(repo_root, replay_root / "green", receipt.green, "GREEN")
        _replay_phase(
            repo_root,
            replay_root / "refactor",
            receipt.refactor,
            "REFACTOR",
        )


def _receipts_from_document(value: object) -> tuple[_Receipt, ...]:
    document = _mapping(value, "receipt document")
    _exact_keys(document, frozenset({"schema_version", "receipts"}), "receipt document")
    if document["schema_version"] != 1:
        raise ReceiptValidationError("unsupported receipt schema version")
    return tuple(
        _receipt(item, f"receipts[{index}]")
        for index, item in enumerate(_sequence(document["receipts"], "receipts"))
    )


def validate_receipt_document(
    value: object,
    *,
    repo_root: Path,
    replay: bool,
) -> ReceiptAudit:
    """Validate trusted receipt JSON and optionally replay every bound phase."""
    receipts = _receipts_from_document(value)
    behavior_ids = tuple(receipt.behavior_id for receipt in receipts)
    if len(set(behavior_ids)) != len(behavior_ids):
        raise ReceiptValidationError("duplicate behavior_id")
    for receipt in receipts:
        _validate_revision_evidence(repo_root, receipt)
        if replay:
            _replay_receipt(repo_root, receipt)
    return ReceiptAudit(
        receipt_count=len(receipts),
        behavior_ids=behavior_ids,
        replayed=replay,
    )


def render_receipt_markdown(value: object) -> str:
    """Render validated-shape receipt data as a deterministic review table."""
    receipts = _receipts_from_document(value)
    lines = [
        "# One-to-One JAX Example TDD Receipts",
        "",
        "Generated from schema v1 machine-readable evidence.",
        "",
        "| Behavior | Native source | Mirror | RED | GREEN | REFACTOR |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(
        f"| `{receipt.behavior_id}` | `{receipt.native_source_id}` | "
        f"`{receipt.mirror_id}` | `{receipt.red.revision}` | "
        f"`{receipt.green.revision}` | `{receipt.refactor.revision}` |"
        for receipt in receipts
    )
    return "\n".join(lines) + "\n"
