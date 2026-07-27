from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest
from examples.jax.tdd_receipts import (
    ReceiptValidationError,
    validate_receipt_document,
)


def _git(repo_root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repo_root: Path, message: str) -> str:
    _git(repo_root, "add", ".")
    _git(repo_root, "commit", "-m", message)
    return _git(repo_root, "rev-parse", "HEAD")


def _blob_sha256(repo_root: Path, revision: str, relative_path: str) -> str:
    payload = subprocess.run(
        ("git", "show", f"{revision}:{relative_path}"),
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def _phase(
    repo_root: Path,
    revision: str,
    *,
    expected_exit: int,
    failure_excerpt: str | None,
) -> dict[str, object]:
    stdout = b""
    stderr = b"mirror behavior absent\n" if expected_exit else b""
    return {
        "revision": revision,
        "command": [sys.executable, "tests/contract_probe.py"],
        "cwd": ".",
        "expected_exit": expected_exit,
        "failure_excerpt": failure_excerpt,
        "combined_output_sha256": hashlib.sha256(stdout + b"\0" + stderr).hexdigest(),
        "source_sha256": {
            "src/state.txt": _blob_sha256(repo_root, revision, "src/state.txt")
        },
        "timestamp_utc": "2026-07-27T12:00:00Z",
    }


def _receipt_document(
    tmp_path: Path,
    *,
    change_green_test: bool = False,
) -> tuple[Path, dict[str, object]]:
    repo_root = tmp_path / "receipt-repo"
    (repo_root / "src").mkdir(parents=True)
    (repo_root / "tests").mkdir()
    _git(repo_root, "init", "--initial-branch=main")
    _git(repo_root, "config", "user.name", "TDD Receipt Test")
    _git(repo_root, "config", "user.email", "tdd-receipt@example.invalid")

    probe = repo_root / "tests" / "contract_probe.py"
    probe.write_text(
        """from pathlib import Path
import sys

state = Path(\"src/state.txt\").read_text(encoding=\"utf-8\").strip()
if state == \"red\":
    print(\"mirror behavior absent\", file=sys.stderr)
    raise SystemExit(7)
""",
        encoding="utf-8",
    )
    state = repo_root / "src" / "state.txt"
    state.write_text("red\n", encoding="utf-8")
    red_revision = _commit(repo_root, "test: add failing mirror contract")

    state.write_text("green\n", encoding="utf-8")
    if change_green_test:
        probe.write_text(probe.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    green_revision = _commit(repo_root, "feat: satisfy mirror contract")

    state.write_text("refactor\n", encoding="utf-8")
    refactor_revision = _commit(repo_root, "refactor: preserve mirror contract")

    test_sha256 = _blob_sha256(repo_root, red_revision, "tests/contract_probe.py")
    document: dict[str, object] = {
        "schema_version": 1,
        "receipts": [
            {
                "behavior_id": "demo-mirror-contract",
                "native_source_id": "1_Simple/demo.py",
                "mirror_id": "demo-mirror",
                "intended_red_failure": "mirror behavior absent",
                "test_sha256": {"tests/contract_probe.py": test_sha256},
                "phases": {
                    "red": _phase(
                        repo_root,
                        red_revision,
                        expected_exit=7,
                        failure_excerpt="mirror behavior absent",
                    ),
                    "green": _phase(
                        repo_root,
                        green_revision,
                        expected_exit=0,
                        failure_excerpt=None,
                    ),
                    "refactor": _phase(
                        repo_root,
                        refactor_revision,
                        expected_exit=0,
                        failure_excerpt=None,
                    ),
                },
            }
        ],
    }
    return repo_root, document


def test_receipt_audit_replays_immutable_red_green_refactor(
    tmp_path: Path,
) -> None:
    repo_root, document = _receipt_document(tmp_path)

    audit = validate_receipt_document(document, repo_root=repo_root, replay=True)

    assert audit.receipt_count == 1
    assert audit.behavior_ids == ("demo-mirror-contract",)
    assert audit.replayed is True


def test_receipt_audit_rejects_test_bytes_changed_after_red(tmp_path: Path) -> None:
    repo_root, document = _receipt_document(tmp_path, change_green_test=True)

    with pytest.raises(
        ReceiptValidationError,
        match="test bytes differ between RED, GREEN, and REFACTOR",
    ):
        validate_receipt_document(document, repo_root=repo_root, replay=False)


def test_receipt_audit_rejects_fabricated_phase_source_hash(tmp_path: Path) -> None:
    repo_root, document = _receipt_document(tmp_path)
    receipts = document["receipts"]
    assert isinstance(receipts, list)
    receipt = receipts[0]
    assert isinstance(receipt, dict)
    phases = receipt["phases"]
    assert isinstance(phases, dict)
    green = phases["green"]
    assert isinstance(green, dict)
    green["source_sha256"] = {"src/state.txt": "0" * 64}

    with pytest.raises(ReceiptValidationError, match="source SHA-256 mismatch"):
        validate_receipt_document(document, repo_root=repo_root, replay=False)


def test_receipt_audit_rejects_wrong_red_failure_discriminator(
    tmp_path: Path,
) -> None:
    repo_root, document = _receipt_document(tmp_path)
    receipts = document["receipts"]
    assert isinstance(receipts, list)
    receipt = receipts[0]
    assert isinstance(receipt, dict)
    receipt["intended_red_failure"] = "different failure"

    with pytest.raises(ReceiptValidationError, match="intended RED failure"):
        validate_receipt_document(document, repo_root=repo_root, replay=True)
