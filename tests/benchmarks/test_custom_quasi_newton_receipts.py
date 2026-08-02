from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from benchmarks.custom_quasi_newton_receipts import publish, validate_all


def _runner_directory(root: Path, *, clean: bool = True) -> Path:
    run = root / "runner-case"
    run.mkdir()
    (run / "raw").mkdir()
    (run / "raw" / "stdout.json").write_text("{}\n", encoding="utf-8")
    (run / "measurements.json").write_text(
        json.dumps(
            {
                "git_commit": "abc123",
                "git_clean": clean,
                "measurements": [
                    {
                        "provider": "custom",
                        "iterations": 2,
                        "status": 0,
                        "success": True,
                        "final_objective": 1.25e-12,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return run


def _publish_receipt(tmp_path: Path, *, clean: bool = True) -> tuple[Path, Path]:
    run = _runner_directory(tmp_path, clean=clean)
    lock = tmp_path / "environment.lock"
    lock.write_text("python==3.11\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "receipt"
    archive = tmp_path / "archive" / "receipt"
    publish(
        (run,),
        environment_lock=lock,
        destination=destination,
        archive_uri=archive.as_uri(),
        repo_root=tmp_path,
    )
    return destination, archive


def test_publish_and_validate_receipt_from_a_fresh_process(tmp_path: Path) -> None:
    destination, _archive = _publish_receipt(tmp_path)

    assert validate_all(destination, repo_root=tmp_path) == 0
    completed = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).resolve().parents[2]
                / "benchmarks"
                / "custom_quasi_newton_receipts.py"
            ),
            "validate-all",
            "--root",
            str(destination),
            "--repo-root",
            str(tmp_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"validated": 1' in completed.stdout


def test_publish_marks_dirty_runner_diagnostic(tmp_path: Path) -> None:
    destination, _archive = _publish_receipt(tmp_path, clean=False)
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["candidate_worktree_clean"] is False
    assert manifest["verdict"] == "diagnostic-pass-not-promotion"


def test_publish_rejects_missing_environment_lock(tmp_path: Path) -> None:
    run = _runner_directory(tmp_path)
    with pytest.raises(ValueError, match="environment lock does not exist"):
        publish(
            (run,),
            environment_lock=tmp_path / "missing.lock",
            destination=tmp_path / "tracked" / "receipt",
            archive_uri=(tmp_path / "archive" / "receipt").as_uri(),
            repo_root=tmp_path,
        )


def test_publish_rejects_archive_aliasing_tracked_destination(tmp_path: Path) -> None:
    run = _runner_directory(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("python==3.11\n", encoding="utf-8")
    destination = tmp_path / "tracked" / "receipt"

    with pytest.raises(ValueError, match="archive must be distinct"):
        publish(
            (run,),
            environment_lock=lock,
            destination=destination,
            archive_uri=destination.as_uri(),
            repo_root=tmp_path,
        )

    assert not destination.exists()


@pytest.mark.parametrize("location", ["tracked", "archive"])
def test_validate_all_rejects_artifact_tampering(tmp_path: Path, location: str) -> None:
    destination, archive = _publish_receipt(tmp_path)
    target = (destination if location == "tracked" else archive) / "metrics.json"
    target.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_all(destination, repo_root=tmp_path)


def test_validate_all_rejects_environment_lock_tampering(tmp_path: Path) -> None:
    destination, _archive = _publish_receipt(tmp_path)
    lock = tmp_path / "environment.lock"
    lock.write_text("python==3.12\n", encoding="utf-8")

    with pytest.raises(ValueError, match="environment lock checksum mismatch"):
        validate_all(destination, repo_root=tmp_path)
