"""Publish and validate custom quasi-Newton measurement receipts.

The runner deliberately writes ignored working directories.  This module is
the small, deterministic boundary that turns one or more runner directories
into a tracked receipt and verifies both the tracked copy and its archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, cast
from urllib.parse import unquote, urlparse

_SCHEMA_VERSION = 1
_JSON_INDENT = 2


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=_JSON_INDENT, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def _archive_path(uri: str, *, repo_root: Path) -> Path:
    if uri.startswith("repo://"):
        return repo_root / uri.removeprefix("repo://")
    parsed = urlparse(uri)
    if parsed.scheme != "file" or not parsed.path:
        raise ValueError(
            f"archive URI must use file:// or repo:// for local validation: {uri!r}"
        )
    return Path(unquote(parsed.path))


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _runner_payload(run: Path) -> dict[str, object]:
    measurements = run / "measurements.json"
    if not measurements.is_file():
        raise ValueError(f"runner directory has no measurements.json: {run}")
    payload = _json_object(measurements)
    rows = payload.get("measurements")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"runner measurements are empty: {measurements}")
    return payload


def _measurement_rows(payload: dict[str, object]) -> list[dict[str, object]]:
    raw_rows = payload["measurements"]
    rows: list[dict[str, object]] = []
    for raw_row in cast(list[object], raw_rows):
        if not isinstance(raw_row, dict):
            raise TypeError("runner measurements must contain JSON objects")
        rows.append(cast(dict[str, object], raw_row))
    return rows


def _all_success(rows: Iterable[dict[str, object]]) -> bool:
    return all(row.get("success") is True for row in rows)


def _relative_artifact_manifest(root: Path) -> list[dict[str, str]]:
    artifacts: list[dict[str, str]] = []
    for path in _iter_files(root):
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        artifacts.append({"path": relative, "sha256": _sha256(path)})
    return artifacts


def _copy_runner_tree(run: Path, destination: Path) -> None:
    for source in _iter_files(run):
        relative = source.relative_to(run)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _summary(
    receipt_id: str,
    run_payloads: list[tuple[Path, dict[str, object]]],
    verdict: str,
) -> str:
    lines = [f"# {receipt_id}", "", f"Verdict: `{verdict}`", ""]
    for run, payload in run_payloads:
        rows = _measurement_rows(payload)
        lines.append(f"## {run.name}")
        for row in rows:
            provider = str(row.get("provider", "unknown"))
            status = row.get("status")
            success = row.get("success")
            iterations = row.get("iterations")
            final_objective = row.get("final_objective")
            lines.append(
                f"- `{provider}`: success={success}, status={status}, "
                f"iterations={iterations}, final objective={final_objective}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def publish(
    runs: tuple[Path, ...],
    *,
    environment_lock: Path,
    destination: Path,
    archive_uri: str,
    repo_root: Path,
) -> Path:
    """Publish runner directories atomically and return the destination."""

    if not runs:
        raise ValueError("at least one --run directory is required")
    if not environment_lock.is_file():
        raise ValueError(f"environment lock does not exist: {environment_lock}")
    if destination.exists():
        raise FileExistsError(f"receipt destination already exists: {destination}")

    run_payloads = [(run, _runner_payload(run)) for run in runs]
    run_names = [run.name for run, _payload in run_payloads]
    if len(set(run_names)) != len(run_names):
        raise ValueError("runner directory names must be unique")
    archive = _archive_path(archive_uri, repo_root=repo_root)
    if destination.resolve() == archive.resolve():
        raise ValueError(
            "receipt archive must be distinct from the tracked destination"
        )
    if archive.exists():
        raise FileExistsError(f"receipt archive destination already exists: {archive}")

    candidate_commits = {
        str(payload.get("git_commit")) for _run, payload in run_payloads
    }
    clean_values = [
        bool(payload.get("git_clean", False)) for _run, payload in run_payloads
    ]
    rows = [row for _run, payload in run_payloads for row in _measurement_rows(payload)]
    verdict = (
        "pass"
        if len(candidate_commits) == 1 and all(clean_values) and _all_success(rows)
        else "diagnostic-pass-not-promotion"
    )
    lock_sha256 = _sha256(environment_lock)

    destination_parent = destination.parent
    destination_parent.mkdir(parents=True, exist_ok=True)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination_parent
    ) as destination_tmp_name, tempfile.TemporaryDirectory(
        prefix=f".{archive.name}.", dir=archive.parent
    ) as archive_tmp_name:
        destination_tmp = Path(destination_tmp_name)
        archive_tmp = Path(archive_tmp_name)
        raw_root = destination_tmp / "raw"
        for run, _payload in run_payloads:
            _copy_runner_tree(run, raw_root / run.name)

        metrics = {
            "schema_version": _SCHEMA_VERSION,
            "receipt_id": destination.name,
            "source_runs": [run.name for run, _payload in run_payloads],
            "candidate_commits": sorted(candidate_commits),
            "candidate_worktree_clean": all(clean_values),
            "environment_lock": str(environment_lock),
            "environment_lock_sha256": lock_sha256,
            "measurements": rows,
            "verdict": verdict,
        }
        _write_json(destination_tmp / "metrics.json", metrics)
        (destination_tmp / "summary.md").write_text(
            _summary(destination.name, run_payloads, verdict), encoding="utf-8"
        )
        manifest = {
            "schema_version": _SCHEMA_VERSION,
            "receipt_id": destination.name,
            "kind": "custom-quasi-newton-runner",
            "candidate_commit": (
                next(iter(candidate_commits)) if len(candidate_commits) == 1 else None
            ),
            "candidate_worktree_clean": all(clean_values),
            "environment_lock": {
                "path": str(environment_lock),
                "sha256": lock_sha256,
            },
            "source_runs": [run.name for run, _payload in run_payloads],
            "archive_uri": archive_uri,
            "artifacts": _relative_artifact_manifest(destination_tmp),
            "verdict": verdict,
        }
        _write_json(destination_tmp / "manifest.json", manifest)
        _copy_runner_tree(destination_tmp, archive_tmp)
        destination_tmp.replace(destination)
        archive_tmp.replace(archive)
    return destination


def _validate_receipt(receipt: Path, *, repo_root: Path) -> None:
    manifest = _json_object(receipt / "manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise TypeError(f"manifest artifacts are invalid: {receipt}")
    archive_uri = manifest.get("archive_uri")
    if not isinstance(archive_uri, str):
        raise TypeError(f"manifest archive URI is invalid: {receipt}")
    archive = _archive_path(archive_uri, repo_root=repo_root)
    environment_lock = manifest.get("environment_lock")
    if isinstance(environment_lock, dict):
        lock_record = cast(dict[str, object], environment_lock)
        lock_path_value = lock_record.get("path")
        lock_sha256 = lock_record.get("sha256")
        if not isinstance(lock_path_value, str) or not isinstance(lock_sha256, str):
            raise TypeError(f"manifest environment lock is invalid: {receipt}")
        lock_path = Path(lock_path_value)
        if not lock_path.is_absolute():
            lock_path = repo_root / lock_path
        if not lock_path.is_file():
            raise FileNotFoundError(f"environment lock missing: {lock_path}")
        if _sha256(lock_path) != lock_sha256:
            raise ValueError(f"environment lock checksum mismatch: {lock_path}")
    for raw_artifact in cast(list[object], artifacts):
        if not isinstance(raw_artifact, dict):
            raise TypeError(f"manifest artifact entry is invalid: {receipt}")
        artifact = cast(dict[str, object], raw_artifact)
        relative = artifact.get("path")
        expected = artifact.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError(f"manifest artifact entry is invalid: {receipt}")
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"manifest artifact path escapes receipt: {relative}")
        tracked = receipt / relative_path
        archived = archive / relative_path
        for path in (tracked, archived):
            if not path.is_file():
                raise FileNotFoundError(f"receipt artifact missing: {path}")
            actual = _sha256(path)
            if actual != expected:
                raise ValueError(
                    f"receipt artifact checksum mismatch: {path} {actual} != {expected}"
                )


def validate_all(root: Path, *, repo_root: Path) -> int:
    manifests = sorted(root.rglob("manifest.json"))
    if not manifests:
        raise ValueError(f"no receipt manifests found under {root}")
    for manifest in manifests:
        _validate_receipt(manifest.parent, repo_root=repo_root)
    print(json.dumps({"validated": len(manifests), "root": str(root)}))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish_parser = subparsers.add_parser("publish")
    publish_parser.add_argument("--run", type=Path, action="append", required=True)
    publish_parser.add_argument("--environment-lock", type=Path, required=True)
    publish_parser.add_argument("--destination", type=Path, required=True)
    publish_parser.add_argument("--archive-uri", required=True)
    publish_parser.add_argument("--repo-root", type=Path, default=Path.cwd())

    validate_parser = subparsers.add_parser("validate-all")
    validate_parser.add_argument("--root", type=Path, required=True)
    validate_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "publish":
        destination = publish(
            tuple(args.run),
            environment_lock=args.environment_lock,
            destination=args.destination,
            archive_uri=args.archive_uri,
            repo_root=args.repo_root,
        )
        print(destination)
        return 0
    return validate_all(args.root, repo_root=args.repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
