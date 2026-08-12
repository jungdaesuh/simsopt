"""Rebind a copied immutable snapshot to its canonical destination root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping, Sequence

from benchmarks.single_stage_compute_graph_phase0_workflow import (
    Phase0WorkflowError,
    _publication,
)
from benchmarks.single_stage_compute_graph_snapshot import (
    SnapshotError,
    canonical_json_bytes,
)


class SnapshotRelocationError(RuntimeError):
    """A snapshot relocation publication failed validation."""


def _canonical_document(path: Path) -> Mapping[str, object]:
    raw = path.read_bytes()
    try:
        document = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise SnapshotRelocationError("source publication is not valid JSON") from error
    if not isinstance(document, dict) or not all(
        isinstance(key, str) for key in document
    ):
        raise SnapshotRelocationError("source publication must be a JSON object")
    if canonical_json_bytes(document) != raw:
        raise SnapshotRelocationError("source publication is not canonical JSON")
    return document


def relocate_snapshot_publication(
    *,
    source_publication_path: Path,
    destination_snapshot_root: Path,
    output_path: Path,
) -> Path:
    """Publish a root-rebound record only after full destination validation."""

    destination_snapshot_root = destination_snapshot_root.absolute()
    output_path = output_path.absolute()
    staging_path = output_path.with_name(f".{output_path.name}.partial")
    if output_path.exists():
        raise SnapshotRelocationError("relocated publication output already exists")
    if staging_path.exists():
        raise SnapshotRelocationError("relocated publication staging already exists")
    source_document = _canonical_document(source_publication_path)
    relocated = {
        **source_document,
        "snapshot_root": str(destination_snapshot_root),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with staging_path.open("xb") as stream:
        stream.write(canonical_json_bytes(relocated))
    try:
        _publication(destination_snapshot_root, staging_path)
    except (OSError, Phase0WorkflowError, SnapshotError) as error:
        raise SnapshotRelocationError(str(error)) from error
    staging_path.rename(output_path)
    return output_path


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-publication", type=Path, required=True)
    parser.add_argument("--destination-snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    options = _parse_args(argv)
    try:
        output = relocate_snapshot_publication(
            source_publication_path=options.source_publication,
            destination_snapshot_root=options.destination_snapshot_root,
            output_path=options.output,
        )
    except (OSError, SnapshotRelocationError) as error:
        sys.stderr.write(f"Snapshot relocation failed: {error}\n")
        return 2
    sys.stdout.write(f"{output}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
