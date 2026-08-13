"""Rewrite the execution-source manifest, refusing undeclared membership drift.

The manifest at ``DIAG4_EXECUTION_SOURCE_MANIFEST_PATH`` is the repository's
single statement of which bytes may execute in a certified run, and every
certification lane asks it after its imports have happened.  Regenerating it is
therefore not a formatting pass: adding a path admits new bytes to the
certified surface, and dropping one silently removes a gate.

This tool exists because that decision must be deliberate.  It recomputes each
member's digest and size, and it refuses to write when the membership the
repository's own rule produces differs from the manifest's current membership
in any way the operator did not name with ``--admit``.  A new file under
``benchmarks/``, ``examples/`` or ``src/`` is exactly such a difference, so a
refreeze that forgot to bump the frozen counts fails here instead of shipping a
manifest no gate agrees with.

The membership rule itself is NOT restated here: it is imported from the
successor-authority module that the certification gates validate against.  A
second spelling of that rule is the twin-constant failure this campaign has
already paid for twice (mistake-book P153).

Run from the repository root with the repository and ``src`` on ``PYTHONPATH``:

    python benchmarks/regenerate_execution_source_manifest.py \\
        --admit benchmarks/new_execution_source.py --expect-count 614
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Final

from benchmarks.single_stage_fullspace_snapshot import (
    JsonValue,
    canonical_json_bytes,
    load_canonical_json_bytes,
)
from benchmarks.single_stage_native_equivalent_quality_successor_authority import (
    DIAG4_EXECUTION_SOURCE_MANIFEST_PATH,
    DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION,
    DIAG5_FROZEN_NUMERICAL_PATHS,
    DIAG5_QUALIFIED_FILE_PATHS,
    _diag4_execution_source_membership,
)

REPOSITORY_ROOT: Final = Path(__file__).resolve(strict=True).parents[1]


class ManifestRegenerationError(RuntimeError):
    """The regeneration refused; the manifest on disk is left untouched."""


def execution_source_membership(repository: Path) -> frozenset[str]:
    """Every path the manifest's membership rule selects in this repository."""

    return _diag4_execution_source_membership(
        repository,
        qualified=dict.fromkeys(DIAG5_QUALIFIED_FILE_PATHS, ""),
        frozen=dict.fromkeys(DIAG5_FROZEN_NUMERICAL_PATHS, ""),
    )


def regenerate_execution_source_manifest(
    *,
    expected_count: int,
    admitted: Sequence[str] = (),
    repository: Path = REPOSITORY_ROOT,
) -> tuple[bytes, tuple[str, ...]]:
    """Build the manifest bytes this repository's membership rule describes.

    Returns the canonical document and the members whose recorded entry the
    regeneration changes, so a caller can show the operator what moved.  Every
    refusal names the paths responsible: a manifest is reviewed by reading the
    difference, not by rerunning the tool.
    """

    manifest_path = repository / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH
    document = load_canonical_json_bytes(manifest_path.read_bytes())
    if not isinstance(document, dict) or not isinstance(document.get("entries"), dict):
        raise ManifestRegenerationError("execution-source manifest is not an entry map")
    previous: dict[str, JsonValue] = document["entries"]

    admitted_paths = frozenset(admitted)
    already_member = sorted(admitted_paths & frozenset(previous))
    if already_member:
        raise ManifestRegenerationError(
            f"admitted paths are already members: {already_member}"
        )
    membership = execution_source_membership(repository)
    unselected = sorted(admitted_paths - membership)
    if unselected:
        raise ManifestRegenerationError(
            f"admitted paths the membership rule does not select: {unselected}"
        )
    declared = frozenset(previous) | admitted_paths
    entering = sorted(membership - declared)
    leaving = sorted(declared - membership)
    if entering or leaving:
        raise ManifestRegenerationError(
            "membership changed without being declared: "
            f"entering={entering} leaving={leaving}"
        )
    if len(membership) != expected_count:
        raise ManifestRegenerationError(
            f"membership holds {len(membership)} paths, --expect-count names "
            f"{expected_count}"
        )

    entries: dict[str, JsonValue] = {}
    changed: list[str] = []
    for relative in sorted(membership):
        source = repository / relative
        if not source.is_file():
            raise ManifestRegenerationError(f"member is not a file on disk: {relative}")
        payload = source.read_bytes()
        entry = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
        }
        entries[relative] = entry
        if entry != previous.get(relative):
            changed.append(relative)
    return (
        canonical_json_bytes(
            {
                "entries": entries,
                "entries_sha256": hashlib.sha256(
                    canonical_json_bytes(entries)
                ).hexdigest(),
                "schema_version": DIAG4_EXECUTION_SOURCE_SCHEMA_VERSION,
            }
        ),
        tuple(changed),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--admit",
        action="append",
        default=[],
        metavar="RELATIVE_PATH",
        help="repository-relative path deliberately entering the manifest",
    )
    parser.add_argument(
        "--expect-count",
        type=int,
        required=True,
        help="entry count the refreeze's frozen constants were bumped to",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    body, changed = regenerate_execution_source_manifest(
        expected_count=arguments.expect_count, admitted=arguments.admit
    )
    (REPOSITORY_ROOT / DIAG4_EXECUTION_SOURCE_MANIFEST_PATH).write_bytes(body)
    document = load_canonical_json_bytes(body)
    assert isinstance(document, dict)
    print(f"entries: {arguments.expect_count}  changed: {len(changed)}")
    for relative in changed:
        print(f"  updated: {relative}")
    print(f"entries_sha256: {document['entries_sha256']}")
    print(f"manifest_sha256: {hashlib.sha256(body).hexdigest()}")
    return 0


__all__ = (
    "REPOSITORY_ROOT",
    "ManifestRegenerationError",
    "execution_source_membership",
    "main",
    "regenerate_execution_source_manifest",
)


if __name__ == "__main__":
    raise SystemExit(main())
