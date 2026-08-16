"""Drift gate for ``docs/jax_example_device_assignment.md``.

That document's single assignment table is the source of truth for which
device each ``examples/jax`` example should be launched on. These tests keep
it in exact one-to-one correspondence with ``examples/jax/manifest.json`` and
keep every claim grounded:

* only the four declared device values and the four declared evidence classes;
* every ``cpu`` row opens with one of three mechanism families, so the reason
  the GPU loses cannot be buried, negated, or implied;
* every ``gpu`` row — and every ``measured-certified`` row — cites a
  git-tracked regular file under ``docs/receipts/``, which is the only kind of
  evidence a reader can open from a clone;
* device and evidence class do not contradict each other;
* no cited in-repo path is missing;
* the document's own legends still enumerate exactly the values accepted here,
  so the vocabulary cannot be deleted out from under the gate.

Adding an example to the manifest without assigning it a device fails here.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT_DOC_PATH = REPO_ROOT / "docs" / "jax_example_device_assignment.md"
MANIFEST_PATH = REPO_ROOT / "examples" / "jax" / "manifest.json"

# Only evidence committed under this prefix can support a performance claim:
# it is the one location a reviewer can read without access to this
# workstation's campaign directories.
RECEIPT_PREFIX = "docs/receipts/"

# The assignment table is located by these header cells, lowercased. Anchoring
# on the header rather than on a position keeps the parse stable when prose or
# further tables are added around it.
TABLE_HEADER_CELLS = (
    "example id",
    "device",
    "evidence class",
    "mechanism / receipt",
)

ALLOWED_DEVICES = ("cpu", "gpu", "either", "unmeasured")
ALLOWED_EVIDENCE_CLASSES = (
    "measured-certified",
    "measured-diagnostic",
    "census-structural",
    "unmeasured",
)
MECHANISM_FAMILIES = ("sequential chain", "tiny problem", "narrow matrix")

# Each vocabulary above must stay enumerated by the document section named
# here, as ``- **value** — explanation`` bullets. Deleting or editing a legend
# without updating this module fails ``test_document_legends_enumerate_...``.
LEGEND_SECTIONS = {
    "Device values": ALLOWED_DEVICES,
    "Evidence-class legend": ALLOWED_EVIDENCE_CLASSES,
    "Mechanism families": MECHANISM_FAMILIES,
}

# Device values that assert a measured or reasoned placement, and so may not
# be paired with the ``unmeasured`` evidence class.
PLACED_DEVICES = ("cpu", "gpu")

_BACKTICKED_SPAN = re.compile(r"`([^`]+)`")
_LEGEND_BULLET = re.compile(r"^- \*\*([^*]+)\*\*")


@dataclass(frozen=True)
class AssignmentRow:
    """One example's device assignment, exactly as the doc states it."""

    example_id: str
    device: str
    evidence_class: str
    mechanism: str


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _parse_assignment_table(document: str) -> list[AssignmentRow]:
    """Return the assignment rows in document order.

    Finds the unique header row whose cells are ``TABLE_HEADER_CELLS``, skips
    the alignment row that follows it, and consumes every subsequent line that
    still starts a table row. Raises if the anchor is missing or duplicated,
    because a silently empty parse would make every assertion below vacuous.
    """
    lines = document.splitlines()
    header_indices = [
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("|")
        and tuple(cell.lower() for cell in _split_table_row(line)) == TABLE_HEADER_CELLS
    ]
    if len(header_indices) != 1:
        raise AssertionError(
            f"{ASSIGNMENT_DOC_PATH.name} must contain exactly one table whose "
            f"header is {' | '.join(TABLE_HEADER_CELLS)}; found "
            f"{len(header_indices)}."
        )

    rows: list[AssignmentRow] = []
    for line in lines[header_indices[0] + 2 :]:
        if not line.lstrip().startswith("|"):
            break
        cells = _split_table_row(line)
        if len(cells) != len(TABLE_HEADER_CELLS):
            raise AssertionError(
                f"assignment row {line!r} has {len(cells)} cells, expected "
                f"{len(TABLE_HEADER_CELLS)}."
            )
        rows.append(AssignmentRow(*cells))
    if not rows:
        raise AssertionError(
            f"{ASSIGNMENT_DOC_PATH.name} has an assignment table header with "
            "no rows under it."
        )
    return rows


def _parse_legend_values(document: str, section_title: str) -> list[str]:
    """Values enumerated by one ``## <section_title>`` legend, in order.

    A legend value is the bolded lead of a top-level bullet; prose emphasis
    elsewhere in the section is ignored. Missing sections return an empty
    list so the caller can report the deletion as the failure it is.
    """
    values: list[str] = []
    in_section = False
    for line in document.splitlines():
        if line.startswith("## "):
            if in_section:
                break
            in_section = line[3:].strip() == section_title
            continue
        if not in_section:
            continue
        match = _LEGEND_BULLET.match(line)
        if match is not None:
            values.append(match.group(1).strip())
    return values


def _cited_in_repo_paths(cell: str) -> list[str]:
    """Repository-relative paths cited inside one table cell.

    A backticked span counts as an in-repo path when it contains ``/`` and is
    neither host-absolute (``/`` or ``~`` rooted) nor a URL. Host-local
    campaign directories and bare identifiers are therefore left alone.
    """
    return [
        span
        for span in _BACKTICKED_SPAN.findall(cell)
        if "/" in span and not span.startswith(("/", "~", "http://", "https://"))
    ]


@lru_cache(maxsize=1)
def _tracked_receipt_files() -> frozenset[str]:
    """Every git-tracked regular file under ``docs/receipts/``.

    ``git ls-files`` lists blobs only, so directories and untracked working-
    tree files are absent by construction — which is exactly the distinction
    a receipt citation has to survive.
    """
    listing = subprocess.run(
        ["git", "ls-files", "-z", "--", RECEIPT_PREFIX],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return frozenset(entry for entry in listing.split("\0") if entry)


def _cited_receipts(cell: str) -> list[str]:
    """Citations in one cell that are tracked receipt files."""
    tracked = _tracked_receipt_files()
    return [path for path in _cited_in_repo_paths(cell) if path in tracked]


ASSIGNMENT_DOC = ASSIGNMENT_DOC_PATH.read_text(encoding="utf-8")
ASSIGNMENT_ROWS = _parse_assignment_table(ASSIGNMENT_DOC)
MANIFEST_EXAMPLE_IDS = [
    example["id"]
    for example in json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))["jax_examples"]
]


def test_every_assigned_example_exists_in_the_manifest() -> None:
    known_ids = set(MANIFEST_EXAMPLE_IDS)
    unknown = [
        row.example_id for row in ASSIGNMENT_ROWS if row.example_id not in known_ids
    ]
    assert not unknown, (
        "assignment table names examples that examples/jax/manifest.json does "
        f"not define: {sorted(unknown)}. Remove the rows or fix the ids."
    )


def test_every_manifest_example_is_assigned_exactly_once() -> None:
    assigned = [row.example_id for row in ASSIGNMENT_ROWS]
    missing = [
        example_id for example_id in MANIFEST_EXAMPLE_IDS if example_id not in assigned
    ]
    duplicated = sorted({row for row in assigned if assigned.count(row) > 1})
    assert not missing, (
        "manifest examples with no device assignment: "
        f"{missing}. Add a row to {ASSIGNMENT_DOC_PATH.name} stating which "
        "device the example should run on, or mark it unmeasured."
    )
    assert not duplicated, (
        f"examples assigned more than once: {duplicated}. One row per example."
    )


def test_device_column_holds_only_declared_values() -> None:
    offenders = {
        row.example_id: row.device
        for row in ASSIGNMENT_ROWS
        if row.device not in ALLOWED_DEVICES
    }
    assert not offenders, f"device values outside {list(ALLOWED_DEVICES)}: {offenders}."


def test_evidence_class_column_holds_only_declared_values() -> None:
    offenders = {
        row.example_id: row.evidence_class
        for row in ASSIGNMENT_ROWS
        if row.evidence_class not in ALLOWED_EVIDENCE_CLASSES
    }
    assert not offenders, (
        "evidence classes outside the doc's legend "
        f"{list(ALLOWED_EVIDENCE_CLASSES)}: {offenders}."
    )


def test_gpu_rows_cite_a_tracked_receipt_file() -> None:
    uncited = [
        row.example_id
        for row in ASSIGNMENT_ROWS
        if row.device == "gpu" and not _cited_receipts(row.mechanism)
    ]
    assert not uncited, (
        "gpu assignments without a git-tracked receipt file under "
        f"{RECEIPT_PREFIX}: {uncited}. A gpu row is a performance claim; a "
        "host-local campaign directory, an untracked path, or a directory is "
        "not evidence a reviewer can open. Downgrade the row to unmeasured "
        "until the receipt is committed."
    )


def test_measured_certified_rows_cite_a_tracked_receipt_file() -> None:
    uncited = [
        row.example_id
        for row in ASSIGNMENT_ROWS
        if row.evidence_class == "measured-certified"
        and not _cited_receipts(row.mechanism)
    ]
    assert not uncited, (
        "measured-certified rows without a git-tracked receipt file under "
        f"{RECEIPT_PREFIX}: {uncited}. Certification is the receipt; without "
        "one the row is at most measured-diagnostic."
    )


def test_unmeasured_device_rows_claim_no_measurement() -> None:
    incoherent = {
        row.example_id: row.evidence_class
        for row in ASSIGNMENT_ROWS
        if row.device == "unmeasured"
        and row.evidence_class not in ("unmeasured", "census-structural")
    }
    assert not incoherent, (
        "rows left unplaced while claiming a measurement: "
        f"{incoherent}. If the example was measured, assign it a device."
    )


def test_placed_device_rows_carry_evidence() -> None:
    incoherent = {
        row.example_id: row.device
        for row in ASSIGNMENT_ROWS
        if row.device in PLACED_DEVICES and row.evidence_class == "unmeasured"
    }
    assert not incoherent, (
        "rows assigned a device on unmeasured evidence: "
        f"{incoherent}. Ground the row or set the device to unmeasured."
    )


def test_cpu_rows_open_with_a_mechanism_family() -> None:
    unexplained = [
        row.example_id
        for row in ASSIGNMENT_ROWS
        if row.device == "cpu" and not row.mechanism.startswith(MECHANISM_FAMILIES)
    ]
    assert not unexplained, (
        "cpu assignments whose mechanism cell does not open with one of "
        f"{list(MECHANISM_FAMILIES)}: {unexplained}. The family must lead the "
        "cell, so that a negated or incidental mention cannot pass for a "
        "reason."
    )


def test_every_cited_in_repo_path_exists() -> None:
    dangling = {
        row.example_id: [
            path
            for path in _cited_in_repo_paths(row.mechanism)
            if not (REPO_ROOT / path).exists()
        ]
        for row in ASSIGNMENT_ROWS
    }
    dangling = {key: paths for key, paths in dangling.items() if paths}
    assert not dangling, (
        f"assignment rows cite repository paths that do not exist: {dangling}."
    )


def test_document_legends_enumerate_the_accepted_vocabularies() -> None:
    drifted = {
        title: _parse_legend_values(ASSIGNMENT_DOC, title)
        for title, expected in LEGEND_SECTIONS.items()
        if tuple(_parse_legend_values(ASSIGNMENT_DOC, title)) != expected
    }
    expected_vocabularies = {
        title: list(expected) for title, expected in LEGEND_SECTIONS.items()
    }
    assert not drifted, (
        "the document's legend sections no longer enumerate exactly the "
        f"values this test accepts {expected_vocabularies}; found {drifted}. "
        "A deleted or reworded legend leaves readers with a vocabulary only "
        "the test knows."
    )
