"""Fail-closed gate over the native-test coverage manifest.

Green side: the repository manifest passes every check in
``scripts/jax_native_unit_coverage.py``, which implements the coverage
contract formerly defined by
``docs/jax_native_unit_test_coverage_implementation_plan.md`` (Draft,
2026-07-29; commit 6fec6e4ca; removed from this branch by the 2026-08-24 docs
curation).

RED side: the contract requires "RED tests first" — proof that the validator
rejects every one of the following manifest defects. Each RED test corrupts
an in-memory or temp-file copy of the manifest (or a throwaway file for the
two AST-level checks) and asserts the rejection, so the checked-in manifest
and its generated document are never touched:

1. an omitted native test file (``native_surface_complete``)
2. a row for a file outside the pinned native surface (``unknown_row``)
3. a stale evidence path (``stale_path``)
4. an evidence line number past the end of its file (``stale_path``)
5. an empty reason (``empty_reason``)
6. a generic stand-in reason such as "unsupported" (``empty_reason``)
7. a stale ``source_tree_hash`` (``source_tree_drift``)
8. an orphan JAX-side ``test_ids`` entry (``orphan_test_id``)
9. a capability referenced by no native file row (``capability_mapping`` /
   ``orphan_reference``)
10. an unreviewed ``native_only`` decision (``unreviewed_decision``)
11. a disposition outside the contract's vocabulary
    (``disposition_vocabulary``)
12. a CLI invocation on a corrupted manifest (non-zero exit, no traceback)
13. a ``native_test_ids``/``test_ids`` entry naming the wrong class for an
    otherwise-real function (``orphan_test_id``)
14. a ``def`` name that is structurally real only inside a docstring, never
    actually defined (``orphan_test_id``; proven by AST parsing, not regex)
15. a hand-edited generated document that no longer matches
    ``render_document(manifest)`` (``generated_doc_drift``)
16. a pinned native-surface file missing from the working tree at hash time
    (``stale_path``, not an uncaught ``FileNotFoundError``)
17. a capability missing a required field ``render_document`` dereferences
    unconditionally, such as ``title`` (``manifest_schema``)
18. a duplicate manifest row for the same path (``duplicate_row``)
19. a missing ``baseline.baseline_commit`` (``manifest_schema``)
20. a ``jax_missing`` capability that declares a passing jax lane
    (``jax_missing_lane_contradiction``)
21. a ``shared_python`` capability without a decision record
    (``unreviewed_decision``)
22. a stale evidence anchor that no longer appears in the cited file
    (``stale_path``)
23. a central ``tolerance_owner`` claim whose bound test file never imports
    ``simsopt_jax.parity_tolerances`` (``tolerance_owner_unsupported``)
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))
sys.modules.pop("scripts", None)

from scripts import jax_native_unit_coverage as COV  # noqa: E402


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    return COV.load_manifest(COV.MANIFEST_PATH)


def _failures(corrupted: dict[str, Any]) -> list[str]:
    return COV.validate(REPO_ROOT, corrupted)


def _assert_rejected(corrupted: dict[str, Any], check: str) -> str:
    """Assert the validator emits ``check`` and return the message it emitted."""
    failures = _failures(corrupted)
    matching = [failure for failure in failures if failure.startswith(f"{check}:")]
    assert matching, (
        f"the validator accepted a manifest it must reject: expected a {check!r} "
        f"failure, got {failures or 'no failures at all'}"
    )
    return matching[0]


def _row(manifest: dict[str, Any], path: str) -> dict[str, Any]:
    for row in manifest["native_test_files"]:
        if row["path"] == path:
            return row
    raise AssertionError(f"{path} has no manifest row")


def _capability(manifest: dict[str, Any], capability_id: str) -> dict[str, Any]:
    for capability in manifest["capabilities"]:
        if capability["id"] == capability_id:
            return capability
    raise AssertionError(f"{capability_id} is not in the manifest")


def test_repository_manifest_satisfies_every_fail_closed_check(manifest):
    """The checked-in manifest passes the validator with no violations."""
    assert COV.validate(REPO_ROOT, manifest) == []


def test_check_mode_exits_zero_without_touching_the_artifacts():
    """Default (read-only) mode reports success and rewrites nothing."""
    manifest_before = COV.MANIFEST_PATH.read_bytes()
    doc_before = COV.DOC_PATH.read_bytes()

    assert COV.main([]) == 0

    assert COV.MANIFEST_PATH.read_bytes() == manifest_before
    assert COV.DOC_PATH.read_bytes() == doc_before


def test_generated_document_is_reproducible_from_the_manifest(manifest):
    """The committed document is exactly what the manifest renders to."""
    surface = COV.native_test_surface(
        REPO_ROOT, manifest["baseline"]["upstream_authority_commit"]
    )
    assert COV.render_document(manifest, surface) == COV.DOC_PATH.read_text(
        encoding="utf-8"
    )


def test_every_native_surface_file_has_exactly_one_row(manifest):
    """The manifest covers the pinned native surface once per file."""
    surface = COV.native_test_surface(
        REPO_ROOT, manifest["baseline"]["upstream_authority_commit"]
    )
    paths = [row["path"] for row in manifest["native_test_files"]]

    assert sorted(paths) == sorted(surface)
    assert len(paths) == len(set(paths))


def test_unclassified_rows_are_listed_and_never_counted_as_covered(manifest):
    """`unclassified` is a listed planning state, not a covered state.

    Constructed on a copy, not asserted against the live manifest's current
    classification state: the live manifest reaching full classification
    (zero unclassified rows) would make a live-state assertion here false
    without the underlying guarantee -- that a row FORCED unclassified is
    listed under "Unclassified" and excluded from "Classified" -- changing
    at all.
    """
    constructed = copy.deepcopy(manifest)
    surface = COV.native_test_surface(
        REPO_ROOT, constructed["baseline"]["upstream_authority_commit"]
    )
    target_path = constructed["native_test_files"][0]["path"]
    _row(constructed, target_path).update(
        {
            "capabilities": [],
            "disposition": COV.UNCLASSIFIED,
            "evidence": [],
            "reason": "",
        }
    )

    document = COV.render_document(constructed, surface)

    unclassified_section = document.split("## Unclassified native test files")[1]
    assert f"`{target_path}`" in unclassified_section, (
        f"{target_path} was forced unclassified but is not listed"
    )
    classified_section = document.split("## Classified native test files")[1].split(
        "## Unclassified native test files"
    )[0]
    assert f"`{target_path}`" not in classified_section, (
        f"{target_path} is unclassified but still counted as classified"
    )
    assert COV.UNCLASSIFIED not in COV.ALLOWED_DISPOSITIONS


def test_red_omitted_native_test_file_is_rejected(manifest):
    """Dropping a native file's row fails, so upstream tests cannot go missing."""
    corrupted = copy.deepcopy(manifest)
    dropped = corrupted["native_test_files"].pop(0)["path"]

    message = _assert_rejected(corrupted, "native_surface_complete")

    assert dropped in message


def test_red_unknown_row_is_rejected(manifest):
    """A row for a file outside the pinned native surface fails."""
    corrupted = copy.deepcopy(manifest)
    corrupted["native_test_files"].append(
        {
            "path": "tests/jax/native_unit_parity/test_force_parity.py",
            "domain": "force",
            "disposition": COV.UNCLASSIFIED,
            "capabilities": [],
            "reason": "",
            "evidence": [],
        }
    )

    message = _assert_rejected(corrupted, "unknown_row")

    assert "test_force_parity.py" in message


def test_red_stale_evidence_path_is_rejected(manifest):
    """Evidence pointing at a file that no longer exists fails."""
    corrupted = copy.deepcopy(manifest)
    _capability(corrupted, "MF-1")["evidence"] = [
        "src/simsopt/field/selffield_deleted_by_a_refactor.py:18"
    ]

    message = _assert_rejected(corrupted, "stale_path")

    assert "selffield_deleted_by_a_refactor.py" in message


def test_red_evidence_line_past_end_of_file_is_rejected(manifest):
    """An evidence line number that ran off the end of its file fails.

    The cited path is outside the pinned native surface (D8), so the entry
    carries a real, currently-valid anchor to isolate this test to the
    line-count check specifically -- an unanchored entry to the same path
    would also fail, but on the anchor-missing check, not this one.
    """
    corrupted = copy.deepcopy(manifest)
    _capability(corrupted, "MF-1")["evidence"] = [
        {
            "anchor": "import jax.numpy as jnp",
            "cite": "src/simsopt/field/selffield.py:900000",
        }
    ]

    message = _assert_rejected(corrupted, "stale_path")

    assert "points past the end" in message


def test_red_empty_reason_is_rejected(manifest):
    """A classified row with no rationale fails."""
    corrupted = copy.deepcopy(manifest)
    _row(corrupted, "tests/field/test_selffieldforces.py")["reason"] = "   "

    message = _assert_rejected(corrupted, "empty_reason")

    assert "test_selffieldforces.py" in message


def test_red_generic_reason_is_rejected(manifest):
    """A generic 'unsupported' label is not a rationale."""
    corrupted = copy.deepcopy(manifest)
    _capability(corrupted, "MF-2")["reason"] = "unsupported"

    message = _assert_rejected(corrupted, "empty_reason")

    assert "MF-2" in message


def test_red_source_tree_hash_drift_is_rejected(manifest):
    """A manifest minted against different native test bytes fails."""
    corrupted = copy.deepcopy(manifest)
    corrupted["baseline"]["source_tree_hash"] = "sha256:" + "0" * 64

    message = _assert_rejected(corrupted, "source_tree_drift")

    assert "re-mint with --write" in message


def test_red_orphan_jax_test_id_is_rejected(manifest):
    """A capability naming a test function that does not exist fails."""
    corrupted = copy.deepcopy(manifest)
    _capability(corrupted, "MF-5")["test_ids"] = [
        "tests/jax/native_unit_parity/test_force_parity.py::test_that_was_renamed_away"
    ]

    message = _assert_rejected(corrupted, "orphan_test_id")

    assert "test_that_was_renamed_away" in message


def test_red_capability_referenced_by_no_native_file_is_rejected(manifest):
    """A capability nothing maps to fails, so records cannot drift loose."""
    corrupted = copy.deepcopy(manifest)
    _row(corrupted, "tests/field/test_coilset.py")["capabilities"] = []

    failures = _failures(corrupted)

    assert any(failure.startswith("capability_mapping:") for failure in failures)
    assert any(
        failure.startswith("orphan_reference:") and "CS-1" in failure
        for failure in failures
    )


def test_red_unreviewed_native_only_decision_is_rejected(manifest):
    """`native_only` without a dated review record fails."""
    corrupted = copy.deepcopy(manifest)
    del _capability(corrupted, "CS-1")["decision"]

    message = _assert_rejected(corrupted, "unreviewed_decision")

    assert "CS-1" in message


def test_red_disposition_outside_the_contract_vocabulary_is_rejected(manifest):
    """A disposition the 2026-07-29 contract does not define fails."""
    corrupted = copy.deepcopy(manifest)
    _capability(corrupted, "MF-2")["disposition"] = "mostly_ported"

    message = _assert_rejected(corrupted, "disposition_vocabulary")

    assert "mostly_ported" in message


def test_red_test_id_naming_the_wrong_class_is_rejected(manifest):
    """A `Class::method` id where `method` is defined in a DIFFERENT class fails.

    ``test_k_square`` is a real function, but it lives in ``SpecialFunctionsTests``,
    not ``CoilForcesTest``; the AST-based check must look inside the NAMED
    class, not just confirm the function exists somewhere in the file.
    """
    corrupted = copy.deepcopy(manifest)
    _capability(corrupted, "MF-1")["native_test_ids"] = [
        "tests/field/test_selffieldforces.py::CoilForcesTest::test_k_square"
    ]

    message = _assert_rejected(corrupted, "orphan_test_id")

    assert "test_k_square" in message


def test_red_def_name_found_only_in_a_docstring_is_rejected(tmp_path):
    """A `def` mentioned only inside a docstring must not satisfy the AST check.

    A raw-text regex search for ``^\\s*def test_ghost\\(`` (MULTILINE) would
    match the indented line inside the docstring below and wrongly accept
    this id. Structural AST parsing does not: the docstring is a string
    literal, not a ``FunctionDef`` node.
    """
    fake_test_file = tmp_path / "test_fake_native.py"
    fake_test_file.write_text(
        '"""Module docstring.\n\n'
        "Example of the old test:\n"
        "    def test_ghost(self):\n"
        "        pass\n"
        '"""\n'
        "import unittest\n\n\n"
        "class Fake(unittest.TestCase):\n"
        "    def test_real(self):\n"
        "        pass\n",
        encoding="utf-8",
    )

    failures = COV._test_id_failures(
        "orphan_test_id",
        tmp_path,
        "row",
        "native_test_ids",
        ["test_fake_native.py::Fake::test_ghost"],
    )

    assert failures, (
        "a def name appearing only inside a docstring must be rejected, not "
        "accepted by a raw-text search"
    )
    assert "test_ghost" in failures[0]


def test_red_stale_evidence_anchor_is_rejected(manifest):
    """An anchor substring that no longer appears in its cited file fails."""
    corrupted = copy.deepcopy(manifest)
    _capability(corrupted, "MF-1")["evidence"] = [
        {
            "anchor": "this substring does not appear in selffield.py at all",
            "cite": "src/simsopt/field/selffield.py:18",
        }
    ]

    message = _assert_rejected(corrupted, "stale_path")

    assert "anchor" in message


def test_red_central_owner_claim_without_import_is_rejected(manifest):
    """A central tolerance-owner claim needs the bound file to import the owner.

    CP-4 binds only ``tests/jax/examples/test_stochastic_samples.py``, which
    never imports ``simsopt_jax.parity_tolerances``; flipping its declared
    owner to the central module must fail closed rather than stand as an
    unverifiable claim.
    """
    corrupted = copy.deepcopy(manifest)
    _capability(corrupted, "CP-4")["tolerance_owner"] = COV.CENTRAL_TOLERANCE_OWNER

    message = _assert_rejected(corrupted, "tolerance_owner_unsupported")

    assert "test_stochastic_samples.py" in message


def test_red_capability_missing_title_is_rejected(manifest):
    """A capability without a title fails closed instead of crashing render_document.

    render_document dereferences capability['title'] unconditionally; without
    this check a manifest could pass validate() and then KeyError inside
    render_document itself (D3c).
    """
    corrupted = copy.deepcopy(manifest)
    del _capability(corrupted, "MF-1")["title"]

    message = _assert_rejected(corrupted, "manifest_schema")

    assert "MF-1" in message


def test_red_duplicate_native_test_file_row_is_rejected(manifest):
    """The same native file appearing twice as a row fails."""
    corrupted = copy.deepcopy(manifest)
    duplicate_row = copy.deepcopy(_row(corrupted, "tests/field/test_coilset.py"))
    corrupted["native_test_files"].append(duplicate_row)

    message = _assert_rejected(corrupted, "duplicate_row")

    assert "test_coilset.py" in message


def test_red_missing_baseline_commit_is_rejected(manifest):
    """An empty baseline.baseline_commit fails the header schema check."""
    corrupted = copy.deepcopy(manifest)
    corrupted["baseline"]["baseline_commit"] = ""

    message = _assert_rejected(corrupted, "manifest_schema")

    assert "baseline_commit" in message


def test_red_jax_missing_capability_claiming_passing_lane_is_rejected(manifest):
    """A jax_missing row cannot claim a passing jax_cpu/jax_gpu lane (D6)."""
    corrupted = copy.deepcopy(manifest)
    _capability(corrupted, "MF-2")["required_lanes"]["jax_cpu"] = "passing"

    message = _assert_rejected(corrupted, "jax_missing_lane_contradiction")

    assert "MF-2" in message


def test_red_shared_python_without_decision_is_rejected(manifest):
    """`shared_python` without a dated decision record fails, like native_only (D7)."""
    corrupted = copy.deepcopy(manifest)
    del _capability(corrupted, "MF-1")["decision"]

    message = _assert_rejected(corrupted, "unreviewed_decision")

    assert "MF-1" in message


def test_red_generated_document_drift_is_rejected(manifest, tmp_path):
    """A hand-edited generated document that no longer matches the manifest fails (D3a)."""
    corrupted_doc_path = tmp_path / "coverage_doc_with_junk.md"
    corrupted_doc_path.write_text(
        COV.DOC_PATH.read_text(encoding="utf-8") + "\nHAND-EDITED JUNK\n",
        encoding="utf-8",
    )

    failures = COV.validate(REPO_ROOT, manifest, doc_path=corrupted_doc_path)

    matching = [f for f in failures if f.startswith("generated_doc_drift:")]
    assert matching, f"expected generated_doc_drift, got {failures or 'no failures'}"


def test_red_missing_pinned_surface_file_is_rejected_not_crashed(manifest, monkeypatch):
    """A pinned surface file missing from disk fails closed instead of crashing (D3b).

    Before the fix, ``source_tree_hash`` called ``.read_bytes()`` on every
    enumerated path unconditionally; a missing file raised an uncaught
    ``FileNotFoundError`` instead of joining the accumulated failures list.
    """
    monkeypatch.setattr(
        COV,
        "native_test_surface",
        lambda repo_root, upstream_commit: (
            "tests/field/test_selffieldforces.py",
            "tests/does_not_exist_on_disk_test_ghost.py",
        ),
    )

    failures = COV.validate(REPO_ROOT, manifest, check_doc=False)

    matching = [f for f in failures if f.startswith("stale_path:")]
    assert any("does_not_exist_on_disk_test_ghost.py" in f for f in matching), (
        f"expected a stale_path failure naming the missing file, got {failures}"
    )


def test_red_corrupted_manifest_makes_the_cli_exit_nonzero(manifest, tmp_path):
    """The command-line entry point fails closed, not just the library call."""
    corrupted = copy.deepcopy(manifest)
    corrupted["native_test_files"].pop(0)
    corrupted_path = tmp_path / "corrupted_manifest.json"
    corrupted_path.write_text(json.dumps(corrupted, indent=2, sort_keys=True) + "\n")

    assert COV.main(["--manifest", str(corrupted_path)]) == 1
