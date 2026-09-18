"""Live receipt destinations for the nested-LS benchmark family.

``72e7a72b0`` deleted ``docs/receipts/`` from the tree while nine drivers
still defaulted their receipts into it. Six of them write without creating the
parent, so a default run raises ``FileNotFoundError``; the rest create the
parent and therefore resurrect the deleted tree and write into it, leaving a
checkout the clean-tree gate then flags. These tests pin the repaired
destination, the directory creation that makes a first run work on a clean
checkout, and the clean-tree exemption that has to follow the destination
rather than the other way round.

Four of the eight drivers run their campaign at module scope, so importing
them here would launch a multi-hour sweep. Those are read as source.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from benchmarks import nested_ls_evidence
from benchmarks import nested_ls_shamanskii_attribution as attribution

REPO = nested_ls_evidence.REPO
EVIDENCE = nested_ls_evidence.EVIDENCE

#: Every driver repaired here. The four marked ``imports_safely=False`` have no
#: ``if __name__ == "__main__"`` guard: importing one runs its benchmark.
DRIVERS: dict[str, bool] = {
    "nested_ls_shamanskii_attribution.py": True,
    "nested_ls_outer_predictor_replay.py": True,
    "nested_ls_outer_claim.py": True,
    "nested_ls_outer_fd0.py": True,
    "nested_ls_a100_banana_omp.py": False,
    "nested_ls_banana_omp_gap.py": False,
    "nested_ls_banana_omp_min_bracket.py": False,
    "nested_ls_f3_b37_gpu_canaries.py": False,
}

CAMPAIGN_SCRIPTS = [name for name, safe in DRIVERS.items() if not safe]


def driver_source(name: str) -> str:
    return (REPO / "benchmarks" / name).read_text(encoding="utf-8")


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=str(repo),
        check=True,
        capture_output=True,
        text=True,
    ).stdout


@pytest.fixture
def receipt_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A real git repository laid out like this one, with nothing excluded.

    ``/.artifacts/`` sits in this checkout's ``.git/info/exclude``, which would
    hide every receipt from ``git status`` and let a broken exemption pass. A
    fresh repository has no such exclude, so the status lines here are the ones
    a clone would produce.
    """

    git(tmp_path, "init", "-q")
    git(tmp_path, "config", "user.email", "test@example.invalid")
    git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "benchmarks").mkdir()
    (tmp_path / "benchmarks" / "driver.py").write_text("x = 1\n", encoding="utf-8")
    git(tmp_path, "add", "benchmarks/driver.py")
    git(tmp_path, "commit", "-q", "-m", "seed")
    (tmp_path / attribution.EVIDENCE_STATUS_PREFIX).mkdir(parents=True)
    monkeypatch.setattr(attribution, "REPO", tmp_path)
    return tmp_path


def test_the_receipt_directory_is_live_and_inside_the_repository() -> None:
    """The destination is a creatable path under the repo, not the dead tree."""

    assert EVIDENCE.is_relative_to(nested_ls_evidence.ARTIFACT_ROOT)
    assert nested_ls_evidence.ARTIFACT_ROOT.is_relative_to(REPO)
    assert (REPO / "benchmarks").is_dir(), "REPO must resolve to the repo root"
    assert "receipts" not in EVIDENCE.parts


@pytest.mark.parametrize("name", sorted(DRIVERS))
def test_no_driver_defaults_into_the_deleted_receipts_tree(name: str) -> None:
    """Each driver takes its destination from the one module that owns it."""

    source = driver_source(name)
    assert '"docs" / "receipts"' not in source, (
        f"{name} still builds a destination under the tree 72e7a72b0 deleted"
    )
    assert "from benchmarks.nested_ls_evidence import EVIDENCE" in source, (
        f"{name} does not bind its destination to the shared receipt directory"
    )


def test_the_imported_driver_writes_through_the_shared_constant() -> None:
    """Binding by import, not by a copy that could be edited on its own."""

    assert attribution.EVIDENCE is EVIDENCE
    assert attribution.OUT_JSON.parent == EVIDENCE


@pytest.mark.parametrize("name", CAMPAIGN_SCRIPTS)
def test_campaign_scripts_create_their_receipt_directory_before_writing(
    name: str,
) -> None:
    """A first run on a clean checkout must not fail on a missing parent.

    Asserted on the source because these four scripts execute their benchmark
    on import. Every module-level ``write_strict_json(TARGET, ...)`` must be
    preceded by ``TARGET.parent.mkdir(...)`` at the same level.
    """

    tree = ast.parse(driver_source(name))
    created: set[str] = set()
    written: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        func = call.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "mkdir"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "parent"
            and isinstance(func.value.value, ast.Name)
        ):
            created.add(func.value.value.id)
        if (
            isinstance(func, ast.Name)
            and func.id == "write_strict_json"
            and call.args
            and isinstance(call.args[0], ast.Name)
        ):
            target = call.args[0].id
            written.append(target)
            assert target in created, (
                f"{name} writes {target} before creating its directory"
            )
    assert written, f"{name} has no module-level receipt write to check"


def test_writing_a_lane_receipt_creates_a_missing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The receipt lands even when no part of the destination exists yet."""

    monkeypatch.setattr(attribution, "EVIDENCE", tmp_path / "absent" / "deeper")
    path = attribution.write_lane_json("cache_only", [])
    assert path.is_file()
    assert path.parent == tmp_path / "absent" / "deeper"


def test_the_exemption_prefix_follows_the_receipt_destination() -> None:
    """The clean-tree exemption covers the destination and nothing wider."""

    assert EVIDENCE.is_relative_to(REPO / attribution.EVIDENCE_STATUS_PREFIX)
    assert attribution.EVIDENCE_STATUS_PREFIX.endswith("/")
    assert attribution.EVIDENCE_STATUS_PREFIX != (
        f"{nested_ls_evidence.ARTIFACT_ROOT.relative_to(REPO).as_posix()}/"
    ), "the whole artifact root is not exemptible; only the receipt directory"


def test_an_untracked_receipt_leaves_the_tree_clean(receipt_repo: Path) -> None:
    """A run that writes its own evidence does not refuse the next run."""

    receipt = receipt_repo / attribution.EVIDENCE_STATUS_PREFIX / "receipt.json"
    receipt.write_text("{}\n", encoding="utf-8")

    assert attribution.git_implementation_dirty() == ""


def test_a_tracked_receipt_modification_still_dirties_the_tree(
    receipt_repo: Path,
) -> None:
    """Exempting untracked output must not hide an edit to a sealed receipt."""

    relative = f"{attribution.EVIDENCE_STATUS_PREFIX}sealed.json"
    sealed = receipt_repo / relative
    sealed.write_text("{}\n", encoding="utf-8")
    git(receipt_repo, "add", relative)
    git(receipt_repo, "commit", "-q", "-m", "seal")
    sealed.write_text('{"edited": true}\n', encoding="utf-8")

    assert relative in attribution.git_implementation_dirty()


def test_a_staged_receipt_still_dirties_the_tree(receipt_repo: Path) -> None:
    """``??`` is the only status the exemption drops."""

    relative = f"{attribution.EVIDENCE_STATUS_PREFIX}staged.json"
    (receipt_repo / relative).write_text("{}\n", encoding="utf-8")
    git(receipt_repo, "add", relative)

    assert relative in attribution.git_implementation_dirty()


def test_untracked_output_elsewhere_under_the_artifact_root_dirties_the_tree(
    receipt_repo: Path,
) -> None:
    """The exemption is one directory, not the whole artifact root."""

    other = receipt_repo / ".artifacts" / "other" / "scratch.json"
    other.parent.mkdir(parents=True)
    other.write_text("{}\n", encoding="utf-8")

    assert ".artifacts/other/scratch.json" in attribution.git_implementation_dirty()


def test_a_tracked_change_elsewhere_under_the_artifact_root_dirties_the_tree(
    receipt_repo: Path,
) -> None:
    """A tracked file is implementation wherever it lives."""

    relative = ".artifacts/other/tracked.json"
    tracked = receipt_repo / relative
    tracked.parent.mkdir(parents=True)
    tracked.write_text("{}\n", encoding="utf-8")
    git(receipt_repo, "add", relative)
    git(receipt_repo, "commit", "-q", "-m", "track")
    tracked.write_text('{"edited": true}\n', encoding="utf-8")

    assert relative in attribution.git_implementation_dirty()
