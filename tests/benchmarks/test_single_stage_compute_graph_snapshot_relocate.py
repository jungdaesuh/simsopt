from __future__ import annotations

import json
from pathlib import Path

import pytest
from benchmarks import single_stage_compute_graph_snapshot_relocate as relocation
from benchmarks.single_stage_compute_graph_snapshot import canonical_json_bytes


def _source_publication(path: Path) -> None:
    path.write_bytes(canonical_json_bytes({"snapshot_root": "/source"}))


def test_relocation_validates_staging_before_exclusive_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "snapshot"
    output = tmp_path / "relocated.json"
    _source_publication(source)
    destination.mkdir()

    def validate(root: Path, path: Path) -> tuple[dict[str, object], tuple[()], str]:
        document = json.loads(path.read_bytes())
        assert root == destination.absolute()
        assert document["snapshot_root"] == str(destination.absolute())
        assert path.name == f".{output.name}.partial"
        return document, (), "a" * 64

    monkeypatch.setattr(relocation, "_publication", validate)

    assert (
        relocation.relocate_snapshot_publication(
            source_publication_path=source,
            destination_snapshot_root=destination,
            output_path=output,
        )
        == output.absolute()
    )
    assert output.is_file()
    assert not output.with_name(f".{output.name}.partial").exists()


def test_relocation_failure_never_publishes_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "snapshot"
    output = tmp_path / "relocated.json"
    _source_publication(source)
    destination.mkdir()

    def reject(_root: Path, _path: Path) -> tuple[dict[str, object], tuple[()], str]:
        raise Phase0WorkflowError("destination bytes differ")

    from benchmarks.single_stage_compute_graph_phase0_workflow import (
        Phase0WorkflowError,
    )

    monkeypatch.setattr(relocation, "_publication", reject)

    with pytest.raises(
        relocation.SnapshotRelocationError,
        match="destination bytes differ",
    ):
        relocation.relocate_snapshot_publication(
            source_publication_path=source,
            destination_snapshot_root=destination,
            output_path=output,
        )
    assert not output.exists()


def test_relocation_rejects_noncanonical_source(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    source.write_text('{"snapshot_root": "/source"}\n', encoding="utf-8")

    with pytest.raises(
        relocation.SnapshotRelocationError,
        match="not canonical JSON",
    ):
        relocation.relocate_snapshot_publication(
            source_publication_path=source,
            destination_snapshot_root=tmp_path / "snapshot",
            output_path=tmp_path / "relocated.json",
        )
