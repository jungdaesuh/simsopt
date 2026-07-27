from __future__ import annotations

from pathlib import Path

import pytest
import examples.jax.parity.publication as publication
from examples.jax.parity.publication import (
    PublicationError,
    begin_run,
    mark_run_failed,
    publish_run,
    require_published_run,
)


def test_run_publication_is_atomic_and_never_overwrites(tmp_path: Path) -> None:
    paths = begin_run(tmp_path, "20260726T160000Z-a1b2c3d4")
    (paths.partial / "summary.json").write_text("{}\n", encoding="utf-8")

    published = publish_run(paths)

    assert published == paths.final
    assert published.is_dir()
    assert not paths.partial.exists()
    assert (published / "COMPLETED.json").is_file()
    assert require_published_run(tmp_path, paths.run_id) == published
    with pytest.raises(PublicationError, match="already exists"):
        begin_run(tmp_path, paths.run_id)


@pytest.mark.parametrize("collision", ["empty-directory", "symlink"])
def test_publish_time_collision_never_replaces_final_entry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    paths = begin_run(tmp_path, "20260726T160004Z-a1b2c3d4")
    (paths.partial / "summary.json").write_text("{}\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("collision-owned\n", encoding="utf-8")

    def create_collision(target: Path) -> None:
        if collision == "empty-directory":
            target.mkdir()
        else:
            target.symlink_to(outside, target_is_directory=True)

    if hasattr(publication, "_rename_no_replace"):
        original_no_replace = publication._rename_no_replace

        def collide_then_publish(source: Path, target: Path) -> None:
            create_collision(target)
            original_no_replace(source, target)

        monkeypatch.setattr(publication, "_rename_no_replace", collide_then_publish)
    else:
        original_rename = Path.rename

        def collide_then_rename(source: Path, target: Path) -> Path:
            create_collision(target)
            return original_rename(source, target)

        monkeypatch.setattr(Path, "rename", collide_then_rename)

    with pytest.raises(PublicationError, match="already exists"):
        publish_run(paths)

    if collision == "empty-directory":
        assert paths.final.is_dir()
        assert not paths.final.is_symlink()
        assert not any(paths.final.iterdir())
    else:
        assert paths.final.is_symlink()
        assert paths.final.resolve() == outside.resolve()
        assert sentinel.read_text(encoding="utf-8") == "collision-owned\n"
    assert (paths.partial / "summary.json").read_text(encoding="utf-8") == "{}\n"


def test_reader_rejects_final_directory_without_completion_marker(
    tmp_path: Path,
) -> None:
    paths = begin_run(tmp_path, "20260726T160005Z-a1b2c3d4")
    (paths.partial / "summary.json").write_text("{}\n", encoding="utf-8")
    published = publish_run(paths)
    (published / "COMPLETED.json").unlink()

    with pytest.raises(PublicationError, match="not a published run"):
        require_published_run(tmp_path, paths.run_id)


def test_concurrent_writer_cannot_claim_existing_partial_run(tmp_path: Path) -> None:
    first = begin_run(tmp_path, "20260726T160001Z-a1b2c3d4")

    with pytest.raises(PublicationError, match="already exists"):
        begin_run(tmp_path, first.run_id)


def test_failed_run_remains_partial_and_cannot_be_audit_input(tmp_path: Path) -> None:
    paths = begin_run(tmp_path, "20260726T160002Z-a1b2c3d4")

    marker = mark_run_failed(paths, "native-cpu returned 2")

    assert marker.is_file()
    assert paths.partial.is_dir()
    assert not paths.final.exists()
    with pytest.raises(PublicationError, match="not a published run"):
        require_published_run(tmp_path, paths.run_id)
    with pytest.raises(PublicationError, match="failed partial run"):
        publish_run(paths)


def test_publish_rejects_incomplete_run_without_summary(tmp_path: Path) -> None:
    paths = begin_run(tmp_path, "20260726T160003Z-a1b2c3d4")

    with pytest.raises(PublicationError, match="summary.json"):
        publish_run(paths)


@pytest.mark.parametrize("run_id", ["../escape", "wave-a", "20260726-a/b"])
def test_run_id_is_bounded_and_canonical(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(PublicationError, match="run ID"):
        begin_run(tmp_path, run_id)
