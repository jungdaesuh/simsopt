from __future__ import annotations

from pathlib import Path

import pytest
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
    assert require_published_run(tmp_path, paths.run_id) == published
    with pytest.raises(PublicationError, match="already exists"):
        begin_run(tmp_path, paths.run_id)


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
