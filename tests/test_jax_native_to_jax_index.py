"""Generated native-to-JAX index integrity."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

from pathlib import Path

import examples.jax.native_to_jax_index as index_module
import pytest
from examples.jax.native_to_jax_index import (
    INDEX_PATH,
    main,
    render_native_to_jax_index,
)


def test_native_to_jax_index_matches_validated_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rendered = render_native_to_jax_index(repo_root=repo_root)

    assert INDEX_PATH.read_text(encoding="utf-8") == rendered
    # One row per official source.
    assert rendered.count("\n| `examples/") == 53
    assert "## Official upstream catalog" in rendered
    assert "examples/1_Simple/periodicfieldline_QA.py" in rendered
    assert "examples/1_Simple/periodicfieldline_QH.py" in rendered


def test_check_accepts_a_current_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    index = tmp_path / "index.md"
    index.write_text("current index")
    monkeypatch.setattr(index_module, "INDEX_PATH", index)
    monkeypatch.setattr(
        index_module, "render_native_to_jax_index", lambda: "current index"
    )
    assert main(["--check"]) == 0
    assert "Index consistent." in capsys.readouterr().out


def test_check_rejects_a_stale_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = tmp_path / "index.md"
    index.write_text("stale index")
    monkeypatch.setattr(index_module, "INDEX_PATH", index)
    monkeypatch.setattr(
        index_module, "render_native_to_jax_index", lambda: "current index"
    )
    with pytest.raises(SystemExit, match="index is stale"):
        main(["--check"])
