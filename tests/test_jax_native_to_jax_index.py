"""Generated native-to-JAX index integrity."""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import json
from dataclasses import replace
from pathlib import Path

import examples.jax.manifest_contracts_v3 as manifest_contracts
import examples.jax.native_to_jax_index as index_module
import pytest
from examples.jax.native_to_jax_index import (
    INDEX_PATH,
    main,
    render_native_to_jax_index,
)
from examples.jax.parity._manifest import ScaleContract


def test_native_to_jax_index_matches_validated_contracts() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    rendered = render_native_to_jax_index(repo_root=repo_root)

    assert INDEX_PATH.read_text(encoding="utf-8") == rendered
    # One row per official source.
    assert rendered.count("\n| `examples/") == 53
    assert "## Official upstream catalog" in rendered
    assert "examples/1_Simple/periodicfieldline_QA.py" in rendered
    assert "examples/1_Simple/periodicfieldline_QH.py" in rendered


def test_dual_scale_index_lists_every_supported_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    examples = json.loads((repo_root / "examples/jax/manifest.json").read_text())
    parity = json.loads((repo_root / "examples/jax/parity_manifest.json").read_text())
    pair = manifest_contracts.load_manifest_contract_pair_documents(
        examples, parity, repo_root=repo_root
    )
    relationships = tuple(
        replace(
            relationship,
            scale_contracts=(
                ScaleContract(
                    "native_default", relationship.comparison_routes, "scheduled"
                ),
            ),
        )
        if relationship.case_id == "native-just-a-quadratic"
        else relationship
        for relationship in pair.parity.relationships
    )
    dual_scale_pair = replace(
        pair, parity=replace(pair.parity, relationships=relationships)
    )
    monkeypatch.setattr(
        index_module,
        "load_manifest_contract_pair_documents",
        lambda *_args, **_kwargs: dual_scale_pair,
    )

    rendered = render_native_to_jax_index(repo_root=repo_root)
    row = next(
        line
        for line in rendered.splitlines()
        if "`examples/1_Simple/just_a_quadratic.py`" in line
    )
    assert row.endswith("| bounded, native_default |")


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
