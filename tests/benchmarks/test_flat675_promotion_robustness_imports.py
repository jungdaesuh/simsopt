"""Import policy and the robustness child's pre-solve origin boundary."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Never

import benchmarks.flat675_promotion_robustness_child as child
import pytest


def test_all_child_imports_are_at_module_scope() -> None:
    tree = ast.parse(Path(child.__file__).read_text())
    top_level_imports = {
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    }
    nested_import_lines = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and node not in top_level_imports
    ]
    assert not nested_import_lines, (
        f"Function/block-local imports: {nested_import_lines}"
    )


def test_production_module_origins_are_accepted() -> None:
    origins = child._require_production_import_origin()
    assert set(origins) == {"simsopt_jax", "simsopt_jax_adapters"}
    assert all(
        child.PRODUCTION_ROOT in Path(origin).parents for origin in origins.values()
    )


def _unexpected_problem_construction(*args: object, **kwargs: object) -> Never:
    pytest.fail("Foreign module origin reached problem construction")


@pytest.mark.parametrize("module_name", ["simsopt_jax", "simsopt_jax_adapters"])
@pytest.mark.parametrize("mode", ["bundle", "constructor"])
def test_foreign_origin_rejected_before_problem_or_output(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, module_name: str, mode: str
) -> None:
    module = getattr(child, module_name)
    foreign_origin = tmp_path / module_name / "__init__.py"
    monkeypatch.setattr(module, "__file__", str(foreign_origin))
    monkeypatch.setattr(child, "load_flat675_bundle", _unexpected_problem_construction)
    monkeypatch.setattr(child, "repository_problem", _unexpected_problem_construction)
    output = tmp_path / "uncreated" / "child.json"
    argv = [
        "--mode",
        mode,
        "--input-manifest",
        str(tmp_path / "manifest.json"),
        "--maxiter",
        "1",
        "--seed",
        "20260819",
        "--output-json",
        str(output),
    ]
    with pytest.raises(RuntimeError, match=f"{module_name} resolved to .*outside"):
        child.main(argv)
    assert not output.parent.exists(), "Rejected origin created output directories"
