"""The isolated-child kernel resolution of ``simsopt_jax_adapters.isolated_kernel``.

A child process must import the compiled ``simsoptpp`` its parent loaded, never
the ``src/simsoptpp/`` C++ source directory, which Python would import as a
namespace package without ``Curve``.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import ast
import os
import subprocess
import sys
from pathlib import Path
from typing import get_args

import pytest
import simsoptpp
import simsopt_jax_adapters.isolated_kernel as isolated_kernel
from simsopt_jax_adapters.isolated_kernel import (
    IsolatedChildPayload,
    isolated_child_command,
    loaded_kernel_directory,
)

_ADAPTERS = Path(isolated_kernel.__file__).resolve().parent
_CHILD = _ADAPTERS / "isolated_kernel_child.py"
_RUNTIME_CODE_LOADERS = frozenset({"exec", "eval", "compile", "__import__"})


def test_loaded_kernel_directory_is_the_parent_of_the_extension() -> None:
    kernel_file = Path(simsoptpp.__file__).resolve()
    assert loaded_kernel_directory() == kernel_file.parent
    assert kernel_file.name.startswith("simsoptpp")


def test_isolated_child_imports_curve_from_the_parent_kernel() -> None:
    completed = subprocess.run(
        isolated_child_command("simsoptpp-curve-name"),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "Curve"


def test_installed_layout_kernel_directory_beats_src_namespace(
    tmp_path: Path,
) -> None:
    """A wheel puts ``simsoptpp*.so`` in site-packages; ``src/simsoptpp/`` must not win."""
    extension = next(loaded_kernel_directory().glob("simsoptpp*.so"))
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    (site_packages / extension.name).symlink_to(extension)
    src = Path(__file__).resolve().parents[3] / "src"
    probe = "from simsoptpp import Curve; print(Curve.__name__)"
    installed = subprocess.run(
        (sys.executable, "-S", "-c", probe),
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join((str(site_packages), str(src))),
        },
        check=False,
        capture_output=True,
        text=True,
    )
    shadowed = subprocess.run(
        (sys.executable, "-S", "-c", probe),
        env={**os.environ, "PYTHONPATH": str(src)},
        check=False,
        capture_output=True,
        text=True,
    )
    assert installed.returncode == 0, installed.stderr
    assert installed.stdout.strip() == "Curve"
    assert shadowed.returncode != 0
    assert "Curve" in shadowed.stderr


@pytest.mark.parametrize("path", (_CHILD, _ADAPTERS / "isolated_kernel.py"))
def test_isolated_child_runs_no_source_text(path: Path) -> None:
    """The child dispatches on a fixed table of functions; it executes no text."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    called = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    imported = {
        name.partition(".")[0]
        for node in ast.walk(tree)
        for name in (
            [alias.name for alias in node.names]
            if isinstance(node, ast.Import)
            else [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else []
        )
    }
    assert not called & _RUNTIME_CODE_LOADERS
    assert not imported & {"runpy", "importlib"}


def test_payload_names_match_the_child_table() -> None:
    tree = ast.parse(_CHILD.read_text(encoding="utf-8"), filename=str(_CHILD))
    table = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "PAYLOADS"
    )
    assert isinstance(table, ast.Call) and isinstance(table.args[0], ast.Dict)
    keys = {key.value for key in table.args[0].keys if isinstance(key, ast.Constant)}
    assert keys == set(get_args(IsolatedChildPayload))


def test_an_unknown_payload_is_refused() -> None:
    command = (*isolated_child_command("simsoptpp-curve-name")[:-1], "missing")
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert completed.returncode != 0
    assert "unknown isolated-child payload 'missing'" in completed.stderr
