"""The isolated-child kernel resolution of ``simsopt_jax_adapters.isolated_kernel``.

A child process must import the compiled ``simsoptpp`` its parent loaded, never
the ``src/simsoptpp/`` C++ source directory, which Python would import as a
namespace package without ``Curve``.
"""

from __future__ import annotations

from jax_test_support import fixture_jax_runtime_guard  # noqa: F401

import os
import subprocess
import sys
from pathlib import Path

import simsoptpp
from simsopt_jax_adapters.isolated_kernel import (
    isolated_child_command,
    loaded_kernel_directory,
)


def test_loaded_kernel_directory_is_the_parent_of_the_extension() -> None:
    kernel_file = Path(simsoptpp.__file__).resolve()
    assert loaded_kernel_directory() == kernel_file.parent
    assert kernel_file.name.startswith("simsoptpp")


def test_isolated_child_imports_curve_from_the_parent_kernel() -> None:
    completed = subprocess.run(
        isolated_child_command(
            ("-c", "from simsoptpp import Curve; print(Curve.__name__)")
        ),
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
