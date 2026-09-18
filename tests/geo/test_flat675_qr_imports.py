"""Fresh-process import coverage for the flat-675 dependency and example modules."""

from __future__ import annotations

import importlib.machinery
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NATIVE_BUILD_ROOT = REPOSITORY_ROOT / "build"


class MissingNativeExtensionError(RuntimeError):
    """The repo-local ``simsoptpp`` extension this boundary imports is absent."""


def _native_extension_directory() -> Path:
    """Return the build directory holding this interpreter's ``simsoptpp``."""

    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        for candidate in sorted(NATIVE_BUILD_ROOT.glob(f"**/simsoptpp{suffix}")):
            return candidate.parent
    raise MissingNativeExtensionError(
        f"no simsoptpp extension under {NATIVE_BUILD_ROOT}; build the native "
        "extension in this worktree before running the flat-675 import boundary"
    )


@pytest.mark.parametrize(
    "module_name",
    (
        "simsopt_jax_adapters.geo.flat675_qr",
        "simsopt_jax_adapters.geo.nested_ls_reduced",
        "simsopt_jax_adapters.geo.flat675",
        "simsopt._examples_runtime",
        "simsopt._examples_runtime.execution",
        "simsopt_jax_adapters.examples",
        "simsopt_jax_adapters.examples.single_stage_flat675",
        "simsopt_jax_adapters.examples.single_stage_flat675_native_twin",
    ),
)
def test_flat675_qr_import_boundary_is_collectible_in_a_fresh_process(
    module_name: str,
) -> None:
    """Each listed module imports alone in a fresh interpreter.

    The list is the two sides of the former flat-675 QR import cycle plus their
    shared leaf, and the modules 2f7d21b75 created or renamed: the private
    example runtime and the relocated flat-675 example implementations.
    """

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), str(_native_extension_directory()))
    )
    completed = subprocess.run(
        (sys.executable, "-c", f"import {module_name}"),
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
