"""Fresh-process import coverage for the flat-675 QR dependency boundary."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
NATIVE_BUILD_DIRECTORY = next((REPOSITORY_ROOT / "build").glob("cp*-*"))


@pytest.mark.parametrize(
    "module_name",
    (
        "simsopt_jax_adapters.geo.flat675_qr",
        "simsopt_jax_adapters.geo.nested_ls_reduced",
        "simsopt_jax_adapters.geo.flat675",
    ),
)
def test_flat675_qr_import_boundary_is_collectible_in_a_fresh_process(
    module_name: str,
) -> None:
    """Each side of the former cycle imports independently in a new interpreter."""

    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(REPOSITORY_ROOT / "src"), str(NATIVE_BUILD_DIRECTORY))
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
