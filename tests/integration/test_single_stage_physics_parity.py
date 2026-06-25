from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("jax")
pytest.importorskip("simsoptpp")


REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_STAGE_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)


def test_single_stage_public_cli_uses_direct_soft_penalty_contract():
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    result = subprocess.run(
        [sys.executable, str(SINGLE_STAGE_SCRIPT), "--help"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--iota-target" in result.stdout
    assert "--constraint-weight" in result.stdout
    assert "--constraint-method" not in result.stdout
    assert "augmented" not in result.stdout.lower()
