from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("jax")
pytest.importorskip("simsoptpp")

from examples.single_stage_optimization.banana_opt.jax_banana_types import (  # noqa: E402
    BANANA_TARGETS,
    DEFAULT_BANANA_ORDER,
    DEFAULT_PROXY_RZ,
    Stage2Weights,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "single_stage_optimization"
    / "STAGE_2"
    / "banana_coil_solver.py"
)


def _run_script(args: list[str]) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    result = subprocess.run(
        [sys.executable, str(STAGE2_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_stage2_cli_exposes_soft_penalty_surface():
    result = _run_script(["--help"])

    assert "soft-penalty" in result.stdout
    assert "--weight-global-curvature-radius" in result.stdout
    assert "--constraint-method" not in result.stdout
    assert "augmented" not in result.stdout.lower()


def test_stage2_defaults_match_reference_driver_contract():
    weights = Stage2Weights()

    assert weights.sqflux == pytest.approx(1.0)
    assert weights.length == pytest.approx(5.0e-2)
    assert weights.ccdist == pytest.approx(1.0e6)
    assert weights.width == pytest.approx(1.0e1)
    assert weights.global_curvature_radius == pytest.approx(1.0e3)
    assert weights.currents == pytest.approx(1.0e6)
    assert BANANA_TARGETS.width_max == pytest.approx(0.197)
    assert BANANA_TARGETS.width_min == pytest.approx(0.050)
    assert BANANA_TARGETS.global_curvature_radius_min == pytest.approx(0.010)
    assert DEFAULT_BANANA_ORDER == 4
    assert DEFAULT_PROXY_RZ == pytest.approx((0.903, 0.0))
