from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytest.importorskip("jax")
pytest.importorskip("simsoptpp")

from examples.single_stage_optimization.banana_opt.jax_banana_types import (  # noqa: E402
    BANANA_TARGETS,
    SingleStageWeights,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SINGLE_STAGE_SCRIPT = (
    REPO_ROOT
    / "examples"
    / "single_stage_optimization"
    / "SINGLE_STAGE"
    / "single_stage_banana_example.py"
)


def _run_script(args: list[str]) -> subprocess.CompletedProcess:
    result = _run_script_raw(args)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _run_script_raw(args: list[str]) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    return subprocess.run(
        [sys.executable, str(SINGLE_STAGE_SCRIPT), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )


def test_single_stage_cli_exposes_seeded_soft_penalty_surface():
    result = _run_script(["--help"])

    assert "soft-penalty" in result.stdout
    assert "--compile-jax-runtime-seed-spec" in result.stdout
    assert "--weight-global-curvature-radius" in result.stdout
    assert "--constraint-method" not in result.stdout
    assert "augmented" not in result.stdout.lower()


def test_single_stage_seed_spec_compilation_writes_chainable_paths(tmp_path):
    stage2_run_dir = tmp_path / "stage2"
    stage2_run_dir.mkdir()
    for artifact_name in (
        "biot_savart_opt.json",
        "surf_opt.json",
        "boozersurface_opt.json",
    ):
        (stage2_run_dir / artifact_name).write_text("{}\n", encoding="utf-8")
    seed_spec = tmp_path / "seed.json"

    _run_script(
        [
            "--compile-jax-runtime-seed-spec",
            "--warm-start-run-dir",
            str(stage2_run_dir),
            "--jax-runtime-seed-spec",
            str(seed_spec),
            "--mpol",
            "1",
            "--ntor",
            "1",
            "--nphi",
            "8",
            "--ntheta",
            "8",
        ]
    )

    payload = json.loads(seed_spec.read_text(encoding="utf-8"))
    assert payload["driver"] == "single_stage_jax_runtime_seed_spec"
    assert payload["biotsavart"] == str(stage2_run_dir / "biot_savart_opt.json")
    assert payload["surface"] == str(stage2_run_dir / "surf_opt.json")
    assert payload["boozersurface"] == str(stage2_run_dir / "boozersurface_opt.json")


def test_single_stage_seed_spec_compilation_rejects_missing_artifacts(tmp_path):
    stage2_run_dir = tmp_path / "stage2"
    seed_spec = tmp_path / "seed.json"

    result = _run_script_raw(
        [
            "--compile-jax-runtime-seed-spec",
            "--warm-start-run-dir",
            str(stage2_run_dir),
            "--jax-runtime-seed-spec",
            str(seed_spec),
            "--mpol",
            "1",
            "--ntor",
            "1",
            "--nphi",
            "8",
            "--ntheta",
            "8",
        ]
    )

    assert result.returncode != 0
    assert "missing Stage 2 artifact" in result.stderr
    assert not seed_spec.exists()


def test_single_stage_defaults_match_reference_driver_contract():
    weights = SingleStageWeights()

    assert weights.nonqs == pytest.approx(1.0)
    assert weights.bres == pytest.approx(1.0e3)
    assert weights.iota == pytest.approx(1.0e4)
    assert weights.length == pytest.approx(5.0e-2)
    assert weights.ccdist == pytest.approx(1.0e6)
    assert weights.width == pytest.approx(1.0e1)
    assert weights.global_curvature_radius == pytest.approx(1.0e3)
    assert BANANA_TARGETS.global_curvature_radius_exp_weight == pytest.approx(0.010)
