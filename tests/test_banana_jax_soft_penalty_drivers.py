from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
pytest.importorskip("simsoptpp")

from examples.single_stage_optimization.banana_opt.jax_banana_drivers import (  # noqa: E402
    CurveSelfDistanceJAX,
    PoloidalExtentJAX,
    ProjectedEllipseWidthJAX,
    banana_base_curve,
    build_biotsavart,
    build_boozer_surface_copy,
)
from examples.single_stage_optimization.banana_opt.jax_banana_types import (  # noqa: E402
    DEFAULT_BANANA_DOFS,
    DEFAULT_PROXY_RZ,
    HBT_BANANA_WS,
)
from simsopt.geo import SurfaceXYZTensorFourier  # noqa: E402
from simsopt_jax_adapters.field.biotsavart_backend import BiotSavartJAX  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_ROOT = REPO_ROOT / "examples" / "single_stage_optimization"
STAGE2_SCRIPT = EXAMPLE_ROOT / "STAGE_2" / "banana_coil_solver.py"
SINGLE_STAGE_SCRIPT = EXAMPLE_ROOT / "SINGLE_STAGE" / "single_stage_banana_example.py"


def _run_driver(script: Path, args: list[str], *, timeout: int) -> subprocess.CompletedProcess:
    env = {
        **os.environ,
        "PYTHONNOUSERSITE": "1",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    result = subprocess.run(
        [sys.executable, str(script), *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def _single_match(paths: Iterable[Path]) -> Path:
    matches = sorted(paths)
    assert len(matches) == 1, matches
    return matches[0]


def _assert_finite_positive_gradient(objective) -> None:
    value = float(objective.J())
    gradient = np.asarray(objective.dJ(), dtype=float)
    assert np.isfinite(value)
    assert value > 0.0
    assert gradient.shape == np.asarray(objective.x).shape
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 0.0


def test_banana_geometry_jax_objectives_have_finite_gradients():
    biotsavart = build_biotsavart(
        tf_current_ka=-80.0,
        tf_fix_current=True,
        banana_current_ka=16.0,
        banana_fix_current=True,
        banana_order=1,
        banana_dofs=dict(DEFAULT_BANANA_DOFS),
        proxy_current_ka=0.0,
        proxy_rz=DEFAULT_PROXY_RZ,
        vf_current_ka=0.0,
        vf_fix_current=True,
    )
    curve = banana_base_curve(BiotSavartJAX(biotsavart.coils))

    objectives = [
        PoloidalExtentJAX(curve, HBT_BANANA_WS.major_radius, theta_target=0.05),
        ProjectedEllipseWidthJAX(
            curve,
            HBT_BANANA_WS.major_radius,
            HBT_BANANA_WS.minor_radius,
        ),
        CurveSelfDistanceJAX(
            curve,
            minimum_distance=2.0,
            neighbor_skip=3,
        ),
    ]
    for objective in objectives:
        _assert_finite_positive_gradient(objective)


def test_build_boozer_surface_copy_regrids_tensor_surface_seed():
    source = SurfaceXYZTensorFourier(
        mpol=2,
        ntor=2,
        nfp=1,
        stellsym=False,
        quadpoints_phi=np.linspace(0.0, 1.0, 7, endpoint=False),
        quadpoints_theta=np.linspace(0.0, 1.0, 8, endpoint=False),
    )

    copied = build_boozer_surface_copy(
        source,
        mpol=2,
        ntor=2,
        nphi=8,
        ntheta=8,
    )

    assert isinstance(copied, SurfaceXYZTensorFourier)
    assert copied.mpol == 2
    assert copied.ntor == 2
    assert copied.quadpoints_phi.size == 8
    assert copied.quadpoints_theta.size == 8
    assert np.all(np.isfinite(copied.gamma()))


@pytest.mark.slow
def test_soft_penalty_cli_stage2_to_single_stage_smoke(tmp_path):
    stage2_root = tmp_path / "stage2"
    single_stage_root = tmp_path / "single_stage"
    single_stage_seed_spec = tmp_path / "single_stage_seed_spec.json"

    _run_driver(
        STAGE2_SCRIPT,
        [
            "--backend",
            "jax",
            "--maxiter",
            "0",
            "--nphi",
            "8",
            "--ntheta",
            "7",
            "--order",
            "1",
            "--output-root",
            str(stage2_root),
            "--skip-postprocess",
        ],
        timeout=90,
    )

    stage2_run_dir = _single_match(stage2_root.glob("outputs-*/R0=*/"))
    assert (stage2_run_dir / "surf_opt.json").exists()
    assert (stage2_run_dir / "biot_savart_opt.json").exists()
    stage2_results = stage2_run_dir / "results.json"
    if not stage2_results.exists():
        rejected_payload = json.loads((stage2_run_dir / "REJECTED.json").read_text())
        assert rejected_payload["accepted"] is False
        assert rejected_payload["artifact_contract"]["accepted_results_filename"] == (
            "results.json"
        )
        assert rejected_payload["artifact_contract"]["diagnostic_marker_filename"] == (
            "REJECTED.json"
        )
        return

    stage2_dir = stage2_run_dir
    stage2_payload = json.loads(stage2_results.read_text())
    assert (stage2_dir / "surf_opt.json").exists()
    assert (stage2_dir / "biot_savart_opt.json").exists()
    assert stage2_payload["driver"] == "stage2_jax_soft_penalty"
    assert stage2_payload["optimizer"]["nfev"] > 0
    assert np.isfinite(stage2_payload["diagnostics"]["J"])
    assert "J_squared_flux" in stage2_payload["diagnostics"]

    _run_driver(
        SINGLE_STAGE_SCRIPT,
        [
            "--backend",
            "jax",
            "--compile-jax-runtime-seed-spec",
            "--warm-start-run-dir",
            str(stage2_dir),
            "--jax-runtime-seed-spec",
            str(single_stage_seed_spec),
            "--mpol",
            "1",
            "--ntor",
            "1",
            "--nphi",
            "8",
            "--ntheta",
            "8",
        ],
        timeout=90,
    )
    assert single_stage_seed_spec.exists()

    _run_driver(
        SINGLE_STAGE_SCRIPT,
        [
            "--backend",
            "jax",
            "--jax-runtime-seed-spec",
            str(single_stage_seed_spec),
            "--maxiter",
            "0",
            "--mpol",
            "1",
            "--ntor",
            "1",
            "--nphi",
            "8",
            "--ntheta",
            "8",
            "--banana-surf-radius",
            "0.15",
            "--output-root",
            str(single_stage_root),
            "--minimal-artifacts",
        ],
        timeout=120,
    )

    single_stage_results = _single_match(
        single_stage_root.glob("mpol=1-ntor=1-*/results.json")
    )
    single_stage_dir = single_stage_results.parent
    single_stage_payload = json.loads(single_stage_results.read_text())
    assert (single_stage_dir / "surf_opt.json").exists()
    assert single_stage_payload["driver"] == "single_stage_jax_soft_penalty"
    assert single_stage_payload["optimizer"]["nfev"] > 0
    assert np.isfinite(single_stage_payload["diagnostics"]["J"])
    assert np.isfinite(single_stage_payload["diagnostics"]["iota"])
    assert "J_non_quasisymmetric_ratio" in single_stage_payload["diagnostics"]
    assert "J_boozer_residual" in single_stage_payload["diagnostics"]
