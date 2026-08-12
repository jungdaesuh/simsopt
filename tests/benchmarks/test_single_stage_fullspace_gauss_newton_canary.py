from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from benchmarks.run_single_stage_fullspace_gauss_newton_canary import (
    EQUALITY_SIZE,
    GN_NORMALIZED_PSD_LOWER_BOUND,
    GPU_UUID,
    OBJECTIVE_RESIDUAL_SIZE,
    ROUTE,
    SCHEMA_VERSION,
    SOURCE_PATHS,
    STATE_SIZE,
    TRUST_RADIUS,
    _canonical_json_bytes,
    _numerical_gates_pass,
    _resource_gates_pass,
    _terminal_status,
    _tracked_paths,
)


def _endpoint(*, objective: float = 1.0, raw_kkt: float = 2.0):
    return SimpleNamespace(
        physical=SimpleNamespace(
            physical_objective=np.asarray(objective, dtype=np.float64),
            raw_kkt_stationarity_infinity_norm=np.asarray(
                raw_kkt,
                dtype=np.float64,
            ),
            all_finite=np.asarray(True),
        )
    )


def _result(
    *,
    residual_valid: bool = True,
    residual_all_finite: bool = True,
    value_defect: float = 0.0,
    gradient_defect: float = 0.0,
    symmetry_defect: float = 0.0,
    normalized_curvature: float = 0.0,
    terminal_normalized_curvature: float = 0.0,
    raw_kkt: float = 2.0,
    objective_residual_size: int = OBJECTIVE_RESIDUAL_SIZE,
):
    return SimpleNamespace(
        both_variants_usable=np.asarray(True),
        all_finite=np.asarray(True),
        objective_residual_size=np.asarray(objective_residual_size, dtype=np.int32),
        residual_reconstruction=SimpleNamespace(
            residual_valid=np.asarray(residual_valid),
            all_finite=np.asarray(residual_all_finite),
            value_scaled_defect=np.asarray(value_defect, dtype=np.float64),
            gradient_scaled_defect=np.asarray(gradient_defect, dtype=np.float64),
        ),
        gauss_newton_hvp_bilinear_symmetry_relative_defect=np.asarray(
            symmetry_defect,
            dtype=np.float64,
        ),
        gauss_newton_probe_normalized_curvature=np.asarray(
            normalized_curvature,
            dtype=np.float64,
        ),
        gauss_newton_terminal_normalized_curvature=np.asarray(
            terminal_normalized_curvature,
            dtype=np.float64,
        ),
        initial=_endpoint(raw_kkt=raw_kkt),
        identity=_endpoint(raw_kkt=raw_kkt),
        gauss_newton=_endpoint(raw_kkt=raw_kkt),
    )


def test_frozen_runtime_and_problem_identity() -> None:
    assert SCHEMA_VERSION == "single-stage-fullspace-gauss-newton-canary-v1"
    assert ROUTE == "CFS-GN1"
    assert GPU_UUID == "GPU-7951f78e-c05d-e01c-303f-d644f4341fe1"
    assert STATE_SIZE == 716
    assert EQUALITY_SIZE == 255
    assert OBJECTIVE_RESIDUAL_SIZE == 2110
    assert TRUST_RADIUS == 2.0**-10


def test_terminal_status_fails_closed() -> None:
    assert _terminal_status(usable=False, supported=True) == "CANARY_NOT_USABLE"
    assert (
        _terminal_status(usable=True, supported=False)
        == "NOT_SUPPORTED_BY_ONE_STEP_CANARY"
    )
    assert (
        _terminal_status(usable=True, supported=True) == "SUPPORTED_BY_ONE_STEP_CANARY"
    )


def test_numerical_gate_enforces_residual_and_gauss_newton_certificates() -> None:
    assert _numerical_gates_pass(_result())
    assert not _numerical_gates_pass(_result(objective_residual_size=2109))
    assert not _numerical_gates_pass(_result(residual_valid=False))
    assert not _numerical_gates_pass(_result(residual_all_finite=False))
    assert not _numerical_gates_pass(_result(value_defect=1.1e-12))
    assert not _numerical_gates_pass(_result(gradient_defect=1.1e-10))
    assert not _numerical_gates_pass(_result(symmetry_defect=1.1e-10))
    assert not _numerical_gates_pass(
        _result(normalized_curvature=GN_NORMALIZED_PSD_LOWER_BOUND - 1.0e-12)
    )
    assert not _numerical_gates_pass(
        _result(terminal_normalized_curvature=(GN_NORMALIZED_PSD_LOWER_BOUND - 1.0e-12))
    )


def test_numerical_gate_rejects_nonfinite_physical_endpoint() -> None:
    assert not _numerical_gates_pass(_result(raw_kkt=float("nan")))


def test_resource_gate_requires_unchanged_manifest_and_memory_below_bound() -> None:
    assert _resource_gates_pass(
        pre_post_manifest_identical=True,
        peak_memory_fraction=0.799,
    )
    assert not _resource_gates_pass(
        pre_post_manifest_identical=False,
        peak_memory_fraction=0.1,
    )
    assert not _resource_gates_pass(
        pre_post_manifest_identical=True,
        peak_memory_fraction=0.8,
    )
    assert not _resource_gates_pass(
        pre_post_manifest_identical=True,
        peak_memory_fraction=float("nan"),
    )


def test_canonical_json_is_strict_and_round_trips() -> None:
    payload = {"terminal_status": "CANARY_NOT_USABLE", "value": None}
    encoded = _canonical_json_bytes(payload)

    assert encoded.endswith(b"\n")
    assert json.loads(encoded) == payload
    with pytest.raises(ValueError, match="Out of range float values"):
        _canonical_json_bytes({"nonfinite": float("nan")})


def test_source_manifest_covers_complete_route_and_excludes_dotenv() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    paths = _tracked_paths(repo_root)
    required = {
        Path("benchmarks/run_single_stage_fullspace_gauss_newton_canary.py"),
        Path("docs/single_stage_jax_gpu_gauss_newton_canary_implementation_plan.md"),
        Path("src/simsopt_jax/core/__init__.py"),
        Path("src/simsopt_jax/core/quasisymmetry.py"),
        Path("src/simsopt_jax/geo/optimizers/projected_hvp_trust_region.py"),
        Path("src/simsopt_jax/objectives/single_stage_fullspace.py"),
        Path("src/simsopt_jax/objectives/single_stage_fullspace_residuals.py"),
        Path("src/simsopt_jax/solve/fullspace_gauss_newton_canary.py"),
        Path("tests/benchmarks/test_single_stage_fullspace_gauss_newton_canary.py"),
        Path("tests/geo/test_fullspace_gauss_newton_canary.py"),
        Path("tests/geo/test_projected_hvp_trust_region.py"),
        Path("tests/jax/objectives/test_single_stage_fullspace_core.py"),
    }

    assert set(SOURCE_PATHS) == required
    assert required.issubset(paths)
    assert all(
        path.name == ".env.example"
        or (path.name != ".env" and not path.name.startswith(".env."))
        for path in paths
    )
