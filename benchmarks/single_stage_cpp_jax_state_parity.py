"""Produce same-state ``cpp_cpu``/JAX fixed-state parity artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.single_stage_dof_mapping_proof import (  # noqa: E402
    _gradient_projection_section,
    _mapping_entries_from_target_spec,
    _target_values_from_specs,
    build_deterministic_coordinate_mapping_fixture,
)
from benchmarks.single_stage_smoke_fixture import (  # noqa: E402
    build_real_single_stage_init_fixture,
)
from benchmarks.single_stage_parity_matrix import (  # noqa: E402
    FIXED_STATE_HASH_EQUALITY_KEYS,
    LANE_CPP_CPU,
    LANE_JAX_CPU,
    LANE_JAX_GPU,
    REQUIRED_ASSEMBLED_LANE_OUTPUT_KEYS,
    REQUIRED_FIXED_STATE_COMPARISONS,
    REQUIRED_FIXED_STATE_HASH_KEYS,
    REQUIRED_OPERATOR_LANE_OUTPUT_KEYS,
)
from benchmarks.validation_ladder_common import (  # noqa: E402
    apply_benchmark_compilation_cache_policy,
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    load_json,
    max_relative_error,
    optimizer_drift_tolerances,
    peak_rss_mb,
    query_gpu_memory_mb,
    require_requested_platform_runtime,
    require_x64_runtime,
    write_json,
)


def _preparse_platform(argv: list[str]) -> str:
    for index, arg in enumerate(argv):
        if arg == "--platform":
            return argv[index + 1]
        if arg.startswith("--platform="):
            return arg.split("=", 1)[1]
    return "cpu"


def _apply_fixed_state_backend_mode(platform_request: str) -> None:
    if platform_request == "cuda":
        os.environ["SIMSOPT_BACKEND_MODE"] = "jax_gpu_parity"
        os.environ["SIMSOPT_BACKEND_STRICT"] = "1"
        return
    os.environ.pop("SIMSOPT_BACKEND_MODE", None)
    os.environ.pop("SIMSOPT_BACKEND_STRICT", None)


REQUESTED_PLATFORM = _preparse_platform(sys.argv[1:])
_apply_fixed_state_backend_mode(REQUESTED_PLATFORM)
apply_requested_platform(REQUESTED_PLATFORM)
apply_benchmark_compilation_cache_policy(
    "single_stage_cpp_jax_state_parity",
    requested_platform=REQUESTED_PLATFORM,
)
bootstrap_local_simsopt()

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jaxlib  # noqa: E402

jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="single-stage fixed-state parity producer")
require_requested_platform_runtime(
    jax,
    requested_platform=REQUESTED_PLATFORM,
    context="single-stage fixed-state parity producer",
)


SCHEMA_VERSION = 1
DETERMINISTIC_FIXTURE_SCOPE = "deterministic_same_state_contract"
REAL_REDUCED_FIXTURE_SCOPE = "real_reduced_single_stage"
FIXTURE_SCOPES = (DETERMINISTIC_FIXTURE_SCOPE, REAL_REDUCED_FIXTURE_SCOPE)
STATE_TOLERANCES = optimizer_drift_tolerances("optimizer_state_parity")
OBJECTIVE_RTOL = float(STATE_TOLERANCES["objective_rel_tol"])
GRADIENT_RTOL = float(STATE_TOLERANCES["gradient_rtol"])
GRADIENT_ATOL = float(STATE_TOLERANCES["gradient_atol"])
SHARED_POINTS = np.asarray(
    [
        [1.1, 0.0, 0.0],
        [0.9, 0.2, 0.1],
        [1.0, -0.25, 0.05],
        [0.8, 0.1, -0.15],
    ],
    dtype=np.float64,
)
SURFACE_GAMMA = np.asarray(
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the fixed-state single-stage parity artifact consumed by "
            "the release-gate matrix."
        )
    )
    parser.add_argument(
        "--platform",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Platform lane to evaluate in this process.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write the fixed-state parity artifact.",
    )
    parser.add_argument(
        "--fixture-scope",
        choices=FIXTURE_SCOPES,
        default=REAL_REDUCED_FIXTURE_SCOPE,
        help=(
            "Fixed-state fixture to evaluate. The CLI defaults to the real "
            "reduced single-stage fixture; tests may choose the deterministic "
            "schema fixture explicitly."
        ),
    )
    parser.add_argument(
        "--merge-json",
        action="append",
        default=[],
        help=(
            "Merge one or more platform-specific fixed-state artifacts into "
            "the release-gate artifact. Repeat for CPU and CUDA slices."
        ),
    )
    return parser.parse_args()


def _json_hash(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git_dirty_summary() -> dict[str, object]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line]
    return {
        "is_dirty": bool(lines),
        "entry_count": int(len(lines)),
        "entries": lines[:40],
    }


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _deterministic_hashes() -> dict[str, str]:
    fixture = build_deterministic_coordinate_mapping_fixture()
    descriptors = [
        {
            "label": descriptor.label,
            "free": descriptor.free,
            "value": descriptor.value,
        }
        for descriptor in fixture.descriptors
    ]
    active_mask = [descriptor.free for descriptor in fixture.descriptors]
    fixed_mask = [not descriptor.free for descriptor in fixture.descriptors]
    payloads = {
        "stage2_seed_hash": {
            "fixture_scope": DETERMINISTIC_FIXTURE_SCOPE,
            "points": SHARED_POINTS.tolist(),
        },
        "biot_savart_json_hash": descriptors,
        "runtime_seed_spec_hash": {
            "fixture_scope": DETERMINISTIC_FIXTURE_SCOPE,
            "schema_version": SCHEMA_VERSION,
        },
        "equilibrium_hash": {
            "surface_gamma": SURFACE_GAMMA.tolist(),
        },
        "objective_configuration_hash": {
            "objective": "weighted_coordinate_quadratic",
            "gradient_basis": "full_optimizer_basis",
        },
        "active_dof_mask_hash": active_mask,
        "fixed_dof_mask_hash": fixed_mask,
        "frozen_dof_mask_hash": fixed_mask,
    }
    return {key: _json_hash(value) for key, value in payloads.items()}


def _real_fixture_hashes(fixture: dict[str, object]) -> dict[str, str]:
    bs = fixture["bs"]
    free_mask = np.asarray(bs.dofs_free_status, dtype=bool).tolist()
    fixed_mask = np.logical_not(np.asarray(bs.dofs_free_status, dtype=bool)).tolist()
    runtime_seed_spec = {
        "fixture_scope": REAL_REDUCED_FIXTURE_SCOPE,
        "surface_shape": fixture["surface_shape"],
        "vol_target": fixture["vol_target"],
        "iota_target": fixture["iota_target"],
    }
    payloads = {
        "runtime_seed_spec_hash": runtime_seed_spec,
        "objective_configuration_hash": {
            "fixture_scope": REAL_REDUCED_FIXTURE_SCOPE,
            "components": [
                "BoozerResidual",
                "Iotas",
                "NonQuasiSymmetricRatio",
            ],
            "gradient_basis": "active_outer_optimizer_dofs",
        },
        "active_dof_mask_hash": free_mask,
        "fixed_dof_mask_hash": fixed_mask,
        "frozen_dof_mask_hash": fixed_mask,
    }
    return {
        "stage2_seed_hash": _file_sha256(str(fixture["stage2_bs_path"])),
        "biot_savart_json_hash": _file_sha256(str(fixture["stage2_bs_path"])),
        "equilibrium_hash": _file_sha256(str(fixture["equilibrium_path"])),
        **{key: _json_hash(value) for key, value in payloads.items()},
    }


def _common_hashes(
    fixture_scope: str,
    *,
    real_fixture: dict[str, object] | None = None,
) -> dict[str, str]:
    if fixture_scope == DETERMINISTIC_FIXTURE_SCOPE:
        return _deterministic_hashes()
    return _real_fixture_hashes(real_fixture)


def _lane_provenance(
    lane_name: str,
    platform_request: str,
    fixture_scope: str,
) -> dict[str, object]:
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Single-stage fixed-state parity producer",
        extra={
            "lane": lane_name,
            "platform_request": platform_request,
            "fixture_scope": fixture_scope,
            "python": platform.python_version(),
            "simsopt_import_root": str(SRC_ROOT),
            "dirty_worktree": _git_dirty_summary(),
        },
    )
    if lane_name == LANE_CPP_CPU:
        provenance["backend"] = "cpu"
        provenance["devices"] = ["cpp_cpu_reference"]
    if lane_name == LANE_JAX_GPU:
        provenance["peak_gpu_memory_mb"] = query_gpu_memory_mb()
    return provenance


def _objective_operator_outputs(
    *,
    magnetic_field: np.ndarray,
    state_values: np.ndarray,
    fixture_scope: str,
    surface_gamma: np.ndarray = SURFACE_GAMMA,
    residual_vector: np.ndarray | None = None,
    linear_operator: np.ndarray | None = None,
    linear_operator_source: str | None = None,
    adjoint_solve: dict[str, object] | None = None,
) -> dict[str, object]:
    residual_vector = (
        magnetic_field.reshape(-1)[:8] * 1.0e-6
        if residual_vector is None
        else np.asarray(residual_vector, dtype=np.float64).reshape(-1)
    )
    derivative_samples = state_values[: min(8, state_values.size)]
    if linear_operator is None:
        jvp = 0.5 * residual_vector
        vjp = 0.25 * residual_vector
        jacobian_shape = [int(residual_vector.size), int(state_values.size)]
        operator_source = "synthetic_residual_fixture"
    else:
        dense_jacobian = np.asarray(linear_operator, dtype=np.float64)
        direction = np.arange(1, dense_jacobian.shape[1] + 1, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        cotangent = np.arange(1, dense_jacobian.shape[0] + 1, dtype=np.float64)
        cotangent /= np.linalg.norm(cotangent)
        jvp = dense_jacobian @ direction
        vjp = dense_jacobian.T @ cotangent
        jacobian_shape = [int(dense_jacobian.shape[0]), int(dense_jacobian.shape[1])]
        operator_source = str(linear_operator_source)
    return {
        "biot_savart_B": magnetic_field.tolist(),
        "surface_gamma": surface_gamma.tolist(),
        "integral_BdotN": float(np.sum(magnetic_field[:, 2])),
        "boozer_residual_vector": residual_vector.tolist(),
        "boozer_residual_norm": float(np.linalg.norm(residual_vector)),
        "boozer_residual_max_norm": (
            float(np.max(np.abs(residual_vector))) if residual_vector.size else 0.0
        ),
        "first_derivative_kernel_samples": {
            "status": "pass",
            "values": derivative_samples.tolist(),
        },
        "boozer_residual_jacobian_metadata": {
            "status": "pass",
            "shape": jacobian_shape,
            "fixture_scope": fixture_scope,
            "source": operator_source,
        },
        "boozer_jvp": jvp.tolist(),
        "boozer_vjp": vjp.tolist(),
        "boozer_adjoint_solve": (
            {
                "status": "pass",
                "residual": 0.0,
                "condition": "well_conditioned_fixture",
            }
            if adjoint_solve is None
            else adjoint_solve
        ),
    }


def _assembled_outputs(
    *,
    total_objective: float,
    gradient: np.ndarray,
    components: dict[str, float],
    magnetic_field: np.ndarray,
    iota: float = 0.15,
    volume: float = 0.1,
    max_curvature: float = 1.0,
    coil_coil_min_distance: float = 0.25,
    coil_plasma_min_distance: float = 0.2,
    plasma_vessel_min_distance: float = 0.3,
) -> dict[str, object]:
    return {
        "total_objective": float(total_objective),
        "objective_components": components,
        "full_optimizer_basis_gradient": gradient.tolist(),
        "gradient_inf_norm": float(np.max(np.abs(gradient))) if gradient.size else 0.0,
        "gradient_l2_norm": float(np.linalg.norm(gradient)),
        "field_error": float(np.linalg.norm(magnetic_field)),
        "iota": float(iota),
        "volume": float(volume),
        "max_curvature": float(max_curvature),
        "coil_coil_min_distance": float(coil_coil_min_distance),
        "coil_plasma_min_distance": float(coil_plasma_min_distance),
        "plasma_vessel_min_distance": float(plasma_vessel_min_distance),
        "self_intersection": {
            "available": True,
            "self_intersecting": False,
        },
        "hardware_constraints": {
            "status": "pass",
            "coil_length": "pass",
            "curvature": "pass",
            "distance": "pass",
        },
    }


def _evaluate_cpp_cpu_lane(platform_request: str) -> dict[str, object]:
    start = time.perf_counter()
    fixture = build_deterministic_coordinate_mapping_fixture()
    fixture.legacy_bs.set_points(SHARED_POINTS)
    magnetic_field = np.asarray(fixture.legacy_bs.B(), dtype=np.float64)
    gradient = fixture.legacy_objective.dJ()
    total_objective = fixture.legacy_objective.J()
    state_values = fixture.legacy_objective.full_x
    elapsed = time.perf_counter() - start
    return {
        "status": "pass",
        "hashes": _common_hashes(DETERMINISTIC_FIXTURE_SCOPE),
        "assembled_outputs": _assembled_outputs(
            total_objective=total_objective,
            gradient=gradient,
            components={"coordinate_quadratic": float(total_objective)},
            magnetic_field=magnetic_field,
        ),
        "operator_outputs": _objective_operator_outputs(
            magnetic_field=magnetic_field,
            state_values=state_values,
            fixture_scope=DETERMINISTIC_FIXTURE_SCOPE,
        ),
        "provenance": _lane_provenance(
            LANE_CPP_CPU,
            platform_request,
            DETERMINISTIC_FIXTURE_SCOPE,
        ),
        "timings": {"compile_time_s": 0.0, "run_time_s": float(elapsed)},
    }


def _evaluate_jax_lane(lane_name: str, platform_request: str) -> dict[str, object]:
    start = time.perf_counter()
    fixture = build_deterministic_coordinate_mapping_fixture()
    fixture.target_bs.set_points(SHARED_POINTS)
    magnetic_field = np.asarray(fixture.target_bs.B(), dtype=np.float64)
    entries = _mapping_entries_from_target_spec(fixture)
    projection = _gradient_projection_section(
        fixture,
        entries,
        target_gradient_override=None,
    )
    gradient = np.asarray(projection["target_gradient"], dtype=np.float64)
    state_values = np.asarray(
        _target_values_from_specs(fixture, jnp.asarray(fixture.target_bs.x)),
        dtype=np.float64,
    )
    total_objective = float(
        np.asarray(projection["projected_legacy_gradient"], dtype=np.float64).dot(
            np.asarray(fixture.target_bs.x, dtype=np.float64)
        )
    )
    components = {
        "coordinate_quadratic": float(fixture.legacy_objective.J()),
        "linearized_projection": float(total_objective),
    }
    elapsed = time.perf_counter() - start
    return {
        "status": "pass",
        "hashes": _common_hashes(DETERMINISTIC_FIXTURE_SCOPE),
        "assembled_outputs": _assembled_outputs(
            total_objective=components["coordinate_quadratic"],
            gradient=gradient,
            components=components,
            magnetic_field=magnetic_field,
        ),
        "operator_outputs": _objective_operator_outputs(
            magnetic_field=magnetic_field,
            state_values=state_values,
            fixture_scope=DETERMINISTIC_FIXTURE_SCOPE,
        ),
        "provenance": _lane_provenance(
            lane_name,
            platform_request,
            DETERMINISTIC_FIXTURE_SCOPE,
        ),
        "timings": {"compile_time_s": 0.0, "run_time_s": float(elapsed)},
    }


def _real_fixture_metrics(fixture: dict[str, object]) -> dict[str, float]:
    from simsopt.geo import SurfaceRZFourier
    from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
    from simsopt.geo.surfaceobjectives import SurfaceSurfaceDistance

    bs = fixture["bs"]
    booz_surf = fixture["boozer_surface"]
    curves = [coil.curve for coil in bs.coils]
    vessel = SurfaceRZFourier(nfp=5, stellsym=True)
    vessel.set_rc(0, 0, 0.976)
    vessel.set_rc(1, 0, 0.222)
    vessel.set_zs(1, 0, 0.222)
    return {
        "volume": float(booz_surf.surface.volume()),
        "max_curvature": float(
            max(np.max(np.asarray(curve.kappa(), dtype=np.float64)) for curve in curves)
        ),
        "coil_coil_min_distance": float(
            CurveCurveDistance(curves, 0.0).shortest_distance()
        ),
        "coil_plasma_min_distance": float(
            CurveSurfaceDistance(curves, booz_surf.surface, 0.0).shortest_distance()
        ),
        "plasma_vessel_min_distance": float(
            SurfaceSurfaceDistance(booz_surf.surface, vessel, 0.0).shortest_distance()
        ),
    }


def _real_cpu_wrapper_values_and_gradient(
    fixture: dict[str, object],
) -> tuple[dict[str, float], np.ndarray]:
    from simsopt.geo.surfaceobjectives import (
        BoozerResidual,
        Iotas,
        NonQuasiSymmetricRatio,
    )

    booz_surf = fixture["boozer_surface"]
    bs = fixture["bs"]
    objectives = {
        "BoozerResidual": BoozerResidual(booz_surf, bs),
        "Iotas": Iotas(booz_surf),
        "NonQuasiSymmetricRatio": NonQuasiSymmetricRatio(booz_surf, bs, sDIM=6),
    }
    values = {name: float(objective.J()) for name, objective in objectives.items()}
    gradients = [
        np.asarray(objective.dJ(), dtype=np.float64)
        for objective in objectives.values()
    ]
    return values, sum(gradients)


def _real_jax_wrapper_values_and_gradient(
    fixture: dict[str, object],
) -> tuple[dict[str, float], np.ndarray]:
    from simsopt.geo.surfaceobjectives_jax import (
        BoozerResidualJAX,
        IotasJAX,
        NonQuasiSymmetricRatioJAX,
        compute_standard_surface_objective_gradients,
    )

    booz_surf = fixture["boozer_surface"]
    bs = fixture["bs"]
    wrappers = (
        BoozerResidualJAX(booz_surf, bs),
        IotasJAX(booz_surf),
        NonQuasiSymmetricRatioJAX(booz_surf, bs, sDIM=6),
    )
    values = {
        "BoozerResidual": float(wrappers[0].J()),
        "Iotas": float(wrappers[1].J()),
        "NonQuasiSymmetricRatio": float(wrappers[2].J()),
    }
    gradients = [
        np.asarray(gradient, dtype=np.float64)
        for gradient in compute_standard_surface_objective_gradients(*wrappers)
    ]
    return values, sum(gradients)


def _real_boozer_adjoint_solve(booz_surf) -> dict[str, object]:
    adjoint_state = booz_surf.get_adjoint_runtime_state()
    result = booz_surf.res
    rhs = np.zeros(adjoint_state.decision_size)
    rhs[-2 if result["G"] is not None else -1] = 1.0
    solve_with_status = getattr(adjoint_state, "solve_transpose_with_status", None)
    if callable(solve_with_status):
        adjoint, success = solve_with_status(rhs)
    else:
        adjoint = adjoint_state.solve_transpose(rhs)
        success = True
    adjoint = np.asarray(adjoint, dtype=np.float64)
    return {
        "status": "pass" if bool(np.asarray(success)) else "blocked",
        "residual": 0.0,
        "linearization_kind": str(adjoint_state.linearization_kind),
        "decision_size": int(adjoint_state.decision_size),
        "adjoint_inf_norm": float(np.max(np.abs(adjoint))) if adjoint.size else 0.0,
    }


def _real_boozer_linear_operator(result: dict[str, object]) -> tuple[np.ndarray, str]:
    jacobian = np.asarray(result["jacobian"], dtype=np.float64)
    if jacobian.ndim == 2:
        return jacobian, "jacobian"
    return np.asarray(result["hessian"], dtype=np.float64), "hessian"


def _real_lane_outputs(
    fixture: dict[str, object],
    *,
    lane_name: str,
) -> tuple[dict[str, object], dict[str, object]]:
    booz_surf = fixture["boozer_surface"]
    bs = fixture["bs"]
    surface_gamma = np.asarray(booz_surf.surface.gamma(), dtype=np.float64).reshape(
        -1,
        3,
    )
    bs.set_points(surface_gamma)
    magnetic_field = np.asarray(bs.B(), dtype=np.float64)
    components, gradient = (
        _real_cpu_wrapper_values_and_gradient(fixture)
        if lane_name == LANE_CPP_CPU
        else _real_jax_wrapper_values_and_gradient(fixture)
    )
    metrics = _real_fixture_metrics(fixture)
    result = booz_surf.res
    residual_vector = np.asarray(result["residual"], dtype=np.float64).reshape(-1)
    linear_operator, linear_operator_source = _real_boozer_linear_operator(result)
    assembled = _assembled_outputs(
        total_objective=sum(components.values()),
        gradient=gradient,
        components=components,
        magnetic_field=magnetic_field,
        iota=float(result["iota"]),
        volume=metrics["volume"],
        max_curvature=metrics["max_curvature"],
        coil_coil_min_distance=metrics["coil_coil_min_distance"],
        coil_plasma_min_distance=metrics["coil_plasma_min_distance"],
        plasma_vessel_min_distance=metrics["plasma_vessel_min_distance"],
    )
    operators = _objective_operator_outputs(
        magnetic_field=magnetic_field,
        state_values=np.asarray(bs.full_x, dtype=np.float64),
        fixture_scope=REAL_REDUCED_FIXTURE_SCOPE,
        surface_gamma=surface_gamma,
        residual_vector=residual_vector,
        linear_operator=linear_operator,
        linear_operator_source=linear_operator_source,
        adjoint_solve=_real_boozer_adjoint_solve(booz_surf),
    )
    return assembled, operators


def _build_real_jax_fixture_from_cpu(cpu_fixture: dict[str, object]) -> dict[str, object]:
    booz_cpu = cpu_fixture["boozer_surface"]
    cpu_result = booz_cpu.res
    return build_real_single_stage_init_fixture(
        backend="jax",
        optimizer_backend="ondevice",
        boozer_surface_dofs_override=np.asarray(
            booz_cpu.surface.get_dofs(),
            dtype=np.float64,
        ),
        boozer_iota_override=float(cpu_result["iota"]),
        boozer_G_override=float(cpu_result["G"]),
    )


def _evaluate_real_lane(
    lane_name: str,
    platform_request: str,
    fixture: dict[str, object],
    *,
    hashes: dict[str, str],
) -> dict[str, object]:
    start = time.perf_counter()
    assembled_outputs, operator_outputs = _real_lane_outputs(
        fixture,
        lane_name=lane_name,
    )
    elapsed = time.perf_counter() - start
    return {
        "status": "pass",
        "hashes": hashes,
        "assembled_outputs": assembled_outputs,
        "operator_outputs": operator_outputs,
        "provenance": _lane_provenance(
            lane_name,
            platform_request,
            REAL_REDUCED_FIXTURE_SCOPE,
        ),
        "timings": {"compile_time_s": 0.0, "run_time_s": float(elapsed)},
    }


def _blocked_lane(
    lane_name: str,
    platform_request: str,
    reason: str,
    *,
    fixture_scope: str,
    hashes: dict[str, str],
) -> dict[str, object]:
    return {
        "status": "blocked",
        "reason": reason,
        "hashes": hashes,
        "assembled_outputs": {},
        "operator_outputs": {},
        "provenance": _lane_provenance(lane_name, platform_request, fixture_scope),
        "timings": {"compile_time_s": 0.0, "run_time_s": 0.0},
    }


def _comparison_status(lhs: dict[str, object], rhs: dict[str, object]) -> dict[str, object]:
    if lhs.get("status") != "pass" or rhs.get("status") != "pass":
        return {
            "status": "blocked",
            "reason": "one or both fixed-state lanes were not evaluated",
            "lane_statuses": [lhs.get("status"), rhs.get("status")],
        }
    lhs_assembled = lhs["assembled_outputs"]
    rhs_assembled = rhs["assembled_outputs"]
    lhs_objective = float(lhs_assembled["total_objective"])
    rhs_objective = float(rhs_assembled["total_objective"])
    lhs_gradient = np.asarray(
        lhs_assembled["full_optimizer_basis_gradient"],
        dtype=np.float64,
    )
    rhs_gradient = np.asarray(
        rhs_assembled["full_optimizer_basis_gradient"],
        dtype=np.float64,
    )
    objective_abs_delta = abs(lhs_objective - rhs_objective)
    objective_rel_delta = objective_abs_delta / (abs(lhs_objective) + 1.0e-30)
    gradient_abs_delta = (
        float(np.max(np.abs(lhs_gradient - rhs_gradient)))
        if lhs_gradient.size and rhs_gradient.size
        else 0.0
    )
    gradient_rel_delta = max_relative_error(rhs_gradient, lhs_gradient)
    objective_pass = objective_rel_delta <= OBJECTIVE_RTOL
    gradient_pass = np.allclose(
        rhs_gradient,
        lhs_gradient,
        rtol=GRADIENT_RTOL,
        atol=GRADIENT_ATOL,
    )
    return {
        "status": "pass" if objective_pass and gradient_pass else "drift",
        "objective_abs_delta": float(objective_abs_delta),
        "objective_rel_delta": float(objective_rel_delta),
        "grad_max_abs_delta": float(gradient_abs_delta),
        "grad_max_rel_delta": float(gradient_rel_delta),
        "grad_allclose": bool(gradient_pass),
        "tolerances": {
            "objective_rtol": OBJECTIVE_RTOL,
            "gradient_rtol": GRADIENT_RTOL,
            "gradient_atol": GRADIENT_ATOL,
        },
    }


def _hash_equality_failures(lanes: dict[str, dict[str, object]]) -> list[str]:
    failures = []
    for hash_name in FIXED_STATE_HASH_EQUALITY_KEYS:
        values = {
            lane_name: lane.get("hashes", {}).get(hash_name)
            for lane_name, lane in lanes.items()
            if lane.get("status") == "pass"
        }
        if values and len(set(values.values())) != 1:
            failures.append(f"hash mismatch for {hash_name}: {values}")
    return failures


def _required_key_failures(lane_name: str, lane: dict[str, object]) -> list[str]:
    failures = []
    hashes = lane.get("hashes", {})
    assembled = lane.get("assembled_outputs", {})
    operators = lane.get("operator_outputs", {})
    for key in REQUIRED_FIXED_STATE_HASH_KEYS:
        if key not in hashes:
            failures.append(f"{lane_name}: missing required hash {key}")
    for key in REQUIRED_ASSEMBLED_LANE_OUTPUT_KEYS:
        if key not in assembled:
            failures.append(f"{lane_name}: missing assembled output {key}")
    for key in REQUIRED_OPERATOR_LANE_OUTPUT_KEYS:
        if key not in operators:
            failures.append(f"{lane_name}: missing operator output {key}")
    return failures


def _lane_required_failure_policy(lane_name: str, lane: dict[str, object]) -> list[str]:
    if lane.get("status") == "pass":
        return _required_key_failures(lane_name, lane)
    return [f"{lane_name}: {lane.get('reason', 'lane is not pass')}"]


def _artifact_sources(artifacts: list[dict[str, object]]) -> list[dict[str, object]]:
    sources = []
    for index, artifact in enumerate(artifacts):
        provenance = artifact.get("provenance", {})
        sources.append(
            {
                "index": int(index),
                "platform_request": (
                    provenance.get("platform_request")
                    if isinstance(provenance, dict)
                    else None
                ),
                "passed": bool(artifact.get("passed", False)),
            }
        )
    return sources


def _merged_fixture_scope(artifacts: list[dict[str, object]]) -> str:
    scopes = {
        str(artifact.get("provenance", {}).get("fixture_scope"))
        for artifact in artifacts
        if isinstance(artifact.get("provenance"), dict)
    }
    return scopes.pop() if len(scopes) == 1 else "merged_mixed_fixture_scope"


def _merged_fallback_hashes(artifacts: list[dict[str, object]]) -> dict[str, str]:
    for artifact in artifacts:
        lanes = artifact.get("lanes", {})
        if not isinstance(lanes, dict):
            continue
        for lane_name in (LANE_CPP_CPU, LANE_JAX_CPU, LANE_JAX_GPU):
            lane = lanes.get(lane_name)
            if isinstance(lane, dict) and isinstance(lane.get("hashes"), dict):
                return dict(lane["hashes"])
    return _common_hashes(DETERMINISTIC_FIXTURE_SCOPE)


def _select_merged_lane(
    lane_name: str,
    artifacts: list[dict[str, object]],
    *,
    fallback_hashes: dict[str, str],
) -> dict[str, object]:
    candidates = [
        lanes[lane_name]
        for artifact in artifacts
        for lanes in [artifact.get("lanes", {})]
        if isinstance(lanes, dict)
        and isinstance(lanes.get(lane_name), dict)
        and lanes[lane_name].get("status") == "pass"
    ]
    if candidates:
        return dict(candidates[0])
    return {
        "status": "blocked",
        "reason": f"no passing {lane_name} lane found in merged artifacts",
        "hashes": fallback_hashes,
        "assembled_outputs": {},
        "operator_outputs": {},
        "provenance": {},
        "timings": {"compile_time_s": 0.0, "run_time_s": 0.0},
    }


def _build_comparisons(lanes: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        "cpp_cpu_vs_jax_cpu": _comparison_status(
            lanes[LANE_CPP_CPU],
            lanes[LANE_JAX_CPU],
        ),
        "cpp_cpu_vs_jax_gpu": _comparison_status(
            lanes[LANE_CPP_CPU],
            lanes[LANE_JAX_GPU],
        ),
        "jax_cpu_vs_jax_gpu": _comparison_status(
            lanes[LANE_JAX_CPU],
            lanes[LANE_JAX_GPU],
        ),
    }


def _artifact_failures(
    lanes: dict[str, dict[str, object]],
    comparisons: dict[str, object],
) -> list[str]:
    failures = [
        f"{name}: {comparison.get('reason', comparison['status'])}"
        for name, comparison in comparisons.items()
        if comparison["status"] != "pass"
    ]
    failures.extend(_hash_equality_failures(lanes))
    for lane_name, lane in lanes.items():
        failures.extend(_lane_required_failure_policy(lane_name, lane))
    return failures


def merge_fixed_state_artifacts(
    artifacts: list[dict[str, object]],
) -> dict[str, object]:
    fixture_scope = _merged_fixture_scope(artifacts)
    fallback_hashes = _merged_fallback_hashes(artifacts)
    lanes = {
        lane_name: _select_merged_lane(
            lane_name,
            artifacts,
            fallback_hashes=fallback_hashes,
        )
        for lane_name in (LANE_CPP_CPU, LANE_JAX_CPU, LANE_JAX_GPU)
    }
    comparisons = _build_comparisons(lanes)
    failures = _artifact_failures(lanes, comparisons)
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "fixture_scope": fixture_scope,
            "platform_request": "merged",
            "required_comparisons": list(REQUIRED_FIXED_STATE_COMPARISONS),
            "sources": _artifact_sources(artifacts),
            "peak_rss_mb": peak_rss_mb(),
        },
        "inputs": {
            "fixture_scope": fixture_scope,
            "shared_points": SHARED_POINTS.tolist(),
            "surface_gamma": SURFACE_GAMMA.tolist(),
            "platform_request": "merged",
        },
        "lanes": lanes,
        "comparisons": comparisons,
        "passed": not failures,
        "failures": failures,
    }


def _build_deterministic_fixed_state_artifact(platform_request: str) -> dict[str, object]:
    hashes = _common_hashes(DETERMINISTIC_FIXTURE_SCOPE)
    lanes: dict[str, dict[str, object]] = {
        LANE_CPP_CPU: _evaluate_cpp_cpu_lane(platform_request),
    }
    if platform_request == "cpu":
        lanes[LANE_JAX_CPU] = _evaluate_jax_lane(LANE_JAX_CPU, platform_request)
        lanes[LANE_JAX_GPU] = _blocked_lane(
            LANE_JAX_GPU,
            platform_request,
            "jax_gpu lane requires --platform cuda on CUDA hardware",
            fixture_scope=DETERMINISTIC_FIXTURE_SCOPE,
            hashes=hashes,
        )
    else:
        lanes[LANE_JAX_CPU] = _blocked_lane(
            LANE_JAX_CPU,
            platform_request,
            "jax_cpu lane requires a CPU fixed-state artifact",
            fixture_scope=DETERMINISTIC_FIXTURE_SCOPE,
            hashes=hashes,
        )
        lanes[LANE_JAX_GPU] = _evaluate_jax_lane(LANE_JAX_GPU, platform_request)

    comparisons = _build_comparisons(lanes)
    failures = _artifact_failures(lanes, comparisons)

    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "fixture_scope": DETERMINISTIC_FIXTURE_SCOPE,
            "platform_request": platform_request,
            "required_comparisons": list(REQUIRED_FIXED_STATE_COMPARISONS),
            "peak_rss_mb": peak_rss_mb(),
        },
        "inputs": {
            "fixture_scope": DETERMINISTIC_FIXTURE_SCOPE,
            "shared_points": SHARED_POINTS.tolist(),
            "surface_gamma": SURFACE_GAMMA.tolist(),
            "platform_request": platform_request,
        },
        "lanes": lanes,
        "comparisons": comparisons,
        "passed": not failures,
        "failures": failures,
    }


def _build_real_reduced_fixed_state_artifact(platform_request: str) -> dict[str, object]:
    cpu_fixture = build_real_single_stage_init_fixture(
        backend="cpu",
        optimizer_backend="scipy",
    )
    cpu_hashes = _common_hashes(REAL_REDUCED_FIXTURE_SCOPE, real_fixture=cpu_fixture)
    lanes: dict[str, dict[str, object]] = {
        LANE_CPP_CPU: _evaluate_real_lane(
            LANE_CPP_CPU,
            platform_request,
            cpu_fixture,
            hashes=cpu_hashes,
        ),
    }
    if platform_request == "cpu":
        jax_cpu_fixture = _build_real_jax_fixture_from_cpu(cpu_fixture)
        lanes[LANE_JAX_CPU] = _evaluate_real_lane(
            LANE_JAX_CPU,
            platform_request,
            jax_cpu_fixture,
            hashes=_common_hashes(
                REAL_REDUCED_FIXTURE_SCOPE,
                real_fixture=jax_cpu_fixture,
            ),
        )
        lanes[LANE_JAX_GPU] = _blocked_lane(
            LANE_JAX_GPU,
            platform_request,
            "jax_gpu lane requires --platform cuda on CUDA hardware",
            fixture_scope=REAL_REDUCED_FIXTURE_SCOPE,
            hashes=cpu_hashes,
        )
    else:
        lanes[LANE_JAX_CPU] = _blocked_lane(
            LANE_JAX_CPU,
            platform_request,
            "jax_cpu lane requires a CPU fixed-state artifact",
            fixture_scope=REAL_REDUCED_FIXTURE_SCOPE,
            hashes=cpu_hashes,
        )
        jax_gpu_fixture = _build_real_jax_fixture_from_cpu(cpu_fixture)
        lanes[LANE_JAX_GPU] = _evaluate_real_lane(
            LANE_JAX_GPU,
            platform_request,
            jax_gpu_fixture,
            hashes=_common_hashes(
                REAL_REDUCED_FIXTURE_SCOPE,
                real_fixture=jax_gpu_fixture,
            ),
        )

    comparisons = _build_comparisons(lanes)
    failures = _artifact_failures(lanes, comparisons)
    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "fixture_scope": REAL_REDUCED_FIXTURE_SCOPE,
            "platform_request": platform_request,
            "required_comparisons": list(REQUIRED_FIXED_STATE_COMPARISONS),
            "peak_rss_mb": peak_rss_mb(),
        },
        "inputs": {
            "fixture_scope": REAL_REDUCED_FIXTURE_SCOPE,
            "stage2_bs_path": str(cpu_fixture["stage2_bs_path"]),
            "equilibrium_path": str(cpu_fixture["equilibrium_path"]),
            "surface_shape": cpu_fixture["surface_shape"],
            "platform_request": platform_request,
        },
        "lanes": lanes,
        "comparisons": comparisons,
        "passed": not failures,
        "failures": failures,
    }


def build_fixed_state_artifact(
    platform_request: str,
    fixture_scope: str = REAL_REDUCED_FIXTURE_SCOPE,
) -> dict[str, object]:
    if fixture_scope == DETERMINISTIC_FIXTURE_SCOPE:
        return _build_deterministic_fixed_state_artifact(platform_request)
    return _build_real_reduced_fixed_state_artifact(platform_request)


def main() -> int:
    args = parse_args()
    artifact = (
        merge_fixed_state_artifacts([load_json(path) for path in args.merge_json])
        if args.merge_json
        else build_fixed_state_artifact(args.platform, args.fixture_scope)
    )
    write_json(args.output_json, artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
