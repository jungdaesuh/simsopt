import argparse
import atexit
import copy
from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass, replace
import faulthandler
from functools import lru_cache
import hashlib
import inspect
import io
import json
import logging
import os
import sys
import time
import types

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, EXAMPLE_ROOT)

SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
SRC_ROOT = os.path.join(SIMSOPT_ROOT, "src")
sys.path.insert(0, SRC_ROOT)
sys.path.insert(0, SIMSOPT_ROOT)
sys.path.insert(0, REPO_ROOT)

from repo_bootstrap import bootstrap_local_simsopt, configure_entrypoint_jax_runtime


configure_entrypoint_jax_runtime(
    sys.argv[1:],
    require_cpu_platform_when_flags=(
        "--diagnostic-callbacks",
        "--record-target-lane-invalid-state-events",
    ),
)

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

bootstrap_local_simsopt(SRC_ROOT)

from alm_utils import (
    ALMSettings,
    alm_result_diagnostics_fields,
    minimize_alm,
    validate_alm_cli_args,
)
from banana_opt.current_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    apply_penalty_traversal_forbidden_box_bounds,
    resolve_loaded_tf_current_A,
)
from banana_opt.artifact_contracts import upgrade_legacy_stage2_artifact_results
from banana_opt.hardware_contracts import (
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_TARGET_M,
    COIL_PLASMA_MIN_DIST_M,
    COIL_VESSEL_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    PLASMA_VESSEL_MIN_DIST_M,
    TF_CURRENT_HARD_LIMIT_A,
    validate_banana_winding_surface_radius,
)
from banana_opt.hardware_constraint_schema import (
    build_hardware_constraint_status,
    build_threshold_overrides,
    hardware_constraint_alm_names,
)
from banana_opt.single_stage_constraints import (
    smooth_max_curvature_signed_constraint,
    smooth_min_curve_curve_signed_constraint,
    smooth_min_curve_surface_signed_constraint,
    smooth_min_surface_surface_signed_constraint,
)
from banana_opt.single_stage_objectives import evaluate_alm_objective

# SIMSOPT imports
from simsopt._core.derivative import Derivative, derivative_dec
from simsopt._core.optimizable import Optimizable, load
from simsopt.config import maybe_initialize_distributed_jax
from simsopt.field import BiotSavart
from simsopt.jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
    as_jax_int32 as _as_jax_int32,
    as_runtime_float64 as _as_runtime_float64,
)
from simsopt.geo import (
    BoozerSurface,
    CurveLength,
    LpCurveCurvature,
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    curves_to_vtk,
)
from simsopt.geo.curve import surfrz_gamma_lin
from simsopt.geo.curveobjectives import (
    CurveCurveDistance,
    CurveSurfaceDistance,
    pairwise_min_distance_pure,
)
import simsopt.geo.surface as surface_module
from simsopt.geo.surface_fourier_jax import (
    build_phi_basis,
    build_theta_basis,
    stellsym_scatter_indices,
    surface_gamma_from_dofs,
    surface_gamma_lin_from_dofs,
)
from simsopt.geo.surfaceobjectives import (
    BoozerResidual,
    Iotas,
    NonQuasiSymmetricRatio,
    SurfaceSurfaceDistance,
    Volume,
    boozer_surface_residual,
    boozer_surface_residual_dB,
)
from simsopt.objectives import QuadraticPenalty
from simsopt.objectives.utilities import forward_backward
from simsopt.field.biotsavart_jax_backend import SingleStageRuntimeSpecBiotSavartJAX
from simsopt.jax_core.curve_geometry import (
    closed_curve_self_intersection_summary,
)
from simsopt.jax_core.field import coil_set_spec_from_dof_extraction_spec
import simsopt.jax_core.specs as jax_specs
from simsopt.jax_core.specs import (
    make_single_stage_runtime_spec,
    make_single_stage_seed_spec,
    make_surface_xyz_tensor_fourier_spec,
)
from simsopt.jax_core.surface_rzfourier import (
    surface_rz_fourier_gamma_from_dofs,
    surface_rz_fourier_spec_from_dofs,
)
from hardware_constraints import (
    apply_hardware_constraint_verdict,
    sanitize_json_payload,
)
from jax_host_boundary import host_array, host_bool, host_float
from plotting_utils import cross_section_plot, norm_field_plot, norm_field_summary
from run_metadata import build_artifact_manifest, build_runtime_provenance

DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = (
    DATABASE_EQUILIBRIA_DIR
    if os.path.isdir(DATABASE_EQUILIBRIA_DIR)
    else os.path.join(EXAMPLE_ROOT, "equilibria")
)
DEFAULT_LOCAL_STAGE2_ROOT = os.path.join(EXAMPLE_ROOT, "STAGE_2")
DEFAULT_DATABASE_STAGE2_ROOT = os.path.join(
    REPO_ROOT, "DATABASE", "COIL_OPTIMIZATION", "outputs"
)
DEFAULT_SINGLE_STAGE_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
CURVATURE_THRESHOLD_FLOOR = 20.0
CURVATURE_THRESHOLD_CEILING = MAX_CURVATURE_INV_M
TARGET_LANE_ACCEPTED_STEP_SYNC_CHOICES = ("per-accept", "final-only")
TARGET_LANE_ACCEPTED_STEP_SYNC_DEFAULT = "final-only"
_REFERENCE_OUTER_MAXLS_DEFAULT = 20
_TARGET_OUTER_MAXLS_BENCHMARK_DEFAULT = 4
_TARGET_OUTER_MAXLS_DEFAULT = 8
_TARGET_OUTER_INITIAL_STEP_SIZE_BENCHMARK_DEFAULT = 1.0e-4
_REFERENCE_OUTER_MAXCOR_DEFAULT = 300
_TARGET_OUTER_MAXCOR_DEFAULT = 20
_TARGET_LANE_BOOZER_BFGS_TOL_BENCHMARK_DEFAULT = 1e-6
_TARGET_LANE_BOOZER_NEWTON_TOL_FULL_MEMORY_DEFAULT = 1e-8
_TARGET_LANE_BOOZER_BFGS_MAXITER_BENCHMARK_DEFAULT = 64
_SINGLE_STAGE_RESULTS_SCHEMA_VERSION = 1
_SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME = "single_stage_jax_runtime_spec.json"
_SINGLE_STAGE_JAX_RUNTIME_SPEC_SCHEMA = "simsopt.single_stage.jax_runtime_spec"
_SINGLE_STAGE_JAX_RUNTIME_SPEC_VERSION = 1
_SINGLE_STAGE_JAX_SELF_INTERSECTION_MODE = "supported-surface-jax"
_JAX_SELF_INTERSECTION_UNSUPPORTED_MESSAGE = (
    "JAX production self-intersection requires a supported serialized/spec-backed "
    "surface; run seed conversion first."
)
_JAX_COMPILE_DIAGNOSTICS_SAMPLE_LIMIT = 25
_SURFACE_SELF_INTERSECTION_BISECTION_STEPS = 48
_SURFACE_SELF_INTERSECTION_TOLERANCE_FACTOR = 1.0e-9
SINGLE_STAGE_THRESHOLDED_PHYSICS_CONSTRAINT_NAMES = (
    "qs_error",
    "boozer_residual",
    "iota_penalty",
    "length_penalty",
)
_SINGLE_STAGE_SEARCH_POLICY_PRESERVE_FIRST = "preserve_first"
_SINGLE_STAGE_SEARCH_POLICY_REPAIR_FIRST = "repair_first"
_SINGLE_STAGE_SEARCH_POLICY_GLOBAL_SEARCH = "global_search"
_TIMED_STAGE_LABELS = frozenset(
    {
        "after_boozer_surface_fit",
        "after_boozer_setup",
        "before_boozer_lbfgs",
        "after_boozer_solve",
        "after_boozer_lbfgs",
        "before_boozer_newton",
        "after_boozer_newton",
        "after_boozer_postprocess",
    }
)
DEFAULT_STAGE2_SEEDS_BY_PLASMA = {
    "wout_nfp22ginsburg_000_014417_iota15.nc": {
        "major_radius": 0.915,
        "toroidal_flux": 0.24,
        "length_weight": 0.0005,
        "cc_weight": 100.0,
        "cc_threshold": 0.05,
        "curvature_weight": 0.0001,
        "curvature_threshold": 40.0,
        "banana_surf_radius": 0.22,
        "order": 2,
    },
    "wout_nfp22ginsburg_000_002084_iota20.nc": {
        "major_radius": 0.975,
        "toroidal_flux": 0.24,
        "length_weight": 0.0005,
        "cc_weight": 100.0,
        "cc_threshold": 0.05,
        "curvature_weight": 0.0001,
        "curvature_threshold": 40.0,
        "banana_surf_radius": 0.22,
        "order": 2,
    },
}


@dataclass(frozen=True)
class SingleStageSearchPolicy:
    donor_class: str
    search_policy: str
    adaptive_failure_penalty_weight: float
    auto_initial_step_scale: float | None = None
    auto_initial_step_maxiter: int | None = None
    invalid_step_retry_budget: int = 0
    retry_step_shrink_factor: float = 0.5


maybe_initialize_distributed_jax()


@dataclass(frozen=True)
class SingleStageOuterOptimizerState:
    coil_dofs: object


jax.tree_util.register_dataclass(
    SingleStageOuterOptimizerState,
    data_fields=["coil_dofs"],
    meta_fields=[],
)


@dataclass(frozen=True)
class ScaledOuterPhaseOptimizerState:
    step_dofs: object
    anchor_dofs: object


@dataclass(frozen=True)
class SingleStageHostPostprocessResult:
    self_intersecting: bool
    self_intersection_check_available: bool


@dataclass(frozen=True)
class SingleStageFinalResultSnapshot:
    final_coil_dofs: object
    solved_surface_state: object
    final_metrics: dict
    final_distances: dict
    hardware_status: dict
    optimizer_result: dict
    optimizer_diagnostics: dict
    timings: dict
    artifact_policy: dict
    boozer_optimizer_method: object
    results_payload: dict
    field_error: object
    self_intersecting: bool
    self_intersection_check_available: bool


jax.tree_util.register_dataclass(
    ScaledOuterPhaseOptimizerState,
    data_fields=["step_dofs", "anchor_dofs"],
    meta_fields=[],
)


def format_compact_float(value):
    return f"{value:g}"


def _perf_counter_s():
    return float(time.perf_counter())


def _elapsed_s(start_s, end_s):
    return float(end_s - start_s)


def _record_timing(timings, key, start_s, end_s):
    timings[key] = _elapsed_s(start_s, end_s)
    return timings[key]


def begin_jax_profile_trace(profile_dir):
    """Start an optional JAX/XProf trace and return a stopper callable."""
    if not profile_dir:
        return None
    resolved_dir = os.path.abspath(profile_dir)
    os.makedirs(resolved_dir, exist_ok=True)
    trace_context = jax.profiler.trace(resolved_dir)
    trace_context.__enter__()

    def _stop_trace():
        trace_context.__exit__(None, None, None)

    return _stop_trace


_NONFINITE_OPTIMIZER_MESSAGE_FRAGMENT = "non-finite objective or gradient"


def _termination_message_indicates_invalid_optimizer_state(message):
    return isinstance(message, str) and (
        _NONFINITE_OPTIMIZER_MESSAGE_FRAGMENT in message.lower()
    )


def extract_optimizer_diagnostics(result, *, ran_optimizer, termination_message=None):
    if not ran_optimizer:
        return {
            "fun": None,
            "fun_finite": None,
            "jac_finite": None,
            "jac_inf_norm": None,
            "x_finite": None,
            "invalid_state": None,
        }
    if result is None:
        if _termination_message_indicates_invalid_optimizer_state(termination_message):
            return {
                "fun": None,
                "fun_finite": False,
                "jac_finite": False,
                "jac_inf_norm": None,
                "x_finite": None,
                "invalid_state": True,
            }
        return {
            "fun": None,
            "fun_finite": None,
            "jac_finite": None,
            "jac_inf_norm": None,
            "x_finite": None,
            "invalid_state": None,
        }

    fun_value = getattr(result, "fun", None)
    try:
        fun_host = float(fun_value)
    except (TypeError, ValueError):
        fun_host = None
    fun_finite = None if fun_host is None else bool(np.isfinite(fun_host))

    def _finite_array_diagnostics(value):
        if value is None:
            return None, None
        try:
            array = _single_stage_host_vector_array(value)
        except Exception:
            return None, None
        finite = bool(np.all(np.isfinite(array)))
        inf_norm = (
            None if (not finite or array.size == 0) else float(np.max(np.abs(array)))
        )
        return finite, inf_norm

    jac_finite, jac_inf_norm = _finite_array_diagnostics(getattr(result, "jac", None))
    x_finite, _ = _finite_array_diagnostics(getattr(result, "x", None))

    invalid_state = (
        (fun_finite is False) or (jac_finite is False) or (x_finite is False)
    )
    if _termination_message_indicates_invalid_optimizer_state(termination_message):
        if fun_finite is None:
            fun_finite = False
        if jac_finite is None:
            jac_finite = False
        invalid_state = True
    return {
        "fun": fun_host if fun_finite else None,
        "fun_finite": fun_finite,
        "jac_finite": jac_finite,
        "jac_inf_norm": jac_inf_norm,
        "x_finite": x_finite,
        "invalid_state": invalid_state,
    }


def summarize_optimizer_result_for_progress(result):
    """Return a compact, JSON-safe optimizer summary for progress checkpoints."""
    if result is None:
        return None
    termination_message = str(getattr(result, "message", ""))
    invalid_step_log = getattr(result, "invalid_step_log", None)
    return {
        "success": bool(getattr(result, "success", False)),
        "iterations": int(getattr(result, "nit", 0)),
        "status": _optional_int(getattr(result, "status", None)),
        "nfev": _optional_int(getattr(result, "nfev", None)),
        "njev": _optional_int(getattr(result, "njev", None)),
        "ls_status": _optional_int(getattr(result, "ls_status", None)),
        "rejected_step_count": _optional_int(
            getattr(result, "rejected_step_count", None)
        ),
        "invalid_step_log": [] if not invalid_step_log else list(invalid_step_log),
        "message": termination_message,
        "diagnostics": extract_optimizer_diagnostics(
            result,
            ran_optimizer=True,
            termination_message=termination_message,
        ),
    }


def _classify_nonfinite_scalar(value):
    if np.isnan(value):
        return "nan"
    if np.isposinf(value):
        return "+inf"
    if np.isneginf(value):
        return "-inf"
    return None


def _summarize_host_scalar(value):
    host_value = float(value)
    finite = bool(np.isfinite(host_value))
    return {
        "value": host_value if finite else None,
        "finite": finite,
        "classification": None if finite else _classify_nonfinite_scalar(host_value),
    }


def _single_stage_host_vector_array(value):
    return np.asarray(
        _single_stage_optimizer_dofs_array(value), dtype=np.float64
    ).reshape(-1)


def _optional_int(value):
    return None if value is None else int(value)


def _summarize_host_gradient(gradient):
    array = _single_stage_host_vector_array(gradient)
    finite_mask = np.isfinite(array)
    all_finite = bool(np.all(finite_mask))
    first_nonfinite_index = None
    first_nonfinite_classification = None
    if not all_finite:
        first_nonfinite_index = int(np.flatnonzero(~finite_mask)[0])
        first_nonfinite_classification = _classify_nonfinite_scalar(
            float(array[first_nonfinite_index])
        )
    return {
        "all_finite": all_finite,
        "inf_norm": None
        if (not all_finite or array.size == 0)
        else float(np.max(np.abs(array))),
        "size": int(array.size),
        "nonfinite_count": int(array.size - int(np.count_nonzero(finite_mask))),
        "first_nonfinite_index": first_nonfinite_index,
        "first_nonfinite_classification": first_nonfinite_classification,
    }


def _summarize_host_vector(vector):
    array = _single_stage_host_vector_array(vector)
    finite_mask = np.isfinite(array)
    all_finite = bool(np.all(finite_mask))
    first_nonfinite_index = None
    first_nonfinite_classification = None
    if not all_finite:
        first_nonfinite_index = int(np.flatnonzero(~finite_mask)[0])
        first_nonfinite_classification = _classify_nonfinite_scalar(
            float(array[first_nonfinite_index])
        )
    return {
        "values": array.tolist(),
        "all_finite": all_finite,
        "inf_norm": None
        if (not all_finite or array.size == 0)
        else float(np.max(np.abs(array))),
        "size": int(array.size),
        "nonfinite_count": int(array.size - int(np.count_nonzero(finite_mask))),
        "first_nonfinite_index": first_nonfinite_index,
        "first_nonfinite_classification": first_nonfinite_classification,
    }


def _optional_host_float(value):
    return None if value is None else host_float(value)


def _host_curve_max_curvature(curve):
    return float(np.max(host_array(curve.kappa(), dtype=np.float64)))


def _build_target_lane_value_and_grad_record(
    *,
    value,
    grad,
    mapped_dofs,
    scaled_dofs=None,
):
    mapped_array = np.asarray(host_array(mapped_dofs), dtype=np.float64).reshape(-1)
    record = {
        "mapped_coil_dofs": mapped_array.tolist(),
        "mapped_coil_dofs_inf_norm": None
        if mapped_array.size == 0
        else float(np.max(np.abs(mapped_array))),
        "mapped_coil_dofs_size": int(mapped_array.size),
        "value": _summarize_host_scalar(value),
        "grad": _summarize_host_gradient(grad),
    }
    if scaled_dofs is not None:
        scaled_array = np.asarray(host_array(scaled_dofs), dtype=np.float64).reshape(-1)
        record["scaled_dofs"] = scaled_array.tolist()
        record["scaled_dofs_inf_norm"] = (
            None if scaled_array.size == 0 else float(np.max(np.abs(scaled_array)))
        )
        record["scaled_dofs_size"] = int(scaled_array.size)
    return record


def _target_lane_record_all_finite(record):
    return bool(record["value"]["finite"] and record["grad"]["all_finite"])


def _resolve_first_nonfinite_target_lane_stage(stage_records):
    for stage_name, record in stage_records:
        if not _target_lane_record_all_finite(record):
            return stage_name
    return None


def _build_target_lane_invalid_state_event(
    *,
    phase,
    iteration,
    step_scale,
    line_search_failed,
    nonfinite_step,
    stalled_step,
    valid_curvature,
    trial_converged,
    ls_status,
):
    return {
        "phase": phase,
        "iteration": int(iteration),
        "step_scale": _summarize_host_scalar(step_scale),
        "line_search_failed": bool(line_search_failed),
        "nonfinite_step": bool(nonfinite_step),
        "stalled_step": bool(stalled_step),
        "valid_curvature": bool(valid_curvature),
        "trial_converged": bool(trial_converged),
        "ls_status": int(ls_status),
    }


def build_target_lane_invalid_state_failure_callback(events, *, phase):
    """Record rejected target-lane L-BFGS trial states for postmortem analysis."""

    def _record(
        iteration,
        trial_x,
        trial_f,
        trial_g,
        search_direction,
        step_vector,
        step_scale,
        line_search_failed,
        nonfinite_step,
        stalled_step,
        valid_curvature,
        trial_converged,
        ls_status,
    ):
        events.append(
            _build_target_lane_invalid_state_event(
                phase=phase,
                iteration=iteration,
                step_scale=step_scale,
                line_search_failed=line_search_failed,
                nonfinite_step=nonfinite_step,
                stalled_step=stalled_step,
                valid_curvature=valid_curvature,
                trial_converged=trial_converged,
                ls_status=ls_status,
            )
            | {
                "trial_value": _summarize_host_scalar(trial_f),
                "trial_x": _summarize_host_vector(trial_x),
                "trial_grad": _summarize_host_vector(trial_g),
                "search_direction": _summarize_host_vector(search_direction),
                "step_vector": _summarize_host_vector(step_vector),
            }
        )

    return _record


def target_lane_diagnostic_callbacks_enabled(args) -> bool:
    """Return whether detailed target-lane host callbacks are explicitly enabled."""

    return bool(
        getattr(args, "diagnostic_callbacks", False)
        or getattr(args, "record_target_lane_invalid_state_events", False)
    )


def record_target_lane_invalid_state_events_enabled(args) -> bool:
    """Return whether detailed target-lane callback payloads are enabled."""

    return target_lane_diagnostic_callbacks_enabled(args)


def resolve_target_lane_invalid_state_failure_callback(
    events,
    *,
    phase,
    use_target_lane: bool,
    args,
):
    """Return the rejected-step diagnostic callback only when explicitly enabled."""

    if (not use_target_lane) or (
        not record_target_lane_invalid_state_events_enabled(args)
    ):
        return None
    return build_target_lane_invalid_state_failure_callback(events, phase=phase)


def extend_target_lane_invalid_state_events_from_result(events, result, *, phase):
    """Append the structured invalid-step result log to the host retry event list."""

    invalid_step_log = getattr(result, "invalid_step_log", None)
    if not invalid_step_log:
        return
    events.extend(
        _build_target_lane_invalid_state_event(
            phase=phase,
            iteration=entry["iteration"],
            step_scale=entry["step_scale"],
            line_search_failed=entry["line_search_failed"],
            nonfinite_step=entry["nonfinite_step"],
            stalled_step=entry["stalled_step"],
            valid_curvature=entry["valid_curvature"],
            trial_converged=entry["trial_converged"],
            ls_status=entry["ls_status"],
        )
        for entry in invalid_step_log
    )


def _target_lane_signature_tree(value):
    """Convert host-side success-filter constants into a compact stable tree."""

    if isinstance(value, dict):
        return {
            key: _target_lane_signature_tree(value[key]) for key in sorted(value.keys())
        }
    if isinstance(value, (list, tuple)):
        return [_target_lane_signature_tree(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": type(value).__name__,
            **{
                field.name: _target_lane_signature_tree(getattr(value, field.name))
                for field in dataclass_fields(value)
            },
        }
    if isinstance(value, jax.Array):
        value = np.asarray(host_array(value))
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if isinstance(value, np.generic):
        return sanitize_json_payload(value)
    return value


def _target_lane_success_filter_cache_signature(payload) -> str:
    """Return one stable digest for the hardware success-filter configuration."""

    normalized_payload = _target_lane_signature_tree(payload)
    encoded_payload = json.dumps(
        normalized_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded_payload).hexdigest()


def _hostify_target_lane_constant_tree(value):
    """Materialize one constant tree on the host for target-lane cache keys."""
    return jax.tree_util.tree_map(
        lambda leaf: (
            host_array(leaf)
            if isinstance(leaf, jax.Array)
            else np.asarray(leaf)
            if isinstance(leaf, np.ndarray)
            else leaf
        ),
        value,
    )


def build_single_stage_problem_contract(
    *,
    plasma_surf_filename,
    file_loc,
    mpol,
    ntor,
    nphi,
    ntheta,
    vol_target,
    iota_target,
    stage2_bs_path,
    stage2_results_path,
    stage2_source,
    stage2_results,
    warm_start_run_dir,
    warm_start_state,
    banana_surf_radius,
    R0,
    s,
    order,
    CONSTRAINT_WEIGHT,
    constraint_method=None,
    alm_formulation=None,
    CC_DIST,
    CC_WEIGHT,
    CS_DIST,
    CS_WEIGHT,
    SS_DIST,
    SURF_DIST_WEIGHT,
    CURVATURE_WEIGHT,
    CURVATURE_THRESHOLD,
    LENGTH_WEIGHT,
    RES_WEIGHT,
    IOTAS_WEIGHT,
    length_target=None,
    banana_current_max_A=None,
    tf_current_limit_A=None,
    optimizer_backend_record,
    boozer_optimizer_backend_record,
    boozer_least_squares_algorithm_record,
    outer_optimizer_method,
    target_lane_sync_record,
    requested_experimental_target_lane_vg,
    use_target_lane_vg,
    target_lane_boozer_bfgs_tol_record,
    target_lane_boozer_bfgs_maxiter_record,
    target_lane_boozer_newton_tol_record,
    target_lane_boozer_newton_maxiter_record,
    single_stage_search_policy=None,
    effective_initial_phase_settings=None,
    args=None,
    MAXITER=None,
    write_restart_artifacts=False,
    write_full_artifacts=False,
):
    if args is None:
        raise ValueError(
            "args is required for build_single_stage_problem_contract — "
            "the runtime_contract section requires argparse fields"
        )
    constraint_method = (
        getattr(args, "constraint_method", "penalty")
        if constraint_method is None
        else constraint_method
    )
    if constraint_method != "alm":
        alm_formulation = None
    elif alm_formulation is None:
        alm_formulation = getattr(args, "alm_formulation", "weighted_sum")
    length_target = (
        COIL_LENGTH_TARGET_M if length_target is None else float(length_target)
    )
    banana_current_max_A = (
        BANANA_CURRENT_HARD_LIMIT_A
        if banana_current_max_A is None
        else float(banana_current_max_A)
    )
    tf_current_limit_A = (
        TF_CURRENT_HARD_LIMIT_A
        if tf_current_limit_A is None
        else float(tf_current_limit_A)
    )
    single_stage_search_policy, effective_initial_phase_settings = (
        resolve_single_stage_contract_policy_context(
            warm_start_state=warm_start_state,
            single_stage_search_policy=single_stage_search_policy,
            effective_initial_phase_settings=effective_initial_phase_settings,
            args=args,
        )
    )
    uses_jax_runtime_seed = getattr(args, "backend", None) == "jax"
    stage2_seed_biot_savart_path = os.path.abspath(stage2_bs_path)
    stage2_seed_jax_runtime_spec_path = None
    if uses_jax_runtime_seed:
        stage2_seed_biot_savart_path = None
        stage2_seed_jax_runtime_spec_path = os.path.abspath(stage2_results_path)
    return {
        "workflow": "single-stage-banana-optimization",
        "equilibrium": {
            "filename": plasma_surf_filename,
            "path": file_loc,
        },
        "resolution": {
            "mpol": int(mpol),
            "ntor": int(ntor),
            "nphi": int(nphi),
            "ntheta": int(ntheta),
        },
        "targets": {
            "volume": float(vol_target),
            "iota": float(iota_target),
        },
        "stage2_seed": {
            "source": stage2_source,
            "requested_source": getattr(args, "stage2_source", stage2_source),
            "biot_savart_path": stage2_seed_biot_savart_path,
            "jax_runtime_spec_path": stage2_seed_jax_runtime_spec_path,
            "results_path": os.path.abspath(stage2_results_path),
            "major_radius": float(R0),
            "toroidal_flux": float(s),
            "banana_surface_radius": float(stage2_results["banana_surf_radius"]),
            "order": int(order),
            "tf_current_limit_enforced": bool(
                getattr(args, "stage2_tf_current_limit_enforced", True)
            ),
            "hardware_seed_validation_enforced": bool(
                getattr(args, "stage2_seed_hardware_validation_enforced", True)
            ),
        },
        "warm_start": {
            "run_dir": None
            if warm_start_run_dir is None
            else os.path.abspath(warm_start_run_dir),
            "surface_path": None
            if warm_start_state is None
            else warm_start_state.get("surface_path"),
            "results_path": None
            if warm_start_state is None
            else warm_start_state.get("results_path"),
            "biot_savart_path": None
            if warm_start_state is None
            else warm_start_state.get("biot_savart_path"),
            "jax_runtime_spec_path": None
            if warm_start_state is None
            else warm_start_state.get("jax_runtime_spec_path"),
            "donor_class": single_stage_search_policy.donor_class,
            "search_policy": single_stage_search_policy.search_policy,
        },
        "objective_weights": {
            "constraint": float(CONSTRAINT_WEIGHT),
            "coil_coil_distance": float(CC_WEIGHT),
            "curve_surface_distance": float(CS_WEIGHT),
            "surface_vessel_distance": float(SURF_DIST_WEIGHT),
            "curvature": float(CURVATURE_WEIGHT),
            "non_qs": 1.0,
            "length": float(LENGTH_WEIGHT),
            "residual": float(RES_WEIGHT),
            "iota": float(IOTAS_WEIGHT),
        },
        "hardware_thresholds": {
            "coil_coil_distance": float(CC_DIST),
            "curve_surface_distance": float(CS_DIST),
            "coil_vessel_clearance": float(COIL_VESSEL_MIN_DIST_M),
            "surface_vessel_distance": float(SS_DIST),
            "curvature": float(CURVATURE_THRESHOLD),
            "coil_length": float(length_target),
            "banana_current": float(banana_current_max_A),
            "tf_current": float(tf_current_limit_A),
        },
        "runtime_contract": {
            "field_backend": args.backend,
            "optimizer_backend": optimizer_backend_record,
            "boozer_optimizer_backend": boozer_optimizer_backend_record,
            "boozer_least_squares_algorithm": boozer_least_squares_algorithm_record,
            "outer_optimizer_method": outer_optimizer_method,
            "constraint_method": constraint_method,
            "alm_formulation": (
                None if constraint_method != "alm" else alm_formulation
            ),
            "target_lane_accepted_step_sync": target_lane_sync_record,
            "experimental_target_lane_value_and_grad": bool(
                requested_experimental_target_lane_vg
            ),
            "target_lane_value_and_grad": bool(use_target_lane_vg),
            "max_iterations": int(MAXITER),
            "maxcor": int(args.maxcor),
            "outer_maxls": int(args.outer_maxls),
            "outer_ftol": args.outer_ftol,
            "target_lane_outer_initial_step_size": args.target_lane_outer_initial_step_size,
            "initial_step_scale": float(
                effective_initial_phase_settings["initial_step_scale"]
            ),
            "initial_step_maxiter": int(
                effective_initial_phase_settings["initial_step_maxiter"]
            ),
            "initial_step_auto_enabled": bool(
                effective_initial_phase_settings["auto_enabled"]
            ),
            "adaptive_failure_penalty_weight": float(
                single_stage_search_policy.adaptive_failure_penalty_weight
            ),
            "invalid_step_retry_budget": int(
                single_stage_search_policy.invalid_step_retry_budget
            ),
            "retry_step_shrink_factor": float(
                single_stage_search_policy.retry_step_shrink_factor
            ),
            "target_lane_boozer_bfgs_tol": target_lane_boozer_bfgs_tol_record,
            "target_lane_boozer_bfgs_maxiter": target_lane_boozer_bfgs_maxiter_record,
            "target_lane_boozer_newton_tol": target_lane_boozer_newton_tol_record,
            "target_lane_boozer_newton_maxiter": target_lane_boozer_newton_maxiter_record,
            "effective_banana_surface_radius": float(banana_surf_radius),
            "benchmark_mode": bool(args.benchmark_mode),
            "minimal_artifacts": bool(args.minimal_artifacts),
            "full_artifacts": bool(getattr(args, "full_artifacts", False)),
            "init_only": bool(args.init_only),
            "profile_target_lane_only": bool(args.profile_target_lane_only),
            "profile_target_lane_batch_size": int(args.profile_target_lane_batch_size),
            "diagnose_target_lane_gradient": bool(
                getattr(args, "diagnose_target_lane_gradient", False)
            ),
            "diagnose_target_lane_first_line_search": bool(
                getattr(args, "diagnose_target_lane_first_line_search", False)
            ),
            "diagnose_target_lane_scaled_phase1": bool(
                getattr(args, "diagnose_target_lane_scaled_phase1", False)
            ),
            "diagnostic_callbacks": bool(
                getattr(args, "diagnostic_callbacks", False)
            ),
            "record_target_lane_invalid_state_events": bool(
                getattr(args, "record_target_lane_invalid_state_events", False)
            ),
            "structured_invalid_step_log": True,
            "disable_target_lane_success_filter": bool(
                args.disable_target_lane_success_filter
            ),
            "alm_max_outer_iters": (
                None if constraint_method != "alm" else int(args.alm_max_outer_iters)
            ),
            "alm_penalty_init": (
                None if constraint_method != "alm" else float(args.alm_penalty_init)
            ),
            "alm_penalty_scale": (
                None if constraint_method != "alm" else float(args.alm_penalty_scale)
            ),
            "alm_penalty_max": (
                None if constraint_method != "alm" else float(args.alm_penalty_max)
            ),
            "alm_feas_tol": (
                None if constraint_method != "alm" else float(args.alm_feas_tol)
            ),
            "alm_stationarity_tol": (
                None if constraint_method != "alm" else float(args.alm_stationarity_tol)
            ),
            "alm_trust_radius_init": (
                None
                if constraint_method != "alm"
                else float(args.alm_trust_radius_init)
            ),
            "alm_trust_radius_min": (
                None if constraint_method != "alm" else float(args.alm_trust_radius_min)
            ),
            "alm_trust_radius_shrink": (
                None
                if constraint_method != "alm"
                else float(args.alm_trust_radius_shrink)
            ),
            "alm_trust_radius_grow": (
                None
                if constraint_method != "alm"
                else float(args.alm_trust_radius_grow)
            ),
            "alm_max_inner_attempts": (
                None if constraint_method != "alm" else int(args.alm_max_inner_attempts)
            ),
            "alm_max_subproblem_continuations": (
                None
                if constraint_method != "alm"
                else int(args.alm_max_subproblem_continuations)
            ),
            "alm_distance_smoothing": (
                None
                if constraint_method != "alm"
                else float(args.alm_distance_smoothing)
            ),
            "alm_curvature_smoothing": (
                None
                if constraint_method != "alm"
                else float(args.alm_curvature_smoothing)
            ),
            "alm_qs_threshold": (
                None if constraint_method != "alm" else args.alm_qs_threshold
            ),
            "alm_boozer_threshold": (
                None if constraint_method != "alm" else args.alm_boozer_threshold
            ),
            "alm_iota_penalty_threshold": (
                None if constraint_method != "alm" else args.alm_iota_penalty_threshold
            ),
            "alm_length_penalty_threshold": (
                None
                if constraint_method != "alm"
                else args.alm_length_penalty_threshold
            ),
            "write_restart_artifacts": bool(write_restart_artifacts),
            "write_full_artifacts": bool(write_full_artifacts),
        },
    }


def build_single_stage_results_envelope(
    *,
    output_root,
    plasma_surf_filename,
    file_loc,
    mpol,
    ntor,
    nphi,
    ntheta,
    vol_target,
    iota_target,
    stage2_bs_path,
    stage2_results_path,
    stage2_source,
    stage2_results,
    warm_start_run_dir,
    warm_start_state,
    banana_surf_radius,
    R0,
    s,
    order,
    CONSTRAINT_WEIGHT,
    constraint_method=None,
    alm_formulation=None,
    CC_DIST,
    CC_WEIGHT,
    CS_DIST,
    CS_WEIGHT,
    SS_DIST,
    SURF_DIST_WEIGHT,
    CURVATURE_WEIGHT,
    CURVATURE_THRESHOLD,
    LENGTH_WEIGHT,
    RES_WEIGHT,
    IOTAS_WEIGHT,
    length_target=None,
    banana_current_max_A=None,
    tf_current_limit_A=None,
    optimizer_backend_record,
    boozer_optimizer_backend_record,
    boozer_least_squares_algorithm_record,
    outer_optimizer_method,
    target_lane_sync_record,
    requested_experimental_target_lane_vg,
    use_target_lane_vg,
    target_lane_boozer_bfgs_tol_record,
    target_lane_boozer_bfgs_maxiter_record,
    target_lane_boozer_newton_tol_record,
    target_lane_boozer_newton_maxiter_record,
    single_stage_search_policy=None,
    effective_initial_phase_settings=None,
    args=None,
    MAXITER=None,
    write_restart_artifacts=False,
    write_full_artifacts=False,
):
    constraint_method = (
        getattr(args, "constraint_method", "penalty")
        if constraint_method is None
        else constraint_method
    )
    if constraint_method != "alm":
        alm_formulation = None
    elif alm_formulation is None:
        alm_formulation = getattr(args, "alm_formulation", "weighted_sum")
    length_target = (
        COIL_LENGTH_TARGET_M if length_target is None else float(length_target)
    )
    banana_current_max_A = (
        BANANA_CURRENT_HARD_LIMIT_A
        if banana_current_max_A is None
        else float(banana_current_max_A)
    )
    tf_current_limit_A = (
        TF_CURRENT_HARD_LIMIT_A
        if tf_current_limit_A is None
        else float(tf_current_limit_A)
    )
    required_files = ["results.json"]
    planned_files = ["results.json"]
    if getattr(args, "diagnose_target_lane_gradient", False):
        required_files.append("target_lane_gradient_diagnosis.json")
        planned_files.append("target_lane_gradient_diagnosis.json")
    if getattr(args, "diagnose_target_lane_first_line_search", False):
        required_files.append("target_lane_first_line_search_diagnosis.json")
        planned_files.append("target_lane_first_line_search_diagnosis.json")
    if getattr(args, "diagnose_target_lane_scaled_phase1", False):
        required_files.append("target_lane_scaled_phase1_diagnosis.json")
        planned_files.append("target_lane_scaled_phase1_diagnosis.json")
    if constraint_method == "alm":
        required_files.append("alm_state.partial.json")
        planned_files.append("alm_state.partial.json")
    if write_restart_artifacts:
        restart_files = single_stage_restart_artifact_filenames(args)
        required_files.extend(restart_files)
        planned_files.extend(restart_files)
    artifacts = build_artifact_manifest(
        output_root,
        required_files=tuple(required_files),
        planned_files=tuple(planned_files),
    )
    artifacts["policy"] = {
        "write_restart_artifacts": bool(write_restart_artifacts),
        "write_full_artifacts": bool(write_full_artifacts),
    }
    return {
        "schema_version": _SINGLE_STAGE_RESULTS_SCHEMA_VERSION,
        "provenance": build_runtime_provenance(
            title="Single-stage banana optimization",
            repo_root=REPO_ROOT,
            script_path=__file__,
            output_root=output_root,
            argv=sys.argv,
            jax_module=jax,
            jaxlib_version=jaxlib.__version__,
        ),
        "artifacts": artifacts,
        "problem_contract": build_single_stage_problem_contract(
            plasma_surf_filename=plasma_surf_filename,
            file_loc=file_loc,
            mpol=mpol,
            ntor=ntor,
            nphi=nphi,
            ntheta=ntheta,
            vol_target=vol_target,
            iota_target=iota_target,
            stage2_bs_path=stage2_bs_path,
            stage2_results_path=stage2_results_path,
            stage2_source=stage2_source,
            stage2_results=stage2_results,
            warm_start_run_dir=warm_start_run_dir,
            warm_start_state=warm_start_state,
            banana_surf_radius=banana_surf_radius,
            R0=R0,
            s=s,
            order=order,
            CONSTRAINT_WEIGHT=CONSTRAINT_WEIGHT,
            constraint_method=constraint_method,
            alm_formulation=alm_formulation,
            CC_DIST=CC_DIST,
            CC_WEIGHT=CC_WEIGHT,
            CS_DIST=CS_DIST,
            CS_WEIGHT=CS_WEIGHT,
            SS_DIST=SS_DIST,
            SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
            CURVATURE_WEIGHT=CURVATURE_WEIGHT,
            CURVATURE_THRESHOLD=CURVATURE_THRESHOLD,
            LENGTH_WEIGHT=LENGTH_WEIGHT,
            RES_WEIGHT=RES_WEIGHT,
            IOTAS_WEIGHT=IOTAS_WEIGHT,
            length_target=length_target,
            banana_current_max_A=banana_current_max_A,
            tf_current_limit_A=tf_current_limit_A,
            optimizer_backend_record=optimizer_backend_record,
            boozer_optimizer_backend_record=boozer_optimizer_backend_record,
            boozer_least_squares_algorithm_record=boozer_least_squares_algorithm_record,
            outer_optimizer_method=outer_optimizer_method,
            target_lane_sync_record=target_lane_sync_record,
            requested_experimental_target_lane_vg=requested_experimental_target_lane_vg,
            use_target_lane_vg=use_target_lane_vg,
            target_lane_boozer_bfgs_tol_record=target_lane_boozer_bfgs_tol_record,
            target_lane_boozer_bfgs_maxiter_record=target_lane_boozer_bfgs_maxiter_record,
            target_lane_boozer_newton_tol_record=target_lane_boozer_newton_tol_record,
            target_lane_boozer_newton_maxiter_record=target_lane_boozer_newton_maxiter_record,
            single_stage_search_policy=single_stage_search_policy,
            effective_initial_phase_settings=effective_initial_phase_settings,
            args=args,
            MAXITER=MAXITER,
            write_restart_artifacts=write_restart_artifacts,
            write_full_artifacts=write_full_artifacts,
        ),
    }


@contextmanager
def maybe_trace_single_stage_phase(label, *, enabled):
    """Annotate one phase in the optional JAX profiler trace."""
    if enabled:
        with jax.profiler.TraceAnnotation(label):
            yield
        return
    yield


def jax_solver_stage_callback_supported():
    """Return whether JAX solver stage callbacks are enabled for this process."""
    forced = os.environ.get("SIMSOPT_FORCE_JAX_SOLVER_STAGE_CALLBACK")
    if forced is not None:
        normalized = forced.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    requested_platforms = os.environ.get("JAX_PLATFORMS")
    if requested_platforms is not None:
        requested = {
            platform.strip().lower()
            for platform in requested_platforms.split(",")
            if platform.strip()
        }
        return "cpu" in requested
    try:
        return any(device.platform == "cpu" for device in jax.devices())
    except RuntimeError:
        return False


def _record_prefixed_stage_timings(timings, stage_marks, *, prefix, solve_start_s):
    if "before_boozer_lbfgs" in stage_marks and "after_boozer_lbfgs" in stage_marks:
        timings[f"{prefix}_lbfgs_s"] = _elapsed_s(
            stage_marks["before_boozer_lbfgs"],
            stage_marks["after_boozer_lbfgs"],
        )
    elif "after_boozer_lbfgs" in stage_marks:
        timings[f"{prefix}_lbfgs_s"] = _elapsed_s(
            solve_start_s, stage_marks["after_boozer_lbfgs"]
        )
    if "before_boozer_newton" in stage_marks and "after_boozer_newton" in stage_marks:
        timings[f"{prefix}_newton_s"] = _elapsed_s(
            stage_marks["before_boozer_newton"],
            stage_marks["after_boozer_newton"],
        )
    if (
        "after_boozer_solve" in stage_marks
        and "after_boozer_postprocess" in stage_marks
    ):
        timings[f"{prefix}_postprocess_s"] = _elapsed_s(
            stage_marks["after_boozer_solve"],
            stage_marks["after_boozer_postprocess"],
        )


def resolve_curvature_threshold(value: float) -> float:
    """Clamp single-stage curvature thresholds into the shared HBT [20, 40] band."""
    return min(
        max(float(value), CURVATURE_THRESHOLD_FLOOR),
        CURVATURE_THRESHOLD_CEILING,
    )


def evaluate_single_stage_hardware_constraints_pure(
    curve_curve_min_dist,
    cc_dist,
    curve_surface_min_dist,
    cs_dist,
    surface_vessel_min_dist,
    ss_dist,
    max_curvature,
    curvature_threshold,
):
    """Evaluate single-stage hardware constraints without leaving the runtime lane."""
    curve_curve_min_dist = _as_jax_float64(curve_curve_min_dist)
    cc_dist = _as_jax_float64(cc_dist)
    curve_surface_min_dist = _as_jax_float64(curve_surface_min_dist)
    cs_dist = _as_jax_float64(cs_dist)
    surface_vessel_min_dist = _as_jax_float64(surface_vessel_min_dist)
    ss_dist = _as_jax_float64(ss_dist)
    max_curvature = _as_jax_float64(max_curvature)
    curvature_threshold = _as_jax_float64(curvature_threshold)

    finite_flags = {
        "curve_curve_min_dist": jnp.isfinite(curve_curve_min_dist),
        "cc_dist": jnp.isfinite(cc_dist),
        "curve_surface_min_dist": jnp.isfinite(curve_surface_min_dist),
        "cs_dist": jnp.isfinite(cs_dist),
        "surface_vessel_min_dist": jnp.isfinite(surface_vessel_min_dist),
        "ss_dist": jnp.isfinite(ss_dist),
        "max_curvature": jnp.isfinite(max_curvature),
        "curvature_threshold": jnp.isfinite(curvature_threshold),
    }
    threshold_flags = {
        "curve_curve_min_dist": curve_curve_min_dist >= cc_dist,
        "curve_surface_min_dist": curve_surface_min_dist >= cs_dist,
        "surface_vessel_min_dist": surface_vessel_min_dist >= ss_dist,
        "max_curvature": max_curvature <= curvature_threshold,
    }
    success = jnp.all(
        jnp.asarray(
            tuple(finite_flags.values()) + tuple(threshold_flags.values()),
            dtype=bool,
        )
    )
    return {
        "success": success,
        "finite_flags": finite_flags,
        "threshold_flags": threshold_flags,
        "curve_curve_min_dist": curve_curve_min_dist,
        "cc_dist": cc_dist,
        "curve_surface_min_dist": curve_surface_min_dist,
        "cs_dist": cs_dist,
        "surface_vessel_min_dist": surface_vessel_min_dist,
        "ss_dist": ss_dist,
        "max_curvature": max_curvature,
        "curvature_threshold": curvature_threshold,
    }


def _hostify_single_stage_hardware_constraints(status):
    """Normalize pure single-stage hardware status at the reporting boundary."""
    host_status = {
        "success": host_bool(status["success"]),
        "curve_curve_min_dist": host_float(status["curve_curve_min_dist"]),
        "cc_dist": host_float(status["cc_dist"]),
        "curve_surface_min_dist": host_float(status["curve_surface_min_dist"]),
        "cs_dist": host_float(status["cs_dist"]),
        "surface_vessel_min_dist": host_float(status["surface_vessel_min_dist"]),
        "ss_dist": host_float(status["ss_dist"]),
        "max_curvature": host_float(status["max_curvature"]),
        "curvature_threshold": host_float(status["curvature_threshold"]),
    }
    finite_flags = {
        metric_name: host_bool(metric_ok)
        for metric_name, metric_ok in status["finite_flags"].items()
    }
    threshold_flags = {
        metric_name: host_bool(metric_ok)
        for metric_name, metric_ok in status["threshold_flags"].items()
    }
    violations = []
    for metric_name, label in (
        ("curve_curve_min_dist", "coil_coil_min_dist"),
        ("cc_dist", "cc_dist"),
        ("curve_surface_min_dist", "coil_surface_min_dist"),
        ("cs_dist", "cs_dist"),
        ("surface_vessel_min_dist", "surface_vessel_min_dist"),
        ("ss_dist", "ss_dist"),
        ("max_curvature", "max_curvature"),
        ("curvature_threshold", "curvature_threshold"),
    ):
        if not finite_flags[metric_name]:
            violations.append(f"{label} {host_status[metric_name]} is not finite")
    if (
        finite_flags["curve_curve_min_dist"]
        and finite_flags["cc_dist"]
        and (not threshold_flags["curve_curve_min_dist"])
    ):
        violations.append(
            "coil_coil_min_dist "
            f"{host_status['curve_curve_min_dist']:.6f} below threshold "
            f"{host_status['cc_dist']:.6f}"
        )
    if (
        finite_flags["curve_surface_min_dist"]
        and finite_flags["cs_dist"]
        and (not threshold_flags["curve_surface_min_dist"])
    ):
        violations.append(
            "coil_surface_min_dist "
            f"{host_status['curve_surface_min_dist']:.6f} below threshold "
            f"{host_status['cs_dist']:.6f}"
        )
    if (
        finite_flags["surface_vessel_min_dist"]
        and finite_flags["ss_dist"]
        and (not threshold_flags["surface_vessel_min_dist"])
    ):
        violations.append(
            "surface_vessel_min_dist "
            f"{host_status['surface_vessel_min_dist']:.6f} below threshold "
            f"{host_status['ss_dist']:.6f}"
        )
    if (
        finite_flags["max_curvature"]
        and finite_flags["curvature_threshold"]
        and (not threshold_flags["max_curvature"])
    ):
        violations.append(
            "max_curvature "
            f"{host_status['max_curvature']:.6f} exceeds threshold "
            f"{host_status['curvature_threshold']:.6f}"
        )
    host_status["violations"] = violations
    return host_status


def evaluate_single_stage_hardware_constraints(
    curve_curve_min_dist,
    cc_dist,
    curve_surface_min_dist,
    cs_dist,
    surface_vessel_min_dist,
    ss_dist,
    max_curvature,
    curvature_threshold,
):
    """Evaluate hard single-stage hardware constraints against realized geometry."""
    return _hostify_single_stage_hardware_constraints(
        evaluate_single_stage_hardware_constraints_pure(
            curve_curve_min_dist,
            cc_dist,
            curve_surface_min_dist,
            cs_dist,
            surface_vessel_min_dist,
            ss_dist,
            max_curvature,
            curvature_threshold,
        )
    )


def _can_evaluate_single_stage_hardware_status(objectives, diagnostics):
    return (
        objectives is not None
        and diagnostics is not None
        and "cc" in objectives
        and "cs" in objectives
        and "surf" in objectives
        and "banana_curve" in diagnostics
    )


def _evaluate_single_stage_hardware_status(objectives, diagnostics):
    curve_curve_min_dist = host_float(objectives["cc"].shortest_distance())
    curve_surface_min_dist = host_float(objectives["cs"].shortest_distance())
    surface_vessel_min_dist = host_float(objectives["surf"].shortest_distance())
    max_curvature = float(np.max(diagnostics["banana_curve"].kappa()))
    return evaluate_single_stage_hardware_constraints(
        curve_curve_min_dist,
        CC_DIST,
        curve_surface_min_dist,
        CS_DIST,
        surface_vessel_min_dist,
        SS_DIST,
        max_curvature,
        CURVATURE_THRESHOLD,
    )


_SINGLE_STAGE_WEIGHTED_REPORTING_FIELDS = (
    "final_non_qs",
    "final_boozer_residual",
    "final_iota_penalty",
    "final_length_penalty",
    "final_curve_curve_penalty",
    "final_curve_surface_penalty",
    "final_surface_vessel_penalty",
    "final_curvature_penalty",
)
_TRACEABLE_REPORTING_FLOAT_FIELDS = (
    "final_non_qs",
    "final_boozer_residual",
    "final_iota_penalty",
    "final_length_penalty",
    "final_curve_curve_penalty",
    "final_curve_surface_penalty",
    "final_surface_vessel_penalty",
    "final_curvature_penalty",
    "coil_length",
    "max_curvature",
    "banana_current_A",
    "field_error",
    "final_volume",
    "final_iota",
)
_TRACEABLE_REPORTING_DISTANCE_FIELDS = (
    "curve_curve_min_dist",
    "curve_surface_min_dist",
    "surface_vessel_min_dist",
)


def total_single_stage_objective_from_reporting_metrics(metrics):
    """Reconstruct the weighted single-stage objective from reporting terms."""
    return float(
        sum(
            float(metrics[field_name])
            for field_name in _SINGLE_STAGE_WEIGHTED_REPORTING_FIELDS
        )
    )


def _hostify_traceable_reporting_metrics(
    metrics,
    *,
    include_distance_metrics,
):
    """Normalize pure target-lane reporting metrics at the explicit host boundary."""
    host_metrics = {}
    if "solver_success" in metrics:
        host_metrics["solver_success"] = host_bool(metrics["solver_success"])
    has_G = metrics.get("has_G")
    if has_G is not None:
        host_has_G = host_bool(has_G)
        host_metrics["has_G"] = host_has_G
        host_metrics["final_G"] = (
            None if not host_has_G else host_float(metrics["final_G"])
        )
    elif "final_G" in metrics:
        final_G = metrics["final_G"]
        host_metrics["final_G"] = None if final_G is None else host_float(final_G)
    for metric_name in _TRACEABLE_REPORTING_FLOAT_FIELDS:
        if metric_name in metrics:
            host_metrics[metric_name] = host_float(metrics[metric_name])
    for metric_name in _TRACEABLE_REPORTING_DISTANCE_FIELDS:
        metric_value = metrics.get(metric_name)
        host_metrics[metric_name] = (
            None
            if (not include_distance_metrics or metric_value is None)
            else host_float(metric_value)
        )
    return host_metrics


def _hostify_traceable_value_and_grad(value_and_grad, x):
    """Evaluate the pure target-lane value/grad contract and host-normalize once."""
    value, grad = value_and_grad(_as_jax_float64(x))
    return (
        host_float(value),
        np.asarray(host_array(grad), dtype=np.float64).reshape(-1),
    )


def total_single_stage_objective_from_traceable_reporting_metrics(metrics):
    """Reconstruct the weighted single-stage objective on the runtime lane."""
    return jnp.sum(
        jnp.asarray(
            [
                metrics[field_name]
                for field_name in _SINGLE_STAGE_WEIGHTED_REPORTING_FIELDS
            ],
            dtype=jnp.float64,
        )
    )


def uses_per_accept_target_lane_sync(sync_policy: str) -> bool:
    """Return whether the target lane should sync/log every accepted step."""
    return sync_policy == "per-accept"


def resolve_effective_target_lane_accepted_step_sync(
    sync_policy: str, *, benchmark_mode: bool
) -> str:
    """Return the effective target-lane sync policy for this run."""
    if benchmark_mode:
        return "final-only"
    return sync_policy


def resolve_target_lane_accepted_step_callback(
    adapter,
    *,
    use_target_lane: bool,
    sync_policy: str,
):
    """Return the accepted-step callback contract for the outer optimizer."""
    if not use_target_lane:
        return adapter.callback
    if uses_per_accept_target_lane_sync(sync_policy):
        return adapter.observe_accepted_step
    return None


def resolve_target_lane_post_run_state_sync(
    adapter,
    *,
    use_target_lane: bool,
    accepted_step_callback,
    scaled_phase_step_scale: float | None = None,
    scaled_phase_anchor_dofs=None,
):
    """Return explicit accepted-state sync for target-lane runs."""
    if not use_target_lane:
        return None
    final_state_sync = (
        adapter.sync_accepted_step
        if accepted_step_callback is None
        else adapter.sync_accepted_step_state
    )

    def explicit_state_sync(result_x):
        final_state_sync(result_x)

    explicit_state_sync.simsopt_skip_failed_attempt_sync = True
    if scaled_phase_step_scale is None:
        return explicit_state_sync

    def sync(result_x):
        explicit_state_sync(
            resolve_scaled_outer_phase_final_dofs(
                scaled_phase_anchor_dofs,
                result_x,
                scaled_phase_step_scale,
                use_target_lane=True,
            )
        )

    sync.simsopt_skip_failed_attempt_sync = True
    return sync


def resolve_target_lane_accepted_step_sync_record(
    *,
    backend: str,
    optimizer_backend: str | None,
    maxiter: int,
    sync_policy: str,
):
    """Return the sync policy only when it can affect the target lane."""
    if backend == "jax" and optimizer_backend == "ondevice" and int(maxiter) > 0:
        return sync_policy
    return None


def should_write_single_stage_full_artifacts(
    benchmark_mode: bool,
    minimal_artifacts: bool,
    *,
    backend: str = "cpu",
    full_artifacts: bool = False,
) -> bool:
    """Return whether the run should emit heavy plotting/VTK artifacts."""
    if benchmark_mode or minimal_artifacts:
        return False
    return backend != "jax" or bool(full_artifacts)


def should_write_single_stage_restart_artifacts(benchmark_mode: bool) -> bool:
    """Return whether the run should emit final JSON artifacts for warm restarts."""
    return not benchmark_mode


def single_stage_restart_artifact_filenames(args):
    """Return restart artifacts required by this backend's warm-start contract."""
    if args.backend == "jax":
        return (_SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME,)
    return (
        "biot_savart_opt.json",
        "surf_opt.json",
        _SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME,
    )


def should_record_single_stage_outer_optimizer_progress(use_target_lane: bool) -> bool:
    """Return whether durable outer-optimizer progress should be written."""
    return bool(use_target_lane)


def use_experimental_target_lane_value_and_grad(
    *,
    backend: str,
    optimizer_backend: str | None,
    enabled: bool,
) -> bool:
    """Return whether the legacy compatibility flag was requested here."""
    return bool(enabled and backend == "jax" and optimizer_backend == "ondevice")


def use_target_lane_value_and_grad(
    *,
    backend: str,
    optimizer_backend: str | None,
) -> bool:
    """Return whether the production fused target-lane
    value/grad contract is active."""
    return bool(backend == "jax" and optimizer_backend == "ondevice")


def resolve_target_lane_value_and_grad_modes(
    *,
    backend: str,
    optimizer_backend: str | None,
    experimental_enabled: bool,
) -> tuple[bool, bool]:
    """Return (requested_legacy_flag, effective_target_lane_value_and_grad)."""
    return (
        use_experimental_target_lane_value_and_grad(
            backend=backend,
            optimizer_backend=optimizer_backend,
            enabled=experimental_enabled,
        ),
        use_target_lane_value_and_grad(
            backend=backend,
            optimizer_backend=optimizer_backend,
        ),
    )


def format_local_stage2_seed_dir(
    major_radius,
    toroidal_flux,
    length_weight,
    cc_weight,
    cc_threshold,
    curvature_weight,
    curvature_threshold,
    banana_surf_radius,
    order,
):
    return (
        f"R0={format_compact_float(major_radius)}"
        f"-s={format_compact_float(toroidal_flux)}"
        f"-LW={format_compact_float(length_weight)}"
        f"-CCW={format_compact_float(cc_weight)}"
        f"-CCT={format_compact_float(cc_threshold)}"
        f"-CW={format_compact_float(curvature_weight)}"
        f"-CT={format_compact_float(curvature_threshold)}"
        f"-SR={banana_surf_radius:0.3f}"
        f"-Order={order}"
    )


def format_database_stage2_seed_dir(
    major_radius,
    toroidal_flux,
    length_weight,
    cc_weight,
    curvature_weight,
    banana_surf_radius,
    order,
):
    return (
        f"MR={format_compact_float(major_radius)}"
        f"-TF={format_compact_float(toroidal_flux)}"
        f"-LW={format_compact_float(length_weight)}"
        f"-CCW={format_compact_float(cc_weight)}"
        f"-CW={format_compact_float(curvature_weight)}"
        f"-SR={format_compact_float(banana_surf_radius)}"
        f"-Order={order}"
    )


def build_stage2_bs_path(args):
    if args.stage2_bs_path:
        return args.stage2_bs_path

    if args.stage2_source == "database":
        seed_dir = format_database_stage2_seed_dir(
            args.stage2_seed_major_radius,
            args.stage2_seed_toroidal_flux,
            args.stage2_seed_length_weight,
            args.stage2_seed_cc_weight,
            args.stage2_seed_curvature_weight,
            args.stage2_seed_banana_surf_radius,
            args.stage2_seed_order,
        )
        return os.path.join(
            args.database_stage2_root,
            f"outputs-{args.plasma_surf_filename}",
            seed_dir,
            "biot_savart_opt.json",
        )

    seed_dir = format_local_stage2_seed_dir(
        args.stage2_seed_major_radius,
        args.stage2_seed_toroidal_flux,
        args.stage2_seed_length_weight,
        args.stage2_seed_cc_weight,
        args.stage2_seed_cc_threshold,
        args.stage2_seed_curvature_weight,
        args.stage2_seed_curvature_threshold,
        args.stage2_seed_banana_surf_radius,
        args.stage2_seed_order,
    )
    candidate = os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
        seed_dir,
        "biot_savart_opt.json",
    )
    if os.path.exists(candidate):
        return candidate

    # Fallback: legacy directory format without CCT/CT segments
    legacy_dir = (
        f"R0={format_compact_float(args.stage2_seed_major_radius)}"
        f"-s={format_compact_float(args.stage2_seed_toroidal_flux)}"
        f"-LW={format_compact_float(args.stage2_seed_length_weight)}"
        f"-CCW={format_compact_float(args.stage2_seed_cc_weight)}"
        f"-CW={format_compact_float(args.stage2_seed_curvature_weight)}"
        f"-SR={args.stage2_seed_banana_surf_radius:0.3f}"
        f"-Order={args.stage2_seed_order}"
    )
    legacy = os.path.join(
        args.local_stage2_root,
        f"outputs-{args.plasma_surf_filename}",
        legacy_dir,
        "biot_savart_opt.json",
    )
    if os.path.exists(legacy):
        print(
            f"Note: found legacy Stage 2 output at {legacy_dir}/ (missing CCT/CT segments)"
        )
        return legacy

    return candidate


def load_stage2_results(stage2_bs_path):
    stage2_results_path = os.path.join(os.path.dirname(stage2_bs_path), "results.json")
    with open(stage2_results_path, "r", encoding="utf-8") as infile:
        stage2_results = json.load(infile)
    stage2_results = upgrade_legacy_stage2_artifact_results(stage2_results)
    return stage2_results_path, stage2_results


def resolve_single_stage_warm_start_paths(run_dir):
    resolved_run_dir = os.path.abspath(run_dir)
    surface_path = os.path.join(resolved_run_dir, "surf_opt.json")
    results_path = os.path.join(resolved_run_dir, "results.json")
    missing_paths = [
        path for path in (surface_path, results_path) if not os.path.exists(path)
    ]
    if missing_paths:
        raise FileNotFoundError(
            "single-stage warm start run directory is missing required artifacts: "
            + ", ".join(missing_paths)
        )
    return surface_path, results_path


def resolve_single_stage_warm_start_biotsavart_path(run_dir):
    candidate = os.path.join(os.path.abspath(run_dir), "biot_savart_opt.json")
    if os.path.exists(candidate):
        return candidate
    return None


@dataclass(frozen=True)
class SerializedSurfaceState:
    surface_class: str
    dofs: np.ndarray
    mpol: int
    ntor: int
    nfp: int
    stellsym: bool
    quadpoints_phi: np.ndarray
    quadpoints_theta: np.ndarray


class DeferredSurfaceXYZTensorFourier:
    """Lazily materialize a host SurfaceXYZTensorFourier only on host-only paths."""

    deferred_surface_class = "SurfaceXYZTensorFourier"

    def __init__(
        self,
        *,
        mpol,
        ntor,
        nfp,
        stellsym,
        quadpoints_phi,
        quadpoints_theta,
        dofs,
    ):
        self.mpol = int(mpol)
        self.ntor = int(ntor)
        self.nfp = int(nfp)
        self.stellsym = bool(stellsym)
        self.quadpoints_phi = np.asarray(quadpoints_phi, dtype=np.float64)
        self.quadpoints_theta = np.asarray(quadpoints_theta, dtype=np.float64)
        self._dofs = _as_jax_float64(dofs)
        self._materialized_surface = None

    def _host_dofs(self):
        return np.asarray(host_array(self._dofs), dtype=np.float64)

    def _materialize_surface(self):
        if self._materialized_surface is None:
            self._materialized_surface = SurfaceXYZTensorFourier(
                mpol=self.mpol,
                ntor=self.ntor,
                nfp=self.nfp,
                stellsym=self.stellsym,
                quadpoints_phi=self.quadpoints_phi,
                quadpoints_theta=self.quadpoints_theta,
            )
        self._materialized_surface.set_dofs(self._host_dofs())
        return self._materialized_surface

    def get_dofs(self):
        return self._dofs

    def set_dofs(self, dofs):
        self._dofs = _as_jax_float64(dofs)
        if self._materialized_surface is not None:
            self._materialized_surface.set_dofs(self._host_dofs())

    @property
    def x(self):
        return self._dofs

    @x.setter
    def x(self, dofs):
        self.set_dofs(dofs)

    def __getattr__(self, name):
        return getattr(self._materialize_surface(), name)


class DeferredVolume:
    """Volume-label proxy that preserves the Boozer public label contract."""

    def __init__(self, surface):
        self.surface = surface

    def J(self):
        return self.surface.volume()

    def dJ_by_dsurfacecoefficients(self):
        return self.surface.dvolume_by_dcoeff()

    def d2J_by_dsurfacecoefficientsdsurfacecoefficients(self):
        return self.surface.d2volume_by_dcoeffdcoeff()


def _decode_gson_array(value, *, field_name):
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} is not a serialized array payload")
    if value.get("@module") != "numpy" or value.get("@class") != "array":
        raise ValueError(f"{field_name} is not a serialized numpy array")
    return np.asarray(value["data"], dtype=np.dtype(value["dtype"]))


def _resolve_simson_root_name(serialized_payload):
    graph = serialized_payload.get("graph")
    if isinstance(graph, dict) and graph.get("$type") == "ref":
        return graph["value"]
    if (
        isinstance(graph, list)
        and len(graph) == 1
        and isinstance(graph[0], dict)
        and graph[0].get("$type") == "ref"
    ):
        return graph[0]["value"]
    raise ValueError("serialized surface payload is missing a single root reference")


def load_serialized_surface_state(surface_path):
    with open(surface_path, "r", encoding="utf-8") as infile:
        serialized_payload = json.load(infile)
    if serialized_payload.get("@class") != "SIMSON":
        raise ValueError("surface payload is not a SIMSON serialization")

    simsopt_objs = serialized_payload.get("simsopt_objs")
    if not isinstance(simsopt_objs, dict):
        raise ValueError("surface payload is missing simsopt_objs")

    surface_name = _resolve_simson_root_name(serialized_payload)
    surface_payload = simsopt_objs.get(surface_name)
    if not isinstance(surface_payload, dict):
        raise ValueError("surface payload root is missing")

    surface_class = surface_payload.get("@class")
    if surface_class not in {"SurfaceRZFourier", "SurfaceXYZTensorFourier"}:
        raise ValueError(f"unsupported serialized surface class: {surface_class}")

    dofs_ref = surface_payload.get("dofs")
    if not isinstance(dofs_ref, dict) or dofs_ref.get("$type") != "ref":
        raise ValueError("surface payload is missing DOF reference")
    dofs_payload = simsopt_objs.get(dofs_ref["value"])
    if not isinstance(dofs_payload, dict):
        raise ValueError("surface DOF payload is missing")

    return SerializedSurfaceState(
        surface_class=surface_class,
        dofs=_decode_gson_array(dofs_payload["x"], field_name="dofs.x"),
        mpol=int(surface_payload["mpol"]),
        ntor=int(surface_payload["ntor"]),
        nfp=int(surface_payload["nfp"]),
        stellsym=bool(surface_payload["stellsym"]),
        quadpoints_phi=_decode_gson_array(
            surface_payload["quadpoints_phi"],
            field_name="quadpoints_phi",
        ),
        quadpoints_theta=_decode_gson_array(
            surface_payload["quadpoints_theta"],
            field_name="quadpoints_theta",
        ),
    )


def _reconstruct_live_surface_from_serialized_state(surface):
    surface_kwargs = {
        "mpol": surface.mpol,
        "ntor": surface.ntor,
        "nfp": surface.nfp,
        "stellsym": surface.stellsym,
        "quadpoints_phi": surface.quadpoints_phi,
        "quadpoints_theta": surface.quadpoints_theta,
    }
    if surface.surface_class == "SurfaceXYZTensorFourier":
        live_surface = SurfaceXYZTensorFourier(**surface_kwargs)
    elif surface.surface_class == "SurfaceRZFourier":
        live_surface = SurfaceRZFourier(**surface_kwargs)
    else:
        raise ValueError(
            f"unsupported serialized surface class: {surface.surface_class}"
        )
    live_surface.set_dofs(np.asarray(surface.dofs, dtype=float))
    return live_surface


def load_single_stage_warm_start_state(run_dir):
    surface_path, results_path = resolve_single_stage_warm_start_paths(run_dir)
    biot_savart_path = resolve_single_stage_warm_start_biotsavart_path(run_dir)
    surface = load_serialized_surface_state(surface_path)
    with open(results_path, "r", encoding="utf-8") as infile:
        results = json.load(infile)
    warm_start_iota = float(results["FINAL_IOTA"])
    warm_start_g = results.get("FINAL_G")
    if warm_start_g is not None:
        warm_start_g = float(warm_start_g)
    return {
        "surface": surface,
        "iota": warm_start_iota,
        "G": warm_start_g,
        "surface_path": surface_path,
        "results_path": results_path,
        "biot_savart_path": biot_savart_path,
    }


def load_single_stage_jax_warm_start_state(run_dir, *, runtime_spec_path=None):
    resolved_run_dir = os.path.abspath(run_dir)
    resolved_runtime_spec_path = resolve_single_stage_jax_runtime_spec_path(
        runtime_spec_path if runtime_spec_path is not None else resolved_run_dir
    )
    if not os.path.exists(resolved_runtime_spec_path):
        raise FileNotFoundError(
            "JAX warm start requires "
            f"{_SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME}; run seed conversion first: "
            f"{resolved_runtime_spec_path}"
        )
    results_path = os.path.join(resolved_run_dir, "results.json")
    return {
        "surface": None,
        "iota": None,
        "G": None,
        "surface_path": None,
        "results_path": results_path if os.path.exists(results_path) else None,
        "biot_savart_path": None,
        "jax_runtime_spec_path": resolved_runtime_spec_path,
    }


def resolve_single_stage_jax_runtime_spec_path(path_or_run_dir):
    resolved = os.path.abspath(path_or_run_dir)
    if os.path.isdir(resolved):
        return os.path.join(resolved, _SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME)
    return resolved


def _single_stage_jax_spec_array_payload(values):
    array = np.asarray(host_array(values), dtype=np.float64)
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "data": array.tolist(),
    }


def _single_stage_jax_spec_array_from_payload(payload, *, field_name):
    if payload.get("dtype") != "float64":
        raise ValueError(f"{field_name} must be a float64 runtime-spec array")
    return np.asarray(payload["data"], dtype=np.float64).reshape(payload["shape"])


_SINGLE_STAGE_JAX_SPEC_DATACLASSES = {
    cls.__name__: cls
    for cls in (
        jax_specs.CoilDofExtractionSpec,
        jax_specs.CoilGroupSpec,
        jax_specs.CoilSetDofExtractionSpec,
        jax_specs.CoilSpec,
        jax_specs.CoilSymmetrySpec,
        jax_specs.CurrentValueSpec,
        jax_specs.CurveCWSFourierRZSpec,
        jax_specs.CurveFilamentSpec,
        jax_specs.CurveHelicalSpec,
        jax_specs.CurvePerturbedSpec,
        jax_specs.CurvePlanarFourierSpec,
        jax_specs.CurveRZFourierSpec,
        jax_specs.CurveXYZFourierSpec,
        jax_specs.FrameRotationSpec,
        jax_specs.GroupedCoilSetSpec,
        jax_specs.OptimizableDofMapSpec,
        jax_specs.SurfaceRZFourierSpec,
        jax_specs.SurfaceXYZFourierSpec,
        jax_specs.SurfaceXYZTensorFourierSpec,
        jax_specs.ZeroRotationSpec,
    )
}


def _single_stage_jax_spec_tree_payload(value):
    if is_dataclass(value):
        return {
            "kind": "dataclass",
            "class": type(value).__name__,
            "fields": {
                field.name: _single_stage_jax_spec_tree_payload(
                    getattr(value, field.name)
                )
                for field in dataclass_fields(value)
            },
        }
    if isinstance(value, jax.Array):
        array = np.asarray(host_array(value))
        return {
            "kind": "array",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.tolist(),
        }
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        return {
            "kind": "array",
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "data": array.tolist(),
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_single_stage_jax_spec_tree_payload(item) for item in value],
        }
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_single_stage_jax_spec_tree_payload(item) for item in value],
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return {"kind": "scalar", "value": value}
    if isinstance(value, np.generic):
        return {"kind": "scalar", "value": value.item()}
    raise TypeError(
        "JAX runtime seed specs can only serialize immutable spec dataclasses, "
        f"arrays, tuples, lists, and scalars; got {type(value).__name__}."
    )


def _single_stage_jax_spec_tree_from_payload(payload, *, field_name):
    kind = payload["kind"]
    if kind == "array":
        return jax.device_put(
            np.asarray(payload["data"], dtype=np.dtype(payload["dtype"])).reshape(
                payload["shape"]
            )
        )
    if kind == "tuple":
        return tuple(
            _single_stage_jax_spec_tree_from_payload(
                item,
                field_name=f"{field_name}[]",
            )
            for item in payload["items"]
        )
    if kind == "list":
        return [
            _single_stage_jax_spec_tree_from_payload(
                item,
                field_name=f"{field_name}[]",
            )
            for item in payload["items"]
        ]
    if kind == "scalar":
        return payload["value"]
    if kind == "dataclass":
        class_name = payload["class"]
        spec_cls = _SINGLE_STAGE_JAX_SPEC_DATACLASSES.get(class_name)
        if spec_cls is None:
            raise ValueError(
                f"{field_name} uses unsupported runtime-spec class {class_name!r}"
            )
        return spec_cls(
            **{
                name: _single_stage_jax_spec_tree_from_payload(
                    value,
                    field_name=f"{field_name}.{name}",
                )
                for name, value in payload["fields"].items()
            }
        )
    raise ValueError(f"{field_name} has unsupported runtime-spec payload kind {kind!r}")


def _single_stage_hardware_constants_payload():
    return {
        "coil_coil_min_dist_m": float(COIL_COIL_MIN_DIST_M),
        "coil_length_target_m": float(COIL_LENGTH_TARGET_M),
        "coil_plasma_min_dist_m": float(COIL_PLASMA_MIN_DIST_M),
        "coil_vessel_min_dist_m": float(COIL_VESSEL_MIN_DIST_M),
        "plasma_vessel_min_dist_m": float(PLASMA_VESSEL_MIN_DIST_M),
        "banana_current_hard_limit_A": float(BANANA_CURRENT_HARD_LIMIT_A),
        "tf_current_hard_limit_A": float(TF_CURRENT_HARD_LIMIT_A),
    }


def _single_stage_hardware_constants_tuple(payload):
    return tuple((str(name), float(value)) for name, value in payload.items())


def build_single_stage_runtime_stage2_seed_payload(
    stage2_results,
    *,
    banana_surf_radius,
):
    return {
        "major_radius": float(stage2_results["MAJOR_RADIUS"]),
        "toroidal_flux": float(stage2_results["TOROIDAL_FLUX"]),
        "order": int(stage2_results["order"]),
        "banana_surf_radius": float(banana_surf_radius),
    }


def stage2_results_from_single_stage_runtime_seed_payload(stage2_seed_payload):
    return {
        "MAJOR_RADIUS": float(stage2_seed_payload["major_radius"]),
        "TOROIDAL_FLUX": float(stage2_seed_payload["toroidal_flux"]),
        "order": int(stage2_seed_payload["order"]),
        "banana_surf_radius": float(stage2_seed_payload["banana_surf_radius"]),
    }


def build_single_stage_jax_runtime_seed_spec_payload(
    surface,
    *,
    iota,
    G,
    mpol,
    ntor,
    quadpoints_phi,
    quadpoints_theta,
    coil_dof_extraction_spec,
    coil_dofs,
    num_tf_coils,
    banana_curve_index,
    tf_current_A,
    banana_current_A,
    stage2_seed,
    surface_dofs=None,
):
    """Build the durable JAX startup spec payload for a canonical seed surface."""
    if G is None:
        raise ValueError(
            "JAX runtime seed spec requires FINAL_G in the donor results; "
            "run seed conversion first."
        )
    target_surface_dofs = (
        np.asarray(host_array(surface_dofs), dtype=np.float64)
        if surface_dofs is not None
        else project_surface_dofs_to_resolution(
            surface,
            mpol=mpol,
            ntor=ntor,
            quadpoints_phi=quadpoints_phi,
            quadpoints_theta=quadpoints_theta,
        )
    )
    nfp = int(surface.nfp)
    stellsym = bool(surface.stellsym)
    coil_dofs_array = np.asarray(host_array(coil_dofs), dtype=np.float64)
    coil_set_spec = coil_set_spec_from_dof_extraction_spec(
        coil_dof_extraction_spec,
        coil_dofs_array,
    )
    return {
        "schema": _SINGLE_STAGE_JAX_RUNTIME_SPEC_SCHEMA,
        "schema_version": _SINGLE_STAGE_JAX_RUNTIME_SPEC_VERSION,
        "surface": {
            "surface_class": "SurfaceXYZTensorFourier",
            "dofs": _single_stage_jax_spec_array_payload(target_surface_dofs),
            "mpol": int(mpol),
            "ntor": int(ntor),
            "nfp": nfp,
            "stellsym": stellsym,
            "quadpoints_phi": _single_stage_jax_spec_array_payload(quadpoints_phi),
            "quadpoints_theta": _single_stage_jax_spec_array_payload(quadpoints_theta),
        },
        "field": {
            "coil_dof_extraction": _single_stage_jax_spec_tree_payload(
                coil_dof_extraction_spec
            ),
            "coil_set": _single_stage_jax_spec_tree_payload(coil_set_spec),
            "coil_dofs": _single_stage_jax_spec_array_payload(coil_dofs_array),
            "num_tf_coils": int(num_tf_coils),
            "banana_curve_index": int(banana_curve_index),
            "tf_current_A": float(tf_current_A),
            "banana_current_A": float(banana_current_A),
        },
        "boozer_init": {
            "iota": float(iota),
            "G": float(G),
        },
        "quadrature": {
            "nphi": int(np.asarray(quadpoints_phi).size),
            "ntheta": int(np.asarray(quadpoints_theta).size),
        },
        "stage2_seed": dict(stage2_seed),
        "target_labels": list(SINGLE_STAGE_THRESHOLDED_PHYSICS_CONSTRAINT_NAMES),
        "hardware_constants": _single_stage_hardware_constants_payload(),
        "self_intersection_mode": _SINGLE_STAGE_JAX_SELF_INTERSECTION_MODE,
    }


def write_single_stage_jax_runtime_seed_spec(path_or_run_dir, **kwargs):
    path = resolve_single_stage_jax_runtime_spec_path(path_or_run_dir)
    write_json_file(
        path,
        build_single_stage_jax_runtime_seed_spec_payload(**kwargs),
    )
    return path


def make_single_stage_half_period_quadpoints(*, nphi, ntheta, nfp):
    quadpoints_phi, quadpoints_theta = surface_module.Surface.get_quadpoints(
        nphi,
        ntheta,
        nfp=nfp,
        range=surface_module.Surface.RANGE_HALF_PERIOD,
    )
    return (
        np.asarray(quadpoints_phi, dtype=np.float64),
        np.asarray(quadpoints_theta, dtype=np.float64),
    )


def resolve_single_stage_runtime_seed_G(warm_start_G, tf_coils):
    if warm_start_G is not None:
        return float(warm_start_G)
    current_sum = sum(abs(coil.current.get_value()) for coil in tf_coils)
    return float(2.0 * np.pi * current_sum * (4 * np.pi * 10 ** (-7) / (2 * np.pi)))


def compile_single_stage_jax_runtime_seed_spec(
    run_dir,
    *,
    mpol,
    ntor,
    nphi,
    ntheta,
    num_tf_coils,
    output_path_or_run_dir=None,
):
    warm_start_state = load_single_stage_warm_start_state(run_dir)
    quadpoints_phi, quadpoints_theta = make_single_stage_half_period_quadpoints(
        nphi=nphi,
        ntheta=ntheta,
        nfp=warm_start_state["surface"].nfp,
    )
    biot_savart_path = warm_start_state["biot_savart_path"]
    if biot_savart_path is None:
        raise FileNotFoundError(
            "JAX runtime seed conversion requires biot_savart_opt.json; "
            "run the donor with restart artifacts first."
        )
    from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

    donor_bs = load(biot_savart_path)
    donor_bs_jax = BiotSavartJAX(donor_bs.coils)
    donor_results_path = warm_start_state["results_path"]
    with open(donor_results_path, "r", encoding="utf-8") as infile:
        donor_results = json.load(infile)
    tf_coils = donor_bs.coils[:num_tf_coils]
    banana_coils = donor_bs.coils[num_tf_coils:]
    tf_current_A = resolve_loaded_tf_current_A(
        donor_results.get("TF_CURRENT_A"),
        tf_coils,
        enforce_limit=False,
    )
    banana_surf_radius = resolve_single_stage_banana_surface_radius(
        types.SimpleNamespace(banana_surf_radius=None),
        donor_results,
    )
    banana_current_A = float(banana_coils[0].current.get_value())
    runtime_spec_destination = (
        output_path_or_run_dir if output_path_or_run_dir is not None else run_dir
    )
    return write_single_stage_jax_runtime_seed_spec(
        runtime_spec_destination,
        surface=warm_start_state["surface"],
        iota=warm_start_state["iota"],
        G=resolve_single_stage_runtime_seed_G(warm_start_state["G"], tf_coils),
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        coil_dof_extraction_spec=donor_bs_jax.coil_dof_extraction_spec(),
        coil_dofs=donor_bs_jax.x.copy(),
        num_tf_coils=num_tf_coils,
        banana_curve_index=int(num_tf_coils),
        tf_current_A=tf_current_A,
        banana_current_A=banana_current_A,
        stage2_seed=build_single_stage_runtime_stage2_seed_payload(
            donor_results,
            banana_surf_radius=banana_surf_radius,
        ),
    )


def compile_requested_single_stage_jax_runtime_seed_spec(args):
    if args.warm_start_run_dir is None:
        raise ValueError(
            "--compile-jax-runtime-seed-spec requires --warm-start-run-dir"
        )
    return compile_single_stage_jax_runtime_seed_spec(
        args.warm_start_run_dir,
        mpol=args.mpol,
        ntor=args.ntor,
        nphi=args.nphi,
        ntheta=args.ntheta,
        num_tf_coils=args.num_tf_coils,
        output_path_or_run_dir=args.jax_runtime_seed_spec,
    )


def load_single_stage_jax_runtime_seed_startup_state(args, *, mpol, ntor, nphi, ntheta):
    runtime_spec_source = (
        args.jax_runtime_seed_spec
        if args.jax_runtime_seed_spec is not None
        else args.warm_start_run_dir
    )
    if runtime_spec_source is None:
        raise FileNotFoundError(
            "JAX startup requires an immutable runtime seed spec; "
            "run seed conversion first."
        )
    runtime_spec_state = load_single_stage_jax_runtime_seed_spec(
        runtime_spec_source,
        mpol=mpol,
        ntor=ntor,
        nphi=nphi,
        ntheta=ntheta,
    )
    return {
        "stage2_bs_path": resolve_single_stage_jax_runtime_spec_path(
            runtime_spec_source
        ),
        "stage2_results_path": runtime_spec_state["path"],
        "stage2_results": stage2_results_from_single_stage_runtime_seed_payload(
            runtime_spec_state["stage2_seed"]
        ),
        "runtime_spec_state": runtime_spec_state,
    }


def _require_matching_single_stage_jax_runtime_surface(
    surface_payload,
    *,
    mpol,
    ntor,
    quadpoints_phi=None,
    quadpoints_theta=None,
    nfp=None,
    stellsym=None,
):
    if surface_payload["surface_class"] != "SurfaceXYZTensorFourier":
        raise ValueError(
            "JAX warm start requires a canonical SurfaceXYZTensorFourier runtime "
            "spec; run seed conversion first."
        )
    if int(surface_payload["mpol"]) != int(mpol) or int(surface_payload["ntor"]) != int(
        ntor
    ):
        raise ValueError(
            "JAX warm-start runtime spec resolution does not match this run; "
            "run seed conversion first."
        )
    if nfp is not None and int(surface_payload["nfp"]) != int(nfp):
        raise ValueError(
            "JAX warm-start runtime spec nfp does not match this run; "
            "run seed conversion first."
        )
    if stellsym is not None and bool(surface_payload["stellsym"]) != bool(stellsym):
        raise ValueError(
            "JAX warm-start runtime spec stellsym does not match this run; "
            "run seed conversion first."
        )
    spec_phi = _single_stage_jax_spec_array_from_payload(
        surface_payload["quadpoints_phi"],
        field_name="surface.quadpoints_phi",
    )
    spec_theta = _single_stage_jax_spec_array_from_payload(
        surface_payload["quadpoints_theta"],
        field_name="surface.quadpoints_theta",
    )
    if quadpoints_phi is not None and not np.array_equal(
        spec_phi,
        np.asarray(quadpoints_phi, dtype=np.float64),
    ):
        raise ValueError(
            "JAX warm-start runtime spec phi quadrature does not match this run; "
            "run seed conversion first."
        )
    if quadpoints_theta is not None and not np.array_equal(
        spec_theta,
        np.asarray(quadpoints_theta, dtype=np.float64),
    ):
        raise ValueError(
            "JAX warm-start runtime spec theta quadrature does not match this run; "
            "run seed conversion first."
        )
    return spec_phi, spec_theta


def load_single_stage_jax_runtime_seed_spec(
    path_or_run_dir,
    *,
    mpol,
    ntor,
    quadpoints_phi=None,
    quadpoints_theta=None,
    nphi=None,
    ntheta=None,
    nfp=None,
    stellsym=None,
):
    path = resolve_single_stage_jax_runtime_spec_path(path_or_run_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "JAX warm start requires "
            f"{_SINGLE_STAGE_JAX_RUNTIME_SPEC_FILENAME}; run seed conversion first: "
            f"{path}"
        )
    with open(path, "r", encoding="utf-8") as infile:
        payload = json.load(infile)
    if payload["schema"] != _SINGLE_STAGE_JAX_RUNTIME_SPEC_SCHEMA:
        raise ValueError("JAX warm-start runtime spec has the wrong schema")
    if int(payload["schema_version"]) != _SINGLE_STAGE_JAX_RUNTIME_SPEC_VERSION:
        raise ValueError("JAX warm-start runtime spec has an unsupported schema version")
    if payload["self_intersection_mode"] != _SINGLE_STAGE_JAX_SELF_INTERSECTION_MODE:
        raise ValueError(
            "JAX warm-start runtime spec must use supported-surface JAX "
            "self-intersection; run seed conversion first."
        )
    surface_payload = payload["surface"]
    spec_phi, spec_theta = _require_matching_single_stage_jax_runtime_surface(
        surface_payload,
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
        nfp=nfp,
        stellsym=stellsym,
    )
    if nphi is not None and int(payload["quadrature"]["nphi"]) != int(nphi):
        raise ValueError(
            "JAX warm-start runtime spec nphi does not match this run; "
            "run seed conversion first."
        )
    if ntheta is not None and int(payload["quadrature"]["ntheta"]) != int(ntheta):
        raise ValueError(
            "JAX warm-start runtime spec ntheta does not match this run; "
            "run seed conversion first."
        )
    surface_dofs = _single_stage_jax_spec_array_from_payload(
        surface_payload["dofs"],
        field_name="surface.dofs",
    )
    surface_spec = make_surface_xyz_tensor_fourier_spec(
        dofs=surface_dofs,
        quadpoints_phi=spec_phi,
        quadpoints_theta=spec_theta,
        nfp=int(surface_payload["nfp"]),
        stellsym=bool(surface_payload["stellsym"]),
        mpol=int(surface_payload["mpol"]),
        ntor=int(surface_payload["ntor"]),
    )
    field_payload = payload["field"]
    coil_dof_extraction_spec = _single_stage_jax_spec_tree_from_payload(
        field_payload["coil_dof_extraction"],
        field_name="field.coil_dof_extraction",
    )
    coil_set_spec = _single_stage_jax_spec_tree_from_payload(
        field_payload["coil_set"],
        field_name="field.coil_set",
    )
    coil_dofs = _single_stage_jax_spec_array_from_payload(
        field_payload["coil_dofs"],
        field_name="field.coil_dofs",
    )
    seed_spec = make_single_stage_seed_spec(
        surface=surface_spec,
        coil_set=coil_set_spec,
        coil_dof_extraction=coil_dof_extraction_spec,
        coil_dofs=coil_dofs,
        boozer_iota=float(payload["boozer_init"]["iota"]),
        boozer_G=float(payload["boozer_init"]["G"]),
        target_labels=payload["target_labels"],
        hardware_constants=_single_stage_hardware_constants_tuple(
            payload["hardware_constants"]
        ),
        self_intersection_mode=payload["self_intersection_mode"],
        schema_version=int(payload["schema_version"]),
        num_tf_coils=int(field_payload["num_tf_coils"]),
        banana_curve_index=int(field_payload["banana_curve_index"]),
        tf_current_A=float(field_payload["tf_current_A"]),
        banana_current_A=float(field_payload["banana_current_A"]),
    )
    runtime_spec = make_single_stage_runtime_spec(
        seed=seed_spec,
        mpol=int(surface_payload["mpol"]),
        ntor=int(surface_payload["ntor"]),
        nfp=int(surface_payload["nfp"]),
        nphi=int(payload["quadrature"]["nphi"]),
        ntheta=int(payload["quadrature"]["ntheta"]),
    )
    return {
        "path": path,
        "payload": payload,
        "runtime_spec": runtime_spec,
        "surface_spec": surface_spec,
        "surface_dofs": runtime_spec.seed.surface.dofs,
        "coil_dof_extraction_spec": runtime_spec.seed.coil_dof_extraction,
        "coil_dofs": runtime_spec.seed.coil_dofs,
        "coil_set_spec": runtime_spec.seed.coil_set,
        "stage2_seed": dict(payload["stage2_seed"]),
        "iota": runtime_spec.seed.boozer_iota[0],
        "G": runtime_spec.seed.boozer_G[0],
    }


def resolve_jax_warm_start_surface_dofs_from_spec(path_or_run_dir, **kwargs):
    return load_single_stage_jax_runtime_seed_spec(path_or_run_dir, **kwargs)[
        "surface_dofs"
    ]


def build_single_stage_surface_from_jax_runtime_spec(runtime_spec):
    surface_spec = runtime_spec.seed.surface
    return DeferredSurfaceXYZTensorFourier(
        mpol=surface_spec.mpol,
        ntor=surface_spec.ntor,
        nfp=surface_spec.nfp,
        stellsym=surface_spec.stellsym,
        quadpoints_phi=host_array(surface_spec.quadpoints_phi, dtype=np.float64),
        quadpoints_theta=host_array(surface_spec.quadpoints_theta, dtype=np.float64),
        dofs=surface_spec.dofs,
    )


def resolve_single_stage_startup_seed_contract(args, *, warm_start_state):
    donor_biot_savart_path = (
        None if warm_start_state is None else warm_start_state.get("biot_savart_path")
    )
    if donor_biot_savart_path is not None and not getattr(
        args,
        "stage2_bs_path_explicit",
        False,
    ):
        return {
            "stage2_bs_path": donor_biot_savart_path,
            "stage2_source": "warm_start_donor",
            "tf_current_limit_enforced": False,
            "seed_hardware_validation_enforced": False,
        }

    stage2_bs_path = build_stage2_bs_path(args)
    if getattr(args, "stage2_bs_path_explicit", False):
        return {
            "stage2_bs_path": stage2_bs_path,
            "stage2_source": "explicit_path",
            "tf_current_limit_enforced": False,
            "seed_hardware_validation_enforced": False,
        }

    return {
        "stage2_bs_path": stage2_bs_path,
        "stage2_source": args.stage2_source,
        "tf_current_limit_enforced": True,
        "seed_hardware_validation_enforced": True,
    }


def classify_single_stage_donor(
    warm_start_state,
    *,
    explicit_surface_warm_start,
):
    """Classify the startup donor quality for continuation policy selection."""
    if warm_start_state is None:
        return "stage2_seed_only"
    surface = warm_start_state["surface"]
    if isinstance(surface, SerializedSurfaceState):
        return "serialized_surface_state"
    if explicit_surface_warm_start and isinstance(
        surface, (SurfaceRZFourier, SurfaceXYZTensorFourier)
    ):
        return "live_supported_surface"
    if explicit_surface_warm_start:
        return "projected_supported_surface"
    return "legacy_surface_object"


def resolve_single_stage_search_policy(
    warm_start_state,
    *,
    explicit_surface_warm_start,
):
    """Resolve donor-aware continuation policy for the outer single-stage loop."""
    donor_class = classify_single_stage_donor(
        warm_start_state,
        explicit_surface_warm_start=explicit_surface_warm_start,
    )
    if donor_class == "serialized_surface_state":
        return SingleStageSearchPolicy(
            donor_class=donor_class,
            search_policy=_SINGLE_STAGE_SEARCH_POLICY_PRESERVE_FIRST,
            adaptive_failure_penalty_weight=1.0,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.35,
        )
    if donor_class in {"stage2_seed_only", "live_supported_surface"}:
        return SingleStageSearchPolicy(
            donor_class=donor_class,
            search_policy=_SINGLE_STAGE_SEARCH_POLICY_REPAIR_FIRST,
            adaptive_failure_penalty_weight=1.5,
            auto_initial_step_scale=0.25,
            auto_initial_step_maxiter=3,
            invalid_step_retry_budget=2,
            retry_step_shrink_factor=0.5,
        )
    return SingleStageSearchPolicy(
        donor_class=donor_class,
        search_policy=_SINGLE_STAGE_SEARCH_POLICY_GLOBAL_SEARCH,
        adaptive_failure_penalty_weight=2.0,
        auto_initial_step_scale=0.1,
        auto_initial_step_maxiter=5,
        invalid_step_retry_budget=2,
        retry_step_shrink_factor=0.5,
    )


def resolve_single_stage_policy_initial_phase_settings(
    search_policy,
    *,
    initial_step_scale,
    initial_step_maxiter,
    initial_step_scale_explicit=False,
    initial_step_maxiter_explicit=False,
    field_backend=None,
    optimizer_backend=None,
):
    """Auto-enable a conservative scaled phase only when the caller left defaults.

    The JAX/ondevice target lane keeps the scaled initial phase as an explicit
    opt-in only. That path creates a second compiled outer-optimizer objective
    boundary, which is useful for diagnosis but too expensive for the default
    reduced single-stage proof/runtime lane.
    """
    if initial_step_scale_explicit or initial_step_maxiter_explicit:
        return {
            "initial_step_scale": float(initial_step_scale),
            "initial_step_maxiter": int(initial_step_maxiter),
            "auto_enabled": False,
        }
    if initial_step_maxiter > 0 or initial_step_scale < 1.0:
        return {
            "initial_step_scale": float(initial_step_scale),
            "initial_step_maxiter": int(initial_step_maxiter),
            "auto_enabled": False,
        }
    if field_backend == "jax" and optimizer_backend == "ondevice":
        return {
            "initial_step_scale": float(initial_step_scale),
            "initial_step_maxiter": int(initial_step_maxiter),
            "auto_enabled": False,
        }
    if search_policy.auto_initial_step_scale is None:
        return {
            "initial_step_scale": float(initial_step_scale),
            "initial_step_maxiter": int(initial_step_maxiter),
            "auto_enabled": False,
        }
    return {
        "initial_step_scale": float(search_policy.auto_initial_step_scale),
        "initial_step_maxiter": int(search_policy.auto_initial_step_maxiter),
        "auto_enabled": True,
    }


def _cli_option_was_provided(raw_argv, option, env_var=None):
    """Return whether a CLI option or its env-backed default was set explicitly."""
    if any(arg == option or arg.startswith(f"{option}=") for arg in raw_argv):
        return True
    return env_var is not None and env_var in os.environ


def snapshot_single_stage_local_incumbent_state(run_dict):
    """Capture one accepted local single-stage state for retry restoration."""
    hardware_status = run_dict.get("hardware_constraint_status")
    coil_dofs = run_dict.get("x_prev")
    if coil_dofs is None:
        coil_dofs = np.zeros_like(host_array(run_dict["dJ"], dtype=np.float64))
    return {
        "coil_dofs": host_array(coil_dofs, dtype=np.float64),
        "sdofs": host_array(run_dict["sdofs"], dtype=np.float64),
        "iota": host_float(run_dict["iota"]),
        "G": host_float(run_dict["G"]),
        "J": host_float(run_dict["J"]),
        "dJ": host_array(run_dict["dJ"], dtype=np.float64),
        "intersecting": bool(run_dict.get("intersecting", False)),
        "self_intersection_check_available": bool(
            run_dict.get("self_intersection_check_available", False)
        ),
        "hardware_constraint_status": copy.deepcopy(hardware_status),
    }


def restore_single_stage_local_incumbent_state(run_dict, incumbent_state):
    """Restore the accepted local state used as a retry anchor."""
    _clear_target_lane_reporting_cache(run_dict)
    run_dict["x_prev"] = host_array(incumbent_state["coil_dofs"], dtype=np.float64)
    run_dict["sdofs"] = host_array(incumbent_state["sdofs"], dtype=np.float64)
    run_dict["iota"] = host_float(incumbent_state["iota"])
    run_dict["G"] = host_float(incumbent_state["G"])
    run_dict["J"] = host_float(incumbent_state["J"])
    run_dict["dJ"] = host_array(incumbent_state["dJ"], dtype=np.float64)
    run_dict["intersecting"] = bool(incumbent_state["intersecting"])
    run_dict["self_intersection_check_available"] = bool(
        incumbent_state["self_intersection_check_available"]
    )
    run_dict["hardware_constraint_status"] = copy.deepcopy(
        incumbent_state["hardware_constraint_status"]
    )
    run_dict["failure_count"] = 0
    run_dict.pop("last_candidate_failure", None)


def single_stage_local_incumbent_eligible(run_dict):
    """Return whether the current run_dict state is safe to preserve for retries."""
    if bool(run_dict.get("intersecting", False)):
        return False
    return bool(
        np.isfinite(host_float(run_dict["J"]))
        and np.all(np.isfinite(host_array(run_dict["dJ"], dtype=np.float64)))
    )


def record_single_stage_local_incumbent(run_dict, *, stage):
    """Track latest and best feasible local states for donor-aware retries."""
    if not single_stage_local_incumbent_eligible(run_dict):
        return False
    incumbent_state = snapshot_single_stage_local_incumbent_state(run_dict)
    incumbent_metric = float(incumbent_state["J"])
    run_dict["latest_local_incumbent"] = incumbent_state
    run_dict["latest_local_metric"] = incumbent_metric
    run_dict["latest_local_stage"] = str(stage)
    best_metric = run_dict.get("best_local_metric")
    if best_metric is not None and incumbent_metric >= float(best_metric):
        return False
    run_dict["best_local_incumbent"] = copy.deepcopy(incumbent_state)
    run_dict["best_local_metric"] = incumbent_metric
    run_dict["best_local_stage"] = str(stage)
    return True


def resolve_single_stage_retry_anchor(run_dict, single_stage_search_policy):
    """Choose the retry anchor that matches the current donor search policy."""
    if (
        single_stage_search_policy.search_policy
        == _SINGLE_STAGE_SEARCH_POLICY_PRESERVE_FIRST
    ):
        best_incumbent = run_dict.get("best_local_incumbent")
        if best_incumbent is not None:
            return best_incumbent, run_dict.get("best_local_stage")
    latest_incumbent = run_dict.get("latest_local_incumbent")
    if latest_incumbent is not None:
        return latest_incumbent, run_dict.get("latest_local_stage")
    best_incumbent = run_dict.get("best_local_incumbent")
    if best_incumbent is not None:
        return best_incumbent, run_dict.get("best_local_stage")
    return None, None


def single_stage_retry_triggered_by_invalid_state(events):
    """Return whether a target-lane failure event is retry-eligible."""
    if not events:
        return False
    last_event = events[-1]
    return bool(
        last_event["line_search_failed"]
        or last_event["nonfinite_step"]
        or last_event["stalled_step"]
    )


def resolve_single_stage_retry_initial_step_size(
    previous_initial_step_size,
    failure_events,
    *,
    single_stage_search_policy,
    retry_index,
):
    """Shrink the next target-lane trial step after an invalid-state failure."""
    del retry_index
    base_step_size = previous_initial_step_size
    if base_step_size is None and failure_events:
        base_step_size = failure_events[-1]["step_scale"]["value"]
    if base_step_size is None or base_step_size <= 0.0:
        base_step_size = 1.0
    return max(
        float(base_step_size)
        * float(single_stage_search_policy.retry_step_shrink_factor),
        1.0e-6,
    )


def resolve_single_stage_contract_policy_context(
    *,
    warm_start_state,
    single_stage_search_policy,
    effective_initial_phase_settings,
    args,
):
    """Backfill reporting metadata when helper callers omit policy inputs."""
    if single_stage_search_policy is None:
        explicit_surface_warm_start = (
            isinstance(warm_start_state, dict) and "surface" in warm_start_state
        )
        if warm_start_state is None or explicit_surface_warm_start:
            single_stage_search_policy = resolve_single_stage_search_policy(
                warm_start_state,
                explicit_surface_warm_start=explicit_surface_warm_start,
            )
        else:
            single_stage_search_policy = SingleStageSearchPolicy(
                donor_class="direct_call_unknown",
                search_policy=_SINGLE_STAGE_SEARCH_POLICY_GLOBAL_SEARCH,
                adaptive_failure_penalty_weight=1.0,
            )
    if effective_initial_phase_settings is None:
        if args is None:
            raise ValueError(
                "args is required when effective_initial_phase_settings is not "
                "provided to resolve_single_stage_contract_policy_context"
            )
        effective_initial_phase_settings = (
            resolve_single_stage_policy_initial_phase_settings(
                single_stage_search_policy,
                initial_step_scale=args.initial_step_scale,
                initial_step_maxiter=args.initial_step_maxiter,
                initial_step_scale_explicit=getattr(
                    args,
                    "initial_step_scale_explicit",
                    False,
                ),
                initial_step_maxiter_explicit=getattr(
                    args,
                    "initial_step_maxiter_explicit",
                    False,
                ),
                field_backend=getattr(args, "backend", None),
                optimizer_backend=getattr(args, "optimizer_backend", None),
            )
        )
    return single_stage_search_policy, effective_initial_phase_settings


def _surface_xyz_tensor_design_matrix(
    *,
    mpol,
    ntor,
    nfp,
    stellsym,
    quadpoints_phi,
    quadpoints_theta,
):
    theta_basis, _ = build_theta_basis(quadpoints_theta, mpol)
    phi_basis, _ = build_phi_basis(quadpoints_phi, ntor, nfp)
    basis_values = phi_basis[:, None, None, :] * theta_basis[None, :, :, None]
    basis_values = jnp.reshape(
        basis_values,
        (
            phi_basis.shape[0],
            theta_basis.shape[0],
            (2 * mpol + 1) * (2 * ntor + 1),
        ),
    )

    two_pi = _as_runtime_float64(2.0 * np.pi, reference=quadpoints_phi)
    phi_angles = two_pi * quadpoints_phi
    cos_phi = jnp.cos(phi_angles)[:, None, None]
    sin_phi = jnp.sin(phi_angles)[:, None, None]
    zeros = _as_runtime_float64(0.0, reference=basis_values) * basis_values

    x_block = jnp.stack(
        (
            basis_values * cos_phi,
            basis_values * sin_phi,
            zeros,
        ),
        axis=2,
    )
    y_block = jnp.stack(
        (
            -basis_values * sin_phi,
            basis_values * cos_phi,
            zeros,
        ),
        axis=2,
    )
    z_block = jnp.stack(
        (
            zeros,
            zeros,
            basis_values,
        ),
        axis=2,
    )
    design_matrix = jnp.reshape(
        jnp.concatenate((x_block, y_block, z_block), axis=3),
        (-1, 3 * basis_values.shape[2]),
    )
    if not stellsym:
        return design_matrix
    scatter_indices = _as_jax_int32(stellsym_scatter_indices(mpol, ntor))
    return jnp.take(design_matrix, scatter_indices, axis=1)


def _surface_xyz_tensor_design_matrix_host(
    *,
    mpol,
    ntor,
    nfp,
    stellsym,
    quadpoints_phi,
    quadpoints_theta,
):
    return host_array(
        _surface_xyz_tensor_design_matrix(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,
            stellsym=stellsym,
            quadpoints_phi=_as_jax_float64(quadpoints_phi),
            quadpoints_theta=_as_jax_float64(quadpoints_theta),
        ),
        dtype=np.float64,
    )


def _host_and_jax_float64(values):
    host_values = host_array(values, dtype=np.float64)
    return host_values, _as_jax_float64(host_values)


def _fit_surface_xyz_tensor_dofs_to_gamma(
    target_gamma,
    *,
    mpol,
    ntor,
    nfp,
    stellsym,
    quadpoints_phi,
    quadpoints_theta,
):
    quadpoints_phi_host = host_array(quadpoints_phi, dtype=np.float64)
    quadpoints_theta_host = host_array(quadpoints_theta, dtype=np.float64)
    target_gamma_host = host_array(target_gamma, dtype=np.float64).reshape(
        quadpoints_phi_host.size,
        quadpoints_theta_host.size,
        3,
    )
    # Keep the reprojection solve on the host. This path is a one-off compatibility
    # fit, and routing it through the GPU solver stack has triggered Hopper-only
    # cuSolver/runtime failures even when the target geometry itself is valid.
    design_matrix_host = _surface_xyz_tensor_design_matrix_host(
        mpol=max(1, int(mpol)),
        ntor=max(1, int(ntor)),
        nfp=int(nfp),
        stellsym=bool(stellsym),
        quadpoints_phi=quadpoints_phi_host,
        quadpoints_theta=quadpoints_theta_host,
    )
    rhs_host = np.reshape(target_gamma_host, (-1,))
    fitted_dofs, _, rank, _ = np.linalg.lstsq(
        design_matrix_host, rhs_host, rcond=None
    )
    fitted_dofs_host = _canonicalize_surface_xyz_tensor_fit_dofs(
        np.asarray(fitted_dofs, dtype=float),
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=quadpoints_phi_host,
        quadpoints_theta=quadpoints_theta_host,
    )
    design_rank = int(rank)
    return fitted_dofs_host, design_rank == int(design_matrix_host.shape[1])


def _quadpoints_cache_key(values):
    return tuple(
        float(value) for value in host_array(values, dtype=np.float64).reshape(-1)
    )


def _surface_xyz_tensor_alias_cache_args(
    *,
    mpol,
    ntor,
    nfp,
    stellsym,
    quadpoints_phi,
    quadpoints_theta,
):
    return (
        int(mpol),
        int(ntor),
        int(nfp),
        bool(stellsym),
        _quadpoints_cache_key(quadpoints_phi),
        _quadpoints_cache_key(quadpoints_theta),
    )


@lru_cache(maxsize=None)
def _surface_xyz_tensor_alias_groups(
    mpol,
    ntor,
    nfp,
    stellsym,
    quadpoints_phi_key,
    quadpoints_theta_key,
):
    design_matrix = _surface_xyz_tensor_design_matrix_host(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=quadpoints_phi_key,
        quadpoints_theta=quadpoints_theta_key,
    )
    ncols = int(design_matrix.shape[1])
    used = np.zeros(ncols, dtype=bool)
    groups = []
    atol = 1.0e-12
    for column_index in range(ncols):
        if used[column_index]:
            continue
        reference = design_matrix[:, column_index]
        members = [(column_index, 1.0)]
        used[column_index] = True
        for other_index in range(column_index + 1, ncols):
            if used[other_index]:
                continue
            candidate = design_matrix[:, other_index]
            if np.allclose(candidate, reference, rtol=0.0, atol=atol):
                members.append((other_index, 1.0))
                used[other_index] = True
            elif np.allclose(candidate, -reference, rtol=0.0, atol=atol):
                members.append((other_index, -1.0))
                used[other_index] = True
        groups.append(
            (
                int(column_index),
                bool(np.max(np.abs(reference)) <= atol),
                tuple((int(index), float(sign)) for index, sign in members),
            )
        )
    return tuple(groups)


@lru_cache(maxsize=None)
def _surface_xyz_tensor_alias_group_host_convention(
    mpol,
    ntor,
    nfp,
    stellsym,
    quadpoints_phi_key,
    quadpoints_theta_key,
):
    """Resolve the legacy host QR representative for each alias group."""
    quadpoints_phi = np.asarray(quadpoints_phi_key, dtype=float)
    quadpoints_theta = np.asarray(quadpoints_theta_key, dtype=float)
    alias_groups = _surface_xyz_tensor_alias_groups(
        int(mpol),
        int(ntor),
        int(nfp),
        bool(stellsym),
        quadpoints_phi_key,
        quadpoints_theta_key,
    )
    design_matrix = None
    host_surface = None

    def get_design_matrix():
        nonlocal design_matrix
        if design_matrix is None:
            design_matrix = _surface_xyz_tensor_design_matrix_host(
                mpol=mpol,
                ntor=ntor,
                nfp=nfp,
                stellsym=stellsym,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )
        return design_matrix

    def get_host_surface():
        nonlocal host_surface
        if host_surface is None:
            host_surface = SurfaceXYZTensorFourier(
                mpol=mpol,
                ntor=ntor,
                nfp=nfp,
                stellsym=stellsym,
                quadpoints_phi=quadpoints_phi,
                quadpoints_theta=quadpoints_theta,
            )
        return host_surface

    conventions = []
    for representative_index, zero_column, members in alias_groups:
        if zero_column or len(members) == 1:
            conventions.append((int(representative_index), 1.0))
            continue
        target_gamma = get_design_matrix()[:, representative_index].reshape(
            quadpoints_phi.size,
            quadpoints_theta.size,
            3,
        )
        resolved_host_surface = get_host_surface()
        resolved_host_surface.least_squares_fit(target_gamma)
        host_dofs = np.asarray(resolved_host_surface.get_dofs(), dtype=float)
        chosen_index, chosen_sign, chosen_value = max(
            (
                (int(index), float(sign), float(host_dofs[index]))
                for index, sign in members
            ),
            key=lambda item: abs(item[2]),
        )
        del chosen_value
        conventions.append((chosen_index, chosen_sign))
    return tuple(conventions)


def _canonicalize_surface_xyz_tensor_fit_dofs(
    fitted_dofs,
    *,
    mpol,
    ntor,
    nfp,
    stellsym,
    quadpoints_phi,
    quadpoints_theta,
):
    """Collapse alias-equivalent coefficients to the host solver convention."""
    alias_cache_args = _surface_xyz_tensor_alias_cache_args(
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    alias_groups = _surface_xyz_tensor_alias_groups(*alias_cache_args)
    host_convention = _surface_xyz_tensor_alias_group_host_convention(*alias_cache_args)
    canonical_dofs = np.zeros_like(np.asarray(fitted_dofs, dtype=float))
    for (representative_index, zero_column, members), (
        host_index,
        host_orientation,
    ) in zip(alias_groups, host_convention):
        if zero_column:
            canonical_dofs[representative_index] = 0.0
            continue
        canonical_value = sum(
            sign * float(fitted_dofs[index]) for index, sign in members
        )
        canonical_dofs[int(host_index)] = float(host_orientation) * canonical_value
    return canonical_dofs


def _target_gamma_from_supported_surface(
    surface,
    *,
    quadpoints_phi,
    quadpoints_theta,
):
    quadpoints_phi_jax = _as_jax_float64(quadpoints_phi)
    quadpoints_theta_jax = _as_jax_float64(quadpoints_theta)
    if isinstance(surface, SerializedSurfaceState):
        surface_class = surface.surface_class
        source_dofs = _as_jax_float64(surface.dofs)
        source_mpol = surface.mpol
        source_ntor = surface.ntor
        source_nfp = surface.nfp
        source_stellsym = surface.stellsym
    else:
        surface_class = type(surface).__name__
        if not hasattr(surface, "get_dofs"):
            return None
        source_dofs = _as_jax_float64(surface.get_dofs())
        source_mpol = int(surface.mpol)
        source_ntor = int(surface.ntor)
        source_nfp = int(surface.nfp)
        source_stellsym = bool(surface.stellsym)

    if surface_class == "SurfaceXYZTensorFourier" or isinstance(
        surface, SurfaceXYZTensorFourier
    ):
        scatter_indices = None
        if source_stellsym:
            scatter_indices = stellsym_scatter_indices(source_mpol, source_ntor)
        return surface_gamma_from_dofs(
            source_dofs,
            quadpoints_phi_jax,
            quadpoints_theta_jax,
            source_mpol,
            source_ntor,
            source_nfp,
            source_stellsym,
            scatter_indices=scatter_indices,
        )
    if surface_class == "SurfaceRZFourier" or isinstance(surface, SurfaceRZFourier):
        source_spec = surface_rz_fourier_spec_from_dofs(
            source_dofs,
            quadpoints_phi=quadpoints_phi_jax,
            quadpoints_theta=quadpoints_theta_jax,
            mpol=source_mpol,
            ntor=source_ntor,
            nfp=source_nfp,
            stellsym=source_stellsym,
        )
        return surface_rz_fourier_gamma_from_dofs(source_spec, source_dofs)
    return None


def _matching_surface_xyz_tensor_dofs(
    surface,
    *,
    mpol,
    ntor,
    quadpoints_phi,
    quadpoints_theta,
):
    if isinstance(surface, SerializedSurfaceState):
        surface_class = surface.surface_class
        source_dofs = surface.dofs
    else:
        surface_class = getattr(
            surface,
            "deferred_surface_class",
            type(surface).__name__,
        )
        if not hasattr(surface, "get_dofs"):
            return None
        source_dofs = surface.get_dofs()
    if surface_class != "SurfaceXYZTensorFourier":
        return None
    if int(surface.mpol) != int(mpol) or int(surface.ntor) != int(ntor):
        return None
    if not np.array_equal(
        np.asarray(surface.quadpoints_phi, dtype=np.float64),
        np.asarray(quadpoints_phi, dtype=np.float64),
    ):
        return None
    if not np.array_equal(
        np.asarray(surface.quadpoints_theta, dtype=np.float64),
        np.asarray(quadpoints_theta, dtype=np.float64),
    ):
        return None
    return np.asarray(host_array(source_dofs), dtype=np.float64)


def project_surface_dofs_to_resolution(
    surface,
    *,
    mpol,
    ntor,
    quadpoints_phi,
    quadpoints_theta,
):
    """Reproject surface geometry onto the requested target resolution."""
    matching_dofs = _matching_surface_xyz_tensor_dofs(
        surface,
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    if matching_dofs is not None:
        return matching_dofs
    target_gamma = _target_gamma_from_supported_surface(
        surface,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    if target_gamma is None:
        raise TypeError(
            "project_surface_dofs_to_resolution only supports "
            "SurfaceXYZTensorFourier, SurfaceRZFourier, and serialized warm-start "
            f"surfaces on the dehybridized path; got {type(surface).__name__}."
        )
    projected_dofs, _ = _fit_surface_xyz_tensor_dofs_to_gamma(
        target_gamma,
        mpol=max(1, int(mpol)),
        ntor=max(1, int(ntor)),
        nfp=int(surface.nfp),
        stellsym=bool(surface.stellsym),
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )
    return projected_dofs


def project_single_stage_warm_start_surface_dofs(
    surface,
    *,
    mpol,
    ntor,
    quadpoints_phi,
    quadpoints_theta,
):
    """Backward-compatible warm-start wrapper around the generic projector."""
    return project_surface_dofs_to_resolution(
        surface,
        mpol=mpol,
        ntor=ntor,
        quadpoints_phi=quadpoints_phi,
        quadpoints_theta=quadpoints_theta,
    )


def build_equilibrium_path(args):
    if args.equilibrium_path is not None:
        return args.equilibrium_path

    candidate_paths = [
        os.path.join(args.equilibria_dir, args.plasma_surf_filename),
        os.path.join(DATABASE_EQUILIBRIA_DIR, args.plasma_surf_filename),
    ]
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    return candidate_paths[0]


def apply_default_stage2_seed_args(args):
    default_seed = DEFAULT_STAGE2_SEEDS_BY_PLASMA.get(args.plasma_surf_filename, {})
    if args.stage2_seed_major_radius is None:
        args.stage2_seed_major_radius = default_seed.get("major_radius", 0.915)
    if args.stage2_seed_toroidal_flux is None:
        args.stage2_seed_toroidal_flux = default_seed.get("toroidal_flux", 0.24)
    if args.stage2_seed_length_weight is None:
        args.stage2_seed_length_weight = default_seed.get("length_weight", 0.0005)
    if args.stage2_seed_cc_weight is None:
        args.stage2_seed_cc_weight = default_seed.get("cc_weight", 100.0)
    if args.stage2_seed_curvature_weight is None:
        args.stage2_seed_curvature_weight = default_seed.get("curvature_weight", 0.0001)
    if args.stage2_seed_cc_threshold is None:
        args.stage2_seed_cc_threshold = default_seed.get("cc_threshold", 0.05)
    if args.stage2_seed_curvature_threshold is None:
        args.stage2_seed_curvature_threshold = default_seed.get(
            "curvature_threshold", 40.0
        )
    args.stage2_seed_curvature_threshold = resolve_curvature_threshold(
        args.stage2_seed_curvature_threshold
    )
    if args.stage2_seed_banana_surf_radius is None:
        args.stage2_seed_banana_surf_radius = default_seed.get(
            "banana_surf_radius", 0.22
        )
    if args.stage2_seed_order is None:
        args.stage2_seed_order = default_seed.get("order", 2)
    return args


def validate_single_stage_current_args(args) -> None:
    banana_current_max_A = float(args.banana_current_max_A)
    if not (0.0 < banana_current_max_A <= BANANA_CURRENT_HARD_LIMIT_A):
        raise ValueError(
            f"--banana-current-max-A must be in the interval "
            f"(0, {BANANA_CURRENT_HARD_LIMIT_A:.0f}]."
        )
    if float(args.length_target) <= 0.0:
        raise ValueError("--length-target must be positive")


def validate_single_stage_alm_formulation_args(args) -> None:
    if args.alm_formulation == "weighted_sum":
        return
    if args.constraint_method != "alm":
        raise ValueError(
            "--alm-formulation=thresholded_physics requires --constraint-method=alm"
        )

    required_thresholds = {
        "--alm-qs-threshold": args.alm_qs_threshold,
        "--alm-boozer-threshold": args.alm_boozer_threshold,
        "--alm-iota-penalty-threshold": args.alm_iota_penalty_threshold,
        "--alm-length-penalty-threshold": args.alm_length_penalty_threshold,
    }
    missing_thresholds = [
        flag_name for flag_name, value in required_thresholds.items() if value is None
    ]
    if missing_thresholds:
        raise ValueError(
            "thresholded_physics ALM formulation requires explicit thresholds for "
            + ", ".join(missing_thresholds)
        )

    negative_thresholds = [
        flag_name
        for flag_name, value in required_thresholds.items()
        if float(value) < 0.0
    ]
    if negative_thresholds:
        raise ValueError(
            "thresholded_physics ALM thresholds must be non-negative: "
            + ", ".join(negative_thresholds)
        )


def single_stage_alm_constraint_names(*, alm_formulation, include_surface_surface):
    available_names = {
        "coil_coil_spacing",
        "coil_surface_spacing",
        "max_curvature",
        "coil_length",
        "banana_current",
    }
    if include_surface_surface:
        available_names.add("surface_vessel_spacing")
    names = list(hardware_constraint_alm_names(names=available_names))
    if alm_formulation == "thresholded_physics":
        names.extend(SINGLE_STAGE_THRESHOLDED_PHYSICS_CONSTRAINT_NAMES)
    return names


def build_single_stage_alm_settings(args):
    return ALMSettings(
        max_outer_iterations=args.alm_max_outer_iters,
        max_subproblem_continuations=args.alm_max_subproblem_continuations,
        penalty_init=args.alm_penalty_init,
        penalty_scale=args.alm_penalty_scale,
        penalty_max=args.alm_penalty_max,
        feasibility_tol=args.alm_feas_tol,
        stationarity_tol=args.alm_stationarity_tol,
        trust_radius_init=(
            None
            if float(args.alm_trust_radius_init) == 0.0
            else float(args.alm_trust_radius_init)
        ),
        trust_radius_min=args.alm_trust_radius_min,
        trust_radius_shrink=args.alm_trust_radius_shrink,
        trust_radius_grow=args.alm_trust_radius_grow,
        max_inner_attempts=args.alm_max_inner_attempts,
    )


def resolve_single_stage_banana_surface_radius(args, stage2_results):
    requested_radius = (
        args.banana_surf_radius
        if args.banana_surf_radius is not None
        else float(stage2_results["banana_surf_radius"])
    )
    return validate_banana_winding_surface_radius(requested_radius)


def _jsonable_value(value):
    if isinstance(value, np.ndarray):
        return _jsonable_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value


def build_single_stage_alm_partial_state(
    run_dict,
    constraint_names,
    history,
    latest_history_entry,
    multipliers,
    penalty,
    *,
    outer_iteration=None,
    termination_message=None,
    optimizer_success=None,
    termination_reason=None,
    inner_optimizer_success=None,
    inner_optimizer_message=None,
    converged_to_tolerances=None,
    restored_best_feasible=None,
    restored_best_feasible_reason=None,
    final_max_feasibility_violation=None,
    final_stationarity_norm=None,
):
    current_objective = None if "J" not in run_dict else float(run_dict["J"])
    return {
        "outer_iteration": None if outer_iteration is None else int(outer_iteration),
        "constraint_names": list(constraint_names),
        "penalty": float(penalty),
        "multipliers": np.asarray(multipliers, dtype=float).tolist(),
        "history_length": int(len(history)),
        "latest_history_entry": _jsonable_value(latest_history_entry),
        "history": _jsonable_value(history),
        "accepted_iterations": int(run_dict.get("accepted_iterations", 0)),
        "current_iteration": int(run_dict.get("it", 0)),
        "current_objective": current_objective,
        "accepted_boozer_stage": run_dict.get("accepted_boozer_stage"),
        "accepted_hardware_status": _jsonable_value(
            run_dict.get("accepted_hardware_status")
        ),
        "trial_hardware_status": _jsonable_value(run_dict.get("trial_hardware_status")),
        "topology_gate_status": _jsonable_value(run_dict.get("topology_gate_status")),
        "termination_message": termination_message,
        "optimizer_success": optimizer_success,
        "termination_reason": termination_reason,
        "inner_optimizer_success": inner_optimizer_success,
        "inner_optimizer_message": inner_optimizer_message,
        "converged_to_tolerances": converged_to_tolerances,
        "restored_best_feasible": restored_best_feasible,
        "restored_best_feasible_reason": restored_best_feasible_reason,
        "final_max_feasibility_violation": final_max_feasibility_violation,
        "final_stationarity_norm": final_stationarity_norm,
    }


def write_single_stage_alm_partial_state(out_dir, payload):
    write_json_file(os.path.join(out_dir, "alm_state.partial.json"), payload)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-stage Boozer/quasi-symmetry optimization from a Stage 2 seed.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=os.environ.get(
            "PLASMA_SURF_FILENAME", "wout_nfp22ginsburg_000_014417_iota15.nc"
        ),
        help="VMEC wout filename under the equilibria directory.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=os.environ.get("EQUILIBRIA_DIR", DEFAULT_EQUILIBRIA_DIR),
        help="Directory that contains the equilibrium wout files.",
    )
    parser.add_argument(
        "--equilibrium-path",
        default=os.environ.get("EQUILIBRIUM_PATH"),
        help="Explicit path to the equilibrium file. Overrides --equilibria-dir.",
    )
    parser.add_argument(
        "--output-root",
        default=os.environ.get(
            "SINGLE_STAGE_OUTPUT_ROOT", DEFAULT_SINGLE_STAGE_OUTPUT_ROOT
        ),
        help="Directory where the single-stage output family will be written.",
    )
    parser.add_argument(
        "--warm-start-run-dir",
        default=os.environ.get("WARM_START_RUN_DIR"),
        help=(
            "Optional prior single-stage run directory containing surf_opt.json "
            "and results.json. When provided, the Boozer initialization reuses "
            "that optimized surface geometry and solved iota/G as a warm start."
        ),
    )
    parser.add_argument(
        "--jax-runtime-seed-spec",
        default=os.environ.get("JAX_RUNTIME_SEED_SPEC"),
        help=(
            "Immutable single-stage JAX runtime seed spec. Required by the "
            "production JAX lane when --warm-start-run-dir is omitted; warm-start "
            "runs read the same artifact from the donor directory unless this "
            "path is provided explicitly."
        ),
    )
    parser.add_argument(
        "--compile-jax-runtime-seed-spec",
        action="store_true",
        help=(
            "Convert --warm-start-run-dir into an immutable JAX runtime seed spec "
            "and exit. Writes --jax-runtime-seed-spec when provided, otherwise "
            "writes into the donor run directory."
        ),
    )
    parser.add_argument(
        "--minimal-artifacts",
        action="store_true",
        help=(
            "Write only the JSON artifacts needed for warm restarts and skip "
            "heavy VTK/plot outputs."
        ),
    )
    parser.add_argument(
        "--full-artifacts",
        action="store_true",
        help=(
            "Opt in to heavy VTK/plot artifacts on JAX runs. CPU/reference runs "
            "write full artifacts by default unless --minimal-artifacts or "
            "--benchmark-mode is set."
        ),
    )
    parser.add_argument(
        "--banana-surf-radius",
        type=float,
        default=float(os.environ["BANANA_SURF_RADIUS"])
        if "BANANA_SURF_RADIUS" in os.environ
        else None,
        help="Coil surface minor radius. Defaults to the Stage 2 seed radius when omitted.",
    )
    parser.add_argument("--nphi", type=int, default=int(os.environ.get("NPHI", "255")))
    parser.add_argument(
        "--ntheta", type=int, default=int(os.environ.get("NTHETA", "64"))
    )
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Build the initial Boozer surface, write init artifacts, and skip the optimizer.",
    )
    parser.add_argument("--mpol", type=int, default=int(os.environ.get("MPOL", "8")))
    parser.add_argument("--ntor", type=int, default=int(os.environ.get("NTOR", "6")))
    parser.add_argument(
        "--vol-target", type=float, default=float(os.environ.get("VOL_TARGET", "0.10"))
    )
    parser.add_argument(
        "--constraint-weight",
        type=float,
        default=float(os.environ.get("CONSTRAINT_WEIGHT", "1.0")),
    )
    parser.add_argument(
        "--maxiter", type=int, default=int(os.environ.get("MAXITER", "300"))
    )
    parser.add_argument(
        "--constraint-method",
        choices=["penalty", "alm"],
        default=os.environ.get("CONSTRAINT_METHOD", "penalty"),
        help="Use the weighted-penalty objective or the augmented Lagrangian outer loop.",
    )
    parser.add_argument(
        "--alm-max-outer-iters",
        type=int,
        default=int(os.environ.get("ALM_MAX_OUTER_ITERS", "10")),
        help="Maximum number of ALM outer iterations (default 10).",
    )
    parser.add_argument(
        "--alm-penalty-init",
        type=float,
        default=float(os.environ.get("ALM_PENALTY_INIT", "1.0")),
        help="Initial ALM penalty parameter (default 1.0).",
    )
    parser.add_argument(
        "--alm-penalty-scale",
        type=float,
        default=float(os.environ.get("ALM_PENALTY_SCALE", "10.0")),
        help="Multiplicative ALM penalty growth factor (default 10.0).",
    )
    parser.add_argument(
        "--alm-penalty-max",
        type=float,
        default=float(os.environ.get("ALM_PENALTY_MAX", "1e8")),
        help="Maximum ALM penalty parameter before capped termination (default 1e8).",
    )
    parser.add_argument(
        "--alm-feas-tol",
        type=float,
        default=float(os.environ.get("ALM_FEAS_TOL", "1e-6")),
        help="ALM max-violation stopping tolerance (default 1e-6).",
    )
    parser.add_argument(
        "--alm-stationarity-tol",
        type=float,
        default=float(os.environ.get("ALM_STATIONARITY_TOL", "1e-6")),
        help="ALM augmented-gradient stopping tolerance (default 1e-6).",
    )
    parser.add_argument(
        "--alm-trust-radius-init",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_INIT", "0.05")),
        help="Initial relative trust radius for bounded ALM inner solves (0 disables bounds).",
    )
    parser.add_argument(
        "--alm-trust-radius-min",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_MIN", "1e-4")),
        help="Minimum relative trust radius for bounded ALM inner solves.",
    )
    parser.add_argument(
        "--alm-trust-radius-shrink",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_SHRINK", "0.5")),
        help="Multiplicative shrink factor for the ALM inner trust radius.",
    )
    parser.add_argument(
        "--alm-trust-radius-grow",
        type=float,
        default=float(os.environ.get("ALM_TRUST_RADIUS_GROW", "1.5")),
        help="Multiplicative growth factor for the ALM inner trust radius after good steps.",
    )
    parser.add_argument(
        "--alm-max-inner-attempts",
        type=int,
        default=int(os.environ.get("ALM_MAX_INNER_ATTEMPTS", "4")),
        help="Maximum number of trust-radius retries per ALM outer iteration.",
    )
    parser.add_argument(
        "--alm-max-subproblem-continuations",
        type=int,
        default=int(os.environ.get("ALM_MAX_SUBPROBLEM_CONTINUATIONS", "20")),
        help="Maximum accepted-feasible continuation solves before forcing an ALM return.",
    )
    parser.add_argument(
        "--alm-distance-smoothing",
        type=float,
        default=float(os.environ.get("ALM_DISTANCE_SMOOTHING", "0.005")),
        help="Distance soft-min temperature for single-stage ALM spacing constraints.",
    )
    parser.add_argument(
        "--alm-curvature-smoothing",
        type=float,
        default=float(os.environ.get("ALM_CURVATURE_SMOOTHING", "0.05")),
        help="Curvature smooth-max temperature for single-stage ALM curvature constraints.",
    )
    parser.add_argument(
        "--alm-formulation",
        choices=["weighted_sum", "thresholded_physics"],
        default=os.environ.get("ALM_FORMULATION", "weighted_sum"),
        help=(
            "ALM objective assembly. 'weighted_sum' keeps physics terms in the base objective; "
            "'thresholded_physics' uses a dummy zero objective and promotes physics terms "
            "to inequality constraints."
        ),
    )
    parser.add_argument(
        "--alm-qs-threshold",
        type=float,
        default=float(os.environ["ALM_QS_THRESHOLD"])
        if "ALM_QS_THRESHOLD" in os.environ
        else None,
        help="thresholded_physics-mode upper bound for the quasi-symmetry objective J_QS.",
    )
    parser.add_argument(
        "--alm-boozer-threshold",
        type=float,
        default=float(os.environ["ALM_BOOZER_THRESHOLD"])
        if "ALM_BOOZER_THRESHOLD" in os.environ
        else None,
        help="thresholded_physics-mode upper bound for the Boozer residual objective.",
    )
    parser.add_argument(
        "--alm-iota-penalty-threshold",
        type=float,
        default=float(os.environ["ALM_IOTA_PENALTY_THRESHOLD"])
        if "ALM_IOTA_PENALTY_THRESHOLD" in os.environ
        else None,
        help="thresholded_physics-mode upper bound for the Jiota penalty objective.",
    )
    parser.add_argument(
        "--alm-length-penalty-threshold",
        type=float,
        default=float(os.environ["ALM_LENGTH_PENALTY_THRESHOLD"])
        if "ALM_LENGTH_PENALTY_THRESHOLD" in os.environ
        else None,
        help="thresholded_physics-mode upper bound for the single-stage length penalty objective.",
    )
    parser.add_argument(
        "--iota-target",
        type=float,
        default=float(os.environ.get("IOTA_TARGET", "0.15")),
    )
    parser.add_argument(
        "--num-tf-coils", type=int, default=int(os.environ.get("NUM_TF_COILS", "20"))
    )
    parser.add_argument(
        "--boozer-stage",
        choices=["initial", "final"],
        default=os.environ.get("BOOZER_STAGE", "initial"),
        help="Use least-squares Boozer residual during initial stage or exact residual during final stage.",
    )
    parser.add_argument(
        "--cc-dist", type=float, default=float(os.environ.get("CC_DIST", "0.05"))
    )
    parser.add_argument(
        "--curvature-threshold",
        type=float,
        default=float(os.environ.get("CURVATURE_THRESHOLD", "40")),
    )
    parser.add_argument(
        "--cc-weight", type=float, default=float(os.environ.get("CC_WEIGHT", "100"))
    )
    parser.add_argument(
        "--curvature-weight",
        type=float,
        default=float(os.environ.get("CURVATURE_WEIGHT", "0.1")),
    )
    parser.add_argument(
        "--length-weight",
        type=float,
        default=float(os.environ.get("SS_LENGTH_WEIGHT", "1")),
        help="Curve length penalty weight (default 1).",
    )
    parser.add_argument(
        "--banana-current-max-A",
        type=float,
        default=float(
            os.environ.get("BANANA_CURRENT_MAX_A", str(BANANA_CURRENT_HARD_LIMIT_A))
        ),
        help=(
            "Maximum allowed magnitude for the banana current in amps. "
            "Penalty mode applies this as a hard box bound; ALM mode still "
            "certifies it at final feasibility."
        ),
    )
    parser.add_argument(
        "--length-target",
        type=float,
        default=float(os.environ.get("SS_LENGTH_TARGET", str(COIL_LENGTH_TARGET_M))),
        help=(
            "Curve length quadratic penalty target in meters. Values above the "
            "hardware contract are clamped back to the shared ceiling."
        ),
    )
    parser.add_argument(
        "--res-weight",
        type=float,
        default=float(os.environ.get("RES_WEIGHT", "1000")),
        help="Boozer residual penalty weight (default 1000).",
    )
    parser.add_argument(
        "--iotas-weight",
        type=float,
        default=float(os.environ.get("IOTAS_WEIGHT", "100")),
        help="Iota target tracking weight (default 100).",
    )
    parser.add_argument(
        "--cs-weight",
        type=float,
        default=float(os.environ.get("CS_WEIGHT", "1")),
        help="Coil-surface distance penalty weight (default 1).",
    )
    parser.add_argument(
        "--cs-dist",
        type=float,
        default=float(os.environ.get("CS_DIST", "0.02")),
        help="Minimum coil-surface distance in meters (default 0.02).",
    )
    parser.add_argument(
        "--surf-dist-weight",
        type=float,
        default=float(os.environ.get("SURF_DIST_WEIGHT", "1000")),
        help="Surface-vessel distance penalty weight (default 1000).",
    )
    parser.add_argument(
        "--ss-dist",
        type=float,
        default=float(os.environ.get("SS_DIST", "0.04")),
        help="Minimum surface-vessel distance in meters (default 0.04).",
    )
    parser.add_argument(
        "--maxcor",
        type=int,
        default=int(os.environ["MAXCOR"]) if "MAXCOR" in os.environ else None,
        help=(
            "L-BFGS memory (number of corrections). Defaults to a tighter budget "
            "on the JAX ondevice lane and the historical budget elsewhere."
        ),
    )
    parser.add_argument(
        "--outer-maxls",
        type=int,
        default=int(os.environ["OUTER_MAXLS"]) if "OUTER_MAXLS" in os.environ else None,
        help=(
            "Maximum strong-Wolfe line-search evaluations per outer L-BFGS step. "
            "Defaults to a tighter budget on the JAX ondevice lane and the "
            "historical budget elsewhere."
        ),
    )
    parser.add_argument(
        "--outer-ftol",
        type=float,
        default=float(os.environ["OUTER_FTOL"]) if "OUTER_FTOL" in os.environ else None,
        help=(
            "Optional outer L-BFGS relative objective tolerance. Defaults to the "
            "mpol-specific production table."
        ),
    )
    parser.add_argument(
        "--target-lane-outer-initial-step-size",
        type=float,
        default=float(os.environ["TARGET_LANE_OUTER_INITIAL_STEP_SIZE"])
        if "TARGET_LANE_OUTER_INITIAL_STEP_SIZE" in os.environ
        else None,
        help=(
            "Optional initial strong-Wolfe trial step for the JAX/ondevice "
            "outer L-BFGS line search. This is mainly useful for proof and "
            "benchmark runs whose first accepted step requires a much smaller "
            "trial scale than the optimizer's default start."
        ),
    )
    parser.add_argument(
        "--initial-step-scale",
        type=float,
        default=float(os.environ.get("OUTER_INITIAL_STEP_SCALE", "1.0")),
        help=(
            "Physical step scale for an optional initial outer-optimization "
            "phase. Values below 1.0 shrink early optimizer moves in a "
            "mathematically consistent scaled coordinate system."
        ),
    )
    parser.add_argument(
        "--initial-step-maxiter",
        type=int,
        default=int(os.environ.get("OUTER_INITIAL_STEP_MAXITER", "0")),
        help=(
            "Maximum outer iterations to spend in the scaled initial phase. "
            "Set to 0 to disable the early-step continuation phase."
        ),
    )
    parser.add_argument(
        "--target-lane-boozer-bfgs-tol",
        type=float,
        default=float(os.environ["TARGET_LANE_BOOZER_BFGS_TOL"])
        if "TARGET_LANE_BOOZER_BFGS_TOL" in os.environ
        else None,
        help=(
            "Temporary Boozer LS tolerance override used only while evaluating "
            "target-lane outer-loop trial points."
        ),
    )
    parser.add_argument(
        "--target-lane-boozer-bfgs-maxiter",
        type=int,
        default=int(os.environ["TARGET_LANE_BOOZER_BFGS_MAXITER"])
        if "TARGET_LANE_BOOZER_BFGS_MAXITER" in os.environ
        else None,
        help=(
            "Temporary Boozer LS iteration cap used only while evaluating "
            "target-lane outer-loop trial points."
        ),
    )
    parser.add_argument(
        "--target-lane-boozer-newton-tol",
        type=float,
        default=float(os.environ["TARGET_LANE_BOOZER_NEWTON_TOL"])
        if "TARGET_LANE_BOOZER_NEWTON_TOL" in os.environ
        else None,
        help=(
            "Temporary Boozer Newton tolerance override used only while "
            "evaluating target-lane outer-loop trial points."
        ),
    )
    parser.add_argument(
        "--target-lane-boozer-newton-maxiter",
        type=int,
        default=int(os.environ["TARGET_LANE_BOOZER_NEWTON_MAXITER"])
        if "TARGET_LANE_BOOZER_NEWTON_MAXITER" in os.environ
        else None,
        help=(
            "Temporary Boozer Newton iteration cap used only while evaluating "
            "target-lane outer-loop trial points."
        ),
    )
    parser.add_argument(
        "--stage2-source",
        choices=["database", "local"],
        default=os.environ.get("STAGE2_SOURCE", "database"),
        help="Resolve the Stage 2 seed from the archive database or from local STAGE_2 outputs.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=os.environ.get("STAGE2_BS_PATH"),
        help="Explicit path to the Stage 2 biot_savart_opt.json seed. Overrides all derived seed settings.",
    )
    parser.add_argument(
        "--local-stage2-root",
        default=os.environ.get("LOCAL_STAGE2_ROOT", DEFAULT_LOCAL_STAGE2_ROOT),
        help="Directory that contains local STAGE_2 outputs-[plasma]/... runs.",
    )
    parser.add_argument(
        "--database-stage2-root",
        default=os.environ.get("DATABASE_STAGE2_ROOT", DEFAULT_DATABASE_STAGE2_ROOT),
        help="Directory that contains DATABASE/COIL_OPTIMIZATION/outputs.",
    )
    parser.add_argument(
        "--stage2-seed-major-radius",
        type=float,
        default=float(os.environ["STAGE2_SEED_MAJOR_RADIUS"])
        if "STAGE2_SEED_MAJOR_RADIUS" in os.environ
        else None,
    )
    parser.add_argument(
        "--stage2-seed-toroidal-flux",
        type=float,
        default=float(os.environ["STAGE2_SEED_TOROIDAL_FLUX"])
        if "STAGE2_SEED_TOROIDAL_FLUX" in os.environ
        else None,
    )
    parser.add_argument(
        "--stage2-seed-length-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_LENGTH_WEIGHT"])
        if "STAGE2_SEED_LENGTH_WEIGHT" in os.environ
        else None,
    )
    parser.add_argument(
        "--stage2-seed-cc-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_CC_WEIGHT"])
        if "STAGE2_SEED_CC_WEIGHT" in os.environ
        else None,
    )
    parser.add_argument(
        "--stage2-seed-curvature-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_CURVATURE_WEIGHT"])
        if "STAGE2_SEED_CURVATURE_WEIGHT" in os.environ
        else None,
    )
    parser.add_argument(
        "--stage2-seed-cc-threshold",
        type=float,
        default=float(os.environ["STAGE2_SEED_CC_THRESHOLD"])
        if "STAGE2_SEED_CC_THRESHOLD" in os.environ
        else None,
    )
    parser.add_argument(
        "--stage2-seed-curvature-threshold",
        type=float,
        default=float(os.environ["STAGE2_SEED_CURVATURE_THRESHOLD"])
        if "STAGE2_SEED_CURVATURE_THRESHOLD" in os.environ
        else None,
    )
    parser.add_argument(
        "--stage2-seed-banana-surf-radius",
        type=float,
        default=float(os.environ["STAGE2_SEED_BANANA_SURF_RADIUS"])
        if "STAGE2_SEED_BANANA_SURF_RADIUS" in os.environ
        else None,
    )
    parser.add_argument(
        "--stage2-seed-order",
        type=int,
        default=int(os.environ["STAGE2_SEED_ORDER"])
        if "STAGE2_SEED_ORDER" in os.environ
        else None,
    )
    parser.add_argument(
        "--backend",
        choices=["cpu", "jax"],
        default=os.environ.get("SIMSOPT_BACKEND", "cpu"),
        help="Field/objective backend: cpu (simsoptpp) or jax (JAX autodiff).",
    )
    parser.add_argument(
        "--optimizer-backend",
        choices=["scipy", "ondevice"],
        default=os.environ.get("OPTIMIZER_BACKEND"),
        help=(
            "JAX outer single-stage optimizer backend. Recorded in the run "
            "fingerprint and used to select the outer optimization path. "
            "Defaults to 'ondevice' on the JAX backend and 'scipy' on the "
            "CPU/reference backend when no explicit override is provided."
        ),
    )
    parser.add_argument(
        "--boozer-optimizer-backend",
        choices=["scipy", "ondevice"],
        default=None,
        help=(
            "Optional override for the inner JAX Boozer LS solve backend. "
            "Defaults to --optimizer-backend when omitted."
        ),
    )
    parser.add_argument(
        "--boozer-least-squares-algorithm",
        choices=["quasi-newton", "lm"],
        default=os.environ.get("BOOZER_LEAST_SQUARES_ALGORITHM"),
        help=(
            "Optional override for the inner JAX Boozer LS algorithm. "
            "Defaults to 'quasi-newton' on all lanes when omitted. "
            "'lm' is explicit opt-in only."
        ),
    )
    parser.add_argument(
        "--boozer-limited-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use L-BFGS instead of full-memory BFGS for the inner JAX Boozer "
            "quasi-Newton stage. Defaults to disabled unless explicitly enabled."
        ),
    )
    parser.add_argument(
        "--target-lane-accepted-step-sync",
        choices=TARGET_LANE_ACCEPTED_STEP_SYNC_CHOICES,
        default=os.environ.get(
            "TARGET_LANE_ACCEPTED_STEP_SYNC",
            TARGET_LANE_ACCEPTED_STEP_SYNC_DEFAULT,
        ),
        help=(
            "How the target ondevice outer lane refreshes mutable state for "
            "accepted-step diagnostics."
        ),
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help=(
            "Skip heavy plotting/VTK artifacts and use the cheap target-lane "
            "final sync path so benchmark/probe runs measure solver time more directly."
        ),
    )
    parser.add_argument(
        "--disable-target-lane-success-filter",
        action="store_true",
        help=(
            "Benchmark/proof-only: bypass the target-lane hard hardware success "
            "filter during outer objective evaluation. Final hardware verdicts "
            "are still applied to the recorded results."
        ),
    )
    parser.add_argument(
        "--profile-target-lane",
        action="store_true",
        help=(
            "Record first-vs-warm timing breakdowns for the traceable target-lane "
            "objective closure suite used to study the outer optimization path."
        ),
    )
    parser.add_argument(
        "--profile-target-lane-only",
        action="store_true",
        help=(
            "Build and profile the target-lane runtime bundle, then skip the "
            "outer optimizer. This is a fast profiling path for JAX/ondevice only."
        ),
    )
    parser.add_argument(
        "--profile-target-lane-batch-size",
        type=int,
        default=1,
        help=(
            "When profiling the JAX/ondevice target lane, also profile a batched "
            "seed-evaluation path over this many nearby deterministic seed points. "
            "Values greater than 1 are additive and do not change optimizer behavior."
        ),
    )
    parser.add_argument(
        "--record-jax-compile-diagnostics",
        action="store_true",
        help=(
            "Record named JAX compile/cache-miss diagnostics for the real "
            "target-lane bundle setup and outer optimizer, then write the summary "
            "into results.json for compile-reuse smoke tests."
        ),
    )
    parser.add_argument(
        "--diagnose-target-lane-gradient",
        action="store_true",
        help=(
            "Build the JAX/ondevice target-lane runtime bundle, evaluate the "
            "exact baseline value/gradient, and write a term-by-term finiteness "
            "report instead of running the outer optimizer."
        ),
    )
    parser.add_argument(
        "--diagnose-target-lane-first-line-search",
        action="store_true",
        help=(
            "Build the JAX/ondevice target-lane runtime bundle, run the first "
            "host-dispatched L-BFGS Wolfe search from the seeded -grad direction, "
            "and write the actual trial alpha/value/derivative trace instead of "
            "running the full outer optimizer."
        ),
    )
    parser.add_argument(
        "--diagnose-target-lane-scaled-phase1",
        action="store_true",
        help=(
            "Run the exact JAX/ondevice scaled initial-phase contract in "
            "diagnostic mode, recording origin/trial/reevaluation finiteness "
            "before and after the phase-1 optimizer."
        ),
    )
    parser.add_argument(
        "--diagnostic-callbacks",
        action="store_true",
        help=(
            "Opt in to target-lane host callbacks for detailed progress and "
            "rejected-step payloads. On explicit CUDA-only runs this also "
            "keeps a CPU lane available for JAX host callbacks."
        ),
    )
    parser.add_argument(
        "--record-target-lane-invalid-state-events",
        action="store_true",
        help=(
            "Deprecated alias for --diagnostic-callbacks. Keeps detailed "
            "target-lane rejected-step payload recording enabled."
        ),
    )
    parser.add_argument(
        "--jax-profile-dir",
        default=os.environ.get("JAX_PROFILE_DIR"),
        help=(
            "Optional output directory for a JAX/XProf trace of the heavy "
            "single-stage phases."
        ),
    )
    parser.add_argument(
        "--experimental-target-lane-value-and-grad",
        action="store_true",
        help=(
            "Legacy compatibility flag. The single-stage JAX ondevice target lane "
            "now uses the fused runtime-bundle (value, grad) contract by default."
        ),
    )
    raw_argv = list(sys.argv[1:])
    args = parser.parse_args()
    args.diagnostic_callbacks = target_lane_diagnostic_callbacks_enabled(args)
    args.record_target_lane_invalid_state_events = bool(args.diagnostic_callbacks)
    args.boozer_least_squares_algorithm_explicit = (
        args.boozer_least_squares_algorithm is not None
    )
    args.stage2_bs_path_explicit = _cli_option_was_provided(
        raw_argv,
        "--stage2-bs-path",
        env_var="STAGE2_BS_PATH",
    )
    args.initial_step_scale_explicit = _cli_option_was_provided(
        raw_argv,
        "--initial-step-scale",
        env_var="OUTER_INITIAL_STEP_SCALE",
    )
    args.initial_step_maxiter_explicit = _cli_option_was_provided(
        raw_argv,
        "--initial-step-maxiter",
        env_var="OUTER_INITIAL_STEP_MAXITER",
    )
    args.optimizer_backend = resolve_single_stage_default_optimizer_backend(
        args.backend,
        args.optimizer_backend,
    )
    args.boozer_least_squares_algorithm = (
        resolve_single_stage_default_boozer_least_squares_algorithm(
            args.backend,
            args.optimizer_backend,
            args.boozer_optimizer_backend,
            args.boozer_least_squares_algorithm,
        )
    )
    args.outer_maxls = resolve_single_stage_outer_maxls(
        args.backend,
        args.optimizer_backend,
        args.outer_maxls,
        benchmark_mode=args.benchmark_mode,
    )
    args.target_lane_outer_initial_step_size = (
        resolve_target_lane_outer_initial_step_size(
            args.backend,
            args.optimizer_backend,
            args.target_lane_outer_initial_step_size,
            benchmark_mode=args.benchmark_mode,
        )
    )
    args.maxcor = resolve_single_stage_outer_maxcor(
        args.backend,
        args.optimizer_backend,
        args.maxcor,
    )
    args.target_lane_boozer_bfgs_tol = resolve_target_lane_boozer_bfgs_tol(
        args.backend,
        args.optimizer_backend,
        args.target_lane_boozer_bfgs_tol,
        benchmark_mode=args.benchmark_mode,
    )
    args.target_lane_boozer_bfgs_maxiter = resolve_target_lane_boozer_bfgs_maxiter(
        args.backend,
        args.optimizer_backend,
        args.target_lane_boozer_bfgs_maxiter,
        benchmark_mode=args.benchmark_mode,
    )
    args.target_lane_boozer_newton_tol = resolve_target_lane_boozer_newton_tol(
        args.backend,
        args.optimizer_backend,
        args.target_lane_boozer_newton_tol,
    )
    args.target_lane_boozer_newton_maxiter = resolve_target_lane_boozer_newton_maxiter(
        args.backend,
        args.optimizer_backend,
        args.target_lane_boozer_newton_maxiter,
    )
    if args.profile_target_lane_only:
        args.profile_target_lane = True
    if args.profile_target_lane_batch_size < 1:
        raise ValueError("--profile-target-lane-batch-size must be at least 1")
    if not (0.0 < args.initial_step_scale <= 1.0):
        raise ValueError("--initial-step-scale must be in (0, 1]")
    if args.initial_step_maxiter < 0:
        raise ValueError("--initial-step-maxiter must be non-negative")
    validate_single_stage_current_args(args)
    validate_single_stage_alm_formulation_args(args)
    if args.constraint_method == "alm":
        validate_alm_cli_args(args)
    if args.constraint_method == "alm" and (
        args.profile_target_lane
        or args.profile_target_lane_only
        or args.diagnose_target_lane_gradient
        or args.diagnose_target_lane_first_line_search
        or args.diagnose_target_lane_scaled_phase1
    ):
        raise ValueError(
            "target-lane profiling and diagnostic flags are only supported with "
            "--constraint-method=penalty"
        )
    return args


class BoozerResidualExact(Optimizable):
    r"""
    This term returns the Boozer residual penalty term

    .. math::
       J = \int_0^{1/n_{\text{fp}}} \int_0^1 \| \mathbf r \|^2 ~d\theta ~d\varphi + w (\text{label.J()-boozer_surface.constraint_weight})^2.

    where

    .. math::
        \mathbf r = \frac{1}{\|\mathbf B\|}[G\mathbf B_\text{BS}(\mathbf x) - ||\mathbf B_\text{BS}(\mathbf x)||^2  (\mathbf x_\varphi + \iota  \mathbf x_\theta)]

    """

    def __init__(self, boozer_surface, bs):
        Optimizable.__init__(self, depends_on=[boozer_surface])
        in_surface = boozer_surface.surface
        self.boozer_surface = boozer_surface

        # same number of points as on the solved surface
        nphis = in_surface.quadpoints_phi.size
        phis = np.linspace(0, 1.0 / in_surface.nfp, nphis * 4, endpoint=False)
        nthetas = in_surface.quadpoints_theta.size
        thetas = np.linspace(0, 1, nthetas * 4, endpoint=False)

        s = SurfaceXYZTensorFourier(
            mpol=in_surface.mpol,
            ntor=in_surface.ntor,
            stellsym=in_surface.stellsym,
            nfp=in_surface.nfp,
            quadpoints_phi=phis,
            quadpoints_theta=thetas,
        )
        s.set_dofs(in_surface.get_dofs())

        print("warning: constraint weight set to 0")
        self.constraint_weight = 0.0
        self.in_surface = in_surface
        self.surface = s
        self.biotsavart = bs
        self.recompute_bell()

    def J(self):
        """
        Return the value of the penalty function.
        """

        if self._J is None:
            self.compute()
        return self._J

    @derivative_dec
    def dJ(self):
        """
        Return the derivative of the penalty function with respect to the coil degrees of freedom.
        """

        if self._dJ is None:
            self.compute()
        return self._dJ

    def recompute_bell(self, parent=None):
        self._J = None
        self._dJ = None

    def _adjoint_coil_derivative(self, dJ_ds, *, iota, G):
        booz_surf = self.boozer_surface
        get_adjoint_runtime_state = getattr(booz_surf, "get_adjoint_runtime_state", None)
        if callable(get_adjoint_runtime_state):
            adjoint_state = get_adjoint_runtime_state()
            rhs = jnp.asarray(dJ_ds, dtype=jnp.float64)
            solve_with_status = getattr(
                adjoint_state,
                "solve_transpose_with_status",
                None,
            )
            if callable(solve_with_status):
                adjoint, success = solve_with_status(rhs)
                if not bool(np.asarray(success)):
                    raise RuntimeError(
                        "BoozerResidualExact adjoint solve failed on the "
                        f"runtime path ({adjoint_state.linearization_kind})."
                    )
            else:
                adjoint = adjoint_state.solve_transpose(rhs)
            projector = getattr(
                adjoint_state,
                "project_coil_adjoint_derivative",
                None,
            )
            if callable(projector):
                return projector(adjoint)
            adjoint_coil_derivative = Derivative({})
            for d_coil_array, coil_group_indices in adjoint_state.stream_group_vjps(
                adjoint
            ):
                adjoint_coil_derivative += (
                    self.biotsavart.coil_cotangents_to_derivative(
                        [d_coil_array],
                        [coil_group_indices],
                    )
                )
            return adjoint_coil_derivative

        linear_solve_factors = booz_surf.res.get("PLU")
        legacy_vjp = booz_surf.res.get("vjp")
        if linear_solve_factors is None or not callable(legacy_vjp):
            raise RuntimeError(
                "BoozerResidualExact requires either "
                "boozer_surface.get_adjoint_runtime_state() or the legacy "
                "PLU/vjp adjoint contract."
            )
        adjoint = forward_backward(*linear_solve_factors, dJ_ds)
        return legacy_vjp(adjoint, booz_surf, iota, G)

    def compute(self):
        if self.boozer_surface.need_to_run_code:
            res = self.boozer_surface.res
            res = self.boozer_surface.run_code(res["iota"], G=res["G"])

        self.surface.set_dofs(self.in_surface.get_dofs())
        self.biotsavart.set_points(self.surface.gamma().reshape((-1, 3)))

        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta

        # compute J
        surface = self.surface
        iota = self.boozer_surface.res["iota"]
        G = self.boozer_surface.res["G"]
        r, J = boozer_surface_residual(
            surface, iota, G, self.biotsavart, derivatives=1, weight_inv_modB=True
        )
        rtil = np.concatenate(
            (
                r / np.sqrt(num_points),
                [
                    np.sqrt(self.constraint_weight)
                    * (self.boozer_surface.label.J() - self.boozer_surface.targetlabel)
                ],
            )
        )
        self._J = 0.5 * np.sum(rtil**2)

        dJ_by_dB = self.dJ_by_dB()
        dJ_by_dcoils = self.biotsavart.B_vjp(dJ_by_dB)

        # dJ_diota, dJ_dG  to the end of dJ_ds are on the end
        dl = np.zeros((J.shape[1],))
        dlabel_dsurface = self.boozer_surface.label.dJ_by_dsurfacecoefficients()
        dl[: dlabel_dsurface.size] = dlabel_dsurface
        Jtil = np.concatenate(
            (J / np.sqrt(num_points), np.sqrt(self.constraint_weight) * dl[None, :]),
            axis=0,
        )
        dJ_ds = Jtil.T @ rtil
        adj_times_dg_dcoil = self._adjoint_coil_derivative(dJ_ds, iota=iota, G=G)
        self._dJ = dJ_by_dcoils - adj_times_dg_dcoil

    def dJ_by_dB(self):
        """
        Return the partial derivative of the objective with respect to the magnetic field
        """

        surface = self.surface
        nphi = self.surface.quadpoints_phi.size
        ntheta = self.surface.quadpoints_theta.size
        num_points = 3 * nphi * ntheta
        r, r_dB = boozer_surface_residual_dB(
            surface,
            self.boozer_surface.res["iota"],
            self.boozer_surface.res["G"],
            self.biotsavart,
            derivatives=0,
            weight_inv_modB=True,
        )

        r /= np.sqrt(num_points)
        r_dB /= np.sqrt(num_points)

        dJ_by_dB = r[:, None] * r_dB
        dJ_by_dB = np.sum(dJ_by_dB.reshape((-1, 3, 3)), axis=1)
        return dJ_by_dB


def initialize_boozer_surface(
    surf_prev,
    mpol,
    ntor,
    bs,
    vol_target,
    constraint_weight,
    iota,
    G0,
    backend="cpu",
    optimizer_backend=None,
    boozer_least_squares_algorithm=None,
    boozer_limited_memory=False,
    bfgs_tol_override=None,
    bfgs_maxiter_override=None,
    newton_tol_override=None,
    newton_maxiter_override=None,
    surface_dofs_override=None,
    iota_override=None,
    G_override=None,
    on_stage=None,
    timings_out=None,
):
    """
    This initializes the boozer surface, using either the boozer "exact" algorithm, or the boozer "least squares" algorithm

    surf_prev: Any instance of simsopt.geo.Surface. This is the initial guess for the boozer surface solver
    mpol: SurfaceXYZTensorFourier resolution (both toroidal and poloidal)
    bs: simsopt.field.BiotSavart or BiotSavartJAX instance
    vol_target: target volume to be enclosed by the boozer surface
    constraint_weight: Set to 1.0 to use Boozer least square, None to use Boozer exact
    iota: initial guess for iota value on the surface
    G0: Value of net current going through the torus hole
    backend: "cpu" or "jax"
    optimizer_backend: optional JAX inner optimizer selector recorded in metadata
    boozer_least_squares_algorithm: optional JAX Boozer LS algorithm override
    boozer_limited_memory: force the JAX Boozer LS solve through ondevice
        limited-memory routing without changing the default contract elsewhere
    bfgs_tol_override: optional first-stage least-squares tolerance override
        for JAX/ondevice Boozer initialization
    bfgs_maxiter_override: optional first-stage least-squares iteration cap
        override for JAX/ondevice Boozer initialization
    newton_tol_override: optional Newton polish tolerance override for
        JAX/ondevice Boozer initialization
    newton_maxiter_override: optional Newton polish iteration cap override
        for JAX/ondevice Boozer initialization
    surface_dofs_override: optional converged surface DOFs to reuse as the
        initial Boozer state instead of the fitted Stage 2 seed surface
    iota_override: optional solved iota warm start for the Boozer replay
    G_override: optional solved G warm start for the Boozer replay
    """

    timings = {} if timings_out is None else timings_out
    stage_marks = {}

    def emit_stage(label, **extra):
        if label in _TIMED_STAGE_LABELS:
            stage_marks[label] = _perf_counter_s()
        if on_stage is not None:
            on_stage(label, **extra)

    def build_jax_stage_options(**extra):
        options = dict(extra)
        if (
            backend == "jax"
            and on_stage is not None
            and jax_solver_stage_callback_supported()
        ):
            options["stage_callback"] = on_stage
        return options

    def resolve_boozer_warm_start():
        solve_iota = float(iota_override) if iota_override is not None else iota
        solve_G = float(G_override) if G_override is not None else G0
        solve_sdofs = (
            None
            if surface_dofs_override is None
            else (
                _as_jax_float64(surface_dofs_override)
                if backend == "jax"
                else np.asarray(surface_dofs_override, dtype=float)
            )
        )
        return solve_iota, solve_G, solve_sdofs

    def run_boozer_solve(boozer_surface, solve_iota, solve_G, solve_sdofs):
        if backend == "jax":
            return boozer_surface.run_code(solve_iota, solve_G, sdofs=solve_sdofs)
        return boozer_surface.run_code(solve_iota, solve_G)

    total_start_s = _perf_counter_s()
    fit_start_s = total_start_s
    initial_surface_dofs = (
        _as_jax_float64(surface_dofs_override)
        if surface_dofs_override is not None
        else _as_jax_float64(
            project_surface_dofs_to_resolution(
                surf_prev,
                mpol=mpol,
                ntor=ntor,
                quadpoints_phi=surf_prev.quadpoints_phi,
                quadpoints_theta=surf_prev.quadpoints_theta,
            )
        )
    )
    initial_surface_dofs_host = np.asarray(
        host_array(initial_surface_dofs), dtype=np.float64
    )

    def build_surface(*, quadpoints_phi, quadpoints_theta):
        if backend == "jax":
            return DeferredSurfaceXYZTensorFourier(
                mpol=mpol,
                ntor=ntor,
                nfp=surf_prev.nfp,
                stellsym=surf_prev.stellsym,
                quadpoints_theta=quadpoints_theta,
                quadpoints_phi=quadpoints_phi,
                dofs=initial_surface_dofs,
            )
        surface = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            nfp=surf_prev.nfp,
            stellsym=surf_prev.stellsym,
            quadpoints_theta=quadpoints_theta,
            quadpoints_phi=quadpoints_phi,
        )
        surface.set_dofs(initial_surface_dofs_host)
        return surface

    def build_volume_label(surface):
        if backend == "jax":
            return DeferredVolume(surface)
        return Volume(surface)

    surf = build_surface(
        quadpoints_theta=surf_prev.quadpoints_theta,
        quadpoints_phi=surf_prev.quadpoints_phi,
    )
    fit_end_s = _perf_counter_s()
    _record_timing(timings, "boozer_surface_fit_s", fit_start_s, fit_end_s)
    emit_stage("after_boozer_surface_fit")

    if backend == "jax":
        from simsopt.geo.boozersurface_jax import (
            BoozerSurfaceJAX,
            build_boozer_surface_runtime_state,
        )

        BoozerCls = BoozerSurfaceJAX
    else:
        BoozerCls = BoozerSurface

    solver_name = "JAX " if backend == "jax" else ""
    setup_start_s = _perf_counter_s()
    if constraint_weight is not None:
        print(f"Generating {solver_name}Boozer least squares surface...")
        vol = build_volume_label(surf)
        options = {"verbose": True}
        if backend == "jax":
            resolved_optimizer_backend = resolve_boozer_optimizer_backend(
                backend,
                "ondevice",
                optimizer_backend,
            )
            options["optimizer_backend"] = resolved_optimizer_backend
            if resolved_optimizer_backend == "ondevice":
                if boozer_limited_memory:
                    options["force_ondevice_limited_memory"] = True
                    options["materialize_dense_linearization"] = False
            if boozer_least_squares_algorithm is not None:
                options["least_squares_algorithm"] = boozer_least_squares_algorithm
            if bfgs_tol_override is not None:
                options["bfgs_tol"] = float(bfgs_tol_override)
            if bfgs_maxiter_override is not None:
                options["bfgs_maxiter"] = int(bfgs_maxiter_override)
            if newton_tol_override is not None:
                options["newton_tol"] = float(newton_tol_override)
            if newton_maxiter_override is not None:
                options["newton_maxiter"] = int(newton_maxiter_override)
            options.update(build_jax_stage_options())
        if backend == "jax":
            boozer_surface = BoozerCls(
                bs,
                surf,
                vol,
                vol_target,
                constraint_weight,
                options=options,
                surface_runtime_state=build_boozer_surface_runtime_state(surf),
            )
        else:
            boozer_surface = BoozerCls(
                bs,
                surf,
                vol,
                vol_target,
                constraint_weight,
                options=options,
            )
        emit_stage(
            "after_boozer_setup",
            boozer_type="ls",
            backend=backend,
        )
    else:
        print(f"Generating {solver_name}Boozer exact surface...")
        exact_quadpoints_theta = np.linspace(0, 1, 2 * mpol + 1, endpoint=False)
        exact_quadpoints_phi = np.linspace(
            0, 1.0 / surf.nfp, 2 * ntor + 1, endpoint=False
        )
        surf_exact = build_surface(
            quadpoints_theta=exact_quadpoints_theta,
            quadpoints_phi=exact_quadpoints_phi,
        )
        vol = build_volume_label(surf_exact)
        if backend == "jax":
            boozer_surface = BoozerCls(
                bs,
                surf_exact,
                vol,
                vol_target,
                None,
                options=build_jax_stage_options(verbose=True),
                surface_runtime_state=build_boozer_surface_runtime_state(surf_exact),
            )
        else:
            boozer_surface = BoozerCls(
                bs,
                surf_exact,
                vol,
                vol_target,
                None,
                options=build_jax_stage_options(verbose=True),
            )
        emit_stage(
            "after_boozer_setup",
            boozer_type="exact",
            backend=backend,
        )
    if (
        backend == "jax"
        and _supported_surface_self_intersection_inputs(boozer_surface.surface)
        is not None
    ):
        prewarm_start_s = _perf_counter_s()
        prewarm_supported_surface_self_intersection(boozer_surface.surface)
        _record_timing(
            timings,
            "jax_compile_prewarm_self_intersection_s",
            prewarm_start_s,
            _perf_counter_s(),
        )
    setup_end_s = _perf_counter_s()
    _record_timing(timings, "boozer_setup_s", setup_start_s, setup_end_s)

    # Run boozer surface algorithm
    solve_iota, solve_G, solve_sdofs = resolve_boozer_warm_start()
    emit_stage("before_boozer_solve")
    solve_start_s = _perf_counter_s()
    res = run_boozer_solve(boozer_surface, solve_iota, solve_G, solve_sdofs)
    solve_end_s = _perf_counter_s()
    _record_timing(timings, "boozer_solve_s", solve_start_s, solve_end_s)
    emit_stage(
        "after_boozer_solve",
        solve_success=host_bool(res["success"]),
        iterations=host_float(res["iter"]),
    )
    print(f"G0 from solve: {host_float(res['G'])}")
    print(f"iota from solve: {host_float(res['iota'])}")

    # Check if boozer algo is successful
    success1 = host_bool(res["success"])  # True if the boozer surface algo converged
    postprocess_start_s = _perf_counter_s()
    self_intersection_status = "not_evaluated"
    if success1:
        (
            self_intersecting,
            self_intersection_check_available,
        ) = evaluate_surface_self_intersection(
            boozer_surface.surface,
            require_supported_surface=_boozer_surface_requires_jax_supported_self_intersection(
                boozer_surface
            ),
        )
        success2 = not self_intersecting  # True if surface is not self intersecting
        self_intersection_status = str(self_intersecting)
    else:
        self_intersecting = False
        self_intersection_check_available = False
        success2 = True
    success = success1 and success2
    if success1 and not self_intersection_check_available:
        print(
            "Skipping surface self-intersection check because "
            "ground+bentley_ottmann or shapely is unavailable."
        )
    if not success:
        print(
            "Boozer initialization failed: "
            f"solve_success={success1}, "
            f"self_intersecting={self_intersection_status}, "
            f"volume={host_float(boozer_surface.surface.volume())}, "
            f"iota_guess={solve_iota}, "
            f"iota_solved={host_float(res['iota'])}"
        )
        raise RuntimeError("Something went wrong with the Boozer solve...")

    emit_stage(
        "after_boozer_postprocess",
        self_intersection_check_available=(
            "true" if self_intersection_check_available else "false"
        ),
    )
    postprocess_end_s = _perf_counter_s()
    _record_timing(
        timings,
        "boozer_postprocess_total_s",
        postprocess_start_s,
        postprocess_end_s,
    )
    _record_timing(timings, "boozer_total_s", total_start_s, postprocess_end_s)
    _record_prefixed_stage_timings(
        timings,
        stage_marks,
        prefix="boozer",
        solve_start_s=solve_start_s,
    )
    return boozer_surface


def normPlot(surf, bs, filename):
    """Plot normal magnetic field — delegates to shared norm_field_plot."""
    mean_abs_relBfinal_norm, _, _, _, _ = norm_field_plot(surf, bs, filename)
    return mean_abs_relBfinal_norm


def diagnostic_field(bs, bs_cpu_diag):
    """Use the CPU field object for artifact/diagnostic paths when available."""
    return bs_cpu_diag if bs_cpu_diag is not None else bs


def build_iota_objective(boozer_surface, iota_cls):
    """Create the backend-matched iota diagnostic/objective wrapper."""
    return iota_cls(boozer_surface)


def resolve_single_stage_iota_metric(
    boozer_surface,
    iota_cls,
    *,
    benchmark_mode,
):
    """Resolve the reported iota metric without redundant benchmark-mode replay."""
    if (
        benchmark_mode
        and boozer_surface.res is not None
        and "iota" in boozer_surface.res
    ):
        return host_float(boozer_surface.res["iota"])
    return host_float(build_iota_objective(boozer_surface, iota_cls).J())


def _resolved_single_stage_boozer_solved_state(boozer_surface):
    """Return the solved Boozer state using the explicit runtime contract when present."""
    get_solved_runtime_state = getattr(boozer_surface, "get_solved_runtime_state", None)
    if callable(get_solved_runtime_state):
        return get_solved_runtime_state()

    return types.SimpleNamespace(
        sdofs=boozer_surface.surface.x,
        iota=boozer_surface.res["iota"],
        G=boozer_surface.res["G"],
    )


def _clear_target_lane_reporting_cache(run_dict):
    """Clear cached target-lane reporting state from the shared run dict."""
    run_dict.pop("target_lane_reporting_metrics", None)
    run_dict.pop("target_lane_reporting_coil_dofs", None)
    run_dict.pop("target_lane_reporting_include_distance_metrics", None)


def _cache_target_lane_reporting_summary(
    run_dict,
    coil_dofs,
    accepted_step_summary,
    *,
    benchmark_mode,
):
    """Persist one accepted-step reporting summary in the array-native run state."""
    reporting_metrics = dict(accepted_step_summary["reporting_metrics"])
    hardware_status = reporting_metrics.get("hardware_status")
    if isinstance(hardware_status, dict):
        reporting_metrics["hardware_status"] = dict(hardware_status)
    run_dict["target_lane_reporting_metrics"] = reporting_metrics
    run_dict["target_lane_reporting_coil_dofs"] = host_array(
        _single_stage_optimizer_dofs_array(coil_dofs),
        dtype=np.float64,
    )
    run_dict["target_lane_reporting_include_distance_metrics"] = bool(
        not benchmark_mode
    )


def _require_cached_target_lane_reporting_metrics(
    run_dict,
    coil_dofs,
    *,
    benchmark_mode,
):
    """Return cached accepted-step reporting metrics when they match the final state."""
    if run_dict is None or not all(
        key in run_dict
        for key in (
            "target_lane_reporting_metrics",
            "target_lane_reporting_coil_dofs",
            "target_lane_reporting_include_distance_metrics",
        )
    ):
        raise RuntimeError(
            "Missing cached target-lane final reporting metrics for results.json. "
            "The JAX final result path must use the accepted-step reporting snapshot "
            "instead of rebuilding host objective wrappers."
        )
    cached_metrics = run_dict["target_lane_reporting_metrics"]
    cached_coil_dofs = run_dict["target_lane_reporting_coil_dofs"]
    include_distance_metrics = run_dict["target_lane_reporting_include_distance_metrics"]
    if (
        cached_metrics is None
        or cached_coil_dofs is None
        or include_distance_metrics is None
        or bool(include_distance_metrics) != bool(not benchmark_mode)
    ):
        raise RuntimeError(
            "Cached target-lane final reporting metrics do not match the "
            "results.json artifact policy."
        )

    final_coil_dofs = host_array(_single_stage_optimizer_dofs_array(coil_dofs))
    if final_coil_dofs.shape != cached_coil_dofs.shape or not np.array_equal(
        final_coil_dofs,
        cached_coil_dofs,
    ):
        raise RuntimeError(
            "Cached target-lane final reporting metrics do not match the final "
            "optimizer DOFs."
        )

    resolved_metrics = dict(cached_metrics)
    hardware_status = resolved_metrics.get("hardware_status")
    if isinstance(hardware_status, dict):
        resolved_metrics["hardware_status"] = dict(hardware_status)
    return resolved_metrics


def resolve_single_stage_final_penalty_metrics(
    *,
    use_target_lane,
    benchmark_mode,
    skip_outer_optimizer,
    boozer_surface,
    bs,
    iota_target,
    coil_dofs,
    outer_objective_config,
    success_filter,
    curvelength,
    j_non_qs,
    j_boozer_residual,
    j_iota,
    j_curve_length,
    j_curve_curve,
    j_curve_surface,
    j_surface_surface,
    j_curvature,
    cc_dist,
    cs_dist,
    ss_dist,
    curvature_threshold,
    run_dict=None,
    init_only=False,
    termination_message=None,
    optimizer_success=None,
):
    """Resolve final reported penalties/hardware metrics for one single-stage run."""
    del init_only, termination_message, optimizer_success
    benchmark_hardware_status = {
        "success": None,
        "violations": ["skipped_in_benchmark_mode"],
    }

    if use_target_lane:
        return _require_cached_target_lane_reporting_metrics(
            run_dict,
            coil_dofs,
            benchmark_mode=benchmark_mode,
        )

    max_curvature = float(np.max(j_curvature.curve.kappa()))
    final_curve_curve_min_dist = None
    final_curve_surface_min_dist = None
    final_surface_vessel_min_dist = None
    if benchmark_mode:
        final_hardware_status = benchmark_hardware_status
    else:
        final_curve_curve_min_dist = host_float(j_curve_curve.shortest_distance())
        final_curve_surface_min_dist = host_float(j_curve_surface.shortest_distance())
        final_surface_vessel_min_dist = host_float(
            j_surface_surface.shortest_distance()
        )
        final_hardware_status = evaluate_single_stage_hardware_constraints(
            final_curve_curve_min_dist,
            cc_dist,
            final_curve_surface_min_dist,
            cs_dist,
            final_surface_vessel_min_dist,
            ss_dist,
            max_curvature,
            curvature_threshold,
        )
    field_error, _, _, _, _, _ = norm_field_summary(
        boozer_surface.surface,
        bs,
    )
    return {
        "final_G": host_float(boozer_surface.res["G"]),
        "final_non_qs": host_float(j_non_qs.J()),
        "final_boozer_residual": host_float(j_boozer_residual.J()),
        "final_iota_penalty": host_float(j_iota.J()),
        "final_length_penalty": host_float(j_curve_length.J()),
        "final_curve_curve_penalty": host_float(j_curve_curve.J()),
        "final_curve_surface_penalty": host_float(j_curve_surface.J()),
        "final_surface_vessel_penalty": host_float(j_surface_surface.J()),
        "final_curvature_penalty": host_float(j_curvature.J()),
        "coil_length": host_float(curvelength.J()),
        "max_curvature": max_curvature,
        "field_error": host_float(field_error),
        "final_volume": host_float(boozer_surface.surface.volume()),
        "final_iota": host_float(boozer_surface.res["iota"]),
        "curve_curve_min_dist": final_curve_curve_min_dist,
        "curve_surface_min_dist": final_curve_surface_min_dist,
        "surface_vessel_min_dist": final_surface_vessel_min_dist,
        "hardware_status": final_hardware_status,
    }


def evaluate_single_stage_artifact_hardware_snapshot(
    *,
    curve_curve_min_dist,
    cc_dist,
    curve_surface_min_dist,
    cs_dist,
    surface_vessel_min_dist,
    ss_dist,
    max_curvature,
    curvature_threshold,
    coil_length,
    length_target,
    banana_current_A,
    banana_current_max_A,
    tf_current_A,
    tf_current_limit_A,
):
    snapshot = {
        "curve_curve_min_dist": _optional_host_float(curve_curve_min_dist),
        "cc_dist": _optional_host_float(cc_dist),
        "curve_surface_min_dist": _optional_host_float(curve_surface_min_dist),
        "cs_dist": _optional_host_float(cs_dist),
        "surface_vessel_min_dist": _optional_host_float(surface_vessel_min_dist),
        "ss_dist": _optional_host_float(ss_dist),
        "max_curvature": _optional_host_float(max_curvature),
        "curvature_threshold": _optional_host_float(curvature_threshold),
        "coil_length": _optional_host_float(coil_length),
        "length_target": _optional_host_float(length_target),
        "banana_current_A": _optional_host_float(banana_current_A),
        "banana_current_max_A": _optional_host_float(banana_current_max_A),
        "tf_current_A": _optional_host_float(tf_current_A),
        "tf_current_limit_A": _optional_host_float(tf_current_limit_A),
    }
    threshold_overrides = build_threshold_overrides(
        (
            ("coil_coil_spacing", snapshot["cc_dist"]),
            ("coil_surface_spacing", snapshot["cs_dist"]),
            ("surface_vessel_spacing", snapshot["ss_dist"]),
            ("max_curvature", snapshot["curvature_threshold"]),
            ("coil_length", snapshot["length_target"]),
            ("banana_current", snapshot["banana_current_max_A"]),
            ("tf_current", snapshot["tf_current_limit_A"]),
        )
    )
    measured_values = {
        "coil_coil_spacing": snapshot["curve_curve_min_dist"],
        "coil_surface_spacing": snapshot["curve_surface_min_dist"],
        "surface_vessel_spacing": snapshot["surface_vessel_min_dist"],
        "max_curvature": snapshot["max_curvature"],
        "coil_length": snapshot["coil_length"],
        "banana_current": snapshot["banana_current_A"],
        "tf_current": snapshot["tf_current_A"],
    }
    artifact_hardware_status = build_hardware_constraint_status(
        measured_values,
        applies_to="artifact",
        threshold_overrides=threshold_overrides,
    )
    return {**snapshot, "artifact_hardware_status": artifact_hardware_status}


def resolve_single_stage_final_banana_current_A(
    *,
    use_target_lane,
    final_metrics,
    banana_current,
):
    """Resolve final banana current from target-lane metrics or restored host state."""
    if use_target_lane:
        return final_metrics["banana_current_A"]
    return banana_current.get_value()


def _failed_single_stage_alm_evaluation(
    run_dict,
    constraint_names,
    *,
    failure_penalty,
):
    objective_value = float(run_dict["J"]) + float(failure_penalty)
    grad = host_array(run_dict["dJ"], dtype=np.float64).copy()
    max_violation = max(abs(objective_value), 1.0)
    feasibility_values = np.full(len(constraint_names), max_violation, dtype=float)
    zero_grad = np.zeros_like(grad)
    return {
        "total": float(objective_value),
        "grad": grad,
        "physics_total": float(objective_value),
        "base_total": float(objective_value),
        "constraint_names": list(constraint_names),
        "constraint_values": feasibility_values.copy(),
        "dual_update_values": feasibility_values.copy(),
        "feasibility_values": feasibility_values.copy(),
        "max_feasibility_violation": float(max_violation),
        "constraint_grads": [zero_grad.copy() for _ in constraint_names],
        "constraint_activity_tolerances": np.zeros(
            len(constraint_names),
            dtype=float,
        ),
        "metric_grad": grad,
        "metric_stationarity_norm": float(np.linalg.norm(grad)),
        "J_QS": float(run_dict["J"]),
        "J_Boozer": float(run_dict["J"]),
        "J_iota": 0.0,
        "J_len": 0.0,
    }


def evaluate_single_stage_alm_problem(
    dofs,
    *,
    run_dict,
    boozer_surface,
    JF,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    curvelength,
    JCurveLength,
    LENGTH_WEIGHT,
    JCurveCurve,
    JCurveSurface,
    JCurvature,
    JSurfSurf,
    curves,
    banana_curve,
    vessel_surface,
    curve_curve_distance,
    curve_surface_distance,
    surface_vessel_distance,
    curvature_threshold,
    distance_smoothing,
    curvature_smoothing,
    constraint_names,
    alm_formulation,
    qs_threshold,
    boozer_threshold,
    iota_penalty_threshold,
    length_penalty_threshold,
    banana_current,
    banana_current_threshold,
    tf_current_A,
    tf_current_limit_A,
    coil_length_threshold,
    multipliers,
    penalty,
):
    candidate_x = _single_stage_optimizer_dofs_array(dofs)
    JF.x = candidate_x
    uses_legacy_warm_start = not _boozer_surface_supports_explicit_surface_warm_start(
        boozer_surface
    )
    if uses_legacy_warm_start:
        _restore_cpu_boozer_state(boozer_surface, run_dict)
        boozer_surface.run_code(run_dict["iota"], run_dict["G"])
    else:
        boozer_surface.run_code(
            run_dict["iota"],
            run_dict["G"],
            sdofs=run_dict["sdofs"],
        )

    success_solve = host_bool(boozer_surface.res["success"])
    is_intersecting = update_self_intersection_status(
        run_dict,
        boozer_surface.surface,
        require_supported_surface=_boozer_surface_requires_jax_supported_self_intersection(
            boozer_surface
        ),
    )
    if not success_solve or is_intersecting:
        failure_penalty, failure_summary = compute_single_stage_failure_penalty(
            candidate_x,
            run_dict,
            boozer_surface,
            success_solve=success_solve,
            is_intersecting=is_intersecting,
            hardware_status=None,
        )
        run_dict["last_candidate_failure"] = failure_summary
        run_dict["trial_hardware_status"] = None
        if uses_legacy_warm_start:
            _restore_cpu_boozer_state(boozer_surface, run_dict)
        return _failed_single_stage_alm_evaluation(
            run_dict,
            constraint_names,
            failure_penalty=failure_penalty,
        )

    evaluation = evaluate_alm_objective(
        np.ones(len(nonQSs), dtype=float),
        nonQSs,
        brs,
        RES_WEIGHT,
        Jiota,
        IOTAS_WEIGHT,
        JCurveLength,
        LENGTH_WEIGHT,
        JCurveCurve,
        JCurveSurface,
        JCurvature,
        np.asarray(multipliers, dtype=float),
        float(penalty),
        objective_optimizable=JF,
        curves=curves,
        curve_curve_min_distance=curve_curve_distance,
        outer_surface=boozer_surface.surface,
        curve_surface_min_distance=curve_surface_distance,
        banana_curve=banana_curve,
        curvature_threshold=curvature_threshold,
        distance_smoothing=distance_smoothing,
        curvature_smoothing=curvature_smoothing,
        constraint_names=constraint_names,
        curve_curve_constraint_fn=smooth_min_curve_curve_signed_constraint,
        curve_surface_constraint_fn=smooth_min_curve_surface_signed_constraint,
        curvature_constraint_fn=smooth_max_curvature_signed_constraint,
        JSurfSurf=JSurfSurf,
        vessel_surface=vessel_surface,
        surface_surface_min_distance=surface_vessel_distance,
        surface_surface_constraint_fn=smooth_min_surface_surface_signed_constraint,
        alm_formulation=alm_formulation,
        qs_threshold=qs_threshold,
        boozer_threshold=boozer_threshold,
        iota_penalty_threshold=iota_penalty_threshold,
        length_penalty_threshold=length_penalty_threshold,
        coil_length_objective=curvelength,
        coil_length_threshold=coil_length_threshold,
        banana_current=banana_current,
        banana_current_threshold=banana_current_threshold,
    )
    trial_hardware_snapshot = evaluate_single_stage_artifact_hardware_snapshot(
        curve_curve_min_dist=JCurveCurve.shortest_distance(),
        cc_dist=curve_curve_distance,
        curve_surface_min_dist=JCurveSurface.shortest_distance(),
        cs_dist=curve_surface_distance,
        surface_vessel_min_dist=(
            None if JSurfSurf is None else JSurfSurf.shortest_distance()
        ),
        ss_dist=surface_vessel_distance,
        max_curvature=_host_curve_max_curvature(banana_curve),
        curvature_threshold=curvature_threshold,
        coil_length=curvelength.J(),
        length_target=coil_length_threshold,
        banana_current_A=banana_current.get_value(),
        banana_current_max_A=banana_current_threshold,
        tf_current_A=tf_current_A,
        tf_current_limit_A=tf_current_limit_A,
    )
    run_dict["trial_hardware_status"] = trial_hardware_snapshot[
        "artifact_hardware_status"
    ]
    return evaluation


def surface_self_intersection_check_available(surface=None):
    """Return whether the optional surface self-intersection backend is present."""
    if (
        surface is not None
        and _supported_surface_self_intersection_inputs(surface) is not None
    ):
        return True
    has_ground = (
        surface_module.get_context is not None
        and surface_module.contour_self_intersects is not None
    )
    return has_ground or getattr(surface_module, "LineString", None) is not None


def _surface_rzfourier_dof_count(*, mpol, ntor, stellsym):
    rc_count = (ntor + 1) + mpol * (2 * ntor + 1)
    tail_count = ntor + mpol * (2 * ntor + 1)
    if stellsym:
        return rc_count + tail_count
    return 2 * rc_count + 2 * tail_count


def _supported_surface_self_intersection_inputs(surface):
    deferred_surface_class = getattr(surface, "deferred_surface_class", None)
    surface_type_name = type(surface).__name__
    if deferred_surface_class == "SurfaceXYZTensorFourier":
        surface_kind = "xyztensorfourier"
    elif surface_type_name == "SurfaceXYZTensorFourier":
        surface_kind = "xyztensorfourier"
    elif surface_type_name == "SurfaceRZFourier":
        surface_kind = "rzfourier"
    else:
        return None

    surface_dofs_getter = getattr(surface, "get_dofs", None)
    if surface_dofs_getter is None or not callable(surface_dofs_getter):
        return None

    surface_dofs = _as_jax_float64(surface_dofs_getter()).reshape(-1)
    mpol = int(surface.mpol)
    ntor = int(surface.ntor)
    stellsym = bool(surface.stellsym)
    scatter_indices = None
    if surface_kind == "xyztensorfourier":
        scatter_indices = (
            _as_jax_int32(stellsym_scatter_indices(mpol, ntor))
            if stellsym
            else None
        )
        expected_dofs = (
            int(scatter_indices.size)
            if stellsym
            else 3 * (2 * mpol + 1) * (2 * ntor + 1)
        )
    else:
        expected_dofs = _surface_rzfourier_dof_count(
            mpol=mpol,
            ntor=ntor,
            stellsym=stellsym,
        )
    if int(surface_dofs.size) != expected_dofs:
        return None

    return {
        "surface_kind": surface_kind,
        "dofs": surface_dofs,
        "mpol": mpol,
        "ntor": ntor,
        "nfp": int(surface.nfp),
        "stellsym": stellsym,
        "scatter_indices": scatter_indices,
        "quadpoints_theta": _as_jax_float64(surface.quadpoints_theta).reshape(-1),
    }


@jax.jit(
    static_argnames=(
        "mpol",
        "ntor",
        "nfp",
        "stellsym",
        "surface_kind",
    )
)
def _surface_phi0_cross_section_from_supported_dofs(
    surface_dofs,
    quadpoints_theta,
    scatter_indices,
    *,
    mpol,
    ntor,
    nfp,
    stellsym,
    surface_kind,
):
    thetas = _as_runtime_float64(quadpoints_theta, reference=surface_dofs).reshape(-1)
    zero_varphi = _as_runtime_float64(0.0, reference=thetas) * thetas

    if surface_kind == "rzfourier":
        return surfrz_gamma_lin(
            zero_varphi,
            thetas,
            mpol,
            ntor,
            surface_dofs,
            nfp,
            stellsym,
        )

    gamma_zero = surface_gamma_lin_from_dofs(
        surface_dofs,
        zero_varphi,
        thetas,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
    )
    two_pi = _as_runtime_float64(2.0 * np.pi, reference=gamma_zero)
    phi0 = jnp.arctan2(gamma_zero[:, 1], gamma_zero[:, 0]) / two_pi
    target_phi = _as_runtime_float64(0.0, reference=phi0)
    target_phi = target_phi - phi0
    target_phi = target_phi + jnp.ceil(-target_phi)

    def shifted_cylindrical_angle(varphi_in):
        gamma = surface_gamma_lin_from_dofs(
            surface_dofs,
            varphi_in,
            thetas,
            mpol,
            ntor,
            nfp,
            stellsym,
            scatter_indices,
        )
        angle = jnp.arctan2(gamma[:, 1], gamma[:, 0]) / two_pi - phi0
        return angle + jnp.ceil(-angle)

    def bisection_step(_iteration, state):
        lower, upper, lower_phi, upper_phi = state
        middle = 0.5 * (lower + upper)
        middle_phi = shifted_cylindrical_angle(middle)
        root_in_lower_interval = (
            (middle_phi - target_phi) * (upper_phi - target_phi)
        ) > _as_runtime_float64(0.0, reference=middle_phi)
        next_lower = jnp.where(root_in_lower_interval, lower, middle)
        next_upper = jnp.where(root_in_lower_interval, middle, upper)
        next_lower_phi = jnp.where(root_in_lower_interval, lower_phi, middle_phi)
        next_upper_phi = jnp.where(root_in_lower_interval, middle_phi, upper_phi)
        return next_lower, next_upper, next_lower_phi, next_upper_phi

    initial_lower = _as_runtime_float64(0.0, reference=thetas) * thetas
    initial_upper = initial_lower + _as_runtime_float64(1.0, reference=thetas)
    lower, upper, _lower_phi, _upper_phi = jax.lax.fori_loop(
        0,
        _SURFACE_SELF_INTERSECTION_BISECTION_STEPS,
        bisection_step,
        (initial_lower, initial_upper, initial_lower, initial_upper),
    )
    return surface_gamma_lin_from_dofs(
        surface_dofs,
        0.5 * (lower + upper),
        thetas,
        mpol,
        ntor,
        nfp,
        stellsym,
        scatter_indices,
    )


def _evaluate_supported_surface_self_intersection(surface):
    supported_inputs = _supported_surface_self_intersection_inputs(surface)
    if supported_inputs is None:
        return None

    return bool(
        host_bool(
            _supported_surface_self_intersection_flag_from_dofs(
                supported_inputs["dofs"],
                supported_inputs["quadpoints_theta"],
                mpol=supported_inputs["mpol"],
                ntor=supported_inputs["ntor"],
                nfp=supported_inputs["nfp"],
                stellsym=supported_inputs["stellsym"],
                surface_kind=supported_inputs["surface_kind"],
                scatter_indices=supported_inputs["scatter_indices"],
            )
        )
    )


def prewarm_supported_surface_self_intersection(surface):
    supported_intersection = _evaluate_supported_surface_self_intersection(surface)
    if supported_intersection is None:
        raise TypeError(_JAX_SELF_INTERSECTION_UNSUPPORTED_MESSAGE)
    return supported_intersection


def _supported_surface_self_intersection_flag_from_dofs(
    surface_dofs,
    quadpoints_theta,
    *,
    mpol,
    ntor,
    nfp,
    stellsym,
    surface_kind,
    scatter_indices,
):
    cross_section = _surface_phi0_cross_section_from_supported_dofs(
        surface_dofs,
        quadpoints_theta,
        scatter_indices,
        mpol=mpol,
        ntor=ntor,
        nfp=nfp,
        stellsym=stellsym,
        surface_kind=surface_kind,
    )
    radial_components = jnp.take(
        cross_section,
        _as_jax_int32(np.array([0, 1], dtype=np.int32)),
        axis=1,
    )
    vertical_component = jnp.reshape(
        jnp.take(
            cross_section,
            _as_jax_int32(np.array([2], dtype=np.int32)),
            axis=1,
        ),
        (cross_section.shape[0],),
    )
    radial_height_curve = jnp.stack(
        (
            jnp.linalg.norm(radial_components, axis=1),
            vertical_component,
            vertical_component * _as_runtime_float64(0.0, reference=vertical_component),
        ),
        axis=1,
    )
    return closed_curve_self_intersection_summary(
        radial_height_curve,
        neighbor_skip=1,
        tolerance_factor=_SURFACE_SELF_INTERSECTION_TOLERANCE_FACTOR,
    )[3]


def build_single_stage_target_lane_self_intersection_success_filter(
    boozer_surface,
    bs,
):
    """Build a pure-JAX self-intersection rejector for the native ALM lane."""
    import jax.numpy as jnp

    from simsopt.jax_core.field import coil_set_spec_from_dof_extraction_spec

    supported_inputs = _supported_surface_self_intersection_inputs(
        boozer_surface.surface
    )
    if supported_inputs is None:
        return None

    surface_intersection_kwargs = {
        "quadpoints_theta": _hostify_target_lane_constant_tree(
            supported_inputs["quadpoints_theta"]
        ),
        "mpol": int(supported_inputs["mpol"]),
        "ntor": int(supported_inputs["ntor"]),
        "nfp": int(supported_inputs["nfp"]),
        "stellsym": bool(supported_inputs["stellsym"]),
        "surface_kind": supported_inputs["surface_kind"],
        "scatter_indices": _hostify_target_lane_constant_tree(
            supported_inputs["scatter_indices"]
        ),
    }

    optimize_G = boozer_surface.res.get("G") is not None
    coil_dof_extraction_spec = (
        None
        if optimize_G
        else _hostify_target_lane_constant_tree(bs.coil_dof_extraction_spec())
    )
    success_filter_signature = _target_lane_success_filter_cache_signature(
        {
            **surface_intersection_kwargs,
            "optimize_G": bool(optimize_G),
            "coil_dof_extraction_spec": coil_dof_extraction_spec,
        }
    )

    def success_filter(coil_dofs, solved_x):
        coil_set_spec = None
        if not optimize_G:
            coil_set_spec = coil_set_spec_from_dof_extraction_spec(
                coil_dof_extraction_spec,
                coil_dofs,
            )
        sdofs, _iota, _G = boozer_surface._unpack_decision_vector_jax(
            solved_x,
            optimize_G,
            coil_set_spec=coil_set_spec,
        )
        return jnp.logical_not(
            _supported_surface_self_intersection_flag_from_dofs(
                sdofs,
                **surface_intersection_kwargs,
            )
        )

    success_filter._traceable_runtime_cache_signature = (
        "single-stage-target-lane-self-intersection-success-filter",
        success_filter_signature,
    )
    return success_filter


def evaluate_surface_self_intersection(surface, *, require_supported_surface=False):
    """Return (intersecting, check_available) for a SIMSOPT surface."""
    supported_intersection = _evaluate_supported_surface_self_intersection(surface)
    if supported_intersection is not None:
        return supported_intersection, True
    if require_supported_surface:
        raise TypeError(_JAX_SELF_INTERSECTION_UNSUPPORTED_MESSAGE)
    check_available = surface_self_intersection_check_available()
    if not check_available:
        return False, False
    return bool(surface.is_self_intersecting()), True


def update_self_intersection_status(run_dict, surface, *, require_supported_surface=False):
    """Refresh self-intersection status in the shared run-state dictionary."""
    (
        run_dict["intersecting"],
        run_dict["self_intersection_check_available"],
    ) = evaluate_surface_self_intersection(
        surface,
        require_supported_surface=require_supported_surface,
    )
    return run_dict["intersecting"]


def get_jax_surface_objective_classes():
    """Load the JAX single-stage objective wrappers on demand."""
    from simsopt.geo.surfaceobjectives_jax import (
        BoozerResidualJAX,
        IotasJAX,
        NonQuasiSymmetricRatioJAX,
    )

    return BoozerResidualJAX, IotasJAX, NonQuasiSymmetricRatioJAX


def get_traceable_single_stage_objective_builder():
    """Load the pure single-stage JAX target objective on demand."""
    from simsopt.geo.surfaceobjectives_jax import make_traceable_objective

    return make_traceable_objective


def get_traceable_single_stage_runtime_bundle_builder():
    """Load the shared single-stage JAX target-lane runtime bundle on demand."""
    from simsopt.geo.surfaceobjectives_jax import (
        make_traceable_objective_runtime_bundle,
    )

    return make_traceable_objective_runtime_bundle


def get_traceable_single_stage_seeded_value_and_grad_builder():
    """Load the optimizer-only seeded target-lane value/gradient helper."""
    from simsopt.geo.surfaceobjectives_jax import (
        make_traceable_objective_seeded_value_and_grad,
    )

    return make_traceable_objective_seeded_value_and_grad


def build_traceable_single_stage_seeded_value_and_grad(
    boozer_surface,
    bs,
    iota_target,
    *,
    outer_objective_config,
    success_filter,
):
    """Build the optimizer-seeded target-lane fused objective."""
    return get_traceable_single_stage_seeded_value_and_grad_builder()(
        boozer_surface,
        bs,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )


def get_traceable_single_stage_alm_runtime_bundle_builder():
    """Load the pure single-stage ALM runtime bundle on demand."""
    from simsopt.geo.surfaceobjectives_jax import (
        make_traceable_single_stage_alm_runtime_bundle,
    )

    return make_traceable_single_stage_alm_runtime_bundle


def get_traceable_single_stage_runtime_diagnostic_builder():
    """Load the compact target-lane baseline diagnosis helper on demand."""
    from simsopt.geo.surfaceobjectives_jax import diagnose_traceable_objective_runtime

    return diagnose_traceable_objective_runtime


def build_single_stage_target_lane_accepted_step_sync(
    boozer_surface,
    bs,
    iota_target,
    *,
    outer_objective_config,
    success_filter,
):
    """Build the array-native accepted-step sync used on the target lane.

    The returned callable refreshes ``run_dict`` from immutable coil/surface
    state only. It intentionally does not touch ``boozer_surface.surface`` or
    any other mutable host-only diagnostic graph state.
    """
    runtime_bundle = get_traceable_single_stage_runtime_bundle_builder()(
        boozer_surface,
        bs,
        iota_target,
        include_profile_suite=False,
        include_host_wrappers=False,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    reporting_metrics_fn = runtime_bundle["reporting_metrics"]
    value_and_grad = runtime_bundle["value_and_grad"]
    forward_result_fn = runtime_bundle.get("forward_result")

    def accepted_step_solve_result(run_dict, coil_dofs):
        if forward_result_fn is None:
            return boozer_surface.run_code_traceable(
                bs.coil_set_spec_from_dofs(coil_dofs),
                _as_jax_float64(run_dict["sdofs"]),
                _as_jax_float64(run_dict["iota"]),
                _as_jax_float64(run_dict["G"]),
            )
        forward_result = forward_result_fn(coil_dofs)
        solve_success = forward_result.get(
            "primal_success",
            forward_result["success"],
        )
        return {
            "success": solve_success,
            "sdofs": forward_result["sdofs"],
            "iota": forward_result["iota"],
            "G": forward_result["G"],
        }

    def sync(run_dict, coil_dofs, *, benchmark_mode, update_run_state=True):
        coil_dofs = _as_jax_float64(_single_stage_optimizer_dofs_array(coil_dofs))
        solve_result = accepted_step_solve_result(run_dict, coil_dofs)
        if not host_bool(solve_result["success"]):
            raise RuntimeError(
                "target-lane accepted-step replay failed while refreshing "
                "single-stage array-native state."
            )

        include_distance_metrics = not benchmark_mode
        traceable_reporting_metrics = reporting_metrics_fn(
            coil_dofs,
            include_distance_metrics=include_distance_metrics,
        )
        reporting_metrics = dict(
            _hostify_traceable_reporting_metrics(
                traceable_reporting_metrics,
                include_distance_metrics=include_distance_metrics,
            )
        )
        if benchmark_mode:
            hardware_status = {
                "success": None,
                "violations": ["skipped_in_benchmark_mode"],
            }
        else:
            hardware_status = _hostify_single_stage_hardware_constraints(
                evaluate_single_stage_hardware_constraints_pure(
                    traceable_reporting_metrics["curve_curve_min_dist"],
                    CC_DIST,
                    traceable_reporting_metrics["curve_surface_min_dist"],
                    CS_DIST,
                    traceable_reporting_metrics["surface_vessel_min_dist"],
                    SS_DIST,
                    traceable_reporting_metrics["max_curvature"],
                    CURVATURE_THRESHOLD,
                )
            )
        reporting_metrics["hardware_status"] = hardware_status
        if update_run_state:
            objective_value, objective_grad = value_and_grad(coil_dofs)
            objective_value = host_float(objective_value)
            objective_grad = host_array(objective_grad, dtype=np.float64)
        else:
            objective_value = host_float(
                total_single_stage_objective_from_traceable_reporting_metrics(
                    traceable_reporting_metrics
                )
            )
            objective_grad = None
        accepted_step_summary = {
            "objective_value": objective_value,
            "reporting_metrics": reporting_metrics,
        }
        if update_run_state:
            snapshot_accepted_step_state_from_values(
                run_dict,
                sdofs=solve_result["sdofs"],
                iota=solve_result["iota"],
                G=solve_result["G"],
                objective_value=objective_value,
                objective_grad=objective_grad,
                store_objective_grad=True,
            )
            run_dict["hardware_constraint_status"] = hardware_status
            record_single_stage_local_incumbent(
                run_dict,
                stage=f"iter_{run_dict.get('it', 0)}",
            )
        _cache_target_lane_reporting_summary(
            run_dict,
            coil_dofs,
            accepted_step_summary,
            benchmark_mode=benchmark_mode,
        )
        return accepted_step_summary

    return sync


def configure_single_stage_target_lane_accepted_step_sync(
    adapter,
    boozer_surface,
    bs,
    iota_target,
    *,
    use_target_lane,
    outer_objective_config,
    success_filter,
):
    """Install the array-native accepted-step sync and disable CPU reevaluation."""
    if not use_target_lane:
        return
    adapter.accepted_step_state_sync = build_single_stage_target_lane_accepted_step_sync(
        boozer_surface,
        bs,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    adapter.reevaluate_before_accept = False


def cache_single_stage_target_lane_reporting_snapshot(
    adapter,
    run_dict,
    coil_dofs,
    *,
    benchmark_mode,
):
    """Prime final reporting from the pure target-lane runtime without host restore."""
    return adapter.accepted_step_state_sync(
        run_dict,
        coil_dofs,
        benchmark_mode=benchmark_mode,
        update_run_state=False,
    )


def cache_single_stage_target_lane_init_reporting_snapshot(
    *,
    boozer_surface,
    bs,
    banana_curve,
    vessel_surface,
    iota_target,
    run_dict,
    coil_dofs,
    benchmark_mode,
    disable_success_filter,
    length_target,
    cc_dist,
    cc_weight,
    cs_dist,
    cs_weight,
    ss_dist,
    surf_dist_weight,
    residual_weight,
    iota_weight,
    length_weight,
    curvature_threshold,
    curvature_weight,
):
    """Prime target-lane final reporting for init-only runs."""
    del disable_success_filter
    target_lane_success_filter = None
    target_lane_outer_objective_config = build_target_lane_outer_objective_config(
        boozer_surface,
        bs,
        banana_curve,
        vessel_surface,
        non_qs_weight=1.0,
        residual_weight=residual_weight,
        iota_weight=iota_weight,
        length_weight=length_weight,
        length_target=length_target,
        curve_curve_threshold=cc_dist,
        curve_curve_weight=cc_weight,
        curve_surface_threshold=cs_dist,
        curve_surface_weight=cs_weight,
        surface_vessel_threshold=ss_dist,
        surface_vessel_weight=surf_dist_weight,
        curvature_threshold=curvature_threshold,
        curvature_weight=curvature_weight,
    )
    sync_target_lane_reporting = build_single_stage_target_lane_accepted_step_sync(
        boozer_surface,
        bs,
        iota_target,
        outer_objective_config=target_lane_outer_objective_config,
        success_filter=target_lane_success_filter,
    )
    return sync_target_lane_reporting(
        run_dict,
        coil_dofs,
        benchmark_mode=benchmark_mode,
        update_run_state=False,
    )


def log_single_stage_target_lane_accepted_step(
    run_dict,
    accepted_step_summary,
    log_path,
):
    """Log the pure target-lane accepted-step summary without host diagnostics."""
    reporting_metrics = accepted_step_summary["reporting_metrics"]
    hardware_status = reporting_metrics["hardware_status"]
    width = 35
    buffer = io.StringIO()
    print("=" * 70, file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(
        f"{'Objective J':{width}} = {accepted_step_summary['objective_value']:.6e}",
        file=buffer,
    )
    print(
        f"{'Iotas (actual)':{width}} = {reporting_metrics['final_iota']:.4f}",
        file=buffer,
    )
    print(
        f"{'Volume':{width}} = {reporting_metrics['final_volume']:.4f}",
        file=buffer,
    )
    print(
        f"{'Curve Length':{width}} = {reporting_metrics['coil_length']:.6e}",
        file=buffer,
    )
    print(
        f"{'Max Curvature':{width}} = {reporting_metrics['max_curvature']:.6e}",
        file=buffer,
    )
    if reporting_metrics["curve_curve_min_dist"] is not None:
        print(
            f"{'Curve-Curve Min Dist':{width}} = "
            f"{reporting_metrics['curve_curve_min_dist']:.6e}",
            file=buffer,
        )
        print(
            f"{'Curve-Surface Min Dist':{width}} = "
            f"{reporting_metrics['curve_surface_min_dist']:.6e}",
            file=buffer,
        )
        print(
            f"{'Surface-Vessel Min Dist':{width}} = "
            f"{reporting_metrics['surface_vessel_min_dist']:.6e}",
            file=buffer,
        )
    print(
        f"{'Host-only diagnostics':{width}} = deferred to final postprocess",
        file=buffer,
    )
    print(
        f"{'Hardware Constraints OK':{width}} = {hardware_status['success']}",
        file=buffer,
    )
    if hardware_status["violations"]:
        print(
            f"{'Hardware Violations':{width}} = {hardware_status['violations']}",
            file=buffer,
        )
    print("=" * 70, file=buffer)

    output_str = buffer.getvalue()
    buffer.close()
    logger.info("%s", output_str)

    with open(log_path, "a") as f:
        f.write(output_str + "\n")

    run_dict["it"] += 1


def _resolve_single_stage_banana_curve_index(bs, banana_curve):
    try:
        return next(
            coil_index
            for coil_index, coil in enumerate(bs.coils)
            if coil.curve is banana_curve
        )
    except StopIteration as exc:
        raise RuntimeError(
            "single-stage target-lane setup could not locate the banana curve "
            "in bs.coils."
        ) from exc


def build_traceable_single_stage_outer_objective_config(
    boozer_surface,
    bs,
    banana_curve,
    vessel_surface,
    *,
    non_qs_weight,
    residual_weight,
    iota_weight,
    length_weight,
    length_target,
    curve_curve_weight,
    curve_curve_threshold,
    curve_surface_weight,
    curve_surface_threshold,
    surface_vessel_weight,
    surface_vessel_threshold,
    curvature_weight,
    curvature_threshold,
    curvature_p_norm=2.0,
):
    """Build the immutable target-lane config for the full single-stage objective."""
    from simsopt.jax_core.surface_rzfourier import (
        surface_rz_fourier_gamma_from_spec,
    )

    surface = boozer_surface.surface
    non_qs_sdim = 20
    banana_curve_index = _resolve_single_stage_banana_curve_index(bs, banana_curve)
    return {
        "non_qs_weight": float(non_qs_weight),
        "non_qs_quadpoints_phi": np.linspace(
            0.0,
            1.0 / surface.nfp,
            2 * non_qs_sdim,
            endpoint=False,
            dtype=np.float64,
        ),
        "non_qs_quadpoints_theta": np.linspace(
            0.0,
            1.0,
            2 * non_qs_sdim,
            endpoint=False,
            dtype=np.float64,
        ),
        "non_qs_axis": 0,
        "residual_weight": float(residual_weight),
        "iota_weight": float(iota_weight),
        "length_weight": float(length_weight),
        "length_target": float(length_target),
        "curve_curve_weight": float(curve_curve_weight),
        "curve_curve_threshold": float(curve_curve_threshold),
        "curve_surface_weight": float(curve_surface_weight),
        "curve_surface_threshold": float(curve_surface_threshold),
        "surface_vessel_weight": float(surface_vessel_weight),
        "surface_vessel_threshold": float(surface_vessel_threshold),
        "curvature_weight": float(curvature_weight),
        "curvature_threshold": float(curvature_threshold),
        "curvature_p_norm": float(curvature_p_norm),
        "banana_curve_index": int(banana_curve_index),
        "vessel_gamma": host_array(
            surface_rz_fourier_gamma_from_spec(vessel_surface.surface_spec()).reshape(
                (-1, 3)
            ),
            dtype=np.float64,
        ),
    }


def build_single_stage_target_lane_hardware_success_filter(
    boozer_surface,
    bs,
    banana_curve,
    vessel_surface,
    *,
    cc_dist,
    cs_dist,
    ss_dist,
    curvature_threshold,
):
    """Build the pure-JAX feasibility filter for the ondevice target lane."""
    import jax.numpy as jnp

    from simsopt.geo.curve import kappa_pure
    from simsopt.geo.boozer_residual_jax import _surface_geometry_from_dofs
    from simsopt.jax_core.curve_geometry import (
        curve_gamma_and_dash_from_spec,
        curve_geometry_from_spec,
    )
    from simsopt.jax_core.field import (
        coil_set_spec_from_dof_extraction_spec,
        coil_specs_from_dof_extraction_spec,
    )
    from simsopt.jax_core.surface_rzfourier import (
        surface_rz_fourier_gamma_from_spec,
    )

    banana_curve_index = _resolve_single_stage_banana_curve_index(bs, banana_curve)

    optimize_G = boozer_surface.res.get("G") is not None
    coil_dof_extraction_spec = _hostify_target_lane_constant_tree(
        bs.coil_dof_extraction_spec()
    )
    surface = boozer_surface.surface
    surface_kind = getattr(boozer_surface, "_surface_geometry_kind", None)
    if surface_kind is None:
        surface_type_name = type(surface).__name__
        if surface_type_name == "SurfaceRZFourier":
            surface_kind = "rzfourier"
        elif surface_type_name == "SurfaceXYZFourier":
            surface_kind = "xyzfourier"
        else:
            surface_kind = "generic"

    surface_quadpoints_phi = getattr(
        boozer_surface,
        "quadpoints_phi",
        np.asarray(surface.quadpoints_phi, dtype=np.float64),
    )
    surface_quadpoints_theta = getattr(
        boozer_surface,
        "quadpoints_theta",
        np.asarray(surface.quadpoints_theta, dtype=np.float64),
    )
    surface_quadpoints_phi = host_array(surface_quadpoints_phi, dtype=np.float64)
    surface_quadpoints_theta = host_array(surface_quadpoints_theta, dtype=np.float64)
    surface_mpol = int(getattr(boozer_surface, "mpol", surface.mpol))
    surface_ntor = int(getattr(boozer_surface, "ntor", surface.ntor))
    surface_nfp = int(getattr(boozer_surface, "nfp", surface.nfp))
    surface_stellsym = bool(getattr(boozer_surface, "stellsym", surface.stellsym))
    surface_scatter_indices = getattr(boozer_surface, "scatter_indices", None)
    if surface_scatter_indices is not None:
        surface_scatter_indices = host_array(
            surface_scatter_indices,
            dtype=np.int32,
        )
    vessel_gamma = host_array(
        surface_rz_fourier_gamma_from_spec(vessel_surface.surface_spec()).reshape(
            (-1, 3)
        ),
        dtype=np.float64,
    )
    cc_dist_host = float(cc_dist)
    cs_dist_host = float(cs_dist)
    ss_dist_host = float(ss_dist)
    curvature_threshold_host = float(curvature_threshold)
    inf = np.float64(np.inf)
    success_filter_signature = _target_lane_success_filter_cache_signature(
        {
            "banana_curve_index": int(banana_curve_index),
            "optimize_G": bool(optimize_G),
            "coil_dof_extraction_spec": coil_dof_extraction_spec,
            "surface_kind": surface_kind,
            "surface_quadpoints_phi": surface_quadpoints_phi,
            "surface_quadpoints_theta": surface_quadpoints_theta,
            "surface_mpol": int(surface_mpol),
            "surface_ntor": int(surface_ntor),
            "surface_nfp": int(surface_nfp),
            "surface_stellsym": bool(surface_stellsym),
            "surface_scatter_indices": surface_scatter_indices,
            "vessel_gamma": vessel_gamma,
            "cc_dist": cc_dist_host,
            "cs_dist": cs_dist_host,
            "ss_dist": ss_dist_host,
            "curvature_threshold": curvature_threshold_host,
        }
    )

    def _coil_gamma_points(coil_spec):
        gamma, _ = curve_gamma_and_dash_from_spec(coil_spec.curve)
        if coil_spec.symmetry.has_rotation:
            gamma = gamma @ coil_spec.symmetry.rotmat
        return gamma.reshape((-1, 3))

    def _curve_curve_min_distance(coil_gammas):
        minimum = _as_runtime_float64(inf, reference=coil_gammas[0])
        for i, gamma_i in enumerate(coil_gammas):
            for gamma_j in coil_gammas[:i]:
                minimum = jnp.minimum(
                    minimum,
                    pairwise_min_distance_pure(gamma_i, gamma_j),
                )
        return minimum

    def _curve_surface_min_distance(coil_gammas, surface_gamma):
        minimum = _as_runtime_float64(inf, reference=surface_gamma)
        for gamma in coil_gammas:
            minimum = jnp.minimum(
                minimum,
                pairwise_min_distance_pure(gamma, surface_gamma),
            )
        return minimum

    def _surface_gamma_from_sdofs(sdofs):
        surface_gamma, _surface_gammadash1, _surface_gammadash2 = (
            _surface_geometry_from_dofs(
                sdofs,
                quadpoints_phi=_as_runtime_float64(
                    surface_quadpoints_phi,
                    reference=sdofs,
                ),
                quadpoints_theta=_as_runtime_float64(
                    surface_quadpoints_theta,
                    reference=sdofs,
                ),
                mpol=surface_mpol,
                ntor=surface_ntor,
                nfp=surface_nfp,
                stellsym=surface_stellsym,
                scatter_indices=surface_scatter_indices,
                surface_kind=surface_kind,
            )
        )
        return surface_gamma.reshape((-1, 3))

    def success_filter(coil_dofs, solved_x):
        coil_set_spec = coil_set_spec_from_dof_extraction_spec(
            coil_dof_extraction_spec,
            coil_dofs,
        )
        coil_specs = coil_specs_from_dof_extraction_spec(
            coil_dof_extraction_spec,
            coil_dofs,
        )
        sdofs, _iota, _G = boozer_surface._unpack_decision_vector_jax(
            solved_x,
            optimize_G,
            coil_set_spec=coil_set_spec,
        )
        surface_gamma = _surface_gamma_from_sdofs(sdofs)
        coil_gammas = tuple(_coil_gamma_points(coil_spec) for coil_spec in coil_specs)
        curve_curve_min_dist = _curve_curve_min_distance(coil_gammas)
        curve_surface_min_dist = _curve_surface_min_distance(
            coil_gammas,
            surface_gamma,
        )
        surface_vessel_min_dist = pairwise_min_distance_pure(
            surface_gamma,
            _as_runtime_float64(vessel_gamma, reference=surface_gamma),
        )
        _gamma, banana_gammadash, banana_gammadashdash = curve_geometry_from_spec(
            coil_specs[banana_curve_index].curve
        )
        max_curvature = jnp.max(kappa_pure(banana_gammadash, banana_gammadashdash))
        metrics = jnp.stack(
            (
                curve_curve_min_dist,
                curve_surface_min_dist,
                surface_vessel_min_dist,
                max_curvature,
            )
        )
        return (
            jnp.all(jnp.isfinite(metrics))
            & (
                curve_curve_min_dist
                >= _as_runtime_float64(
                    cc_dist_host,
                    reference=curve_curve_min_dist,
                )
            )
            & (
                curve_surface_min_dist
                >= _as_runtime_float64(
                    cs_dist_host,
                    reference=curve_surface_min_dist,
                )
            )
            & (
                surface_vessel_min_dist
                >= _as_runtime_float64(
                    ss_dist_host,
                    reference=surface_vessel_min_dist,
                )
            )
            & (
                max_curvature
                <= _as_runtime_float64(
                    curvature_threshold_host,
                    reference=max_curvature,
                )
            )
        )

    success_filter._traceable_runtime_cache_signature = (
        "single-stage-target-lane-hardware-success-filter",
        success_filter_signature,
    )
    return success_filter


def build_target_lane_outer_objectives(
    boozer_surface,
    bs,
    iota_target,
    *,
    use_value_and_grad: bool,
    profile_target_lane: bool,
    profile_batch_size: int = 1,
    outer_objective_config=None,
    success_filter=None,
):
    """Build the target-lane objective(s) needed by the selected outer-loop mode."""
    target_scalar_objective = None
    target_value_and_grad_objective = None
    target_optimizer_initial_value_and_grad = None
    target_lane_profile = None
    runtime_bundle = None

    needs_runtime_bundle = True
    if needs_runtime_bundle:
        runtime_bundle = get_traceable_single_stage_runtime_bundle_builder()(
            boozer_surface,
            bs,
            iota_target,
            include_profile_suite=profile_target_lane,
            include_host_wrappers=False,
            outer_objective_config=outer_objective_config,
            success_filter=success_filter,
        )

    target_scalar_objective = runtime_bundle["objective"]
    if use_value_and_grad:
        seeded_value_and_grad = build_traceable_single_stage_seeded_value_and_grad(
            boozer_surface,
            bs,
            iota_target,
            outer_objective_config=outer_objective_config,
            success_filter=success_filter,
        )
        target_value_and_grad_objective = seeded_value_and_grad.value_and_grad
        target_optimizer_initial_value_and_grad = (
            seeded_value_and_grad.optimizer_initial_value_and_grad
        )

    if profile_target_lane:
        target_lane_profile = profile_traceable_target_lane_objective(
            runtime_bundle["profile_suite"],
            build_target_lane_profile_coil_dofs(bs.x.copy()),
        )
        target_lane_profile["profile_point_kind"] = "baseline_perturbed"
        if profile_batch_size > 1:
            target_lane_profile["batched_seed_profile"] = (
                profile_traceable_target_lane_seed_batch(
                    runtime_bundle["profile_suite"],
                    build_target_lane_profile_batch_coil_dofs(
                        bs.x.copy(),
                        batch_size=profile_batch_size,
                    ),
                )
            )
            target_lane_profile["batched_seed_profile"]["profile_point_kind"] = (
                "baseline_perturbed_batch"
            )

    return (
        target_scalar_objective,
        target_value_and_grad_objective,
        target_lane_profile,
        target_optimizer_initial_value_and_grad,
    )


def build_target_lane_profile_coil_dofs(coil_dofs):
    """Return a deterministic non-baseline probe point for target-lane profiling.

    The traceable single-stage objective has an exact-baseline fast path, so
    profiling the unmodified seed DOFs mostly measures the bootstrap shortcut
    instead of the real optimizer hot path. Perturb one finite entry by the
    smallest possible float64 step so profiling still stays arbitrarily close
    to the seed while forcing the general execution path.
    """
    profile_dofs = host_array(coil_dofs, dtype=np.float64).copy()
    finite_indices = np.flatnonzero(np.isfinite(profile_dofs))
    if finite_indices.size == 0:
        return _as_jax_float64(profile_dofs)

    profile_index = int(finite_indices[0])
    current_value = profile_dofs[profile_index]
    if current_value == 0.0:
        profile_dofs[profile_index] = np.finfo(np.float64).eps
    else:
        profile_dofs[profile_index] = np.nextafter(current_value, np.inf)
    return _as_jax_float64(profile_dofs)


def build_target_lane_profile_batch_coil_dofs(coil_dofs, *, batch_size):
    """Return deterministic nearby seed points for batched target-lane profiling."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    baseline_dofs = host_array(coil_dofs, dtype=np.float64).copy().reshape(-1)
    batched_profile_dofs = np.repeat(baseline_dofs[None, :], batch_size, axis=0)
    finite_indices = np.flatnonzero(np.isfinite(baseline_dofs))
    if finite_indices.size == 0:
        return _as_jax_float64(batched_profile_dofs)

    for batch_index in range(batch_size):
        perturb_index = int(finite_indices[batch_index % finite_indices.size])
        perturb_scale = max(abs(float(baseline_dofs[perturb_index])), 1.0)
        perturb_delta = (
            np.finfo(np.float64).eps * perturb_scale * float(batch_index + 1)
        )
        perturb_sign = 1.0 if batch_index % 2 == 0 else -1.0
        batched_profile_dofs[batch_index, perturb_index] = (
            baseline_dofs[perturb_index] + perturb_sign * perturb_delta
        )
    return _as_jax_float64(batched_profile_dofs)


def build_target_lane_outer_objective_config(
    boozer_surface,
    bs,
    banana_curve,
    VV,
    *,
    non_qs_weight,
    residual_weight,
    iota_weight,
    length_weight,
    length_target,
    curve_curve_threshold,
    curve_curve_weight,
    curve_surface_threshold,
    curve_surface_weight,
    surface_vessel_threshold,
    surface_vessel_weight,
    curvature_threshold,
    curvature_weight,
):
    """Build the structured target-lane outer-objective contract once."""
    return build_traceable_single_stage_outer_objective_config(
        boozer_surface,
        bs,
        banana_curve,
        VV,
        non_qs_weight=non_qs_weight,
        residual_weight=residual_weight,
        iota_weight=iota_weight,
        length_weight=length_weight,
        length_target=length_target,
        curve_curve_weight=curve_curve_weight,
        curve_curve_threshold=curve_curve_threshold,
        curve_surface_weight=curve_surface_weight,
        curve_surface_threshold=curve_surface_threshold,
        surface_vessel_weight=surface_vessel_weight,
        surface_vessel_threshold=surface_vessel_threshold,
        curvature_weight=curvature_weight,
        curvature_threshold=curvature_threshold,
    )


def build_traceable_single_stage_alm_runtime_config(
    *,
    constraint_names,
    alm_formulation,
    distance_smoothing,
    curvature_smoothing,
    qs_threshold,
    boozer_threshold,
    iota_penalty_threshold,
    length_penalty_threshold,
    banana_current_threshold,
):
    """Build the immutable pure-JAX ALM runtime config for the inner solve."""

    def optional_threshold(value):
        return None if value is None else float(value)

    return {
        "constraint_names": tuple(str(name) for name in constraint_names),
        "alm_formulation": str(alm_formulation),
        "distance_smoothing": float(distance_smoothing),
        "curvature_smoothing": float(curvature_smoothing),
        "qs_threshold": optional_threshold(qs_threshold),
        "boozer_threshold": optional_threshold(boozer_threshold),
        "iota_penalty_threshold": optional_threshold(iota_penalty_threshold),
        "length_penalty_threshold": optional_threshold(length_penalty_threshold),
        "banana_current_threshold": float(banana_current_threshold),
    }


def build_target_lane_gradient_diagnosis(
    boozer_surface,
    bs,
    banana_curve,
    VV,
    iota_target,
    *,
    success_filter,
    non_qs_weight,
    residual_weight,
    iota_weight,
    length_weight,
    length_target,
    cc_dist,
    cc_weight,
    cs_dist,
    cs_weight,
    ss_dist,
    surf_dist_weight,
    curvature_threshold,
    curvature_weight,
):
    """Return a compact baseline target-lane finiteness report."""
    outer_objective_config = build_target_lane_outer_objective_config(
        boozer_surface,
        bs,
        banana_curve,
        VV,
        non_qs_weight=non_qs_weight,
        residual_weight=residual_weight,
        iota_weight=iota_weight,
        length_weight=length_weight,
        length_target=length_target,
        curve_curve_threshold=cc_dist,
        curve_curve_weight=cc_weight,
        curve_surface_threshold=cs_dist,
        curve_surface_weight=cs_weight,
        surface_vessel_threshold=ss_dist,
        surface_vessel_weight=surf_dist_weight,
        curvature_threshold=curvature_threshold,
        curvature_weight=curvature_weight,
    )
    return get_traceable_single_stage_runtime_diagnostic_builder()(
        boozer_surface,
        bs,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )


def build_target_lane_first_line_search_diagnosis(
    value_and_grad,
    dofs,
    *,
    initial_value_and_grad=None,
    initial_step_size=None,
    maxls,
    gtol,
):
    """Trace the exact first target-lane L-BFGS line search from the seed state."""
    from simsopt.geo.optimizer_jax_private._lbfgs import (
        _line_search_value_and_grad_host,
    )

    x0 = np.asarray(host_array(dofs), dtype=np.float64).reshape(-1)

    def eval_target_value_and_grad(x):
        return _hostify_traceable_value_and_grad(value_and_grad, x)

    if initial_value_and_grad is None:
        f0, g0 = eval_target_value_and_grad(x0)
    else:
        f0 = host_float(initial_value_and_grad[0])
        g0 = np.asarray(host_array(initial_value_and_grad[1]), dtype=np.float64)
        g0 = g0.reshape(x0.shape)

    p0 = -g0
    dphi0 = float(np.dot(g0, p0))
    p0_norm_sq = float(np.dot(p0, p0))
    c1 = 1.0e-4
    c2 = 0.9
    trial_records = []

    def eval_trial(x):
        trial_x = np.asarray(host_array(x), dtype=np.float64).reshape(x0.shape)
        if p0_norm_sq == 0.0:
            alpha = 0.0
        else:
            alpha = float(np.dot(trial_x - x0, p0) / p0_norm_sq)
        value, grad = eval_target_value_and_grad(trial_x)
        dphi = float(np.dot(grad, p0))
        armijo_rhs = float(f0 + c1 * alpha * dphi0)
        trial_records.append(
            {
                "trial_index": int(len(trial_records)),
                "alpha": _summarize_host_scalar(alpha),
                "value": _summarize_host_scalar(value),
                "objective_delta": _summarize_host_scalar(float(value - f0)),
                "linearized_objective_delta": _summarize_host_scalar(
                    float(alpha * dphi0)
                ),
                "directional_derivative": _summarize_host_scalar(dphi),
                "armijo_rhs": _summarize_host_scalar(armijo_rhs),
                "armijo_satisfied": bool(value <= armijo_rhs),
                "curvature_satisfied": bool(abs(dphi) <= -c2 * dphi0),
                "grad": _summarize_host_gradient(grad),
            }
        )
        return value, grad

    line_search_result = _line_search_value_and_grad_host(
        eval_trial,
        x0,
        p0,
        f0,
        g0,
        initial_step_size=initial_step_size,
        maxiter=maxls,
    )
    alpha_star = float(line_search_result.a_k)
    f_star = float(line_search_result.f_k)
    g_star = np.asarray(line_search_result.g_k, dtype=np.float64).reshape(x0.shape)
    step = alpha_star * p0
    y = g_star - g0
    step_norm = float(np.linalg.norm(step))
    y_norm = float(np.linalg.norm(y))
    yts = float(np.dot(y, step))
    yty = float(np.dot(y, y))
    step_eps = float(np.sqrt(np.finfo(np.float64).eps))
    step_tol = step_eps * max(1.0, float(np.linalg.norm(x0)))
    function_change = abs(float(f0 - f_star))
    objective_tol = step_eps * max(abs(float(f0)), abs(float(f_star)))
    gradient_change = y_norm
    gradient_tol = step_eps * max(
        float(np.linalg.norm(g0)),
        float(np.linalg.norm(g_star)),
    )
    converged = bool(np.max(np.abs(g_star)) < float(gtol))
    stalled_step = bool(
        (not converged)
        and step_norm <= step_tol
        and function_change <= objective_tol
        and gradient_change <= gradient_tol
    )
    nonfinite_step = bool(
        (not np.isfinite(f_star))
        or (not np.all(np.isfinite(g_star)))
        or (not np.all(np.isfinite(step)))
        or (not np.all(np.isfinite(x0 + step)))
        or (not np.all(np.isfinite(y)))
    )
    curvature_tol = step_eps * step_norm * y_norm
    valid_curvature = bool(
        np.isfinite(yts)
        and np.isfinite(yty)
        and yts > curvature_tol
        and yty > 0.0
    )
    rejected_step = bool(line_search_result.failed or nonfinite_step or stalled_step)
    return {
        "initial": {
            "dofs": _summarize_host_vector(x0),
            "value": _summarize_host_scalar(f0),
            "grad": _summarize_host_gradient(g0),
            "search_direction": _summarize_host_vector(p0),
            "directional_derivative": _summarize_host_scalar(dphi0),
        },
        "line_search": {
            "initial_step_size": (
                None if initial_step_size is None else float(initial_step_size)
            ),
            "maxls": int(maxls),
            "failed": bool(line_search_result.failed),
            "status": int(line_search_result.status),
            "nit": int(line_search_result.nit),
            "nfev": int(line_search_result.nfev),
            "ngev": int(line_search_result.ngev),
            "alpha": _summarize_host_scalar(alpha_star),
            "value": _summarize_host_scalar(f_star),
            "directional_derivative": _summarize_host_scalar(
                float(np.dot(g_star, p0))
            ),
            "trace": trial_records,
        },
        "optimizer_step": {
            "would_accept": bool(not rejected_step),
            "would_reject": rejected_step,
            "line_search_failed": bool(line_search_result.failed),
            "nonfinite_step": nonfinite_step,
            "stalled_step": stalled_step,
            "valid_curvature": valid_curvature,
            "converged": converged,
            "step": _summarize_host_vector(step),
            "step_norm": _summarize_host_scalar(step_norm),
            "step_tolerance": _summarize_host_scalar(step_tol),
            "function_change": _summarize_host_scalar(function_change),
            "objective_tolerance": _summarize_host_scalar(objective_tol),
            "gradient_change": _summarize_host_scalar(gradient_change),
            "gradient_tolerance": _summarize_host_scalar(gradient_tol),
            "y_dot_s": _summarize_host_scalar(yts),
            "y_dot_y": _summarize_host_scalar(yty),
        },
        "all_finite": bool(
            np.isfinite(f0)
            and np.all(np.isfinite(g0))
            and all(record["value"]["finite"] for record in trial_records)
        ),
    }


def build_target_lane_scaled_phase1_diagnosis(
    boozer_surface,
    bs,
    banana_curve,
    VV,
    iota_target,
    *,
    anchor_dofs,
    contract,
    phase1_maxiter,
    step_scale,
    ftol,
    gtol,
    maxcor,
    outer_maxls,
    callback,
    success_filter,
    non_qs_weight,
    residual_weight,
    iota_weight,
    length_weight,
    length_target,
    cc_dist,
    cc_weight,
    cs_dist,
    cs_weight,
    ss_dist,
    surf_dist_weight,
    curvature_threshold,
    curvature_weight,
    checkpoint_path=None,
):
    """Diagnose the scaled target-lane phase-1 path around the real continuation seam."""
    if not (0.0 < step_scale < 1.0):
        raise ValueError("Scaled phase-1 diagnosis requires step_scale in (0, 1).")
    if phase1_maxiter < 1:
        raise ValueError("Scaled phase-1 diagnosis requires phase1_maxiter >= 1.")

    stage_names = (
        "anchor",
        "scaled_origin",
        "steepest_descent_trial",
        "scaled_origin_after_trial",
        "optimizer_scaled_state",
        "optimizer_mapped_state",
        "scaled_origin_after_optimizer",
    )
    stage_records = {}
    optimizer_payload = None

    def _build_payload(checkpoint_stage, *, diagnosis_complete):
        completed_stage_names = [
            stage_name for stage_name in stage_names if stage_name in stage_records
        ]
        completed_stage_records = [
            (stage_name, stage_records[stage_name])
            for stage_name in completed_stage_names
        ]
        first_nonfinite_stage = _resolve_first_nonfinite_target_lane_stage(
            completed_stage_records
        )
        payload = {
            "contract_method": contract.method,
            "callback_enabled": bool(callback is not None),
            "step_scale": float(step_scale),
            "phase1_maxiter": int(phase1_maxiter),
            "checkpoint_stage": checkpoint_stage,
            "completed_stages": completed_stage_names,
            "diagnosis_complete": bool(diagnosis_complete),
            "optimizer": optimizer_payload,
            "all_finite": (
                first_nonfinite_stage is None if diagnosis_complete else None
            ),
            "all_finite_so_far": first_nonfinite_stage is None,
            "first_nonfinite_stage": first_nonfinite_stage,
        }
        payload.update(
            {stage_name: stage_records.get(stage_name) for stage_name in stage_names}
        )
        return payload

    def _persist_payload(checkpoint_stage, *, diagnosis_complete):
        if checkpoint_path is None:
            return
        write_json_file(
            checkpoint_path,
            _build_payload(
                checkpoint_stage,
                diagnosis_complete=diagnosis_complete,
            ),
        )

    _persist_payload("starting", diagnosis_complete=False)

    outer_objective_config = build_target_lane_outer_objective_config(
        boozer_surface,
        bs,
        banana_curve,
        VV,
        non_qs_weight=non_qs_weight,
        residual_weight=residual_weight,
        iota_weight=iota_weight,
        length_weight=length_weight,
        length_target=length_target,
        curve_curve_threshold=cc_dist,
        curve_curve_weight=cc_weight,
        curve_surface_threshold=cs_dist,
        curve_surface_weight=cs_weight,
        surface_vessel_threshold=ss_dist,
        surface_vessel_weight=surf_dist_weight,
        curvature_threshold=curvature_threshold,
        curvature_weight=curvature_weight,
    )
    runtime_bundle = get_traceable_single_stage_runtime_bundle_builder()(
        boozer_surface,
        bs,
        iota_target,
        include_host_wrappers=False,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    seeded_value_and_grad = build_traceable_single_stage_seeded_value_and_grad(
        boozer_surface,
        bs,
        iota_target,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    _persist_payload("runtime_bundle_ready", diagnosis_complete=False)
    phase1_fun, phase1_callback = build_scaled_outer_problem(
        seeded_value_and_grad.value_and_grad,
        callback,
        anchor_dofs,
        step_scale,
        anchor_in_state=True,
    )
    phase1_optimizer_initial_value_and_grad = (
        build_scaled_outer_problem_initial_value_and_grad(
            seeded_value_and_grad.optimizer_initial_value_and_grad,
            anchor_dofs,
            step_scale,
            anchor_in_state=True,
        )
    )
    anchor_host_dofs = np.asarray(host_array(anchor_dofs), dtype=np.float64).reshape(-1)

    def _base_host_value_and_grad(x):
        return _hostify_traceable_value_and_grad(runtime_bundle["value_and_grad"], x)

    def _phase1_host_value_and_grad(z):
        if isinstance(z, ScaledOuterPhaseOptimizerState):
            z = z.step_dofs
        scaled_dofs = np.asarray(host_array(z), dtype=np.float64).reshape(-1)
        mapped_dofs = anchor_host_dofs + float(step_scale) * scaled_dofs
        value, grad = _base_host_value_and_grad(mapped_dofs)
        return value, float(step_scale) * grad, mapped_dofs, scaled_dofs

    def _evaluate_scaled_state(z):
        value, grad, mapped_dofs, scaled_dofs = _phase1_host_value_and_grad(z)
        return _build_target_lane_value_and_grad_record(
            value=value,
            grad=grad,
            mapped_dofs=mapped_dofs,
            scaled_dofs=scaled_dofs,
        )

    def _evaluate_mapped_state(mapped_dofs):
        value, grad = _base_host_value_and_grad(mapped_dofs)
        return _build_target_lane_value_and_grad_record(
            value=value,
            grad=grad,
            mapped_dofs=mapped_dofs,
        )

    phase1_dofs = build_scaled_outer_phase_initial_dofs(
        anchor_dofs, use_target_lane=True
    )
    phase1_optimizer_dofs = build_target_lane_scaled_outer_phase_state(
        anchor_dofs,
        phase1_dofs,
    )
    stage_records["anchor"] = _evaluate_mapped_state(anchor_host_dofs)
    _persist_payload("anchor", diagnosis_complete=False)
    stage_records["scaled_origin"] = _evaluate_scaled_state(phase1_dofs)
    _persist_payload("scaled_origin", diagnosis_complete=False)
    _, origin_grad, _, _ = _phase1_host_value_and_grad(phase1_dofs)
    stage_records["steepest_descent_trial"] = _evaluate_scaled_state(-origin_grad)
    _persist_payload("steepest_descent_trial", diagnosis_complete=False)
    stage_records["scaled_origin_after_trial"] = _evaluate_scaled_state(phase1_dofs)
    _persist_payload("scaled_origin_after_trial", diagnosis_complete=False)

    phase1_res = run_single_stage_optimizer(
        phase1_fun,
        phase1_optimizer_dofs,
        callback=phase1_callback,
        contract=contract,
        maxiter=phase1_maxiter,
        ftol=ftol,
        gtol=gtol,
        maxcor=maxcor,
        outer_maxls=outer_maxls,
        scalar_fun=None,
        optimizer_initial_value_and_grad=phase1_optimizer_initial_value_and_grad,
    )
    optimizer_payload = {
        "success": bool(phase1_res.success),
        "iterations": int(phase1_res.nit),
        "message": str(phase1_res.message),
        "status": None
        if getattr(phase1_res, "status", None) is None
        else int(phase1_res.status),
        "nfev": None
        if getattr(phase1_res, "nfev", None) is None
        else int(phase1_res.nfev),
        "njev": None
        if getattr(phase1_res, "njev", None) is None
        else int(phase1_res.njev),
        "ls_status": None
        if getattr(phase1_res, "ls_status", None) is None
        else int(phase1_res.ls_status),
        "diagnostics": extract_optimizer_diagnostics(
            phase1_res,
            ran_optimizer=True,
            termination_message=str(phase1_res.message),
        ),
    }
    _persist_payload("optimizer_finished", diagnosis_complete=False)
    stage_records["optimizer_scaled_state"] = _evaluate_scaled_state(phase1_res.x)
    _persist_payload("optimizer_scaled_state", diagnosis_complete=False)
    optimizer_mapped_dofs = resolve_scaled_outer_phase_final_dofs(
        anchor_dofs,
        phase1_res.x,
        step_scale,
        use_target_lane=True,
    )
    stage_records["optimizer_mapped_state"] = _evaluate_mapped_state(
        optimizer_mapped_dofs
    )
    _persist_payload("optimizer_mapped_state", diagnosis_complete=False)
    stage_records["scaled_origin_after_optimizer"] = _evaluate_scaled_state(phase1_dofs)
    _persist_payload("scaled_origin_after_optimizer", diagnosis_complete=False)
    final_payload = _build_payload("completed", diagnosis_complete=True)
    _persist_payload("completed", diagnosis_complete=True)
    return final_payload


def prepare_target_lane_outer_objectives(
    boozer_surface,
    bs,
    banana_curve,
    VV,
    iota_target,
    *,
    use_target_lane: bool,
    use_value_and_grad: bool,
    profile_target_lane: bool,
    profile_batch_size: int,
    disable_success_filter: bool,
    non_qs_weight,
    residual_weight,
    iota_weight,
    length_weight,
    length_target,
    cc_dist,
    cc_weight,
    cs_dist,
    cs_weight,
    ss_dist,
    surf_dist_weight,
    curvature_threshold,
    curvature_weight,
):
    """Build target-lane outer objectives with smooth penalty terms only."""
    del disable_success_filter
    target_scalar_objective = None
    target_value_and_grad_objective = None
    target_optimizer_initial_value_and_grad = None
    target_lane_profile = None
    target_lane_success_filter = None
    target_lane_outer_objective_config = None

    if not use_target_lane:
        return (
            target_scalar_objective,
            target_value_and_grad_objective,
            target_optimizer_initial_value_and_grad,
            target_lane_profile,
            target_lane_success_filter,
        )

    target_lane_outer_objective_config = build_target_lane_outer_objective_config(
        boozer_surface,
        bs,
        banana_curve,
        VV,
        non_qs_weight=non_qs_weight,
        residual_weight=residual_weight,
        iota_weight=iota_weight,
        length_weight=length_weight,
        length_target=length_target,
        curve_curve_weight=cc_weight,
        curve_curve_threshold=cc_dist,
        curve_surface_weight=cs_weight,
        curve_surface_threshold=cs_dist,
        surface_vessel_weight=surf_dist_weight,
        surface_vessel_threshold=ss_dist,
        curvature_weight=curvature_weight,
        curvature_threshold=curvature_threshold,
    )

    (
        target_scalar_objective,
        target_value_and_grad_objective,
        target_lane_profile,
        target_optimizer_initial_value_and_grad,
    ) = build_target_lane_outer_objectives(
        boozer_surface,
        bs,
        iota_target,
        use_value_and_grad=use_value_and_grad,
        profile_target_lane=profile_target_lane,
        profile_batch_size=profile_batch_size,
        outer_objective_config=target_lane_outer_objective_config,
        success_filter=target_lane_success_filter,
    )
    if target_scalar_objective is None:
        target_scalar_objective = _scalar_objective_from_value_and_grad(
            target_value_and_grad_objective
        )
    return (
        target_scalar_objective,
        target_value_and_grad_objective,
        target_optimizer_initial_value_and_grad,
        target_lane_profile,
        target_lane_success_filter,
    )


def select_boozer_residual_class(use_jax, boozer_kind):
    """Select the stage- and backend-matched Boozer residual wrapper."""
    if boozer_kind == "exact":
        return BoozerResidualExact
    if use_jax:
        boozer_residual_cls, _, _ = get_jax_surface_objective_classes()
        return boozer_residual_cls
    return BoozerResidual


def build_boozer_residual_objective(boozer_surface, bs_obj, boozer_residual_cls):
    """Create the stage- and backend-matched Boozer residual wrapper."""
    return boozer_residual_cls(boozer_surface, bs_obj)


def resolve_boozer_optimizer_backend(
    field_backend,
    optimizer_backend,
    boozer_optimizer_backend=None,
):
    """Resolve the inner Boozer LS backend for the active runtime contract."""
    if field_backend != "jax":
        return None
    effective_backend = (
        optimizer_backend
        if boozer_optimizer_backend is None
        else boozer_optimizer_backend
    )
    if effective_backend != "ondevice":
        raise ValueError(
            "Single-stage JAX backend requires boozer_optimizer_backend="
            "'ondevice'. The SciPy/reference Boozer lane is "
            "CPU/reference-only."
        )
    return effective_backend


def resolve_single_stage_default_boozer_least_squares_algorithm(
    field_backend,
    optimizer_backend,
    boozer_optimizer_backend=None,
    boozer_least_squares_algorithm=None,
):
    """Resolve the effective inner Boozer LS algorithm for the active lane."""
    if boozer_least_squares_algorithm is not None or field_backend != "jax":
        return boozer_least_squares_algorithm
    from simsopt.geo.boozersurface_jax import (
        default_least_squares_algorithm_for_backend,
    )

    effective_boozer_backend = resolve_boozer_optimizer_backend(
        field_backend,
        optimizer_backend,
        boozer_optimizer_backend,
    )
    return default_least_squares_algorithm_for_backend(effective_boozer_backend)


def resolve_single_stage_boozer_limited_memory(
    field_backend,
    optimizer_backend,
    boozer_optimizer_backend=None,
    boozer_limited_memory=None,
):
    """Resolve the effective single-stage Boozer quasi-Newton memory policy."""
    if field_backend != "jax":
        return False
    effective_boozer_backend = resolve_boozer_optimizer_backend(
        field_backend,
        optimizer_backend,
        boozer_optimizer_backend,
    )
    if effective_boozer_backend != "ondevice":
        return False
    if boozer_limited_memory is not None:
        return bool(boozer_limited_memory)
    return False


def resolve_single_stage_default_optimizer_backend(
    field_backend,
    optimizer_backend=None,
):
    """Resolve the implicit single-stage outer-loop optimizer lane."""
    if optimizer_backend is not None:
        return optimizer_backend
    if field_backend == "jax":
        return "ondevice"
    return "scipy"


def resolve_single_stage_outer_maxls(
    field_backend,
    optimizer_backend,
    outer_maxls=None,
    *,
    benchmark_mode=False,
):
    """Resolve the effective outer L-BFGS line-search budget for this lane."""
    if outer_maxls is not None:
        resolved = int(outer_maxls)
    elif benchmark_mode and field_backend == "jax" and optimizer_backend == "ondevice":
        resolved = _TARGET_OUTER_MAXLS_BENCHMARK_DEFAULT
    elif field_backend == "jax" and optimizer_backend == "ondevice":
        resolved = _TARGET_OUTER_MAXLS_DEFAULT
    else:
        resolved = _REFERENCE_OUTER_MAXLS_DEFAULT
    if resolved < 1:
        raise ValueError("outer_maxls must be at least 1.")
    return resolved


def resolve_target_lane_outer_initial_step_size(
    field_backend,
    optimizer_backend,
    initial_step_size=None,
    *,
    benchmark_mode=False,
):
    """Resolve the optional first-trial outer L-BFGS step size for target-lane runs."""
    if initial_step_size is not None:
        resolved = float(initial_step_size)
    elif benchmark_mode and field_backend == "jax" and optimizer_backend == "ondevice":
        resolved = _TARGET_OUTER_INITIAL_STEP_SIZE_BENCHMARK_DEFAULT
    else:
        return None
    if resolved <= 0.0:
        raise ValueError("target_lane_outer_initial_step_size must be positive.")
    return resolved


def resolve_single_stage_outer_maxcor(
    field_backend,
    optimizer_backend,
    maxcor=None,
):
    """Resolve the effective outer L-BFGS correction budget for this lane."""
    if maxcor is not None:
        resolved = int(maxcor)
    elif field_backend == "jax" and optimizer_backend == "ondevice":
        resolved = _TARGET_OUTER_MAXCOR_DEFAULT
    else:
        resolved = _REFERENCE_OUTER_MAXCOR_DEFAULT
    if resolved < 1:
        raise ValueError("maxcor must be at least 1.")
    return resolved


def resolve_target_lane_boozer_bfgs_tol(
    field_backend,
    optimizer_backend,
    target_lane_boozer_bfgs_tol=None,
    *,
    benchmark_mode=False,
):
    """Resolve the temporary Boozer LS tolerance override for target-lane trials."""
    if target_lane_boozer_bfgs_tol is not None:
        resolved = float(target_lane_boozer_bfgs_tol)
    elif (
        field_backend == "jax"
        and optimizer_backend == "ondevice"
        and benchmark_mode
    ):
        resolved = _TARGET_LANE_BOOZER_BFGS_TOL_BENCHMARK_DEFAULT
    else:
        return None
    if resolved <= 0.0:
        raise ValueError("target_lane_boozer_bfgs_tol must be positive.")
    return resolved


def resolve_target_lane_boozer_bfgs_maxiter(
    field_backend,
    optimizer_backend,
    target_lane_boozer_bfgs_maxiter=None,
    *,
    benchmark_mode=False,
):
    """Resolve the temporary Boozer LS iteration cap for target-lane trials."""
    if target_lane_boozer_bfgs_maxiter is not None:
        resolved = int(target_lane_boozer_bfgs_maxiter)
    elif field_backend == "jax" and optimizer_backend == "ondevice" and benchmark_mode:
        resolved = _TARGET_LANE_BOOZER_BFGS_MAXITER_BENCHMARK_DEFAULT
    else:
        return None
    if resolved < 1:
        raise ValueError("target_lane_boozer_bfgs_maxiter must be at least 1.")
    return resolved


def resolve_target_lane_boozer_newton_tol(
    field_backend,
    optimizer_backend,
    target_lane_boozer_newton_tol=None,
):
    """Resolve the temporary Boozer Newton tolerance override for target-lane trials."""
    if target_lane_boozer_newton_tol is not None:
        resolved = float(target_lane_boozer_newton_tol)
    else:
        return None
    if resolved <= 0.0:
        raise ValueError("target_lane_boozer_newton_tol must be positive.")
    return resolved


def resolve_target_lane_boozer_newton_maxiter(
    field_backend,
    optimizer_backend,
    target_lane_boozer_newton_maxiter=None,
):
    """Resolve the temporary Boozer Newton iteration cap for target-lane trials."""
    if target_lane_boozer_newton_maxiter is not None:
        resolved = int(target_lane_boozer_newton_maxiter)
    else:
        return None
    if resolved < 1:
        raise ValueError("target_lane_boozer_newton_maxiter must be at least 1.")
    return resolved


@contextmanager
def temporary_boozer_surface_option_overrides(boozer_surface, **overrides):
    """Temporarily override mutable BoozerSurface options for one scoped phase."""
    applied = {key: value for key, value in overrides.items() if value is not None}
    if not applied:
        yield
        return

    original = {key: boozer_surface.options.get(key) for key in applied}
    boozer_surface.options.update(applied)
    try:
        yield
    finally:
        boozer_surface.options.update(original)


_SINGLE_STAGE_COMPONENT_LABEL = "the single-stage outer loop"


def resolve_single_stage_optimizer_contract(field_backend, optimizer_backend):
    """Resolve the optimizer contract for the single-stage outer loop."""
    from simsopt.geo.optimizer_jax import (
        resolve_reference_outer_loop_optimizer_contract,
        resolve_target_outer_loop_optimizer_contract,
    )

    if field_backend == "jax":
        return resolve_target_outer_loop_optimizer_contract(
            field_backend,
            optimizer_backend,
            component_label=_SINGLE_STAGE_COMPONENT_LABEL,
        )
    return resolve_reference_outer_loop_optimizer_contract(
        field_backend,
        optimizer_backend,
        component_label=_SINGLE_STAGE_COMPONENT_LABEL,
    )


def resolve_single_stage_alm_inner_optimizer_contract(
    constraint_method,
    outer_contract,
):
    """Resolve the ALM inner optimizer contract for the single-stage lane."""
    from simsopt.geo.optimizer_jax import TargetOptimizerContract

    if constraint_method != "alm":
        return None
    return (
        outer_contract if isinstance(outer_contract, TargetOptimizerContract) else None
    )


def resolve_single_stage_outer_optimizer_method(field_backend, optimizer_backend):
    """Return the shared optimizer adapter method for the outer single-stage loop."""
    return resolve_single_stage_optimizer_contract(
        field_backend, optimizer_backend
    ).method


def _single_stage_optimizer_dofs_array(x):
    """Normalize outer-loop DOFs to the float array contract used in this file."""
    if isinstance(x, SingleStageOuterOptimizerState):
        x = x.coil_dofs
    if isinstance(x, ScaledOuterPhaseOptimizerState):
        x = x.step_dofs
    return host_array(x, dtype=np.float64)


def _single_stage_outer_optimizer_state(x):
    if isinstance(x, SingleStageOuterOptimizerState):
        coil_dofs = x.coil_dofs
    else:
        coil_dofs = x
    return SingleStageOuterOptimizerState(coil_dofs=_as_jax_float64(coil_dofs))


def _single_stage_target_optimizer_dofs(x):
    """Normalize target-lane optimizer inputs without collapsing pytrees."""
    if isinstance(x, ScaledOuterPhaseOptimizerState):
        return ScaledOuterPhaseOptimizerState(
            step_dofs=_as_jax_float64(x.step_dofs),
            anchor_dofs=_as_jax_float64(x.anchor_dofs),
        )
    return _as_jax_float64(_single_stage_optimizer_dofs_array(x))


def build_target_lane_scaled_outer_phase_state(anchor_dofs, step_dofs):
    """Thread fixed target-lane anchors as dynamic optimizer state."""
    return ScaledOuterPhaseOptimizerState(
        step_dofs=_as_jax_float64(step_dofs),
        anchor_dofs=_as_jax_float64(anchor_dofs),
    )


def _scaled_outer_problem_coordinates(z, anchor_x):
    if isinstance(z, ScaledOuterPhaseOptimizerState):
        step_dofs = z.step_dofs
        anchor_dofs = jax.lax.stop_gradient(z.anchor_dofs)
        return step_dofs, anchor_dofs
    return z, anchor_x


def _scaled_outer_problem_step_and_anchor(z, anchor_x, *, anchor_in_state):
    """Resolve scaled-phase step and anchor coordinates for one optimizer input."""
    if anchor_in_state:
        if not isinstance(z, ScaledOuterPhaseOptimizerState):
            raise TypeError(
                "anchor_in_state=True requires ScaledOuterPhaseOptimizerState inputs."
            )
        return z.step_dofs, jax.lax.stop_gradient(z.anchor_dofs)
    return _scaled_outer_problem_coordinates(z, anchor_x)


def _scaled_outer_problem_gradient(z, grad, scale):
    if isinstance(z, ScaledOuterPhaseOptimizerState):
        runtime_reference = _scaled_outer_problem_runtime_reference(
            grad,
            scale,
            z.step_dofs,
            z.anchor_dofs,
        )
        step_scale = (
            float(scale)
            if runtime_reference is None
            else _as_runtime_float64(scale, reference=runtime_reference)
        )
        anchor_scale = (
            0.0
            if runtime_reference is None
            else _as_runtime_float64(0.0, reference=runtime_reference)
        )
        anchor_dofs = (
            z.anchor_dofs
            if runtime_reference is None
            else _as_runtime_float64(z.anchor_dofs, reference=runtime_reference)
        )
        return ScaledOuterPhaseOptimizerState(
            step_dofs=step_scale * grad,
            anchor_dofs=anchor_scale * anchor_dofs,
        )
    runtime_reference = _scaled_outer_problem_runtime_reference(grad, scale)
    step_scale = (
        float(scale)
        if runtime_reference is None
        else _as_runtime_float64(scale, reference=runtime_reference)
    )
    return step_scale * grad


def should_force_strict_target_lane_final_sync(
    *,
    use_target_lane,
    res_nit,
    optimizer_status,
    accepted_step_callback,
    trial_boozer_override_active,
):
    """Return whether the final accepted state must be re-synced strictly."""
    del accepted_step_callback, trial_boozer_override_active
    return (
        bool(use_target_lane)
        and int(res_nit) > 0
        and target_lane_result_status_allows_state_sync(optimizer_status)
    )


def target_lane_result_status_allows_state_sync(status) -> bool:
    """Return whether a target-lane result status preserves a syncable state."""
    if status is None:
        return True
    return int(status) != 6


def target_lane_result_has_syncable_state(result) -> bool:
    """Return whether a target-lane optimizer result should update the graph."""
    return int(getattr(result, "nit", 0)) > 0 and (
        target_lane_result_status_allows_state_sync(getattr(result, "status", None))
    )


def target_lane_result_objective_metric(result):
    """Return a finite optimizer objective metric when the result exposes one."""
    value = getattr(result, "fun", None)
    if value is None:
        return None
    metric = float(value)
    if not np.isfinite(metric):
        return None
    return metric


def single_stage_stage_label(stage):
    """Return the serialized stage label used in retry summaries."""
    return None if stage is None else str(stage)


def _wrap_single_stage_outer_scalar_objective(fun):
    if fun is None:
        return None

    def wrapped(state):
        return fun(_single_stage_outer_optimizer_state(state).coil_dofs)

    return wrapped


def _wrap_single_stage_outer_value_and_grad_objective(fun):
    if fun is None:
        return None

    def wrapped(state):
        optimizer_state = _single_stage_outer_optimizer_state(state)
        value, grad = fun(optimizer_state.coil_dofs)
        return value, SingleStageOuterOptimizerState(coil_dofs=grad)

    return wrapped


def _scalar_objective_from_value_and_grad(fun):
    """Derive a scalar objective from a fused ``(value, grad)`` callable."""
    if fun is None:
        return None

    def wrapped(state):
        value, _ = fun(state)
        return value

    return wrapped


def _scaled_outer_problem_runtime_reference(*values):
    """Return one runtime leaf when scaled-phase coordinates already touch JAX."""
    array_reference = None
    for value in values:
        for leaf in jax.tree_util.tree_leaves(value):
            if isinstance(leaf, jax.core.Tracer):
                return leaf
            if array_reference is None and isinstance(leaf, jax.Array):
                array_reference = leaf
    return array_reference


def _resolve_scaled_outer_problem_point(anchor_dofs, step_dofs, step_scale):
    """Resolve one scaled-phase point with the explicit target-lane staging contract."""
    runtime_reference = _scaled_outer_problem_runtime_reference(anchor_dofs, step_dofs)
    scale = (
        float(step_scale)
        if runtime_reference is None
        else _as_runtime_float64(step_scale, reference=runtime_reference)
    )
    x = resolve_scaled_outer_phase_final_dofs(
        anchor_dofs,
        step_dofs,
        step_scale,
        use_target_lane=runtime_reference is not None,
    )
    return x, scale


def build_scaled_outer_problem(
    base_fun,
    base_callback,
    anchor_x,
    step_scale,
    *,
    anchor_in_state=False,
):
    """Scale a value-and-gradient outer problem around an anchor point."""
    if not (0.0 < step_scale <= 1.0):
        raise ValueError("step_scale must be in (0, 1]")

    def scaled_fun(z):
        step_dofs, resolved_anchor = _scaled_outer_problem_step_and_anchor(
            z,
            anchor_x,
            anchor_in_state=anchor_in_state,
        )
        x, scale = _resolve_scaled_outer_problem_point(
            resolved_anchor,
            step_dofs,
            step_scale,
        )
        value, grad = base_fun(x)
        return value, _scaled_outer_problem_gradient(z, grad, scale)

    if base_callback is None:
        return scaled_fun, None

    def scaled_callback(z):
        step_dofs, resolved_anchor = _scaled_outer_problem_step_and_anchor(
            z,
            anchor_x,
            anchor_in_state=anchor_in_state,
        )
        x, _ = _resolve_scaled_outer_problem_point(
            resolved_anchor,
            step_dofs,
            step_scale,
        )
        base_callback(x)

    return scaled_fun, scaled_callback


def build_scaled_outer_problem_initial_value_and_grad(
    initial_value_and_grad,
    anchor_x,
    step_scale,
    *,
    anchor_in_state=False,
):
    """Map one fused target-lane seed into scaled optimizer coordinates."""
    if initial_value_and_grad is None:
        return None
    value, grad = initial_value_and_grad
    if anchor_in_state:
        zero_step_dofs = build_scaled_outer_phase_initial_dofs(
            anchor_x,
            use_target_lane=True,
        )
        seed_state = build_target_lane_scaled_outer_phase_state(
            anchor_x,
            zero_step_dofs,
        )
    else:
        seed_state = build_scaled_outer_phase_initial_dofs(
            anchor_x,
            use_target_lane=True,
        )
    return value, _scaled_outer_problem_gradient(seed_state, grad, step_scale)


def build_scaled_outer_scalar_problem(
    base_fun,
    base_callback,
    anchor_x,
    step_scale,
    *,
    anchor_in_state=False,
):
    """Scale a scalar outer problem around an anchor point."""
    if not (0.0 < step_scale <= 1.0):
        raise ValueError("step_scale must be in (0, 1]")

    def scaled_fun(z):
        step_dofs, resolved_anchor = _scaled_outer_problem_step_and_anchor(
            z,
            anchor_x,
            anchor_in_state=anchor_in_state,
        )
        x, _ = _resolve_scaled_outer_problem_point(
            resolved_anchor,
            step_dofs,
            step_scale,
        )
        return base_fun(x)

    if base_callback is None:
        return scaled_fun, None

    def scaled_callback(z):
        step_dofs, resolved_anchor = _scaled_outer_problem_step_and_anchor(
            z,
            anchor_x,
            anchor_in_state=anchor_in_state,
        )
        x, _ = _resolve_scaled_outer_problem_point(
            resolved_anchor,
            step_dofs,
            step_scale,
        )
        base_callback(x)

    return scaled_fun, scaled_callback


def _block_tree_until_ready(tree):
    """Synchronize a pytree of possible JAX arrays before returning timing data."""
    import jax

    for leaf in jax.tree_util.tree_leaves(tree):
        block_until_ready = getattr(leaf, "block_until_ready", None)
        if block_until_ready is not None:
            block_until_ready()
    return tree


def _profile_tree_call(fn, *args):
    """Measure dispatch and ready time for one profiled callable invocation."""
    start_s = _perf_counter_s()
    out = fn(*args)
    dispatch_end_s = _perf_counter_s()
    _block_tree_until_ready(out)
    end_s = _perf_counter_s()
    return out, {
        "dispatch_s": _elapsed_s(start_s, dispatch_end_s),
        "ready_s": _elapsed_s(dispatch_end_s, end_s),
        "total_s": _elapsed_s(start_s, end_s),
    }


def _profile_tree_callable_pair(fn, *args):
    """Return first-call vs warm-call timings for one profiled callable."""
    first_out, first = _profile_tree_call(fn, *args)
    warm_out, warm = _profile_tree_call(fn, *args)
    return warm_out, {
        "first": first,
        "warm": warm,
        "compile_overhead_s": max(first["total_s"] - warm["total_s"], 0.0),
    }


def _increment_diagnostic_counter(counts, key):
    counts[key] = counts.get(key, 0) + 1


class _JaxCompileDiagnosticsRecorder(logging.Handler):
    """Collect compact JAX compile and cache-miss diagnostics for probe runs."""

    def __init__(self, *, sample_limit=_JAX_COMPILE_DIAGNOSTICS_SAMPLE_LIMIT):
        super().__init__(level=logging.WARNING)
        self.sample_limit = int(sample_limit)
        self.compile_event_count = 0
        self.cache_miss_count = 0
        self.compile_target_parse_miss_count = 0
        self.cache_miss_site_parse_miss_count = 0
        self.compile_targets = {}
        self.cache_miss_sites = {}
        self.compile_messages = []
        self.cache_miss_messages = []

    @staticmethod
    def _parse_compile_target(message):
        prefix = "Compiling "
        start = message.find(prefix)
        if start < 0:
            return None
        suffix = message[start + len(prefix) :]
        return suffix.split(" with ", 1)[0].strip() or None

    @staticmethod
    def _parse_cache_miss_site(message):
        prefix = "TRACING CACHE MISS at "
        start = message.find(prefix)
        if start < 0:
            return None
        suffix = message[start + len(prefix) :]
        return suffix.split(" (", 1)[0].strip() or None

    def emit(self, record):
        message = record.getMessage()
        if "Compiling " in message:
            self.compile_event_count += 1
            target = self._parse_compile_target(message)
            if target is not None:
                _increment_diagnostic_counter(self.compile_targets, target)
            else:
                self.compile_target_parse_miss_count += 1
            if len(self.compile_messages) < self.sample_limit:
                self.compile_messages.append(message)
        if "TRACING CACHE MISS" in message:
            self.cache_miss_count += 1
            site = self._parse_cache_miss_site(message)
            if site is not None:
                _increment_diagnostic_counter(self.cache_miss_sites, site)
            else:
                self.cache_miss_site_parse_miss_count += 1
            if len(self.cache_miss_messages) < self.sample_limit:
                self.cache_miss_messages.append(message)

    def summary(self):
        return {
            "compile_event_count": int(self.compile_event_count),
            "cache_miss_count": int(self.cache_miss_count),
            "compile_target_parse_miss_count": int(
                self.compile_target_parse_miss_count
            ),
            "cache_miss_site_parse_miss_count": int(
                self.cache_miss_site_parse_miss_count
            ),
            "compile_targets": {
                key: int(self.compile_targets[key])
                for key in sorted(self.compile_targets)
            },
            "cache_miss_sites": {
                key: int(self.cache_miss_sites[key])
                for key in sorted(self.cache_miss_sites)
            },
            "compile_messages": list(self.compile_messages),
            "cache_miss_messages": list(self.cache_miss_messages),
        }


@contextmanager
def maybe_record_jax_compile_diagnostics(enabled):
    """Capture named JAX compile/cache-miss diagnostics for the wrapped section."""
    if not enabled:
        yield None
        return

    logger = logging.getLogger("jax")
    recorder = _JaxCompileDiagnosticsRecorder()
    previous_level = logger.level
    previous_propagate = logger.propagate
    override_level = (
        previous_level == logging.NOTSET or previous_level > logging.WARNING
    )
    if override_level:
        logger.setLevel(logging.WARNING)
    logger.propagate = False
    previous_explain_cache_misses = bool(jax.config.jax_explain_cache_misses)
    logger.addHandler(recorder)
    try:
        jax.config.update("jax_explain_cache_misses", True)
        with jax.log_compiles(True):
            yield recorder
    finally:
        logger.removeHandler(recorder)
        jax.config.update(
            "jax_explain_cache_misses",
            previous_explain_cache_misses,
        )
        logger.propagate = previous_propagate
        if override_level:
            logger.setLevel(previous_level)


def profile_traceable_target_lane_objective(profile_suite, coil_dofs):
    """Profile the traceable target-lane closures at one representative DOF point."""
    profiled = {}
    forward_result, profiled["forward_result"] = _profile_tree_callable_pair(
        profile_suite["forward_result"],
        coil_dofs,
    )
    _, profiled["forward_value"] = _profile_tree_callable_pair(
        profile_suite["forward_value"],
        coil_dofs,
    )
    _, profiled["warmstart_predict"] = _profile_tree_callable_pair(
        profile_suite["warmstart_predict"],
        coil_dofs,
    )
    solve_result, profiled["inner_solve"] = _profile_tree_callable_pair(
        profile_suite["inner_solve"],
        coil_dofs,
    )
    solved_x = forward_result["x"]
    solved_linear_solve_factors = forward_result["linear_solve_factors"]
    _, profiled["surface_geometry"] = _profile_tree_callable_pair(
        profile_suite["surface_geometry"],
        solved_x,
    )
    _, profiled["field_eval"] = _profile_tree_callable_pair(
        profile_suite["field_eval"],
        coil_dofs,
        solved_x,
    )
    _, profiled["solved_total_objective"] = _profile_tree_callable_pair(
        profile_suite["solved_total_objective"],
        coil_dofs,
        solved_x,
    )
    _, profiled["solved_total_gradient"] = _profile_tree_callable_pair(
        profile_suite["solved_total_gradient"],
        coil_dofs,
        solved_x,
        solved_linear_solve_factors,
    )
    _, profiled["value_and_grad_pipeline"] = _profile_tree_callable_pair(
        profile_suite["value_and_grad_pipeline"],
        coil_dofs,
    )
    profiled["solve_success"] = host_bool(solve_result["success"])
    return profiled


def profile_traceable_target_lane_seed_batch(profile_suite, coil_dofs_batch):
    """Profile the batched seed-evaluation path for nearby target-lane points."""
    (value_batch, grad_batch), profiled_pipeline = _profile_tree_callable_pair(
        profile_suite["batched_value_and_grad_pipeline"],
        coil_dofs_batch,
    )
    host_values = np.asarray(host_array(value_batch), dtype=np.float64).reshape(-1)
    batch_size = int(host_values.size)
    host_gradients = np.asarray(host_array(grad_batch), dtype=np.float64).reshape(
        batch_size, -1
    )
    value_finite_mask = np.isfinite(host_values)
    gradient_finite_mask = np.isfinite(host_gradients)
    return {
        "batch_size": batch_size,
        "value_and_grad_pipeline": profiled_pipeline,
        "all_values_finite": bool(np.all(value_finite_mask)),
        "all_gradients_finite": bool(np.all(gradient_finite_mask)),
        "max_value_abs": None
        if not np.all(value_finite_mask)
        else float(np.max(np.abs(host_values))),
        "max_gradient_inf_norm": None
        if not np.all(gradient_finite_mask)
        else float(np.max(np.max(np.abs(host_gradients), axis=1))),
        "first_total_s_per_seed": profiled_pipeline["first"]["total_s"]
        / float(batch_size),
        "warm_total_s_per_seed": profiled_pipeline["warm"]["total_s"]
        / float(batch_size),
    }


def resolve_single_stage_outer_optimizer_initial_dofs(
    JF,
    bs,
    *,
    use_target_lane,
):
    """Return the optimizer-space DOFs for the selected outer-loop contract."""
    if use_target_lane:
        return _as_jax_float64(bs.x.copy())
    return _single_stage_optimizer_dofs_array(JF.x.copy())


def build_scaled_outer_phase_initial_dofs(dofs, *, use_target_lane):
    """Return zero-origin optimizer coordinates for a scaled initial phase."""
    if use_target_lane:
        return _as_runtime_float64(0.0, reference=dofs) * dofs
    return np.zeros_like(_single_stage_optimizer_dofs_array(dofs))


def resolve_scaled_outer_phase_final_dofs(
    anchor_dofs,
    step_dofs,
    step_scale,
    *,
    use_target_lane,
):
    """Map scaled-phase optimizer coordinates back to the original DOF basis."""
    if isinstance(step_dofs, ScaledOuterPhaseOptimizerState):
        anchor_dofs = step_dofs.anchor_dofs
        step_dofs = step_dofs.step_dofs
    if use_target_lane:
        runtime_reference = None
        for candidate in (anchor_dofs, step_dofs):
            if isinstance(candidate, jax.core.Tracer):
                runtime_reference = candidate
                break
            if isinstance(candidate, jax.Array):
                runtime_reference = candidate
                break
        if isinstance(runtime_reference, jax.core.Tracer):
            anchor_runtime = _as_runtime_float64(
                anchor_dofs, reference=runtime_reference
            )
            step_runtime = _as_runtime_float64(step_dofs, reference=runtime_reference)
            scale = _as_runtime_float64(step_scale, reference=runtime_reference)
            return anchor_runtime + scale * step_runtime
        if isinstance(runtime_reference, jax.Array):
            runtime_sharding = runtime_reference.sharding

            def _explicit_runtime_array(value):
                if isinstance(value, jax.Array):
                    return _as_runtime_float64(value, reference=runtime_reference)
                host_value = np.asarray(host_array(value), dtype=np.float64)
                return jax.device_put(host_value, device=runtime_sharding)

            anchor_runtime = _explicit_runtime_array(anchor_dofs)
            step_runtime = _explicit_runtime_array(step_dofs)
            scale = _as_runtime_float64(step_scale, reference=runtime_reference)
            return anchor_runtime + scale * step_runtime
        return _as_jax_float64(anchor_dofs) + _as_jax_float64(step_scale) * (
            _as_jax_float64(step_dofs)
        )
    return _single_stage_optimizer_dofs_array(anchor_dofs) + float(step_scale) * (
        _single_stage_optimizer_dofs_array(step_dofs)
    )


def build_single_stage_scaled_phase_retry_state(anchor_dofs):
    """Build a zero-step scaled optimizer state anchored at accepted physical DOFs."""
    return build_target_lane_scaled_outer_phase_state(
        anchor_dofs,
        build_scaled_outer_phase_initial_dofs(anchor_dofs, use_target_lane=True),
    )


def resolve_single_stage_outer_dof_setter(
    JF,
    bs,
    *,
    use_target_lane,
):
    """Return the graph update hook for the selected outer-loop contract."""
    target = bs if use_target_lane else JF

    def _set_dofs(x):
        target.x = _single_stage_optimizer_dofs_array(x)

    return _set_dofs


def run_single_stage_optimizer(
    fun,
    dofs,
    *,
    contract,
    maxiter,
    ftol,
    gtol,
    maxcor,
    outer_maxls,
    callback,
    progress_callback=None,
    scalar_fun=None,
    target_lane_initial_step_size=None,
    failure_callback=None,
    optimizer_initial_value_and_grad=None,
):
    """Run the single-stage outer optimization through the lane-specific adapters."""
    from simsopt.geo.optimizer_jax import (
        ReferenceOptimizerContract,
        TargetOptimizerContract,
        reference_minimize,
        target_minimize,
    )

    optimizer_dofs = dofs
    is_target_lane = isinstance(contract, TargetOptimizerContract)
    if fun is not None:
        optimizer_fun = fun
        value_and_grad = True
    elif is_target_lane:
        if scalar_fun is None:
            raise RuntimeError(
                "Single-stage target-lane optimization requires either the fused "
                "value-and-gradient objective or a scalar JAX objective."
            )
        optimizer_fun = scalar_fun
        value_and_grad = False
    else:
        raise RuntimeError(
            "Single-stage optimization requires an explicit value-and-gradient "
            "objective for the selected lane."
        )
    if is_target_lane:
        optimizer_dofs = _single_stage_target_optimizer_dofs(dofs)
        target_minimize_kwargs = {
            "method": contract.method,
            "tol": gtol,
            "maxiter": maxiter,
            "options": {
                "maxcor": int(maxcor),
                "ftol": float(ftol),
                "maxls": int(outer_maxls),
            },
            "value_and_grad": value_and_grad,
            "callback": callback,
            "progress_callback": progress_callback,
        }
        if target_lane_initial_step_size is not None:
            target_minimize_kwargs["options"]["initial_step_size"] = float(
                target_lane_initial_step_size
            )
        if failure_callback is not None:
            target_minimize_kwargs["failure_callback"] = failure_callback
        if optimizer_initial_value_and_grad is not None:
            target_minimize_kwargs["initial_value_and_grad"] = (
                optimizer_initial_value_and_grad
            )
        return target_minimize(
            optimizer_fun,
            optimizer_dofs,
            **target_minimize_kwargs,
        )
    if not isinstance(contract, ReferenceOptimizerContract):
        raise RuntimeError(
            f"Unsupported single-stage optimizer contract {type(contract)!r}."
        )
    if failure_callback is not None:
        raise ValueError(
            "Single-stage reference-lane optimization does not support "
            "failure_callback; use the target ondevice lane for "
            "line-search failure diagnostics."
        )
    if optimizer_initial_value_and_grad is not None:
        raise ValueError(
            "Single-stage reference-lane optimization does not support "
            "optimizer_initial_value_and_grad."
        )
    return reference_minimize(
        optimizer_fun,
        optimizer_dofs,
        method=contract.method,
        tol=gtol,
        maxiter=maxiter,
        options={
            "maxcor": int(maxcor),
            "ftol": float(ftol),
            "maxls": int(outer_maxls),
        },
        value_and_grad=True,
        callback=callback,
        progress_callback=progress_callback,
    )


def run_single_stage_target_lane_optimizer_with_retries(
    fun,
    dofs,
    *,
    phase,
    callback,
    retry_callback,
    result_state_sync,
    contract,
    maxiter,
    ftol,
    gtol,
    maxcor,
    outer_maxls,
    scalar_fun,
    progress_callback=None,
    target_lane_initial_step_size,
    failure_callback,
    optimizer_initial_value_and_grad=None,
    invalid_state_events,
    run_dict,
    single_stage_search_policy,
    retry_dofs_factory=None,
    restored_result_x_factory=None,
    progress_event_callback=None,
):
    """Retry invalid-state target-lane failures from preserved local anchors."""

    def default_retry_dofs_factory(anchor_state):
        return host_array(anchor_state["coil_dofs"], dtype=np.float64)

    if retry_dofs_factory is None:
        retry_dofs_factory = default_retry_dofs_factory

    if restored_result_x_factory is None:
        restored_result_x_factory = retry_dofs_factory

    def record_progress_event(label, **extra):
        if progress_event_callback is None:
            return
        progress_event_callback(label, phase=phase, **extra)

    def sync_failed_attempt_state(result, *, event_start_index):
        if result_state_sync is None:
            return
        if getattr(result_state_sync, "simsopt_skip_failed_attempt_sync", False):
            return
        if int(getattr(result, "nit", 0)) <= 0 or result.success:
            return
        if single_stage_retry_triggered_by_invalid_state(
            invalid_state_events[event_start_index:]
        ):
            result_state_sync(result.x)

    best_retry_result = None
    best_retry_metric = None
    best_retry_anchor_state = None
    best_retry_anchor_stage = None

    def record_best_syncable_result(result):
        nonlocal best_retry_result
        nonlocal best_retry_metric
        nonlocal best_retry_anchor_state
        nonlocal best_retry_anchor_stage
        if not target_lane_result_has_syncable_state(result):
            return
        metric = target_lane_result_objective_metric(result)
        if metric is None:
            return
        if best_retry_metric is not None and metric >= best_retry_metric:
            return
        anchor_state = run_dict.get("latest_local_incumbent")
        anchor_stage = run_dict.get("latest_local_stage")
        if anchor_state is not None:
            anchor_metric = float(anchor_state["J"])
            if not np.isclose(anchor_metric, metric, rtol=1.0e-12, atol=1.0e-12):
                anchor_state = None
                anchor_stage = None
        best_retry_result = result
        best_retry_metric = metric
        best_retry_anchor_state = (
            None if anchor_state is None else copy.deepcopy(anchor_state)
        )
        best_retry_anchor_stage = anchor_stage

    initial_step_size = target_lane_initial_step_size
    event_start = len(invalid_state_events)
    optimizer_kwargs = {}
    if optimizer_initial_value_and_grad is not None:
        optimizer_kwargs["optimizer_initial_value_and_grad"] = (
            optimizer_initial_value_and_grad
        )
    record_progress_event(
        f"{phase}_attempt_0_started",
        attempt_index=0,
        maxiter=int(maxiter),
        initial_step_size=None
        if initial_step_size is None
        else float(initial_step_size),
        optimizer_dofs=_summarize_host_vector(dofs),
    )
    result = run_single_stage_optimizer(
        fun,
        dofs,
        callback=callback,
        contract=contract,
        maxiter=maxiter,
        ftol=ftol,
        gtol=gtol,
        maxcor=maxcor,
        outer_maxls=outer_maxls,
        scalar_fun=scalar_fun,
        progress_callback=progress_callback,
        target_lane_initial_step_size=initial_step_size,
        failure_callback=failure_callback,
        **optimizer_kwargs,
    )
    record_progress_event(
        f"{phase}_attempt_0_returned",
        attempt_index=0,
        result=summarize_optimizer_result_for_progress(result),
    )
    extend_target_lane_invalid_state_events_from_result(
        invalid_state_events,
        result,
        phase=phase,
    )
    total_nit = int(getattr(result, "nit", 0))
    total_nfev = int(getattr(result, "nfev", 0)) if hasattr(result, "nfev") else None
    total_njev = int(getattr(result, "njev", 0)) if hasattr(result, "njev") else None
    retry_summary = {
        "attempt_count": 0,
        "attempts": [],
        "restored_preserved_local_state": False,
        "restored_preserved_local_stage": None,
    }
    sync_failed_attempt_state(result, event_start_index=event_start)
    record_best_syncable_result(result)
    for retry_index in range(single_stage_search_policy.invalid_step_retry_budget):
        new_events = invalid_state_events[event_start:]
        if result.success or (
            not single_stage_retry_triggered_by_invalid_state(new_events)
        ):
            break
        anchor_state, anchor_stage = resolve_single_stage_retry_anchor(
            run_dict,
            single_stage_search_policy,
        )
        if anchor_state is None:
            break
        restore_single_stage_local_incumbent_state(run_dict, anchor_state)
        initial_step_size = resolve_single_stage_retry_initial_step_size(
            initial_step_size,
            new_events,
            single_stage_search_policy=single_stage_search_policy,
            retry_index=retry_index,
        )
        retry_summary["attempt_count"] += 1
        retry_summary["attempts"].append(
            {
                "retry_index": int(retry_index + 1),
                "anchor_stage": single_stage_stage_label(anchor_stage),
                "anchor_metric": float(anchor_state["J"]),
                "initial_step_size": float(initial_step_size),
                "triggered_by_invalid_state": True,
            }
        )
        record_progress_event(
            f"{phase}_retry_{retry_index + 1}_started",
            attempt_index=int(retry_index + 1),
            anchor_stage=single_stage_stage_label(anchor_stage),
            anchor_metric=float(anchor_state["J"]),
            initial_step_size=float(initial_step_size),
            remaining_maxiter=int(max(int(maxiter) - total_nit, 1)),
        )
        event_start = len(invalid_state_events)
        remaining_maxiter = max(int(maxiter) - total_nit, 1)
        optimizer_initial_value_and_grad = None
        result = run_single_stage_optimizer(
            fun,
            retry_dofs_factory(anchor_state),
            callback=retry_callback,
            contract=contract,
            maxiter=remaining_maxiter,
            ftol=ftol,
            gtol=gtol,
            maxcor=maxcor,
            outer_maxls=outer_maxls,
            scalar_fun=scalar_fun,
            progress_callback=progress_callback,
            target_lane_initial_step_size=initial_step_size,
            failure_callback=failure_callback,
        )
        record_progress_event(
            f"{phase}_retry_{retry_index + 1}_returned",
            attempt_index=int(retry_index + 1),
            result=summarize_optimizer_result_for_progress(result),
        )
        extend_target_lane_invalid_state_events_from_result(
            invalid_state_events,
            result,
            phase=phase,
        )
        sync_failed_attempt_state(result, event_start_index=event_start)
        record_best_syncable_result(result)
        total_nit += int(getattr(result, "nit", 0))
        if total_nfev is not None:
            total_nfev += int(getattr(result, "nfev", 0))
        if total_njev is not None:
            total_njev += int(getattr(result, "njev", 0))
    if (
        (not result.success)
        and retry_summary["attempt_count"] > 0
        and best_retry_result is not None
        and best_retry_result is not result
    ):
        result = best_retry_result
        if best_retry_anchor_state is not None:
            restore_single_stage_local_incumbent_state(
                run_dict,
                best_retry_anchor_state,
            )
            result.x = restored_result_x_factory(best_retry_anchor_state)
            result.restored_preserved_local_state = True
            result.restored_preserved_local_stage = single_stage_stage_label(
                best_retry_anchor_stage
            )
            retry_summary["restored_preserved_local_state"] = True
            retry_summary["restored_preserved_local_stage"] = (
                result.restored_preserved_local_stage
            )
            record_progress_event(
                f"{phase}_restored_preserved_local_state",
                anchor_stage=result.restored_preserved_local_stage,
                anchor_metric=float(best_retry_anchor_state["J"]),
            )
    result.nit = total_nit
    if total_nfev is not None:
        result.nfev = total_nfev
    if total_njev is not None:
        result.njev = total_njev
    if (not result.success) and (not target_lane_result_has_syncable_state(result)):
        anchor_state, anchor_stage = resolve_single_stage_retry_anchor(
            run_dict,
            single_stage_search_policy,
        )
        if anchor_state is not None:
            restore_single_stage_local_incumbent_state(run_dict, anchor_state)
            result.x = restored_result_x_factory(anchor_state)
            result.restored_preserved_local_state = True
            result.restored_preserved_local_stage = single_stage_stage_label(
                anchor_stage
            )
            retry_summary["restored_preserved_local_state"] = True
            retry_summary["restored_preserved_local_stage"] = (
                result.restored_preserved_local_stage
            )
            record_progress_event(
                f"{phase}_restored_preserved_local_state",
                anchor_stage=result.restored_preserved_local_stage,
                anchor_metric=float(anchor_state["J"]),
            )
    return result, retry_summary


logger = logging.getLogger(__name__)

_DIAG_LABELS = {
    "qs": "nonQS ratio",
    "boozer": "Boozer Residual",
    "iota_penalty": "ι Penalty",
    "length": "Curve Length Penalty",
    "cc": "Curve-Curve Penalty",
    "cs": "Curve-Surface Penalty",
    "surf": "Surf-Vessel Penalty",
    "curvature": "Curvature Penalty",
}


def write_json_file(path, payload):
    """Write a sanitized JSON payload to disk."""
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as outfile:
        json.dump(sanitize_json_payload(payload), outfile, indent=2, allow_nan=False)


def build_stage_progress_recorder(path):
    """Build a small JSON progress recorder for long-running staged workflows."""
    completed_stages = []
    stage_payloads = {}

    def record_stage(label, **extra):
        if label not in completed_stages:
            completed_stages.append(label)
        stage_payloads[label] = dict(extra)
        write_json_file(
            path,
            {
                "current_stage": label,
                "completed_stages": list(completed_stages),
                "stages": dict(stage_payloads),
            },
        )

    return record_stage


def build_event_progress_recorder(path):
    """Build a JSON event recorder that preserves chronological history."""
    events = []
    started_s = _perf_counter_s()

    def record_event(label, **extra):
        event = {
            "label": label,
            "event_index": int(len(events)),
            "event_elapsed_s": float(_perf_counter_s() - started_s),
            **dict(extra),
        }
        events.append(event)
        write_json_file(
            path,
            {
                "current_event": label,
                "event_count": int(len(events)),
                "events": list(events),
            },
        )

    return record_event


def resolve_warm_start_boozer_init_overrides(
    *,
    warm_start_state,
    explicit_surface_warm_start,
    field_backend,
    optimizer_backend,
    boozer_optimizer_backend,
    boozer_least_squares_algorithm,
    boozer_least_squares_algorithm_explicit,
    target_lane_boozer_bfgs_tol,
    target_lane_boozer_bfgs_maxiter,
):
    """Choose a conservative Boozer init policy for warm-started baselines.

    The target-lane trial budget is intentionally aggressive, but warm-start
    initialization seeds the baseline traceable runtime state and implicit-diff
    factorization. Keep that baseline solve on a stricter floor so gradient
    diagnostics do not start from an under-resolved anchor. Only force the
    historical quasi-Newton LS path for legacy warm starts that cannot replay
    an explicit surface state. When explicit surface DOFs are available, keep
    the caller-selected LS algorithm so the continuation baseline matches the
    proven warm-start path.
    """
    if warm_start_state is None:
        return {
            "least_squares_algorithm_override": None,
            "bfgs_tol_override": None,
            "bfgs_maxiter_override": None,
            "newton_tol_override": None,
            "newton_maxiter_override": None,
        }

    least_squares_algorithm_override = None
    if (
        field_backend == "jax"
        and not explicit_surface_warm_start
        and not bool(boozer_least_squares_algorithm_explicit)
        and boozer_least_squares_algorithm in {None, "lm"}
        and resolve_boozer_optimizer_backend(
            field_backend,
            optimizer_backend,
            boozer_optimizer_backend,
        )
        == "ondevice"
    ):
        least_squares_algorithm_override = "quasi-newton"

    bfgs_tol_override = None
    if target_lane_boozer_bfgs_tol is not None:
        bfgs_tol_override = min(float(target_lane_boozer_bfgs_tol), 1.0e-8)

    bfgs_maxiter_override = None
    if target_lane_boozer_bfgs_maxiter is not None:
        bfgs_maxiter_override = max(int(target_lane_boozer_bfgs_maxiter), 128)

    return {
        "least_squares_algorithm_override": least_squares_algorithm_override,
        "bfgs_tol_override": bfgs_tol_override,
        "bfgs_maxiter_override": bfgs_maxiter_override,
        "newton_tol_override": None,
        "newton_maxiter_override": None,
    }


def resolve_target_lane_boozer_init_base_overrides(
    *,
    field_backend,
    optimizer_backend,
    boozer_limited_memory=False,
    target_lane_boozer_bfgs_tol,
    target_lane_boozer_bfgs_maxiter,
    target_lane_boozer_newton_tol,
    target_lane_boozer_newton_maxiter,
):
    """Return the baseline target-lane init overrides for JAX/ondevice solves."""
    if field_backend != "jax" or optimizer_backend != "ondevice":
        return {
            "least_squares_algorithm_override": None,
            "bfgs_tol_override": None,
            "bfgs_maxiter_override": None,
            "newton_tol_override": None,
            "newton_maxiter_override": None,
        }

    bfgs_tol_override = (
        None
        if target_lane_boozer_bfgs_tol is None
        else min(float(target_lane_boozer_bfgs_tol), 1.0e-8)
    )
    newton_tol_override = (
        _TARGET_LANE_BOOZER_NEWTON_TOL_FULL_MEMORY_DEFAULT
        if (
            target_lane_boozer_newton_tol is None
            and not bool(boozer_limited_memory)
        )
        else (
            None
            if target_lane_boozer_newton_tol is None
            else float(target_lane_boozer_newton_tol)
        )
    )

    return {
        "least_squares_algorithm_override": None,
        "bfgs_tol_override": bfgs_tol_override,
        "bfgs_maxiter_override": (
            None
            if target_lane_boozer_bfgs_maxiter is None
            else max(int(target_lane_boozer_bfgs_maxiter), 128)
        ),
        "newton_tol_override": newton_tol_override,
        "newton_maxiter_override": (
            None
            if target_lane_boozer_newton_maxiter is None
            else int(target_lane_boozer_newton_maxiter)
        ),
    }


def _restore_cpu_boozer_state(boozer_surface, run_dict):
    """Restore CPU BoozerSurface warm-start state from run_dict snapshot."""
    boozer_surface.surface.x = run_dict["sdofs"]
    boozer_surface.res["iota"] = run_dict["iota"]
    boozer_surface.res["G"] = run_dict["G"]


def _boozer_surface_supports_explicit_surface_warm_start(boozer_surface):
    support = getattr(
        boozer_surface,
        "supports_explicit_surface_warm_start",
        None,
    )
    if support is not None:
        return bool(support)

    run_code_signature = inspect.signature(boozer_surface.run_code)
    return "sdofs" in run_code_signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in run_code_signature.parameters.values()
    )


def _boozer_surface_requires_jax_supported_self_intersection(boozer_surface):
    return boozer_surface.__class__.__name__ == "BoozerSurfaceJAX"


def _update_line_search_state(x, run_dict):
    """Track step size and increment line-search counter."""
    dx = np.linalg.norm(x - run_dict["x_prev"])
    run_dict["x_prev"] = x.copy()
    logger.info("Step size: %.2e", dx)
    run_dict["lscount"] += 1


def _single_stage_failure_residual_inf(boozer_surface):
    residual = boozer_surface.res.get("residual")
    if residual is None:
        residual = boozer_surface.res.get("fun")
    if residual is None:
        return 0.0
    residual_array = np.asarray(host_array(residual, dtype=np.float64), dtype=float)
    if residual_array.size == 0:
        return 0.0
    return float(np.linalg.norm(residual_array.reshape(-1), ord=np.inf))


def _single_stage_hardware_violation_score(hardware_status):
    if hardware_status is None or hardware_status["success"]:
        return 0.0
    score = 0.25 * float(len(hardware_status["violations"]))
    for distance_name, threshold_name in (
        ("curve_curve_min_dist", "cc_dist"),
        ("curve_surface_min_dist", "cs_dist"),
        ("surface_vessel_min_dist", "ss_dist"),
    ):
        distance = float(hardware_status.get(distance_name, 0.0))
        threshold = float(hardware_status.get(threshold_name, 0.0))
        if threshold > 0.0:
            score += max(threshold - distance, 0.0) / threshold
    curvature_threshold = float(hardware_status.get("curvature_threshold", 0.0))
    max_curvature = float(hardware_status.get("max_curvature", 0.0))
    if curvature_threshold > 0.0:
        score += max(max_curvature - curvature_threshold, 0.0) / curvature_threshold
    return score


def compute_single_stage_failure_penalty(
    x,
    run_dict,
    boozer_surface,
    *,
    success_solve,
    is_intersecting,
    hardware_status,
):
    """Scale failure penalties by donor policy, step size, and solve quality."""
    failure_weight = float(run_dict.get("adaptive_failure_penalty_weight", 1.0))
    last_objective = float(run_dict["J"])
    penalty_base = max(abs(last_objective), 1.0)
    previous_x = np.asarray(run_dict["x_prev"], dtype=float)
    step_norm = float(np.linalg.norm(np.asarray(x, dtype=float) - previous_x))
    reference_norm = max(float(np.linalg.norm(previous_x)), 1.0)
    step_ratio = step_norm / reference_norm
    residual_inf = (
        0.0 if success_solve else _single_stage_failure_residual_inf(boozer_surface)
    )
    hardware_score = _single_stage_hardware_violation_score(hardware_status)
    failure_count = int(run_dict.get("failure_count", 0))
    multiplier = failure_weight
    multiplier += min(step_ratio, 4.0)
    multiplier += min(residual_inf, 4.0)
    multiplier += hardware_score
    multiplier += 0.5 if is_intersecting else 0.0
    multiplier += 0.25 * float(failure_count)
    penalty = penalty_base * multiplier
    return penalty, {
        "step_norm": step_norm,
        "step_ratio": step_ratio,
        "residual_inf": residual_inf,
        "hardware_score": hardware_score,
        "failure_count": failure_count,
        "intersecting": bool(is_intersecting),
        "solver_success": bool(success_solve),
        "search_policy": run_dict.get("search_policy"),
        "donor_class": run_dict.get("donor_class"),
        "penalty": penalty,
        "penalty_multiplier": multiplier,
    }


def _evaluate_candidate_impl(
    x,
    run_dict,
    boozer_surface,
    JF,
    objectives=None,
    diagnostics=None,
):
    """Evaluate a candidate coil configuration against the mutable objective graph.

    Runs the inner Boozer solve with warm-start from ``run_dict`` and
    returns ``(J, dJ)``.

    On success: ``J = JF.J()``, ``dJ = JF.dJ()``.
    On failure: ``J = run_dict["J"] + penalty``, ``dJ = run_dict["dJ"]``
    (gradient-inconsistent by design — see plan documentation).

    The caller (``SingleStageAdapter.__call__``) sets ``JF.x = x``
    before calling this function.  This function mutates ``run_dict``
    (tracking state) and, on the CPU path, mutates
    ``boozer_surface.surface.x`` / ``boozer_surface.res`` for
    warm-start and failure rollback.

    Args:
        x: Candidate coil DOFs from the optimizer.
        run_dict: Mutable optimization state dict (the single source of truth).
        boozer_surface: The Boozer surface adapter.
        JF: Composite objective (``Optimizable``).

    Returns:
        (J, dJ): Objective value and gradient.
    """
    uses_legacy_warm_start = not _boozer_surface_supports_explicit_surface_warm_start(
        boozer_surface
    )
    if uses_legacy_warm_start:
        _restore_cpu_boozer_state(boozer_surface, run_dict)
        boozer_surface.run_code(run_dict["iota"], run_dict["G"])
    else:
        boozer_surface.run_code(
            run_dict["iota"], run_dict["G"], sdofs=run_dict["sdofs"]
        )
    success_solve = host_bool(boozer_surface.res["success"])
    is_intersecting = update_self_intersection_status(
        run_dict,
        boozer_surface.surface,
        require_supported_surface=_boozer_surface_requires_jax_supported_self_intersection(
            boozer_surface
        ),
    )
    success = success_solve and not is_intersecting

    if success and _can_evaluate_single_stage_hardware_status(objectives, diagnostics):
        hardware_status = _evaluate_single_stage_hardware_status(
            objectives,
            diagnostics,
        )
        run_dict["hardware_constraint_status"] = hardware_status
        success = hardware_status["success"]
    else:
        run_dict.pop("hardware_constraint_status", None)

    if success:
        run_dict["failure_count"] = 0
        run_dict.pop("last_candidate_failure", None)
        J = JF.J()
        dJ = JF.dJ()
        logger.info("Volume: %s", host_float(boozer_surface.surface.volume()))
        logger.info("Iota: %s", host_float(boozer_surface.res["iota"]))
    else:
        if not success_solve:
            logger.warning("Boozer solver failed")
        if is_intersecting:
            logger.warning("Surface is self-intersecting")
        hardware_status = run_dict.get("hardware_constraint_status")
        if hardware_status is not None and not hardware_status["success"]:
            logger.warning("Hardware constraints violated")

        # Elevated J triggers line-search backtracking.
        # Returning dJ (not the derivative of J) is intentionally
        # gradient-inconsistent: it produces y_k=0 if the step is ever
        # accepted, safely skipping the L-BFGS Hessian update via the
        # ys > 0 guard.
        failure_penalty, failure_summary = compute_single_stage_failure_penalty(
            x,
            run_dict,
            boozer_surface,
            success_solve=success_solve,
            is_intersecting=is_intersecting,
            hardware_status=hardware_status,
        )
        logger.warning(
            "Single-stage failure penalty: policy=%s donor_class=%s "
            "step_norm=%.2e residual_inf=%.2e hardware_score=%.2e "
            "multiplier=%.2e penalty=%.2e",
            failure_summary["search_policy"],
            failure_summary["donor_class"],
            failure_summary["step_norm"],
            failure_summary["residual_inf"],
            failure_summary["hardware_score"],
            failure_summary["penalty_multiplier"],
            failure_summary["penalty"],
        )
        J = run_dict["J"] + failure_penalty
        dJ = run_dict["dJ"].copy()
        run_dict["failure_count"] = int(run_dict.get("failure_count", 0)) + 1
        run_dict["last_candidate_failure"] = failure_summary

        if uses_legacy_warm_start:
            _restore_cpu_boozer_state(boozer_surface, run_dict)

    return J, dJ


def _refresh_single_stage_runtime_state(
    run_dict,
    boozer_surface,
):
    """Refresh only the solved mutable state for an accepted target-lane point."""
    uses_legacy_warm_start = not _boozer_surface_supports_explicit_surface_warm_start(
        boozer_surface
    )
    if uses_legacy_warm_start:
        _restore_cpu_boozer_state(boozer_surface, run_dict)
        boozer_surface.run_code(run_dict["iota"], run_dict["G"])
    else:
        boozer_surface.run_code(
            run_dict["iota"], run_dict["G"], sdofs=run_dict["sdofs"]
        )
    success = host_bool(
        boozer_surface.res["success"]
    ) and not update_self_intersection_status(
        run_dict,
        boozer_surface.surface,
        require_supported_surface=_boozer_surface_requires_jax_supported_self_intersection(
            boozer_surface
        ),
    )
    if not success and uses_legacy_warm_start:
        _restore_cpu_boozer_state(boozer_surface, run_dict)
    return success


def evaluate_candidate(
    x,
    run_dict,
    boozer_surface,
    JF,
    objectives=None,
    diagnostics=None,
):
    """Evaluate a candidate coil configuration.

    This is the line-search/objective entry used by the explicit outer
    optimizer contract.  Tracks line-search state before evaluating.
    """
    _update_line_search_state(x, run_dict)
    return _evaluate_candidate_impl(
        x,
        run_dict,
        boozer_surface,
        JF,
        objectives,
        diagnostics,
    )


def snapshot_accepted_step_state_from_values(
    run_dict,
    *,
    sdofs,
    iota,
    G,
    objective_value=None,
    objective_grad=None,
    store_objective_grad=True,
):
    """Persist accepted-step state from explicit array/scalar values."""
    run_dict["lscount"] = 0
    _clear_target_lane_reporting_cache(run_dict)
    run_dict["sdofs"] = host_array(sdofs, dtype=np.float64)
    run_dict["iota"] = host_float(iota)
    run_dict["G"] = host_float(G)
    if objective_value is not None:
        run_dict["J"] = host_float(objective_value)
    if store_objective_grad:
        if objective_value is None:
            raise ValueError(
                "objective_value is required when store_objective_grad=True"
            )
        if objective_grad is None:
            raise ValueError(
                "objective_grad is required when store_objective_grad=True"
            )
        run_dict["dJ"] = host_array(objective_grad)
    return run_dict["J"], run_dict["dJ"]


def snapshot_accepted_step_state(
    run_dict,
    boozer_surface,
    JF,
    *,
    objective_value=None,
    objective_grad=None,
    store_objective_grad=True,
):
    """Persist the accepted-step solver/objective state into run_dict."""
    if store_objective_grad:
        if objective_value is None:
            objective_value = JF.J()
        if objective_grad is None:
            objective_grad = JF.dJ()
    return snapshot_accepted_step_state_from_values(
        run_dict,
        sdofs=boozer_surface.surface.x,
        iota=boozer_surface.res["iota"],
        G=boozer_surface.res["G"],
        objective_value=objective_value,
        objective_grad=objective_grad,
        store_objective_grad=store_objective_grad,
    )


def accept_step(
    run_dict,
    boozer_surface,
    JF,
    bs,
    objectives,
    diagnostics_refs,
    log_path,
    *,
    objective_value=None,
    objective_grad=None,
):
    """Update state and log diagnostics on an accepted optimizer step.

    Called by the optimizer callback. Snapshots the current Optimizable
    state into ``run_dict`` and evaluates per-component diagnostics.

    Does not persistently mutate any Optimizable object.  The
    BiotSavart field's evaluation points are saved before the B·n
    diagnostic and restored afterward.

    Args:
        run_dict: Mutable optimization state dict (updated in place).
        boozer_surface: The Boozer surface adapter.
        JF: Composite objective (``Optimizable``).
        bs: Biot-Savart field object.
        objectives: Dict of named objective components for diagnostics.
        diagnostics_refs: Dict of extra diagnostic objects (banana_curve, etc.).
        log_path: Path to the iteration log file.
    """
    snapshot_kwargs = {}
    if objective_value is not None or objective_grad is not None:
        snapshot_kwargs = {
            "objective_value": objective_value,
            "objective_grad": objective_grad,
            "store_objective_grad": True,
        }
    J, grad = snapshot_accepted_step_state(
        run_dict,
        boozer_surface,
        JF,
        **snapshot_kwargs,
    )

    # Per-component diagnostics
    diag = {}
    for name, obj in objectives.items():
        grad = host_array(obj.dJ())
        diag[name] = (host_float(obj.J()), float(np.linalg.norm(grad)))

    iota_obj = diagnostics_refs["iota"]
    banana_curve = diagnostics_refs["banana_curve"]
    curvelength_obj = diagnostics_refs["curvelength"]

    iota_str = f"{host_float(iota_obj.J()):.4f}"
    volume_str = f"{host_float(boozer_surface.surface.volume()):.4f}"

    gamma = banana_curve.gamma()
    max_r = np.max(np.sqrt(gamma[:, 0] ** 2 + gamma[:, 1] ** 2))
    max_z = np.max(np.abs(gamma[:, 2]))
    length = host_float(curvelength_obj.J())
    hardware_status = _evaluate_single_stage_hardware_status(
        objectives,
        diagnostics_refs,
    )
    run_dict["hardware_constraint_status"] = hardware_status
    max_curvature = hardware_status["max_curvature"]
    curvecurve_min = hardware_status["curve_curve_min_dist"]
    curvesurf_min = hardware_status["curve_surface_min_dist"]
    surface_vessel_min = hardware_status["surface_vessel_min_dist"]

    # Save bs evaluation points so we can restore after the diagnostic
    _bs_pts_before = None
    if isinstance(bs, BiotSavart):
        _bs_pts_before = bs.get_points_cart_ref().copy()
    elif hasattr(bs, "_points_jax") and bs._points_jax is not None:
        _bs_pts_before = bs._points_jax  # JAX arrays are immutable; no copy needed

    bs.set_points(boozer_surface.surface.gamma().reshape((-1, 3)))
    unitn = boozer_surface.surface.unitnormal()
    BdotN = host_float(
        np.mean(np.abs(np.sum(host_array(bs.B()).reshape(unitn.shape) * unitn, axis=2)))
    )

    # Restore bs state — no persistent mutation
    if _bs_pts_before is not None:
        bs.set_points(_bs_pts_before)
    update_self_intersection_status(
        run_dict,
        boozer_surface.surface,
        require_supported_surface=_boozer_surface_requires_jax_supported_self_intersection(
            boozer_surface
        ),
    )
    record_single_stage_local_incumbent(
        run_dict,
        stage=f"iter_{run_dict.get('it', 0)}",
    )

    width = 35
    buffer = io.StringIO()
    print("=" * 70, file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(f"{'Objective J':{width}} = {J:.6e}", file=buffer)
    print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
    for name, (val, gnorm) in diag.items():
        label = _DIAG_LABELS.get(name, name)
        extra = ""
        if name == "cc":
            extra = f" (min={curvecurve_min:.3e})"
        elif name == "cs":
            extra = f" (min={curvesurf_min:.3e})"
        print(f"{label:{width}} = {val:.6e}{extra} (dJ = {gnorm:.6e})", file=buffer)
    print(f"{'Iotas (actual)':{width}} = {iota_str}", file=buffer)
    print(f"{'Volume':{width}} = {volume_str}", file=buffer)
    print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)
    check_status = (
        "available"
        if run_dict["self_intersection_check_available"]
        else "skipped (dependency unavailable)"
    )
    print(f"{'Intersecting':{width}} = {run_dict['intersecting']}", file=buffer)
    print(f"{'Self-intersection check':{width}} = {check_status}", file=buffer)
    print(f"{'Max Curve R':{width}} = {max_r:.6e}", file=buffer)
    print(f"{'Max Curve Z':{width}} = {max_z:.6e}", file=buffer)
    print(f"{'Max Curvature':{width}} = {max_curvature:.6e}", file=buffer)
    print(f"{'Curve Length':{width}} = {length:.6e}", file=buffer)
    print(
        f"{'Surface-Vessel Min Dist':{width}} = {surface_vessel_min:.6e}", file=buffer
    )
    print(
        f"{'Hardware Constraints OK':{width}} = {hardware_status['success']}",
        file=buffer,
    )
    if hardware_status["violations"]:
        print(
            f"{'Hardware Violations':{width}} = {hardware_status['violations']}",
            file=buffer,
        )
    print("=" * 70, file=buffer)

    output_str = buffer.getvalue()
    buffer.close()
    logger.info("%s", output_str)

    with open(log_path, "a") as f:
        f.write(output_str + "\n")

    run_dict["it"] += 1


class SingleStageAdapter:
    """Stateful adapter wrapping evaluate_candidate/accept_step for L-BFGS.

    Carries all optimization state explicitly so the outer loop does not
    depend on module-level globals.  Provides ``__call__`` for the objective
    and ``callback`` for accepted-step updates.
    """

    def __init__(
        self,
        run_dict,
        boozer_surface,
        JF,
        bs,
        objectives,
        diagnostics,
        log_path,
        reevaluate_before_accept=False,
        apply_coil_dofs=None,
        benchmark_mode=False,
        accepted_step_state_sync=None,
    ):
        self.run_dict = run_dict
        self.boozer_surface = boozer_surface
        self.JF = JF
        self.bs = bs
        self.objectives = objectives
        self.diagnostics = diagnostics
        self.log_path = log_path
        self.reevaluate_before_accept = bool(reevaluate_before_accept)
        self.benchmark_mode = bool(benchmark_mode)
        self.accepted_step_state_sync = accepted_step_state_sync
        self.apply_coil_dofs = (
            apply_coil_dofs
            if apply_coil_dofs is not None
            else (
                lambda x: setattr(self.JF, "x", _single_stage_optimizer_dofs_array(x))
            )
        )

    def _reevaluate_accepted_step(self, x):
        """Refresh accepted-step state on the mutable graph for diagnostics.

        The ondevice optimizer evaluates through JAX autodiff without
        updating the Optimizable graph.  This re-evaluation at the
        accepted point refreshes the mutable state that diagnostics
        and ``accept_step`` depend on.
        """
        x_array = _single_stage_optimizer_dofs_array(x)
        self.apply_coil_dofs(x_array)
        objective_value, objective_grad = _evaluate_candidate_impl(
            x_array,
            self.run_dict,
            self.boozer_surface,
            self.JF,
            self.objectives,
            self.diagnostics,
        )
        self.run_dict["x_prev"] = x_array.copy()
        return objective_value, objective_grad

    def _refresh_accepted_step_runtime_state(self, x):
        """Refresh only the mutable solved state for a benchmark accepted step."""
        x_array = _single_stage_optimizer_dofs_array(x)
        self.apply_coil_dofs(x_array)
        self.run_dict.pop("hardware_constraint_status", None)
        success = _refresh_single_stage_runtime_state(
            self.run_dict,
            self.boozer_surface,
        )
        self.run_dict["x_prev"] = x_array.copy()
        return success

    def _sync_target_lane_accepted_step_summary(self, x, *, update_run_state):
        """Run the pure target-lane accepted-step sync with optional state commit."""
        accepted_step_summary = self.accepted_step_state_sync(
            self.run_dict,
            x,
            benchmark_mode=self.benchmark_mode,
            update_run_state=update_run_state,
        )
        if update_run_state:
            self.run_dict["x_prev"] = _single_stage_optimizer_dofs_array(x).copy()
        return accepted_step_summary

    def _log_target_lane_accepted_step(self, accepted_step_summary):
        """Write the accepted-step summary when host-side logging is enabled."""
        if self.benchmark_mode:
            return
        log_single_stage_target_lane_accepted_step(
            self.run_dict,
            accepted_step_summary,
            self.log_path,
        )

    def sync_accepted_step_state(self, x):
        """Refresh the accepted step without emitting per-accept logging."""
        if self.accepted_step_state_sync is None:
            self.sync_accepted_step(x)
            return
        self._sync_target_lane_accepted_step_summary(x, update_run_state=True)

    def observe_accepted_step(self, x):
        """Emit target-lane accepted-step observability without state mutation."""
        if self.accepted_step_state_sync is None:
            self.sync_accepted_step(x)
            return
        accepted_step_summary = self._sync_target_lane_accepted_step_summary(
            x,
            update_run_state=False,
        )
        self._log_target_lane_accepted_step(accepted_step_summary)

    def sync_accepted_step(self, x):
        """Refresh mutable state if needed, then snapshot one accepted step."""
        if self.accepted_step_state_sync is not None:
            accepted_step_summary = self._sync_target_lane_accepted_step_summary(
                x,
                update_run_state=True,
            )
            self._log_target_lane_accepted_step(accepted_step_summary)
            return
        objective_value = None
        objective_grad = None
        if self.benchmark_mode:
            if self.reevaluate_before_accept and (
                not self._refresh_accepted_step_runtime_state(x)
            ):
                objective_value, objective_grad = self._reevaluate_accepted_step(x)
            snapshot_accepted_step_state(
                self.run_dict,
                self.boozer_surface,
                self.JF,
                objective_value=objective_value,
                objective_grad=objective_grad,
                store_objective_grad=objective_value is not None,
            )
            return
        if self.reevaluate_before_accept:
            objective_value, objective_grad = self._reevaluate_accepted_step(x)
        accept_step(
            self.run_dict,
            self.boozer_surface,
            self.JF,
            self.bs,
            self.objectives,
            self.diagnostics,
            self.log_path,
            objective_value=objective_value,
            objective_grad=objective_grad,
        )

    def __call__(self, x):
        """Objective for L-BFGS — delegates to evaluate_candidate.

        Sets ``JF.x = x`` to update coil DOFs on the Optimizable graph
        before delegating.  This is the only place the outer loop writes
        candidate coil DOFs onto ``JF``; ``evaluate_candidate`` then
        updates ``run_dict`` and warm-start state as part of the explicit
        outer-loop contract.
        """
        x_array = _single_stage_optimizer_dofs_array(x)
        self.apply_coil_dofs(x_array)
        return evaluate_candidate(
            x_array,
            self.run_dict,
            self.boozer_surface,
            self.JF,
            self.objectives,
            self.diagnostics,
        )

    def callback(self, x):
        """Accepted-step callback — delegates to accept_step.

        No persistent Optimizable mutation.
        """
        self.sync_accepted_step(x)


def snapshot_to_pytree(
    JF,
    boozer_surface,
    bs,
    *,
    num_tf_coils,
    coil_dofs_override=None,
    evaluate_initial_objective=True,
):
    """Extract pre-optimization state from the Optimizable graph.

    Converts the mutable Optimizable graph into plain arrays and metadata.
    The returned ``run_dict`` serves as the mutable accepted-state
    container for :class:`SingleStageAdapter`, and ``static_config``
    captures frozen geometry (TF coil ``gamma()``, currents) that does
    not change during optimization.

    Args:
        JF: Composite objective (``Optimizable``).
        boozer_surface: Boozer surface adapter.
        bs: Biot-Savart field object with ``.coils``.
        num_tf_coils: Number of TF coils (first ``num_tf_coils`` in
            ``bs.coils`` are frozen; the rest are banana coils).
        coil_dofs_override: Optional optimizer-lane coil DOFs to store in
            ``run_dict``.  The JAX target lane optimizes ``bs.x`` directly,
            while ``JF.x`` belongs to the legacy Optimizable graph.
        evaluate_initial_objective: Whether to evaluate the legacy host
            objective and gradient while snapshotting.  Target-lane runs seed
            these fields from the fused JAX value/gradient after the runtime
            bundle is built, avoiding an expensive duplicate host-gradient path.

    Returns:
        (coil_dofs, run_dict, static_config):
        - coil_dofs: Starting DOFs for the optimizer.
        - run_dict: Mutable accepted-state dict for ``SingleStageAdapter``.
        - static_config: Frozen arrays and metadata.

    Raises:
        RuntimeError: If Boozer surface has not been solved or solve failed.
    """
    if boozer_surface.res is None or not boozer_surface.res.get("success", False):
        raise RuntimeError(
            "snapshot_to_pytree requires a successful Boozer solve; "
            "call initialize_boozer_surface() first."
        )
    coil_dofs = (
        JF.x.copy()
        if coil_dofs_override is None
        else host_array(coil_dofs_override, dtype=np.float64).copy()
    )
    coils = bs.coils
    tf_coils = coils[:num_tf_coils]

    initial_objective_pending = not bool(evaluate_initial_objective)
    if evaluate_initial_objective:
        initial_objective = host_float(JF.J())
        initial_objective_grad = host_array(JF.dJ())
    else:
        initial_objective = float("nan")
        initial_objective_grad = np.zeros_like(coil_dofs, dtype=np.float64)
    solved_state = _resolved_single_stage_boozer_solved_state(boozer_surface)

    run_dict = {
        "sdofs": host_array(solved_state.sdofs, dtype=np.float64),
        "iota": host_float(solved_state.iota),
        "G": host_float(solved_state.G),
        "J": initial_objective,
        "dJ": initial_objective_grad,
        "initial_objective": initial_objective,
        "initial_objective_pending": initial_objective_pending,
        "it": 1,
        "lscount": 0,
        "failure_count": 0,
        "x_prev": coil_dofs.copy(),
        "intersecting": False,
        "self_intersection_check_available": (
            surface_self_intersection_check_available(boozer_surface.surface)
        ),
        "latest_local_incumbent": None,
        "latest_local_metric": None,
        "latest_local_stage": None,
        "best_local_incumbent": None,
        "best_local_metric": None,
        "best_local_stage": None,
    }

    static_config = {
        "num_tf_coils": num_tf_coils,
        "tf_gamma": [c.curve.gamma().copy() for c in tf_coils],
        "tf_gammadash": [c.curve.gammadash().copy() for c in tf_coils],
        "tf_currents": [host_float(c.current.get_value()) for c in tf_coils],
    }

    return coil_dofs, run_dict, static_config


def seed_single_stage_initial_objective_from_values(
    run_dict,
    *,
    objective_value,
    objective_grad,
):
    """Seed the initial accepted objective from an already-computed lane value."""
    run_dict["J"] = host_float(objective_value)
    run_dict["dJ"] = host_array(objective_grad, dtype=np.float64)
    run_dict["initial_objective"] = host_float(objective_value)
    run_dict["initial_objective_pending"] = False


def restore_from_pytree(
    JF,
    boozer_surface,
    run_dict,
    coil_dofs=None,
    *,
    apply_coil_dofs=None,
    diagnostic_bs=None,
):
    """Write optimization state back into the Optimizable graph.

    Restores coil DOFs, surface DOFs, and the warm-start scalars
    (``iota``, ``G``) so post-optimization consumers see values
    consistent with the last accepted step.

    Note: only the solved-value state is restored directly here. Dense
    linearization artifacts are optional compatibility outputs now, so
    adjoint consumers must rebuild the runtime state by re-solving after
    ``surface.x`` or ``JF.x`` dirties the Boozer surface
    (``need_to_run_code = True``). The next access through an
    ``IotasJAX`` / ``NonQuasiSymmetricRatioJAX`` wrapper will trigger
    ``_ensure_solved`` and refresh the full runtime contract
    automatically.

    Args:
        JF: Composite objective (``Optimizable``).
        boozer_surface: Boozer surface adapter.
        run_dict: Final accepted-state dict from the optimizer.
        coil_dofs: Final coil DOFs from the optimizer result. If None,
            the coil DOFs in the graph are left unchanged.
    """
    if coil_dofs is not None:
        coil_dofs = _single_stage_optimizer_dofs_array(coil_dofs)
        if apply_coil_dofs is None:
            JF.x = coil_dofs
        else:
            apply_coil_dofs(coil_dofs)
        if diagnostic_bs is not None:
            diagnostic_bs.x = coil_dofs
    boozer_surface.surface.x = run_dict["sdofs"]
    boozer_surface.res["iota"] = run_dict["iota"]
    boozer_surface.res["G"] = run_dict["G"]


def single_stage_host_postprocess_required(
    *,
    use_target_lane,
    write_full_artifacts,
):
    """Return whether final host graph restore/export is part of this run."""
    return (not use_target_lane) or bool(write_full_artifacts)


def single_stage_host_artifact_export_required(
    *,
    use_target_lane,
    write_restart_artifacts,
    write_full_artifacts,
):
    """Return whether final artifact export needs the host object graph."""
    return bool(write_full_artifacts or (write_restart_artifacts and not use_target_lane))


def single_stage_final_host_restore_required(
    *,
    skip_outer_optimizer,
    use_target_lane,
    host_state_restored_for_final,
):
    """Return whether final export must replay optimizer DOFs into host objects."""
    return bool(
        (not host_state_restored_for_final)
        and (not skip_outer_optimizer or use_target_lane)
    )


def require_single_stage_jax_target_lane(*, use_jax, use_target_lane):
    if use_jax and not use_target_lane:
        raise ValueError(
            "JAX production startup consumes immutable runtime seed specs only on "
            "the target optimizer lane; use --constraint-method penalty with "
            "--optimizer-backend ondevice, or use the CPU/reference lane."
        )


def restore_single_stage_host_state(
    *,
    use_target_lane,
    JF,
    boozer_surface,
    run_dict,
    coil_dofs,
    apply_coil_dofs,
    bs_diag,
    record_outer_optimizer_event,
):
    """Restore the mutable host object graph at an explicit I/O boundary."""
    record_outer_optimizer_event(
        "host_state_restore_started",
        coil_dofs=_summarize_host_vector(coil_dofs),
    )
    host_state_restore_start_s = _perf_counter_s()
    restore_from_pytree(
        JF,
        boozer_surface,
        run_dict,
        coil_dofs=coil_dofs,
        apply_coil_dofs=apply_coil_dofs,
        diagnostic_bs=bs_diag if use_target_lane else None,
    )
    record_outer_optimizer_event(
        "host_state_restore_returned",
        elapsed_s=float(_perf_counter_s() - host_state_restore_start_s),
    )


def write_single_stage_final_runtime_seed_spec(
    *,
    output_dir,
    surface,
    solved_surface_state,
    field_source,
    final_coil_dofs,
    num_tf_coils,
    tf_current_A,
    banana_current_A,
    stage2_seed,
):
    return write_single_stage_jax_runtime_seed_spec(
        output_dir,
        surface=surface,
        surface_dofs=solved_surface_state["sdofs"],
        iota=solved_surface_state["iota"],
        G=solved_surface_state["G"],
        mpol=surface.mpol,
        ntor=surface.ntor,
        quadpoints_phi=surface.quadpoints_phi,
        quadpoints_theta=surface.quadpoints_theta,
        coil_dof_extraction_spec=field_source.coil_dof_extraction_spec(),
        coil_dofs=final_coil_dofs,
        num_tf_coils=num_tf_coils,
        banana_curve_index=int(num_tf_coils),
        tf_current_A=tf_current_A,
        banana_current_A=banana_current_A,
        stage2_seed=stage2_seed,
    )


def export_requested_single_stage_artifacts(
    *,
    solved_surface_state,
    coil_dofs,
    num_tf_coils,
    tf_current_A,
    banana_current_A,
    stage2_seed,
    output_dir,
    boozer_surface,
    bs_diag,
    surf_coils,
    hbt,
    VV,
    write_restart_artifacts,
    write_host_restart_artifacts,
    write_full_artifacts,
    timings,
):
    """Export requested host artifacts from a restored host object graph."""
    final_artifacts_start_s = _perf_counter_s()
    if write_host_restart_artifacts:
        bs_diag.save(os.path.join(output_dir, "biot_savart_opt.json"))
        boozer_surface.surface.save(os.path.join(output_dir, "surf_opt.json"))
    if write_restart_artifacts:
        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

        runtime_seed_bs = BiotSavartJAX(bs_diag.coils)
        write_single_stage_final_runtime_seed_spec(
            output_dir=output_dir,
            surface=boozer_surface.surface,
            solved_surface_state=solved_surface_state,
            field_source=runtime_seed_bs,
            final_coil_dofs=coil_dofs,
            num_tf_coils=num_tf_coils,
            tf_current_A=tf_current_A,
            banana_current_A=banana_current_A,
            stage2_seed=stage2_seed,
        )

    if write_full_artifacts:
        artifact_coils = bs_diag.coils
        curves_to_vtk(
            [coil.curve for coil in artifact_coils],
            os.path.join(output_dir, "curves_opt"),
            close=True,
        )
        bs_diag.set_points(boozer_surface.surface.gamma().reshape((-1, 3)))
        unitn = boozer_surface.surface.unitnormal()
        pointData = {
            "B_N/B": np.sum(
                bs_diag.B().reshape(unitn.shape) * unitn,
                axis=2,
            )[:, :, None]
            / np.sqrt(np.sum(bs_diag.B().reshape(unitn.shape) ** 2, axis=2))[
                :, :, None
            ]
        }
        boozer_surface.surface.to_vtk(
            os.path.join(output_dir, "surf_opt"),
            extra_data=pointData,
        )
        normPlot(
            boozer_surface.surface,
            bs_diag,
            os.path.join(output_dir, "NormPlotOptimized"),
        )
        cross_section_plot(
            surf_coils,
            boozer_surface.surface,
            artifact_coils[num_tf_coils].curve,
            os.path.join(output_dir, "CrossSectionOptimized"),
            hbt,
            VV,
        )
    _record_timing(
        timings,
        "final_artifacts_s",
        final_artifacts_start_s,
        _perf_counter_s(),
    )


def run_single_stage_host_diagnostics(
    *,
    boozer_surface,
    run_dict,
    timings,
):
    """Run explicit host diagnostics after host state restoration."""
    final_diagnostics_start_s = _perf_counter_s()
    self_intersecting = update_self_intersection_status(
        run_dict,
        boozer_surface.surface,
        require_supported_surface=_boozer_surface_requires_jax_supported_self_intersection(
            boozer_surface
        ),
    )
    _record_timing(
        timings,
        "final_host_diagnostics_s",
        final_diagnostics_start_s,
        _perf_counter_s(),
    )
    return SingleStageHostPostprocessResult(
        self_intersecting=bool(self_intersecting),
        self_intersection_check_available=bool(
            run_dict["self_intersection_check_available"]
        ),
    )


def build_single_stage_final_result_snapshot(
    *,
    final_coil_dofs,
    run_dict,
    final_metrics,
    final_distances,
    hardware_status,
    optimizer_result,
    optimizer_diagnostics,
    timings,
    write_restart_artifacts,
    write_full_artifacts,
    boozer_optimizer_method,
    field_error,
    self_intersecting,
    self_intersection_check_available,
):
    """Collect final result truth before host serialization."""
    return SingleStageFinalResultSnapshot(
        final_coil_dofs=host_array(final_coil_dofs, dtype=np.float64).copy(),
        solved_surface_state={
            "sdofs": host_array(run_dict["sdofs"], dtype=np.float64).copy(),
            "iota": host_float(run_dict["iota"]),
            "G": host_float(run_dict["G"]),
        },
        final_metrics=dict(final_metrics),
        final_distances=dict(final_distances),
        hardware_status=dict(hardware_status),
        optimizer_result=dict(optimizer_result),
        optimizer_diagnostics=dict(optimizer_diagnostics),
        timings=dict(timings),
        artifact_policy={
            "write_restart_artifacts": bool(write_restart_artifacts),
            "write_full_artifacts": bool(write_full_artifacts),
        },
        boozer_optimizer_method=boozer_optimizer_method,
        results_payload={},
        field_error=field_error,
        self_intersecting=bool(self_intersecting),
        self_intersection_check_available=bool(self_intersection_check_available),
    )


def summarize_single_stage_final_optimizer_result(
    *,
    result,
    ran_optimizer,
    iterations,
    optimizer_success,
    termination_message,
):
    """Return the final optimizer status values used by results.json."""
    if not ran_optimizer:
        return {
            "iterations": int(iterations),
            "success": bool(optimizer_success),
            "termination_message": termination_message,
            "status": None,
            "nfev": None,
            "njev": None,
            "ls_status": None,
        }
    return {
        "iterations": int(iterations),
        "success": bool(optimizer_success),
        "termination_message": termination_message,
        "status": int(getattr(result, "status", -1)),
        "nfev": int(getattr(result, "nfev", 0)),
        "njev": int(getattr(result, "njev", 0)),
        "ls_status": _optional_int(getattr(result, "ls_status", None)),
    }


def with_single_stage_results_payload(snapshot, results):
    """Return a final snapshot carrying the exact results.json payload."""
    payload = dict(results)
    payload["TIMINGS"] = snapshot.timings
    return replace(snapshot, results_payload=payload)


def write_single_stage_results_json(output_dir, snapshot):
    """Write results.json from the finalized result snapshot payload."""
    write_json_file(os.path.join(output_dir, "results.json"), snapshot.results_payload)


# Convergence tolerances for different mpol values (module-level for testability)
ftol_by_mpol = {
    8: 1e-5,
    9: 5e-6,
    10: 1e-6,
    11: 5e-7,
    12: 1e-7,
    13: 5e-8,
    14: 1e-8,
    15: 5e-9,
    16: 1e-9,
    17: 5e-10,
    18: 1e-10,
}
gtol_by_mpol = {
    8: 1e-2,
    9: 5e-3,
    10: 1e-3,
    11: 5e-4,
    12: 1e-4,
    13: 5e-5,
    14: 1e-5,
    15: 5e-6,
    16: 1e-6,
    17: 5e-7,
    18: 1e-7,
}


if __name__ == "__main__":
    run_wall_start_s = _perf_counter_s()

    if not logging.root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logging.root.addHandler(handler)
        logging.root.setLevel(logging.INFO)

    # ==============================================================================
    # CONFIGURATION PARAMETERS
    # ==============================================================================
    args = apply_default_stage2_seed_args(parse_args())
    if args.compile_jax_runtime_seed_spec:
        compiled_spec_path = compile_requested_single_stage_jax_runtime_seed_spec(args)
        print(f"Wrote JAX runtime seed spec: {compiled_spec_path}")
        sys.exit(0)

    OUT_DIR = args.output_root
    os.makedirs(OUT_DIR, exist_ok=True)
    fatal_error_log_stream = open(
        os.path.join(OUT_DIR, "fatal_error.log"),
        "w",
        encoding="utf-8",
        buffering=1,
    )
    # atexit runs LIFO: registering close() first then faulthandler.disable()
    # guarantees disable fires before close, so a late fault signal cannot
    # write to an already-closed stream. Exception paths are covered too.
    atexit.register(fatal_error_log_stream.close)
    atexit.register(faulthandler.disable)
    faulthandler.enable(file=fatal_error_log_stream, all_threads=True)
    startup_progress_path = os.path.join(OUT_DIR, "startup_progress.json")
    startup_progress = {"completed_stages": [], "timings": {}}

    def _mark_startup_progress(stage_name, *, start_s=None):
        startup_progress["current_stage"] = stage_name
        startup_progress["completed_stages"].append(stage_name)
        if start_s is not None:
            startup_progress["timings"][f"{stage_name}_s"] = _elapsed_s(
                start_s,
                _perf_counter_s(),
            )
        write_json_file(startup_progress_path, startup_progress)

    _mark_startup_progress("output_root_ready")
    stop_jax_profile_trace = begin_jax_profile_trace(args.jax_profile_dir)
    jax_profile_enabled = stop_jax_profile_trace is not None
    args.curvature_threshold = resolve_curvature_threshold(args.curvature_threshold)
    use_jax = args.backend == "jax"
    warm_start_state_load_start_s = _perf_counter_s()
    warm_start_state = (
        None
        if args.warm_start_run_dir is None
        else (
            load_single_stage_jax_warm_start_state(
                args.warm_start_run_dir,
                runtime_spec_path=args.jax_runtime_seed_spec,
            )
            if use_jax
            else load_single_stage_warm_start_state(args.warm_start_run_dir)
        )
    )
    _mark_startup_progress(
        "warm_start_state_loaded",
        start_s=warm_start_state_load_start_s,
    )
    nphi = args.nphi
    ntheta = args.ntheta
    mpol = args.mpol
    ntor = args.ntor
    warm_start_runtime_spec_state = None
    stage2_seed_setup_start_s = _perf_counter_s()
    if use_jax:
        jax_seed_startup_state = load_single_stage_jax_runtime_seed_startup_state(
            args,
            mpol=mpol,
            ntor=ntor,
            nphi=nphi,
            ntheta=ntheta,
        )
        warm_start_runtime_spec_state = jax_seed_startup_state["runtime_spec_state"]
        stage2_bs_path = jax_seed_startup_state["stage2_bs_path"]
        stage2_results_path = jax_seed_startup_state["stage2_results_path"]
        stage2_results = jax_seed_startup_state["stage2_results"]
        stage2_source = "jax_runtime_seed_spec"
        stage2_tf_current_limit_enforced = False
        stage2_seed_hardware_validation_enforced = False
    else:
        stage2_seed_contract = resolve_single_stage_startup_seed_contract(
            args,
            warm_start_state=warm_start_state,
        )
        stage2_bs_path = stage2_seed_contract["stage2_bs_path"]
        stage2_source = stage2_seed_contract["stage2_source"]
        stage2_tf_current_limit_enforced = stage2_seed_contract[
            "tf_current_limit_enforced"
        ]
        stage2_seed_hardware_validation_enforced = stage2_seed_contract[
            "seed_hardware_validation_enforced"
        ]
        stage2_results_path, stage2_results = load_stage2_results(stage2_bs_path)
    args.stage2_tf_current_limit_enforced = stage2_tf_current_limit_enforced
    args.stage2_seed_hardware_validation_enforced = (
        stage2_seed_hardware_validation_enforced
    )
    _mark_startup_progress(
        "stage2_seed_resolved",
        start_s=stage2_seed_setup_start_s,
    )
    R0 = float(stage2_results["MAJOR_RADIUS"])
    s = float(stage2_results["TOROIDAL_FLUX"])
    order = int(stage2_results["order"])

    if use_jax:
        if args.banana_surf_radius is not None and not np.isclose(
            float(args.banana_surf_radius),
            float(stage2_results["banana_surf_radius"]),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "JAX runtime seed spec banana_surf_radius does not match this run; "
                "run seed conversion first."
            )
        banana_surf_radius = validate_banana_winding_surface_radius(
            float(stage2_results["banana_surf_radius"])
        )
    else:
        banana_surf_radius = resolve_single_stage_banana_surface_radius(
            args,
            stage2_results,
        )
    stage2_seed_payload = build_single_stage_runtime_stage2_seed_payload(
        stage2_results,
        banana_surf_radius=banana_surf_radius,
    )
    banana_surf_nfp = 5

    # Optimization targets and weights
    vol_target = args.vol_target
    CONSTRAINT_WEIGHT = args.constraint_weight
    CONSTRAINT_METHOD = args.constraint_method
    ALM_FORMULATION = args.alm_formulation
    MAXITER = args.maxiter
    iota_target = args.iota_target
    num_tf_coils = args.num_tf_coils

    boozer_type = {"initial": "least_squares", "final": "exact"}  # example
    stage = args.boozer_stage
    if use_jax:
        if int(warm_start_runtime_spec_state["runtime_spec"].seed.num_tf_coils) != int(
            num_tf_coils
        ):
            raise ValueError(
                "JAX runtime seed spec num_tf_coils does not match this run; "
                "run seed conversion first."
            )
        if warm_start_state is not None:
            warm_start_state = {
                **warm_start_state,
                "iota": host_float(warm_start_runtime_spec_state["iota"]),
                "G": host_float(warm_start_runtime_spec_state["G"]),
                "jax_runtime_spec_path": warm_start_runtime_spec_state["path"],
            }

    # ==============================================================================
    # SURFACE GEOMETRY DEFINITIONS
    # ==============================================================================
    surface_geometry_start_s = _perf_counter_s()
    # The outer vacuum vessel of HBT, R0 = 0.976, a = 0.222
    # Solely for visualization purposes
    VV = SurfaceRZFourier(nfp=5, stellsym=True)
    VV.set_rc(0, 0, 0.976)
    VV.set_rc(1, 0, 0.222)
    VV.set_zs(1, 0, 0.222)

    # The proposed new HBT LCFS
    hbt = SurfaceRZFourier(nfp=5, stellsym=True)
    hbt.set_rc(0, 0, 0.9115)  # R0 of LCFS semi-circle center
    hbt.set_rc(1, 0, 0.1605)  # Minor radius (thick metal walls)
    hbt.set_zs(1, 0, 0.152)  # Z extent = ±0.152 m (flat top/bottom)

    # The surface the coils can lie on from Jeff - R0 = 0.976 and a=0.22
    surf_coils = SurfaceRZFourier(nfp=banana_surf_nfp, stellsym=True)
    surf_coils.set_rc(0, 0, 0.976)
    surf_coils.set_rc(1, 0, banana_surf_radius)
    surf_coils.set_zs(1, 0, banana_surf_radius)
    _mark_startup_progress(
        "surface_geometry_ready",
        start_s=surface_geometry_start_s,
    )

    # ==============================================================================
    # LOAD EQUILIBRIUM AND COILS
    # ==============================================================================
    plasma_surf_filename = args.plasma_surf_filename
    file_loc = build_equilibrium_path(args)
    write_full_artifacts = should_write_single_stage_full_artifacts(
        args.benchmark_mode,
        args.minimal_artifacts,
        backend=args.backend,
        full_artifacts=args.full_artifacts,
    )
    write_restart_artifacts = should_write_single_stage_restart_artifacts(
        args.benchmark_mode
    )

    biotsavart_wrap_start_s = _perf_counter_s()
    if use_jax:
        _, IotasJAX, NonQuasiSymmetricRatioJAX = get_jax_surface_objective_classes()
        bs = SingleStageRuntimeSpecBiotSavartJAX(
            warm_start_runtime_spec_state["runtime_spec"]
        )
        bs_cpu_diag = None
        iota_cls = IotasJAX
    else:
        stage2_bs_load_start_s = _perf_counter_s()
        bs = load(stage2_bs_path)
        _mark_startup_progress("stage2_bs_loaded", start_s=stage2_bs_load_start_s)
        bs_cpu_diag = None
        iota_cls = Iotas
    _mark_startup_progress("biotsavart_ready", start_s=biotsavart_wrap_start_s)

    bs_diag = diagnostic_field(bs, bs_cpu_diag)

    # Initialize the boundary magnetic surface and scale it to the target major radius
    plasma_surface_load_start_s = _perf_counter_s()
    if use_jax:
        surf = build_single_stage_surface_from_jax_runtime_spec(
            warm_start_runtime_spec_state["runtime_spec"]
        )
    else:
        surf = SurfaceRZFourier.from_wout(
            file_loc, range="half period", nphi=nphi, ntheta=ntheta, s=s
        )
        # scale the surface down to the target appropriate major radius
        surf.set_dofs(surf.get_dofs() * R0 / surf.major_radius())
    _mark_startup_progress(
        "plasma_surface_loaded",
        start_s=plasma_surface_load_start_s,
    )
    warm_start_surface_project_start_s = _perf_counter_s()
    if use_jax:
        warm_start_surface_dofs = warm_start_runtime_spec_state["surface_dofs"]
    elif warm_start_state is None:
        warm_start_surface_dofs = None
    else:
        warm_start_surface_dofs = project_surface_dofs_to_resolution(
            warm_start_state["surface"],
            mpol=mpol,
            ntor=ntor,
            quadpoints_phi=surf.quadpoints_phi,
            quadpoints_theta=surf.quadpoints_theta,
        )
    _mark_startup_progress(
        "warm_start_surface_projected",
        start_s=warm_start_surface_project_start_s,
    )
    single_stage_search_policy = resolve_single_stage_search_policy(
        warm_start_state,
        explicit_surface_warm_start=warm_start_surface_dofs is not None,
    )
    effective_initial_phase_settings = (
        resolve_single_stage_policy_initial_phase_settings(
            single_stage_search_policy,
            initial_step_scale=args.initial_step_scale,
            initial_step_maxiter=args.initial_step_maxiter,
            initial_step_scale_explicit=getattr(
                args,
                "initial_step_scale_explicit",
                False,
            ),
            initial_step_maxiter_explicit=getattr(
                args,
                "initial_step_maxiter_explicit",
                False,
            ),
            field_backend=args.backend,
            optimizer_backend=args.optimizer_backend,
        )
    )
    effective_initial_step_scale = effective_initial_phase_settings[
        "initial_step_scale"
    ]
    effective_initial_step_maxiter = effective_initial_phase_settings[
        "initial_step_maxiter"
    ]

    # Extract coil information
    coils = bs.coils
    curves = [c.curve for c in coils]
    tf_coils = coils[:num_tf_coils]
    tf_curves = [c.curve for c in tf_coils]
    banana_coils = coils[num_tf_coils:]
    banana_curves = [c.curve for c in banana_coils]
    banana_curve = banana_curves[0]
    stage2_tf_current_A = (
        float(warm_start_runtime_spec_state["runtime_spec"].seed.tf_current_A)
        if use_jax
        else resolve_loaded_tf_current_A(
            stage2_results.get("TF_CURRENT_A"),
            tf_coils,
            enforce_limit=stage2_tf_current_limit_enforced,
        )
    )
    if (
        not stage2_tf_current_limit_enforced
        and stage2_tf_current_A > TF_CURRENT_HARD_LIMIT_A
    ):
        print(
            "Warning: loaded continuation seed TF current "
            f"{stage2_tf_current_A:.6f} A exceeds the configured hard limit "
            f"{TF_CURRENT_HARD_LIMIT_A:.6f} A. Startup is continuing because "
            f"the seed source {stage2_source!r} was supplied explicitly rather "
            "than derived from the default Stage 2 hardware-valid archive."
        )
    current_sum = sum(abs(c.current.get_value()) for c in tf_coils)

    if CONSTRAINT_METHOD == "penalty":
        banana_seed_current_A = float(banana_coils[0].current.get_value())
        if (
            not stage2_seed_hardware_validation_enforced
            and banana_seed_current_A > float(args.banana_current_max_A)
        ):
            print(
                "Warning: loaded continuation seed banana current "
                f"{banana_seed_current_A:.6f} A exceeds the configured traversal "
                f"box bound {float(args.banana_current_max_A):.6f} A. Startup is "
                "continuing because the seed source was supplied explicitly rather "
                "than derived from the default Stage 2 hardware-valid archive."
            )
        apply_penalty_traversal_forbidden_box_bounds(
            bound_targets={"banana_current": banana_coils[0].current},
            requested_thresholds={"banana_current": args.banana_current_max_A},
            seed_values={"banana_current": banana_seed_current_A},
            validate_seed=stage2_seed_hardware_validation_enforced,
            seed_context="Loaded Stage 2 seed banana_current",
        )

    # Calculate G0 parameter from TF coil currents
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 10 ** (-7) / (2 * np.pi))

    # ==============================================================================
    # OPTIMIZATION SETUP
    # ==============================================================================
    print(f"\n===== Starting single stage optimization for mpol = {mpol} =====")
    print(
        "Single-stage search policy "
        f"(donor_class={single_stage_search_policy.donor_class}, "
        f"policy={single_stage_search_policy.search_policy}, "
        f"adaptive_failure_penalty_weight="
        f"{single_stage_search_policy.adaptive_failure_penalty_weight:g}, "
        f"invalid_step_retry_budget="
        f"{single_stage_search_policy.invalid_step_retry_budget}, "
        f"retry_step_shrink_factor="
        f"{single_stage_search_policy.retry_step_shrink_factor:g}, "
        f"initial_phase_auto_enabled={effective_initial_phase_settings['auto_enabled']})"
    )

    optimizer_backend_record = args.optimizer_backend if args.backend == "jax" else None
    boozer_optimizer_backend_record = resolve_boozer_optimizer_backend(
        args.backend,
        args.optimizer_backend,
        args.boozer_optimizer_backend,
    )
    boozer_least_squares_algorithm_record = (
        getattr(args, "boozer_least_squares_algorithm", None)
        if args.backend == "jax"
        else None
    )
    target_lane_boozer_bfgs_tol_record = (
        args.target_lane_boozer_bfgs_tol
        if args.backend == "jax" and optimizer_backend_record == "ondevice"
        else None
    )
    target_lane_boozer_bfgs_maxiter_record = (
        args.target_lane_boozer_bfgs_maxiter
        if args.backend == "jax" and optimizer_backend_record == "ondevice"
        else None
    )
    target_lane_boozer_newton_tol_record = (
        args.target_lane_boozer_newton_tol
        if args.backend == "jax" and optimizer_backend_record == "ondevice"
        else None
    )
    target_lane_boozer_newton_maxiter_record = (
        args.target_lane_boozer_newton_maxiter
        if args.backend == "jax" and optimizer_backend_record == "ondevice"
        else None
    )
    boozer_optimizer_backend_hash_record = (
        args.boozer_optimizer_backend if args.backend == "jax" else None
    )
    effective_target_lane_sync_policy = (
        resolve_effective_target_lane_accepted_step_sync(
            args.target_lane_accepted_step_sync,
            benchmark_mode=args.benchmark_mode,
        )
    )
    (
        requested_experimental_target_lane_vg,
        use_target_lane_vg,
    ) = resolve_target_lane_value_and_grad_modes(
        backend=args.backend,
        optimizer_backend=optimizer_backend_record,
        experimental_enabled=args.experimental_target_lane_value_and_grad,
    )
    target_lane_sync_record = resolve_target_lane_accepted_step_sync_record(
        backend=args.backend,
        optimizer_backend=optimizer_backend_record,
        maxiter=args.maxiter,
        sync_policy=effective_target_lane_sync_policy,
    )
    effective_boozer_limited_memory = resolve_single_stage_boozer_limited_memory(
        args.backend,
        optimizer_backend_record,
        boozer_optimizer_backend_record,
        args.boozer_limited_memory,
    )
    base_boozer_init_overrides = resolve_target_lane_boozer_init_base_overrides(
        field_backend=args.backend,
        optimizer_backend=optimizer_backend_record,
        boozer_limited_memory=effective_boozer_limited_memory,
        target_lane_boozer_bfgs_tol=target_lane_boozer_bfgs_tol_record,
        target_lane_boozer_bfgs_maxiter=target_lane_boozer_bfgs_maxiter_record,
        target_lane_boozer_newton_tol=target_lane_boozer_newton_tol_record,
        target_lane_boozer_newton_maxiter=target_lane_boozer_newton_maxiter_record,
    )
    warm_start_boozer_init_overrides = resolve_warm_start_boozer_init_overrides(
        warm_start_state=warm_start_state,
        explicit_surface_warm_start=warm_start_surface_dofs is not None,
        field_backend=args.backend,
        optimizer_backend=optimizer_backend_record,
        boozer_optimizer_backend=boozer_optimizer_backend_record,
        boozer_least_squares_algorithm=boozer_least_squares_algorithm_record,
        boozer_least_squares_algorithm_explicit=getattr(
            args,
            "boozer_least_squares_algorithm_explicit",
            False,
        ),
        target_lane_boozer_bfgs_tol=target_lane_boozer_bfgs_tol_record,
        target_lane_boozer_bfgs_maxiter=target_lane_boozer_bfgs_maxiter_record,
    )
    effective_boozer_init_overrides = dict(base_boozer_init_overrides)
    for key, value in warm_start_boozer_init_overrides.items():
        if value is not None:
            effective_boozer_init_overrides[key] = value
    effective_boozer_init_least_squares_algorithm = (
        effective_boozer_init_overrides["least_squares_algorithm_override"]
        or boozer_least_squares_algorithm_record
    )

    config_parts = [
        str(stage2_bs_path),
        str(stage),
        str(CONSTRAINT_WEIGHT),
        str(CONSTRAINT_METHOD),
        str(ALM_FORMULATION if CONSTRAINT_METHOD == "alm" else None),
        str(args.alm_qs_threshold if CONSTRAINT_METHOD == "alm" else None),
        str(args.alm_boozer_threshold if CONSTRAINT_METHOD == "alm" else None),
        str(args.alm_iota_penalty_threshold if CONSTRAINT_METHOD == "alm" else None),
        str(args.alm_length_penalty_threshold if CONSTRAINT_METHOD == "alm" else None),
        str(vol_target),
        str(iota_target),
        str(args.cc_dist),
        str(args.cc_weight),
        str(args.curvature_weight),
        str(args.curvature_threshold),
        str(args.length_weight),
        str(args.length_target),
        str(args.res_weight),
        str(args.iotas_weight),
        str(args.cs_weight),
        str(args.cs_dist),
        str(args.surf_dist_weight),
        str(args.ss_dist),
        str(args.banana_current_max_A),
        str(args.maxcor),
        str(args.outer_maxls),
        str(args.outer_ftol),
        str(args.target_lane_outer_initial_step_size),
        str(effective_initial_phase_settings["initial_step_scale"]),
        str(effective_initial_phase_settings["initial_step_maxiter"]),
        str(target_lane_boozer_bfgs_tol_record),
        str(target_lane_boozer_bfgs_maxiter_record),
        str(target_lane_boozer_newton_tol_record),
        str(target_lane_boozer_newton_maxiter_record),
        str(single_stage_search_policy.donor_class),
        str(single_stage_search_policy.search_policy),
        str(single_stage_search_policy.adaptive_failure_penalty_weight),
        str(single_stage_search_policy.invalid_step_retry_budget),
        str(single_stage_search_policy.retry_step_shrink_factor),
        str(effective_initial_phase_settings["auto_enabled"]),
        str(banana_surf_radius),
        str(nphi),
        str(ntheta),
        str(args.init_only),
        str(args.benchmark_mode),
        str(args.disable_target_lane_success_filter),
        str(args.profile_target_lane),
        str(args.profile_target_lane_only),
        str(args.diagnose_target_lane_gradient),
        str(getattr(args, "diagnose_target_lane_first_line_search", False)),
        str(args.minimal_artifacts),
        str(args.backend),
        str(optimizer_backend_record),
        str(target_lane_sync_record),
        str(use_target_lane_vg),
        str(args.maxiter),
        str(args.num_tf_coils),
        str(stage2_tf_current_A),
        str(file_loc),
    ]
    if boozer_optimizer_backend_hash_record is not None:
        config_parts.append(str(boozer_optimizer_backend_hash_record))
    if effective_boozer_init_least_squares_algorithm is not None:
        config_parts.append(str(effective_boozer_init_least_squares_algorithm))
    config_parts.append(str(effective_boozer_limited_memory))
    config_str = "|".join(config_parts)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    OUT_DIR_ITER = OUT_DIR + f"/mpol={mpol}-ntor={ntor}-{config_hash}"
    os.makedirs(OUT_DIR_ITER, exist_ok=True)
    _mark_startup_progress("optimizer_output_dir_ready")
    boozer_init_progress = build_stage_progress_recorder(
        os.path.join(OUT_DIR_ITER, "boozer_init_progress.json")
    )
    outer_contract = resolve_single_stage_optimizer_contract(
        args.backend,
        args.optimizer_backend,
    )
    from simsopt.geo.optimizer_jax import TargetOptimizerContract

    use_target_lane = CONSTRAINT_METHOD == "penalty" and isinstance(
        outer_contract,
        TargetOptimizerContract,
    )
    require_single_stage_jax_target_lane(
        use_jax=use_jax,
        use_target_lane=use_target_lane,
    )
    outer_optimizer_progress = (
        build_event_progress_recorder(
            os.path.join(OUT_DIR_ITER, "outer_optimizer_progress.json")
        )
        if should_record_single_stage_outer_optimizer_progress(use_target_lane)
        else None
    )

    def record_outer_optimizer_event(label, **extra):
        if outer_optimizer_progress is None:
            return
        outer_optimizer_progress(label, **extra)

    record_outer_optimizer_event(
        "pre_optimizer_startup_ready",
        use_target_lane=bool(use_target_lane),
        output_dir=OUT_DIR_ITER,
        optimizer_method=getattr(outer_contract, "method", None),
    )
    boozer_init_progress(
        "starting",
        backend=args.backend,
        optimizer_backend=boozer_optimizer_backend_record,
        warm_start=("true" if warm_start_state is not None else "false"),
        requested_boozer_least_squares_algorithm=(
            boozer_least_squares_algorithm_record
        ),
        effective_boozer_least_squares_algorithm=(
            effective_boozer_init_least_squares_algorithm
        ),
        requested_boozer_limited_memory=args.boozer_limited_memory,
        effective_boozer_limited_memory=effective_boozer_limited_memory,
        boozer_least_squares_algorithm_override=effective_boozer_init_overrides[
            "least_squares_algorithm_override"
        ],
        bfgs_tol_override=effective_boozer_init_overrides["bfgs_tol_override"],
        bfgs_maxiter_override=effective_boozer_init_overrides["bfgs_maxiter_override"],
        newton_tol_override=effective_boozer_init_overrides["newton_tol_override"],
        newton_maxiter_override=effective_boozer_init_overrides[
            "newton_maxiter_override"
        ],
    )

    # Initialize Boozer surface with target parameters
    timings = {}
    record_outer_optimizer_event("boozer_init_started")
    jax_seed_iota_override = (
        None
        if warm_start_runtime_spec_state is None
        else host_float(warm_start_runtime_spec_state["iota"])
    )
    jax_seed_G_override = (
        None
        if warm_start_runtime_spec_state is None
        else host_float(warm_start_runtime_spec_state["G"])
    )
    with maybe_trace_single_stage_phase(
        "single_stage.initialize_boozer_surface",
        enabled=jax_profile_enabled,
    ):
        boozer_surface = initialize_boozer_surface(
            surf,
            mpol,
            ntor,
            bs,
            vol_target,
            CONSTRAINT_WEIGHT,
            iota_target,
            G0,
            backend=args.backend,
            optimizer_backend=boozer_optimizer_backend_record,
            boozer_least_squares_algorithm=(
                effective_boozer_init_least_squares_algorithm
            ),
            boozer_limited_memory=effective_boozer_limited_memory,
            bfgs_tol_override=effective_boozer_init_overrides["bfgs_tol_override"],
            bfgs_maxiter_override=effective_boozer_init_overrides[
                "bfgs_maxiter_override"
            ],
            newton_tol_override=effective_boozer_init_overrides["newton_tol_override"],
            newton_maxiter_override=effective_boozer_init_overrides[
                "newton_maxiter_override"
            ],
            surface_dofs_override=warm_start_surface_dofs,
            iota_override=jax_seed_iota_override
            if use_jax
            else (None if warm_start_state is None else warm_start_state["iota"]),
            G_override=jax_seed_G_override
            if use_jax
            else (None if warm_start_state is None else warm_start_state["G"]),
            on_stage=boozer_init_progress,
            timings_out=timings,
        )
    boozer_init_progress("completed")
    record_outer_optimizer_event(
        "boozer_init_returned",
        solve_success=bool(boozer_surface.res.get("success", False)),
        iterations=None
        if boozer_surface.res.get("iter", None) is None
        else _summarize_host_scalar(boozer_surface.res["iter"]),
    )

    # ==============================================================================
    # SAVE INITIAL STATE
    # ==============================================================================
    initial_artifacts_start_s = _perf_counter_s()
    record_outer_optimizer_event(
        "initial_diagnostics_started",
        write_full_artifacts=bool(write_full_artifacts),
    )
    if write_full_artifacts:
        # Save initial coil configurations
        curves_to_vtk(curves, OUT_DIR_ITER + "/curves_init", close=True)
        bs_diag.save(OUT_DIR_ITER + "/biot_savart_init.json")

        # Save initial surface with magnetic field normal component data
        bs_diag.set_points(boozer_surface.surface.gamma().reshape((-1, 3)))
        unitn = boozer_surface.surface.unitnormal()
        pointData = {
            "B_N/B": np.sum(bs_diag.B().reshape(unitn.shape) * unitn, axis=2)[
                :, :, None
            ]
            / np.sqrt(np.sum(bs_diag.B().reshape(unitn.shape) ** 2, axis=2))[:, :, None]
        }
        boozer_surface.surface.to_vtk(OUT_DIR_ITER + "/surf_init", extra_data=pointData)
        boozer_surface.surface.save(OUT_DIR_ITER + "/surf_init.json")
    print(f"Volume: {host_float(boozer_surface.surface.volume())}")

    # Generate initial diagnostic plots
    if write_full_artifacts:
        initial_field_error = normPlot(
            boozer_surface.surface, bs_diag, OUT_DIR_ITER + "/NormPlotInitial"
        )
        cross_section_plot(
            surf_coils,
            boozer_surface.surface,
            banana_curve,
            OUT_DIR_ITER + "/CrossSectionInitial",
            hbt,
            VV,
        )
    else:
        initial_field_error, _, _, _, _, _ = norm_field_summary(
            boozer_surface.surface,
            bs_diag,
        )
    initial_volume = host_float(boozer_surface.surface.volume())
    initial_iota = resolve_single_stage_iota_metric(
        boozer_surface,
        iota_cls,
        benchmark_mode=args.benchmark_mode,
    )
    initial_max_curvature = _host_curve_max_curvature(banana_curve)
    _record_timing(
        timings,
        "initial_artifacts_s",
        initial_artifacts_start_s,
        _perf_counter_s(),
    )
    record_outer_optimizer_event(
        "initial_diagnostics_returned",
        elapsed_s=float(_perf_counter_s() - initial_artifacts_start_s),
        initial_volume=_summarize_host_scalar(initial_volume),
        initial_iota=_summarize_host_scalar(initial_iota),
        initial_field_error=_summarize_host_scalar(initial_field_error),
        initial_max_curvature=_summarize_host_scalar(initial_max_curvature),
    )

    # ==============================================================================
    # DEFINE OBJECTIVE FUNCTION COMPONENTS
    # ==============================================================================
    objective_setup_start_s = _perf_counter_s()
    record_outer_optimizer_event("objective_setup_started")
    # Biot-Savart field calculation
    if use_jax:
        bs_obj = bs
    else:
        bs_obj = BiotSavart(coils)

    boozer_residual_cls = select_boozer_residual_class(
        use_jax=use_jax,
        boozer_kind=boozer_type[stage],
    )

    # Quasi-symmetry and Boozer coordinate residuals
    if use_jax:
        nonQSs = [NonQuasiSymmetricRatioJAX(boozer_surface, bs_obj)]
    else:
        nonQSs = [NonQuasiSymmetricRatio(boozer_surface, bs_obj)]
    brs = [build_boozer_residual_objective(boozer_surface, bs_obj, boozer_residual_cls)]

    # Objective function weights and parameters
    LENGTH_WEIGHT = args.length_weight
    RES_WEIGHT = args.res_weight
    IOTAS_WEIGHT = args.iotas_weight
    CC_WEIGHT = args.cc_weight
    CC_DIST = max(args.cc_dist, COIL_COIL_MIN_DIST_M)
    CS_WEIGHT = args.cs_weight
    CS_DIST = max(args.cs_dist, COIL_PLASMA_MIN_DIST_M)
    SURF_DIST_WEIGHT = args.surf_dist_weight
    SS_DIST = max(args.ss_dist, PLASMA_VESSEL_MIN_DIST_M)
    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = args.curvature_threshold

    # Individual objective terms
    iota = build_iota_objective(boozer_surface, iota_cls)
    curvelength = CurveLength(banana_curves[0])
    length_target = min(float(args.length_target), COIL_LENGTH_TARGET_M)

    Jiota = QuadraticPenalty(iota, iota_target)
    JnonQSRatio = sum(nonQSs)
    JBoozerResidual = sum(brs)
    JCurveLength = QuadraticPenalty(curvelength, length_target, "max")
    JCurveCurve = CurveCurveDistance(curves, CC_DIST)
    JCurveSurface = CurveSurfaceDistance(curves, boozer_surface.surface, CS_DIST)
    JSurfSurf = SurfaceSurfaceDistance(boozer_surface.surface, VV, SS_DIST)
    JCurvature = LpCurveCurvature(banana_curves[0], 2, CURVATURE_THRESHOLD)

    # Combined objective function
    JF = (
        JnonQSRatio
        + RES_WEIGHT * JBoozerResidual
        + IOTAS_WEIGHT * Jiota
        + LENGTH_WEIGHT * JCurveLength
        + CC_WEIGHT * JCurveCurve
        + CS_WEIGHT * JCurveSurface
        + SURF_DIST_WEIGHT * JSurfSurf
        + CURVATURE_WEIGHT * JCurvature
    )
    _record_timing(
        timings,
        "objective_setup_s",
        objective_setup_start_s,
        _perf_counter_s(),
    )
    record_outer_optimizer_event(
        "objective_setup_returned",
        elapsed_s=float(_perf_counter_s() - objective_setup_start_s),
    )

    # ==============================================================================
    # SNAPSHOT PRE-OPTIMIZATION STATE
    # ==============================================================================
    dof_setter = resolve_single_stage_outer_dof_setter(
        JF,
        bs,
        use_target_lane=use_target_lane,
    )
    dofs = resolve_single_stage_outer_optimizer_initial_dofs(
        JF,
        bs,
        use_target_lane=use_target_lane,
    )
    snapshot_start_s = _perf_counter_s()
    record_outer_optimizer_event(
        "snapshot_started",
        evaluate_initial_objective=not bool(use_target_lane),
        optimizer_dofs=_summarize_host_vector(dofs),
    )
    dofs, run_dict, static_config = snapshot_to_pytree(
        JF,
        boozer_surface,
        bs,
        num_tf_coils=num_tf_coils,
        coil_dofs_override=dofs,
        evaluate_initial_objective=not use_target_lane,
    )
    record_outer_optimizer_event(
        "snapshot_returned",
        elapsed_s=float(_perf_counter_s() - snapshot_start_s),
        initial_objective_pending=bool(
            run_dict.get("initial_objective_pending", False)
        ),
    )
    outer_optimizer_method_record = (
        "alm" if CONSTRAINT_METHOD == "alm" else outer_contract.method
    )
    alm_inner_optimizer_contract = resolve_single_stage_alm_inner_optimizer_contract(
        CONSTRAINT_METHOD,
        outer_contract,
    )
    run_dict["x_prev"] = _single_stage_optimizer_dofs_array(dofs).copy()
    run_dict["donor_class"] = single_stage_search_policy.donor_class
    run_dict["search_policy"] = single_stage_search_policy.search_policy
    run_dict["adaptive_failure_penalty_weight"] = (
        single_stage_search_policy.adaptive_failure_penalty_weight
    )
    run_dict["initial_phase_auto_enabled"] = bool(
        effective_initial_phase_settings["auto_enabled"]
    )
    run_dict["constraint_method"] = CONSTRAINT_METHOD
    run_dict["alm_formulation"] = (
        ALM_FORMULATION if CONSTRAINT_METHOD == "alm" else None
    )
    run_dict["accepted_iterations"] = 0
    run_dict["accepted_boozer_stage"] = stage
    run_dict["accepted_hardware_status"] = None
    run_dict["trial_hardware_status"] = None
    objectives = {
        "qs": JnonQSRatio,
        "boozer": JBoozerResidual,
        "iota_penalty": Jiota,
        "length": JCurveLength,
        "cc": JCurveCurve,
        "cs": JCurveSurface,
        "surf": JSurfSurf,
        "curvature": JCurvature,
    }
    diagnostics_refs = {
        "iota": iota,
        "banana_curve": banana_curve,
        "curvelength": curvelength,
    }
    initial_hardware_start_s = _perf_counter_s()
    record_outer_optimizer_event("initial_hardware_status_started")
    run_dict["hardware_constraint_status"] = _evaluate_single_stage_hardware_status(
        objectives,
        diagnostics_refs,
    )
    record_outer_optimizer_event(
        "initial_hardware_status_returned",
        elapsed_s=float(_perf_counter_s() - initial_hardware_start_s),
        success=run_dict["hardware_constraint_status"].get("success"),
    )
    update_self_intersection_status(
        run_dict,
        boozer_surface.surface,
        require_supported_surface=_boozer_surface_requires_jax_supported_self_intersection(
            boozer_surface
        ),
    )
    if not bool(run_dict.get("initial_objective_pending", False)):
        record_single_stage_local_incumbent(run_dict, stage="initial")
    target_lane_profile = None
    target_lane_optimizer_initial_value_and_grad = None
    adapter = SingleStageAdapter(
        run_dict=run_dict,
        boozer_surface=boozer_surface,
        JF=JF,
        bs=bs,
        objectives=objectives,
        diagnostics=diagnostics_refs,
        log_path=OUT_DIR_ITER + "/log.txt",
        reevaluate_before_accept=False,
        apply_coil_dofs=dof_setter,
        benchmark_mode=args.benchmark_mode,
        accepted_step_state_sync=None,
    )

    # ==============================================================================
    # RUN OPTIMIZATION
    # ==============================================================================
    # Get convergence tolerances for current mpol
    ftol = (
        float(args.outer_ftol)
        if args.outer_ftol is not None
        else ftol_by_mpol.get(mpol, 1e-5 if mpol < 8 else 1e-10)
    )
    gtol = gtol_by_mpol.get(mpol, 1e-2 if mpol < 8 else 1e-7)
    phase1_iterations = None
    phase1_termination_message = None
    phase1_success = None
    main_phase_iterations = None
    target_lane_gradient_diagnosis = None
    target_lane_first_line_search_diagnosis = None
    target_lane_scaled_phase1_diagnosis = None
    target_lane_invalid_state_events = []
    target_lane_invalid_state_diagnostic_events = []
    phase1_retry_summary = {
        "attempt_count": 0,
        "attempts": [],
        "restored_preserved_local_state": False,
        "restored_preserved_local_stage": None,
    }
    target_lane_retry_summary = {
        "attempt_count": 0,
        "attempts": [],
        "restored_preserved_local_state": False,
        "restored_preserved_local_stage": None,
    }
    jax_compile_diagnostics = None
    record_target_lane_invalid_state_events = (
        record_target_lane_invalid_state_events_enabled(args)
    )
    diagnostic_target_lane_callbacks = bool(
        use_target_lane and target_lane_diagnostic_callbacks_enabled(args)
    )

    def build_outer_optimizer_progress_callback(phase):
        if outer_optimizer_progress is None:
            return None

        def _record(iteration, fun_value, grad_inf):
            record_outer_optimizer_event(
                f"{phase}_progress_iter_{int(iteration)}",
                phase=phase,
                iteration=int(iteration),
                fun_value=_summarize_host_scalar(fun_value),
                grad_inf=_summarize_host_scalar(grad_inf),
            )

        return _record

    def wrap_outer_optimizer_failure_callback(callback, *, phase):
        if callback is None or outer_optimizer_progress is None:
            return callback

        def _record(
            iteration,
            trial_x,
            trial_f,
            trial_g,
            search_direction,
            step_vector,
            step_scale,
            line_search_failed,
            nonfinite_step,
            stalled_step,
            valid_curvature,
            trial_converged,
            ls_status,
        ):
            callback(
                iteration,
                trial_x,
                trial_f,
                trial_g,
                search_direction,
                step_vector,
                step_scale,
                line_search_failed,
                nonfinite_step,
                stalled_step,
                valid_curvature,
                trial_converged,
                ls_status,
            )
            record_outer_optimizer_event(
                f"{phase}_failure_iter_{int(iteration)}",
                phase=phase,
                iteration=int(iteration),
                line_search_failed=bool(line_search_failed),
                nonfinite_step=bool(nonfinite_step),
                stalled_step=bool(stalled_step),
                valid_curvature=bool(valid_curvature),
                trial_converged=bool(trial_converged),
                ls_status=int(ls_status),
                step_scale=_summarize_host_scalar(step_scale),
                trial_value=_summarize_host_scalar(trial_f),
                trial_grad=_summarize_host_vector(trial_g),
                trial_x=_summarize_host_vector(trial_x),
                search_direction=_summarize_host_vector(search_direction),
                step_vector=_summarize_host_vector(step_vector),
            )

        return _record

    skip_outer_optimizer = bool(
        args.init_only
        or (
            CONSTRAINT_METHOD == "penalty"
            and (
                args.profile_target_lane_only
                or args.diagnose_target_lane_gradient
                or args.diagnose_target_lane_first_line_search
                or args.diagnose_target_lane_scaled_phase1
            )
        )
    )
    res = None
    alm_result = None
    target_lane_trial_boozer_override_active = False
    accepted_step_callback = None
    target_lane_success_filter = None
    host_state_restored_for_final = False
    host_postprocess_result = None
    record_outer_optimizer_event(
        "starting",
        use_target_lane=bool(use_target_lane),
        maxiter=int(MAXITER),
        initial_step_scale=float(effective_initial_step_scale),
        initial_step_maxiter=int(effective_initial_step_maxiter),
        target_lane_outer_initial_step_size=args.target_lane_outer_initial_step_size,
        record_target_lane_invalid_state_events=bool(
            record_target_lane_invalid_state_events
        ),
        diagnostic_callbacks=diagnostic_target_lane_callbacks,
    )
    if diagnostic_target_lane_callbacks:
        print(
            "Enabling target-lane accepted-step, progress, and failure host "
            "callbacks for diagnostic run."
        )

    if args.init_only:
        res_nit = 0
        optimizer_success = True
        termination_message = "init_only"
        final_volume = initial_volume
        final_iota = initial_iota
        final_max_curvature = initial_max_curvature
        fieldError = initial_field_error
        print("Skipping single-stage optimizer because --init-only was provided.")
    else:
        outer_optimizer_start_s = _perf_counter_s()
        outer_optimizer_run_start_s = outer_optimizer_start_s
        if CONSTRAINT_METHOD == "alm":
            alm_settings = build_single_stage_alm_settings(args)
            alm_constraint_names = single_stage_alm_constraint_names(
                alm_formulation=ALM_FORMULATION,
                include_surface_surface=JSurfSurf is not None,
            )
            alm_partial_state = {"history": []}
            initial_alm_multipliers = np.zeros(len(alm_constraint_names), dtype=float)
            initial_alm_penalty = float(args.alm_penalty_init)
            alm_target_value_and_grad = None
            alm_target_success_filter = None
            if alm_inner_optimizer_contract is not None:
                alm_target_success_filter = (
                    build_single_stage_target_lane_self_intersection_success_filter(
                        boozer_surface,
                        bs,
                    )
                )
                alm_outer_objective_config = build_target_lane_outer_objective_config(
                    boozer_surface,
                    bs,
                    banana_curve,
                    VV,
                    non_qs_weight=1.0,
                    residual_weight=RES_WEIGHT,
                    iota_weight=IOTAS_WEIGHT,
                    length_weight=LENGTH_WEIGHT,
                    length_target=length_target,
                    curve_curve_threshold=CC_DIST,
                    curve_curve_weight=CC_WEIGHT,
                    curve_surface_threshold=CS_DIST,
                    curve_surface_weight=CS_WEIGHT,
                    surface_vessel_threshold=SS_DIST,
                    surface_vessel_weight=SURF_DIST_WEIGHT,
                    curvature_threshold=CURVATURE_THRESHOLD,
                    curvature_weight=CURVATURE_WEIGHT,
                )
                alm_runtime_config = build_traceable_single_stage_alm_runtime_config(
                    constraint_names=alm_constraint_names,
                    alm_formulation=ALM_FORMULATION,
                    distance_smoothing=args.alm_distance_smoothing,
                    curvature_smoothing=args.alm_curvature_smoothing,
                    qs_threshold=args.alm_qs_threshold,
                    boozer_threshold=args.alm_boozer_threshold,
                    iota_penalty_threshold=args.alm_iota_penalty_threshold,
                    length_penalty_threshold=args.alm_length_penalty_threshold,
                    banana_current_threshold=args.banana_current_max_A,
                )
                alm_runtime_bundle = (
                    get_traceable_single_stage_alm_runtime_bundle_builder()(
                        boozer_surface,
                        bs,
                        iota_target,
                        outer_objective_config=alm_outer_objective_config,
                        alm_config=alm_runtime_config,
                        success_filter=alm_target_success_filter,
                    )
                )
                alm_target_value_and_grad = alm_runtime_bundle["value_and_grad"]

            def set_alm_runtime_state(multipliers, penalty, *, outer_iteration=None):
                run_dict["alm_multipliers"] = np.asarray(
                    multipliers, dtype=float
                ).copy()
                run_dict["alm_penalty"] = float(penalty)
                if outer_iteration is not None:
                    run_dict["alm_outer_iteration"] = int(outer_iteration)

            def emit_alm_partial_state(
                multipliers,
                penalty,
                *,
                outer_iteration=None,
                latest_history_entry=None,
                termination_message=None,
                optimizer_success=None,
                termination_reason=None,
                inner_optimizer_success=None,
                inner_optimizer_message=None,
                converged_to_tolerances=None,
                restored_best_feasible=None,
                restored_best_feasible_reason=None,
                final_max_feasibility_violation=None,
                final_stationarity_norm=None,
            ):
                payload = build_single_stage_alm_partial_state(
                    run_dict,
                    alm_constraint_names,
                    alm_partial_state["history"],
                    latest_history_entry,
                    multipliers,
                    penalty,
                    outer_iteration=outer_iteration,
                    termination_message=termination_message,
                    optimizer_success=optimizer_success,
                    termination_reason=termination_reason,
                    inner_optimizer_success=inner_optimizer_success,
                    inner_optimizer_message=inner_optimizer_message,
                    converged_to_tolerances=converged_to_tolerances,
                    restored_best_feasible=restored_best_feasible,
                    restored_best_feasible_reason=restored_best_feasible_reason,
                    final_max_feasibility_violation=final_max_feasibility_violation,
                    final_stationarity_norm=final_stationarity_norm,
                )
                write_single_stage_alm_partial_state(OUT_DIR_ITER, payload)

            def accepted_callback(x):
                adapter.callback(x)
                run_dict["accepted_iterations"] = (
                    int(run_dict.get("accepted_iterations", 0)) + 1
                )
                run_dict["accepted_boozer_stage"] = stage
                run_dict["accepted_hardware_status"] = copy.deepcopy(
                    run_dict.get("hardware_constraint_status")
                )

            def evaluate_problem(inner_x, multipliers, penalty):
                set_alm_runtime_state(multipliers, penalty)
                return evaluate_single_stage_alm_problem(
                    inner_x,
                    run_dict=run_dict,
                    boozer_surface=boozer_surface,
                    JF=JF,
                    nonQSs=nonQSs,
                    brs=brs,
                    RES_WEIGHT=RES_WEIGHT,
                    Jiota=Jiota,
                    IOTAS_WEIGHT=IOTAS_WEIGHT,
                    curvelength=curvelength,
                    JCurveLength=JCurveLength,
                    LENGTH_WEIGHT=LENGTH_WEIGHT,
                    JCurveCurve=JCurveCurve,
                    JCurveSurface=JCurveSurface,
                    JCurvature=JCurvature,
                    JSurfSurf=JSurfSurf,
                    curves=curves,
                    banana_curve=banana_curve,
                    vessel_surface=VV,
                    curve_curve_distance=CC_DIST,
                    curve_surface_distance=CS_DIST,
                    surface_vessel_distance=SS_DIST,
                    curvature_threshold=CURVATURE_THRESHOLD,
                    distance_smoothing=args.alm_distance_smoothing,
                    curvature_smoothing=args.alm_curvature_smoothing,
                    constraint_names=alm_constraint_names,
                    alm_formulation=ALM_FORMULATION,
                    qs_threshold=args.alm_qs_threshold,
                    boozer_threshold=args.alm_boozer_threshold,
                    iota_penalty_threshold=args.alm_iota_penalty_threshold,
                    length_penalty_threshold=args.alm_length_penalty_threshold,
                    banana_current=banana_coils[0].current,
                    banana_current_threshold=args.banana_current_max_A,
                    tf_current_A=stage2_tf_current_A,
                    tf_current_limit_A=TF_CURRENT_HARD_LIMIT_A,
                    coil_length_threshold=length_target,
                    multipliers=multipliers,
                    penalty=penalty,
                )

            def outer_state_callback(outer_iteration, multipliers, penalty):
                set_alm_runtime_state(
                    multipliers,
                    penalty,
                    outer_iteration=outer_iteration,
                )
                print(
                    f"[ALM] outer_iteration={outer_iteration}, "
                    f"multipliers={np.asarray(multipliers, dtype=float).tolist()}, "
                    f"penalty={float(penalty):.3e}"
                )
                emit_alm_partial_state(
                    multipliers,
                    penalty,
                    outer_iteration=outer_iteration,
                )

            def history_callback(history, latest_history_entry, multipliers, penalty):
                alm_partial_state["history"] = history
                emit_alm_partial_state(
                    multipliers,
                    penalty,
                    outer_iteration=(
                        None
                        if latest_history_entry is None
                        else latest_history_entry.get("outer_iteration")
                    ),
                    latest_history_entry=latest_history_entry,
                )

            def snapshot_accepted_state():
                return snapshot_single_stage_local_incumbent_state(run_dict)

            def restore_incumbent_state(incumbent_state):
                restore_single_stage_local_incumbent_state(run_dict, incumbent_state)
                dof_setter(run_dict["x_prev"])

            set_alm_runtime_state(
                initial_alm_multipliers,
                initial_alm_penalty,
                outer_iteration=0,
            )
            emit_alm_partial_state(
                initial_alm_multipliers,
                initial_alm_penalty,
                outer_iteration=0,
            )
            res = minimize_alm(
                dofs,
                alm_constraint_names,
                evaluate_problem,
                alm_settings,
                {
                    "maxiter": int(MAXITER),
                    "maxcor": int(args.maxcor),
                    "ftol": float(ftol),
                    "gtol": float(gtol),
                    "maxls": int(args.outer_maxls),
                },
                inner_optimizer_contract=alm_inner_optimizer_contract,
                target_inner_value_and_grad=alm_target_value_and_grad,
                accepted_callback=accepted_callback,
                outer_state_callback=outer_state_callback,
                history_callback=history_callback,
                snapshot_accepted_state_fn=snapshot_accepted_state,
                restore_incumbent_state_fn=restore_incumbent_state,
                initial_multipliers=initial_alm_multipliers,
                initial_penalty=initial_alm_penalty,
            )
            alm_result = res
            alm_partial_state["history"] = [
                dict(entry) for entry in getattr(res, "history", [])
            ]
            set_alm_runtime_state(
                res.multipliers,
                res.penalty,
                outer_iteration=getattr(res, "outer_iterations", None),
            )
            emit_alm_partial_state(
                res.multipliers,
                res.penalty,
                outer_iteration=getattr(res, "outer_iterations", None),
                latest_history_entry=(
                    None if not getattr(res, "history", None) else res.history[-1]
                ),
                termination_message=str(res.message),
                optimizer_success=bool(res.success),
                termination_reason=getattr(res, "termination_reason", None),
                inner_optimizer_success=getattr(res, "optimizer_success", None),
                inner_optimizer_message=getattr(res, "optimizer_message", None),
                converged_to_tolerances=getattr(
                    res,
                    "converged_to_tolerances",
                    None,
                ),
                restored_best_feasible=getattr(res, "restored_best_feasible", None),
                restored_best_feasible_reason=getattr(
                    res,
                    "restored_best_feasible_reason",
                    None,
                ),
                final_max_feasibility_violation=getattr(
                    res,
                    "final_max_feasibility_violation",
                    None,
                ),
                final_stationarity_norm=getattr(
                    res,
                    "final_stationarity_norm",
                    None,
                ),
            )
            res_nit = int(res.nit)
            main_phase_iterations = int(res.nit)
            termination_message = str(res.message)
            optimizer_success = bool(res.success)
            outer_optimizer_end_s = _perf_counter_s()
            _record_timing(
                timings,
                "outer_optimizer_s",
                outer_optimizer_run_start_s,
                outer_optimizer_end_s,
            )
            if single_stage_host_postprocess_required(
                use_target_lane=use_target_lane,
                write_full_artifacts=write_full_artifacts,
            ) and not use_target_lane:
                restore_single_stage_host_state(
                    use_target_lane=use_target_lane,
                    JF=JF,
                    boozer_surface=boozer_surface,
                    run_dict=run_dict,
                    coil_dofs=res.x,
                    apply_coil_dofs=dof_setter,
                    bs_diag=bs_diag,
                    record_outer_optimizer_event=record_outer_optimizer_event,
                )
                host_postprocess_result = run_single_stage_host_diagnostics(
                    boozer_surface=boozer_surface,
                    run_dict=run_dict,
                    timings=timings,
                )
                host_state_restored_for_final = True
        elif use_target_lane:
            print(
                "Preparing target-lane outer objective/runtime bundle "
                f"(method={outer_contract.method}, maxiter={MAXITER})..."
            )
        jax_compile_diagnostics_recorder = None
        if CONSTRAINT_METHOD != "alm":
            target_lane_trial_boozer_overrides = {
                "bfgs_tol": target_lane_boozer_bfgs_tol_record,
                "bfgs_maxiter": target_lane_boozer_bfgs_maxiter_record,
                "newton_tol": target_lane_boozer_newton_tol_record,
                "newton_maxiter": target_lane_boozer_newton_maxiter_record,
            }
            target_lane_trial_boozer_override_active = any(
                value is not None
                for value in target_lane_trial_boozer_overrides.values()
            )
            target_lane_bundle_setup_start_s = _perf_counter_s()
            record_outer_optimizer_event(
                "target_lane_bundle_setup_started",
                method=outer_contract.method,
                maxiter=int(MAXITER),
                use_value_and_grad=bool(use_target_lane_vg),
                profile_target_lane=bool(args.profile_target_lane),
                success_filter_disabled=bool(args.disable_target_lane_success_filter),
                trial_boozer_overrides={
                    key: None if value is None else float(value)
                    for key, value in target_lane_trial_boozer_overrides.items()
                },
            )
            with maybe_record_jax_compile_diagnostics(
                bool(args.record_jax_compile_diagnostics and use_target_lane)
            ) as jax_compile_diagnostics_recorder, temporary_boozer_surface_option_overrides(
                boozer_surface,
                **target_lane_trial_boozer_overrides,
            ):
                with maybe_trace_single_stage_phase(
                    "single_stage.target_lane_bundle_setup",
                    enabled=jax_profile_enabled and use_target_lane,
                ):
                    (
                        target_scalar_objective,
                        target_value_and_grad_objective,
                        target_lane_optimizer_initial_value_and_grad,
                        target_lane_profile,
                        target_lane_success_filter,
                    ) = prepare_target_lane_outer_objectives(
                        boozer_surface,
                        bs,
                        banana_curve,
                        VV,
                        iota_target,
                        use_target_lane=use_target_lane,
                        use_value_and_grad=use_target_lane_vg,
                        profile_target_lane=args.profile_target_lane,
                        profile_batch_size=args.profile_target_lane_batch_size,
                        disable_success_filter=args.disable_target_lane_success_filter,
                        non_qs_weight=1.0,
                        residual_weight=RES_WEIGHT,
                        iota_weight=IOTAS_WEIGHT,
                        length_weight=LENGTH_WEIGHT,
                        length_target=length_target,
                        cc_dist=CC_DIST,
                        cc_weight=CC_WEIGHT,
                        cs_dist=CS_DIST,
                        cs_weight=CS_WEIGHT,
                        ss_dist=SS_DIST,
                        surf_dist_weight=SURF_DIST_WEIGHT,
                        curvature_threshold=CURVATURE_THRESHOLD,
                        curvature_weight=CURVATURE_WEIGHT,
                    )
                target_lane_outer_objective_config = (
                    build_target_lane_outer_objective_config(
                        boozer_surface,
                        bs,
                        banana_curve,
                        VV,
                        non_qs_weight=1.0,
                        residual_weight=RES_WEIGHT,
                        iota_weight=IOTAS_WEIGHT,
                        length_weight=LENGTH_WEIGHT,
                        length_target=length_target,
                        curve_curve_threshold=CC_DIST,
                        curve_curve_weight=CC_WEIGHT,
                        curve_surface_threshold=CS_DIST,
                        curve_surface_weight=CS_WEIGHT,
                        surface_vessel_threshold=SS_DIST,
                        surface_vessel_weight=SURF_DIST_WEIGHT,
                        curvature_threshold=CURVATURE_THRESHOLD,
                        curvature_weight=CURVATURE_WEIGHT,
                    )
                )
                configure_single_stage_target_lane_accepted_step_sync(
                    adapter,
                    boozer_surface,
                    bs,
                    iota_target,
                    use_target_lane=use_target_lane,
                    outer_objective_config=target_lane_outer_objective_config,
                    success_filter=target_lane_success_filter,
                )
                if use_target_lane:
                    _record_timing(
                        timings,
                        "target_lane_bundle_setup_s",
                        outer_optimizer_start_s,
                        _perf_counter_s(),
                    )
                    outer_optimizer_run_start_s = _perf_counter_s()
                    print("Target-lane outer objective/runtime bundle ready.")
                    record_outer_optimizer_event(
                        "target_lane_bundle_setup_returned",
                        elapsed_s=float(
                            _perf_counter_s() - target_lane_bundle_setup_start_s
                        ),
                        scalar_objective_available=target_scalar_objective is not None,
                        value_and_grad_objective_available=(
                            target_value_and_grad_objective is not None
                        ),
                        optimizer_initial_value_and_grad_available=(
                            target_lane_optimizer_initial_value_and_grad is not None
                        ),
                    )
                    cache_single_stage_target_lane_reporting_snapshot(
                        adapter,
                        run_dict,
                        dofs,
                        benchmark_mode=bool(args.benchmark_mode),
                    )
                    if bool(run_dict.get("initial_objective_pending", False)):
                        if target_lane_optimizer_initial_value_and_grad is None:
                            raise RuntimeError(
                                "Target-lane startup skipped the legacy initial "
                                "objective/gradient snapshot, but the fused "
                                "target-lane initial value/gradient was not "
                                "available to seed the retry state."
                            )
                        (
                            initial_target_value,
                            initial_target_grad,
                        ) = target_lane_optimizer_initial_value_and_grad
                        seed_single_stage_initial_objective_from_values(
                            run_dict,
                            objective_value=initial_target_value,
                            objective_grad=initial_target_grad,
                        )
                        record_single_stage_local_incumbent(
                            run_dict,
                            stage="initial",
                        )
                        record_outer_optimizer_event(
                            "target_lane_initial_objective_seeded",
                            objective_value=_summarize_host_scalar(
                                initial_target_value
                            ),
                            objective_grad=_summarize_host_vector(
                                initial_target_grad
                            ),
                        )
                if args.diagnose_target_lane_scaled_phase1:
                    if not use_target_lane:
                        raise RuntimeError(
                            "--diagnose-target-lane-scaled-phase1 requires the JAX "
                            "ondevice single-stage target lane."
                        )
                    if (
                        effective_initial_step_maxiter < 1
                        or effective_initial_step_scale >= 1.0
                    ):
                        raise RuntimeError(
                            "--diagnose-target-lane-scaled-phase1 requires an active "
                            "scaled initial phase: set --initial-step-maxiter >= 1 "
                            "and --initial-step-scale < 1."
                        )
                    accepted_step_callback = resolve_target_lane_accepted_step_callback(
                        adapter,
                        use_target_lane=use_target_lane,
                        sync_policy=effective_target_lane_sync_policy,
                    )
                    target_lane_scaled_phase1_diagnosis = (
                        build_target_lane_scaled_phase1_diagnosis(
                            boozer_surface,
                            bs,
                            banana_curve,
                            VV,
                            iota_target,
                            anchor_dofs=dofs,
                            contract=outer_contract,
                            phase1_maxiter=min(MAXITER, effective_initial_step_maxiter),
                            step_scale=effective_initial_step_scale,
                            ftol=ftol,
                            gtol=gtol,
                            maxcor=args.maxcor,
                            outer_maxls=args.outer_maxls,
                            callback=accepted_step_callback,
                            success_filter=target_lane_success_filter,
                            non_qs_weight=1.0,
                            residual_weight=RES_WEIGHT,
                            iota_weight=IOTAS_WEIGHT,
                            length_weight=LENGTH_WEIGHT,
                            length_target=length_target,
                            cc_dist=CC_DIST,
                            cc_weight=CC_WEIGHT,
                            cs_dist=CS_DIST,
                            cs_weight=CS_WEIGHT,
                            ss_dist=SS_DIST,
                            surf_dist_weight=SURF_DIST_WEIGHT,
                            curvature_threshold=CURVATURE_THRESHOLD,
                            curvature_weight=CURVATURE_WEIGHT,
                            checkpoint_path=os.path.join(
                                OUT_DIR_ITER,
                                "target_lane_scaled_phase1_diagnosis.json",
                            ),
                        )
                    )
                    outer_optimizer_end_s = _perf_counter_s()
                    _record_timing(
                        timings,
                        "target_lane_scaled_phase1_diagnosis_s",
                        outer_optimizer_start_s,
                        outer_optimizer_end_s,
                    )
                    phase1_iterations = target_lane_scaled_phase1_diagnosis[
                        "optimizer"
                    ]["iterations"]
                    if accepted_step_callback is not None and phase1_iterations > 0:
                        target_lane_restore_start_s = _perf_counter_s()
                        with maybe_trace_single_stage_phase(
                            "single_stage.target_lane_scaled_phase1_restore",
                            enabled=jax_profile_enabled,
                        ):
                            adapter.sync_accepted_step(dofs)
                        _record_timing(
                            timings,
                            "target_lane_scaled_phase1_restore_s",
                            target_lane_restore_start_s,
                            _perf_counter_s(),
                        )
                    phase1_termination_message = target_lane_scaled_phase1_diagnosis[
                        "optimizer"
                    ]["message"]
                    phase1_success = target_lane_scaled_phase1_diagnosis["optimizer"][
                        "success"
                    ]
                    res_nit = phase1_iterations
                    optimizer_success = bool(
                        target_lane_scaled_phase1_diagnosis["all_finite"]
                    )
                    termination_message = "diagnose_target_lane_scaled_phase1"
                    final_volume = initial_volume
                    final_iota = initial_iota
                    final_max_curvature = initial_max_curvature
                    fieldError = initial_field_error
                    first_nonfinite_stage = target_lane_scaled_phase1_diagnosis[
                        "first_nonfinite_stage"
                    ]
                    if first_nonfinite_stage is None:
                        print(
                            "Skipping the normal single-stage optimizer because "
                            "--diagnose-target-lane-scaled-phase1 was provided; "
                            "all recorded scaled phase-1 states stayed finite."
                        )
                    else:
                        print(
                            "Skipping the normal single-stage optimizer because "
                            "--diagnose-target-lane-scaled-phase1 was provided; "
                            f"first non-finite stage is {first_nonfinite_stage}."
                        )
                    res = types.SimpleNamespace(
                        x=np.asarray(
                            target_lane_scaled_phase1_diagnosis[
                                "optimizer_mapped_state"
                            ]["mapped_coil_dofs"],
                            dtype=np.float64,
                        ),
                        nit=phase1_iterations,
                        success=phase1_success,
                        message=phase1_termination_message,
                        status=target_lane_scaled_phase1_diagnosis["optimizer"][
                            "status"
                        ],
                        nfev=target_lane_scaled_phase1_diagnosis["optimizer"]["nfev"],
                        njev=target_lane_scaled_phase1_diagnosis["optimizer"]["njev"],
                        ls_status=target_lane_scaled_phase1_diagnosis["optimizer"][
                            "ls_status"
                        ],
                    )
                elif args.diagnose_target_lane_gradient:
                    if not use_target_lane:
                        raise RuntimeError(
                            "--diagnose-target-lane-gradient requires the JAX "
                            "ondevice single-stage target lane."
                        )
                    target_lane_gradient_diagnosis = (
                        build_target_lane_gradient_diagnosis(
                            boozer_surface,
                            bs,
                            banana_curve,
                            VV,
                            iota_target,
                            success_filter=target_lane_success_filter,
                            non_qs_weight=1.0,
                            residual_weight=RES_WEIGHT,
                            iota_weight=IOTAS_WEIGHT,
                            length_weight=LENGTH_WEIGHT,
                            length_target=length_target,
                            cc_dist=CC_DIST,
                            cc_weight=CC_WEIGHT,
                            cs_dist=CS_DIST,
                            cs_weight=CS_WEIGHT,
                            ss_dist=SS_DIST,
                            surf_dist_weight=SURF_DIST_WEIGHT,
                            curvature_threshold=CURVATURE_THRESHOLD,
                            curvature_weight=CURVATURE_WEIGHT,
                        )
                    )
                    outer_optimizer_end_s = _perf_counter_s()
                    _record_timing(
                        timings,
                        "target_lane_gradient_diagnosis_s",
                        outer_optimizer_start_s,
                        outer_optimizer_end_s,
                    )
                    total_diag = target_lane_gradient_diagnosis["total"]
                    res_nit = 0
                    optimizer_success = bool(
                        total_diag["value"]["finite"]
                        and total_diag["grad"]["all_finite"]
                    )
                    termination_message = "diagnose_target_lane_gradient"
                    final_volume = initial_volume
                    final_iota = initial_iota
                    final_max_curvature = initial_max_curvature
                    fieldError = initial_field_error
                    first_nonfinite_term = target_lane_gradient_diagnosis[
                        "first_nonfinite_term"
                    ]
                    if first_nonfinite_term is None:
                        print(
                            "Skipping single-stage optimizer because "
                            "--diagnose-target-lane-gradient was provided; "
                            "baseline target-lane value/gradient are finite."
                        )
                    else:
                        print(
                            "Skipping single-stage optimizer because "
                            "--diagnose-target-lane-gradient was provided; "
                            f"first non-finite term is {first_nonfinite_term}."
                        )
                elif args.diagnose_target_lane_first_line_search:
                    if not use_target_lane:
                        raise RuntimeError(
                            "--diagnose-target-lane-first-line-search requires the "
                            "JAX ondevice single-stage target lane."
                        )
                    if target_value_and_grad_objective is None:
                        raise RuntimeError(
                            "--diagnose-target-lane-first-line-search requires the "
                            "target-lane value-and-gradient objective."
                        )
                    target_lane_first_line_search_diagnosis = (
                        build_target_lane_first_line_search_diagnosis(
                            target_value_and_grad_objective,
                            dofs,
                            initial_value_and_grad=(
                                target_lane_optimizer_initial_value_and_grad
                            ),
                            initial_step_size=(
                                args.target_lane_outer_initial_step_size
                            ),
                            maxls=args.outer_maxls,
                            gtol=gtol,
                        )
                    )
                    outer_optimizer_end_s = _perf_counter_s()
                    _record_timing(
                        timings,
                        "target_lane_first_line_search_diagnosis_s",
                        outer_optimizer_start_s,
                        outer_optimizer_end_s,
                    )
                    res_nit = 0
                    optimizer_success = bool(
                        target_lane_first_line_search_diagnosis["optimizer_step"][
                            "would_accept"
                        ]
                    )
                    termination_message = "diagnose_target_lane_first_line_search"
                    final_volume = initial_volume
                    final_iota = initial_iota
                    final_max_curvature = initial_max_curvature
                    fieldError = initial_field_error
                    print(
                        "Skipping single-stage optimizer because "
                        "--diagnose-target-lane-first-line-search was provided; "
                        "wrote first L-BFGS line-search trace."
                    )
                elif args.profile_target_lane_only:
                    if not use_target_lane:
                        raise RuntimeError(
                            "--profile-target-lane-only requires the JAX ondevice "
                            "single-stage target lane."
                        )
                    outer_optimizer_end_s = _perf_counter_s()
                    _record_timing(
                        timings,
                        "target_lane_profile_only_s",
                        outer_optimizer_start_s,
                        outer_optimizer_end_s,
                    )
                    res_nit = 0
                    optimizer_success = True
                    termination_message = "profile_target_lane_only"
                    final_volume = initial_volume
                    final_iota = initial_iota
                    final_max_curvature = initial_max_curvature
                    fieldError = initial_field_error
                    print(
                        "Skipping single-stage optimizer because "
                        "--profile-target-lane-only was provided."
                    )
                else:
                    accepted_step_callback = resolve_target_lane_accepted_step_callback(
                        adapter,
                        use_target_lane=use_target_lane,
                        sync_policy=effective_target_lane_sync_policy,
                    )
                    if use_target_lane and not diagnostic_target_lane_callbacks:
                        accepted_step_callback = None
                    retry_step_callback = accepted_step_callback
                    target_lane_phase1_state_sync = None
                    target_lane_post_run_state_sync = (
                        resolve_target_lane_post_run_state_sync(
                            adapter,
                            use_target_lane=use_target_lane,
                            accepted_step_callback=accepted_step_callback,
                        )
                    )
                    phase1_dofs = dofs
                    phase1_base_fun = (
                        target_value_and_grad_objective if use_target_lane else adapter
                    )
                    phase1_base_scalar_fun = (
                        target_scalar_objective if use_target_lane else None
                    )
                    phase1_fun = phase1_base_fun
                    phase1_scalar_fun = phase1_base_scalar_fun
                    phase1_callback = accepted_step_callback
                    phase1_retry_callback = phase1_callback
                    phase1_post_run_state_sync = None
                    phase1_final_dofs = None
                    phase1_optimizer_initial_value_and_grad = None
                    main_optimizer_initial_value_and_grad = (
                        target_lane_optimizer_initial_value_and_grad
                    )
                    remaining_maxiter = MAXITER
                    if (
                        effective_initial_step_maxiter > 0
                        and effective_initial_step_scale < 1.0
                    ):
                        phase1_maxiter = min(MAXITER, effective_initial_step_maxiter)
                        if phase1_maxiter > 0:
                            phase1_anchor_dofs = dofs
                            phase1_dofs = build_scaled_outer_phase_initial_dofs(
                                phase1_anchor_dofs,
                                use_target_lane=use_target_lane,
                            )
                            if use_target_lane:
                                if phase1_fun is not None:
                                    phase1_fun, phase1_callback = (
                                        build_scaled_outer_problem(
                                            phase1_base_fun,
                                            accepted_step_callback,
                                            phase1_anchor_dofs,
                                            effective_initial_step_scale,
                                        )
                                    )
                                    phase1_optimizer_initial_value_and_grad = (
                                        build_scaled_outer_problem_initial_value_and_grad(
                                            target_lane_optimizer_initial_value_and_grad,
                                            phase1_anchor_dofs,
                                            effective_initial_step_scale,
                                        )
                                    )
                                    main_optimizer_initial_value_and_grad = None
                                    phase1_scalar_fun = None
                                else:
                                    (
                                        phase1_scalar_fun,
                                        phase1_callback,
                                    ) = build_scaled_outer_scalar_problem(
                                        phase1_base_scalar_fun,
                                        accepted_step_callback,
                                        phase1_anchor_dofs,
                                        effective_initial_step_scale,
                                    )
                                phase1_retry_callback = phase1_callback
                                target_lane_phase1_state_sync = (
                                    resolve_target_lane_post_run_state_sync(
                                        adapter,
                                        use_target_lane=use_target_lane,
                                        accepted_step_callback=accepted_step_callback,
                                        scaled_phase_step_scale=(
                                            effective_initial_step_scale
                                        ),
                                        scaled_phase_anchor_dofs=phase1_anchor_dofs,
                                    )
                                )
                                phase1_post_run_state_sync = (
                                    resolve_target_lane_post_run_state_sync(
                                        adapter,
                                        use_target_lane=use_target_lane,
                                        accepted_step_callback=phase1_callback,
                                        scaled_phase_step_scale=(
                                            effective_initial_step_scale
                                        ),
                                        scaled_phase_anchor_dofs=phase1_anchor_dofs,
                                    )
                                )
                                if phase1_retry_callback is None:
                                    if phase1_base_fun is not None:
                                        _, phase1_retry_callback = (
                                            build_scaled_outer_problem(
                                                phase1_base_fun,
                                                retry_step_callback,
                                                phase1_anchor_dofs,
                                                effective_initial_step_scale,
                                            )
                                        )
                                    else:
                                        _, phase1_retry_callback = (
                                            build_scaled_outer_scalar_problem(
                                                phase1_base_scalar_fun,
                                                retry_step_callback,
                                                phase1_anchor_dofs,
                                                effective_initial_step_scale,
                                            )
                                        )
                            else:
                                phase1_fun, phase1_callback = (
                                    build_scaled_outer_problem(
                                        phase1_fun,
                                        accepted_step_callback,
                                        phase1_anchor_dofs,
                                        effective_initial_step_scale,
                                    )
                                )
                            print(
                                "Starting scaled initial outer phase "
                                f"(step_scale={effective_initial_step_scale}, "
                                f"maxiter={phase1_maxiter})..."
                            )
                            record_outer_optimizer_event(
                                "phase1_started",
                                phase="phase1",
                                maxiter=int(phase1_maxiter),
                                step_scale=float(effective_initial_step_scale),
                                optimizer_dofs=_summarize_host_vector(phase1_dofs),
                            )
                            phase1_failure_callback = (
                                resolve_target_lane_invalid_state_failure_callback(
                                    target_lane_invalid_state_diagnostic_events,
                                    phase="phase1",
                                    use_target_lane=use_target_lane,
                                    args=args,
                                )
                            )
                            phase1_failure_callback = (
                                wrap_outer_optimizer_failure_callback(
                                    phase1_failure_callback,
                                    phase="phase1",
                                )
                            )
                            phase1_progress_callback = (
                                build_outer_optimizer_progress_callback("phase1")
                                if diagnostic_target_lane_callbacks
                                else None
                            )
                            phase1_start_s = _perf_counter_s()
                            with maybe_trace_single_stage_phase(
                                "single_stage.outer_optimizer_initial_phase",
                                enabled=jax_profile_enabled,
                            ):
                                if use_target_lane:
                                    phase1_res, phase1_retry_summary = (
                                        run_single_stage_target_lane_optimizer_with_retries(
                                            phase1_fun,
                                            phase1_dofs,
                                            phase="phase1",
                                            callback=phase1_callback,
                                            retry_callback=phase1_retry_callback,
                                            result_state_sync=phase1_post_run_state_sync,
                                            contract=outer_contract,
                                            maxiter=phase1_maxiter,
                                            ftol=ftol,
                                            gtol=gtol,
                                            maxcor=args.maxcor,
                                            outer_maxls=args.outer_maxls,
                                            scalar_fun=phase1_scalar_fun,
                                            progress_callback=phase1_progress_callback,
                                            target_lane_initial_step_size=None,
                                            failure_callback=phase1_failure_callback,
                                            optimizer_initial_value_and_grad=(
                                                phase1_optimizer_initial_value_and_grad
                                            ),
                                            invalid_state_events=(
                                                target_lane_invalid_state_events
                                            ),
                                            run_dict=run_dict,
                                            single_stage_search_policy=(
                                                single_stage_search_policy
                                            ),
                                            retry_dofs_factory=(
                                                lambda anchor_state: (
                                                    build_scaled_outer_phase_initial_dofs(
                                                        host_array(
                                                            anchor_state["coil_dofs"],
                                                            dtype=np.float64,
                                                        ),
                                                        use_target_lane=True,
                                                    )
                                                )
                                            ),
                                            restored_result_x_factory=(
                                                lambda anchor_state: (
                                                    build_scaled_outer_phase_initial_dofs(
                                                        host_array(
                                                            anchor_state["coil_dofs"],
                                                            dtype=np.float64,
                                                        ),
                                                        use_target_lane=True,
                                                    )
                                                )
                                            ),
                                            progress_event_callback=(
                                                record_outer_optimizer_event
                                            ),
                                        )
                                    )
                                else:
                                    phase1_res = run_single_stage_optimizer(
                                        phase1_fun,
                                        phase1_dofs,
                                        callback=phase1_callback,
                                        contract=outer_contract,
                                        maxiter=phase1_maxiter,
                                        ftol=ftol,
                                        gtol=gtol,
                                        maxcor=args.maxcor,
                                        outer_maxls=args.outer_maxls,
                                        scalar_fun=phase1_scalar_fun,
                                        progress_callback=phase1_progress_callback,
                                        failure_callback=phase1_failure_callback,
                                    )
                            _record_timing(
                                timings,
                                "outer_optimizer_initial_phase_s",
                                phase1_start_s,
                                _perf_counter_s(),
                            )
                            record_outer_optimizer_event(
                                "phase1_returned",
                                phase="phase1",
                                elapsed_s=timings.get(
                                    "outer_optimizer_initial_phase_s"
                                ),
                                result=summarize_optimizer_result_for_progress(
                                    phase1_res
                                ),
                                retry_summary=phase1_retry_summary,
                            )
                            phase1_iterations = int(phase1_res.nit)
                            phase1_termination_message = str(phase1_res.message)
                            if phase1_retry_summary["attempt_count"] > 0:
                                phase1_termination_message = (
                                    f"{phase1_termination_message}; "
                                    f"retry_attempts={phase1_retry_summary['attempt_count']}"
                                )
                            phase1_success = bool(phase1_res.success)
                            print(phase1_res.message)
                            phase1_final_dofs = resolve_scaled_outer_phase_final_dofs(
                                phase1_anchor_dofs,
                                phase1_res.x,
                                effective_initial_step_scale,
                                use_target_lane=use_target_lane,
                            )
                            if (
                                target_lane_phase1_state_sync is not None
                                and target_lane_result_has_syncable_state(phase1_res)
                            ):
                                target_lane_phase1_sync_start_s = _perf_counter_s()
                                record_outer_optimizer_event(
                                    "target_lane_initial_phase_sync_started",
                                    phase="phase1",
                                    result=summarize_optimizer_result_for_progress(
                                        phase1_res
                                    ),
                                )
                                with maybe_trace_single_stage_phase(
                                    "single_stage.target_lane_initial_phase_sync",
                                    enabled=jax_profile_enabled,
                                ):
                                    target_lane_phase1_state_sync(phase1_res.x)
                                _record_timing(
                                    timings,
                                    "target_lane_initial_phase_sync_s",
                                    target_lane_phase1_sync_start_s,
                                    _perf_counter_s(),
                                )
                                record_outer_optimizer_event(
                                    "target_lane_initial_phase_sync_returned",
                                    phase="phase1",
                                    elapsed_s=timings.get(
                                        "target_lane_initial_phase_sync_s"
                                    ),
                                )
                            if phase1_iterations > 0:
                                dofs = phase1_final_dofs
                                run_dict["x_prev"] = _single_stage_optimizer_dofs_array(
                                    dofs
                                ).copy()
                            remaining_maxiter = max(MAXITER - phase1_iterations, 0)
                    run_main_optimizer = (
                        remaining_maxiter > 0 or phase1_termination_message is None
                    )
                    if use_target_lane and run_main_optimizer:
                        print(
                            "Starting target-lane outer optimizer "
                            f"(sync={effective_target_lane_sync_policy}, "
                            f"outer_maxls={args.outer_maxls}, "
                            f"initial_step_size={args.target_lane_outer_initial_step_size}, "
                            f"boozer_bfgs_tol={target_lane_boozer_bfgs_tol_record}, "
                            f"boozer_bfgs_maxiter={target_lane_boozer_bfgs_maxiter_record}, "
                            f"boozer_newton_tol={target_lane_boozer_newton_tol_record}, "
                            f"boozer_newton_maxiter={target_lane_boozer_newton_maxiter_record}, "
                            f"remaining_maxiter={remaining_maxiter})..."
                        )
                        record_outer_optimizer_event(
                            "phase2_started",
                            phase="phase2",
                            maxiter=int(remaining_maxiter),
                            initial_step_size=args.target_lane_outer_initial_step_size,
                            optimizer_dofs=_summarize_host_vector(dofs),
                        )
                    if run_main_optimizer:
                        main_failure_callback = (
                            resolve_target_lane_invalid_state_failure_callback(
                                target_lane_invalid_state_diagnostic_events,
                                phase="phase2",
                                use_target_lane=use_target_lane,
                                args=args,
                            )
                        )
                        main_failure_callback = wrap_outer_optimizer_failure_callback(
                            main_failure_callback,
                            phase="phase2",
                        )
                        main_progress_callback = (
                            build_outer_optimizer_progress_callback("phase2")
                            if diagnostic_target_lane_callbacks
                            else None
                        )
                        main_optimizer_start_s = _perf_counter_s()
                        with maybe_trace_single_stage_phase(
                            "single_stage.outer_optimizer",
                            enabled=jax_profile_enabled,
                        ):
                            if use_target_lane:
                                res, target_lane_retry_summary = (
                                    run_single_stage_target_lane_optimizer_with_retries(
                                        target_value_and_grad_objective,
                                        dofs,
                                        phase="phase2",
                                        callback=accepted_step_callback,
                                        retry_callback=retry_step_callback,
                                        result_state_sync=target_lane_post_run_state_sync,
                                        contract=outer_contract,
                                        maxiter=remaining_maxiter,
                                        ftol=ftol,
                                        gtol=gtol,
                                        maxcor=args.maxcor,
                                        outer_maxls=args.outer_maxls,
                                        scalar_fun=target_scalar_objective,
                                        progress_callback=main_progress_callback,
                                        target_lane_initial_step_size=(
                                            args.target_lane_outer_initial_step_size
                                        ),
                                        failure_callback=main_failure_callback,
                                        optimizer_initial_value_and_grad=(
                                            main_optimizer_initial_value_and_grad
                                        ),
                                        invalid_state_events=target_lane_invalid_state_events,
                                        run_dict=run_dict,
                                        single_stage_search_policy=single_stage_search_policy,
                                        progress_event_callback=(
                                            record_outer_optimizer_event
                                        ),
                                    )
                                )
                            else:
                                res = run_single_stage_optimizer(
                                    adapter,
                                    dofs,
                                    callback=accepted_step_callback,
                                    contract=outer_contract,
                                    maxiter=remaining_maxiter,
                                    ftol=ftol,
                                    gtol=gtol,
                                    maxcor=args.maxcor,
                                    outer_maxls=args.outer_maxls,
                                    scalar_fun=target_scalar_objective,
                                    progress_callback=main_progress_callback,
                                    target_lane_initial_step_size=None,
                                    failure_callback=main_failure_callback,
                                )
                        _record_timing(
                            timings,
                            "outer_optimizer_main_s",
                            main_optimizer_start_s,
                            _perf_counter_s(),
                        )
                        record_outer_optimizer_event(
                            "phase2_returned",
                            phase="phase2",
                            elapsed_s=timings.get("outer_optimizer_main_s"),
                            result=summarize_optimizer_result_for_progress(res),
                            retry_summary=target_lane_retry_summary,
                        )
                        termination_message = str(res.message)
                        if target_lane_retry_summary["attempt_count"] > 0:
                            termination_message = (
                                f"{termination_message}; "
                                f"retry_attempts={target_lane_retry_summary['attempt_count']}"
                            )
                        optimizer_success = bool(res.success)
                        main_phase_iterations = int(res.nit)
                        res_nit = (phase1_iterations or 0) + int(res.nit)
                        print(res.message)
                        if (
                            use_target_lane
                            and args.benchmark_mode
                            and target_lane_post_run_state_sync is not None
                            and target_lane_result_has_syncable_state(res)
                        ):
                            target_lane_sync_start_s = _perf_counter_s()
                            record_outer_optimizer_event(
                                "target_lane_final_sync_started",
                                phase="phase2",
                                result=summarize_optimizer_result_for_progress(res),
                            )
                            with maybe_trace_single_stage_phase(
                                "single_stage.target_lane_final_sync",
                                enabled=jax_profile_enabled,
                            ):
                                target_lane_post_run_state_sync(res.x)
                            _record_timing(
                                timings,
                                "target_lane_final_sync_s",
                                target_lane_sync_start_s,
                                _perf_counter_s(),
                            )
                            record_outer_optimizer_event(
                                "target_lane_final_sync_returned",
                                phase="phase2",
                                elapsed_s=timings.get("target_lane_final_sync_s"),
                            )
                        if phase1_termination_message is not None:
                            termination_message = (
                                f"phase1={phase1_termination_message}; "
                                f"phase2={termination_message}"
                            )
                    else:
                        record_outer_optimizer_event(
                            "phase2_skipped",
                            phase="phase2",
                            phase1_termination_message=phase1_termination_message,
                            remaining_maxiter=int(remaining_maxiter),
                        )
                        res = phase1_res
                        if phase1_final_dofs is not None:
                            res.x = phase1_final_dofs
                        res_nit = phase1_iterations or 0
                        termination_message = (
                            phase1_termination_message or "phase1_only"
                        )
                        optimizer_success = bool(phase1_success)
                    outer_optimizer_end_s = _perf_counter_s()
                    _record_timing(
                        timings,
                        "outer_optimizer_s",
                        outer_optimizer_run_start_s,
                        outer_optimizer_end_s,
                    )
        if jax_compile_diagnostics_recorder is not None:
            jax_compile_diagnostics = jax_compile_diagnostics_recorder.summary()

        if (not args.benchmark_mode) and should_force_strict_target_lane_final_sync(
            use_target_lane=use_target_lane,
            res_nit=res_nit,
            optimizer_status=None if res is None else getattr(res, "status", None),
            accepted_step_callback=accepted_step_callback,
            trial_boozer_override_active=target_lane_trial_boozer_override_active,
        ):
            target_lane_sync_start_s = _perf_counter_s()
            record_outer_optimizer_event(
                "target_lane_strict_final_sync_started",
                phase="final",
                result=summarize_optimizer_result_for_progress(res),
            )
            with maybe_trace_single_stage_phase(
                "single_stage.target_lane_final_sync",
                enabled=jax_profile_enabled,
            ):
                target_lane_post_run_state_sync(res.x)
            _record_timing(
                timings,
                "target_lane_final_sync_s",
                target_lane_sync_start_s,
                _perf_counter_s(),
            )
            record_outer_optimizer_event(
                "target_lane_strict_final_sync_returned",
                phase="final",
                elapsed_s=timings.get("target_lane_final_sync_s"),
            )
        if (
            not skip_outer_optimizer
            and not host_state_restored_for_final
            and single_stage_host_postprocess_required(
                use_target_lane=use_target_lane,
                write_full_artifacts=write_full_artifacts,
            )
            and not use_target_lane
        ):
            restore_single_stage_host_state(
                use_target_lane=use_target_lane,
                JF=JF,
                boozer_surface=boozer_surface,
                run_dict=run_dict,
                coil_dofs=res.x,
                apply_coil_dofs=dof_setter,
                bs_diag=bs_diag,
                record_outer_optimizer_event=record_outer_optimizer_event,
            )
            host_postprocess_result = run_single_stage_host_diagnostics(
                boozer_surface=boozer_surface,
                run_dict=run_dict,
                timings=timings,
            )
            host_state_restored_for_final = True

    if use_target_lane and skip_outer_optimizer:
        target_lane_init_reporting_start_s = _perf_counter_s()
        cache_single_stage_target_lane_init_reporting_snapshot(
            boozer_surface=boozer_surface,
            bs=bs,
            banana_curve=banana_curve,
            vessel_surface=VV,
            iota_target=iota_target,
            run_dict=run_dict,
            coil_dofs=dofs,
            benchmark_mode=bool(args.benchmark_mode),
            disable_success_filter=bool(args.disable_target_lane_success_filter),
            length_target=length_target,
            cc_dist=CC_DIST,
            cc_weight=CC_WEIGHT,
            cs_dist=CS_DIST,
            cs_weight=CS_WEIGHT,
            ss_dist=SS_DIST,
            surf_dist_weight=SURF_DIST_WEIGHT,
            residual_weight=RES_WEIGHT,
            iota_weight=IOTAS_WEIGHT,
            length_weight=LENGTH_WEIGHT,
            curvature_threshold=CURVATURE_THRESHOLD,
            curvature_weight=CURVATURE_WEIGHT,
        )
        target_lane_init_reporting_end_s = _perf_counter_s()
        target_lane_init_reporting_snapshot_s = _record_timing(
            timings,
            "target_lane_init_reporting_snapshot_s",
            target_lane_init_reporting_start_s,
            target_lane_init_reporting_end_s,
        )
        timings["jax_compile_prewarm_reporting_snapshot_s"] = (
            target_lane_init_reporting_snapshot_s
        )

    final_penalty_metrics = resolve_single_stage_final_penalty_metrics(
        use_target_lane=use_target_lane,
        benchmark_mode=bool(args.benchmark_mode),
        skip_outer_optimizer=skip_outer_optimizer,
        boozer_surface=boozer_surface,
        bs=bs,
        iota_target=iota_target,
        coil_dofs=bs.x.copy() if res is None else res.x,
        outer_objective_config=None,
        success_filter=target_lane_success_filter,
        curvelength=curvelength,
        j_non_qs=JnonQSRatio,
        j_boozer_residual=JBoozerResidual,
        j_iota=Jiota,
        j_curve_length=JCurveLength,
        j_curve_curve=JCurveCurve,
        j_curve_surface=JCurveSurface,
        j_surface_surface=JSurfSurf,
        j_curvature=JCurvature,
        cc_dist=CC_DIST,
        cs_dist=CS_DIST,
        ss_dist=SS_DIST,
        curvature_threshold=CURVATURE_THRESHOLD,
        run_dict=run_dict,
        init_only=args.init_only,
        termination_message=termination_message,
        optimizer_success=optimizer_success,
    )
    final_volume = float(final_penalty_metrics["final_volume"])
    final_iota = float(final_penalty_metrics["final_iota"])
    final_max_curvature = float(final_penalty_metrics["max_curvature"])
    fieldError = final_penalty_metrics["field_error"]
    final_banana_current_A = resolve_single_stage_final_banana_current_A(
        use_target_lane=use_target_lane,
        final_metrics=final_penalty_metrics,
        banana_current=banana_coils[0].current,
    )
    final_self_intersecting = bool(run_dict["intersecting"])
    final_self_intersection_check_available = bool(
        run_dict["self_intersection_check_available"]
    )
    if host_postprocess_result is not None:
        final_self_intersecting = host_postprocess_result.self_intersecting
        final_self_intersection_check_available = (
            host_postprocess_result.self_intersection_check_available
        )
    final_coil_length = float(final_penalty_metrics["coil_length"])
    print(f"Volume: {final_volume}")
    print(f"Iota: {final_iota}")
    print(f"Max Curvature: {final_max_curvature}")
    final_hardware_metrics_start_s = _perf_counter_s()
    final_artifact_hardware_snapshot = None
    with maybe_trace_single_stage_phase(
        "single_stage.final_hardware_metrics",
        enabled=jax_profile_enabled,
    ):
        if args.benchmark_mode:
            final_curve_curve_min_dist = final_penalty_metrics["curve_curve_min_dist"]
            final_curve_surface_min_dist = final_penalty_metrics[
                "curve_surface_min_dist"
            ]
            final_surface_vessel_min_dist = final_penalty_metrics[
                "surface_vessel_min_dist"
            ]
            final_hardware_status = final_penalty_metrics["hardware_status"]
        else:
            final_artifact_hardware_snapshot = (
                evaluate_single_stage_artifact_hardware_snapshot(
                    curve_curve_min_dist=final_penalty_metrics["curve_curve_min_dist"],
                    cc_dist=CC_DIST,
                    curve_surface_min_dist=final_penalty_metrics[
                        "curve_surface_min_dist"
                    ],
                    cs_dist=CS_DIST,
                    surface_vessel_min_dist=final_penalty_metrics[
                        "surface_vessel_min_dist"
                    ],
                    ss_dist=SS_DIST,
                    max_curvature=final_penalty_metrics["max_curvature"],
                    curvature_threshold=CURVATURE_THRESHOLD,
                    coil_length=final_coil_length,
                    length_target=length_target,
                    banana_current_A=final_banana_current_A,
                    banana_current_max_A=args.banana_current_max_A,
                    tf_current_A=stage2_tf_current_A,
                    tf_current_limit_A=TF_CURRENT_HARD_LIMIT_A,
                )
            )
            final_curve_curve_min_dist = final_artifact_hardware_snapshot[
                "curve_curve_min_dist"
            ]
            final_curve_surface_min_dist = final_artifact_hardware_snapshot[
                "curve_surface_min_dist"
            ]
            final_surface_vessel_min_dist = final_artifact_hardware_snapshot[
                "surface_vessel_min_dist"
            ]
            final_hardware_status = final_artifact_hardware_snapshot[
                "artifact_hardware_status"
            ]
        if not args.benchmark_mode:
            optimizer_success, termination_message = apply_hardware_constraint_verdict(
                optimizer_success,
                termination_message,
                final_hardware_status,
                init_only=args.init_only,
            )
    _record_timing(
        timings,
        "final_hardware_metrics_s",
        final_hardware_metrics_start_s,
        _perf_counter_s(),
    )
    final_distances = {
        "curve_curve_min_dist": final_curve_curve_min_dist,
        "curve_surface_min_dist": final_curve_surface_min_dist,
        "surface_vessel_min_dist": final_surface_vessel_min_dist,
    }
    final_optimizer_result = summarize_single_stage_final_optimizer_result(
        result=res,
        ran_optimizer=not skip_outer_optimizer,
        iterations=res_nit,
        optimizer_success=optimizer_success,
        termination_message=termination_message,
    )
    optimizer_diag = extract_optimizer_diagnostics(
        None if skip_outer_optimizer else res,
        ran_optimizer=not skip_outer_optimizer,
        termination_message=termination_message,
    )
    timings["script_total_s"] = _elapsed_s(run_wall_start_s, _perf_counter_s())
    final_result_snapshot = build_single_stage_final_result_snapshot(
        final_coil_dofs=bs.x.copy() if res is None else res.x,
        run_dict=run_dict,
        final_metrics=final_penalty_metrics,
        final_distances=final_distances,
        hardware_status=final_hardware_status,
        optimizer_result=final_optimizer_result,
        optimizer_diagnostics=optimizer_diag,
        timings=timings,
        write_restart_artifacts=write_restart_artifacts,
        write_full_artifacts=write_full_artifacts,
        boozer_optimizer_method=boozer_surface.res.get("optimizer_method"),
        field_error=fieldError,
        self_intersecting=final_self_intersecting,
        self_intersection_check_available=final_self_intersection_check_available,
    )
    results = {
        **build_single_stage_results_envelope(
            output_root=OUT_DIR_ITER,
            plasma_surf_filename=plasma_surf_filename,
            file_loc=file_loc,
            mpol=mpol,
            ntor=ntor,
            nphi=nphi,
            ntheta=ntheta,
            vol_target=vol_target,
            iota_target=iota_target,
            stage2_bs_path=stage2_bs_path,
            stage2_results_path=stage2_results_path,
            stage2_source=stage2_source,
            stage2_results=stage2_results,
            warm_start_run_dir=args.warm_start_run_dir,
            warm_start_state=warm_start_state,
            banana_surf_radius=banana_surf_radius,
            R0=R0,
            s=s,
            order=order,
            CONSTRAINT_WEIGHT=CONSTRAINT_WEIGHT,
            constraint_method=CONSTRAINT_METHOD,
            alm_formulation=ALM_FORMULATION,
            CC_DIST=CC_DIST,
            CC_WEIGHT=CC_WEIGHT,
            CS_DIST=CS_DIST,
            CS_WEIGHT=CS_WEIGHT,
            SS_DIST=SS_DIST,
            SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
            CURVATURE_WEIGHT=CURVATURE_WEIGHT,
            CURVATURE_THRESHOLD=CURVATURE_THRESHOLD,
            LENGTH_WEIGHT=LENGTH_WEIGHT,
            RES_WEIGHT=RES_WEIGHT,
            IOTAS_WEIGHT=IOTAS_WEIGHT,
            length_target=length_target,
            banana_current_max_A=args.banana_current_max_A,
            tf_current_limit_A=TF_CURRENT_HARD_LIMIT_A,
            optimizer_backend_record=optimizer_backend_record,
            boozer_optimizer_backend_record=boozer_optimizer_backend_record,
            boozer_least_squares_algorithm_record=boozer_least_squares_algorithm_record,
            outer_optimizer_method=outer_optimizer_method_record,
            target_lane_sync_record=target_lane_sync_record,
            requested_experimental_target_lane_vg=requested_experimental_target_lane_vg,
            use_target_lane_vg=use_target_lane_vg,
            target_lane_boozer_bfgs_tol_record=target_lane_boozer_bfgs_tol_record,
            target_lane_boozer_bfgs_maxiter_record=target_lane_boozer_bfgs_maxiter_record,
            target_lane_boozer_newton_tol_record=target_lane_boozer_newton_tol_record,
            target_lane_boozer_newton_maxiter_record=target_lane_boozer_newton_maxiter_record,
            single_stage_search_policy=single_stage_search_policy,
            effective_initial_phase_settings=effective_initial_phase_settings,
            args=args,
            MAXITER=MAXITER,
            write_restart_artifacts=write_restart_artifacts,
            write_full_artifacts=write_full_artifacts,
        ),
        "PLASMA_SURF_FILENAME": plasma_surf_filename,
        "PLASMA_SURF_PATH": file_loc,
        "WARM_START_RUN_DIR": None
        if args.warm_start_run_dir is None
        else os.path.abspath(args.warm_start_run_dir),
        "WARM_START_SURFACE_PATH": None
        if warm_start_state is None
        else warm_start_state["surface_path"],
        "WARM_START_RESULTS_PATH": None
        if warm_start_state is None
        else warm_start_state["results_path"],
        "WARM_START_BIOT_SAVART_PATH": None
        if warm_start_state is None
        else warm_start_state.get("biot_savart_path"),
        "WARM_START_JAX_RUNTIME_SPEC_PATH": None
        if warm_start_state is None
        else warm_start_state.get("jax_runtime_spec_path"),
        "JAX_RUNTIME_SEED_SPEC_PATH": None
        if warm_start_runtime_spec_state is None
        else warm_start_runtime_spec_state["path"],
        "STAGE2_SOURCE": stage2_source,
        "STAGE2_SOURCE_REQUESTED": args.stage2_source,
        "STAGE2_BS_PATH": stage2_bs_path,
        "STAGE2_RESULTS_PATH": stage2_results_path,
        "STAGE2_SEED_MAJOR_RADIUS": R0,
        "STAGE2_SEED_TOROIDAL_FLUX": s,
        "STAGE2_SEED_BANANA_SURF_RADIUS": float(stage2_results["banana_surf_radius"]),
        "STAGE2_SEED_ORDER": order,
        "STAGE2_TF_CURRENT_LIMIT_ENFORCED": bool(stage2_tf_current_limit_enforced),
        "STAGE2_SEED_HARDWARE_VALIDATION_ENFORCED": bool(
            stage2_seed_hardware_validation_enforced
        ),
        "mpol": mpol,
        "ntor": ntor,
        "nphi": nphi,
        "ntheta": ntheta,
        "boozer_stage": stage,
        "CONSTRAINT_WEIGHT": CONSTRAINT_WEIGHT,
        "CONSTRAINT_METHOD": CONSTRAINT_METHOD,
        "ALM_FORMULATION": ALM_FORMULATION if CONSTRAINT_METHOD == "alm" else None,
        "CC_DIST": CC_DIST,
        "CC_WEIGHT": CC_WEIGHT,
        "CS_DIST": CS_DIST,
        "CS_WEIGHT": CS_WEIGHT,
        "SS_DIST": SS_DIST,
        "SURF_DIST_WEIGHT": SURF_DIST_WEIGHT,
        "CURVATURE_WEIGHT": CURVATURE_WEIGHT,
        "CURVATURE_THRESHOLD": CURVATURE_THRESHOLD,
        "NON_QS_WEIGHT": 1.0,
        "LENGTH_WEIGHT": LENGTH_WEIGHT,
        "RES_WEIGHT": RES_WEIGHT,
        "IOTAS_WEIGHT": IOTAS_WEIGHT,
        "MAJOR_RADIUS": R0,
        "TOROIDAL_FLUX": s,
        "banana_surf_radius": banana_surf_radius,
        "order": order,
        "LENGTH_TARGET": float(length_target),
        "BANANA_CURRENT_MAX_A": float(args.banana_current_max_A),
        "STAGE2_TF_CURRENT_A": float(stage2_tf_current_A),
        "TF_CURRENT_LIMIT_A": float(TF_CURRENT_HARD_LIMIT_A),
        "backend": args.backend,
        "optimizer_backend": optimizer_backend_record,
        "boozer_optimizer_backend": boozer_optimizer_backend_record,
        "boozer_least_squares_algorithm": boozer_least_squares_algorithm_record,
        "boozer_limited_memory_requested": args.boozer_limited_memory,
        "boozer_limited_memory": effective_boozer_limited_memory,
        "boozer_optimizer_method": final_result_snapshot.boozer_optimizer_method,
        "outer_optimizer_method": outer_optimizer_method_record,
        "target_lane_accepted_step_sync": target_lane_sync_record,
        "experimental_target_lane_value_and_grad": requested_experimental_target_lane_vg,
        "target_lane_value_and_grad": use_target_lane_vg,
        "benchmark_mode": bool(args.benchmark_mode),
        "disable_target_lane_success_filter": bool(
            args.disable_target_lane_success_filter
        ),
        "single_stage_donor_class": single_stage_search_policy.donor_class,
        "single_stage_search_policy": single_stage_search_policy.search_policy,
        "adaptive_failure_penalty_weight": (
            single_stage_search_policy.adaptive_failure_penalty_weight
        ),
        "invalid_step_retry_budget": single_stage_search_policy.invalid_step_retry_budget,
        "retry_step_shrink_factor": single_stage_search_policy.retry_step_shrink_factor,
        "initial_phase_auto_enabled": bool(
            effective_initial_phase_settings["auto_enabled"]
        ),
        "profile_target_lane": bool(args.profile_target_lane),
        "profile_target_lane_only": bool(args.profile_target_lane_only),
        "profile_target_lane_batch_size": int(args.profile_target_lane_batch_size),
        "diagnose_target_lane_gradient": bool(args.diagnose_target_lane_gradient),
        "diagnose_target_lane_first_line_search": bool(
            getattr(args, "diagnose_target_lane_first_line_search", False)
        ),
        "diagnose_target_lane_scaled_phase1": bool(
            args.diagnose_target_lane_scaled_phase1
        ),
        "diagnostic_callbacks": bool(diagnostic_target_lane_callbacks),
        "record_target_lane_invalid_state_events": bool(
            record_target_lane_invalid_state_events
        ),
        "structured_invalid_step_log": bool(use_target_lane),
        "minimal_artifacts": bool(args.minimal_artifacts),
        "full_artifacts": bool(args.full_artifacts),
        "init_only": args.init_only,
        "max_iterations": MAXITER,
        "maxcor": args.maxcor,
        "outer_maxls": args.outer_maxls,
        "outer_ftol": args.outer_ftol,
        "target_lane_outer_initial_step_size": args.target_lane_outer_initial_step_size,
        "initial_phase_retry_attempts": int(phase1_retry_summary["attempt_count"]),
        "initial_phase_retry_attempt_details": phase1_retry_summary["attempts"],
        "initial_phase_restored_preserved_local_state": bool(
            phase1_retry_summary["restored_preserved_local_state"]
        ),
        "initial_phase_restored_preserved_local_stage": phase1_retry_summary[
            "restored_preserved_local_stage"
        ],
        "target_lane_retry_attempts": int(target_lane_retry_summary["attempt_count"]),
        "target_lane_retry_attempt_details": target_lane_retry_summary["attempts"],
        "target_lane_restored_preserved_local_state": bool(
            target_lane_retry_summary["restored_preserved_local_state"]
        ),
        "target_lane_restored_preserved_local_stage": target_lane_retry_summary[
            "restored_preserved_local_stage"
        ],
        "initial_step_scale": effective_initial_step_scale,
        "initial_step_maxiter": effective_initial_step_maxiter,
        "requested_initial_step_scale": args.initial_step_scale,
        "requested_initial_step_maxiter": args.initial_step_maxiter,
        "target_lane_boozer_bfgs_tol": target_lane_boozer_bfgs_tol_record,
        "target_lane_boozer_bfgs_maxiter": target_lane_boozer_bfgs_maxiter_record,
        "target_lane_boozer_newton_tol": target_lane_boozer_newton_tol_record,
        "target_lane_boozer_newton_maxiter": target_lane_boozer_newton_maxiter_record,
        "INITIAL_PHASE_ITERATIONS": phase1_iterations,
        "INITIAL_PHASE_TERMINATION_MESSAGE": phase1_termination_message,
        "INITIAL_PHASE_SUCCESS": phase1_success,
        "iterations": final_result_snapshot.optimizer_result["iterations"],
        "TERMINATION_MESSAGE": final_result_snapshot.optimizer_result[
            "termination_message"
        ],
        "OPTIMIZER_SUCCESS": final_result_snapshot.optimizer_result["success"],
        "OPTIMIZER_STATUS": final_result_snapshot.optimizer_result["status"],
        "OPTIMIZER_NFEV": final_result_snapshot.optimizer_result["nfev"],
        "OPTIMIZER_NJEV": final_result_snapshot.optimizer_result["njev"],
        "OPTIMIZER_LS_STATUS": final_result_snapshot.optimizer_result["ls_status"],
        "TARGET_VOLUME": float(vol_target),
        "TARGET_IOTA": float(iota_target),
        "FINAL_VOLUME": float(final_result_snapshot.final_metrics["final_volume"]),
        "FINAL_IOTA": float(final_result_snapshot.final_metrics["final_iota"]),
        "FINAL_G": final_result_snapshot.final_metrics["final_G"],
        "FINAL_NON_QS": final_result_snapshot.final_metrics["final_non_qs"],
        "FINAL_BOOZER_RESIDUAL": final_result_snapshot.final_metrics[
            "final_boozer_residual"
        ],
        "FINAL_IOTA_PENALTY": final_result_snapshot.final_metrics[
            "final_iota_penalty"
        ],
        "FINAL_LENGTH_PENALTY": final_result_snapshot.final_metrics[
            "final_length_penalty"
        ],
        "FINAL_CURVE_CURVE_PENALTY": final_result_snapshot.final_metrics[
            "final_curve_curve_penalty"
        ],
        "FINAL_CURVE_SURFACE_PENALTY": final_result_snapshot.final_metrics[
            "final_curve_surface_penalty"
        ],
        "FINAL_SURFACE_VESSEL_PENALTY": final_result_snapshot.final_metrics[
            "final_surface_vessel_penalty"
        ],
        "FINAL_CURVATURE_PENALTY": final_result_snapshot.final_metrics[
            "final_curvature_penalty"
        ],
        "FIELD_ERROR": float(final_result_snapshot.field_error),
        "SELF_INTERSECTING": final_result_snapshot.self_intersecting,
        "SELF_INTERSECTION_CHECK_AVAILABLE": (
            final_result_snapshot.self_intersection_check_available
        ),
        "MAX_CURVATURE": float(final_result_snapshot.final_metrics["max_curvature"]),
        "COIL_LENGTH": float(final_result_snapshot.final_metrics["coil_length"]),
        "CURVE_CURVE_MIN_DIST": final_result_snapshot.final_distances[
            "curve_curve_min_dist"
        ],
        "CURVE_SURFACE_MIN_DIST": final_result_snapshot.final_distances[
            "curve_surface_min_dist"
        ],
        "SURFACE_VESSEL_MIN_DIST": final_result_snapshot.final_distances[
            "surface_vessel_min_dist"
        ],
        "COIL_VESSEL_MIN_DIST_M": COIL_VESSEL_MIN_DIST_M,
        "BANANA_CURRENT_A": float(final_banana_current_A),
        "TF_CURRENT_A": float(stage2_tf_current_A),
        "HARDWARE_CONSTRAINTS_OK": final_result_snapshot.hardware_status["success"],
        "HARDWARE_CONSTRAINT_VIOLATIONS": final_result_snapshot.hardware_status[
            "violations"
        ],
        "INITIAL_VOLUME": float(initial_volume),
        "INITIAL_IOTA": float(initial_iota),
        "INITIAL_FIELD_ERROR": float(initial_field_error),
        "INITIAL_MAX_CURVATURE": float(initial_max_curvature),
        "num_tf_coils": static_config["num_tf_coils"],
        "tf_currents": static_config["tf_currents"],
    }
    if alm_result is not None:
        results.update(
            {
                "ALM_PARTIAL_STATE_FILENAME": "alm_state.partial.json",
                "ALM_OUTER_ITERATIONS": int(alm_result.outer_iterations),
                "ALM_FINAL_PENALTY": float(alm_result.penalty),
                "ALM_CONSTRAINT_NAMES": list(alm_result.constraint_names),
                "ALM_CONSTRAINT_VALUES": list(alm_result.constraint_values),
                "ALM_SOLVER_CONSTRAINT_VALUES": list(
                    alm_result.solver_constraint_values
                ),
                "ALM_MULTIPLIERS": list(alm_result.multipliers),
                "ALM_TRUST_RADIUS": alm_result.trust_radius,
                "ALM_TERMINATION_REASON": getattr(
                    alm_result,
                    "termination_reason",
                    None,
                ),
                "ALM_CONVERGED": getattr(
                    alm_result,
                    "converged_to_tolerances",
                    None,
                ),
                "ALM_CONVERGED_TO_TOLERANCES": getattr(
                    alm_result,
                    "converged_to_tolerances",
                    None,
                ),
                "ALM_RESTORED_BEST_FEASIBLE": getattr(
                    alm_result,
                    "restored_best_feasible",
                    None,
                ),
                "ALM_RESTORED_BEST_FEASIBLE_REASON": getattr(
                    alm_result,
                    "restored_best_feasible_reason",
                    None,
                ),
                "ALM_INNER_OPTIMIZER_SUCCESS": getattr(
                    alm_result,
                    "optimizer_success",
                    None,
                ),
                "ALM_INNER_OPTIMIZER_MESSAGE": getattr(
                    alm_result,
                    "optimizer_message",
                    None,
                ),
                "ALM_FINAL_MAX_FEASIBILITY_VIOLATION": getattr(
                    alm_result,
                    "final_max_feasibility_violation",
                    None,
                ),
                "ALM_FINAL_STATIONARITY_NORM": getattr(
                    alm_result,
                    "final_stationarity_norm",
                    None,
                ),
                "ALM_FINAL_FEASIBILITY_TOL": getattr(
                    alm_result,
                    "final_feasibility_tolerance",
                    None,
                ),
                "ALM_FINAL_STATIONARITY_TOL": getattr(
                    alm_result,
                    "final_stationarity_tolerance",
                    None,
                ),
                "ALM_FINAL_MULTIPLIERS": list(alm_result.multipliers),
                "ALM_FINAL_CONSTRAINT_VALUES": list(alm_result.constraint_values),
                "ALM_FINAL_SOLVER_CONSTRAINT_VALUES": list(
                    alm_result.solver_constraint_values
                ),
                "ALM_FINAL_TRUST_RADIUS": alm_result.trust_radius,
                **alm_result_diagnostics_fields(alm_result),
            }
        )
    initial_objective = float(run_dict["initial_objective"])
    final_objective = final_result_snapshot.optimizer_diagnostics["fun"]
    objective_decrease = (
        None if final_objective is None else initial_objective - float(final_objective)
    )
    results["OPTIMIZER_FUN"] = final_result_snapshot.optimizer_diagnostics["fun"]
    results["OPTIMIZER_FUN_FINITE"] = final_result_snapshot.optimizer_diagnostics[
        "fun_finite"
    ]
    results["OPTIMIZER_JAC_FINITE"] = final_result_snapshot.optimizer_diagnostics[
        "jac_finite"
    ]
    results["OPTIMIZER_JAC_INF_NORM"] = final_result_snapshot.optimizer_diagnostics[
        "jac_inf_norm"
    ]
    results["OPTIMIZER_X_FINITE"] = final_result_snapshot.optimizer_diagnostics[
        "x_finite"
    ]
    results["OPTIMIZER_INVALID_STATE"] = final_result_snapshot.optimizer_diagnostics[
        "invalid_state"
    ]
    results["INITIAL_OBJECTIVE"] = initial_objective
    results["FINAL_OBJECTIVE"] = final_objective
    results["OBJECTIVE_DECREASE"] = objective_decrease
    if use_target_lane and (
        target_lane_invalid_state_events
        or final_result_snapshot.optimizer_diagnostics["invalid_state"]
    ):
        results["TARGET_LANE_INVALID_STATE_DIAGNOSIS"] = {
            "event_count": int(len(target_lane_invalid_state_events)),
            "event_recording_enabled": True,
            "diagnostic_callback_recording_enabled": bool(
                diagnostic_target_lane_callbacks
            ),
            "initial_phase": {
                "iterations": phase1_iterations,
                "termination_message": phase1_termination_message,
                "success": phase1_success,
            },
            "main_phase": {
                "iterations": main_phase_iterations,
                "termination_message": termination_message,
                "success": optimizer_success,
            },
            "events": target_lane_invalid_state_events,
            "diagnostic_events": target_lane_invalid_state_diagnostic_events,
            "note": (
                None
                if target_lane_invalid_state_events
                else (
                    "optimizer reported an invalid target-lane state without a "
                    "structured invalid-step solver-result entry"
                )
            ),
        }
    if target_lane_profile is not None:
        results["TARGET_LANE_PROFILE"] = target_lane_profile
    if jax_compile_diagnostics is not None:
        results["JAX_COMPILE_DIAGNOSTICS"] = jax_compile_diagnostics
    if target_lane_gradient_diagnosis is not None:
        results["TARGET_LANE_GRADIENT_DIAGNOSIS"] = target_lane_gradient_diagnosis
    if target_lane_first_line_search_diagnosis is not None:
        results["TARGET_LANE_FIRST_LINE_SEARCH_DIAGNOSIS"] = (
            target_lane_first_line_search_diagnosis
        )
    if target_lane_scaled_phase1_diagnosis is not None:
        results["TARGET_LANE_SCALED_PHASE1_DIAGNOSIS"] = (
            target_lane_scaled_phase1_diagnosis
        )
    if args.jax_profile_dir:
        results["JAX_PROFILE_DIR"] = os.path.abspath(args.jax_profile_dir)
    if stop_jax_profile_trace is not None:
        stop_jax_profile_trace()
    if target_lane_gradient_diagnosis is not None:
        write_json_file(
            os.path.join(OUT_DIR_ITER, "target_lane_gradient_diagnosis.json"),
            target_lane_gradient_diagnosis,
        )
    if target_lane_first_line_search_diagnosis is not None:
        write_json_file(
            os.path.join(OUT_DIR_ITER, "target_lane_first_line_search_diagnosis.json"),
            target_lane_first_line_search_diagnosis,
        )
    if target_lane_scaled_phase1_diagnosis is not None:
        write_json_file(
            os.path.join(OUT_DIR_ITER, "target_lane_scaled_phase1_diagnosis.json"),
            target_lane_scaled_phase1_diagnosis,
        )
    if jax_compile_diagnostics is not None:
        write_json_file(
            os.path.join(OUT_DIR_ITER, "jax_compile_diagnostics.json"),
            jax_compile_diagnostics,
        )
    if "TARGET_LANE_INVALID_STATE_DIAGNOSIS" in results:
        write_json_file(
            os.path.join(OUT_DIR_ITER, "target_lane_invalid_state_diagnosis.json"),
            results["TARGET_LANE_INVALID_STATE_DIAGNOSIS"],
        )
    if use_target_lane and write_restart_artifacts:
        write_single_stage_final_runtime_seed_spec(
            output_dir=OUT_DIR_ITER,
            surface=boozer_surface.surface,
            solved_surface_state=final_result_snapshot.solved_surface_state,
            field_source=bs,
            final_coil_dofs=final_result_snapshot.final_coil_dofs,
            num_tf_coils=num_tf_coils,
            tf_current_A=stage2_tf_current_A,
            banana_current_A=final_banana_current_A,
            stage2_seed=stage2_seed_payload,
        )
    if single_stage_host_artifact_export_required(
        use_target_lane=use_target_lane,
        write_restart_artifacts=write_restart_artifacts,
        write_full_artifacts=write_full_artifacts,
    ):
        if single_stage_final_host_restore_required(
            skip_outer_optimizer=skip_outer_optimizer,
            use_target_lane=use_target_lane,
            host_state_restored_for_final=host_state_restored_for_final,
        ):
            restore_single_stage_host_state(
                use_target_lane=use_target_lane,
                JF=JF,
                boozer_surface=boozer_surface,
                run_dict=run_dict,
                coil_dofs=final_result_snapshot.final_coil_dofs,
                apply_coil_dofs=dof_setter,
                bs_diag=bs_diag,
                record_outer_optimizer_event=record_outer_optimizer_event,
            )
        export_requested_single_stage_artifacts(
            solved_surface_state=final_result_snapshot.solved_surface_state,
            coil_dofs=final_result_snapshot.final_coil_dofs,
            num_tf_coils=num_tf_coils,
            tf_current_A=stage2_tf_current_A,
            banana_current_A=final_banana_current_A,
            stage2_seed=stage2_seed_payload,
            output_dir=OUT_DIR_ITER,
            boozer_surface=boozer_surface,
            bs_diag=bs_diag,
            surf_coils=surf_coils,
            hbt=hbt,
            VV=VV,
            write_restart_artifacts=write_restart_artifacts and not use_target_lane,
            write_host_restart_artifacts=write_restart_artifacts and not use_target_lane,
            write_full_artifacts=write_full_artifacts,
            timings=timings,
        )
    final_result_snapshot = replace(final_result_snapshot, timings=dict(timings))
    final_result_snapshot = with_single_stage_results_payload(
        final_result_snapshot,
        results,
    )
    write_single_stage_results_json(OUT_DIR_ITER, final_result_snapshot)
