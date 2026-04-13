import argparse
from contextlib import contextmanager
from dataclasses import dataclass, fields as dataclass_fields, is_dataclass
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


configure_entrypoint_jax_runtime(sys.argv[1:])

import jax
import jaxlib
import numpy as np

bootstrap_local_simsopt(SRC_ROOT)

# SIMSOPT imports
from simsopt._core.derivative import derivative_dec
from simsopt._core.optimizable import Optimizable, load
from simsopt.config import maybe_initialize_distributed_jax
from simsopt.field import BiotSavart
from simsopt.jax_core._math_utils import (
    as_jax_float64 as _as_jax_float64,
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
from simsopt.geo.curveobjectives import (
    CurveCurveDistance,
    CurveSurfaceDistance,
    pairwise_min_distance_pure,
)
import simsopt.geo.surface as surface_module
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
CURVATURE_THRESHOLD_CEILING = 40.0
TARGET_LANE_ACCEPTED_STEP_SYNC_CHOICES = ("per-accept", "final-only")
TARGET_LANE_ACCEPTED_STEP_SYNC_DEFAULT = "final-only"
_REFERENCE_OUTER_MAXLS_DEFAULT = 20
_TARGET_OUTER_MAXLS_BENCHMARK_DEFAULT = 4
_TARGET_OUTER_MAXLS_DEFAULT = 8
_TARGET_OUTER_INITIAL_STEP_SIZE_BENCHMARK_DEFAULT = 1.0e-4
_REFERENCE_OUTER_MAXCOR_DEFAULT = 300
_TARGET_OUTER_MAXCOR_DEFAULT = 20
_TARGET_LANE_BOOZER_BFGS_TOL_DEFAULT = 1e-8
_TARGET_LANE_BOOZER_BFGS_TOL_BENCHMARK_DEFAULT = 1e-6
_TARGET_LANE_BOOZER_BFGS_MAXITER_BENCHMARK_DEFAULT = 64
_SINGLE_STAGE_RESULTS_SCHEMA_VERSION = 1
_JAX_COMPILE_DIAGNOSTICS_SAMPLE_LIMIT = 25
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
            array = np.asarray(host_array(value), dtype=np.float64)
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
        (fun_finite is False)
        or (jac_finite is False)
        or (x_finite is False)
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


def _summarize_host_gradient(gradient):
    array = np.asarray(host_array(gradient), dtype=np.float64).reshape(-1)
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
    array = np.asarray(host_array(vector), dtype=np.float64).reshape(-1)
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
            {
                "phase": phase,
                "iteration": int(iteration),
                "step_scale": _summarize_host_scalar(step_scale),
                "line_search_failed": bool(line_search_failed),
                "nonfinite_step": bool(nonfinite_step),
                "stalled_step": bool(stalled_step),
                "valid_curvature": bool(valid_curvature),
                "trial_converged": bool(trial_converged),
                "ls_status": int(ls_status),
                "trial_value": _summarize_host_scalar(trial_f),
                "trial_x": _summarize_host_vector(trial_x),
                "trial_grad": _summarize_host_vector(trial_g),
                "search_direction": _summarize_host_vector(search_direction),
                "step_vector": _summarize_host_vector(step_vector),
            }
        )

    return _record


def record_target_lane_invalid_state_events_enabled(args) -> bool:
    """Return whether rejected target-lane trial events should be recorded."""

    return bool(getattr(args, "record_target_lane_invalid_state_events", False))


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
    R0,
    s,
    order,
    CONSTRAINT_WEIGHT,
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
    args,
    MAXITER,
    write_restart_artifacts,
    write_full_artifacts,
):
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
            "biot_savart_path": os.path.abspath(stage2_bs_path),
            "results_path": os.path.abspath(stage2_results_path),
            "major_radius": float(R0),
            "toroidal_flux": float(s),
            "banana_surface_radius": float(stage2_results["banana_surf_radius"]),
            "order": int(order),
        },
        "warm_start": {
            "run_dir": None
            if warm_start_run_dir is None
            else os.path.abspath(warm_start_run_dir),
            "surface_path": None
            if warm_start_state is None
            else warm_start_state["surface_path"],
            "results_path": None
            if warm_start_state is None
            else warm_start_state["results_path"],
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
            "surface_vessel_distance": float(SS_DIST),
            "curvature": float(CURVATURE_THRESHOLD),
        },
        "runtime_contract": {
            "field_backend": args.backend,
            "optimizer_backend": optimizer_backend_record,
            "boozer_optimizer_backend": boozer_optimizer_backend_record,
            "boozer_least_squares_algorithm": boozer_least_squares_algorithm_record,
            "outer_optimizer_method": outer_optimizer_method,
            "target_lane_accepted_step_sync": target_lane_sync_record,
            "experimental_target_lane_value_and_grad": bool(
                requested_experimental_target_lane_vg
            ),
            "target_lane_value_and_grad": bool(use_target_lane_vg),
            "max_iterations": int(MAXITER),
            "maxcor": int(args.maxcor),
            "outer_maxls": int(args.outer_maxls),
            "target_lane_outer_initial_step_size": args.target_lane_outer_initial_step_size,
            "initial_step_scale": float(args.initial_step_scale),
            "initial_step_maxiter": int(args.initial_step_maxiter),
            "target_lane_boozer_bfgs_tol": target_lane_boozer_bfgs_tol_record,
            "target_lane_boozer_bfgs_maxiter": target_lane_boozer_bfgs_maxiter_record,
            "target_lane_boozer_newton_tol": target_lane_boozer_newton_tol_record,
            "target_lane_boozer_newton_maxiter": target_lane_boozer_newton_maxiter_record,
            "benchmark_mode": bool(args.benchmark_mode),
            "minimal_artifacts": bool(args.minimal_artifacts),
            "init_only": bool(args.init_only),
            "profile_target_lane_only": bool(args.profile_target_lane_only),
            "profile_target_lane_batch_size": int(args.profile_target_lane_batch_size),
            "diagnose_target_lane_gradient": bool(
                getattr(args, "diagnose_target_lane_gradient", False)
            ),
            "diagnose_target_lane_scaled_phase1": bool(
                getattr(args, "diagnose_target_lane_scaled_phase1", False)
            ),
            "record_target_lane_invalid_state_events": bool(
                getattr(args, "record_target_lane_invalid_state_events", False)
            ),
            "disable_target_lane_success_filter": bool(
                args.disable_target_lane_success_filter
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
    R0,
    s,
    order,
    CONSTRAINT_WEIGHT,
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
    args,
    MAXITER,
    write_restart_artifacts,
    write_full_artifacts,
):
    required_files = ["results.json"]
    planned_files = ["results.json"]
    if getattr(args, "diagnose_target_lane_gradient", False):
        required_files.append("target_lane_gradient_diagnosis.json")
        planned_files.append("target_lane_gradient_diagnosis.json")
    if getattr(args, "diagnose_target_lane_scaled_phase1", False):
        required_files.append("target_lane_scaled_phase1_diagnosis.json")
        planned_files.append("target_lane_scaled_phase1_diagnosis.json")
    if write_restart_artifacts:
        required_files.extend(("biot_savart_opt.json", "surf_opt.json"))
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
            R0=R0,
            s=s,
            order=order,
            CONSTRAINT_WEIGHT=CONSTRAINT_WEIGHT,
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
    if (
        "before_boozer_lbfgs" in stage_marks
        and "after_boozer_lbfgs" in stage_marks
    ):
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
    curve_curve_min_dist = float(curve_curve_min_dist)
    cc_dist = float(cc_dist)
    curve_surface_min_dist = float(curve_surface_min_dist)
    cs_dist = float(cs_dist)
    surface_vessel_min_dist = float(surface_vessel_min_dist)
    ss_dist = float(ss_dist)
    max_curvature = float(max_curvature)
    curvature_threshold = float(curvature_threshold)
    violations = []
    for metric_name, metric_value in (
        ("coil_coil_min_dist", curve_curve_min_dist),
        ("cc_dist", cc_dist),
        ("coil_surface_min_dist", curve_surface_min_dist),
        ("cs_dist", cs_dist),
        ("surface_vessel_min_dist", surface_vessel_min_dist),
        ("ss_dist", ss_dist),
        ("max_curvature", max_curvature),
        ("curvature_threshold", curvature_threshold),
    ):
        if not np.isfinite(metric_value):
            violations.append(f"{metric_name} {metric_value} is not finite")
    if curve_curve_min_dist < cc_dist:
        violations.append(
            f"coil_coil_min_dist {curve_curve_min_dist:.6f} below threshold {cc_dist:.6f}"
        )
    if curve_surface_min_dist < cs_dist:
        violations.append(
            f"coil_surface_min_dist {curve_surface_min_dist:.6f} below threshold {cs_dist:.6f}"
        )
    if surface_vessel_min_dist < ss_dist:
        violations.append(
            f"surface_vessel_min_dist {surface_vessel_min_dist:.6f} below threshold {ss_dist:.6f}"
        )
    if max_curvature > curvature_threshold:
        violations.append(
            f"max_curvature {max_curvature:.6f} exceeds threshold {curvature_threshold:.6f}"
        )
    return {
        "success": len(violations) == 0,
        "violations": violations,
        "curve_curve_min_dist": curve_curve_min_dist,
        "cc_dist": cc_dist,
        "curve_surface_min_dist": curve_surface_min_dist,
        "cs_dist": cs_dist,
        "surface_vessel_min_dist": surface_vessel_min_dist,
        "ss_dist": ss_dist,
        "max_curvature": max_curvature,
        "curvature_threshold": curvature_threshold,
    }


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
        return adapter.callback
    return None


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
) -> bool:
    """Return whether the run should emit heavy plotting/VTK artifacts."""
    return (not benchmark_mode) and (not minimal_artifacts)


def should_write_single_stage_restart_artifacts(benchmark_mode: bool) -> bool:
    """Return whether the run should emit final JSON artifacts for warm restarts."""
    return not benchmark_mode


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
    return stage2_results_path, stage2_results


def resolve_single_stage_warm_start_paths(run_dir):
    resolved_run_dir = os.path.abspath(run_dir)
    surface_path = os.path.join(resolved_run_dir, "surf_opt.json")
    results_path = os.path.join(resolved_run_dir, "results.json")
    missing_paths = [
        path
        for path in (surface_path, results_path)
        if not os.path.exists(path)
    ]
    if missing_paths:
        raise FileNotFoundError(
            "single-stage warm start run directory is missing required artifacts: "
            + ", ".join(missing_paths)
        )
    return surface_path, results_path


def load_single_stage_warm_start_state(run_dir):
    surface_path, results_path = resolve_single_stage_warm_start_paths(run_dir)
    surface = load(surface_path)
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
    }


def project_single_stage_warm_start_surface_dofs(
    surface,
    *,
    mpol,
    ntor,
    quadpoints_phi,
    quadpoints_theta,
):
    projected_surface = SurfaceXYZTensorFourier(
        mpol=max(1, int(mpol)),
        ntor=max(1, int(ntor)),
        nfp=int(getattr(surface, "nfp", 5)),
        stellsym=bool(getattr(surface, "stellsym", True)),
        quadpoints_theta=quadpoints_theta,
        quadpoints_phi=quadpoints_phi,
    )
    if isinstance(surface, SurfaceXYZTensorFourier):
        evaluation_surface = SurfaceXYZTensorFourier(
            mpol=max(1, int(getattr(surface, "mpol", mpol))),
            ntor=max(1, int(getattr(surface, "ntor", ntor))),
            nfp=int(getattr(surface, "nfp", 5)),
            stellsym=bool(getattr(surface, "stellsym", True)),
            quadpoints_theta=quadpoints_theta,
            quadpoints_phi=quadpoints_phi,
        )
        evaluation_surface.set_dofs(np.asarray(surface.get_dofs(), dtype=float))
        target_gamma = np.asarray(evaluation_surface.gamma(), dtype=float)
    else:
        target_gamma = np.stack(
            [
                np.asarray(
                    surface.cross_section(float(phi), thetas=quadpoints_theta),
                    dtype=float,
                )
                for phi in np.asarray(quadpoints_phi, dtype=float)
            ],
            axis=0,
        )
    projected_surface.least_squares_fit(target_gamma)
    return np.asarray(projected_surface.get_dofs(), dtype=float)


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
        "--minimal-artifacts",
        action="store_true",
        help=(
            "Write only the JSON artifacts needed for warm restarts and skip "
            "heavy VTK/plot outputs."
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
        default=int(os.environ["OUTER_MAXLS"])
        if "OUTER_MAXLS" in os.environ
        else None,
        help=(
            "Maximum strong-Wolfe line-search evaluations per outer L-BFGS step. "
            "Defaults to a tighter budget on the JAX ondevice lane and the "
            "historical budget elsewhere."
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
            "Defaults to 'lm' on the JAX ondevice lane and 'quasi-newton' "
            "elsewhere when omitted."
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
        "--diagnose-target-lane-scaled-phase1",
        action="store_true",
        help=(
            "Run the exact JAX/ondevice scaled initial-phase contract in "
            "diagnostic mode, recording origin/trial/reevaluation finiteness "
            "before and after the phase-1 optimizer."
        ),
    )
    parser.add_argument(
        "--record-target-lane-invalid-state-events",
        action="store_true",
        help=(
            "Opt in to rejected-step target-lane failure diagnostics. This "
            "installs host-callback postmortem recording on the ondevice outer "
            "optimizer and is disabled by default for production throughput."
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
    args = parser.parse_args()
    args.boozer_least_squares_algorithm_explicit = (
        args.boozer_least_squares_algorithm is not None
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
    args.target_lane_boozer_bfgs_maxiter = (
        resolve_target_lane_boozer_bfgs_maxiter(
            args.backend,
            args.optimizer_backend,
            args.target_lane_boozer_bfgs_maxiter,
            benchmark_mode=args.benchmark_mode,
        )
    )
    args.target_lane_boozer_newton_tol = resolve_target_lane_boozer_newton_tol(
        args.backend,
        args.optimizer_backend,
        args.target_lane_boozer_newton_tol,
    )
    args.target_lane_boozer_newton_maxiter = (
        resolve_target_lane_boozer_newton_maxiter(
            args.backend,
            args.optimizer_backend,
            args.target_lane_boozer_newton_maxiter,
        )
    )
    if args.profile_target_lane_only:
        args.profile_target_lane = True
    if args.profile_target_lane_batch_size < 1:
        raise ValueError("--profile-target-lane-batch-size must be at least 1")
    if not (0.0 < args.initial_step_scale <= 1.0):
        raise ValueError("--initial-step-scale must be in (0, 1]")
    if args.initial_step_maxiter < 0:
        raise ValueError("--initial-step-maxiter must be non-negative")
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

        booz_surf = self.boozer_surface
        P, L, U = booz_surf.res["PLU"]
        dconstraint_dcoils_vjp = booz_surf.res["vjp"]

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

        adj = forward_backward(P, L, U, dJ_ds)

        adj_times_dg_dcoil = dconstraint_dcoils_vjp(adj, booz_surf, iota, G)
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
        for warm-started JAX Boozer initialization
    bfgs_maxiter_override: optional first-stage least-squares iteration cap
        override for warm-started JAX Boozer initialization
    newton_tol_override: optional Newton polish tolerance override for
        warm-started JAX Boozer initialization
    newton_maxiter_override: optional Newton polish iteration cap override
        for warm-started JAX Boozer initialization
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
        if backend == "jax" and on_stage is not None and jax_solver_stage_callback_supported():
            options["stage_callback"] = on_stage
        return options

    def resolve_boozer_warm_start():
        solve_iota = float(iota_override) if iota_override is not None else iota
        solve_G = float(G_override) if G_override is not None else G0
        solve_sdofs = (
            None
            if surface_dofs_override is None
            else np.asarray(surface_dofs_override, dtype=float)
        )
        return solve_iota, solve_G, solve_sdofs

    def run_boozer_solve(boozer_surface, solve_iota, solve_G, solve_sdofs):
        if backend == "jax":
            return boozer_surface.run_code(solve_iota, solve_G, sdofs=solve_sdofs)
        return boozer_surface.run_code(solve_iota, solve_G)

    total_start_s = _perf_counter_s()
    fit_start_s = total_start_s
    surf = SurfaceXYZTensorFourier(
        mpol=mpol,
        ntor=ntor,
        nfp=5,
        stellsym=True,
        quadpoints_theta=surf_prev.quadpoints_theta,
        quadpoints_phi=surf_prev.quadpoints_phi,
    )
    if surface_dofs_override is None:
        surf.least_squares_fit(surf_prev.gamma())
    else:
        surf.set_dofs(np.asarray(surface_dofs_override, dtype=float))
    fit_end_s = _perf_counter_s()
    _record_timing(timings, "boozer_surface_fit_s", fit_start_s, fit_end_s)
    emit_stage("after_boozer_surface_fit")

    if backend == "jax":
        from simsopt.geo.boozersurface_jax import BoozerSurfaceJAX

        BoozerCls = BoozerSurfaceJAX
    else:
        BoozerCls = BoozerSurface

    solver_name = "JAX " if backend == "jax" else ""
    setup_start_s = _perf_counter_s()
    if constraint_weight is not None:
        print(f"Generating {solver_name}Boozer least squares surface...")
        vol = Volume(surf)
        options = {"verbose": True}
        if backend == "jax":
            resolved_optimizer_backend = resolve_boozer_optimizer_backend(
                backend,
                "ondevice",
                optimizer_backend,
            )
            options["optimizer_backend"] = resolved_optimizer_backend
            if boozer_least_squares_algorithm is not None:
                options["least_squares_algorithm"] = boozer_least_squares_algorithm
            if resolved_optimizer_backend == "ondevice" and boozer_limited_memory:
                options["force_ondevice_limited_memory"] = True
            if bfgs_tol_override is not None:
                options["bfgs_tol"] = float(bfgs_tol_override)
            if bfgs_maxiter_override is not None:
                options["bfgs_maxiter"] = int(bfgs_maxiter_override)
            if newton_tol_override is not None:
                options["newton_tol"] = float(newton_tol_override)
            if newton_maxiter_override is not None:
                options["newton_maxiter"] = int(newton_maxiter_override)
            options.update(build_jax_stage_options())
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
        surf_exact = SurfaceXYZTensorFourier(
            mpol=mpol,
            ntor=ntor,
            nfp=5,
            stellsym=True,
            quadpoints_theta=np.linspace(0, 1, 2 * mpol + 1, endpoint=False),
            quadpoints_phi=np.linspace(0, 1.0 / surf.nfp, 2 * ntor + 1, endpoint=False),
            dofs=surf.dofs,
        )
        vol = Volume(surf_exact)
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
    (
        self_intersecting,
        self_intersection_check_available,
    ) = evaluate_surface_self_intersection(boozer_surface.surface)
    success2 = not self_intersecting  # True if surface is not self intersecting
    success = success1 and success2
    if not self_intersection_check_available:
        print(
            "Skipping surface self-intersection check because "
            "ground+bentley_ottmann or shapely is unavailable."
        )
    if not success:
        print(
            "Boozer initialization failed: "
            f"solve_success={success1}, "
            f"self_intersecting={self_intersecting}, "
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
    if benchmark_mode and boozer_surface.res is not None and "iota" in boozer_surface.res:
        return host_float(boozer_surface.res["iota"])
    return host_float(build_iota_objective(boozer_surface, iota_cls).J())


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

    if use_target_lane and not skip_outer_optimizer:
        runtime_bundle = get_traceable_single_stage_runtime_bundle_builder()(
            boozer_surface,
            bs,
            iota_target,
            include_profile_suite=False,
            include_host_wrappers=True,
            outer_objective_config=outer_objective_config,
            success_filter=success_filter,
        )
        metrics = dict(runtime_bundle["host_reporting_metrics"](_as_jax_float64(coil_dofs)))
        if benchmark_mode:
            metrics["curve_curve_min_dist"] = None
            metrics["curve_surface_min_dist"] = None
            metrics["surface_vessel_min_dist"] = None
            metrics["hardware_status"] = benchmark_hardware_status
        else:
            metrics["hardware_status"] = evaluate_single_stage_hardware_constraints(
                metrics["curve_curve_min_dist"],
                cc_dist,
                metrics["curve_surface_min_dist"],
                cs_dist,
                metrics["surface_vessel_min_dist"],
                ss_dist,
                metrics["max_curvature"],
                curvature_threshold,
            )
        return metrics

    max_curvature = float(np.max(j_curvature.curve.kappa()))
    final_curve_curve_min_dist = None
    final_curve_surface_min_dist = None
    final_surface_vessel_min_dist = None
    if benchmark_mode:
        final_hardware_status = benchmark_hardware_status
    else:
        final_curve_curve_min_dist = host_float(j_curve_curve.shortest_distance())
        final_curve_surface_min_dist = host_float(j_curve_surface.shortest_distance())
        final_surface_vessel_min_dist = host_float(j_surface_surface.shortest_distance())
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
        "curve_curve_min_dist": final_curve_curve_min_dist,
        "curve_surface_min_dist": final_curve_surface_min_dist,
        "surface_vessel_min_dist": final_surface_vessel_min_dist,
        "hardware_status": final_hardware_status,
    }


def surface_self_intersection_check_available():
    """Return whether the optional surface self-intersection backend is present."""
    has_ground = (
        surface_module.get_context is not None
        and surface_module.contour_self_intersects is not None
    )
    return has_ground or getattr(surface_module, "LineString", None) is not None


def evaluate_surface_self_intersection(surface):
    """Return (intersecting, check_available) for a SIMSOPT surface."""
    check_available = surface_self_intersection_check_available()
    if not check_available:
        return False, False
    return bool(surface.is_self_intersecting()), True


def update_self_intersection_status(run_dict, surface):
    """Refresh self-intersection status in the shared run-state dictionary."""
    (
        run_dict["intersecting"],
        run_dict["self_intersection_check_available"],
    ) = evaluate_surface_self_intersection(surface)
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


def get_traceable_single_stage_runtime_diagnostic_builder():
    """Load the compact target-lane baseline diagnosis helper on demand."""
    from simsopt.geo.surfaceobjectives_jax import diagnose_traceable_objective_runtime

    return diagnose_traceable_objective_runtime


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
    import jax
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

    def _hostify_constant_tree(value):
        return jax.tree_util.tree_map(
            lambda leaf: host_array(leaf)
            if isinstance(leaf, jax.Array)
            else np.asarray(leaf)
            if isinstance(leaf, np.ndarray)
            else leaf,
            value,
        )

    optimize_G = boozer_surface.res.get("G") is not None
    coil_dof_extraction_spec = _hostify_constant_tree(bs.coil_dof_extraction_spec())
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

    if use_value_and_grad:
        target_value_and_grad_objective = runtime_bundle["value_and_grad"]
    else:
        target_scalar_objective = runtime_bundle["objective"]

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
            target_lane_profile["batched_seed_profile"][
                "profile_point_kind"
            ] = "baseline_perturbed_batch"

    return (
        target_scalar_objective,
        target_value_and_grad_objective,
        target_lane_profile,
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
            {
                stage_name: stage_records.get(stage_name)
                for stage_name in stage_names
            }
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
        include_host_wrappers=True,
        outer_objective_config=outer_objective_config,
        success_filter=success_filter,
    )
    _persist_payload("runtime_bundle_ready", diagnosis_complete=False)
    phase1_fun, phase1_callback = build_scaled_outer_problem(
        runtime_bundle["value_and_grad"],
        callback,
        anchor_dofs,
        step_scale,
        anchor_in_state=True,
    )
    anchor_host_dofs = np.asarray(host_array(anchor_dofs), dtype=np.float64).reshape(-1)

    def _base_host_value_and_grad(x):
        value, grad = runtime_bundle["host_value_and_grad"](_as_jax_float64(x))
        return float(value), np.asarray(grad, dtype=np.float64).reshape(-1)

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

    phase1_dofs = build_scaled_outer_phase_initial_dofs(anchor_dofs, use_target_lane=True)
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
    """Build target-lane outer objectives with an optional hard success filter."""
    target_scalar_objective = None
    target_value_and_grad_objective = None
    target_lane_profile = None
    target_lane_success_filter = None
    target_lane_outer_objective_config = None

    if not use_target_lane:
        return (
            target_scalar_objective,
            target_value_and_grad_objective,
            target_lane_profile,
            target_lane_success_filter,
        )

    if not disable_success_filter:
        target_lane_success_filter = (
            build_single_stage_target_lane_hardware_success_filter(
                boozer_surface,
                bs,
                banana_curve,
                VV,
                cc_dist=cc_dist,
                cs_dist=cs_dist,
                ss_dist=ss_dist,
                curvature_threshold=curvature_threshold,
            )
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
    return (
        target_scalar_objective,
        target_value_and_grad_objective,
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
    elif field_backend == "jax" and optimizer_backend == "ondevice":
        resolved = (
            _TARGET_LANE_BOOZER_BFGS_TOL_BENCHMARK_DEFAULT
            if benchmark_mode
            else _TARGET_LANE_BOOZER_BFGS_TOL_DEFAULT
        )
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
    applied = {
        key: value for key, value in overrides.items() if value is not None
    }
    if not applied:
        yield
        return

    original = {
        key: boozer_surface.options.get(key)
        for key in applied
    }
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


def _scaled_outer_problem_gradient(z, grad, scale):
    if isinstance(z, ScaledOuterPhaseOptimizerState):
        anchor_scale = _as_runtime_float64(0.0, reference=z.anchor_dofs)
        return ScaledOuterPhaseOptimizerState(
            step_dofs=scale * grad,
            anchor_dofs=anchor_scale * z.anchor_dofs,
        )
    return scale * grad


def should_force_strict_target_lane_final_sync(
    *,
    use_target_lane,
    res_nit,
    accepted_step_callback,
    trial_boozer_override_active,
):
    """Return whether the final accepted state must be re-synced strictly."""
    return bool(use_target_lane) and int(res_nit) > 0 and (
        accepted_step_callback is None or trial_boozer_override_active
    )


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

    def resolve_scale(reference):
        if isinstance(reference, jax.core.Tracer):
            return np.float64(step_scale)
        if isinstance(reference, jax.Array):
            return _as_runtime_float64(step_scale, reference=reference)
        return float(step_scale)

    def scaled_fun(z):
        if anchor_in_state:
            if not isinstance(z, ScaledOuterPhaseOptimizerState):
                raise TypeError(
                    "anchor_in_state=True requires ScaledOuterPhaseOptimizerState inputs."
                )
            step_dofs = z.step_dofs
            resolved_anchor = jax.lax.stop_gradient(z.anchor_dofs)
        else:
            step_dofs, resolved_anchor = _scaled_outer_problem_coordinates(z, anchor_x)
        scale = resolve_scale(step_dofs)
        x = resolved_anchor + scale * step_dofs
        value, grad = base_fun(x)
        return value, _scaled_outer_problem_gradient(z, grad, scale)

    if base_callback is None:
        return scaled_fun, None

    def scaled_callback(z):
        if anchor_in_state:
            if not isinstance(z, ScaledOuterPhaseOptimizerState):
                raise TypeError(
                    "anchor_in_state=True requires ScaledOuterPhaseOptimizerState inputs."
                )
            step_dofs = z.step_dofs
            resolved_anchor = jax.lax.stop_gradient(z.anchor_dofs)
        else:
            step_dofs, resolved_anchor = _scaled_outer_problem_coordinates(z, anchor_x)
        scale = resolve_scale(step_dofs)
        base_callback(resolved_anchor + scale * step_dofs)

    return scaled_fun, scaled_callback


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

    def resolve_scale(reference):
        if isinstance(reference, jax.core.Tracer):
            return np.float64(step_scale)
        if isinstance(reference, jax.Array):
            return _as_runtime_float64(step_scale, reference=reference)
        return float(step_scale)

    def scaled_fun(z):
        if anchor_in_state:
            if not isinstance(z, ScaledOuterPhaseOptimizerState):
                raise TypeError(
                    "anchor_in_state=True requires ScaledOuterPhaseOptimizerState inputs."
                )
            step_dofs = z.step_dofs
            resolved_anchor = jax.lax.stop_gradient(z.anchor_dofs)
        else:
            step_dofs, resolved_anchor = _scaled_outer_problem_coordinates(z, anchor_x)
        scale = resolve_scale(step_dofs)
        return base_fun(resolved_anchor + scale * step_dofs)

    if base_callback is None:
        return scaled_fun, None

    def scaled_callback(z):
        if anchor_in_state:
            if not isinstance(z, ScaledOuterPhaseOptimizerState):
                raise TypeError(
                    "anchor_in_state=True requires ScaledOuterPhaseOptimizerState inputs."
                )
            step_dofs = z.step_dofs
            resolved_anchor = jax.lax.stop_gradient(z.anchor_dofs)
        else:
            step_dofs, resolved_anchor = _scaled_outer_problem_coordinates(z, anchor_x)
        scale = resolve_scale(step_dofs)
        base_callback(resolved_anchor + scale * step_dofs)

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
                key: int(self.compile_targets[key]) for key in sorted(self.compile_targets)
            },
            "cache_miss_sites": {
                key: int(self.cache_miss_sites[key]) for key in sorted(self.cache_miss_sites)
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
    override_level = previous_level == logging.NOTSET or previous_level > logging.WARNING
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
    solved_plu = forward_result["plu"]
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
        solved_plu,
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
        if runtime_reference is not None:
            anchor_runtime = _as_runtime_float64(anchor_dofs, reference=runtime_reference)
            step_runtime = _as_runtime_float64(step_dofs, reference=runtime_reference)
            scale = _as_runtime_float64(step_scale, reference=runtime_reference)
            return anchor_runtime + scale * step_runtime
        return _as_jax_float64(anchor_dofs) + _as_jax_float64(step_scale) * (
            _as_jax_float64(step_dofs)
        )
    return _single_stage_optimizer_dofs_array(anchor_dofs) + float(step_scale) * (
        _single_stage_optimizer_dofs_array(step_dofs)
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
    scalar_fun=None,
    target_lane_initial_step_size=None,
    failure_callback=None,
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
        }
        if target_lane_initial_step_size is not None:
            target_minimize_kwargs["options"]["initial_step_size"] = float(
                target_lane_initial_step_size
            )
        if failure_callback is not None:
            target_minimize_kwargs["failure_callback"] = failure_callback
        return target_minimize(
            optimizer_fun,
            optimizer_dofs,
            **target_minimize_kwargs,
        )
    if not isinstance(contract, ReferenceOptimizerContract):
        raise RuntimeError(
            f"Unsupported single-stage optimizer contract {type(contract)!r}."
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
    )


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


def _update_line_search_state(x, run_dict):
    """Track step size and increment line-search counter."""
    dx = np.linalg.norm(x - run_dict["x_prev"])
    run_dict["x_prev"] = x.copy()
    logger.info("Step size: %.2e", dx)
    run_dict["lscount"] += 1


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
    is_intersecting = update_self_intersection_status(run_dict, boozer_surface.surface)
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
        J = run_dict["J"] + max(abs(run_dict["J"]), 1.0)
        dJ = run_dict["dJ"].copy()

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
    success = host_bool(boozer_surface.res["success"]) and not update_self_intersection_status(
        run_dict, boozer_surface.surface
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
    run_dict["lscount"] = 0
    run_dict["sdofs"] = boozer_surface.surface.x.copy()
    run_dict["iota"] = host_float(boozer_surface.res["iota"])
    run_dict["G"] = host_float(boozer_surface.res["G"])
    if store_objective_grad:
        if objective_value is None:
            objective_value = JF.J()
        if objective_grad is None:
            objective_grad = JF.dJ()
        run_dict["J"] = host_float(objective_value)
        run_dict["dJ"] = host_array(objective_grad)
    return run_dict["J"], run_dict["dJ"]


def accept_step(
    run_dict, boozer_surface, JF, bs, objectives, diagnostics_refs, log_path
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
    J, grad = snapshot_accepted_step_state(run_dict, boozer_surface, JF)

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
    update_self_intersection_status(run_dict, boozer_surface.surface)

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

    def sync_accepted_step(self, x):
        """Refresh mutable state if needed, then snapshot one accepted step."""
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


def snapshot_to_pytree(JF, boozer_surface, bs, *, num_tf_coils):
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
    coil_dofs = JF.x.copy()
    coils = bs.coils
    tf_coils = coils[:num_tf_coils]

    initial_objective = host_float(JF.J())
    initial_objective_grad = host_array(JF.dJ())

    run_dict = {
        "sdofs": boozer_surface.surface.x.copy(),
        "iota": host_float(boozer_surface.res["iota"]),
        "G": host_float(boozer_surface.res["G"]),
        "J": initial_objective,
        "dJ": initial_objective_grad,
        "initial_objective": initial_objective,
        "it": 1,
        "lscount": 0,
        "x_prev": coil_dofs.copy(),
        "intersecting": False,
        "self_intersection_check_available": (
            surface_self_intersection_check_available()
        ),
    }

    static_config = {
        "num_tf_coils": num_tf_coils,
        "tf_gamma": [c.curve.gamma().copy() for c in tf_coils],
        "tf_gammadash": [c.curve.gammadash().copy() for c in tf_coils],
        "tf_currents": [float(c.current.get_value()) for c in tf_coils],
    }

    return coil_dofs, run_dict, static_config


def restore_from_pytree(
    JF,
    boozer_surface,
    run_dict,
    coil_dofs=None,
    *,
    apply_coil_dofs=None,
):
    """Write optimization state back into the Optimizable graph.

    Restores coil DOFs, surface DOFs, and the warm-start scalars
    (``iota``, ``G``) so post-optimization consumers see values
    consistent with the last accepted step.

    Note: ``res["success"]``, ``res["PLU"]``, and ``res["vjp"]`` are
    NOT directly restored.  Setting ``surface.x`` or ``JF.x`` marks
    the Boozer surface dirty (``need_to_run_code = True``), so the
    next access through an ``IotasJAX`` / ``NonQuasiSymmetricRatioJAX``
    wrapper will trigger ``_ensure_solved`` which re-runs the inner
    solve and refreshes the full ``res`` dict automatically.

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
    boozer_surface.surface.x = run_dict["sdofs"]
    boozer_surface.res["iota"] = run_dict["iota"]
    boozer_surface.res["G"] = run_dict["G"]


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
    OUT_DIR = args.output_root
    os.makedirs(OUT_DIR, exist_ok=True)
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
    stage2_seed_setup_start_s = _perf_counter_s()
    stage2_bs_path = build_stage2_bs_path(args)
    stage2_results_path, stage2_results = load_stage2_results(stage2_bs_path)
    _mark_startup_progress(
        "stage2_seed_resolved",
        start_s=stage2_seed_setup_start_s,
    )
    R0 = float(stage2_results["MAJOR_RADIUS"])
    s = float(stage2_results["TOROIDAL_FLUX"])
    order = int(stage2_results.get("order", args.stage2_seed_order))

    banana_surf_radius = (
        args.banana_surf_radius
        if args.banana_surf_radius is not None
        else float(stage2_results["banana_surf_radius"])
    )
    banana_surf_nfp = 5
    nphi = args.nphi
    ntheta = args.ntheta
    mpol = args.mpol
    ntor = args.ntor

    # Optimization targets and weights
    vol_target = args.vol_target
    CONSTRAINT_WEIGHT = args.constraint_weight
    MAXITER = args.maxiter
    iota_target = args.iota_target
    num_tf_coils = args.num_tf_coils

    boozer_type = {"initial": "least_squares", "final": "exact"}  # example
    stage = args.boozer_stage

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
    stage2_bs_load_start_s = _perf_counter_s()
    bs = load(stage2_bs_path)
    _mark_startup_progress("stage2_bs_loaded", start_s=stage2_bs_load_start_s)
    warm_start_state_load_start_s = _perf_counter_s()
    warm_start_state = (
        None
        if args.warm_start_run_dir is None
        else load_single_stage_warm_start_state(args.warm_start_run_dir)
    )
    _mark_startup_progress(
        "warm_start_state_loaded",
        start_s=warm_start_state_load_start_s,
    )

    use_jax = args.backend == "jax"
    write_full_artifacts = should_write_single_stage_full_artifacts(
        args.benchmark_mode,
        args.minimal_artifacts,
    )
    write_restart_artifacts = should_write_single_stage_restart_artifacts(
        args.benchmark_mode
    )

    # JAX backend: wrap the loaded BiotSavart coils in BiotSavartJAX.
    # Keep a CPU reference for diagnostics and artifact output.
    bs_cpu_diag = bs if use_jax else None
    biotsavart_wrap_start_s = _perf_counter_s()
    if use_jax:
        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

        _, IotasJAX, NonQuasiSymmetricRatioJAX = get_jax_surface_objective_classes()
        bs = BiotSavartJAX(bs.coils)
        iota_cls = IotasJAX
    else:
        iota_cls = Iotas
    _mark_startup_progress("biotsavart_ready", start_s=biotsavart_wrap_start_s)

    bs_diag = diagnostic_field(bs, bs_cpu_diag)

    # Initialize the boundary magnetic surface and scale it to the target major radius
    plasma_surface_load_start_s = _perf_counter_s()
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
    warm_start_surface_dofs = (
        None
        if warm_start_state is None
        else project_single_stage_warm_start_surface_dofs(
            warm_start_state["surface"],
            mpol=mpol,
            ntor=ntor,
            quadpoints_phi=surf.quadpoints_phi,
            quadpoints_theta=surf.quadpoints_theta,
        )
    )
    _mark_startup_progress(
        "warm_start_surface_projected",
        start_s=warm_start_surface_project_start_s,
    )

    # Extract coil information
    coils = bs.coils
    curves = [c.curve for c in coils]
    tf_coils = coils[:num_tf_coils]
    tf_curves = [c.curve for c in tf_coils]
    banana_coils = coils[num_tf_coils:]
    banana_curves = [c.curve for c in banana_coils]
    banana_curve = banana_curves[0]
    current_sum = sum(abs(c.current.get_value()) for c in tf_coils)

    # Calculate G0 parameter from TF coil currents
    G0 = 2.0 * np.pi * current_sum * (4 * np.pi * 10 ** (-7) / (2 * np.pi))

    # ==============================================================================
    # OPTIMIZATION SETUP
    # ==============================================================================
    print(f"\n===== Starting single stage optimization for mpol = {mpol} =====")

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
    effective_warm_start_boozer_least_squares_algorithm = (
        warm_start_boozer_init_overrides["least_squares_algorithm_override"]
        or boozer_least_squares_algorithm_record
    )

    config_parts = [
        str(stage2_bs_path),
        str(stage),
        str(CONSTRAINT_WEIGHT),
        str(vol_target),
        str(iota_target),
        str(args.cc_dist),
        str(args.cc_weight),
        str(args.curvature_weight),
        str(args.curvature_threshold),
        str(args.length_weight),
        str(args.res_weight),
        str(args.iotas_weight),
        str(args.cs_weight),
        str(args.cs_dist),
        str(args.surf_dist_weight),
        str(args.ss_dist),
        str(args.maxcor),
        str(args.outer_maxls),
        str(args.target_lane_outer_initial_step_size),
        str(args.initial_step_scale),
        str(args.initial_step_maxiter),
        str(target_lane_boozer_bfgs_tol_record),
        str(target_lane_boozer_bfgs_maxiter_record),
        str(target_lane_boozer_newton_tol_record),
        str(target_lane_boozer_newton_maxiter_record),
        str(banana_surf_radius),
        str(nphi),
        str(ntheta),
        str(args.init_only),
        str(args.benchmark_mode),
        str(args.disable_target_lane_success_filter),
        str(args.profile_target_lane),
        str(args.profile_target_lane_only),
        str(args.diagnose_target_lane_gradient),
        str(args.minimal_artifacts),
        str(args.backend),
        str(optimizer_backend_record),
        str(target_lane_sync_record),
        str(use_target_lane_vg),
        str(args.maxiter),
        str(args.num_tf_coils),
        str(file_loc),
    ]
    if boozer_optimizer_backend_hash_record is not None:
        config_parts.append(str(boozer_optimizer_backend_hash_record))
    if effective_warm_start_boozer_least_squares_algorithm is not None:
        config_parts.append(str(effective_warm_start_boozer_least_squares_algorithm))
    config_str = "|".join(config_parts)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    OUT_DIR_ITER = OUT_DIR + f"/mpol={mpol}-ntor={ntor}-{config_hash}"
    os.makedirs(OUT_DIR_ITER, exist_ok=True)
    _mark_startup_progress("optimizer_output_dir_ready")
    boozer_init_progress = build_stage_progress_recorder(
        os.path.join(OUT_DIR_ITER, "boozer_init_progress.json")
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
            effective_warm_start_boozer_least_squares_algorithm
        ),
        boozer_least_squares_algorithm_override=warm_start_boozer_init_overrides[
            "least_squares_algorithm_override"
        ],
        bfgs_tol_override=warm_start_boozer_init_overrides["bfgs_tol_override"],
        bfgs_maxiter_override=warm_start_boozer_init_overrides["bfgs_maxiter_override"],
        newton_tol_override=warm_start_boozer_init_overrides["newton_tol_override"],
        newton_maxiter_override=warm_start_boozer_init_overrides[
            "newton_maxiter_override"
        ],
    )

    # Initialize Boozer surface with target parameters
    timings = {}
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
                effective_warm_start_boozer_least_squares_algorithm
            ),
            bfgs_tol_override=warm_start_boozer_init_overrides["bfgs_tol_override"],
            bfgs_maxiter_override=warm_start_boozer_init_overrides[
                "bfgs_maxiter_override"
            ],
            newton_tol_override=warm_start_boozer_init_overrides[
                "newton_tol_override"
            ],
            newton_maxiter_override=warm_start_boozer_init_overrides[
                "newton_maxiter_override"
            ],
            surface_dofs_override=warm_start_surface_dofs,
            iota_override=None if warm_start_state is None else warm_start_state["iota"],
            G_override=None if warm_start_state is None else warm_start_state["G"],
            on_stage=boozer_init_progress,
            timings_out=timings,
        )
    boozer_init_progress("completed")

    # ==============================================================================
    # SAVE INITIAL STATE
    # ==============================================================================
    initial_artifacts_start_s = _perf_counter_s()
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
    initial_max_curvature = np.max(banana_curve.kappa())
    _record_timing(
        timings,
        "initial_artifacts_s",
        initial_artifacts_start_s,
        _perf_counter_s(),
    )

    # ==============================================================================
    # DEFINE OBJECTIVE FUNCTION COMPONENTS
    # ==============================================================================
    objective_setup_start_s = _perf_counter_s()
    # Biot-Savart field calculation
    if use_jax:
        bs_obj = BiotSavartJAX(coils)
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
    CC_DIST = max(args.cc_dist, 0.05)  # Hardware minimum: 5cm coil-coil spacing
    CS_WEIGHT = args.cs_weight
    CS_DIST = max(args.cs_dist, 0.02)  # Hardware minimum: 2cm coil-surface clearance
    SURF_DIST_WEIGHT = args.surf_dist_weight
    SS_DIST = max(args.ss_dist, 0.04)  # Hardware minimum: 4cm surface-vessel clearance
    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = args.curvature_threshold

    # Individual objective terms
    iota = build_iota_objective(boozer_surface, iota_cls)
    curvelength = CurveLength(banana_curves[0])
    length_target = host_float(curvelength.J())

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

    # ==============================================================================
    # SNAPSHOT PRE-OPTIMIZATION STATE
    # ==============================================================================
    dofs, run_dict, static_config = snapshot_to_pytree(
        JF, boozer_surface, bs, num_tf_coils=num_tf_coils
    )
    outer_contract = resolve_single_stage_optimizer_contract(
        args.backend,
        args.optimizer_backend,
    )
    from simsopt.geo.optimizer_jax import TargetOptimizerContract

    use_target_lane = isinstance(outer_contract, TargetOptimizerContract)
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
    run_dict["x_prev"] = _single_stage_optimizer_dofs_array(dofs).copy()
    target_lane_profile = None
    adapter = SingleStageAdapter(
        run_dict=run_dict,
        boozer_surface=boozer_surface,
        JF=JF,
        bs=bs,
        objectives={
            "qs": JnonQSRatio,
            "boozer": JBoozerResidual,
            "iota_penalty": Jiota,
            "length": JCurveLength,
            "cc": JCurveCurve,
            "cs": JCurveSurface,
            "surf": JSurfSurf,
            "curvature": JCurvature,
        },
        diagnostics={
            "iota": iota,
            "banana_curve": banana_curve,
            "curvelength": curvelength,
        },
        log_path=OUT_DIR_ITER + "/log.txt",
        reevaluate_before_accept=use_target_lane,
        apply_coil_dofs=dof_setter,
        benchmark_mode=args.benchmark_mode,
    )

    # ==============================================================================
    # RUN OPTIMIZATION
    # ==============================================================================
    # Get convergence tolerances for current mpol
    ftol = ftol_by_mpol.get(mpol, 1e-5 if mpol < 8 else 1e-10)
    gtol = gtol_by_mpol.get(mpol, 1e-2 if mpol < 8 else 1e-7)
    phase1_iterations = None
    phase1_termination_message = None
    phase1_success = None
    main_phase_iterations = None
    target_lane_gradient_diagnosis = None
    target_lane_scaled_phase1_diagnosis = None
    target_lane_invalid_state_events = []
    jax_compile_diagnostics = None
    record_target_lane_invalid_state_events = (
        record_target_lane_invalid_state_events_enabled(args)
    )
    skip_outer_optimizer = bool(
        args.init_only
        or args.profile_target_lane_only
        or args.diagnose_target_lane_gradient
        or args.diagnose_target_lane_scaled_phase1
    )
    res = None

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
        if use_target_lane:
            print(
                "Preparing target-lane outer objective/runtime bundle "
                f"(method={outer_contract.method}, maxiter={MAXITER})..."
            )
        target_lane_trial_boozer_overrides = {
            "bfgs_tol": target_lane_boozer_bfgs_tol_record,
            "bfgs_maxiter": target_lane_boozer_bfgs_maxiter_record,
            "newton_tol": target_lane_boozer_newton_tol_record,
            "newton_maxiter": target_lane_boozer_newton_maxiter_record,
        }
        target_lane_trial_boozer_override_active = any(
            value is not None for value in target_lane_trial_boozer_overrides.values()
        )
        accepted_step_callback = None
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
            if use_target_lane:
                _record_timing(
                    timings,
                    "target_lane_bundle_setup_s",
                    outer_optimizer_start_s,
                    _perf_counter_s(),
                )
                outer_optimizer_run_start_s = _perf_counter_s()
                print("Target-lane outer objective/runtime bundle ready.")
            if args.diagnose_target_lane_scaled_phase1:
                if not use_target_lane:
                    raise RuntimeError(
                        "--diagnose-target-lane-scaled-phase1 requires the JAX "
                        "ondevice single-stage target lane."
                    )
                if args.initial_step_maxiter < 1 or args.initial_step_scale >= 1.0:
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
                        phase1_maxiter=min(MAXITER, args.initial_step_maxiter),
                        step_scale=args.initial_step_scale,
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
                phase1_iterations = target_lane_scaled_phase1_diagnosis["optimizer"][
                    "iterations"
                ]
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
                    status=target_lane_scaled_phase1_diagnosis["optimizer"]["status"],
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
                target_lane_gradient_diagnosis = build_target_lane_gradient_diagnosis(
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
                    total_diag["value"]["finite"] and total_diag["grad"]["all_finite"]
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
                phase1_dofs = dofs
                phase1_callback = accepted_step_callback
                phase1_fun = target_value_and_grad_objective if use_target_lane else adapter
                phase1_scalar_fun = target_scalar_objective if use_target_lane else None
                phase1_final_dofs = None
                remaining_maxiter = MAXITER
                if args.initial_step_maxiter > 0 and args.initial_step_scale < 1.0:
                    phase1_maxiter = min(MAXITER, args.initial_step_maxiter)
                    if phase1_maxiter > 0:
                        phase1_anchor_dofs = dofs
                        phase1_dofs = build_scaled_outer_phase_initial_dofs(
                            phase1_anchor_dofs,
                            use_target_lane=use_target_lane,
                        )
                        if use_target_lane:
                            phase1_dofs = build_target_lane_scaled_outer_phase_state(
                                phase1_anchor_dofs,
                                phase1_dofs,
                            )
                        if use_target_lane:
                            if phase1_fun is not None:
                                phase1_fun, phase1_callback = build_scaled_outer_problem(
                                    phase1_fun,
                                    accepted_step_callback,
                                    phase1_anchor_dofs,
                                    args.initial_step_scale,
                                    anchor_in_state=True,
                                )
                                phase1_scalar_fun = None
                            else:
                                (
                                    phase1_scalar_fun,
                                    phase1_callback,
                                ) = build_scaled_outer_scalar_problem(
                                    phase1_scalar_fun,
                                    accepted_step_callback,
                                    phase1_anchor_dofs,
                                    args.initial_step_scale,
                                    anchor_in_state=True,
                                )
                        else:
                            phase1_fun, phase1_callback = build_scaled_outer_problem(
                                phase1_fun,
                                accepted_step_callback,
                                phase1_anchor_dofs,
                                args.initial_step_scale,
                            )
                        print(
                            "Starting scaled initial outer phase "
                            f"(step_scale={args.initial_step_scale}, "
                            f"maxiter={phase1_maxiter})..."
                        )
                        phase1_failure_callback = (
                            resolve_target_lane_invalid_state_failure_callback(
                                target_lane_invalid_state_events,
                                phase="phase1",
                                use_target_lane=use_target_lane,
                                args=args,
                            )
                        )
                        phase1_start_s = _perf_counter_s()
                        with maybe_trace_single_stage_phase(
                            "single_stage.outer_optimizer_initial_phase",
                            enabled=jax_profile_enabled,
                        ):
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
                                failure_callback=phase1_failure_callback,
                            )
                        _record_timing(
                            timings,
                            "outer_optimizer_initial_phase_s",
                            phase1_start_s,
                            _perf_counter_s(),
                        )
                        phase1_iterations = int(phase1_res.nit)
                        phase1_termination_message = str(phase1_res.message)
                        phase1_success = bool(phase1_res.success)
                        print(phase1_res.message)
                        phase1_final_dofs = resolve_scaled_outer_phase_final_dofs(
                            phase1_anchor_dofs,
                            phase1_res.x,
                            args.initial_step_scale,
                            use_target_lane=use_target_lane,
                        )
                        if (
                            use_target_lane
                            and accepted_step_callback is None
                            and phase1_iterations > 0
                        ):
                            target_lane_phase1_sync_start_s = _perf_counter_s()
                            with maybe_trace_single_stage_phase(
                                "single_stage.target_lane_initial_phase_sync",
                                enabled=jax_profile_enabled,
                            ):
                                adapter.sync_accepted_step(phase1_final_dofs)
                            _record_timing(
                                timings,
                                "target_lane_initial_phase_sync_s",
                                target_lane_phase1_sync_start_s,
                                _perf_counter_s(),
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
                if run_main_optimizer:
                    main_failure_callback = (
                        resolve_target_lane_invalid_state_failure_callback(
                            target_lane_invalid_state_events,
                            phase="phase2",
                            use_target_lane=use_target_lane,
                            args=args,
                        )
                    )
                    main_optimizer_start_s = _perf_counter_s()
                    with maybe_trace_single_stage_phase(
                        "single_stage.outer_optimizer",
                        enabled=jax_profile_enabled,
                    ):
                        res = run_single_stage_optimizer(
                            target_value_and_grad_objective if use_target_lane else adapter,
                            dofs,
                            callback=accepted_step_callback,
                            contract=outer_contract,
                            maxiter=remaining_maxiter,
                            ftol=ftol,
                            gtol=gtol,
                            maxcor=args.maxcor,
                            outer_maxls=args.outer_maxls,
                            scalar_fun=target_scalar_objective,
                            target_lane_initial_step_size=(
                                args.target_lane_outer_initial_step_size
                                if use_target_lane
                                else None
                            ),
                            failure_callback=main_failure_callback,
                        )
                    _record_timing(
                        timings,
                        "outer_optimizer_main_s",
                        main_optimizer_start_s,
                        _perf_counter_s(),
                    )
                    termination_message = str(res.message)
                    optimizer_success = bool(res.success)
                    main_phase_iterations = int(res.nit)
                    res_nit = (phase1_iterations or 0) + int(res.nit)
                    print(res.message)
                    if (
                        use_target_lane
                        and args.benchmark_mode
                        and accepted_step_callback is None
                        and int(res.nit) > 0
                    ):
                        target_lane_sync_start_s = _perf_counter_s()
                        with maybe_trace_single_stage_phase(
                            "single_stage.target_lane_final_sync",
                            enabled=jax_profile_enabled,
                        ):
                            adapter.sync_accepted_step(res.x)
                        _record_timing(
                            timings,
                            "target_lane_final_sync_s",
                            target_lane_sync_start_s,
                            _perf_counter_s(),
                        )
                    if phase1_termination_message is not None:
                        termination_message = (
                            f"phase1={phase1_termination_message}; "
                            f"phase2={termination_message}"
                        )
                else:
                    res = phase1_res
                    if phase1_final_dofs is not None:
                        res.x = phase1_final_dofs
                    res_nit = phase1_iterations or 0
                    termination_message = phase1_termination_message or "phase1_only"
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

        # The target lane keeps the mutable graph synchronized through the
        # accepted-step seam, so re-restoring the same accepted state here only
        # dirties BoozerSurfaceJAX and forces an extra no-op solve during final
        # diagnostics. The CPU/reference lane still needs the explicit restore.
        if not skip_outer_optimizer and not use_target_lane:
            restore_from_pytree(
                JF,
                boozer_surface,
                run_dict,
                coil_dofs=res.x,
                apply_coil_dofs=dof_setter,
            )
        if (not args.benchmark_mode) and should_force_strict_target_lane_final_sync(
            use_target_lane=use_target_lane,
            res_nit=res_nit,
            accepted_step_callback=accepted_step_callback,
            trial_boozer_override_active=target_lane_trial_boozer_override_active,
        ):
            target_lane_sync_start_s = _perf_counter_s()
            with maybe_trace_single_stage_phase(
                "single_stage.target_lane_final_sync",
                enabled=jax_profile_enabled,
            ):
                adapter.sync_accepted_step(res.x)
            _record_timing(
                timings,
                "target_lane_final_sync_s",
                target_lane_sync_start_s,
                _perf_counter_s(),
            )

        # ==============================================================================
        # SAVE OPTIMIZED STATE
        # ==============================================================================
        if not skip_outer_optimizer:
            final_artifacts_start_s = _perf_counter_s()
            if write_restart_artifacts:
                bs_diag.save(OUT_DIR_ITER + "/biot_savart_opt.json")

                boozer_surface.surface.save(OUT_DIR_ITER + "/surf_opt.json")
                if write_full_artifacts:
                    # Save optimized coil configurations
                    curves_to_vtk(curves, OUT_DIR_ITER + "/curves_opt", close=True)

                    # Save optimized surface with magnetic field normal component data
                    bs_diag.set_points(boozer_surface.surface.gamma().reshape((-1, 3)))
                    unitn = boozer_surface.surface.unitnormal()
                    pointData = {
                        "B_N/B": np.sum(
                            bs_diag.B().reshape(unitn.shape) * unitn, axis=2
                        )[:, :, None]
                        / np.sqrt(
                            np.sum(bs_diag.B().reshape(unitn.shape) ** 2, axis=2)
                        )[:, :, None]
                    }
                    boozer_surface.surface.to_vtk(
                        OUT_DIR_ITER + "/surf_opt",
                        extra_data=pointData,
                    )

            final_volume = host_float(boozer_surface.surface.volume())
            final_iota = resolve_single_stage_iota_metric(
                boozer_surface,
                iota_cls,
                benchmark_mode=args.benchmark_mode,
            )
            final_max_curvature = np.max(banana_curve.kappa())
            print(f"Volume: {final_volume}")
            print(f"Iota: {final_iota}")
            print(f"Max Curvature: {final_max_curvature}")

            # Generate final diagnostic plots
            if write_full_artifacts:
                fieldError = normPlot(
                    boozer_surface.surface,
                    bs_diag,
                    OUT_DIR_ITER + "/NormPlotOptimized",
                )
                cross_section_plot(
                    surf_coils,
                    boozer_surface.surface,
                    banana_curve,
                    OUT_DIR_ITER + "/CrossSectionOptimized",
                    hbt,
                    VV,
                )
            else:
                fieldError, _, _, _, _, _ = norm_field_summary(
                    boozer_surface.surface,
                    bs_diag,
                )
            _record_timing(
                timings,
                "final_artifacts_s",
                final_artifacts_start_s,
                _perf_counter_s(),
            )

    final_target_lane_outer_objective_config = None
    if use_target_lane and not skip_outer_optimizer:
        final_target_lane_outer_objective_config = build_target_lane_outer_objective_config(
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
    final_penalty_metrics = resolve_single_stage_final_penalty_metrics(
        use_target_lane=use_target_lane,
        benchmark_mode=bool(args.benchmark_mode),
        skip_outer_optimizer=skip_outer_optimizer,
        boozer_surface=boozer_surface,
        bs=bs,
        iota_target=iota_target,
        coil_dofs=bs.x.copy() if res is None else res.x,
        outer_objective_config=final_target_lane_outer_objective_config,
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
        init_only=args.init_only,
        termination_message=termination_message,
        optimizer_success=optimizer_success,
    )
    if use_target_lane and not skip_outer_optimizer:
        final_max_curvature = float(final_penalty_metrics["max_curvature"])
    final_self_intersecting = update_self_intersection_status(
        run_dict, boozer_surface.surface
    )
    final_self_intersection_check_available = run_dict[
        "self_intersection_check_available"
    ]
    final_coil_length = float(final_penalty_metrics["coil_length"])
    final_hardware_metrics_start_s = _perf_counter_s()
    with maybe_trace_single_stage_phase(
        "single_stage.final_hardware_metrics",
        enabled=jax_profile_enabled,
    ):
        final_curve_curve_min_dist = final_penalty_metrics["curve_curve_min_dist"]
        final_curve_surface_min_dist = final_penalty_metrics["curve_surface_min_dist"]
        final_surface_vessel_min_dist = final_penalty_metrics[
            "surface_vessel_min_dist"
        ]
        final_hardware_status = final_penalty_metrics["hardware_status"]
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
            stage2_source=args.stage2_source,
            stage2_results=stage2_results,
            warm_start_run_dir=args.warm_start_run_dir,
            warm_start_state=warm_start_state,
            R0=R0,
            s=s,
            order=order,
            CONSTRAINT_WEIGHT=CONSTRAINT_WEIGHT,
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
            optimizer_backend_record=optimizer_backend_record,
            boozer_optimizer_backend_record=boozer_optimizer_backend_record,
            boozer_least_squares_algorithm_record=boozer_least_squares_algorithm_record,
            outer_optimizer_method=outer_contract.method,
            target_lane_sync_record=target_lane_sync_record,
            requested_experimental_target_lane_vg=requested_experimental_target_lane_vg,
            use_target_lane_vg=use_target_lane_vg,
            target_lane_boozer_bfgs_tol_record=target_lane_boozer_bfgs_tol_record,
            target_lane_boozer_bfgs_maxiter_record=target_lane_boozer_bfgs_maxiter_record,
            target_lane_boozer_newton_tol_record=target_lane_boozer_newton_tol_record,
            target_lane_boozer_newton_maxiter_record=target_lane_boozer_newton_maxiter_record,
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
        "STAGE2_SOURCE": args.stage2_source,
        "STAGE2_BS_PATH": stage2_bs_path,
        "STAGE2_RESULTS_PATH": stage2_results_path,
        "STAGE2_SEED_MAJOR_RADIUS": R0,
        "STAGE2_SEED_TOROIDAL_FLUX": s,
        "STAGE2_SEED_BANANA_SURF_RADIUS": float(stage2_results["banana_surf_radius"]),
        "STAGE2_SEED_ORDER": order,
        "mpol": mpol,
        "ntor": ntor,
        "nphi": nphi,
        "ntheta": ntheta,
        "boozer_stage": stage,
        "CONSTRAINT_WEIGHT": CONSTRAINT_WEIGHT,
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
        "backend": args.backend,
        "optimizer_backend": optimizer_backend_record,
        "boozer_optimizer_backend": boozer_optimizer_backend_record,
        "boozer_least_squares_algorithm": boozer_least_squares_algorithm_record,
        "boozer_optimizer_method": boozer_surface.res.get("optimizer_method"),
        "outer_optimizer_method": outer_contract.method,
        "target_lane_accepted_step_sync": target_lane_sync_record,
        "experimental_target_lane_value_and_grad": requested_experimental_target_lane_vg,
        "target_lane_value_and_grad": use_target_lane_vg,
        "benchmark_mode": bool(args.benchmark_mode),
        "disable_target_lane_success_filter": bool(
            args.disable_target_lane_success_filter
        ),
        "profile_target_lane": bool(args.profile_target_lane),
        "profile_target_lane_only": bool(args.profile_target_lane_only),
        "profile_target_lane_batch_size": int(args.profile_target_lane_batch_size),
        "diagnose_target_lane_gradient": bool(args.diagnose_target_lane_gradient),
        "diagnose_target_lane_scaled_phase1": bool(
            args.diagnose_target_lane_scaled_phase1
        ),
        "record_target_lane_invalid_state_events": bool(
            record_target_lane_invalid_state_events
        ),
        "minimal_artifacts": bool(args.minimal_artifacts),
        "init_only": args.init_only,
        "max_iterations": MAXITER,
        "maxcor": args.maxcor,
        "outer_maxls": args.outer_maxls,
        "target_lane_outer_initial_step_size": args.target_lane_outer_initial_step_size,
        "initial_step_scale": args.initial_step_scale,
        "initial_step_maxiter": args.initial_step_maxiter,
        "target_lane_boozer_bfgs_tol": target_lane_boozer_bfgs_tol_record,
        "target_lane_boozer_bfgs_maxiter": target_lane_boozer_bfgs_maxiter_record,
        "target_lane_boozer_newton_tol": target_lane_boozer_newton_tol_record,
        "target_lane_boozer_newton_maxiter": target_lane_boozer_newton_maxiter_record,
        "INITIAL_PHASE_ITERATIONS": phase1_iterations,
        "INITIAL_PHASE_TERMINATION_MESSAGE": phase1_termination_message,
        "INITIAL_PHASE_SUCCESS": phase1_success,
        "iterations": res_nit,
        "TERMINATION_MESSAGE": termination_message,
        "OPTIMIZER_SUCCESS": optimizer_success,
        "OPTIMIZER_STATUS": None
        if skip_outer_optimizer
        else int(getattr(res, "status", -1)),
        "OPTIMIZER_NFEV": None
        if skip_outer_optimizer
        else int(getattr(res, "nfev", 0)),
        "OPTIMIZER_NJEV": None
        if skip_outer_optimizer
        else int(getattr(res, "njev", 0)),
        "OPTIMIZER_LS_STATUS": None
        if skip_outer_optimizer or getattr(res, "ls_status", None) is None
        else int(res.ls_status),
        "TARGET_VOLUME": float(vol_target),
        "TARGET_IOTA": float(iota_target),
        "FINAL_VOLUME": float(final_volume),
        "FINAL_IOTA": float(final_iota),
        "FINAL_G": final_penalty_metrics["final_G"],
        "FINAL_NON_QS": final_penalty_metrics["final_non_qs"],
        "FINAL_BOOZER_RESIDUAL": final_penalty_metrics["final_boozer_residual"],
        "FINAL_IOTA_PENALTY": final_penalty_metrics["final_iota_penalty"],
        "FINAL_LENGTH_PENALTY": final_penalty_metrics["final_length_penalty"],
        "FINAL_CURVE_CURVE_PENALTY": final_penalty_metrics[
            "final_curve_curve_penalty"
        ],
        "FINAL_CURVE_SURFACE_PENALTY": final_penalty_metrics[
            "final_curve_surface_penalty"
        ],
        "FINAL_SURFACE_VESSEL_PENALTY": final_penalty_metrics[
            "final_surface_vessel_penalty"
        ],
        "FINAL_CURVATURE_PENALTY": final_penalty_metrics[
            "final_curvature_penalty"
        ],
        "FIELD_ERROR": float(fieldError),
        "SELF_INTERSECTING": final_self_intersecting,
        "SELF_INTERSECTION_CHECK_AVAILABLE": final_self_intersection_check_available,
        "MAX_CURVATURE": float(final_max_curvature),
        "COIL_LENGTH": final_coil_length,
        "CURVE_CURVE_MIN_DIST": final_curve_curve_min_dist,
        "CURVE_SURFACE_MIN_DIST": final_curve_surface_min_dist,
        "SURFACE_VESSEL_MIN_DIST": final_surface_vessel_min_dist,
        "HARDWARE_CONSTRAINTS_OK": final_hardware_status["success"],
        "HARDWARE_CONSTRAINT_VIOLATIONS": final_hardware_status["violations"],
        "INITIAL_VOLUME": float(initial_volume),
        "INITIAL_IOTA": float(initial_iota),
        "INITIAL_FIELD_ERROR": float(initial_field_error),
        "INITIAL_MAX_CURVATURE": float(initial_max_curvature),
        "num_tf_coils": static_config["num_tf_coils"],
        "tf_currents": static_config["tf_currents"],
    }
    optimizer_diag = extract_optimizer_diagnostics(
        None if skip_outer_optimizer else res,
        ran_optimizer=not skip_outer_optimizer,
        termination_message=termination_message,
    )
    initial_objective = float(run_dict["initial_objective"])
    final_objective = optimizer_diag["fun"]
    objective_decrease = (
        None
        if final_objective is None
        else initial_objective - float(final_objective)
    )
    results["OPTIMIZER_FUN"] = optimizer_diag["fun"]
    results["OPTIMIZER_FUN_FINITE"] = optimizer_diag["fun_finite"]
    results["OPTIMIZER_JAC_FINITE"] = optimizer_diag["jac_finite"]
    results["OPTIMIZER_JAC_INF_NORM"] = optimizer_diag["jac_inf_norm"]
    results["OPTIMIZER_X_FINITE"] = optimizer_diag["x_finite"]
    results["OPTIMIZER_INVALID_STATE"] = optimizer_diag["invalid_state"]
    results["INITIAL_OBJECTIVE"] = initial_objective
    results["FINAL_OBJECTIVE"] = final_objective
    results["OBJECTIVE_DECREASE"] = objective_decrease
    if use_target_lane and (
        target_lane_invalid_state_events or optimizer_diag["invalid_state"]
    ):
        results["TARGET_LANE_INVALID_STATE_DIAGNOSIS"] = {
            "event_count": int(len(target_lane_invalid_state_events)),
            "event_recording_enabled": bool(record_target_lane_invalid_state_events),
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
            "note": (
                None
                if target_lane_invalid_state_events
                else (
                    "optimizer reported an invalid target-lane state without a "
                    "recorded rejected trial payload because rejected-step event "
                    "recording was disabled"
                    if not record_target_lane_invalid_state_events
                    else "optimizer reported an invalid target-lane state without "
                    "a recorded rejected trial payload"
                )
            ),
        }
    timings["script_total_s"] = _elapsed_s(run_wall_start_s, _perf_counter_s())
    results["TIMINGS"] = timings
    if target_lane_profile is not None:
        results["TARGET_LANE_PROFILE"] = target_lane_profile
    if jax_compile_diagnostics is not None:
        results["JAX_COMPILE_DIAGNOSTICS"] = jax_compile_diagnostics
    if target_lane_gradient_diagnosis is not None:
        results["TARGET_LANE_GRADIENT_DIAGNOSIS"] = target_lane_gradient_diagnosis
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
    write_json_file(os.path.join(OUT_DIR_ITER, "results.json"), results)
