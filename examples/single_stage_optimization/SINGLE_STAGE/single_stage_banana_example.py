from __future__ import annotations

import argparse
import copy
import hashlib
import os
import io
import json
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import astuple, dataclass, fields, replace
from pathlib import Path, PurePath
from types import SimpleNamespace
import numpy as np
from scipy.optimize import minimize

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if EXAMPLE_ROOT not in sys.path:
    sys.path.insert(0, EXAMPLE_ROOT)

from import_provenance import configure_local_simsopt_imports

EXAMPLE_ROOT, SIMSOPT_ROOT, SRC_ROOT = configure_local_simsopt_imports(__file__)

# SIMSOPT imports
from simsopt._core.optimizable import Optimizable
from simsopt.geo import (
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    BoozerSurface,
    CurveCWSFourierCPP,
    curves_to_vtk,
    CurveLength,
    LpCurveCurvature,
)
from simsopt.geo.surfaceobjectives import (
    Volume,
    BoozerResidual,
    Iotas,
    NonQuasiSymmetricRatio,
    SurfaceSurfaceDistance,
)
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
from simsopt.field import (
    BiotSavart,
)
from simsopt.objectives import QuadraticPenalty
from simsopt._core.optimizable import load
from simsopt._core.derivative import Derivative

from alm_utils import (
    ALMSettings,
    alm_result_diagnostics_fields,
    augmented_inequality_objective,  # noqa: F401 - legacy module-level export.
    minimize_alm,
    validate_alm_cli_args,
)
from plotting_utils import norm_field_plot, cross_section_plot
from topology_scorer import (
    safe_score_topology as _safe_score_topology_impl,
    topology_transport_diagnostics_not_evaluated as _topology_transport_diagnostics_not_evaluated,
)
from workflow_helpers import (
    Stage2SeedSpec,
    format_database_stage2_seed_dir,
    format_database_stage2_seed_dir_without_init_current,
    format_legacy_database_stage2_seed_dir,
    format_legacy_local_stage2_seed_dir,
    format_local_stage2_seed_dir,
    format_local_stage2_seed_dir_without_init_current,
    format_local_stage2_seed_dir_without_tf,
)
from workflow_runner_common import load_stage2_artifact_results
from banana_opt.artifact_contracts import (
    STAGE2_SEED_CONTRACT_HASH_KEY,
    upgrade_legacy_stage2_artifact_results,
)
from banana_opt.boozer_residuals import (  # noqa: F401 - re-exported for importlib-loaded tests
    BoozerResidualExact,
    RefinedBoozerResidual,
)
from banana_opt.constraint_contract import (
    apply_offspec_engineering_override_reason,
    build_constraint_metadata,
    resolve_constraint_contract_from_wire_names as _resolve_constraint_contract_from_wire_names_impl,
)
from banana_opt.coil_order_upgrade import upgrade_loaded_seed_biot_savart_order
from banana_opt.basin_hopping import run_basin_hopping, telemetry_values as basin_telemetry_values
from banana_opt.basin_hopping import (  # noqa: F401 - re-exported for importlib-loaded tests
    _normalized_step_rms as basin_normalized_step_rms,
)
from banana_opt.frontier_constraints import (
    apply_frontier_search_contract_penalties as _apply_frontier_search_contract_penalties_impl,
    annotate_frontier_search_eval as _annotate_frontier_search_eval_impl,
    evaluate_frontier_hard_invalidation as _evaluate_frontier_hard_invalidation_impl,
    evaluate_frontier_hardware_search_contract as _evaluate_frontier_hardware_search_contract_impl,
    evaluate_frontier_hardware_search_penalty as _evaluate_frontier_hardware_search_penalty_impl,
    evaluate_frontier_topology_search_contract as _evaluate_frontier_topology_search_contract_impl,
    evaluate_frontier_topology_search_penalty as _evaluate_frontier_topology_search_penalty_impl,
    evaluate_frontier_trust_penalty as _evaluate_frontier_trust_penalty_impl,
    evaluate_frontier_trust_status as _evaluate_frontier_trust_status_impl,
    hardware_violation_ratios as _hardware_violation_ratios,
)
from banana_opt.frontier_conditioning import (
    FRONTIER_CONDITIONING_SCHEMA_VERSION,
    build_frontier_conditioning_gate,
    build_frontier_conditioning_report,
)
from banana_opt.frontier_solver_checkpoint import (
    load_solver_checkpoint,
    restore_incumbent_from_solver_checkpoint,
    restore_optional_incumbent,
    solver_checkpoint_path,
    write_solver_checkpoint,
    build_solver_checkpoint_payload,
)
from banana_opt.banana_current_replay import (
    BANANA_CURRENT_REJECTED_TRIAL_REPLAY_FILENAME,
    BANANA_CURRENT_REJECTED_TRIAL_REPLAY_SCHEMA_VERSION,
    BANANA_CURRENT_REPLAY_CONTEXT_FILENAME,
    banana_current_rejected_trial_replay_path,
    build_banana_current_replay_context_state,
    build_replayed_candidate_x,
    load_banana_current_replay_context,
    record_banana_current_replay_context_snapshot,
    restore_banana_current_replay_incumbent,
    validate_banana_current_replay_coordinate_contract,
    validate_banana_current_replay_context_contract,
    set_banana_current_replay_context_contract,
    write_banana_current_replay_context_artifact,
)
from banana_opt.current_contracts import (
    DEFAULT_FINITE_CURRENT_MODE,
    infer_uniform_coil_current_A as _infer_uniform_coil_current_A,
    resolve_penalty_traversal_forbidden_box_bounds,
    resolve_plasma_current_settings_for_num_surfaces as _resolve_plasma_current_settings_for_num_surfaces_impl,
)
from banana_opt.hardware_contracts import (
    BANANA_CURRENT_HARD_LIMIT_A,
    ACCEPT_OFFSPEC_R0_SEED_ENV,
    ACCEPT_OFFSPEC_R0_SEED_HELP,
    BANANA_WINDING_MINOR_RADIUS_M,
    COIL_COIL_MIN_DIST_M,
    COIL_LENGTH_HARD_LIMIT_M,
    COIL_LENGTH_TARGET_M,
    COIL_PLASMA_MIN_DIST_M,
    MAX_CURVATURE_INV_M,
    PLASMA_VESSEL_MIN_DIST_M,
    TF_CURRENT_HARD_LIMIT_A,
    VACUUM_VESSEL_MAJOR_RADIUS_M,
    env_flag,
    is_major_radius_offspec,
    validate_major_radius,
)
from banana_opt.hardware_constraint_schema import (
    build_hardware_constraint_artifact_payload_fields,
    hardware_constraint_alm_names,
)
from banana_opt.incumbents import (
    restore_single_stage_incumbent_state,
    snapshot_single_stage_incumbent_state,
)
from banana_opt.lbfgsb_defaults import DEFAULT_LBFGSB_MAXCOR
from banana_opt.reference_surfaces import build_banana_reference_surfaces
from banana_opt.single_stage_phase1 import (  # noqa: F401 — re-exported for test access via importlib
    DEFAULT_PHASE1_CONFIG,
    Phase1Config,
    _FRONTIER_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
    _FRONTIER_FEASIBLE_START_PHASE1_SCALE,
    _PENALTY_FEASIBLE_START_LOCAL_MAXITER,
    _PENALTY_FEASIBLE_START_LOCAL_MAX_ATTEMPTS,
    _PENALTY_FEASIBLE_START_LOCAL_RADIUS_SHRINK,
    _PENALTY_FEASIBLE_START_LOCAL_RELATIVE_RADIUS,
    _PENALTY_FEASIBLE_START_MIN_ACCEPTED_STEP_RMS,
    _PENALTY_FEASIBLE_START_PHASE2_RADIUS_SCALE,
    _PENALTY_FEASIBLE_START_REJECT_RADIUS_SHRINK,
    _PENALTY_FEASIBLE_START_SAFE_STEP_RMS_LIMIT,
    _SEED_REGIME_BRIDGE_ONLY as _PHASE1_SEED_REGIME_BRIDGE_ONLY,
    _SEED_REGIME_PRESERVE_FIRST as _PHASE1_SEED_REGIME_PRESERVE_FIRST,
    _SEED_REGIME_REPAIR_FIRST as _PHASE1_SEED_REGIME_REPAIR_FIRST,
    build_penalty_phase2_bounds,
    build_phase1_config,
    resolve_initial_step_phase_maxiter,
    resolve_penalty_phase1_settings,
    run_penalty_phase1,
)
from banana_opt.stage2_single_stage_handoff import (
    build_equilibrium_path as _build_equilibrium_path_impl,
    compute_tf_G0 as _compute_tf_G0_impl,
    initialize_boozer_surface as _initialize_boozer_surface_impl,
    load_warm_start_boozer_seed,
    partition_loaded_stage2_coils as _partition_loaded_stage2_coils_impl,
    resolve_warm_start_boozer_surface_path,
    resolve_single_stage_banana_surf_radius as _resolve_single_stage_banana_surf_radius_impl,
    resolve_stage2_finite_current_mode as _resolve_stage2_finite_current_mode_impl,
    resolve_stage2_num_tf_coils as _resolve_stage2_num_tf_coils_impl,
    resolve_stage2_tf_current_A as _resolve_stage2_tf_current_A_impl,
    validate_loaded_stage2_coils_partition as _validate_loaded_stage2_coils_partition_impl,
    validate_stage2_seed_contract as _validate_stage2_seed_contract_impl,
)
from banana_opt.single_stage_geometry import (
    build_scipy_bounds,
    build_surface_configs as _build_surface_configs_impl,
    build_surface_search_gate,
    build_surface_search_weights,
    broken_topology_gate_status,
    collect_surface_run_metadata,
    disabled_topology_gate_status,
    evaluate_single_stage_hardware_constraints as _evaluate_single_stage_hardware_constraints,
    evaluate_single_stage_hardware_snapshot,
    evaluate_single_stage_search_hardware_snapshot,
    evaluate_surface_stack,
    evaluate_topology_gate as _evaluate_topology_gate_impl,
    restore_surface_states,
    save_surface_artifacts,
    snapshot_surface_states,
    solve_surface_stack_at_dofs,
    compute_single_stage_surface_vessel_min_dist as _compute_single_stage_surface_vessel_min_dist,
    topology_gate_deficit as _topology_gate_deficit,
    topology_gate_state as _topology_gate_state,
)
from banana_opt.single_stage_geometry import (  # noqa: F401 - re-exported for importlib-loaded tests
    build_local_relative_bounds,
    build_scaled_local_outer_bounds,
    build_scaled_outer_bounds,
    build_scaled_outer_problem,
    topology_gate_rejection_increment,
)
from banana_opt.single_stage_constraints import (
    smooth_max_curvature_signed_constraint as _smooth_max_curvature_signed_constraint,
    smooth_min_curve_curve_signed_constraint as _smooth_min_curve_curve_signed_constraint,
    smooth_min_curve_surface_signed_constraint as _smooth_min_curve_surface_signed_constraint,
    smooth_min_surface_surface_signed_constraint as _smooth_min_surface_surface_signed_constraint,
)
from banana_opt.single_stage_search_policy import (
    CurvatureTraversalPolicy,
    HardwareSearchPolicy,
    SearchContext,
    decide_curvature_traversal,
    hardware_rejection_increment,
)
from banana_opt.single_stage_objectives import (
    apply_frontier_scalarization_override as _apply_frontier_scalarization_override_impl,
    average_surface_objectives as _average_surface_objectives_impl,
    build_total_objective as _build_total_objective_impl,
    evaluate_base_objective as _evaluate_base_objective_impl,
    evaluate_total_objective as _evaluate_total_objective_impl,
    evaluate_alm_objective as _evaluate_alm_objective_impl,
)
from banana_opt.single_stage_banana_current_mode import (
    BANANA_CURRENT_COORDINATE_SCALING_NONE,
    BANANA_CURRENT_COORDINATE_SCALING_SEED_RELATIVE,
    BANANA_CURRENT_MODE_INDEPENDENT,
    BANANA_CURRENT_MODE_SHARED,
    SingleStageBananaCurrentState,
    apply_single_stage_penalty_banana_current_bounds,
    build_single_stage_banana_current_state,
    build_single_stage_banana_current_payload_fields,
    resolve_single_stage_banana_current_state as _resolve_single_stage_banana_current_state_impl,
    resolve_banana_current_coordinate_spec,
)
from banana_opt.frontier_scalarization import (
    FRONTIER_REFERENCE_MODE_ACHIEVEMENT,
    FRONTIER_REFERENCE_MODE_EPSILON,
    FRONTIER_REFERENCE_MODE_REFERENCE_POINTS,
)
from banana_opt.surface_mode_contracts import (
    DEFAULT_INNER_SURFACE_RATIO,
    EXPERIMENTAL_MULTISURFACE,
    PUBLISHED_MULTISURFACE,
    SINGLE_SURFACE,
    SURFACE_MODE_CHOICES,
    SurfaceModeContract,
    build_surface_mode_contract as _build_surface_mode_contract_impl,
    build_surface_mode_metadata as _build_surface_mode_metadata_impl,
    resolve_surface_mode_inner_surface_ratio,
    surface_mode_supports_alm,
    surface_mode_supports_boozer_stage_refinement,
    surface_mode_supports_topology_gate,
    validate_surface_mode_runtime_support,
)
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
DATABASE_EQUILIBRIA_DIR = os.path.join(REPO_ROOT, "DATABASE", "EQUILIBRIA")
DEFAULT_EQUILIBRIA_DIR = DATABASE_EQUILIBRIA_DIR if os.path.isdir(DATABASE_EQUILIBRIA_DIR) else os.path.join(EXAMPLE_ROOT, "equilibria")
DEFAULT_LOCAL_STAGE2_ROOT = os.path.join(EXAMPLE_ROOT, "STAGE_2")
DEFAULT_DATABASE_STAGE2_ROOT = os.path.join(REPO_ROOT, "DATABASE", "COIL_OPTIMIZATION", "outputs")
DEFAULT_SINGLE_STAGE_OUTPUT_ROOT = os.path.join(SCRIPT_DIR, "outputs")
LEGACY_STAGE2_BANANA_WINDING_MINOR_RADIUS_M = 0.22
DEFAULT_HARDWARE_SEARCH_MODE = "hard"
DEFAULT_HARDWARE_SEARCH_SOFT_ITERATIONS = 0
DEFAULT_CURVATURE_TRAVERSAL_BAND = 0.0
DEFAULT_CURVATURE_TRAVERSAL_EVAL_BUDGET = 0
_DEFAULT_SINGLE_STAGE_SEED_REGIME = "auto"
_SINGLE_STAGE_SEED_REGIME_AUTO = "auto"
# Derive from single_stage_phase1 to keep one canonical source of truth for string values.
_SINGLE_STAGE_SEED_REGIME_PRESERVE_FIRST = _PHASE1_SEED_REGIME_PRESERVE_FIRST
_SINGLE_STAGE_SEED_REGIME_REPAIR_FIRST = _PHASE1_SEED_REGIME_REPAIR_FIRST
_SINGLE_STAGE_SEED_REGIME_BRIDGE_ONLY = _PHASE1_SEED_REGIME_BRIDGE_ONLY
_SINGLE_STAGE_SEED_REGIME_GLOBAL_SEARCH = "global_search"
SINGLE_STAGE_THRESHOLDED_PHYSICS_CONSTRAINT_NAMES = (
    "qs_error",
    "boozer_residual",
    "iota_penalty",
    "length_penalty",
)
DEFAULT_STAGE2_SEEDS_BY_PLASMA = {
    "wout_nfp22ginsburg_000_014417_iota15.nc": {
        "major_radius": VACUUM_VESSEL_MAJOR_RADIUS_M,
        "toroidal_flux": 0.24,
        "length_weight": 0.0005,
        "cc_weight": 100.0,
        "cc_threshold": COIL_COIL_MIN_DIST_M,
        "curvature_weight": 0.0001,
        "curvature_threshold": MAX_CURVATURE_INV_M,
        "banana_surf_radius": BANANA_WINDING_MINOR_RADIUS_M,
        "tf_current_A": TF_CURRENT_HARD_LIMIT_A,
        "order": 2,
        "banana_init_current_A": 1.0e4,
    },
    "wout_nfp22ginsburg_000_002084_iota20.nc": {
        "major_radius": VACUUM_VESSEL_MAJOR_RADIUS_M,
        "toroidal_flux": 0.24,
        "length_weight": 0.0005,
        "cc_weight": 100.0,
        "cc_threshold": COIL_COIL_MIN_DIST_M,
        "curvature_weight": 0.0001,
        "curvature_threshold": MAX_CURVATURE_INV_M,
        "banana_surf_radius": BANANA_WINDING_MINOR_RADIUS_M,
        "tf_current_A": TF_CURRENT_HARD_LIMIT_A,
        "order": 2,
        "banana_init_current_A": 1.0e4,
    },
}
def add_confinement_surrogate_args(parser):
    parser.add_argument(
        "--confinement-objective-weight",
        type=float,
        default=float(os.environ.get("CONFINEMENT_OBJECTIVE_WEIGHT", "0.0")),
        help="Checkpoint-ranking weight for the tail-sensitive confinement surrogate (0 = disabled).",
    )
    parser.add_argument(
        "--confinement-surrogate-worst-k",
        type=int,
        default=int(os.environ.get("CONFINEMENT_SURROGATE_WORST_K", "3")),
        help="Number of worst field lines emphasized by the confinement surrogate (default 3).",
    )
    parser.add_argument(
        "--confinement-surrogate-early-threshold",
        type=float,
        default=float(os.environ.get("CONFINEMENT_SURROGATE_EARLY_THRESHOLD", "0.2")),
        help="Normalized exit-time threshold below which lines count as early exits (default 0.2).",
    )
    parser.add_argument(
        "--confinement-surrogate-mean-weight",
        type=float,
        default=float(os.environ.get("CONFINEMENT_SURROGATE_MEAN_WEIGHT", "0.2")),
        help="Weight on mean line loss in the checkpoint confinement surrogate (default 0.2).",
    )
    parser.add_argument(
        "--confinement-surrogate-worst-weight",
        type=float,
        default=float(os.environ.get("CONFINEMENT_SURROGATE_WORST_WEIGHT", "0.6")),
        help="Weight on worst-k line loss in the checkpoint confinement surrogate (default 0.6).",
    )
    parser.add_argument(
        "--confinement-surrogate-early-weight",
        type=float,
        default=float(os.environ.get("CONFINEMENT_SURROGATE_EARLY_WEIGHT", "0.2")),
        help="Weight on early-exit fraction in the checkpoint confinement surrogate (default 0.2).",
    )


def _resolve_unique_stage2_match(patterns, note):
    from glob import glob as _glob

    matches = []
    for pattern in patterns:
        matches.extend(_glob(pattern))
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        print(f"Note: found {note} at {os.path.dirname(unique_matches[0])}")
        return unique_matches[0]
    if len(unique_matches) > 1:
        match_dirs = "\n".join(f"  - {os.path.dirname(match)}" for match in unique_matches)
        raise FileNotFoundError(
            "Multiple Stage 2 outputs match the requested seed specification. "
            "Pass --stage2-bs-path explicitly to choose one.\n"
            f"Matches:\n{match_dirs}"
        )
    return None


def _stage2_outputs_parent(root, plasma_surf_filename):
    return os.path.join(root, f"outputs-{plasma_surf_filename}")


def _wataru_stage2_patterns(parent, seed_dir, *, include_constraint_variants):
    suffixes = ["-FCM=wataru_proxy_field-*"]
    if include_constraint_variants:
        suffixes = [
            "-FCM=wataru_proxy_field-*-CM=penalty",
            "-FCM=wataru_proxy_field-*-CM=penalty-BH=*",
            "-FCM=wataru_proxy_field-*-CM=alm-*",
            "-FCM=wataru_proxy_field-*-CM=alm-*-BH=*",
        ]
    return [
        os.path.join(parent, seed_dir + suffix, "biot_savart_opt.json")
        for suffix in suffixes
    ]


def _stage2_seed_spec_from_args(args, *, banana_surf_radius: float | None = None):
    return Stage2SeedSpec(
        plasma_surf_filename=args.plasma_surf_filename,
        major_radius=args.stage2_seed_major_radius,
        toroidal_flux=args.stage2_seed_toroidal_flux,
        length_weight=args.stage2_seed_length_weight,
        cc_weight=args.stage2_seed_cc_weight,
        cc_threshold=args.stage2_seed_cc_threshold,
        curvature_weight=args.stage2_seed_curvature_weight,
        curvature_threshold=args.stage2_seed_curvature_threshold,
        banana_surf_radius=(
            args.stage2_seed_banana_surf_radius
            if banana_surf_radius is None
            else banana_surf_radius
        ),
        tf_current_A=args.stage2_seed_tf_current_A,
        order=args.stage2_seed_order,
        banana_init_current_A=args.stage2_seed_banana_init_current_A,
    )


def _iter_stage2_seed_specs_for_lookup(seed_spec):
    yield seed_spec, None
    if abs(seed_spec.banana_surf_radius - BANANA_WINDING_MINOR_RADIUS_M) <= 1.0e-12:
        yield (
            Stage2SeedSpec(
                plasma_surf_filename=seed_spec.plasma_surf_filename,
                major_radius=seed_spec.major_radius,
                toroidal_flux=seed_spec.toroidal_flux,
                length_weight=seed_spec.length_weight,
                cc_weight=seed_spec.cc_weight,
                cc_threshold=seed_spec.cc_threshold,
                curvature_weight=seed_spec.curvature_weight,
                curvature_threshold=seed_spec.curvature_threshold,
                banana_surf_radius=LEGACY_STAGE2_BANANA_WINDING_MINOR_RADIUS_M,
                tf_current_A=seed_spec.tf_current_A,
                order=seed_spec.order,
                banana_init_current_A=seed_spec.banana_init_current_A,
                banana_current_max_A=seed_spec.banana_current_max_A,
            ),
            (
                "legacy banana winding surface radius "
                f"{LEGACY_STAGE2_BANANA_WINDING_MINOR_RADIUS_M:.3f} m"
            ),
        )


def build_stage2_bs_path(args):
    if args.stage2_bs_path:
        return args.stage2_bs_path

    seed_spec = _stage2_seed_spec_from_args(args)
    lookup_specs = tuple(_iter_stage2_seed_specs_for_lookup(seed_spec))

    if args.stage2_source == "database":
        parent = _stage2_outputs_parent(
            args.database_stage2_root,
            args.plasma_surf_filename,
        )
        for lookup_spec, compatibility_note in lookup_specs:
            note_suffix = "" if compatibility_note is None else f"; {compatibility_note}"
            seed_dir = format_database_stage2_seed_dir(lookup_spec)
            candidate = os.path.join(
                parent,
                seed_dir,
                "biot_savart_opt.json",
            )
            if os.path.exists(candidate):
                if compatibility_note is not None:
                    print(
                        f"Note: found legacy Stage 2 database output at {seed_dir}/ "
                        f"({compatibility_note})"
                    )
                return candidate

            legacy_init_dir = format_database_stage2_seed_dir_without_init_current(lookup_spec)
            legacy_init = os.path.join(
                parent,
                legacy_init_dir,
                "biot_savart_opt.json",
            )
            if os.path.exists(legacy_init):
                print(
                    f"Note: found legacy Stage 2 database output at {legacy_init_dir}/ "
                    f"(missing INITC segment{note_suffix})"
                )
                return legacy_init

            legacy_dir = format_legacy_database_stage2_seed_dir(lookup_spec)
            legacy = os.path.join(
                parent,
                legacy_dir,
                "biot_savart_opt.json",
            )
            if os.path.exists(legacy):
                print(
                    f"Note: found legacy Stage 2 database output at {legacy_dir}/ "
                    f"(missing TFC segment{note_suffix})"
                )
                return legacy
            wataru_match = _resolve_unique_stage2_match(
                _wataru_stage2_patterns(
                    parent,
                    seed_dir,
                    include_constraint_variants=False,
                ),
                "current Wataru Stage 2 database output"
                if compatibility_note is None
                else f"legacy-compatible Wataru Stage 2 database output ({compatibility_note})",
            )
            if wataru_match is not None:
                return wataru_match
        return candidate

    parent = _stage2_outputs_parent(
        args.local_stage2_root,
        args.plasma_surf_filename,
    )
    for lookup_spec, compatibility_note in lookup_specs:
        note_suffix = "" if compatibility_note is None else f"; {compatibility_note}"
        seed_dir = format_local_stage2_seed_dir(lookup_spec)
        legacy_init_dir = format_local_stage2_seed_dir_without_init_current(lookup_spec)
        current_penalty_candidate = os.path.join(
            parent,
            seed_dir + "-CM=penalty",
            "biot_savart_opt.json",
        )
        if os.path.exists(current_penalty_candidate):
            if compatibility_note is not None:
                print(
                    f"Note: found legacy Stage 2 output at {seed_dir}-CM=penalty/ "
                    f"({compatibility_note})"
                )
            return current_penalty_candidate

        legacy_init_penalty_candidate = os.path.join(
            parent,
            legacy_init_dir + "-CM=penalty",
            "biot_savart_opt.json",
        )
        if os.path.exists(legacy_init_penalty_candidate):
            print(
                f"Note: found legacy Stage 2 output at {legacy_init_dir}/ "
                f"(missing INITC segment{note_suffix})"
            )
            return legacy_init_penalty_candidate

        candidate = os.path.join(
            parent,
            seed_dir,
            "biot_savart_opt.json",
        )
        if os.path.exists(candidate):
            print(
                f"Note: found legacy Stage 2 output at {seed_dir}/ "
                f"(missing constraint-method segment{note_suffix})"
            )
            return candidate

        legacy_init_candidate = os.path.join(
            parent,
            legacy_init_dir,
            "biot_savart_opt.json",
        )
        if os.path.exists(legacy_init_candidate):
            print(
                f"Note: found legacy Stage 2 output at {legacy_init_dir}/ "
                f"(missing INITC and constraint-method segments{note_suffix})"
            )
            return legacy_init_candidate

        no_tfc_dir = format_local_stage2_seed_dir_without_tf(lookup_spec)
        no_tfc_candidate = os.path.join(
            parent,
            no_tfc_dir,
            "biot_savart_opt.json",
        )
        if os.path.exists(no_tfc_candidate):
            print(
                f"Note: found legacy Stage 2 output at {no_tfc_dir}/ "
                f"(missing TFC segment{note_suffix})"
            )
            return no_tfc_candidate

        legacy_dir = format_legacy_local_stage2_seed_dir(lookup_spec)
        legacy = os.path.join(
            parent,
            legacy_dir,
            "biot_savart_opt.json",
        )
        if os.path.exists(legacy):
            print(
                f"Note: found legacy Stage 2 output at {legacy_dir}/ "
                f"(missing CCT/CT segments{note_suffix})"
            )
            return legacy

        current_matches = _resolve_unique_stage2_match(
            [
                os.path.join(parent, seed_dir + "-CM=penalty-BH=*", "biot_savart_opt.json"),
                os.path.join(parent, seed_dir + "-CM=alm-*", "biot_savart_opt.json"),
                os.path.join(parent, seed_dir + "-CM=alm-*-BH=*", "biot_savart_opt.json"),
                os.path.join(parent, legacy_init_dir + "-CM=penalty-BH=*", "biot_savart_opt.json"),
                os.path.join(parent, legacy_init_dir + "-CM=alm-*", "biot_savart_opt.json"),
                os.path.join(parent, legacy_init_dir + "-CM=alm-*-BH=*", "biot_savart_opt.json"),
            ],
            "current Stage 2 output"
            if compatibility_note is None
            else f"legacy Stage 2 output ({compatibility_note})",
        )
        if current_matches is not None:
            return current_matches

        wataru_matches = _resolve_unique_stage2_match(
            _wataru_stage2_patterns(
                parent,
                seed_dir,
                include_constraint_variants=True,
            ),
            "current Wataru Stage 2 output"
            if compatibility_note is None
            else f"legacy-compatible Wataru Stage 2 output ({compatibility_note})",
        )
        if wataru_matches is not None:
            return wataru_matches

        no_tfc_matches = _resolve_unique_stage2_match(
            [
                os.path.join(parent, no_tfc_dir + "-CM=penalty", "biot_savart_opt.json"),
                os.path.join(parent, no_tfc_dir + "-CM=penalty-BH=*", "biot_savart_opt.json"),
                os.path.join(parent, no_tfc_dir + "-BH=*", "biot_savart_opt.json"),
            ],
            "legacy Stage 2 output (missing TFC segment)"
            if compatibility_note is None
            else f"legacy Stage 2 output (missing TFC segment; {compatibility_note})",
        )
        if no_tfc_matches is not None:
            return no_tfc_matches

        legacy_matches = _resolve_unique_stage2_match(
            [
                os.path.join(parent, legacy_dir + "-CM=penalty", "biot_savart_opt.json"),
                os.path.join(parent, legacy_dir + "-CM=penalty-BH=*", "biot_savart_opt.json"),
                os.path.join(parent, legacy_dir + "-BH=*", "biot_savart_opt.json"),
            ],
            "legacy Stage 2 output (missing CCT/CT segments)"
            if compatibility_note is None
            else f"legacy Stage 2 output (missing CCT/CT segments; {compatibility_note})",
        )
        if legacy_matches is not None:
            return legacy_matches

    return current_penalty_candidate


def infer_uniform_tf_current_A(tf_coils):
    return _infer_uniform_coil_current_A(tf_coils)


def resolve_stage2_tf_current_A(stage2_results, tf_coils):
    return _resolve_stage2_tf_current_A_impl(stage2_results, tf_coils)


def resolve_stage2_num_tf_coils(stage2_results, requested_num_tf_coils):
    return _resolve_stage2_num_tf_coils_impl(stage2_results, requested_num_tf_coils)


def resolve_single_stage_banana_surf_radius(stage2_results, requested_banana_surf_radius):
    return _resolve_single_stage_banana_surf_radius_impl(
        stage2_results,
        requested_banana_surf_radius,
    )


def validate_loaded_stage2_coils_partition(
    coils,
    *,
    stage2_results,
    requested_num_tf_coils,
):
    _validate_loaded_stage2_coils_partition_impl(
        coils,
        stage2_results=stage2_results,
        requested_num_tf_coils=requested_num_tf_coils,
    )


def compute_tf_G0(tf_coils):
    return _compute_tf_G0_impl(tf_coils)


def resolve_stage2_finite_current_mode(stage2_results, requested_finite_current_mode):
    return _resolve_stage2_finite_current_mode_impl(
        stage2_results,
        requested_finite_current_mode,
    )


def partition_loaded_stage2_coils(coils, stage2_results, requested_num_tf_coils):
    return _partition_loaded_stage2_coils_impl(
        coils,
        stage2_results=stage2_results,
        requested_num_tf_coils=requested_num_tf_coils,
    )


def load_stage2_seed_biot_savart(
    stage2_bs_path,
    *,
    stage2_results,
    num_tf_coils,
    seed_order_upgrade=None,
):
    bs = load(stage2_bs_path)
    coil_partitions = partition_loaded_stage2_coils(
        bs.coils,
        stage2_results,
        num_tf_coils,
    )
    if seed_order_upgrade is None:
        return bs, coil_partitions
    loaded_master_banana_curve = next(
        coil.curve
        for coil in coil_partitions.banana_coils
        if isinstance(coil.curve, CurveCWSFourierCPP)
    )
    if int(seed_order_upgrade) == int(loaded_master_banana_curve.order):
        return bs, coil_partitions
    upgraded_bs, _, upgraded_banana_coils = upgrade_loaded_seed_biot_savart_order(
        bs,
        banana_coils=coil_partitions.banana_coils,
        tf_coils=coil_partitions.tf_coils,
        proxy_coils=coil_partitions.proxy_coils,
        vf_coils=coil_partitions.vf_coils,
        new_order=int(seed_order_upgrade),
    )
    upgraded_partitions = replace(
        coil_partitions,
        banana_coils=tuple(upgraded_banana_coils),
    )
    return upgraded_bs, upgraded_partitions


def resolve_plasma_current_settings(
    args,
    *,
    finite_current_mode="wataru_proxy_field",
    default_plasma_current_A=0.0,
    num_surfaces=1,
):
    settings = _resolve_plasma_current_settings_for_num_surfaces_impl(
        raw_boozer_I=args.boozer_I,
        plasma_current_A=args.plasma_current_A,
        finite_current_mode=finite_current_mode,
        default_plasma_current_A=default_plasma_current_A,
        num_surfaces=num_surfaces,
        requested_finite_current_mode=getattr(args, "finite_current_mode", None),
    )
    return {
        "boozer_I": settings.boozer_I,
        "plasma_current_A": settings.plasma_current_A,
        "input_source": settings.input_source,
        "boozer_current_convention": settings.boozer_current_convention,
        "mode": settings.mode,
        "effective_mode": settings.effective_mode,
    }


def resolve_single_stage_banana_current_state(
    biot_savart,
    coil_partitions,
    *,
    mode,
    coordinate_scaling=BANANA_CURRENT_COORDINATE_SCALING_NONE,
):
    return _resolve_single_stage_banana_current_state_impl(
        biot_savart,
        coil_partitions,
        mode=mode,
        coordinate_scaling=coordinate_scaling,
    )


def coerce_single_stage_banana_current_state(
    banana_current_state: SingleStageBananaCurrentState | None = None,
    *,
    banana_coils=None,
) -> SingleStageBananaCurrentState | None:
    if banana_current_state is not None:
        return banana_current_state
    if not banana_coils:
        return None
    return build_single_stage_banana_current_state(
        banana_coils,
        mode=BANANA_CURRENT_MODE_SHARED,
    )


def resolve_surface_mode_contract(args, *, warn_on_legacy_mapping=True):
    should_warn = (
        bool(warn_on_legacy_mapping)
        and getattr(args, "surface_mode", None) is None
        and int(getattr(args, "num_surfaces", 1)) != 1
    )
    return _build_surface_mode_contract_impl(
        requested_surface_mode=getattr(args, "surface_mode", None),
        legacy_num_surfaces=getattr(args, "num_surfaces", 1),
        legacy_inner_surface_ratio=getattr(
            args,
            "inner_surface_ratio",
            DEFAULT_INNER_SURFACE_RATIO,
        ),
        warn_on_legacy_mapping=should_warn,
    )


def build_equilibrium_path(args):
    return _build_equilibrium_path_impl(
        args.plasma_surf_filename,
        args.equilibria_dir,
        equilibrium_path=args.equilibrium_path,
        database_equilibria_dir=DATABASE_EQUILIBRIA_DIR,
    )


_STAGE2_SEED_NON_CONSTRAINT_DEFAULTS = {
    "toroidal_flux": 0.24,
    "length_weight": 0.0005,
    "cc_weight": 100.0,
    "curvature_weight": 0.0001,
    "order": 2,
    "banana_init_current_A": 1.0e4,
}

_STAGE2_SEED_CONTRACT_KEYS = (
    "tf_current_A",
    "banana_current_max_A",
    "length_target",
    "cc_threshold",
    "curvature_threshold",
    "banana_surf_radius",
    "target_lcfs_max_major_radius_m",
    "target_lcfs_max_minor_radius_m",
)


def apply_default_stage2_seed_args(args):
    """Populate stage2 seed args using the shared constraint contract resolver.

    The resolver owns all engineering and geometry constraint fields; the
    plasma-specific profile in :data:`DEFAULT_STAGE2_SEEDS_BY_PLASMA` supplies
    weights, toroidal flux, Fourier order, and the banana initialization
    current. Fixed vessel/winding geometry is routed automatically and is not
    subject to CLI override.
    """
    plasma_profile = DEFAULT_STAGE2_SEEDS_BY_PLASMA.get(
        args.plasma_surf_filename,
        {},
    )
    constraint_profile = {
        key: plasma_profile[key]
        for key in _STAGE2_SEED_CONTRACT_KEYS
        if key in plasma_profile
    }
    cli_seed_layer = {
        "tf_current_A": args.stage2_seed_tf_current_A,
        "cc_threshold": args.stage2_seed_cc_threshold,
        "curvature_threshold": args.stage2_seed_curvature_threshold,
        "banana_surf_radius": args.stage2_seed_banana_surf_radius,
    }
    contract, _trace = _resolve_constraint_contract_from_wire_names_impl(
        profile=constraint_profile,
        cli_overrides=cli_seed_layer,
        accept_offspec_major_radius=bool(
            getattr(args, "accept_offspec_r0_seed", False)
        ),
        offspec_major_radius_m=args.stage2_seed_major_radius,
    )
    if args.stage2_seed_major_radius is None:
        args.stage2_seed_major_radius = contract["VACUUM_VESSEL_MAJOR_RADIUS_M"]
    if args.stage2_seed_cc_threshold is None:
        args.stage2_seed_cc_threshold = contract["CC_THRESHOLD"]
    if args.stage2_seed_curvature_threshold is None:
        args.stage2_seed_curvature_threshold = contract["CURVATURE_THRESHOLD"]
    if args.stage2_seed_banana_surf_radius is None:
        args.stage2_seed_banana_surf_radius = contract["banana_surf_radius"]
    if args.stage2_seed_tf_current_A is None:
        args.stage2_seed_tf_current_A = contract["TF_CURRENT_A"]
    for attr_name, default_value in _STAGE2_SEED_NON_CONSTRAINT_DEFAULTS.items():
        arg_attr = f"stage2_seed_{attr_name}"
        if getattr(args, arg_attr) is None:
            setattr(
                args,
                arg_attr,
                plasma_profile.get(attr_name, default_value),
            )
    return args


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run single-stage Boozer/quasi-symmetry optimization from a Stage 2 seed.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=os.environ.get("PLASMA_SURF_FILENAME", "wout_nfp22ginsburg_000_014417_iota15.nc"),
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
        default=os.environ.get("SINGLE_STAGE_OUTPUT_ROOT", DEFAULT_SINGLE_STAGE_OUTPUT_ROOT),
        help="Directory where the single-stage output family will be written.",
    )
    parser.add_argument(
        "--resume-solver-checkpoint",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--warm-start-surface-stem",
        default=None,
        help=(
            "Optional stem for saved single-stage surface artifacts "
            "(for example /path/to/surf_best_feasible). When set, the single-stage "
            "initializer reuses the saved Boozer surface geometry/iota/G as its "
            "starting seed."
        ),
    )
    parser.add_argument(
        "--banana-surf-radius",
        type=float,
        default=float(os.environ["BANANA_SURF_RADIUS"]) if "BANANA_SURF_RADIUS" in os.environ else None,
        help="Coil surface minor radius. Defaults to the Stage 2 seed radius when omitted.",
    )
    parser.add_argument("--nphi", type=int, default=int(os.environ.get("NPHI", "255")))
    parser.add_argument("--ntheta", type=int, default=int(os.environ.get("NTHETA", "64")))
    parser.add_argument(
        "--init-only",
        action="store_true",
        help="Build the initial Boozer surface, write init artifacts, and skip the optimizer.",
    )
    parser.add_argument("--mpol", type=int, default=int(os.environ.get("MPOL", "8")))
    parser.add_argument("--ntor", type=int, default=int(os.environ.get("NTOR", "6")))
    parser.add_argument(
        "--single-stage-goal-mode",
        choices=["target", "frontier"],
        default=os.environ.get("SINGLE_STAGE_GOAL_MODE", "target"),
        help=(
            "Single-stage physics-goal contract. 'target' preserves the existing target-based "
            "formulation. 'frontier' uses a normalized tradeoff score that rewards higher "
            "iota and larger volume, penalizes QA error and Boozer residual relative to the "
            "seed, and rejects candidates whose Boozer residual exceeds the frontier trust "
            "threshold. Defaults to the SINGLE_STAGE_GOAL_MODE environment variable when set, "
            "otherwise 'target'."
        ),
    )
    parser.add_argument(
        "--frontier-volume-weight",
        type=float,
        default=float(os.environ["FRONTIER_VOLUME_WEIGHT"]) if "FRONTIER_VOLUME_WEIGHT" in os.environ else None,
        help=(
            "Independent volume-reward weight for frontier mode. Normalised against the "
            "legacy baseline (100) to set effective_volume_weight. When omitted, the volume "
            "weight falls back to --iotas-weight. Ignored in target mode."
        ),
    )
    parser.add_argument("--frontier-reference-iota", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-reference-iota-scale", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-reference-volume", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-reference-volume-scale", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-reference-qa", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-reference-boozer", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-boozer-trust-threshold", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-boozer-trust-penalty-scale", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--frontier-scalarization-type",
        choices=[
            FRONTIER_SCALARIZATION_TYPE_WEIGHT_SCHEDULE,
            FRONTIER_SCALARIZATION_TYPE_REFERENCE_POINT,
            FRONTIER_SCALARIZATION_TYPE_ACHIEVEMENT,
            FRONTIER_SCALARIZATION_TYPE_EPSILON,
        ],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--frontier-chebyshev-rho", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-chebyshev-sharpness", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-chebyshev-weight-iota", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-chebyshev-weight-volume", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-chebyshev-weight-qa", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-chebyshev-weight-boozer", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--epsilon-constraint-qa-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--epsilon-constraint-boozer-max", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--frontier-epsilon-penalty-weight", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--vol-target",
        type=float,
        default=float(os.environ.get("VOL_TARGET", "0.10")),
        help=(
            "Outer Boozer-surface target volume used to construct the numerical surface solve. "
            "In frontier mode this remains a solver reference, not an outer optimization target."
        ),
    )
    parser.add_argument(
        "--constraint-weight",
        type=float,
        default=float(os.environ.get("CONSTRAINT_WEIGHT", "1.0")),
        help=(
            "Boozer constraint weight. Use a non-negative value for least-squares mode "
            "(default 1.0). Use a negative value to select the exact Boozer Newton solver."
        ),
    )
    parser.add_argument(
        "--boozer-I",
        type=float,
        default=float(os.environ["BOOZER_I"]) if "BOOZER_I" in os.environ else None,
        help=(
            "Expert/internal override for the solver-facing BoozerSurface I input. "
            "Prefer --plasma-current-A for the standard Wataru-style mu0*I_A path."
        ),
    )
    parser.add_argument(
        "--plasma-current-A",
        type=float,
        default=float(os.environ["PLASMA_CURRENT_A"]) if "PLASMA_CURRENT_A" in os.environ else None,
        help=(
            "User-facing enclosed toroidal plasma current in physical SI amperes. "
            "In single-surface mode this is converted to BoozerSurface I using "
            "Wataru's mu0*I_A convention."
        ),
    )
    parser.add_argument(
        "--finite-current-mode",
        choices=["boozer_surrogate", "wataru_proxy_field"],
        default=os.environ.get("FINITE_CURRENT_MODE"),
        help=(
            "Finite-current interpretation for the loaded Stage 2 donor. When omitted, "
            "single-stage reload uses the donor artifact metadata. Ignored when "
            "--num-surfaces=1 unless set to wataru_proxy_field, because the "
            "single-surface path is locked to the Wataru proxy-field contract."
        ),
    )
    parser.add_argument("--maxiter", type=int, default=int(os.environ.get("MAXITER", "300")))
    parser.add_argument(
        "--surface-mode",
        choices=SURFACE_MODE_CHOICES,
        default=os.environ.get("SURFACE_MODE"),
        help=(
            "Surface physics contract selector. "
            f"{SINGLE_SURFACE!r} preserves the current one-surface baseline, "
            f"{EXPERIMENTAL_MULTISURFACE!r} preserves the current custom two-surface "
            "continuation lane, and "
            f"{PUBLISHED_MULTISURFACE!r} reserves the future published-aligned "
            "multisurface contract. When omitted, legacy --num-surfaces mapping is used."
        ),
    )
    parser.add_argument(
        "--num-surfaces",
        type=int,
        choices=[1, 2],
        default=int(os.environ.get("NUM_SURFACES", "1")),
        help=(
            "Legacy surface selector retained for backward compatibility. "
            "Use --surface-mode for new runs. Legacy mapping keeps 1 -> single_surface "
            "and 2 -> experimental_multisurface."
        ),
    )
    parser.add_argument(
        "--inner-surface-ratio",
        type=float,
        default=float(
            os.environ.get(
                "INNER_SURFACE_RATIO",
                str(DEFAULT_INNER_SURFACE_RATIO),
            )
        ),
        help=(
            "Legacy inner-surface ratio used by the experimental two-surface contract. "
            "Ignored by single_surface. Reserved for the future published contract."
        ),
    )
    parser.add_argument(
        "--surface-gap-threshold",
        type=float,
        default=float(os.environ.get("SURFACE_GAP_THRESHOLD", "0.0")),
        help="Minimum allowed point-cloud gap between adjacent optimized Boozer surfaces in multi-surface mode.",
    )
    parser.add_argument(
        "--multisurface-ramp-iterations",
        type=int,
        default=int(os.environ.get("MULTISURFACE_RAMP_ITERATIONS", "5")),
        help=(
            "Number of accepted outer iterations over which inner-surface search weight ramps "
            "from --inner-surface-initial-weight to 1.0 in two-surface mode. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--inner-surface-initial-weight",
        type=float,
        default=float(os.environ.get("INNER_SURFACE_INITIAL_WEIGHT", "0.0")),
        help=(
            "Initial search weight applied to inner-surface QS/Boozer terms in two-surface mode. "
            "Must be between 0 and 1."
        ),
    )
    parser.add_argument(
        "--multisurface-initial-step-scale",
        type=float,
        default=float(os.environ.get("MULTISURFACE_INITIAL_STEP_SCALE", "1.0")),
        help=(
            "Physical step scale for an optional first outer-optimization phase. "
            "Values below 1.0 shrink early L-BFGS-B moves in a mathematically consistent scaled coordinate system."
        ),
    )
    parser.add_argument(
        "--multisurface-initial-step-maxiter",
        type=int,
        default=int(os.environ.get("MULTISURFACE_INITIAL_STEP_MAXITER", "0")),
        help=(
            "Maximum outer iterations to run in the scaled first phase. "
            "Set to 0 to disable the early-step continuation phase."
        ),
    )
    parser.add_argument(
        "--topology-gate-fieldlines",
        type=int,
        default=int(os.environ.get("TOPOLOGY_GATE_FIELDLINES", "4")),
        help=(
            "Number of cheap outer-surface field lines used for the search-time topology gate in "
            "multi-surface mode. Set to 0 to disable."
        ),
    )
    parser.add_argument(
        "--topology-gate-tmax",
        type=float,
        default=float(os.environ.get("TOPOLOGY_GATE_TMAX", "2.0")),
        help="Integration horizon for the search-time topology gate field-line traces.",
    )
    parser.add_argument(
        "--topology-gate-tol",
        type=float,
        default=float(os.environ.get("TOPOLOGY_GATE_TOL", "1e-7")),
        help="Integrator tolerance for the search-time topology gate field-line traces.",
    )
    parser.add_argument(
        "--topology-gate-survival-threshold",
        type=float,
        default=float(os.environ.get("TOPOLOGY_GATE_SURVIVAL_THRESHOLD", "0.25")),
        help=(
            "Minimum survival fraction required by the cheap search-time topology gate in "
            "multi-surface mode."
        ),
    )
    parser.add_argument(
        "--topology-gate-penalty-scale",
        type=float,
        default=float(os.environ.get("TOPOLOGY_GATE_PENALTY_SCALE", "4.0")),
        help=(
            "Scale factor for topology-deficit rejection severity. Applied only when a "
            "candidate fails the search-time topology gate and the solver falls back to "
            "the last accepted gradient."
        ),
    )
    parser.add_argument(
        "--hardware-search-mode",
        choices=["hard", "warn", "adaptive"],
        default=os.environ.get("HARDWARE_SEARCH_MODE", DEFAULT_HARDWARE_SEARCH_MODE),
        help=(
            "Search-time policy for realized hardware violations: hard reject, warn only, "
            "or adaptive softening during early continuation."
        ),
    )
    parser.add_argument(
        "--hardware-search-soft-iterations",
        type=int,
        default=int(
            os.environ.get(
                "HARDWARE_SEARCH_SOFT_ITERATIONS",
                str(DEFAULT_HARDWARE_SEARCH_SOFT_ITERATIONS),
            )
        ),
        help=(
            "For --hardware-search-mode=adaptive, allow warning-only handling only while the "
            "multisurface search gate remains relaxed (gate_scale < 1.0). When this value is "
            "positive, it caps the number of accepted relaxed-gate iterations that may use the "
            "warning-only path before reverting to hard rejection."
        ),
    )
    parser.add_argument(
        "--curvature-traversal-band",
        type=float,
        default=float(
            os.environ.get(
                "CURVATURE_TRAVERSAL_BAND",
                str(DEFAULT_CURVATURE_TRAVERSAL_BAND),
            )
        ),
        help=(
            "Relative over-curvature band allowed to reach Boozer before hard search-time "
            "curvature rejection. A value of 0.05 allows trials up to 5 percent above "
            "--curvature-threshold while budget remains."
        ),
    )
    parser.add_argument(
        "--curvature-traversal-eval-budget",
        type=int,
        default=int(
            os.environ.get(
                "CURVATURE_TRAVERSAL_EVAL_BUDGET",
                str(DEFAULT_CURVATURE_TRAVERSAL_EVAL_BUDGET),
            )
        ),
        help=(
            "Per accepted iteration budget for Boozer evaluations whose cheap precheck "
            "curvature is above --curvature-threshold but inside --curvature-traversal-band."
        ),
    )
    parser.add_argument(
        "--ftol",
        type=float,
        default=float(os.environ.get("FTOL", "1e-15")),
        help="L-BFGS-B function tolerance (default: 1e-15).",
    )
    parser.add_argument(
        "--gtol",
        type=float,
        default=float(os.environ.get("GTOL", "1e-15")),
        help="L-BFGS-B gradient tolerance (default: 1e-15).",
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
        # Single-stage uses 0.05 (tighter tracking of hard max curvature)
        # vs Stage 2's 0.25 (broader softmax window). The gap is intentional:
        # single-stage curvature responds more sensitively during full Boozer
        # surface optimization, and a tighter smoothing prevents phantom
        # constraint activation from distant curvature peaks.
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
        default=float(os.environ["ALM_QS_THRESHOLD"]) if "ALM_QS_THRESHOLD" in os.environ else None,
        help="thresholded_physics-mode upper bound for the quasi-symmetry objective J_QS.",
    )
    parser.add_argument(
        "--alm-boozer-threshold",
        type=float,
        default=float(os.environ["ALM_BOOZER_THRESHOLD"]) if "ALM_BOOZER_THRESHOLD" in os.environ else None,
        help="thresholded_physics-mode upper bound for the Boozer residual objective.",
    )
    parser.add_argument(
        "--alm-iota-penalty-threshold",
        type=float,
        default=(
            float(os.environ["ALM_IOTA_PENALTY_THRESHOLD"])
            if "ALM_IOTA_PENALTY_THRESHOLD" in os.environ
            else None
        ),
        help="thresholded_physics-mode upper bound for the Jiota penalty objective.",
    )
    parser.add_argument(
        "--alm-length-penalty-threshold",
        type=float,
        default=(
            float(os.environ["ALM_LENGTH_PENALTY_THRESHOLD"])
            if "ALM_LENGTH_PENALTY_THRESHOLD" in os.environ
            else None
        ),
        help="thresholded_physics-mode upper bound for the single-stage length penalty objective.",
    )
    parser.add_argument(
        "--iota-target",
        type=float,
        # Set this explicitly from the equilibrium and working surface.
        # The 0.15 value is only a historical fallback.
        default=float(os.environ.get("IOTA_TARGET", "0.15")),
        help=(
            "Target-mode iota penalty center and Boozer initialization guess. In frontier mode "
            "the outer objective no longer targets this value."
        ),
    )
    parser.add_argument("--num-tf-coils", type=int, default=int(os.environ.get("NUM_TF_COILS", "20")))
    parser.add_argument(
        "--boozer-stage",
        choices=["initial", "final"],
        default=os.environ.get("BOOZER_STAGE", "initial"),
        help="Use least-squares Boozer residual during initial stage or exact residual during final stage.",
    )
    parser.add_argument(
        "--boozer-stage-refinement",
        action="store_true",
        help=(
            "Penalty-mode single-surface only: after an initial-stage run, restart from the best "
            "accepted hardware-feasible incumbent with a different Boozer residual stage."
        ),
    )
    parser.add_argument(
        "--refinement-boozer-stage",
        choices=["initial", "final"],
        default=os.environ.get("REFINEMENT_BOOZER_STAGE", "final"),
        help=(
            "Residual stage to use for the optional refinement restart. In v1 this changes only "
            "the Boozer residual objective term, not the underlying Boozer initialization method."
        ),
    )
    parser.add_argument(
        "--refinement-maxiter",
        type=int,
        default=int(os.environ.get("REFINEMENT_MAXITER", "100")),
        help="Maximum L-BFGS-B iterations for the optional Boozer-stage refinement restart.",
    )
    parser.add_argument(
        "--refinement-chunk-maxiter",
        type=int,
        default=int(os.environ.get("REFINEMENT_CHUNK_MAXITER", "20")),
        help="Maximum L-BFGS-B iterations per optional Boozer-stage refinement chunk.",
    )
    parser.add_argument(
        "--refinement-max-stalled-chunks",
        type=int,
        default=int(os.environ.get("REFINEMENT_MAX_STALLED_CHUNKS", "2")),
        help="Abort refinement after this many consecutive chunks without accepted-state improvement.",
    )
    parser.add_argument("--cc-dist", type=float, default=float(os.environ.get("CC_DIST", str(COIL_COIL_MIN_DIST_M))))
    parser.add_argument("--curvature-threshold", type=float, default=float(os.environ.get("CURVATURE_THRESHOLD", str(MAX_CURVATURE_INV_M))))
    parser.add_argument("--cc-weight", type=float, default=float(os.environ.get("CC_WEIGHT", "100")))
    parser.add_argument("--curvature-weight", type=float, default=float(os.environ.get("CURVATURE_WEIGHT", "0.1")))
    parser.add_argument("--length-weight", type=float, default=float(os.environ.get("SS_LENGTH_WEIGHT", "1")),
                        help="Curve length penalty weight (default 1).")
    parser.add_argument(
        "--banana-current-max-A",
        type=float,
        default=float(
            os.environ.get("BANANA_CURRENT_MAX_A", str(BANANA_CURRENT_HARD_LIMIT_A))
        ),
        help=(
            "Maximum allowed magnitude for the banana current in amps. "
            "Penalty/L-BFGS-B mode applies this as a hard box bound; "
            "ALM mode still rechecks it at final feasibility but may "
            "temporarily traverse over-cap values during search."
        ),
    )
    parser.add_argument(
        "--single-stage-banana-current-mode",
        choices=[BANANA_CURRENT_MODE_SHARED, BANANA_CURRENT_MODE_INDEPENDENT],
        default=os.environ.get(
            "SINGLE_STAGE_BANANA_CURRENT_MODE",
            BANANA_CURRENT_MODE_SHARED,
        ),
        help=(
            "Single-stage banana-current control contract. 'shared' preserves the "
            "historical single shared banana-current DOF. 'independent' creates one "
            "current DOF per loaded banana coil while preserving the loaded Stage 2 "
            "current state at the handoff."
        ),
    )
    parser.add_argument(
        "--single-stage-banana-current-coordinate-scaling",
        choices=[
            BANANA_CURRENT_COORDINATE_SCALING_NONE,
            BANANA_CURRENT_COORDINATE_SCALING_SEED_RELATIVE,
        ],
        default=os.environ.get(
            "SINGLE_STAGE_BANANA_CURRENT_COORDINATE_SCALING",
            BANANA_CURRENT_COORDINATE_SCALING_NONE,
        ),
        help=(
            "Optimizer coordinate scaling for independent banana-current controls. "
            "'none' keeps optimizer coordinates in amps. 'seed-relative' wraps each "
            "independent current as ScaledCurrent(Current(I_seed/|I_seed|), |I_seed|), "
            "so L-BFGS-B sees order-one current coordinates while physical currents "
            "and artifacts remain in amps."
        ),
    )
    parser.add_argument(
        "--banana-current-diagnostics",
        action="store_true",
        default=env_flag("BANANA_CURRENT_DIAGNOSTICS"),
        help=(
            "Emit explicit banana-current coordinate diagnostics for the optimizer "
            "seed, each accepted iterate, and recent rejected trial points. "
            "Writes banana_current_diagnostics.json next to results.json."
        ),
    )
    parser.add_argument(
        "--banana-current-fd-diagnostics",
        action="store_true",
        default=env_flag("BANANA_CURRENT_FD_DIAGNOSTICS"),
        help=(
            "Augment banana-current diagnostics with a seed-only finite-difference "
            "probe that compares banana-current coordinates against a matched set of "
            "noncurrent optimizer coordinates using symmetric in-bounds relative "
            "perturbations through the live single-stage objective path."
        ),
    )
    parser.add_argument(
        "--banana-current-fd-relative-step-fraction",
        type=float,
        default=float(
            os.environ.get(
                "BANANA_CURRENT_FD_RELATIVE_STEP_FRACTION",
                "0.01",
            )
        ),
        help=(
            "Relative coordinate perturbation used by --banana-current-fd-diagnostics. "
            "A value of 0.01 means symmetric +/-1%% perturbations, clipped to stay "
            "inside the active optimizer bounds."
        ),
    )
    parser.add_argument(
        "--banana-current-replay-diagnostics-path",
        default=os.environ.get("BANANA_CURRENT_REPLAY_DIAGNOSTICS_PATH"),
        help=(
            "Offline replay study input banana_current_diagnostics.json. "
            "When set, the script replays stored rejected trial current blocks "
            "from their accepted incumbents through the live surface/topology/"
            "hardware evaluation path. Requires banana_current_replay_context.json "
            "next to the diagnostics input, unless --banana-current-replay-context-path "
            "is provided."
        ),
    )
    parser.add_argument(
        "--banana-current-replay-context-path",
        default=os.environ.get("BANANA_CURRENT_REPLAY_CONTEXT_PATH"),
        help=(
            "Optional banana_current_replay_context.json override for the replay "
            "study. Defaults to the sibling replay-context artifact next to the "
            "diagnostics file."
        ),
    )
    parser.add_argument(
        "--banana-current-replay-output-path",
        default=os.environ.get("BANANA_CURRENT_REPLAY_OUTPUT_PATH"),
        help=(
            "Optional output path for the rejected-trial replay study artifact. "
            "Defaults to banana_current_rejected_trial_replay.json next to the "
            "diagnostics input."
        ),
    )
    parser.add_argument(
        "--length-target",
        type=float,
        default=float(os.environ.get("SS_LENGTH_TARGET", str(COIL_LENGTH_TARGET_M))),
        help=(
            "Curve length quadratic penalty target in meters (applies to banana_curves[0] via "
            f"QuadraticPenalty(..., 'max')). Defaults to the preferred hardware target of "
            f"{COIL_LENGTH_TARGET_M:.1f} m, with a hard ceiling at "
            f"{COIL_LENGTH_HARD_LIMIT_M:.1f} m. Passing a larger value is clamped back to "
            "the hardware ceiling; passing a smaller value makes the run stricter."
        ),
    )
    parser.add_argument("--res-weight", type=float, default=float(os.environ.get("RES_WEIGHT", "1000")),
                        help="Boozer residual penalty weight (default 1000).")
    parser.add_argument("--iotas-weight", type=float, default=float(os.environ.get("IOTAS_WEIGHT", "100")),
                        help="Iota target tracking weight (default 100).")
    parser.add_argument("--cs-weight", type=float, default=float(os.environ.get("CS_WEIGHT", "1")),
                        help="Coil-surface distance penalty weight (default 1).")
    parser.add_argument("--cs-dist", type=float, default=float(os.environ.get("CS_DIST", str(COIL_PLASMA_MIN_DIST_M))),
                        help="Minimum coil-surface distance in meters (default 0.015 = 1.5 cm, HBT spec).")
    parser.add_argument("--surf-dist-weight", type=float, default=float(os.environ.get("SURF_DIST_WEIGHT", "1000")),
                        help="Surface-vessel distance penalty weight (default 1000).")
    parser.add_argument("--ss-dist", type=float, default=float(os.environ.get("SS_DIST", str(PLASMA_VESSEL_MIN_DIST_M))),
                        help="Minimum surface-vessel distance in meters (default 0.04).")
    parser.add_argument(
        "--maxcor",
        type=int,
        default=int(os.environ.get("MAXCOR", str(DEFAULT_LBFGSB_MAXCOR))),
        help=(
            "L-BFGS-B memory (number of correction pairs, "
            f"default {DEFAULT_LBFGSB_MAXCOR})."
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
        "--stage2-seed-surf-path",
        default=os.environ.get("STAGE2_SEED_SURF_PATH"),
        help=(
            "Optional saved surface or Boozer-surface artifact from the Stage 2 seed lane. "
            "When set, the Boozer initializer reuses the saved surface geometry/iota/G "
            "as its starting seed."
        ),
    )
    parser.add_argument(
        "--seed-order-upgrade",
        type=int,
        default=(
            int(os.environ["SEED_ORDER_UPGRADE"])
            if "SEED_ORDER_UPGRADE" in os.environ
            else None
        ),
        help=(
            "Optional Fourier order upgrade applied to the loaded Stage 2 "
            "seed before rebuilding the banana symmetry family."
        ),
    )
    parser.add_argument(
        "--constraint-profile-label",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--constraint-override-reason",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--allow-offspec-engineering-constraints",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--seed-regime",
        choices=[
            _SINGLE_STAGE_SEED_REGIME_AUTO,
            _SINGLE_STAGE_SEED_REGIME_PRESERVE_FIRST,
            _SINGLE_STAGE_SEED_REGIME_REPAIR_FIRST,
            _SINGLE_STAGE_SEED_REGIME_BRIDGE_ONLY,
            _SINGLE_STAGE_SEED_REGIME_GLOBAL_SEARCH,
        ],
        default=os.environ.get("SEED_REGIME", _DEFAULT_SINGLE_STAGE_SEED_REGIME),
        help=(
            "Single-stage startup regime. 'preserve_first' protects a good donor, "
            "'repair_first' attempts a bounded local feasibility recovery, "
            "'bridge_only' runs a short local bridge solve from a clean initializer, "
            "and 'global_search' skips the bounded startup lane."
        ),
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
        default=float(os.environ["STAGE2_SEED_MAJOR_RADIUS"]) if "STAGE2_SEED_MAJOR_RADIUS" in os.environ else None,
    )
    parser.add_argument(
        "--accept-offspec-r0-seed",
        action="store_true",
        default=env_flag(ACCEPT_OFFSPEC_R0_SEED_ENV),
        help=ACCEPT_OFFSPEC_R0_SEED_HELP,
    )
    parser.add_argument(
        "--stage2-seed-toroidal-flux",
        type=float,
        default=float(os.environ["STAGE2_SEED_TOROIDAL_FLUX"]) if "STAGE2_SEED_TOROIDAL_FLUX" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-length-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_LENGTH_WEIGHT"]) if "STAGE2_SEED_LENGTH_WEIGHT" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-cc-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_CC_WEIGHT"]) if "STAGE2_SEED_CC_WEIGHT" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-curvature-weight",
        type=float,
        default=float(os.environ["STAGE2_SEED_CURVATURE_WEIGHT"]) if "STAGE2_SEED_CURVATURE_WEIGHT" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-cc-threshold",
        type=float,
        default=float(os.environ["STAGE2_SEED_CC_THRESHOLD"]) if "STAGE2_SEED_CC_THRESHOLD" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-curvature-threshold",
        type=float,
        default=float(os.environ["STAGE2_SEED_CURVATURE_THRESHOLD"]) if "STAGE2_SEED_CURVATURE_THRESHOLD" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-banana-surf-radius",
        type=float,
        default=float(os.environ["STAGE2_SEED_BANANA_SURF_RADIUS"]) if "STAGE2_SEED_BANANA_SURF_RADIUS" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-tf-current-A",
        type=float,
        default=float(os.environ["STAGE2_SEED_TF_CURRENT_A"]) if "STAGE2_SEED_TF_CURRENT_A" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-order",
        type=int,
        default=int(os.environ["STAGE2_SEED_ORDER"]) if "STAGE2_SEED_ORDER" in os.environ else None,
    )
    parser.add_argument(
        "--stage2-seed-banana-init-current-A",
        type=float,
        default=(
            float(os.environ["STAGE2_SEED_BANANA_INIT_CURRENT_A"])
            if "STAGE2_SEED_BANANA_INIT_CURRENT_A" in os.environ
            else None
        ),
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=int(os.environ.get("CHECKPOINT_EVERY", "0")),
        help="Save checkpoint artifacts every N accepted iterations (0 = disabled, default).",
    )
    parser.add_argument(
        "--topology-scorer-every",
        type=int,
        default=int(os.environ.get("TOPOLOGY_SCORER_EVERY", "0")),
        help="Run medium-fidelity topology scorer every N accepted iterations (0 = disabled, default).",
    )
    parser.add_argument(
        "--topology-scorer-nfieldlines",
        type=int,
        default=int(os.environ.get("TOPOLOGY_SCORER_NFIELDLINES", "12")),
        help="Number of field lines for callback topology scorer (default 12).",
    )
    parser.add_argument(
        "--topology-scorer-tmax",
        type=float,
        default=float(os.environ.get("TOPOLOGY_SCORER_TMAX", "50.0")),
        help="Integration horizon for callback topology scorer (default 50.0).",
    )
    add_confinement_surrogate_args(parser)
    parser.add_argument(
        "--basin-hops",
        type=int,
        default=int(os.environ.get("BASIN_HOPS", "0")),
        help="Number of basin-hopping restarts (0 = single L-BFGS-B, default).",
    )
    parser.add_argument(
        "--basin-stepsize",
        type=float,
        default=float(os.environ.get("BASIN_STEPSIZE", "0.01")),
        help="Perturbation scale for basin-hopping (default 0.01).",
    )
    parser.add_argument(
        "--basin-temperature",
        type=float,
        default=float(os.environ.get("BASIN_TEMPERATURE", "1.0")),
        help="Metropolis temperature for basin-hopping uphill acceptance (default 1.0).",
    )
    parser.add_argument(
        "--basin-niter-success",
        type=int,
        default=int(os.environ.get("BASIN_NITER_SUCCESS", "0")),
        help="Stop basin-hopping early after this many hops without improvement (0 = disabled, default).",
    )
    parser.add_argument(
        "--basin-seed",
        type=int,
        default=int(os.environ.get("BASIN_SEED", "-1")),
        help="RNG seed for basin-hopping (-1 = random). Set for reproducibility.",
    )
    return parser.parse_args()


def initialize_boozer_surface(
    surf_prev,
    mpol,
    ntor,
    bs,
    vol_target,
    constraint_weight,
    iota,
    G0,
    boozer_I=0.0,
    *,
    initial_surface_guess=None,
    nfp=5,
):
    return _initialize_boozer_surface_impl(
        surf_prev,
        mpol,
        ntor,
        bs,
        vol_target,
        constraint_weight,
        iota,
        G0,
        boozer_I,
        initial_surface_guess=initial_surface_guess,
        nfp=nfp,
        surface_cls=SurfaceXYZTensorFourier,
        volume_cls=Volume,
        boozer_surface_cls=BoozerSurface,
    )


def build_surface_configs(
    file_loc,
    nphi,
    ntheta,
    seed_label,
    major_radius,
    outer_target_volume,
    num_surfaces,
    inner_surface_ratio,
):
    return _build_surface_configs_impl(
        file_loc,
        nphi,
        ntheta,
        seed_label,
        major_radius,
        outer_target_volume,
        num_surfaces,
        inner_surface_ratio,
        surface_factory=SurfaceRZFourier,
    )


def _resolved_optional_path_string(raw_path):
    if raw_path is None:
        return None
    return str(Path(raw_path).expanduser().resolve())


def resolve_initial_boozer_surface_seed(
    *,
    config_name,
    default_surface,
    default_iota,
    default_G,
    stage2_seed_surface,
    warm_start_surface_stem,
):
    if stage2_seed_surface is not None and config_name == "outer":
        return (
            stage2_seed_surface.surface,
            stage2_seed_surface.surface,
            (
                default_iota
                if stage2_seed_surface.iota is None
                else stage2_seed_surface.iota
            ),
            default_G if stage2_seed_surface.G is None else stage2_seed_surface.G,
            None,
        )
    if warm_start_surface_stem is None:
        return default_surface, None, default_iota, default_G, None

    warm_start_surface_path = resolve_warm_start_boozer_surface_path(
        warm_start_surface_stem,
        surface_name=config_name,
    )
    warm_start_seed = load_warm_start_boozer_seed(warm_start_surface_path)
    return (
        warm_start_seed.surface,
        warm_start_seed.surface,
        default_iota if warm_start_seed.iota is None else warm_start_seed.iota,
        default_G if warm_start_seed.G is None else warm_start_seed.G,
        str(warm_start_seed.source_path),
    )


def build_hbt_reference_surfaces(nfp, banana_surf_radius):
    surfaces = build_banana_reference_surfaces(nfp, banana_surf_radius)
    return (
        surfaces.vessel,
        surfaces.lcfs_clearance_reference,
        surfaces.coil_winding_surface,
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
    *,
    coil_length=None,
    length_target=None,
    tf_current_A=None,
    tf_current_limit_A=None,
    banana_current_A=None,
    banana_current_max_A=None,
):
    return _evaluate_single_stage_hardware_constraints(
        curve_curve_min_dist,
        cc_dist,
        curve_surface_min_dist,
        cs_dist,
        surface_vessel_min_dist,
        ss_dist,
        max_curvature,
        curvature_threshold,
        coil_length=coil_length,
        length_target=length_target,
        tf_current_A=tf_current_A,
        tf_current_limit_A=tf_current_limit_A,
        banana_current_A=banana_current_A,
        banana_current_max_A=banana_current_max_A,
    )


def compute_single_stage_surface_vessel_min_dist(
    surface_vessel_distance_obj,
    surface_status,
    outer_surface=None,
    vessel_surface=None,
):
    return _compute_single_stage_surface_vessel_min_dist(
        surface_vessel_distance_obj,
        surface_status,
        outer_surface,
        vessel_surface,
    )


def current_single_stage_hardware_snapshot_kwargs(*, coil_length=None):
    curvelength_obj = globals().get("curvelength")
    resolved_coil_length = None
    if curvelength_obj is not None:
        resolved_coil_length = float(curvelength_obj.J())
    if coil_length is not None:
        resolved_coil_length = float(coil_length)
    resolved_length_target = globals().get("length_target")
    resolved_tf_current_A = globals().get("stage2_tf_current_A")
    banana_current_state = globals().get("banana_current_state")
    resolved_banana_current_A = (
        None
        if banana_current_state is None
        else banana_current_state.control_current_A()
    )
    args_value = globals().get("args")
    resolved_banana_current_max_A = getattr(
        args_value,
        "banana_current_max_A",
        BANANA_CURRENT_HARD_LIMIT_A,
    )
    penalty_box_bounds = resolve_penalty_traversal_forbidden_box_bounds(
        {"banana_current": resolved_banana_current_max_A},
    )
    penalty_banana_current_max_A = penalty_box_bounds["banana_current"]
    return {
        "coil_length": resolved_coil_length,
        "length_target": resolved_length_target,
        "tf_current_A": resolved_tf_current_A,
        "tf_current_limit_A": (
            None if resolved_tf_current_A is None else TF_CURRENT_HARD_LIMIT_A
        ),
        "banana_current_A": resolved_banana_current_A,
        "banana_current_max_A": (
            None
            if resolved_banana_current_A is None
            else float(penalty_banana_current_max_A)
        ),
    }


def current_single_stage_alm_banana_current():
    banana_current_state = globals().get("banana_current_state")
    if banana_current_state is not None:
        representative_current_A = banana_current_state.representative_current_A()
        if representative_current_A is None:
            raise ValueError(
                "single-stage ALM banana-current constraints require "
                "--single-stage-banana-current-mode=shared"
            )
        return banana_current_state.currents[0]
    banana_coils_value = globals().get("banana_coils")
    if banana_coils_value:
        return banana_coils_value[0].current
    return None


def evaluate_topology_gate(surface, bfield, nfieldlines, tmax, tol, survival_threshold):
    return _evaluate_topology_gate_impl(
        surface,
        bfield,
        nfieldlines,
        tmax,
        tol,
        survival_threshold,
    )


def topology_gate_deficit(status):
    return _topology_gate_deficit(status)


def average_surface_objectives(objectives, weights=None):
    return _average_surface_objectives_impl(objectives, weights=weights)


def confinement_surrogate_kwargs():
    return {
        "surrogate_worst_k": CONFINEMENT_SURROGATE_WORST_K,
        "surrogate_early_exit_threshold": CONFINEMENT_SURROGATE_EARLY_THRESHOLD,
        "surrogate_mean_weight": CONFINEMENT_SURROGATE_MEAN_WEIGHT,
        "surrogate_worst_weight": CONFINEMENT_SURROGATE_WORST_WEIGHT,
        "surrogate_early_weight": CONFINEMENT_SURROGATE_EARLY_WEIGHT,
    }


def checkpoint_confinement_objective(proxy_objective, topology_result, confinement_weight):
    return float(proxy_objective) + float(confinement_weight) * float(topology_result["confinement_loss"])


def _format_topology_error(error):
    return str(error) or repr(error)


def safe_evaluate_topology_gate(surface, bfield, nfieldlines, tmax, tol, survival_threshold):
    try:
        return evaluate_topology_gate(
            surface,
            bfield,
            nfieldlines,
            tmax,
            tol,
            survival_threshold,
        )
    except Exception as error:
        return broken_topology_gate_status(
            tmax,
            tol,
            survival_threshold,
            nfieldlines=nfieldlines,
            error_message=_format_topology_error(error),
            error_type=type(error).__name__,
        )


def safe_score_topology(
    surface,
    bfield,
    *,
    nfieldlines,
    tmax,
    tol=1e-7,
    **kwargs,
):
    return _safe_score_topology_impl(
        surface,
        bfield,
        nfieldlines=nfieldlines,
        tmax=tmax,
        tol=tol,
        **kwargs,
    )


def validate_confinement_surrogate_args(args):
    if args.confinement_objective_weight < 0.0:
        raise ValueError("--confinement-objective-weight must be non-negative")
    if args.confinement_surrogate_worst_k <= 0:
        raise ValueError("--confinement-surrogate-worst-k must be positive")
    if not (0.0 < args.confinement_surrogate_early_threshold <= 1.0):
        raise ValueError("--confinement-surrogate-early-threshold must be in (0, 1]")
    if min(
        args.confinement_surrogate_mean_weight,
        args.confinement_surrogate_worst_weight,
        args.confinement_surrogate_early_weight,
    ) < 0.0:
        raise ValueError("--confinement-surrogate-* weights must be non-negative")


def validate_single_stage_current_args(args):
    banana_current_max_A = float(args.banana_current_max_A)
    banana_current_mode = getattr(
        args,
        "single_stage_banana_current_mode",
        BANANA_CURRENT_MODE_SHARED,
    )
    constraint_method = getattr(args, "constraint_method", "penalty")
    allow_offspec_engineering_constraints = bool(
        getattr(args, "allow_offspec_engineering_constraints", False)
    )
    if banana_current_mode not in {
        BANANA_CURRENT_MODE_SHARED,
        BANANA_CURRENT_MODE_INDEPENDENT,
    }:
        raise ValueError(
            "--single-stage-banana-current-mode must be one of {shared, independent}"
        )
    if banana_current_mode == BANANA_CURRENT_MODE_INDEPENDENT and constraint_method == "alm":
        raise ValueError(
            "--single-stage-banana-current-mode=independent is not supported with "
            "--constraint-method=alm"
        )
    if banana_current_max_A <= 0.0:
        raise ValueError("--banana-current-max-A must be positive.")
    if (
        banana_current_max_A > BANANA_CURRENT_HARD_LIMIT_A
        and not allow_offspec_engineering_constraints
    ):
        raise ValueError(
            f"--banana-current-max-A must be in the interval "
            f"(0, {BANANA_CURRENT_HARD_LIMIT_A:.0f}] unless "
            "--allow-offspec-engineering-constraints is set."
        )


@dataclass(frozen=True)
class RunIdentityConfig:
    stage2_bs_path: str
    stage: str
    boozer_stage_refinement: bool
    refinement_boozer_stage: str
    refinement_maxiter: int
    refinement_chunk_maxiter: int
    refinement_max_stalled_chunks: int
    constraint_weight: float
    constraint_method: str
    alm_formulation: str
    alm_qs_threshold: float | None
    alm_boozer_threshold: float | None
    alm_iota_penalty_threshold: float | None
    alm_length_penalty_threshold: float | None
    single_stage_goal_mode: str | None
    vol_target: float
    iota_target: float
    boozer_I: float
    plasma_current_A: float
    cc_dist: float
    cc_weight: float
    curvature_weight: float
    curvature_threshold: float
    banana_surf_radius: float
    banana_current_max_A: float
    single_stage_banana_current_mode: str
    single_stage_banana_current_coordinate_scaling: str
    num_banana_current_controls: int
    nphi: int
    ntheta: int
    init_only: bool
    basin_hops: int
    basin_stepsize: float
    basin_temperature: float
    basin_niter_success: int
    rng_seed: int | None
    ftol: float | None
    gtol: float | None
    alm_max_outer_iters: int
    alm_penalty_init: float
    alm_penalty_scale: float
    alm_penalty_max: float
    alm_feas_tol: float
    alm_stationarity_tol: float
    num_surfaces: int
    inner_surface_ratio: float
    surface_gap_threshold: float
    multisurface_ramp_iterations: int
    inner_surface_initial_weight: float
    multisurface_initial_step_scale: float
    multisurface_initial_step_maxiter: int
    topology_gate_fieldlines: int
    topology_gate_tmax: float
    topology_gate_tol: float
    topology_gate_survival_threshold: float
    topology_gate_penalty_scale: float
    hardware_search_mode: str
    hardware_search_soft_iterations: int
    curvature_traversal_band: float
    curvature_traversal_eval_budget: int
    topology_scorer_every: int
    topology_scorer_nfieldlines: int
    topology_scorer_tmax: float
    confinement_objective_weight: float
    confinement_surrogate_worst_k: int
    confinement_surrogate_early_threshold: float
    confinement_surrogate_mean_weight: float
    confinement_surrogate_worst_weight: float
    confinement_surrogate_early_weight: float
    alm_trust_radius_init: float
    alm_trust_radius_min: float
    alm_trust_radius_shrink: float
    alm_trust_radius_grow: float
    alm_max_inner_attempts: int
    alm_max_subproblem_continuations: int
    alm_distance_smoothing: float
    alm_curvature_smoothing: float
    warm_start_surface_stem: str | None = None
    stage2_seed_surf_path: str | None = None
    seed_regime: str | None = None


@dataclass(frozen=True)
class PreservedTimeoutReplayConfig:
    plasma_surf_filename: str | None
    plasma_surf_path: str | None
    stage2_bs_path: str | None
    stage2_results_path: str | None
    mpol: int | None
    ntor: int | None
    nphi: int | None
    ntheta: int | None
    constraint_weight: float | None
    constraint_method: str | None
    alm_formulation: str | None
    max_iterations: int | None
    target_volume: float | None
    target_iota: float | None
    requested_seed_regime: str | None = None
    effective_seed_regime: str | None = None
    single_stage_goal_mode: str | None = None
    single_stage_banana_current_mode: str | None = None
    single_stage_banana_current_coordinate_scaling: str | None = None
    num_banana_current_controls: int | None = None
    single_stage_goal_mode_impl: str | None = None
    boozer_surface_target_volumes: tuple[float, ...] | None = None
    frontier_iota_reference: float | None = None
    frontier_iota_scale: float | None = None
    frontier_volume_reference: float | None = None
    frontier_volume_scale: float | None = None
    frontier_qs_reference: float | None = None
    frontier_boozer_reference: float | None = None
    frontier_boozer_trust_threshold: float | None = None
    frontier_boozer_trust_penalty_scale: float | None = None
    frontier_effective_qs_weight: float | None = None
    frontier_effective_boozer_weight: float | None = None
    frontier_effective_iota_weight: float | None = None
    frontier_effective_volume_weight: float | None = None
    frontier_scalarization_type: str | None = None
    frontier_chebyshev_rho: float | None = None
    frontier_chebyshev_sharpness: float | None = None
    frontier_chebyshev_weight_iota: float | None = None
    frontier_chebyshev_weight_volume: float | None = None
    frontier_chebyshev_weight_qa: float | None = None
    frontier_chebyshev_weight_boozer: float | None = None
    epsilon_constraint_qa_max: float | None = None
    epsilon_constraint_boozer_max: float | None = None
    frontier_epsilon_penalty_weight: float | None = None
    stage2_seed_surf_path: str | None = None
    major_radius: float = VACUUM_VESSEL_MAJOR_RADIUS_M


@dataclass(frozen=True)
class PreservedTimeoutALMState:
    penalty: float
    multipliers: np.ndarray


@dataclass(frozen=True)
class FrontierGoalConfig:
    iota_reference: float
    iota_scale: float
    volume_reference: float
    volume_scale: float
    qs_reference: float
    boozer_reference: float
    boozer_trust_threshold: float
    boozer_trust_penalty_scale: float
    effective_qs_weight: float
    effective_boozer_weight: float
    effective_iota_weight: float
    effective_volume_weight: float
    scalarization_type: str
    chebyshev_rho: float
    chebyshev_sharpness: float
    chebyshev_weight_iota: float
    chebyshev_weight_volume: float
    chebyshev_weight_qa: float
    chebyshev_weight_boozer: float
    epsilon_constraint_qa_max: float | None
    epsilon_constraint_boozer_max: float | None
    epsilon_penalty_weight: float


FRONTIER_GOAL_MODE_IMPL = "frontier_tradeoff_score_v2"
FRONTIER_SCALARIZATION_TYPE_WEIGHT_SCHEDULE = "weight_schedule_v1"
FRONTIER_SCALARIZATION_TYPE_REFERENCE_POINT = FRONTIER_REFERENCE_MODE_REFERENCE_POINTS
FRONTIER_SCALARIZATION_TYPE_ACHIEVEMENT = FRONTIER_REFERENCE_MODE_ACHIEVEMENT
FRONTIER_SCALARIZATION_TYPE_EPSILON = FRONTIER_REFERENCE_MODE_EPSILON
_FRONTIER_LEGACY_RES_WEIGHT_BASELINE = 1000.0
_FRONTIER_LEGACY_IOTA_WEIGHT_BASELINE = 100.0
_FRONTIER_LEGACY_VOLUME_WEIGHT_BASELINE = 100.0
_FRONTIER_CHEBYSHEV_SHARPNESS = 12.0
_FRONTIER_EPSILON_PENALTY_WEIGHT = 4.0


class BoundedImprovementReward(Optimizable):
    """Smooth bounded reward that increases as a metric improves past a reference."""

    def __init__(self, metric_objective, reference, scale):
        self.metric_objective = metric_objective
        self.reference = float(reference)
        self.scale = float(scale)
        if not np.isfinite(self.reference):
            raise ValueError("BoundedImprovementReward requires a finite reference")
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("BoundedImprovementReward requires a positive finite scale")
        depends_on = [metric_objective] if isinstance(metric_objective, Optimizable) else []
        super().__init__(depends_on=depends_on)

    def _scaled_delta(self):
        return (float(self.metric_objective.J()) - self.reference) / self.scale

    def J(self):
        return -np.tanh(self._scaled_delta())

    def dJ(self, partials=False):
        delta = self._scaled_delta()
        prefactor = -(1.0 - np.tanh(delta) ** 2) / self.scale
        if partials:
            partial_gradient = self.metric_objective.dJ(partials=True)
            if not callable(partial_gradient):
                return prefactor * partial_gradient
            if isinstance(self.metric_objective, Optimizable):
                return prefactor * Derivative(
                    {
                        self.metric_objective: np.asarray(
                            partial_gradient(self.metric_objective),
                            dtype=float,
                        )
                    }
                )
            return lambda objective_optimizable: prefactor * np.asarray(
                partial_gradient(objective_optimizable),
                dtype=float,
            )
        return prefactor * np.asarray(self.metric_objective.dJ(), dtype=float)


def _normalized_frontier_weight(raw_weight, legacy_baseline):
    return float(raw_weight) / float(legacy_baseline)


def _frontier_override_or_default(override_value, default_value, *, minimum=None):
    value = default_value if override_value is None else float(override_value)
    if minimum is not None:
        value = max(float(minimum), value)
    return float(value)


def build_frontier_goal_config(
    *,
    initial_iota,
    initial_volume,
    initial_qs_objective,
    initial_boozer_objective,
    res_weight,
    iotas_weight,
    volume_weight=None,
    iota_reference_override=None,
    iota_scale_override=None,
    volume_reference_override=None,
    volume_scale_override=None,
    qs_reference_override=None,
    boozer_reference_override=None,
    boozer_trust_threshold_override=None,
    boozer_trust_penalty_scale_override=None,
    scalarization_type=None,
    chebyshev_rho_override=None,
    chebyshev_sharpness_override=None,
    chebyshev_weight_iota_override=None,
    chebyshev_weight_volume_override=None,
    chebyshev_weight_qa_override=None,
    chebyshev_weight_boozer_override=None,
    epsilon_constraint_qa_max_override=None,
    epsilon_constraint_boozer_max_override=None,
    epsilon_penalty_weight_override=None,
):
    if volume_weight is None:
        volume_weight = iotas_weight
    default_qs_reference = max(abs(float(initial_qs_objective)), 1e-6)
    default_boozer_reference = max(abs(float(initial_boozer_objective)), 1e-6)
    qs_reference = _frontier_override_or_default(
        qs_reference_override,
        default_qs_reference,
        minimum=1e-6,
    )
    boozer_reference = _frontier_override_or_default(
        boozer_reference_override,
        default_boozer_reference,
        minimum=1e-6,
    )
    boozer_trust_threshold = _frontier_override_or_default(
        boozer_trust_threshold_override,
        max(10.0 * boozer_reference, 1e-5),
        minimum=1e-5,
    )
    resolved_scalarization_type = (
        FRONTIER_SCALARIZATION_TYPE_WEIGHT_SCHEDULE
        if scalarization_type is None
        else str(scalarization_type)
    )
    return FrontierGoalConfig(
        iota_reference=_frontier_override_or_default(
            iota_reference_override,
            float(initial_iota),
        ),
        iota_scale=_frontier_override_or_default(
            iota_scale_override,
            max(abs(float(initial_iota)) * 0.25, 0.05),
            minimum=1e-6,
        ),
        volume_reference=_frontier_override_or_default(
            volume_reference_override,
            float(initial_volume),
        ),
        volume_scale=_frontier_override_or_default(
            volume_scale_override,
            max(abs(float(initial_volume)) * 0.10, 0.01),
            minimum=1e-6,
        ),
        qs_reference=qs_reference,
        boozer_reference=boozer_reference,
        boozer_trust_threshold=boozer_trust_threshold,
        boozer_trust_penalty_scale=_frontier_override_or_default(
            boozer_trust_penalty_scale_override,
            5.0 * boozer_trust_threshold,
            minimum=1e-6,
        ),
        effective_qs_weight=1.0,
        effective_boozer_weight=_normalized_frontier_weight(
            res_weight,
            _FRONTIER_LEGACY_RES_WEIGHT_BASELINE,
        ),
        effective_iota_weight=_normalized_frontier_weight(
            iotas_weight,
            _FRONTIER_LEGACY_IOTA_WEIGHT_BASELINE,
        ),
        effective_volume_weight=_normalized_frontier_weight(
            volume_weight,
            _FRONTIER_LEGACY_VOLUME_WEIGHT_BASELINE,
        ),
        scalarization_type=resolved_scalarization_type,
        chebyshev_rho=_frontier_override_or_default(
            chebyshev_rho_override,
            1.0e-3,
            minimum=0.0,
        ),
        chebyshev_sharpness=_frontier_override_or_default(
            chebyshev_sharpness_override,
            _FRONTIER_CHEBYSHEV_SHARPNESS,
            minimum=1.0e-12,
        ),
        chebyshev_weight_iota=_frontier_override_or_default(
            chebyshev_weight_iota_override,
            1.0,
            minimum=1.0e-12,
        ),
        chebyshev_weight_volume=_frontier_override_or_default(
            chebyshev_weight_volume_override,
            1.0,
            minimum=1.0e-12,
        ),
        chebyshev_weight_qa=_frontier_override_or_default(
            chebyshev_weight_qa_override,
            1.0,
            minimum=1.0e-12,
        ),
        chebyshev_weight_boozer=_frontier_override_or_default(
            chebyshev_weight_boozer_override,
            1.0,
            minimum=1.0e-12,
        ),
        epsilon_constraint_qa_max=(
            None
            if epsilon_constraint_qa_max_override is None
            else max(float(epsilon_constraint_qa_max_override), 1.0e-6)
        ),
        epsilon_constraint_boozer_max=(
            None
            if epsilon_constraint_boozer_max_override is None
            else max(float(epsilon_constraint_boozer_max_override), 1.0e-6)
        ),
        epsilon_penalty_weight=_frontier_override_or_default(
            epsilon_penalty_weight_override,
            _FRONTIER_EPSILON_PENALTY_WEIGHT,
            minimum=0.0,
        ),
    )


PRESERVED_TIMEOUT_REPLAY_CONFIG = PreservedTimeoutReplayConfig(
    plasma_surf_filename="",
    plasma_surf_path="",
    stage2_bs_path="",
    stage2_seed_surf_path=None,
    stage2_results_path="",
    mpol=0,
    ntor=0,
    nphi=0,
    ntheta=0,
    constraint_weight=None,
    constraint_method="penalty",
    alm_formulation=None,
    max_iterations=0,
    target_volume=0.0,
    target_iota=0.0,
    requested_seed_regime=None,
    effective_seed_regime=None,
    single_stage_goal_mode=None,
    single_stage_banana_current_mode=None,
    single_stage_banana_current_coordinate_scaling=None,
    num_banana_current_controls=None,
    single_stage_goal_mode_impl=None,
    boozer_surface_target_volumes=None,
    frontier_iota_reference=None,
    frontier_iota_scale=None,
    frontier_volume_reference=None,
    frontier_volume_scale=None,
    frontier_qs_reference=None,
    frontier_boozer_reference=None,
    frontier_boozer_trust_threshold=None,
    frontier_boozer_trust_penalty_scale=None,
    frontier_effective_qs_weight=None,
    frontier_effective_boozer_weight=None,
    frontier_effective_iota_weight=None,
    frontier_effective_volume_weight=None,
    frontier_scalarization_type=None,
    frontier_chebyshev_rho=None,
    frontier_chebyshev_sharpness=None,
    frontier_chebyshev_weight_iota=None,
    frontier_chebyshev_weight_volume=None,
    frontier_chebyshev_weight_qa=None,
    frontier_chebyshev_weight_boozer=None,
    epsilon_constraint_qa_max=None,
    epsilon_constraint_boozer_max=None,
    frontier_epsilon_penalty_weight=None,
)


def make_run_identity_config(
    args,
    stage2_bs_path,
    stage,
    constraint_weight,
    constraint_method,
    vol_target,
    iota_target,
    boozer_I,
    plasma_current_A,
    banana_surf_radius,
    nphi,
    ntheta,
    rng_seed,
    *,
    surface_mode_contract: SurfaceModeContract | None = None,
    effective_num_surfaces: int | None = None,
    effective_inner_surface_ratio: float | None = None,
    num_banana_current_controls: int = 1,
):
    resolved_contract = (
        resolve_surface_mode_contract(args, warn_on_legacy_mapping=False)
        if surface_mode_contract is None
        else surface_mode_contract
    )
    resolved_num_surfaces = (
        resolved_contract.num_surfaces
        if effective_num_surfaces is None
        else int(effective_num_surfaces)
    )
    resolved_inner_surface_ratio = resolve_surface_mode_inner_surface_ratio(
        resolved_contract,
        fallback_inner_surface_ratio=(
            args.inner_surface_ratio
            if effective_inner_surface_ratio is None
            else effective_inner_surface_ratio
        ),
    )
    return RunIdentityConfig(
        stage2_bs_path=stage2_bs_path,
        stage2_seed_surf_path=_resolved_optional_path_string(
            getattr(args, "stage2_seed_surf_path", None)
        ),
        stage=stage,
        boozer_stage_refinement=bool(args.boozer_stage_refinement),
        refinement_boozer_stage=args.refinement_boozer_stage,
        refinement_maxiter=args.refinement_maxiter,
        refinement_chunk_maxiter=args.refinement_chunk_maxiter,
        refinement_max_stalled_chunks=args.refinement_max_stalled_chunks,
        constraint_weight=constraint_weight,
        constraint_method=constraint_method,
        alm_formulation=getattr(args, "alm_formulation", "weighted_sum"),
        alm_qs_threshold=getattr(args, "alm_qs_threshold", None),
        alm_boozer_threshold=getattr(args, "alm_boozer_threshold", None),
        alm_iota_penalty_threshold=getattr(args, "alm_iota_penalty_threshold", None),
        alm_length_penalty_threshold=getattr(args, "alm_length_penalty_threshold", None),
        single_stage_goal_mode=(
            # Preserve legacy run fingerprints for explicit/implicit target-mode equivalence.
            args.single_stage_goal_mode if args.single_stage_goal_mode != "target" else None
        ),
        vol_target=vol_target,
        iota_target=iota_target,
        boozer_I=boozer_I,
        plasma_current_A=plasma_current_A,
        cc_dist=args.cc_dist,
        cc_weight=args.cc_weight,
        curvature_weight=args.curvature_weight,
        curvature_threshold=args.curvature_threshold,
        banana_surf_radius=banana_surf_radius,
        banana_current_max_A=getattr(
            args,
            "banana_current_max_A",
            BANANA_CURRENT_HARD_LIMIT_A,
        ),
        single_stage_banana_current_mode=getattr(
            args,
            "single_stage_banana_current_mode",
            BANANA_CURRENT_MODE_SHARED,
        ),
        single_stage_banana_current_coordinate_scaling=getattr(
            args,
            "single_stage_banana_current_coordinate_scaling",
            BANANA_CURRENT_COORDINATE_SCALING_NONE,
        ),
        num_banana_current_controls=int(num_banana_current_controls),
        nphi=nphi,
        ntheta=ntheta,
        init_only=args.init_only,
        basin_hops=args.basin_hops,
        basin_stepsize=args.basin_stepsize,
        basin_temperature=getattr(args, "basin_temperature", 1.0),
        basin_niter_success=getattr(args, "basin_niter_success", 0),
        rng_seed=rng_seed,
        ftol=args.ftol,
        gtol=args.gtol,
        alm_max_outer_iters=args.alm_max_outer_iters,
        alm_penalty_init=args.alm_penalty_init,
        alm_penalty_scale=args.alm_penalty_scale,
        alm_penalty_max=args.alm_penalty_max,
        alm_feas_tol=args.alm_feas_tol,
        alm_stationarity_tol=args.alm_stationarity_tol,
        num_surfaces=resolved_num_surfaces,
        inner_surface_ratio=resolved_inner_surface_ratio,
        surface_gap_threshold=args.surface_gap_threshold,
        multisurface_ramp_iterations=args.multisurface_ramp_iterations,
        inner_surface_initial_weight=args.inner_surface_initial_weight,
        multisurface_initial_step_scale=args.multisurface_initial_step_scale,
        multisurface_initial_step_maxiter=args.multisurface_initial_step_maxiter,
        topology_gate_fieldlines=args.topology_gate_fieldlines,
        topology_gate_tmax=args.topology_gate_tmax,
        topology_gate_tol=args.topology_gate_tol,
        topology_gate_survival_threshold=args.topology_gate_survival_threshold,
        topology_gate_penalty_scale=args.topology_gate_penalty_scale,
        hardware_search_mode=args.hardware_search_mode,
        hardware_search_soft_iterations=args.hardware_search_soft_iterations,
        curvature_traversal_band=args.curvature_traversal_band,
        curvature_traversal_eval_budget=args.curvature_traversal_eval_budget,
        topology_scorer_every=args.topology_scorer_every,
        topology_scorer_nfieldlines=args.topology_scorer_nfieldlines,
        topology_scorer_tmax=args.topology_scorer_tmax,
        confinement_objective_weight=args.confinement_objective_weight,
        confinement_surrogate_worst_k=args.confinement_surrogate_worst_k,
        confinement_surrogate_early_threshold=args.confinement_surrogate_early_threshold,
        confinement_surrogate_mean_weight=args.confinement_surrogate_mean_weight,
        confinement_surrogate_worst_weight=args.confinement_surrogate_worst_weight,
        confinement_surrogate_early_weight=args.confinement_surrogate_early_weight,
        alm_trust_radius_init=args.alm_trust_radius_init,
        alm_trust_radius_min=args.alm_trust_radius_min,
        alm_trust_radius_shrink=args.alm_trust_radius_shrink,
        alm_trust_radius_grow=args.alm_trust_radius_grow,
        alm_max_inner_attempts=args.alm_max_inner_attempts,
        alm_max_subproblem_continuations=args.alm_max_subproblem_continuations,
        alm_distance_smoothing=args.alm_distance_smoothing,
        alm_curvature_smoothing=args.alm_curvature_smoothing,
        warm_start_surface_stem=_resolved_optional_path_string(
            getattr(args, "warm_start_surface_stem", None)
        ),
        seed_regime=(
            None
            if getattr(args, "seed_regime", _DEFAULT_SINGLE_STAGE_SEED_REGIME)
            == _SINGLE_STAGE_SEED_REGIME_AUTO
            else args.seed_regime
        ),
    )


def build_run_identity_config(config):
    values = []
    for field, value in zip(fields(config), astuple(config)):
        if (
            field.name == "single_stage_banana_current_coordinate_scaling"
            and value == BANANA_CURRENT_COORDINATE_SCALING_NONE
        ):
            continue
        values.append(value)
    return "|".join(str(value) for value in values)


def validate_boozer_stage_refinement_args(
    args,
    constraint_weight,
    *,
    surface_mode_contract: SurfaceModeContract | None = None,
):
    if not args.boozer_stage_refinement:
        return
    resolved_contract = (
        resolve_surface_mode_contract(args, warn_on_legacy_mapping=False)
        if surface_mode_contract is None
        else surface_mode_contract
    )
    if args.constraint_method != "penalty":
        raise ValueError("--boozer-stage-refinement currently requires --constraint-method=penalty")
    if not surface_mode_supports_boozer_stage_refinement(resolved_contract):
        raise ValueError(
            "--boozer-stage-refinement currently requires "
            f"--surface-mode={SINGLE_SURFACE} (legacy --num-surfaces=1)"
        )
    if args.basin_hops > 0:
        raise ValueError("--boozer-stage-refinement is not supported with --basin-hops > 0")
    if args.boozer_stage != "initial":
        raise ValueError("--boozer-stage-refinement currently requires --boozer-stage=initial")
    if args.refinement_boozer_stage != "final":
        raise ValueError("--boozer-stage-refinement currently requires --refinement-boozer-stage=final")
    if constraint_weight is None:
        raise ValueError(
            "--boozer-stage-refinement currently requires least-squares Boozer initialization "
            "(--constraint-weight >= 0)"
        )
    if args.refinement_maxiter <= 0:
        raise ValueError("--refinement-maxiter must be positive when --boozer-stage-refinement is enabled")
    if args.refinement_chunk_maxiter <= 0:
        raise ValueError("--refinement-chunk-maxiter must be positive when --boozer-stage-refinement is enabled")
    if args.refinement_max_stalled_chunks <= 0:
        raise ValueError(
            "--refinement-max-stalled-chunks must be positive when --boozer-stage-refinement is enabled"
        )


def validate_surface_mode_constraint_args(
    args,
    *,
    surface_mode_contract: SurfaceModeContract,
):
    if args.constraint_method == "alm" and not surface_mode_supports_alm(
        surface_mode_contract
    ):
        raise ValueError(
            "--constraint-method=alm currently requires "
            f"--surface-mode={SINGLE_SURFACE} (legacy --num-surfaces=1)"
        )


def validate_single_stage_alm_formulation_args(args):
    if args.alm_formulation == "weighted_sum":
        return
    if args.single_stage_goal_mode == "frontier":
        raise ValueError(
            "--single-stage-goal-mode=frontier is not compatible with "
            "--alm-formulation=thresholded_physics because that ALM formulation assumes "
            "an upper-bounded Jiota penalty objective"
        )
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
        flag_name for flag_name, value in required_thresholds.items() if float(value) < 0.0
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


def evaluate_search_topology_gate(
    num_surfaces,
    outer_surface,
    bfield,
    *,
    surface_mode_contract: SurfaceModeContract | None = None,
):
    if TOPOLOGY_GATE_FIELDLINES <= 0:
        return disabled_topology_gate_status(
            TOPOLOGY_GATE_TMAX,
            TOPOLOGY_GATE_TOL,
            TOPOLOGY_GATE_SURVIVAL_THRESHOLD,
        )
    if surface_mode_contract is not None:
        if not surface_mode_supports_topology_gate(surface_mode_contract):
            return disabled_topology_gate_status(
                TOPOLOGY_GATE_TMAX,
                TOPOLOGY_GATE_TOL,
                TOPOLOGY_GATE_SURVIVAL_THRESHOLD,
            )
    elif num_surfaces <= 1:
        return disabled_topology_gate_status(
            TOPOLOGY_GATE_TMAX,
            TOPOLOGY_GATE_TOL,
            TOPOLOGY_GATE_SURVIVAL_THRESHOLD,
        )
    return safe_evaluate_topology_gate(
        outer_surface,
        bfield,
        TOPOLOGY_GATE_FIELDLINES,
        TOPOLOGY_GATE_TMAX,
        TOPOLOGY_GATE_TOL,
        TOPOLOGY_GATE_SURVIVAL_THRESHOLD,
    )


def skipped_topology_gate_status():
    return {
        "enabled": False,
        "evaluated": False,
        "success": None,
        "survived_lines": None,
        "survival_fraction": None,
        "first_exit_time": None,
        "first_exit_angle": None,
        "first_exit_reason": None,
        "stop_reason_counts": None,
        "transport_diagnostics": _topology_transport_diagnostics_not_evaluated(
            "topology_gate_not_evaluated"
        ),
    }


def final_topology_gate_for_results(
    init_only,
    num_surfaces,
    outer_surface,
    bfield,
    *,
    surface_mode_contract: SurfaceModeContract | None = None,
):
    if init_only:
        return skipped_topology_gate_status()
    status = evaluate_search_topology_gate(
        num_surfaces,
        outer_surface,
        bfield,
        surface_mode_contract=surface_mode_contract,
    )
    status["evaluated"] = True
    return status


def frontier_mode_enabled():
    return globals().get("SINGLE_STAGE_GOAL_MODE", "target") == "frontier"


def current_frontier_goal_mode_impl():
    if not frontier_mode_enabled():
        return "target"
    frontier_goal_config = current_frontier_goal_config()
    if frontier_goal_config is None:
        return FRONTIER_GOAL_MODE_IMPL
    if frontier_goal_config.scalarization_type == FRONTIER_SCALARIZATION_TYPE_ACHIEVEMENT:
        return "frontier_achievement_chebyshev_v1"
    if frontier_goal_config.scalarization_type == FRONTIER_SCALARIZATION_TYPE_EPSILON:
        return "frontier_epsilon_constraint_v1"
    return FRONTIER_GOAL_MODE_IMPL


def current_frontier_goal_config():
    return globals().get("FRONTIER_GOAL_CONFIG")


def require_frontier_goal_config(frontier_goal_config):
    if frontier_goal_config is None:
        raise ValueError("frontier goal mode requires frontier_goal_config")
    return frontier_goal_config


def measure_frontier_reference_metrics(stage, surface_data, coils):
    reference_terms = build_boozer_derived_objective_terms(stage, surface_data, coils)
    return (
        float(average_surface_objectives(reference_terms["nonQSs"]).J()),
        float(average_surface_objectives(reference_terms["brs"]).J()),
    )


def evaluate_frontier_trust_status(search_eval):
    frontier_goal_config = current_frontier_goal_config()
    return _evaluate_frontier_trust_status_impl(
        search_eval,
        enabled=frontier_mode_enabled(),
        threshold=(
            None
            if frontier_goal_config is None
            else frontier_goal_config.boozer_trust_threshold
        ),
    )


def evaluate_frontier_trust_penalty(search_eval):
    frontier_goal_config = current_frontier_goal_config()
    return _evaluate_frontier_trust_penalty_impl(
        search_eval,
        enabled=frontier_mode_enabled(),
        threshold=(
            None
            if frontier_goal_config is None
            else frontier_goal_config.boozer_trust_threshold
        ),
        penalty_scale=(
            None
            if frontier_goal_config is None
            else frontier_goal_config.boozer_trust_penalty_scale
        ),
    )


def annotate_frontier_search_eval(search_eval):
    frontier_goal_config = current_frontier_goal_config()
    return _annotate_frontier_search_eval_impl(
        search_eval,
        enabled=frontier_mode_enabled(),
        threshold=(
            None
            if frontier_goal_config is None
            else frontier_goal_config.boozer_trust_threshold
        ),
        penalty_scale=(
            None
            if frontier_goal_config is None
            else frontier_goal_config.boozer_trust_penalty_scale
        ),
    )


def preserved_incumbent_eligible(run_dict):
    surface_status = run_dict.get("surface_status")
    search_eval = run_dict.get("search_eval")
    if surface_status is None or not surface_status.get("success", False):
        return False
    if bool(run_dict.get("intersecting", False)):
        return False
    if search_eval is None or "total" not in search_eval:
        return False
    if search_eval.get("finite_eval_ok") is False:
        return False
    if frontier_mode_enabled() and not bool(search_eval.get("frontier_trust_ok", False)):
        return False
    return bool(np.isfinite(float(search_eval["total"])))


def topology_gate_allows_incumbent(run_dict):
    topology_status = run_dict.get("topology_gate_status")
    return not (
        topology_status is not None
        and bool(topology_status.get("enabled", False))
        and not bool(topology_status.get("success", False))
    )


def refinement_eligible_incumbent(run_dict):
    hardware_status = run_dict.get("accepted_hardware_status")
    return refinement_eligible_for_hardware_status(run_dict, hardware_status)


def refinement_eligible_for_hardware_status(run_dict, hardware_status):
    if hardware_status is None or not hardware_status.get("success", False):
        return False
    if not topology_gate_allows_incumbent(run_dict):
        return False
    return preserved_incumbent_eligible(run_dict)


def repair_eligible_incumbent(run_dict):
    if not topology_gate_allows_incumbent(run_dict):
        return False
    return preserved_incumbent_eligible(run_dict)


def hardware_violation_score(hardware_status):
    if hardware_status is None:
        return float("inf")
    if bool(hardware_status.get("success", False)):
        return 0.0
    violation_score = float(
        sum(
            float(value)
            for value in _hardware_violation_ratios(hardware_status).values()
        )
    )
    if violation_score > 0.0:
        return violation_score
    violations = hardware_status.get("violations")
    if violations is None:
        return 1.0
    return float(max(len(violations), 1))


def repair_progress_state(run_dict):
    if not repair_eligible_incumbent(run_dict):
        return (1, float("inf"))
    return (
        0,
        float(hardware_violation_score(run_dict.get("accepted_hardware_status"))),
    )


def accepted_search_metric(run_dict):
    search_eval = run_dict["search_eval"]
    if frontier_mode_enabled():
        return float(search_eval.get("frontier_rank_total", search_eval["total"]))
    return float(search_eval["total"])


def snapshot_diagnostic_incumbent_state(run_dict):
    search_eval = diagnostic_search_eval_for_current_state(run_dict)
    if search_eval is run_dict["search_eval"]:
        return snapshot_single_stage_incumbent_state(run_dict)
    compact_search_eval = run_dict["search_eval"]
    run_dict["search_eval"] = search_eval
    try:
        return snapshot_single_stage_incumbent_state(run_dict)
    finally:
        run_dict["search_eval"] = compact_search_eval


def normalize_diagnostic_incumbent_for_stage(
    run_dict,
    incumbent,
    incumbent_stage,
    current_stage,
    rebuild_stage_objective_bundle,
):
    if incumbent is None or search_eval_has_diagnostics(incumbent.search_eval):
        return incumbent
    current_state = snapshot_single_stage_incumbent_state(run_dict)
    restore_single_stage_incumbent_state(run_dict, incumbent)
    try:
        rebuild_stage_objective_bundle(incumbent_stage)
        refresh_accepted_search_state(run_dict, incumbent_stage)
        return snapshot_diagnostic_incumbent_state(run_dict)
    finally:
        restore_single_stage_incumbent_state(run_dict, current_state)
        rebuild_stage_objective_bundle(current_stage)
        refresh_accepted_search_state(run_dict, current_stage)


def maybe_update_best_accepted_incumbent(run_dict, incumbent_stage):
    if not preserved_incumbent_eligible(run_dict):
        return False
    metric = accepted_search_metric(run_dict)
    best_metric = run_dict.get("best_accepted_metric")
    if best_metric is not None and metric >= best_metric:
        return False
    run_dict["best_accepted_incumbent"] = snapshot_diagnostic_incumbent_state(run_dict)
    run_dict["best_accepted_metric"] = metric
    run_dict["best_accepted_stage"] = str(incumbent_stage)
    return True


def maybe_update_best_feasible_incumbent(run_dict, incumbent_stage):
    if not refinement_eligible_incumbent(run_dict):
        return False
    metric = accepted_search_metric(run_dict)
    best_metric = run_dict.get("best_feasible_metric")
    if best_metric is not None and metric >= best_metric:
        return False
    run_dict["best_feasible_incumbent"] = snapshot_diagnostic_incumbent_state(run_dict)
    run_dict["best_feasible_metric"] = metric
    run_dict["best_feasible_stage"] = str(incumbent_stage)
    return True


def frontier_goal_mode_warning_message(frontier_goal_config):
    if frontier_goal_config.scalarization_type == FRONTIER_SCALARIZATION_TYPE_ACHIEVEMENT:
        return (
            "INFO: --single-stage-goal-mode=frontier uses an achievement/Chebyshev "
            "tradeoff score: iota, volume, QA error, and Boozer residual are compared "
            f"against the lane reference point (iota_ref={frontier_goal_config.iota_reference:.6f}, "
            f"volume_ref={frontier_goal_config.volume_reference:.6f}, "
            f"qa_ref={frontier_goal_config.qs_reference:.6e}, "
            f"boozer_ref={frontier_goal_config.boozer_reference:.6e}), combined with "
            f"rho={frontier_goal_config.chebyshev_rho:.6e}, and Boozer residuals above "
            f"{frontier_goal_config.boozer_trust_threshold:.6e} still incur the frontier "
            "trust penalty during search."
        )
    if frontier_goal_config.scalarization_type == FRONTIER_SCALARIZATION_TYPE_EPSILON:
        return (
            "INFO: --single-stage-goal-mode=frontier uses an epsilon-constrained "
            "tradeoff score: the seed-relative frontier objective stays active, but "
            "QA and/or Boozer residual threshold violations add smooth search penalties "
            f"(qa_max={frontier_goal_config.epsilon_constraint_qa_max}, "
            f"boozer_max={frontier_goal_config.epsilon_constraint_boozer_max}) while "
            f"Boozer residuals above {frontier_goal_config.boozer_trust_threshold:.6e} "
            "still incur the frontier trust penalty during search."
        )
    return (
        "INFO: --single-stage-goal-mode=frontier uses a normalized tradeoff score: "
        "QA and Boozer residual are normalized to the seed, iota and volume use bounded "
        f"improvement rewards referenced to the seed (iota_ref={frontier_goal_config.iota_reference:.6f}, "
        f"volume_ref={frontier_goal_config.volume_reference:.6f}), and Boozer residuals above "
        f"{frontier_goal_config.boozer_trust_threshold:.6e} incur a smooth threshold-relative trust penalty "
        "during search and still fail final frontier certification. "
        "The legacy --res-weight and --iotas-weight inputs are rescaled relative to their historical "
        "defaults so matched target/frontier runs stay in the same rough objective range."
    )


def apply_frontier_scalarization_override(objective_eval, *, alm_formulation="weighted_sum"):
    frontier_goal_config = current_frontier_goal_config()
    surface_iota_terms_local = globals().get("surface_iota_terms")
    surface_iota_term = (
        None
        if not surface_iota_terms_local
        else surface_iota_terms_local[-1]
    )
    return _apply_frontier_scalarization_override_impl(
        objective_eval,
        enabled=frontier_mode_enabled(),
        frontier_goal_config=frontier_goal_config,
        surface_iota_term=surface_iota_term,
        surface_volume_term=globals().get("surface_volume_term"),
        effective_res_weight=globals().get("EFFECTIVE_RES_WEIGHT", 0.0),
        effective_iotas_weight=globals().get("EFFECTIVE_IOTAS_WEIGHT", 0.0),
        effective_volume_weight=globals().get("EFFECTIVE_VOLUME_WEIGHT", 0.0),
        length_weight=globals().get("LENGTH_WEIGHT", 0.0),
        cc_weight=globals().get("CC_WEIGHT", 0.0),
        cs_weight=globals().get("CS_WEIGHT", 0.0),
        curvature_weight=globals().get("CURVATURE_WEIGHT", 0.0),
        surf_dist_weight=globals().get("SURF_DIST_WEIGHT", 0.0),
        objective_optimizable=globals().get("JF"),
        alm_formulation=alm_formulation,
        alm_multipliers=globals().get("ALM_MULTIPLIERS"),
        alm_penalty=globals().get("ALM_PENALTY"),
    )


def resolve_single_stage_goal_objective_terms(
    *,
    goal_mode,
    frontier_goal_config,
    JnonQSRatio,
    JBoozerResidual,
    RES_WEIGHT,
    IOTAS_WEIGHT,
):
    if goal_mode == "target":
        return {
            "JnonQSRatioObjective": JnonQSRatio,
            "JBoozerResidualObjective": JBoozerResidual,
            "effective_res_weight": RES_WEIGHT,
            "effective_iotas_weight": IOTAS_WEIGHT,
            "effective_volume_weight": 0.0,
        }
    if goal_mode == "frontier":
        frontier_goal_config = require_frontier_goal_config(frontier_goal_config)
        return {
            "JnonQSRatioObjective": (1.0 / frontier_goal_config.qs_reference) * JnonQSRatio,
            "JBoozerResidualObjective": (1.0 / frontier_goal_config.boozer_reference)
            * JBoozerResidual,
            "effective_res_weight": frontier_goal_config.effective_boozer_weight,
            "effective_iotas_weight": frontier_goal_config.effective_iota_weight,
            "effective_volume_weight": frontier_goal_config.effective_volume_weight,
        }
    raise ValueError(f"Unsupported single-stage goal mode {goal_mode!r}")


def resolve_current_surface_objective_terms(RES_WEIGHT, IOTAS_WEIGHT):
    """The *RES_WEIGHT* / *IOTAS_WEIGHT* fallbacks are only reached in test
    harnesses that call the evaluation wrappers without initialising the full
    objective bundle; in production the effective weights always come from the
    globals set by ``apply_single_stage_objective_bundle``.
    """
    objective_nonqs = JnonQSRatioObjective
    objective_boozer = JBoozerResidualObjective
    return {
        "JNonQSObjective": objective_nonqs,
        "JBoozerObjective": objective_boozer,
        "effective_res_weight": RES_WEIGHT if objective_boozer is None else EFFECTIVE_RES_WEIGHT,
        "effective_iotas_weight": (
            IOTAS_WEIGHT if objective_nonqs is None else EFFECTIVE_IOTAS_WEIGHT
        ),
        "JVolume": JVolume,
        "effective_volume_weight": 0.0 if JVolume is None else EFFECTIVE_VOLUME_WEIGHT,
    }


def normalize_optimizer_termination_message(
    message,
    *,
    success,
    status=None,
    invalid_state_rejects_total=None,
    surface_solve_rejects=None,
    hardware_rejects=None,
    topology_gate_rejects=None,
):
    if message is None:
        text = ""
    elif isinstance(message, (bytes, bytearray)):
        text = bytes(message).decode("utf-8", errors="replace")
    else:
        text = str(message)
    if success:
        return text
    if text.strip() not in {"ABNORMAL:", "ABNORMAL"}:
        return text

    details = ["empty SciPy L-BFGS-B task"]
    if status is not None:
        details.append(f"status={int(status)}")
    if invalid_state_rejects_total is not None:
        details.append(f"invalid_state_rejects={int(invalid_state_rejects_total)}")
    if surface_solve_rejects is not None:
        details.append(f"surface_solve_rejects={int(surface_solve_rejects)}")
    if hardware_rejects is not None:
        details.append(f"hardware_rejects={int(hardware_rejects)}")
    if topology_gate_rejects is not None:
        details.append(f"topology_gate_rejects={int(topology_gate_rejects)}")
    return f"ABNORMAL: {'; '.join(details)}"


def build_single_stage_iota_objective(
    surface_iota_term,
    iota_target,
    *,
    goal_mode,
    frontier_goal_config=None,
):
    """Construct the iota term used in the single-stage scalar objective.

    In ``target`` mode this returns ``QuadraticPenalty(iota, iota_target)``,
    whose ``J`` is ``0.5 * (iota - iota_target)**2`` (typically O(1e-3) near the
    target). In ``frontier`` mode this returns a smooth bounded reward that
    measures improvement relative to the fixed seed/reference iota. Minimizing
    this term therefore rewards higher iota without introducing an explicit
    outer target or an unbounded linear ``-iota`` direction. ``iota_target`` is
    unused in frontier mode and is accepted only for caller-API symmetry.
    """
    if goal_mode == "target":
        return QuadraticPenalty(surface_iota_term, iota_target)
    if goal_mode == "frontier":
        frontier_goal_config = require_frontier_goal_config(frontier_goal_config)
        return BoundedImprovementReward(
            surface_iota_term,
            frontier_goal_config.iota_reference,
            frontier_goal_config.iota_scale,
        )
    raise ValueError(f"Unsupported single-stage goal mode {goal_mode!r}")


def build_single_stage_volume_objective(surface_volume_term, *, goal_mode, frontier_goal_config=None):
    if goal_mode == "target":
        return None
    if goal_mode == "frontier":
        frontier_goal_config = require_frontier_goal_config(frontier_goal_config)
        return BoundedImprovementReward(
            surface_volume_term,
            frontier_goal_config.volume_reference,
            frontier_goal_config.volume_scale,
        )
    raise ValueError(f"Unsupported single-stage goal mode {goal_mode!r}")


def build_best_feasible_results_summary(
    run_dict,
    curve_curve_distance_obj,
    curve_surface_distance_obj,
    surface_surface_distance_obj,
    banana_curve,
    curvelength_obj,
    cc_dist,
    cs_dist,
    ss_dist,
    curvature_threshold,
    length_target,
    tf_current_A,
    banana_current_state: SingleStageBananaCurrentState | None = None,
    banana_current_max_A=None,
    outer_surface=None,
    vessel_surface=None,
    banana_coils=None,
):
    resolved_banana_current_state = coerce_single_stage_banana_current_state(
        banana_current_state,
        banana_coils=banana_coils,
    )
    incumbent = run_dict.get("best_feasible_incumbent")
    if (
        incumbent is None
        or run_dict.get("J") is None
        or run_dict.get("dJ") is None
        or run_dict.get("search_eval") is None
    ):
        return {
            "BEST_FEASIBLE_AVAILABLE": False,
            "BEST_FEASIBLE_STAGE": None,
            "BEST_FEASIBLE_SEARCH_OBJECTIVE_J": None,
            "BEST_FEASIBLE_BASE_OBJECTIVE_J": None,
            "BEST_FEASIBLE_QA_OBJECTIVE": None,
            "BEST_FEASIBLE_BOOZER_OBJECTIVE": None,
            "BEST_FEASIBLE_FRONTIER_RANK_OBJECTIVE_J": None,
            "BEST_FEASIBLE_FRONTIER_TRUST_OK": None,
            "BEST_FEASIBLE_FINAL_IOTA": None,
            "BEST_FEASIBLE_FINAL_VOLUME": None,
            **build_single_stage_banana_current_payload_fields(
                None,
                prefix="BEST_FEASIBLE_",
            ),
            **build_hardware_constraint_artifact_payload_fields(
                None,
                prefix="BEST_FEASIBLE_",
            ),
            "BEST_FEASIBLE_SURFACE_STACK_OK": None,
            "BEST_FEASIBLE_SELF_INTERSECTING": None,
            "BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_SUCCESS": None,
            "BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_STATE": None,
            "BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_ERROR": None,
            "BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_DIAGNOSTICS": None,
            "BEST_FEASIBLE_FINAL_TOPOLOGY_TRANSPORT_DIAGNOSTICS": None,
        }

    current_state = snapshot_single_stage_incumbent_state(run_dict)
    restore_single_stage_incumbent_state(run_dict, incumbent)
    try:
        hardware_snapshot = evaluate_single_stage_hardware_snapshot(
            curve_curve_distance_obj,
            cc_dist,
            curve_surface_distance_obj,
            cs_dist,
            surface_surface_distance_obj,
            run_dict["surface_status"],
            ss_dist,
            banana_curve,
            curvature_threshold,
            outer_surface,
            vessel_surface,
            coil_length=float(curvelength_obj.J()),
            length_target=length_target,
            tf_current_A=tf_current_A,
            tf_current_limit_A=TF_CURRENT_HARD_LIMIT_A,
            banana_current_A=(
                None
                if resolved_banana_current_state is None
                else resolved_banana_current_state.control_current_A()
            ),
            banana_current_max_A=banana_current_max_A,
        )
        surface_status = run_dict["surface_status"]
        search_eval = run_dict["search_eval"]
        topology_status = run_dict["topology_gate_status"]
        return {
            "BEST_FEASIBLE_AVAILABLE": True,
            "BEST_FEASIBLE_STAGE": run_dict.get("best_feasible_stage"),
            "BEST_FEASIBLE_SEARCH_OBJECTIVE_J": float(search_eval["total"]),
            "BEST_FEASIBLE_BASE_OBJECTIVE_J": float(search_eval.get("physics_total", search_eval["total"])),
            "BEST_FEASIBLE_QA_OBJECTIVE": float(search_eval.get("J_QS")) if search_eval.get("J_QS") is not None else None,
            "BEST_FEASIBLE_BOOZER_OBJECTIVE": (
                float(search_eval.get("J_Boozer")) if search_eval.get("J_Boozer") is not None else None
            ),
            "BEST_FEASIBLE_FRONTIER_RANK_OBJECTIVE_J": (
                float(search_eval.get("frontier_rank_total"))
                if search_eval.get("frontier_rank_total") is not None
                else None
            ),
            "BEST_FEASIBLE_FRONTIER_TRUST_OK": search_eval.get("frontier_trust_ok"),
            "BEST_FEASIBLE_FINAL_IOTA": float(surface_status["iotas"][-1]),
            "BEST_FEASIBLE_FINAL_VOLUME": float(surface_status["volumes"][-1]),
            **build_single_stage_banana_current_payload_fields(
                resolved_banana_current_state,
                prefix="BEST_FEASIBLE_",
            ),
            **build_hardware_constraint_artifact_payload_fields(
                hardware_snapshot,
                prefix="BEST_FEASIBLE_",
            ),
            "BEST_FEASIBLE_SURFACE_STACK_OK": bool(surface_status["success"]),
            "BEST_FEASIBLE_SELF_INTERSECTING": bool(any(surface_status["self_intersections"])),
            "BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_SUCCESS": bool(topology_status["success"]),
            "BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_STATE": topology_status.get("state"),
            "BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_ERROR": topology_status.get("evaluation_error"),
            "BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_DIAGNOSTICS": build_topology_gate_diagnostics(
                topology_status,
                artifact_role="best_feasible_final_topology_gate",
            ),
            "BEST_FEASIBLE_FINAL_TOPOLOGY_TRANSPORT_DIAGNOSTICS": topology_status.get(
                "transport_diagnostics"
            ),
        }
    finally:
        restore_single_stage_incumbent_state(run_dict, current_state)


def build_boozer_derived_objective_terms(stage, surface_data, coils):
    boozer_surfaces = [entry["boozer_surface"] for entry in surface_data]
    boozer_objective_biot_savarts = [BiotSavart(coils) for _surface in boozer_surfaces]
    surface_iota_terms = [Iotas(surface) for surface in boozer_surfaces]
    nonQSs = [
        NonQuasiSymmetricRatio(surface, boozer_objective_biot_savarts[index])
        for index, surface in enumerate(boozer_surfaces)
    ]
    boozer_residual_cls = BoozerResidualExact if stage == "final" else BoozerResidual
    brs = [
        boozer_residual_cls(surface, boozer_objective_biot_savarts[index])
        for index, surface in enumerate(boozer_surfaces)
    ]
    return {
        "boozer_objective_biot_savarts": boozer_objective_biot_savarts,
        "surface_iota_terms": surface_iota_terms,
        "nonQSs": nonQSs,
        "brs": brs,
    }


def build_single_stage_objective_bundle(
    stage,
    surface_data,
    coils,
    curves,
    banana_curves,
    iota_target,
    RES_WEIGHT,
    IOTAS_WEIGHT,
    LENGTH_WEIGHT,
    CC_WEIGHT,
    CC_DIST,
    CS_WEIGHT,
    CS_DIST,
    CURVATURE_WEIGHT,
    CURVATURE_THRESHOLD,
    length_target=None,
    SURF_DIST_WEIGHT=0.0,
    vessel_surface=None,
    vessel_gap_threshold=0.0,
    goal_mode="target",
    frontier_goal_config=None,
):
    boozer_terms = build_boozer_derived_objective_terms(stage, surface_data, coils)
    surface_iota_terms = boozer_terms["surface_iota_terms"]
    nonQSs = boozer_terms["nonQSs"]
    brs = boozer_terms["brs"]

    curvelength = CurveLength(banana_curves[0])
    if length_target is None:
        length_target = COIL_LENGTH_TARGET_M
    outer_surface = surface_data[-1]["boozer_surface"].surface
    surface_volume_term = Volume(outer_surface)
    Jiota = build_single_stage_iota_objective(
        surface_iota_terms[-1],
        iota_target,
        goal_mode=goal_mode,
        frontier_goal_config=frontier_goal_config,
    )
    JVolume = build_single_stage_volume_objective(
        surface_volume_term,
        goal_mode=goal_mode,
        frontier_goal_config=frontier_goal_config,
    )
    JnonQSRatio = average_surface_objectives(nonQSs)
    JBoozerResidual = average_surface_objectives(brs)
    goal_objective_terms = resolve_single_stage_goal_objective_terms(
        goal_mode=goal_mode,
        frontier_goal_config=frontier_goal_config,
        JnonQSRatio=JnonQSRatio,
        JBoozerResidual=JBoozerResidual,
        RES_WEIGHT=RES_WEIGHT,
        IOTAS_WEIGHT=IOTAS_WEIGHT,
    )
    JCurveLength = QuadraticPenalty(curvelength, length_target, "max")
    JCurveCurve = CurveCurveDistance(curves, CC_DIST)
    JCurveSurface = CurveSurfaceDistance(curves, outer_surface, CS_DIST)
    JSurfSurf = (
        SurfaceSurfaceDistance(outer_surface, vessel_surface, vessel_gap_threshold)
        if len(surface_data) == 1 and vessel_surface is not None
        else None
    )
    JCurvature = LpCurveCurvature(banana_curves[0], 2, CURVATURE_THRESHOLD)
    JF = build_total_objective(
        goal_objective_terms["JnonQSRatioObjective"],
        goal_objective_terms["effective_res_weight"],
        goal_objective_terms["JBoozerResidualObjective"],
        goal_objective_terms["effective_iotas_weight"],
        Jiota,
        goal_objective_terms["effective_volume_weight"],
        JVolume,
        LENGTH_WEIGHT,
        JCurveLength,
        CC_WEIGHT,
        JCurveCurve,
        CS_WEIGHT,
        JCurveSurface,
        CURVATURE_WEIGHT,
        JCurvature,
        SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
        JSurfSurf=JSurfSurf,
    )
    return {
        "surface_iota_terms": surface_iota_terms,
        "boozer_objective_biot_savarts": boozer_terms[
            "boozer_objective_biot_savarts"
        ],
        "nonQSs": nonQSs,
        "brs": brs,
        "curvelength": curvelength,
        "length_target": length_target,
        "Jiota": Jiota,
        "surface_volume_term": surface_volume_term,
        "JVolume": JVolume,
        "JnonQSRatio": JnonQSRatio,
        "JnonQSRatioObjective": goal_objective_terms["JnonQSRatioObjective"],
        "JBoozerResidual": JBoozerResidual,
        "JBoozerResidualObjective": goal_objective_terms["JBoozerResidualObjective"],
        "effective_res_weight": goal_objective_terms["effective_res_weight"],
        "effective_iotas_weight": goal_objective_terms["effective_iotas_weight"],
        "effective_volume_weight": goal_objective_terms["effective_volume_weight"],
        "frontier_goal_config": frontier_goal_config,
        "JCurveLength": JCurveLength,
        "JCurveCurve": JCurveCurve,
        "JCurveSurface": JCurveSurface,
        "JSurfSurf": JSurfSurf,
        "JCurvature": JCurvature,
        "JF": JF,
    }


def apply_single_stage_objective_bundle(objective_bundle):
    global surface_iota_terms
    global nonQSs
    global brs
    global curvelength
    global surface_volume_term
    global Jiota
    global JVolume
    global JnonQSRatio
    global JnonQSRatioObjective
    global JBoozerResidual
    global JBoozerResidualObjective
    global EFFECTIVE_RES_WEIGHT
    global EFFECTIVE_IOTAS_WEIGHT
    global EFFECTIVE_VOLUME_WEIGHT
    global FRONTIER_GOAL_CONFIG
    global JCurveLength
    global JCurveCurve
    global JCurveSurface
    global JSurfSurf
    global JCurvature
    global JF

    surface_iota_terms = objective_bundle["surface_iota_terms"]
    nonQSs = objective_bundle["nonQSs"]
    brs = objective_bundle["brs"]
    curvelength = objective_bundle["curvelength"]
    surface_volume_term = objective_bundle["surface_volume_term"]
    Jiota = objective_bundle["Jiota"]
    JVolume = objective_bundle["JVolume"]
    JnonQSRatio = objective_bundle["JnonQSRatio"]
    JnonQSRatioObjective = objective_bundle["JnonQSRatioObjective"]
    JBoozerResidual = objective_bundle["JBoozerResidual"]
    JBoozerResidualObjective = objective_bundle["JBoozerResidualObjective"]
    EFFECTIVE_RES_WEIGHT = objective_bundle["effective_res_weight"]
    EFFECTIVE_IOTAS_WEIGHT = objective_bundle["effective_iotas_weight"]
    EFFECTIVE_VOLUME_WEIGHT = objective_bundle["effective_volume_weight"]
    FRONTIER_GOAL_CONFIG = objective_bundle["frontier_goal_config"]
    JCurveLength = objective_bundle["JCurveLength"]
    JCurveCurve = objective_bundle["JCurveCurve"]
    JCurveSurface = objective_bundle["JCurveSurface"]
    JSurfSurf = objective_bundle["JSurfSurf"]
    JCurvature = objective_bundle["JCurvature"]
    JF = objective_bundle["JF"]


def refresh_accepted_search_state(run_dict, accepted_stage):
    JF.x = run_dict["accepted_x"].copy()
    restore_surface_states(surface_data, run_dict["surface_state"])
    current_search_weights = build_surface_search_weights(
        len(surface_data),
        run_dict["accepted_iterations"],
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
    )
    current_search_eval = evaluate_search_objective(current_search_weights)
    run_dict["J"] = current_search_eval["total"]
    run_dict["dJ"] = current_search_eval["grad"].copy()
    run_dict["search_eval"] = current_search_eval
    run_dict["frontier_trust_status"] = evaluate_frontier_trust_status(current_search_eval)
    run_dict["accepted_boozer_stage"] = accepted_stage
    run_dict["x_prev"] = run_dict["accepted_x"].copy()
    run_dict["trial_hardware_status"] = None
    run_dict.pop("last_successful_eval", None)
    run_dict.pop("last_successful_eval_weights", None)
    return float(current_search_eval["total"])


def evaluate_incumbent_metric_for_stage(
    run_dict,
    incumbent,
    comparison_stage,
    rebuild_stage_objective_bundle,
):
    restore_single_stage_incumbent_state(run_dict, incumbent)
    rebuild_stage_objective_bundle(comparison_stage)
    return refresh_accepted_search_state(run_dict, comparison_stage)


def restore_incumbent_for_stage(
    run_dict,
    incumbent,
    incumbent_stage,
    rebuild_stage_objective_bundle,
):
    restore_single_stage_incumbent_state(run_dict, incumbent)
    objective_bundle = rebuild_stage_objective_bundle(incumbent_stage)
    refresh_accepted_search_state(run_dict, incumbent_stage)
    return objective_bundle


def refinement_improves_phase1_metric(
    phase1_metric,
    phase1_stage,
    run_dict,
    refinement_incumbent,
    rebuild_stage_objective_bundle,
):
    if refinement_incumbent is None:
        return None, False
    refinement_metric = evaluate_incumbent_metric_for_stage(
        run_dict,
        refinement_incumbent,
        phase1_stage,
        rebuild_stage_objective_bundle,
    )
    return refinement_metric, refinement_metric <= phase1_metric


def reported_boozer_stage(requested_stage, final_source_stage):
    return str(requested_stage if final_source_stage is None else final_source_stage)


def refinement_chunk_improves_metric(reference_metric, candidate_metric):
    return candidate_metric is not None and candidate_metric < reference_metric


def run_refinement_chunk(
    run_dict,
    seed_incumbent,
    refinement_stage,
    rebuild_stage_objective_bundle,
    refinement_chunk_maxiter,
    maxcor,
    ftol,
    gtol,
):
    restore_incumbent_for_stage(
        run_dict,
        seed_incumbent,
        refinement_stage,
        rebuild_stage_objective_bundle,
    )
    run_dict["best_feasible_incumbent"] = None
    run_dict["best_feasible_metric"] = None
    run_dict["best_feasible_stage"] = None
    maybe_update_best_feasible_incumbent(run_dict, refinement_stage)
    bounds = current_optimizer_bounds()
    with temporary_run_dict_value(run_dict, "active_optimizer_bounds", bounds):
        chunk_result = minimize(
            fun,
            run_dict["accepted_x"].copy(),
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            callback=callback,
            options={
                "maxiter": refinement_chunk_maxiter,
                "maxcor": maxcor,
                "ftol": ftol,
                "gtol": gtol,
            },
        )
    return chunk_result, run_dict.get("best_feasible_incumbent")


def run_chunked_refinement(
    run_dict,
    phase1_incumbent,
    phase1_metric,
    phase1_stage,
    refinement_stage,
    rebuild_stage_objective_bundle,
    refinement_maxiter,
    refinement_chunk_maxiter,
    refinement_max_stalled_chunks,
    maxcor,
    ftol,
    gtol,
):
    total_iterations = 0
    chunk_count = 0
    stalled_chunks = 0
    best_metric = phase1_metric
    current_seed_incumbent = phase1_incumbent
    best_refinement_incumbent = None
    last_chunk_success = False
    last_chunk_message = None
    abort_reason = None

    while total_iterations < refinement_maxiter:
        chunk_budget = min(refinement_chunk_maxiter, refinement_maxiter - total_iterations)
        chunk_result, chunk_incumbent = run_refinement_chunk(
            run_dict,
            current_seed_incumbent,
            refinement_stage,
            rebuild_stage_objective_bundle,
            chunk_budget,
            maxcor,
            ftol,
            gtol,
        )
        chunk_count += 1
        total_iterations += int(chunk_result.nit)
        last_chunk_success = bool(chunk_result.success)
        last_chunk_message = str(chunk_result.message)

        chunk_metric, _ = refinement_improves_phase1_metric(
            best_metric,
            phase1_stage,
            run_dict,
            chunk_incumbent,
            rebuild_stage_objective_bundle,
        )
        if refinement_chunk_improves_metric(best_metric, chunk_metric):
            best_metric = float(chunk_metric)
            best_refinement_incumbent = chunk_incumbent
            current_seed_incumbent = chunk_incumbent
            stalled_chunks = 0
            if last_chunk_success:
                abort_reason = "converged_after_improvement"
                break
            continue

        stalled_chunks += 1
        if total_iterations >= refinement_maxiter:
            abort_reason = (
                "budget_exhausted_after_improvement"
                if best_refinement_incumbent is not None
                else "budget_exhausted_without_improvement"
            )
            break
        if last_chunk_success:
            abort_reason = (
                "converged_without_additional_improvement"
                if best_refinement_incumbent is not None
                else "converged_without_improvement"
            )
            break
        if stalled_chunks >= refinement_max_stalled_chunks:
            abort_reason = (
                "stalled_after_improvement"
                if best_refinement_incumbent is not None
                else "stalled_without_improvement"
            )
            break

    if abort_reason is None and total_iterations >= refinement_maxiter:
        abort_reason = (
            "budget_exhausted_after_improvement"
            if best_refinement_incumbent is not None
            else "budget_exhausted_without_improvement"
        )

    termination_message = last_chunk_message
    if abort_reason is not None:
        termination_message = (
            abort_reason
            if termination_message is None
            else f"{termination_message}; {abort_reason}"
        )
    return {
        "best_incumbent": best_refinement_incumbent,
        "best_metric": None if best_refinement_incumbent is None else best_metric,
        "iterations": total_iterations,
        "chunks": chunk_count,
        "termination_message": termination_message,
        "abort_reason": abort_reason,
        "success": last_chunk_success,
    }


def summarize_refinement_result(refinement_result, total_iterations, accepted_x):
    termination_message = refinement_result["termination_message"]
    optimizer_success = bool(refinement_result["success"])
    result = SimpleNamespace(
        x=accepted_x.copy(),
        nit=total_iterations,
        message=termination_message,
        success=optimizer_success,
    )
    return termination_message, optimizer_success, result


def current_optimizer_bounds():
    return build_scipy_bounds(JF.lower_bounds, JF.upper_bounds)


def resolve_single_stage_seed_regime(
    requested_seed_regime,
    run_dict,
    *,
    constraint_method,
    num_surfaces,
    basin_hops,
    init_only,
):
    """Resolve the solver-side seed regime.

    Keep this logic aligned with `_resolve_single_stage_seed_regime()` in
    `autoresearch/scripts/run_one.py`; the two repos intentionally share this
    routing contract.
    """
    requested_regime = str(
        requested_seed_regime or _DEFAULT_SINGLE_STAGE_SEED_REGIME
    )
    if constraint_method != "penalty":
        return _SINGLE_STAGE_SEED_REGIME_GLOBAL_SEARCH
    if init_only or basin_hops > 0 or num_surfaces != 1:
        return _SINGLE_STAGE_SEED_REGIME_GLOBAL_SEARCH
    if requested_regime != _SINGLE_STAGE_SEED_REGIME_AUTO:
        return requested_regime
    if refinement_eligible_incumbent(run_dict):
        return _SINGLE_STAGE_SEED_REGIME_PRESERVE_FIRST
    if preserved_incumbent_eligible(run_dict):
        return _SINGLE_STAGE_SEED_REGIME_BRIDGE_ONLY
    return _SINGLE_STAGE_SEED_REGIME_REPAIR_FIRST


def penalty_feasible_start_local_preservation_enabled(
    run_dict,
    *,
    constraint_method,
    num_surfaces,
    basin_hops,
    init_only,
):
    if constraint_method != "penalty":
        return False
    if init_only or basin_hops > 0 or num_surfaces != 1:
        return False
    if int(run_dict.get("accepted_iterations", 0)) != 0:
        return False
    return refinement_eligible_incumbent(run_dict)


def evaluate_total_objective(
    surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    JCurveCurve,
    CC_WEIGHT,
    JCurveSurface,
    CS_WEIGHT,
    JCurvature,
    CURVATURE_WEIGHT,
    JSurfSurf=None,
    SURF_DIST_WEIGHT=0.0,
    include_diagnostics=True,
):
    objective_terms = resolve_current_surface_objective_terms(RES_WEIGHT, IOTAS_WEIGHT)
    return apply_frontier_scalarization_override(
        _evaluate_total_objective_impl(
            surface_weights,
            nonQSs,
            brs,
            objective_terms["effective_res_weight"],
            Jiota,
            objective_terms["effective_iotas_weight"],
            JCurveLength,
            LENGTH_WEIGHT,
            JCurveCurve,
            CC_WEIGHT,
            JCurveSurface,
            CS_WEIGHT,
            JCurvature,
            CURVATURE_WEIGHT,
            JSurfSurf=JSurfSurf,
            SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
            JNonQSObjective=objective_terms["JNonQSObjective"],
            JBoozerObjective=objective_terms["JBoozerObjective"],
            JVolume=objective_terms["JVolume"],
            VOLUME_WEIGHT=objective_terms["effective_volume_weight"],
            objective_optimizable=globals().get("JF"),
            include_diagnostics=include_diagnostics,
        ),
        alm_formulation="weighted_sum",
    )


def evaluate_base_objective(
    surface_weights,
    nonQSs,
    brs,
    RES_WEIGHT,
    Jiota,
    IOTAS_WEIGHT,
    JCurveLength,
    LENGTH_WEIGHT,
    *,
    include_diagnostics=True,
):
    objective_terms = resolve_current_surface_objective_terms(RES_WEIGHT, IOTAS_WEIGHT)
    return _evaluate_base_objective_impl(
        surface_weights,
        nonQSs,
        brs,
        objective_terms["effective_res_weight"],
        Jiota,
        objective_terms["effective_iotas_weight"],
        objective_terms["JVolume"],
        objective_terms["effective_volume_weight"],
        JCurveLength,
        LENGTH_WEIGHT,
        alm_formulation=(
            ALM_FORMULATION if CONSTRAINT_METHOD == "alm" else "weighted_sum"
        ),
        JNonQSObjective=objective_terms["JNonQSObjective"],
        JBoozerObjective=objective_terms["JBoozerObjective"],
        include_diagnostics=include_diagnostics,
    )


def evaluate_alm_objective(
    surface_weights,
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
    multipliers,
    penalty,
    JSurfSurf=None,
    include_diagnostics=True,
):
    objective_terms = resolve_current_surface_objective_terms(RES_WEIGHT, IOTAS_WEIGHT)
    return apply_frontier_scalarization_override(
        _evaluate_alm_objective_impl(
            surface_weights,
            nonQSs,
            brs,
            objective_terms["effective_res_weight"],
            Jiota,
            objective_terms["effective_iotas_weight"],
            objective_terms["JVolume"],
            objective_terms["effective_volume_weight"],
            JCurveLength,
            LENGTH_WEIGHT,
            JCurveCurve,
            JCurveSurface,
            JCurvature,
            multipliers,
            penalty,
            objective_optimizable=JF,
            curves=curves,
            curve_curve_min_distance=CC_DIST,
            outer_surface=outer_surface_data["boozer_surface"].surface,
            curve_surface_min_distance=CS_DIST,
            banana_curve=banana_curve,
            curvature_threshold=CURVATURE_THRESHOLD,
            distance_smoothing=args.alm_distance_smoothing,
            curvature_smoothing=args.alm_curvature_smoothing,
            constraint_names=single_stage_alm_constraint_names(
                alm_formulation=args.alm_formulation,
                include_surface_surface=JSurfSurf is not None,
            ),
            curve_curve_constraint_fn=_smooth_min_curve_curve_signed_constraint,
            curve_surface_constraint_fn=_smooth_min_curve_surface_signed_constraint,
            curvature_constraint_fn=_smooth_max_curvature_signed_constraint,
            JSurfSurf=JSurfSurf,
            vessel_surface=VV,
            surface_surface_min_distance=SS_DIST,
            surface_surface_constraint_fn=_smooth_min_surface_surface_signed_constraint,
            alm_formulation=args.alm_formulation,
            qs_threshold=args.alm_qs_threshold,
            boozer_threshold=args.alm_boozer_threshold,
            iota_penalty_threshold=args.alm_iota_penalty_threshold,
            length_penalty_threshold=args.alm_length_penalty_threshold,
            coil_length_objective=curvelength,
            coil_length_threshold=length_target,
            banana_current=current_single_stage_alm_banana_current(),
            banana_current_threshold=args.banana_current_max_A,
            JNonQSObjective=objective_terms["JNonQSObjective"],
            JBoozerObjective=objective_terms["JBoozerObjective"],
            include_diagnostics=include_diagnostics,
        ),
        alm_formulation=args.alm_formulation,
    )


def evaluate_search_objective(surface_weights, *, include_diagnostics=None):
    if include_diagnostics is None:
        include_diagnostics = frontier_mode_enabled()
    if globals().get("CONSTRAINT_METHOD") == "alm":
        return annotate_frontier_search_eval(
            evaluate_alm_objective(
                surface_weights,
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
                ALM_MULTIPLIERS,
                ALM_PENALTY,
                JSurfSurf=JSurfSurf,
                include_diagnostics=include_diagnostics,
            )
        )
    return annotate_frontier_search_eval(
        evaluate_total_objective(
            surface_weights,
            nonQSs,
            brs,
            RES_WEIGHT,
            Jiota,
            IOTAS_WEIGHT,
            JCurveLength,
            LENGTH_WEIGHT,
            JCurveCurve,
            CC_WEIGHT,
            JCurveSurface,
            CS_WEIGHT,
            JCurvature,
            CURVATURE_WEIGHT,
            JSurfSurf=JSurfSurf,
            SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
            include_diagnostics=include_diagnostics,
        )
    )


def set_alm_runtime_state(multipliers, penalty):
    global ALM_MULTIPLIERS, ALM_PENALTY
    ALM_MULTIPLIERS = np.asarray(multipliers, dtype=float).copy()
    ALM_PENALTY = float(penalty)


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
            None if float(args.alm_trust_radius_init) == 0.0 else args.alm_trust_radius_init
        ),
        trust_radius_min=args.alm_trust_radius_min,
        trust_radius_shrink=args.alm_trust_radius_shrink,
        trust_radius_grow=args.alm_trust_radius_grow,
        max_inner_attempts=args.alm_max_inner_attempts,
    )


def _jsonable_value(value):
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, np.ndarray):
        return _jsonable_value(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]
    return value


def write_json_artifact(path, payload):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as outfile:
        json.dump(_jsonable_value(payload), outfile, indent=2)
    os.replace(temp_path, path)


def append_jsonl_artifact(path, payload):
    archive_path = os.path.abspath(path)
    serialized = json.dumps(_jsonable_value(payload)) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=os.path.dirname(archive_path),
        prefix=f".{os.path.basename(archive_path)}.",
        suffix=".tmp",
        delete=False,
    ) as outfile:
        temp_path = outfile.name
        if os.path.exists(archive_path):
            with open(archive_path, encoding="utf-8") as infile:
                shutil.copyfileobj(infile, outfile)
        outfile.write(serialized)
        outfile.flush()
        os.fsync(outfile.fileno())
    os.replace(temp_path, archive_path)


@contextmanager
def temporary_run_dict_value(run_dict, key, value):
    previous_value = run_dict.get(key)
    run_dict[key] = value
    try:
        yield
    finally:
        run_dict[key] = previous_value


SINGLE_STAGE_BANANA_CURRENT_DIAGNOSTICS_SCHEMA_VERSION = (
    "single_stage_banana_current_diagnostics_v3"
)
BANANA_CURRENT_DIAGNOSTICS_FILENAME = "banana_current_diagnostics.json"
BANANA_CURRENT_DIAGNOSTIC_REJECT_REPORT_LIMIT = 20
BANANA_CURRENT_FD_NONCURRENT_SELECTION = "largest_abs_baseline_value"


def _safe_optional_mean(values):
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_optional_median(values):
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=float)))


def _safe_optional_ratio(numerator, denominator):
    if numerator is None or denominator is None or float(denominator) == 0.0:
        return None
    return float(numerator) / float(denominator)


def _sanitize_banana_current_fd_probe_result(result):
    if isinstance(result, dict):
        objective_total = result.get("objective_total")
        resolved_success = bool(
            result.get("success", objective_total is not None)
        )
        resolved_surface_success = bool(
            result.get("surface_success", resolved_success)
        )
        return {
            "success": resolved_success,
            "surface_success": resolved_surface_success,
            "objective_total": (
                None if objective_total is None else float(objective_total)
            ),
            "rejection_reason": result.get("rejection_reason"),
            "topology_state": result.get("topology_state"),
            "hardware_ok": (
                None
                if "hardware_ok" not in result
                else bool(result.get("hardware_ok"))
            ),
        }
    return {
        "success": True,
        "surface_success": True,
        "objective_total": float(result),
        "rejection_reason": None,
        "topology_state": None,
        "hardware_ok": None,
    }


def _build_banana_current_fd_trial_probe(
    probe_result,
    *,
    trial_value,
    baseline_total,
):
    probe = _sanitize_banana_current_fd_probe_result(probe_result)
    objective_total = probe["objective_total"]
    probe["trial_value"] = float(trial_value)
    probe["objective_delta_total"] = (
        None
        if objective_total is None
        else float(objective_total - float(baseline_total))
    )
    return probe


def _relative_bounded_coordinate_step(
    value,
    lower_bound,
    upper_bound,
    *,
    relative_step_fraction,
):
    requested_step = abs(float(value)) * float(relative_step_fraction)
    if requested_step <= 0.0:
        return requested_step, 0.0
    applied_step = requested_step
    if np.isfinite(lower_bound):
        applied_step = min(applied_step, float(value) - float(lower_bound))
    if np.isfinite(upper_bound):
        applied_step = min(applied_step, float(upper_bound) - float(value))
    return requested_step, max(float(applied_step), 0.0)


def _select_largest_abs_noncurrent_coordinate_indices(
    x,
    coordinate_indices,
    *,
    max_count,
):
    x = np.asarray(x, dtype=float)
    current_index_set = {int(index) for index in coordinate_indices}
    ranked_indices = sorted(
        (
            int(index)
            for index in range(x.size)
            if int(index) not in current_index_set
        ),
        key=lambda index: abs(float(x[index])),
        reverse=True,
    )
    return tuple(ranked_indices[: max(0, int(max_count))])


def _build_relative_fd_coordinate_group(
    *,
    baseline_x,
    baseline_total,
    coordinate_indices,
    coordinate_dof_names,
    lower_bounds,
    upper_bounds,
    relative_step_fraction,
    objective_probe_fn,
    selection,
):
    baseline_x = np.asarray(baseline_x, dtype=float)
    baseline_total = float(baseline_total)
    lower_bounds = np.asarray(lower_bounds, dtype=float)
    upper_bounds = np.asarray(upper_bounds, dtype=float)
    samples = []
    max_abs_objective_deltas = []
    abs_central_differences = []

    for coordinate_index, coordinate_dof_name in zip(
        coordinate_indices,
        coordinate_dof_names,
    ):
        baseline_value = float(baseline_x[coordinate_index])
        requested_step, applied_step = _relative_bounded_coordinate_step(
            baseline_value,
            lower_bounds[coordinate_index],
            upper_bounds[coordinate_index],
            relative_step_fraction=relative_step_fraction,
        )
        sample = {
            "coordinate_index": int(coordinate_index),
            "coordinate_dof_name": str(coordinate_dof_name),
            "baseline_value": baseline_value,
            "requested_step": float(requested_step),
            "applied_step": float(applied_step),
            "step_clipped_to_bounds": bool(
                requested_step > 0.0 and applied_step < requested_step
            ),
            "applied_relative_step_fraction": (
                None
                if baseline_value == 0.0
                else float(applied_step / abs(baseline_value))
            ),
        }
        if applied_step <= 0.0:
            sample.update(
                {
                    "minus_probe": None,
                    "plus_probe": None,
                    "central_difference_per_unit": None,
                    "max_abs_objective_delta": None,
                    "mean_abs_objective_delta": None,
                    "step_status": "unavailable",
                }
            )
            samples.append(sample)
            continue

        minus_x = baseline_x.copy()
        minus_x[coordinate_index] = baseline_value - applied_step
        plus_x = baseline_x.copy()
        plus_x[coordinate_index] = baseline_value + applied_step
        minus_probe = _build_banana_current_fd_trial_probe(
            objective_probe_fn(minus_x),
            trial_value=baseline_value - applied_step,
            baseline_total=baseline_total,
        )
        plus_probe = _build_banana_current_fd_trial_probe(
            objective_probe_fn(plus_x),
            trial_value=baseline_value + applied_step,
            baseline_total=baseline_total,
        )

        successful_abs_deltas = [
            abs(float(probe["objective_delta_total"]))
            for probe in (minus_probe, plus_probe)
            if probe["objective_delta_total"] is not None
        ]
        central_difference_per_unit = None
        if (
            minus_probe["objective_total"] is not None
            and plus_probe["objective_total"] is not None
        ):
            central_difference_per_unit = float(
                (plus_probe["objective_total"] - minus_probe["objective_total"])
                / (2.0 * applied_step)
            )
            abs_central_differences.append(abs(central_difference_per_unit))
        max_abs_objective_delta = (
            None
            if not successful_abs_deltas
            else float(max(successful_abs_deltas))
        )
        mean_abs_objective_delta = (
            None
            if not successful_abs_deltas
            else float(np.mean(np.asarray(successful_abs_deltas, dtype=float)))
        )
        if max_abs_objective_delta is not None:
            max_abs_objective_deltas.append(max_abs_objective_delta)
        sample.update(
            {
                "minus_probe": minus_probe,
                "plus_probe": plus_probe,
                "central_difference_per_unit": central_difference_per_unit,
                "max_abs_objective_delta": max_abs_objective_delta,
                "mean_abs_objective_delta": mean_abs_objective_delta,
                "step_status": "ok",
            }
        )
        samples.append(sample)

    bidirectional_success_count = sum(
        sample["central_difference_per_unit"] is not None for sample in samples
    )
    successful_coordinate_count = sum(
        sample["max_abs_objective_delta"] is not None for sample in samples
    )
    return {
        "selection": str(selection),
        "relative_step_fraction": float(relative_step_fraction),
        "coordinate_indices": [int(index) for index in coordinate_indices],
        "coordinate_dof_names": [str(name) for name in coordinate_dof_names],
        "samples": samples,
        "summary": {
            "requested_coordinate_count": int(len(coordinate_indices)),
            "sample_count": int(len(samples)),
            "successful_coordinate_count": int(successful_coordinate_count),
            "bidirectional_success_count": int(bidirectional_success_count),
            "mean_max_abs_objective_delta": _safe_optional_mean(
                max_abs_objective_deltas
            ),
            "median_max_abs_objective_delta": _safe_optional_median(
                max_abs_objective_deltas
            ),
            "mean_abs_central_difference_per_unit": _safe_optional_mean(
                abs_central_differences
            ),
            "median_abs_central_difference_per_unit": _safe_optional_median(
                abs_central_differences
            ),
        },
    }


def build_banana_current_finite_difference_probe(
    objective_optimizable,
    banana_current_state,
    x,
    *,
    baseline_total,
    objective_probe_fn,
    active_optimizer_bounds=None,
    relative_step_fraction,
):
    coordinate_spec = resolve_banana_current_coordinate_spec(
        objective_optimizable,
        banana_current_state,
    )
    if not coordinate_spec.indices:
        return None

    baseline_total = float(baseline_total)
    lower_bounds, upper_bounds = _optimizer_bounds_arrays(
        objective_optimizable,
        active_optimizer_bounds=active_optimizer_bounds,
    )
    baseline_x = np.asarray(x, dtype=float)
    current_indices = tuple(int(index) for index in coordinate_spec.indices)
    noncurrent_indices = _select_largest_abs_noncurrent_coordinate_indices(
        baseline_x,
        current_indices,
        max_count=len(current_indices),
    )
    objective_dof_names = tuple(
        str(name) for name in objective_optimizable.dof_names
    )
    current_group = _build_relative_fd_coordinate_group(
        baseline_x=baseline_x,
        baseline_total=baseline_total,
        coordinate_indices=current_indices,
        coordinate_dof_names=coordinate_spec.dof_names,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        relative_step_fraction=relative_step_fraction,
        objective_probe_fn=objective_probe_fn,
        selection="banana_current_coordinates",
    )
    noncurrent_group = _build_relative_fd_coordinate_group(
        baseline_x=baseline_x,
        baseline_total=baseline_total,
        coordinate_indices=noncurrent_indices,
        coordinate_dof_names=tuple(
            objective_dof_names[index] for index in noncurrent_indices
        ),
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        relative_step_fraction=relative_step_fraction,
        objective_probe_fn=objective_probe_fn,
        selection=BANANA_CURRENT_FD_NONCURRENT_SELECTION,
    )
    current_summary = current_group["summary"]
    noncurrent_summary = noncurrent_group["summary"]
    return {
        "baseline_total": baseline_total,
        "relative_step_fraction": float(relative_step_fraction),
        "noncurrent_selection": BANANA_CURRENT_FD_NONCURRENT_SELECTION,
        "current_group": current_group,
        "matched_noncurrent_group": noncurrent_group,
        "comparison": {
            "current_to_noncurrent_mean_max_abs_objective_delta_ratio": (
                _safe_optional_ratio(
                    current_summary["mean_max_abs_objective_delta"],
                    noncurrent_summary["mean_max_abs_objective_delta"],
                )
            ),
            "current_to_noncurrent_median_max_abs_objective_delta_ratio": (
                _safe_optional_ratio(
                    current_summary["median_max_abs_objective_delta"],
                    noncurrent_summary["median_max_abs_objective_delta"],
                )
            ),
            "current_to_noncurrent_mean_abs_central_difference_ratio": (
                _safe_optional_ratio(
                    current_summary["mean_abs_central_difference_per_unit"],
                    noncurrent_summary["mean_abs_central_difference_per_unit"],
                )
            ),
            "current_to_noncurrent_median_abs_central_difference_ratio": (
                _safe_optional_ratio(
                    current_summary["median_abs_central_difference_per_unit"],
                    noncurrent_summary["median_abs_central_difference_per_unit"],
                )
            ),
        },
    }


def _bound_is_active(value, bound):
    return bool(
        np.isfinite(bound)
        and np.isclose(
            value,
            bound,
            atol=1.0e-6,
            rtol=1.0e-9,
        )
    )


def _active_bound_mask(values, bounds):
    return np.asarray(
        [_bound_is_active(value, bound) for value, bound in zip(values, bounds)],
        dtype=bool,
    )


def _banana_current_bound_activity(
    coordinate_values_A,
    coordinate_lower_bounds_A,
    coordinate_upper_bounds_A,
):
    entries = []
    for value_A, lower_bound_A, upper_bound_A in zip(
        coordinate_values_A,
        coordinate_lower_bounds_A,
        coordinate_upper_bounds_A,
    ):
        lower_is_finite = bool(np.isfinite(lower_bound_A))
        upper_is_finite = bool(np.isfinite(upper_bound_A))
        entries.append(
            {
                "lower_bound_A": None if not lower_is_finite else float(lower_bound_A),
                "upper_bound_A": None if not upper_is_finite else float(upper_bound_A),
                "distance_to_lower_A": (
                    None if not lower_is_finite else float(value_A - lower_bound_A)
                ),
                "distance_to_upper_A": (
                    None if not upper_is_finite else float(upper_bound_A - value_A)
                ),
                "at_lower_bound": bool(
                    lower_is_finite
                    and _bound_is_active(value_A, lower_bound_A)
                ),
                "at_upper_bound": bool(
                    upper_is_finite
                    and _bound_is_active(value_A, upper_bound_A)
                ),
            }
        )
    return entries


def _optimizer_bounds_arrays(objective_optimizable, active_optimizer_bounds=None):
    default_lower_bounds = np.asarray(
        objective_optimizable.lower_bounds,
        dtype=float,
    )
    default_upper_bounds = np.asarray(
        objective_optimizable.upper_bounds,
        dtype=float,
    )
    if active_optimizer_bounds is None:
        return default_lower_bounds, default_upper_bounds

    if hasattr(active_optimizer_bounds, "lb") and hasattr(active_optimizer_bounds, "ub"):
        lower_bounds = np.asarray(active_optimizer_bounds.lb, dtype=float)
        upper_bounds = np.asarray(active_optimizer_bounds.ub, dtype=float)
    else:
        bounds = np.asarray(active_optimizer_bounds, dtype=float)
        if bounds.ndim != 2 or bounds.shape[1] != 2:
            raise ValueError(
                "Active optimizer bounds must be a scipy Bounds object or a "
                "sequence of (lower, upper) pairs."
            )
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
    if (
        lower_bounds.shape != default_lower_bounds.shape
        or upper_bounds.shape != default_upper_bounds.shape
    ):
        raise ValueError(
            "Active optimizer bounds must match the objective coordinate shape."
        )
    return lower_bounds, upper_bounds


def _gradient_report_key(prefix, suffix):
    return suffix if not prefix else f"{prefix}_{suffix}"


def _safe_gradient_ratio(numerator_l2, denominator_l2):
    if denominator_l2 == 0.0:
        return None
    return float(numerator_l2 / denominator_l2)


def _physical_current_values_from_optimizer_coordinates(
    optimizer_values,
    scale_factors_A,
):
    return np.asarray(optimizer_values, dtype=float) * np.asarray(
        scale_factors_A,
        dtype=float,
    )


def _physical_current_bounds_from_optimizer_bounds(
    optimizer_lower_bounds,
    optimizer_upper_bounds,
    scale_factors_A,
):
    lower = np.asarray(optimizer_lower_bounds, dtype=float)
    upper = np.asarray(optimizer_upper_bounds, dtype=float)
    scale = np.asarray(scale_factors_A, dtype=float)
    lower_A = lower * scale
    upper_A = upper * scale
    return np.minimum(lower_A, upper_A), np.maximum(lower_A, upper_A)


def _physical_current_gradients_from_optimizer_gradients(
    optimizer_gradients,
    scale_factors_A,
):
    return np.asarray(optimizer_gradients, dtype=float) / np.asarray(
        scale_factors_A,
        dtype=float,
    )


def _empty_gradient_report_fields(*, prefix=""):
    return {
        _gradient_report_key(prefix, "coordinate_gradients"): None,
        _gradient_report_key(prefix, "coordinate_gradient_l2"): None,
        _gradient_report_key(prefix, "noncurrent_gradient_l2"): None,
        _gradient_report_key(prefix, "full_gradient_l2"): None,
        _gradient_report_key(prefix, "coordinate_to_noncurrent_gradient_ratio"): None,
        _gradient_report_key(prefix, "coordinate_to_full_gradient_ratio"): None,
    }


def _empty_optimizer_coordinate_gradient_report_fields(*, prefix=""):
    return {
        _gradient_report_key(prefix, "optimizer_coordinate_gradients"): None,
        _gradient_report_key(prefix, "optimizer_coordinate_gradient_l2"): None,
        _gradient_report_key(
            prefix,
            "optimizer_coordinate_to_noncurrent_gradient_ratio",
        ): None,
        _gradient_report_key(
            prefix,
            "optimizer_coordinate_to_full_gradient_ratio",
        ): None,
    }


def _gradient_report_fields(
    gradient,
    coordinate_indices,
    noncurrent_mask,
    *,
    prefix="",
    coordinate_scale_factors_A=None,
):
    gradient = np.asarray(gradient, dtype=float)
    optimizer_coordinate_gradient = gradient[coordinate_indices]
    coordinate_gradient = (
        optimizer_coordinate_gradient
        if coordinate_scale_factors_A is None
        else _physical_current_gradients_from_optimizer_gradients(
            optimizer_coordinate_gradient,
            coordinate_scale_factors_A,
        )
    )
    noncurrent_gradient = gradient[noncurrent_mask]
    physical_report_gradient = gradient.copy()
    physical_report_gradient[coordinate_indices] = coordinate_gradient
    coordinate_gradient_l2 = float(np.linalg.norm(coordinate_gradient))
    noncurrent_gradient_l2 = float(np.linalg.norm(noncurrent_gradient))
    full_gradient_l2 = float(np.linalg.norm(physical_report_gradient))
    return {
        _gradient_report_key(prefix, "coordinate_gradients"): coordinate_gradient.tolist(),
        _gradient_report_key(prefix, "coordinate_gradient_l2"): coordinate_gradient_l2,
        _gradient_report_key(prefix, "noncurrent_gradient_l2"): noncurrent_gradient_l2,
        _gradient_report_key(prefix, "full_gradient_l2"): full_gradient_l2,
        _gradient_report_key(
            prefix,
            "coordinate_to_noncurrent_gradient_ratio",
        ): _safe_gradient_ratio(
            coordinate_gradient_l2,
            noncurrent_gradient_l2,
        ),
        _gradient_report_key(
            prefix,
            "coordinate_to_full_gradient_ratio",
        ): _safe_gradient_ratio(
            coordinate_gradient_l2,
            full_gradient_l2,
        ),
    }


def _optimizer_coordinate_gradient_report_fields(
    gradient,
    coordinate_indices,
    noncurrent_mask,
    *,
    prefix="",
):
    coordinate_gradient = gradient[coordinate_indices]
    noncurrent_gradient = gradient[noncurrent_mask]
    coordinate_gradient_l2 = float(np.linalg.norm(coordinate_gradient))
    noncurrent_gradient_l2 = float(np.linalg.norm(noncurrent_gradient))
    full_gradient_l2 = float(np.linalg.norm(gradient))
    return {
        _gradient_report_key(
            prefix,
            "optimizer_coordinate_gradients",
        ): coordinate_gradient.tolist(),
        _gradient_report_key(
            prefix,
            "optimizer_coordinate_gradient_l2",
        ): coordinate_gradient_l2,
        _gradient_report_key(
            prefix,
            "optimizer_coordinate_to_noncurrent_gradient_ratio",
        ): _safe_gradient_ratio(
            coordinate_gradient_l2,
            noncurrent_gradient_l2,
        ),
        _gradient_report_key(
            prefix,
            "optimizer_coordinate_to_full_gradient_ratio",
        ): _safe_gradient_ratio(
            coordinate_gradient_l2,
            full_gradient_l2,
        ),
    }


def _project_lbfgsb_gradient(
    gradient,
    x,
    lower_bounds,
    upper_bounds,
):
    gradient_array = np.asarray(gradient, dtype=float)
    x_array = np.asarray(x, dtype=float)
    lower_bounds_array = np.asarray(lower_bounds, dtype=float)
    upper_bounds_array = np.asarray(upper_bounds, dtype=float)
    if (
        gradient_array.shape != x_array.shape
        or gradient_array.shape != lower_bounds_array.shape
        or gradient_array.shape != upper_bounds_array.shape
    ):
        raise ValueError(
            "Projected-gradient diagnostics require gradient, coordinates, and "
            "bounds arrays with matching shapes."
        )

    lower_active = _active_bound_mask(x_array, lower_bounds_array)
    upper_active = _active_bound_mask(x_array, upper_bounds_array)
    projected_gradient = gradient_array.copy()
    fixed_coordinate_mask = lower_active & upper_active
    projected_gradient[fixed_coordinate_mask] = 0.0
    projected_gradient[lower_active & ~fixed_coordinate_mask & (gradient_array > 0.0)] = 0.0
    projected_gradient[upper_active & ~fixed_coordinate_mask & (gradient_array < 0.0)] = 0.0
    return projected_gradient


def build_banana_current_coordinate_report(
    objective_optimizable,
    banana_current_state,
    x,
    grad,
    *,
    active_optimizer_bounds=None,
    phase,
    accepted_iteration=None,
    line_search_evaluations=None,
    step_norm=None,
    rejection_reason=None,
):
    if banana_current_state is None:
        return None
    coordinate_spec = resolve_banana_current_coordinate_spec(
        objective_optimizable,
        banana_current_state,
    )
    if not coordinate_spec.indices:
        return None

    lower_bounds, upper_bounds = _optimizer_bounds_arrays(
        objective_optimizable,
        active_optimizer_bounds=active_optimizer_bounds,
    )
    coordinate_indices = np.asarray(coordinate_spec.indices, dtype=int)
    coordinate_scale_factors_A = np.asarray(
        coordinate_spec.scale_factors_A,
        dtype=float,
    )
    optimizer_coordinate_values = np.asarray(x, dtype=float)[coordinate_indices]
    optimizer_coordinate_lower_bounds = lower_bounds[coordinate_indices]
    optimizer_coordinate_upper_bounds = upper_bounds[coordinate_indices]
    coordinate_values_A = _physical_current_values_from_optimizer_coordinates(
        optimizer_coordinate_values,
        coordinate_scale_factors_A,
    )
    coordinate_lower_bounds_A, coordinate_upper_bounds_A = (
        _physical_current_bounds_from_optimizer_bounds(
            optimizer_coordinate_lower_bounds,
            optimizer_coordinate_upper_bounds,
            coordinate_scale_factors_A,
        )
    )
    report = {
        "phase": str(phase),
        "accepted_iteration": (
            None if accepted_iteration is None else int(accepted_iteration)
        ),
        "line_search_evaluations": (
            None if line_search_evaluations is None else int(line_search_evaluations)
        ),
        "step_norm": None if step_norm is None else float(step_norm),
        "rejection_reason": rejection_reason,
        "coordinate_indices": coordinate_indices.tolist(),
        "coordinate_dof_names": list(coordinate_spec.dof_names),
        "coordinate_values_A": coordinate_values_A.tolist(),
        "coordinate_lower_bounds_A": coordinate_lower_bounds_A.tolist(),
        "coordinate_upper_bounds_A": coordinate_upper_bounds_A.tolist(),
        "optimizer_coordinate_values": optimizer_coordinate_values.tolist(),
        "optimizer_coordinate_lower_bounds": optimizer_coordinate_lower_bounds.tolist(),
        "optimizer_coordinate_upper_bounds": optimizer_coordinate_upper_bounds.tolist(),
        "current_coordinate_scale_factors_A": coordinate_scale_factors_A.tolist(),
        "bound_activity": _banana_current_bound_activity(
            coordinate_values_A,
            coordinate_lower_bounds_A,
            coordinate_upper_bounds_A,
        ),
    }
    if grad is None:
        report.update(_empty_gradient_report_fields())
        report.update(_empty_gradient_report_fields(prefix="projected"))
        report.update(_empty_optimizer_coordinate_gradient_report_fields())
        report.update(
            _empty_optimizer_coordinate_gradient_report_fields(prefix="projected")
        )
        return report

    gradient = np.asarray(grad, dtype=float)
    noncurrent_mask = np.ones(gradient.size, dtype=bool)
    noncurrent_mask[coordinate_indices] = False
    report.update(
        _gradient_report_fields(
            gradient,
            coordinate_indices,
            noncurrent_mask,
            coordinate_scale_factors_A=coordinate_scale_factors_A,
        )
    )
    report.update(
        _optimizer_coordinate_gradient_report_fields(
            gradient,
            coordinate_indices,
            noncurrent_mask,
        )
    )
    projected_gradient = _project_lbfgsb_gradient(
        gradient,
        np.asarray(x, dtype=float),
        lower_bounds,
        upper_bounds,
    )
    report.update(
        _gradient_report_fields(
            projected_gradient,
            coordinate_indices,
            noncurrent_mask,
            prefix="projected",
            coordinate_scale_factors_A=coordinate_scale_factors_A,
        )
    )
    report.update(
        _optimizer_coordinate_gradient_report_fields(
            projected_gradient,
            coordinate_indices,
            noncurrent_mask,
            prefix="projected",
        )
    )
    return report


def _add_banana_current_seed_delta(report, *, seed_coordinate_values_A):
    coordinate_values_A = np.asarray(report["coordinate_values_A"], dtype=float)
    seed_values_A = np.asarray(seed_coordinate_values_A, dtype=float)
    if coordinate_values_A.shape != seed_values_A.shape:
        raise ValueError(
            "Banana current diagnostics require seed and current coordinate vectors "
            "to share the same shape."
        )
    delta_from_seed_A = coordinate_values_A - seed_values_A
    report["delta_from_seed_A"] = delta_from_seed_A.tolist()
    report["max_abs_delta_from_seed_A"] = float(np.max(np.abs(delta_from_seed_A)))
    return report


def build_banana_current_diagnostics_state(
    objective_optimizable,
    banana_current_state,
    x,
    grad,
    *,
    active_optimizer_bounds=None,
    accepted_iteration,
    baseline_total=None,
    finite_difference_probe_fn=None,
    finite_difference_relative_step_fraction=None,
):
    seed_report = build_banana_current_coordinate_report(
        objective_optimizable,
        banana_current_state,
        x,
        grad,
        active_optimizer_bounds=active_optimizer_bounds,
        phase="seed",
        accepted_iteration=accepted_iteration,
    )
    if seed_report is None:
        return None
    seed_coordinate_values_A = tuple(seed_report["coordinate_values_A"])
    _add_banana_current_seed_delta(
        seed_report,
        seed_coordinate_values_A=seed_coordinate_values_A,
    )
    seed_finite_difference_probe = None
    if finite_difference_probe_fn is not None:
        if baseline_total is None:
            raise ValueError(
                "Banana current finite-difference diagnostics require baseline_total."
            )
        if finite_difference_relative_step_fraction is None:
            raise ValueError(
                "Banana current finite-difference diagnostics require a positive "
                "relative-step fraction."
            )
        baseline_total = float(baseline_total)
        finite_difference_relative_step_fraction = float(
            finite_difference_relative_step_fraction
        )
        seed_finite_difference_probe = build_banana_current_finite_difference_probe(
            objective_optimizable,
            banana_current_state,
            x,
            baseline_total=baseline_total,
            objective_probe_fn=finite_difference_probe_fn,
            active_optimizer_bounds=active_optimizer_bounds,
            relative_step_fraction=finite_difference_relative_step_fraction,
        )
    return {
        "schema_version": SINGLE_STAGE_BANANA_CURRENT_DIAGNOSTICS_SCHEMA_VERSION,
        "enabled": True,
        "mode": banana_current_state.mode,
        "num_control_currents": int(banana_current_state.num_control_currents()),
        "seed_currents_A": [float(value) for value in seed_coordinate_values_A],
        "configured_seed_currents_A": [
            float(value) for value in banana_current_state.seed_currents_A
        ],
        "seed_report": seed_report,
        "seed_finite_difference_probe": seed_finite_difference_probe,
        "latest_accepted_report": None,
        "latest_rejected_trial_report": None,
        "accepted_reports": [],
        "recent_rejected_trial_reports": [],
        "rejected_trial_reports_recorded": 0,
        "rejected_trial_reports_dropped": 0,
    }


def record_banana_current_diagnostics_report(
    diagnostics_state,
    report,
    *,
    rejected_trial=False,
):
    if diagnostics_state is None or report is None:
        return
    seed_coordinate_values_A = diagnostics_state["seed_report"]["coordinate_values_A"]
    _add_banana_current_seed_delta(
        report,
        seed_coordinate_values_A=seed_coordinate_values_A,
    )
    if rejected_trial:
        diagnostics_state["latest_rejected_trial_report"] = report
        diagnostics_state["rejected_trial_reports_recorded"] = int(
            diagnostics_state.get("rejected_trial_reports_recorded", 0)
        ) + 1
        rejected_reports = diagnostics_state.setdefault(
            "recent_rejected_trial_reports",
            [],
        )
        rejected_reports.append(report)
        if len(rejected_reports) > BANANA_CURRENT_DIAGNOSTIC_REJECT_REPORT_LIMIT:
            rejected_reports.pop(0)
            diagnostics_state["rejected_trial_reports_dropped"] = int(
                diagnostics_state.get("rejected_trial_reports_dropped", 0)
            ) + 1
        return
    diagnostics_state["latest_accepted_report"] = report
    diagnostics_state.setdefault("accepted_reports", []).append(report)


def finalize_banana_current_diagnostics_report(
    out_dir,
    diagnostics_state,
    report,
    *,
    rejected_trial=False,
):
    if diagnostics_state is None or report is None:
        return
    record_banana_current_diagnostics_report(
        diagnostics_state,
        report,
        rejected_trial=rejected_trial,
    )
    emit_banana_current_diagnostics_report(report)
    write_banana_current_diagnostics_artifact(
        out_dir,
        diagnostics_state,
    )


def emit_banana_current_diagnostics_report(report):
    if report is None:
        return
    print(
        "[banana-current-diagnostics] "
        + json.dumps(_jsonable_value(report), sort_keys=True)
    )


def write_banana_current_diagnostics_artifact(out_dir, diagnostics_state):
    if diagnostics_state is None:
        return
    write_json_artifact(
        os.path.join(out_dir, BANANA_CURRENT_DIAGNOSTICS_FILENAME),
        diagnostics_state,
    )


def evaluate_banana_current_fd_probe(
    trial_x,
    *,
    reference_x,
    reference_surface_state,
    accepted_iterations,
):
    reference_x = np.asarray(reference_x, dtype=float)
    search_gate = build_surface_search_gate(
        len(surface_data),
        accepted_iterations,
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
        SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
        SS_DIST if len(surface_data) > 1 else 0.0,
    )
    search_surface_weights = build_surface_search_weights(
        len(surface_data),
        accepted_iterations,
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
    )
    stack_status = solve_surface_stack_at_dofs(
        np.asarray(trial_x, dtype=float),
        JF,
        surface_data,
        reference_surface_state,
        vessel_surface=VV if len(surface_data) > 1 else None,
        surface_gap_threshold=search_gate["surface_gap_threshold"],
        vessel_gap_threshold=search_gate["vessel_gap_threshold"],
        enforce_nesting=search_gate["enforce_nesting"],
    )
    try:
        if not stack_status["success"]:
            return {
                "success": False,
                "surface_success": False,
                "objective_total": None,
                "final_iota": None,
                "final_volume": None,
                "rejection_reason": "surface_solve",
                "topology_state": None,
                "hardware_ok": None,
            }
        objective_eval = evaluate_search_objective(search_surface_weights)
        topology_status = evaluate_search_topology_gate(
            len(surface_data),
            surface_data[-1]["boozer_surface"].surface,
            bs,
            surface_mode_contract=globals().get("surface_mode_contract"),
        )
        topology_state = _topology_gate_state(topology_status)
        hardware_snapshot = evaluate_single_stage_hardware_snapshot(
            JCurveCurve,
            CC_DIST,
            JCurveSurface,
            CS_DIST,
            JSurfSurf,
            stack_status,
            SS_DIST,
            banana_curve,
            CURVATURE_THRESHOLD,
            **current_single_stage_hardware_snapshot_kwargs(),
        )
        hardware_status = hardware_snapshot["search_hardware_status"]
        rejection_reason = None
        if topology_state == "broken":
            rejection_reason = "topology_broken"
        elif not hardware_status["success"]:
            rejection_reason = "hardware"
        outer_surface = surface_data[-1]["boozer_surface"].surface
        return {
            "success": True,
            "surface_success": True,
            "objective_total": float(objective_eval["total"]),
            "final_iota": float(Iotas(surface_data[-1]["boozer_surface"]).J()),
            "final_volume": float(Volume(outer_surface).J()),
            "rejection_reason": rejection_reason,
            "topology_state": topology_state,
            "hardware_ok": bool(hardware_status["success"]),
        }
    finally:
        JF.x = reference_x.copy()
        restore_surface_states(surface_data, reference_surface_state)


def _load_banana_current_replay_context_state(
    diagnostics_path,
    *,
    replay_context_path=None,
):
    diagnostics_artifact_path = Path(diagnostics_path)
    resolved_context_path = (
        diagnostics_artifact_path.with_name(BANANA_CURRENT_REPLAY_CONTEXT_FILENAME)
        if replay_context_path is None
        else Path(replay_context_path)
    )
    if replay_context_path is not None and not resolved_context_path.is_file():
        raise FileNotFoundError(
            "Explicit banana-current replay context artifact was not found: "
            f"{resolved_context_path}"
        )
    if resolved_context_path.is_file():
        return (
            load_banana_current_replay_context(
                resolved_context_path,
                require_replay_contract=True,
            ),
            {
                "kind": "replay_context",
                "path": str(resolved_context_path),
            },
        )
    raise FileNotFoundError(
        "Banana-current replay requires a replay context artifact alongside the "
        "diagnostics input."
    )


def run_banana_current_rejected_trial_replay_study(
    diagnostics_path,
    *,
    replay_context_path=None,
    replay_output_path=None,
):
    diagnostics_artifact_path = Path(diagnostics_path)
    diagnostics_payload = json.loads(
        diagnostics_artifact_path.read_text(encoding="utf-8")
    )
    if diagnostics_payload.get("mode") != banana_current_state.mode:
        raise ValueError(
            "Banana-current replay diagnostics mode does not match the live "
            "single-stage banana-current mode."
        )
    if int(diagnostics_payload.get("num_control_currents", -1)) != int(
        banana_current_state.num_control_currents()
    ):
        raise ValueError(
            "Banana-current replay diagnostics control-count does not match the "
            "live single-stage banana-current configuration."
        )
    live_coordinate_spec = resolve_banana_current_coordinate_spec(
        JF,
        banana_current_state,
    )
    validate_banana_current_replay_coordinate_contract(
        diagnostics_payload["seed_report"],
        live_dof_names=live_coordinate_spec.dof_names,
        live_scale_factors_A=live_coordinate_spec.scale_factors_A,
    )
    live_coordinate_indices = np.asarray(live_coordinate_spec.indices, dtype=int)
    live_coordinate_scale_factors_A = np.asarray(
        live_coordinate_spec.scale_factors_A,
        dtype=float,
    )
    replay_context_state, replay_context_source = (
        _load_banana_current_replay_context_state(
            diagnostics_artifact_path,
            replay_context_path=replay_context_path,
        )
    )
    validate_banana_current_replay_context_contract(
        replay_context_state,
        diagnostics_payload,
    )
    rejected_reports = list(
        diagnostics_payload.get("recent_rejected_trial_reports", [])
    )
    original_incumbent = snapshot_single_stage_incumbent_state(run_dict)
    original_stage = run_dict.get("accepted_boozer_stage", stage)
    original_accepted_iterations = int(run_dict.get("accepted_iterations", 0))
    replay_reports = []
    try:
        for report_index, report in enumerate(rejected_reports):
            accepted_iteration = int(report["accepted_iteration"])
            if str(accepted_iteration) not in replay_context_state["accepted_incumbents"]:
                replay_reports.append(
                    {
                        "report_index": int(report_index),
                        "accepted_iteration": accepted_iteration,
                        "original_rejection_reason": report.get("rejection_reason"),
                        "replay_status": "missing_incumbent_context",
                    }
                )
                continue
            replay_stage, replay_incumbent = restore_banana_current_replay_incumbent(
                replay_context_state,
                accepted_iteration,
            )
            run_dict["accepted_iterations"] = accepted_iteration
            restore_incumbent_for_stage(
                run_dict,
                replay_incumbent,
                replay_stage,
                rebuild_stage_objective_bundle,
            )
            report_dof_names = tuple(report["coordinate_dof_names"])
            if report_dof_names != tuple(live_coordinate_spec.dof_names):
                raise ValueError(
                    "Banana-current replay report DOF names do not match the live "
                    "banana-current coordinate contract."
                )
            replay_optimizer_coordinates = np.asarray(
                report["optimizer_coordinate_values"],
                dtype=float,
            )
            reference_x = np.asarray(run_dict["accepted_x"], dtype=float).copy()
            trial_x = build_replayed_candidate_x(
                reference_x,
                live_coordinate_indices,
                replay_optimizer_coordinates,
            )
            incumbent_optimizer_coordinates = reference_x[live_coordinate_indices]
            replay_probe = evaluate_banana_current_fd_probe(
                trial_x,
                reference_x=reference_x,
                reference_surface_state=copy.deepcopy(run_dict["surface_state"]),
                accepted_iterations=accepted_iteration,
            )
            replay_success = bool(
                replay_probe["success"]
                and replay_probe["rejection_reason"] is None
            )
            replay_reports.append(
                {
                    "report_index": int(report_index),
                    "accepted_iteration": accepted_iteration,
                    "accepted_boozer_stage": replay_stage,
                    "original_rejection_reason": report.get("rejection_reason"),
                    "replay_status": "completed",
                    "replay_success": replay_success,
                    "replay_result": replay_probe,
                    "max_abs_delta_from_incumbent_A": float(
                        np.max(
                            np.abs(
                                (
                                    replay_optimizer_coordinates
                                    - incumbent_optimizer_coordinates
                                )
                                * live_coordinate_scale_factors_A
                            )
                        )
                    ),
                    "max_abs_delta_from_seed_A": float(
                        report.get("max_abs_delta_from_seed_A", 0.0)
                    ),
                }
            )
    finally:
        run_dict["accepted_iterations"] = original_accepted_iterations
        restore_incumbent_for_stage(
            run_dict,
            original_incumbent,
            original_stage,
            rebuild_stage_objective_bundle,
        )

    rejection_counts = {}
    rescue_counts = {}
    completed_replays = 0
    rescued_replays = 0
    for replay_report in replay_reports:
        rejection_reason = replay_report.get("original_rejection_reason")
        rejection_counts[rejection_reason] = rejection_counts.get(rejection_reason, 0) + 1
        if replay_report.get("replay_status") != "completed":
            continue
        completed_replays += 1
        if replay_report.get("replay_success"):
            rescued_replays += 1
            rescue_counts[rejection_reason] = rescue_counts.get(rejection_reason, 0) + 1

    artifact_path = (
        banana_current_rejected_trial_replay_path(diagnostics_artifact_path.parent)
        if replay_output_path is None
        else Path(replay_output_path)
    )
    replay_summary = {
        "schema_version": BANANA_CURRENT_REJECTED_TRIAL_REPLAY_SCHEMA_VERSION,
        "diagnostics_path": str(diagnostics_artifact_path),
        "artifact_path": str(artifact_path),
        "replay_context_source": replay_context_source,
        "rejected_trial_reports_recorded": int(
            diagnostics_payload.get("rejected_trial_reports_recorded", 0)
        ),
        "rejected_trial_reports_dropped": int(
            diagnostics_payload.get("rejected_trial_reports_dropped", 0)
        ),
        "stored_rejected_trial_reports": len(rejected_reports),
        "completed_replays": int(completed_replays),
        "rescued_replays": int(rescued_replays),
        "rejection_counts": rejection_counts,
        "rescue_counts": rescue_counts,
        "reports": replay_reports,
    }
    write_json_artifact(str(artifact_path), replay_summary)
    print(
        "[banana-current-replay] "
        + json.dumps(
            {
                "artifact_path": str(artifact_path),
                "stored_rejected_trial_reports": len(rejected_reports),
                "completed_replays": int(completed_replays),
                "rescued_replays": int(rescued_replays),
            },
            sort_keys=True,
        )
    )
    return replay_summary


SINGLE_STAGE_TOPOLOGY_DIAGNOSTICS_SCHEMA_VERSION = (
    "single_stage_topology_diagnostics_v1"
)


def _topology_gate_has_payload(status):
    return any(
        key in status
        for key in (
            "success",
            "state",
            "broken",
            "survived_lines",
            "survival_fraction",
        )
    )


def _topology_gate_outcome(status):
    has_gate_payload = _topology_gate_has_payload(status)
    enabled = bool(status.get("enabled", has_gate_payload))
    evaluated = bool(status.get("evaluated", has_gate_payload))
    state = status.get("state")
    if not evaluated:
        return "not_evaluated"
    if not enabled:
        return "disabled"
    if bool(status.get("broken", False)) or state == "broken":
        return "broken"
    if bool(status.get("success", False)) or state == "feasible":
        return "pass"
    return "reject"


def _topology_gate_reason(status, outcome):
    if outcome == "not_evaluated":
        return "not_evaluated"
    if outcome == "disabled":
        return "gate_disabled"
    if outcome == "broken":
        return status.get("evaluation_error_type") or "broken_evaluation"
    if outcome == "pass":
        return "survival_threshold_met"
    return status.get("first_exit_reason") or "survival_threshold_not_met"


def _format_topology_gate_summary(status, outcome):
    if outcome == "not_evaluated":
        return "Topology gate was not evaluated for this artifact."
    if outcome == "disabled":
        return "Topology gate was disabled for this artifact."
    if outcome == "broken":
        error_type = status.get("evaluation_error_type")
        error_message = status.get("evaluation_error")
        if error_type and error_message:
            return f"Topology evaluation broke: {error_type}: {error_message}"
        if error_message:
            return f"Topology evaluation broke: {error_message}"
        return "Topology evaluation broke."
    survived_lines = status.get("survived_lines")
    nfieldlines = status.get("nfieldlines")
    survival_fraction = status.get("survival_fraction")
    survival_threshold = status.get("survival_threshold")
    summary = (
        f"Topology gate {outcome}: survived {survived_lines}/{nfieldlines} lines "
        f"(fraction={survival_fraction}, threshold={survival_threshold})."
    )
    if outcome == "reject" and status.get("first_exit_reason") is not None:
        summary = (
            f"{summary} First exit reason={status.get('first_exit_reason')} "
            f"at t={status.get('first_exit_time')} phi={status.get('first_exit_angle')}."
        )
    return summary


def build_topology_gate_diagnostics(status, *, artifact_role):
    outcome = _topology_gate_outcome(status)
    has_gate_payload = _topology_gate_has_payload(status)
    return {
        "schema_version": SINGLE_STAGE_TOPOLOGY_DIAGNOSTICS_SCHEMA_VERSION,
        "kind": "gate",
        "artifact_role": str(artifact_role),
        "outcome": outcome,
        "reason": _topology_gate_reason(status, outcome),
        "summary": _format_topology_gate_summary(status, outcome),
        "enabled": bool(status.get("enabled", has_gate_payload)),
        "evaluated": bool(status.get("evaluated", has_gate_payload)),
        "state": status.get("state"),
        "status": _jsonable_value(status),
    }


def _topology_score_outcome(entry):
    return "broken" if bool(entry.get("topology_broken", False)) else "scored"


def _format_topology_score_summary(entry, outcome):
    if outcome == "broken":
        error_type = entry.get("topology_error_type")
        error_message = entry.get("topology_error")
        if error_type and error_message:
            return f"Topology scoring broke: {error_type}: {error_message}"
        if error_message:
            return f"Topology scoring broke: {error_message}"
        return "Topology scoring broke."
    return (
        "Topology score recorded at accepted iteration "
        f"{entry.get('accepted_iteration')}: survival="
        f"{entry.get('survived_lines')}/{entry.get('nfieldlines')}, "
        f"confinement_score={entry.get('confinement_score')}, "
        f"confinement_loss={entry.get('confinement_loss')}."
    )


def build_topology_score_diagnostics(entry, *, artifact_role):
    outcome = _topology_score_outcome(entry)
    return {
        "schema_version": SINGLE_STAGE_TOPOLOGY_DIAGNOSTICS_SCHEMA_VERSION,
        "kind": "score",
        "artifact_role": str(artifact_role),
        "outcome": outcome,
        "reason": (
            entry.get("topology_error_type")
            if outcome == "broken"
            else "score_recorded"
        ),
        "summary": _format_topology_score_summary(entry, outcome),
        "accepted_iteration": entry.get("accepted_iteration"),
        "state": entry.get("topology_state"),
        "entry": _jsonable_value(entry),
    }


def optional_topology_score_diagnostics(entry, *, artifact_role):
    if entry is None:
        return None
    return build_topology_score_diagnostics(
        entry,
        artifact_role=artifact_role,
    )


def write_topology_checkpoint_artifacts(
    out_dir,
    *,
    artifact_role,
    topology_entry,
    biotsavart,
    surface_data,
):
    os.makedirs(out_dir, exist_ok=True)
    write_json_artifact(
        os.path.join(out_dir, "topology_diagnostics.json"),
        build_topology_score_diagnostics(
            topology_entry,
            artifact_role=artifact_role,
        ),
    )
    biotsavart.save(os.path.join(out_dir, "biot_savart.json"))
    save_surface_artifacts(
        surface_data,
        biotsavart,
        out_dir,
        "surf",
        also_write_outer_legacy=False,
    )


def build_preserved_timeout_alm_state(
    *,
    constraint_method,
    penalty,
    multipliers,
) -> PreservedTimeoutALMState | None:
    if constraint_method != "alm":
        return None
    return PreservedTimeoutALMState(
        penalty=float(penalty),
        multipliers=np.asarray(multipliers, dtype=float).copy(),
    )


def current_preserved_timeout_alm_state() -> PreservedTimeoutALMState | None:
    return build_preserved_timeout_alm_state(
        constraint_method=PRESERVED_TIMEOUT_REPLAY_CONFIG.constraint_method,
        penalty=ALM_PENALTY,
        multipliers=ALM_MULTIPLIERS,
    )


def current_preserved_timeout_replay_config() -> PreservedTimeoutReplayConfig:
    replay_config = globals().get("PRESERVED_TIMEOUT_REPLAY_CONFIG", PRESERVED_TIMEOUT_REPLAY_CONFIG)
    stage2_bs_path = globals().get("stage2_bs_path")
    stage2_results_path = globals().get("stage2_results_path")
    frontier_goal_config = globals().get("FRONTIER_GOAL_CONFIG", replay_config)
    surface_data_value = globals().get("surface_data")
    banana_current_state = globals().get("banana_current_state")
    replay_banana_current_mode = replay_config.single_stage_banana_current_mode
    replay_banana_current_coordinate_scaling = (
        replay_config.single_stage_banana_current_coordinate_scaling
    )
    replay_num_banana_current_controls = replay_config.num_banana_current_controls
    if banana_current_state is not None:
        replay_banana_current_mode = banana_current_state.mode
        replay_banana_current_coordinate_scaling = banana_current_state.coordinate_scaling
        replay_num_banana_current_controls = (
            banana_current_state.num_control_currents()
        )

    def frontier_replay_value(replay_attr, config_attr):
        if isinstance(frontier_goal_config, FrontierGoalConfig):
            return getattr(frontier_goal_config, config_attr)
        return getattr(replay_config, replay_attr)

    return PreservedTimeoutReplayConfig(
        plasma_surf_filename=globals().get("plasma_surf_filename", replay_config.plasma_surf_filename),
        plasma_surf_path=globals().get("file_loc", replay_config.plasma_surf_path),
        stage2_bs_path=(
            replay_config.stage2_bs_path if stage2_bs_path is None else str(stage2_bs_path)
        ),
        stage2_seed_surf_path=globals().get(
            "stage2_seed_surf_path",
            replay_config.stage2_seed_surf_path,
        ),
        stage2_results_path=(
            replay_config.stage2_results_path if stage2_results_path is None else str(stage2_results_path)
        ),
        mpol=globals().get("mpol", replay_config.mpol),
        ntor=globals().get("ntor", replay_config.ntor),
        nphi=globals().get("nphi", replay_config.nphi),
        ntheta=globals().get("ntheta", replay_config.ntheta),
        constraint_weight=globals().get("CONSTRAINT_WEIGHT", replay_config.constraint_weight),
        constraint_method=globals().get("CONSTRAINT_METHOD", replay_config.constraint_method),
        alm_formulation=globals().get("ALM_FORMULATION", replay_config.alm_formulation),
        max_iterations=globals().get("MAXITER", replay_config.max_iterations),
        target_volume=globals().get("vol_target", replay_config.target_volume),
        target_iota=globals().get("iota_target", replay_config.target_iota),
        requested_seed_regime=globals().get(
            "REQUESTED_SEED_REGIME",
            replay_config.requested_seed_regime,
        ),
        effective_seed_regime=globals().get(
            "EFFECTIVE_SEED_REGIME",
            replay_config.effective_seed_regime,
        ),
        single_stage_goal_mode=globals().get("SINGLE_STAGE_GOAL_MODE", replay_config.single_stage_goal_mode),
        single_stage_banana_current_mode=replay_banana_current_mode,
        single_stage_banana_current_coordinate_scaling=(
            replay_banana_current_coordinate_scaling
        ),
        num_banana_current_controls=replay_num_banana_current_controls,
        single_stage_goal_mode_impl=current_frontier_goal_mode_impl(),
        boozer_surface_target_volumes=(
            replay_config.boozer_surface_target_volumes
            if surface_data_value is None
            else tuple(float(entry["target_volume"]) for entry in surface_data_value)
        ),
        frontier_iota_reference=frontier_replay_value("frontier_iota_reference", "iota_reference"),
        frontier_iota_scale=frontier_replay_value("frontier_iota_scale", "iota_scale"),
        frontier_volume_reference=frontier_replay_value(
            "frontier_volume_reference",
            "volume_reference",
        ),
        frontier_volume_scale=frontier_replay_value("frontier_volume_scale", "volume_scale"),
        frontier_qs_reference=frontier_replay_value("frontier_qs_reference", "qs_reference"),
        frontier_boozer_reference=frontier_replay_value(
            "frontier_boozer_reference",
            "boozer_reference",
        ),
        frontier_boozer_trust_threshold=frontier_replay_value(
            "frontier_boozer_trust_threshold",
            "boozer_trust_threshold",
        ),
        frontier_boozer_trust_penalty_scale=frontier_replay_value(
            "frontier_boozer_trust_penalty_scale",
            "boozer_trust_penalty_scale",
        ),
        frontier_effective_qs_weight=frontier_replay_value(
            "frontier_effective_qs_weight",
            "effective_qs_weight",
        ),
        frontier_effective_boozer_weight=frontier_replay_value(
            "frontier_effective_boozer_weight",
            "effective_boozer_weight",
        ),
        frontier_effective_iota_weight=frontier_replay_value(
            "frontier_effective_iota_weight",
            "effective_iota_weight",
        ),
        frontier_effective_volume_weight=frontier_replay_value(
            "frontier_effective_volume_weight",
            "effective_volume_weight",
        ),
        frontier_scalarization_type=frontier_replay_value(
            "frontier_scalarization_type",
            "scalarization_type",
        ),
        frontier_chebyshev_rho=frontier_replay_value(
            "frontier_chebyshev_rho",
            "chebyshev_rho",
        ),
        frontier_chebyshev_sharpness=frontier_replay_value(
            "frontier_chebyshev_sharpness",
            "chebyshev_sharpness",
        ),
        frontier_chebyshev_weight_iota=frontier_replay_value(
            "frontier_chebyshev_weight_iota",
            "chebyshev_weight_iota",
        ),
        frontier_chebyshev_weight_volume=frontier_replay_value(
            "frontier_chebyshev_weight_volume",
            "chebyshev_weight_volume",
        ),
        frontier_chebyshev_weight_qa=frontier_replay_value(
            "frontier_chebyshev_weight_qa",
            "chebyshev_weight_qa",
        ),
        frontier_chebyshev_weight_boozer=frontier_replay_value(
            "frontier_chebyshev_weight_boozer",
            "chebyshev_weight_boozer",
        ),
        epsilon_constraint_qa_max=frontier_replay_value(
            "epsilon_constraint_qa_max",
            "epsilon_constraint_qa_max",
        ),
        epsilon_constraint_boozer_max=frontier_replay_value(
            "epsilon_constraint_boozer_max",
            "epsilon_constraint_boozer_max",
        ),
        frontier_epsilon_penalty_weight=frontier_replay_value(
            "frontier_epsilon_penalty_weight",
            "epsilon_penalty_weight",
        ),
    )


_PRESERVED_TIMEOUT_RESULTS_FILENAMES = {
    "best_feasible": "results_best_feasible.partial.json",
    "best_accepted": "results_best_accepted.partial.json",
}
_PRESERVED_TIMEOUT_BS_FILENAMES = {
    "best_feasible": "biot_savart_best_feasible.json",
    "best_accepted": "biot_savart_best_accepted.json",
}
_PRESERVED_TIMEOUT_SURFACE_STEMS = {
    "best_feasible": "surf_best_feasible",
    "best_accepted": "surf_best_accepted",
}


def _preserved_timeout_artifact_name(names_by_kind, preservation_kind):
    if preservation_kind not in names_by_kind:
        raise ValueError(f"Unsupported preservation kind {preservation_kind!r}")
    return names_by_kind[preservation_kind]


def compute_surface_field_metrics(surf, bs):
    if hasattr(surf, "normal"):
        n = surf.normal()
        absn = np.linalg.norm(n, axis=2)
        unitn = n * (1.0 / absn)[:, :, None]
    else:
        unitn = surf.unitnormal()
        absn = np.linalg.norm(unitn, axis=2)
        unitn = unitn * (1.0 / absn)[:, :, None]
    surf_area = absn.reshape((-1, 1)) / float(absn.size)
    bs.set_points(surf.gamma().reshape((-1, 3)))
    field = bs.B().reshape(unitn.shape)
    bdotn = np.sum(field * unitn, axis=2)
    modb = np.sqrt(np.sum(field**2, axis=2))
    field_error = float(
        np.sum(np.abs((bdotn / modb).reshape((-1, 1))) * surf_area) / np.sum(surf_area)
    )
    mean_abs_bdotn = float(np.mean(np.abs(bdotn)))
    return field_error, mean_abs_bdotn


def build_preserved_timeout_results_payload(
    *,
    replay_config: PreservedTimeoutReplayConfig,
    preservation_kind,
    incumbent_stage,
    run_dict,
    objective_eval,
    field_error,
    final_iota,
    final_volume,
    hardware_snapshot,
    banana_current_state: SingleStageBananaCurrentState | None = None,
    coil_length,
    accepted_iteration,
    alm_runtime_state: PreservedTimeoutALMState | None = None,
):
    """Build preserved-timeout results.

    FINAL_SOURCE_STAGE and PRESERVED_TIMEOUT_SALVAGE_STAGE intentionally match
    because preserved timeout artifacts snapshot the same incumbent stage rather
    than a separate post-timeout refinement stage.
    """
    search_eval = run_dict["search_eval"]
    artifact_hardware_snapshot = dict(hardware_snapshot)
    if artifact_hardware_snapshot.get("coil_length") is None:
        artifact_hardware_snapshot["coil_length"] = float(coil_length)
    hardware_status = hardware_snapshot["artifact_hardware_status"]
    final_feasibility_ok = refinement_eligible_for_hardware_status(
        run_dict,
        hardware_status,
    )
    source_stage = str(incumbent_stage)
    is_frontier_mode = replay_config.single_stage_goal_mode == "frontier"
    payload = {
        "PLASMA_SURF_FILENAME": replay_config.plasma_surf_filename,
        "PLASMA_SURF_PATH": replay_config.plasma_surf_path,
        "STAGE2_BS_PATH": replay_config.stage2_bs_path,
        "STAGE2_SEED_SURF_PATH": replay_config.stage2_seed_surf_path,
        "STAGE2_RESULTS_PATH": replay_config.stage2_results_path,
        "mpol": replay_config.mpol,
        "ntor": replay_config.ntor,
        "nphi": replay_config.nphi,
        "ntheta": replay_config.ntheta,
        "CONSTRAINT_WEIGHT": (
            None if replay_config.constraint_weight is None else float(replay_config.constraint_weight)
        ),
        "CONSTRAINT_METHOD": replay_config.constraint_method,
        "ALM_FORMULATION": (
            replay_config.alm_formulation if replay_config.constraint_method == "alm" else None
        ),
        "REQUESTED_SEED_REGIME": replay_config.requested_seed_regime,
        "EFFECTIVE_SEED_REGIME": replay_config.effective_seed_regime,
        "init_only": False,
        "max_iterations": replay_config.max_iterations,
        "iterations": int(accepted_iteration),
        "TARGET_VOLUME": (
            None
            if is_frontier_mode or replay_config.target_volume is None
            else float(replay_config.target_volume)
        ),
        "TARGET_IOTA": (
            None
            if is_frontier_mode or replay_config.target_iota is None
            else float(replay_config.target_iota)
        ),
        "BOOZER_SURFACE_TARGET_VOLUMES": (
            None
            if replay_config.boozer_surface_target_volumes is None
            else list(replay_config.boozer_surface_target_volumes)
        ),
        "SINGLE_STAGE_GOAL_MODE": replay_config.single_stage_goal_mode,
        "SINGLE_STAGE_GOAL_MODE_IMPL": replay_config.single_stage_goal_mode_impl,
        "TERMINATION_MESSAGE": f"preserved_{preservation_kind}_partial",
        "OPTIMIZER_SUCCESS": False,
        "FINAL_SOURCE_STAGE": source_stage,
        "FIELD_ERROR": float(field_error),
        "OBJECTIVE_J": float(run_dict["J"]),
        "BASE_OBJECTIVE_J": float(search_eval.get("base_total", search_eval["total"])),
        "SEARCH_OBJECTIVE_J": float(search_eval["total"]),
        "FRONTIER_RANK_OBJECTIVE_J": (
            None
            if not is_frontier_mode
            else float(search_eval.get("frontier_rank_total", search_eval["total"]))
        ),
        "FINAL_IOTA": float(final_iota),
        "FINAL_VOLUME": float(final_volume),
        "FINAL_FEASIBILITY_OK": bool(final_feasibility_ok),
        "SELF_INTERSECTING": bool(run_dict["intersecting"]),
        **build_single_stage_banana_current_payload_fields(banana_current_state),
        **build_hardware_constraint_artifact_payload_fields(artifact_hardware_snapshot),
        **search_step_metrics_payload(run_dict),
        "FINAL_TOPOLOGY_GATE_SUCCESS": bool(run_dict["topology_gate_status"]["success"]),
        "FINAL_TOPOLOGY_GATE_STATE": run_dict["topology_gate_status"].get("state"),
        "FINAL_TOPOLOGY_GATE_ERROR": run_dict["topology_gate_status"].get("evaluation_error"),
        "FINAL_TOPOLOGY_GATE_DIAGNOSTICS": build_topology_gate_diagnostics(
            run_dict["topology_gate_status"],
            artifact_role=f"{preservation_kind}_final_topology_gate",
        ),
        "FINAL_TOPOLOGY_TRANSPORT_DIAGNOSTICS": run_dict["topology_gate_status"].get(
            "transport_diagnostics"
        ),
        "NONQS_RATIO": float(objective_eval["J_QS"]),
        "BOOZER_RESIDUAL": float(objective_eval["J_Boozer"]),
        "FRONTIER_TRUST_OK": search_eval.get("frontier_trust_ok"),
        "FRONTIER_BOOZER_TRUST_THRESHOLD": search_eval.get("frontier_boozer_trust_threshold"),
        "FRONTIER_BOOZER_TRUST_EXCESS": search_eval.get("frontier_boozer_trust_excess"),
        "FRONTIER_REFERENCE_IOTA": replay_config.frontier_iota_reference,
        "FRONTIER_REFERENCE_IOTA_SCALE": replay_config.frontier_iota_scale,
        "FRONTIER_REFERENCE_VOLUME": replay_config.frontier_volume_reference,
        "FRONTIER_REFERENCE_VOLUME_SCALE": replay_config.frontier_volume_scale,
        "FRONTIER_REFERENCE_QA": replay_config.frontier_qs_reference,
        "FRONTIER_REFERENCE_BOOZER": replay_config.frontier_boozer_reference,
        "FRONTIER_SCALARIZATION_TYPE": replay_config.frontier_scalarization_type,
        "FRONTIER_CHEBYSHEV_RHO": replay_config.frontier_chebyshev_rho,
        "FRONTIER_CHEBYSHEV_SHARPNESS": replay_config.frontier_chebyshev_sharpness,
        "FRONTIER_CHEBYSHEV_WEIGHT_IOTA": replay_config.frontier_chebyshev_weight_iota,
        "FRONTIER_CHEBYSHEV_WEIGHT_VOLUME": replay_config.frontier_chebyshev_weight_volume,
        "FRONTIER_CHEBYSHEV_WEIGHT_QA": replay_config.frontier_chebyshev_weight_qa,
        "FRONTIER_CHEBYSHEV_WEIGHT_BOOZER": replay_config.frontier_chebyshev_weight_boozer,
        "EPSILON_CONSTRAINT_QA_MAX": replay_config.epsilon_constraint_qa_max,
        "EPSILON_CONSTRAINT_BOOZER_MAX": replay_config.epsilon_constraint_boozer_max,
        "FRONTIER_EPSILON_PENALTY_WEIGHT": replay_config.frontier_epsilon_penalty_weight,
        "FRONTIER_EFFECTIVE_QA_WEIGHT": replay_config.frontier_effective_qs_weight,
        "FRONTIER_EFFECTIVE_BOOZER_WEIGHT": replay_config.frontier_effective_boozer_weight,
        "FRONTIER_EFFECTIVE_IOTA_WEIGHT": replay_config.frontier_effective_iota_weight,
        "FRONTIER_EFFECTIVE_VOLUME_WEIGHT": replay_config.frontier_effective_volume_weight,
        "FRONTIER_VOLUME_OBJECTIVE": search_eval.get("J_volume"),
        "FRONTIER_TRUST_PENALTY": search_eval.get("frontier_trust_penalty"),
        "FRONTIER_CONTRACT_PENALTY": search_eval.get("frontier_contract_penalty"),
        "FRONTIER_EPSILON_PENALTY": search_eval.get("frontier_epsilon_penalty"),
        "FRONTIER_HARDWARE_PENALTY": search_eval.get("frontier_hardware_penalty"),
        "FRONTIER_HARDWARE_MAX_VIOLATION_RATIO": search_eval.get(
            "frontier_hardware_max_violation_ratio"
        ),
        "FRONTIER_TOPOLOGY_PENALTY": search_eval.get("frontier_topology_penalty"),
        "FRONTIER_TOPOLOGY_DEFICIT": search_eval.get("frontier_topology_deficit"),
        "FRONTIER_BOOZER_TRUST_PENALTY_SCALE": search_eval.get(
            "frontier_boozer_trust_penalty_scale"
        ),
        "FRONTIER_BOOZER_TRUST_EXCESS_RATIO": search_eval.get(
            "frontier_boozer_trust_excess_ratio"
        ),
        "PRESERVED_TIMEOUT_SALVAGE_AVAILABLE": True,
        "PRESERVED_TIMEOUT_SALVAGE_KIND": preservation_kind,
        "PRESERVED_TIMEOUT_SALVAGE_STAGE": source_stage,
        "MAJOR_RADIUS": float(replay_config.major_radius),
        "R0_OFF_SPEC": is_major_radius_offspec(replay_config.major_radius),
    }
    if replay_config.constraint_method == "alm":
        if alm_runtime_state is None:
            raise ValueError("alm_runtime_state is required when constraint_method='alm'")
        payload.update(
            {
                "ALM_OUTER_ITERATIONS": run_dict.get("alm_outer_iteration"),
                "ALM_FINAL_MAX_FEASIBILITY_VIOLATION": search_eval.get(
                    "max_feasibility_violation",
                    search_eval.get("max_violation"),
                ),
                "ALM_FINAL_STATIONARITY_NORM": search_eval.get(
                    "metric_stationarity_norm",
                    search_eval.get("stationarity_norm"),
                ),
                "ALM_FINAL_FEASIBILITY_TOL": run_dict.get("alm_feasibility_tolerance"),
                "ALM_FINAL_STATIONARITY_TOL": run_dict.get("alm_stationarity_tolerance"),
                "ALM_FINAL_PENALTY": float(alm_runtime_state.penalty),
                "ALM_FINAL_MULTIPLIERS": np.asarray(
                    alm_runtime_state.multipliers,
                    dtype=float,
                ).tolist(),
                "ALM_FINAL_CONSTRAINT_VALUES": _jsonable_value(
                    search_eval.get("constraint_values")
                ),
            }
        )
    return payload


def write_preserved_timeout_artifacts(
    out_dir,
    *,
    preservation_kind,
    results_payload,
    biotsavart,
    surface_data,
):
    results_filename = _preserved_timeout_artifact_name(
        _PRESERVED_TIMEOUT_RESULTS_FILENAMES,
        preservation_kind,
    )
    bs_filename = _preserved_timeout_artifact_name(
        _PRESERVED_TIMEOUT_BS_FILENAMES,
        preservation_kind,
    )
    surface_stem = _preserved_timeout_artifact_name(
        _PRESERVED_TIMEOUT_SURFACE_STEMS,
        preservation_kind,
    )
    write_json_artifact(os.path.join(out_dir, results_filename), results_payload)
    biotsavart.save(os.path.join(out_dir, bs_filename))
    for entry in surface_data:
        boozer_surface = entry["boozer_surface"]
        surface = boozer_surface.surface
        path_stem = os.path.join(out_dir, f"{surface_stem}_{entry['name']}")
        surface.save(path_stem + ".json")
        boozer_surface.save(path_stem + "_boozer_surface.json")


def preserved_timeout_artifact_paths(out_dir, preservation_kind, surface_data):
    results_filename = _preserved_timeout_artifact_name(
        _PRESERVED_TIMEOUT_RESULTS_FILENAMES,
        preservation_kind,
    )
    bs_filename = _preserved_timeout_artifact_name(
        _PRESERVED_TIMEOUT_BS_FILENAMES,
        preservation_kind,
    )
    surface_stem = _preserved_timeout_artifact_name(
        _PRESERVED_TIMEOUT_SURFACE_STEMS,
        preservation_kind,
    )
    artifact_paths = [
        os.path.join(out_dir, results_filename),
        os.path.join(out_dir, bs_filename),
    ]
    for entry in surface_data:
        path_stem = os.path.join(out_dir, f"{surface_stem}_{entry['name']}")
        artifact_paths.extend(
            [
                path_stem + ".json",
                path_stem + "_boozer_surface.json",
            ]
        )
    return artifact_paths


def remove_preserved_timeout_artifacts(out_dir, *, preservation_kind, surface_data):
    for artifact_path in preserved_timeout_artifact_paths(
        out_dir,
        preservation_kind,
        surface_data,
    ):
        if os.path.exists(artifact_path):
            os.remove(artifact_path)


def diagnostic_search_eval_for_current_state(run_dict):
    search_eval = run_dict["search_eval"]
    if search_eval_has_diagnostics(search_eval):
        return search_eval
    surface_weights = search_eval.get("surface_weights")
    if surface_weights is None:
        surface_weights = build_surface_search_weights(
            len(surface_data),
            run_dict.get("accepted_iterations", 0),
            MULTISURFACE_RAMP_ITERATIONS,
            INNER_SURFACE_INITIAL_WEIGHT,
        )
    return evaluate_search_objective(
        np.asarray(surface_weights, dtype=float),
        include_diagnostics=True,
    )


def write_preserved_timeout_artifacts_for_current_state(
    out_dir,
    *,
    preservation_kind,
    incumbent_stage,
    run_dict,
    bs,
    surface_data,
    hardware_snapshot,
    field_error,
    coil_length,
):
    write_preserved_timeout_artifacts(
        out_dir,
        preservation_kind=preservation_kind,
        results_payload=build_preserved_timeout_results_payload(
            replay_config=PRESERVED_TIMEOUT_REPLAY_CONFIG,
            preservation_kind=preservation_kind,
            incumbent_stage=incumbent_stage,
            run_dict=run_dict,
            objective_eval=diagnostic_search_eval_for_current_state(run_dict),
            field_error=field_error,
            final_iota=run_dict["surface_status"]["iotas"][-1],
            final_volume=run_dict["surface_status"]["volumes"][-1],
            hardware_snapshot=hardware_snapshot,
            banana_current_state=globals().get("banana_current_state"),
            coil_length=coil_length,
            accepted_iteration=int(run_dict.get("accepted_iterations", 0)),
            alm_runtime_state=current_preserved_timeout_alm_state(),
        ),
        biotsavart=bs,
        surface_data=surface_data,
    )


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
    partial_path = os.path.join(out_dir, "alm_state.partial.json")
    write_json_artifact(partial_path, payload)


def build_single_stage_solver_checkpoint_state(
    run_dict,
    *,
    requested_maxiter,
    runtime_maxiter,
    accepted_stage,
    goal_mode,
    constraint_method,
    stage2_bs_path,
    out_dir_iter,
    alm_state=None,
):
    return build_solver_checkpoint_payload(
        goal_mode=goal_mode,
        constraint_method=constraint_method,
        stage2_bs_path=str(stage2_bs_path),
        requested_maxiter=int(requested_maxiter),
        runtime_maxiter=int(runtime_maxiter),
        accepted_iterations=int(run_dict.get("accepted_iterations", 0)),
        accepted_boozer_stage=str(accepted_stage),
        accepted_incumbent=snapshot_diagnostic_incumbent_state(run_dict),
        best_accepted_incumbent=run_dict.get("best_accepted_incumbent"),
        best_accepted_stage=run_dict.get("best_accepted_stage"),
        best_accepted_metric=run_dict.get("best_accepted_metric"),
        best_feasible_incumbent=run_dict.get("best_feasible_incumbent"),
        best_feasible_stage=run_dict.get("best_feasible_stage"),
        best_feasible_metric=run_dict.get("best_feasible_metric"),
        out_dir_iter=str(out_dir_iter),
        run_counters={
            "it": run_dict.get("it", 0),
            "invalid_state_rejects_total": run_dict.get(
                "invalid_state_rejects_total",
                0,
            ),
            "topology_gate_rejects": run_dict.get("topology_gate_rejects", 0),
            "hardware_rejects": run_dict.get("hardware_rejects", 0),
            "curvature_precheck_rejects": run_dict.get(
                "curvature_precheck_rejects",
                0,
            ),
            "curvature_overcap_boozer_evals": run_dict.get(
                "curvature_overcap_boozer_evals",
                0,
            ),
            "surface_solve_rejects": run_dict.get("surface_solve_rejects", 0),
            "frontier_trust_rejects": run_dict.get("frontier_trust_rejects", 0),
        },
        alm_state=alm_state,
        conditioning_seed_report=run_dict.get(
            "frontier_conditioning_seed_report"
        ),
        conditioning_first_accepted_report=run_dict.get(
            "frontier_conditioning_first_accepted_report"
        ),
    )


def write_single_stage_solver_checkpoint_state(
    out_dir,
    run_dict,
    *,
    requested_maxiter,
    runtime_maxiter,
    accepted_stage,
    goal_mode,
    constraint_method,
    stage2_bs_path,
    out_dir_iter,
    alm_state=None,
):
    checkpoint_payload = build_single_stage_solver_checkpoint_state(
        run_dict,
        requested_maxiter=requested_maxiter,
        runtime_maxiter=runtime_maxiter,
        accepted_stage=accepted_stage,
        goal_mode=goal_mode,
        constraint_method=constraint_method,
        stage2_bs_path=stage2_bs_path,
        out_dir_iter=out_dir_iter,
        alm_state=alm_state,
    )
    write_solver_checkpoint(
        solver_checkpoint_path(out_dir),
        checkpoint_payload,
    )


def current_solver_checkpoint_alm_state():
    alm_state = current_preserved_timeout_alm_state()
    if alm_state is None:
        return None
    return {
        "penalty": float(alm_state.penalty),
        "multipliers": np.asarray(alm_state.multipliers, dtype=float).copy(),
    }


def build_total_objective(
    JnonQSRatio,
    RES_WEIGHT,
    JBoozerResidual,
    IOTAS_WEIGHT,
    Jiota,
    VOLUME_WEIGHT,
    JVolume,
    LENGTH_WEIGHT,
    JCurveLength,
    CC_WEIGHT,
    JCurveCurve,
    CS_WEIGHT,
    JCurveSurface,
    CURVATURE_WEIGHT,
    JCurvature,
    SURF_DIST_WEIGHT=0.0,
    JSurfSurf=None,
):
    return _build_total_objective_impl(
        JnonQSRatio,
        RES_WEIGHT,
        JBoozerResidual,
        IOTAS_WEIGHT,
        Jiota,
        VOLUME_WEIGHT,
        JVolume,
        LENGTH_WEIGHT,
        JCurveLength,
        CC_WEIGHT,
        JCurveCurve,
        CS_WEIGHT,
        JCurveSurface,
        CURVATURE_WEIGHT,
        JCurvature,
        SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
        JSurfSurf=JSurfSurf,
    )


def finalize_surface_stack(x, objective, surface_data, run_state, vessel_surface=None, surface_gap_threshold=0.0, vessel_gap_threshold=0.0):
    status = solve_surface_stack_at_dofs(
        x,
        objective,
        surface_data,
        run_state["surface_state"],
        vessel_surface=vessel_surface,
        surface_gap_threshold=surface_gap_threshold,
        vessel_gap_threshold=vessel_gap_threshold,
        enforce_nesting=True,
    )
    if status["success"]:
        run_state["accepted_x"] = np.asarray(x).copy()
        run_state["surface_state"] = snapshot_surface_states(surface_data)
        run_state["surface_status"] = status
        run_state["J"] = objective.J()
        run_state["dJ"] = objective.dJ().copy()
    else:
        print("Final optimized state rejected; falling back to last accepted state.")
        objective.x = run_state["accepted_x"].copy()
        restore_surface_states(surface_data, run_state["surface_state"])
        run_state["surface_status"] = evaluate_surface_stack(
            surface_data,
            vessel_surface=vessel_surface,
            surface_gap_threshold=surface_gap_threshold,
            vessel_gap_threshold=vessel_gap_threshold,
        )
    run_state["intersecting"] = any(run_state["surface_status"]["self_intersections"])
    return run_state["surface_status"]

def normPlot(surf, bs, filename):
    """Plot normal magnetic field — delegates to shared norm_field_plot."""
    mean_abs_relBfinal_norm, _, _, _, _ = norm_field_plot(surf, bs, filename)
    return mean_abs_relBfinal_norm


def new_search_step_metrics():
    return {
        "evaluations": 0,
        "accepted_evaluations": 0,
        "rejected_evaluations": 0,
        "rejected_after_surface_solve": 0,
        "rejected_before_surface_solve": 0,
        "surface_solve_rejects": 0,
        "topology_rejects": 0,
        "hardware_rejects": 0,
        "curvature_rejects": 0,
        "curvature_precheck_rejects": 0,
        "curvature_overcap_boozer_evals": 0,
        "other_rejects": 0,
        "objective_evaluations": 0,
        "fast_objective_evaluations": 0,
        "diagnostic_objective_evaluations": 0,
        "curvature_precheck_seconds": 0.0,
        "surface_solve_seconds": 0.0,
        "objective_eval_seconds": 0.0,
        "topology_gate_seconds": 0.0,
        "hardware_snapshot_seconds": 0.0,
        "total_seconds": 0.0,
        "last_rejection_reason": None,
        "last_step_norm": None,
        "last_rejection_increment": None,
    }


def search_step_metrics_for_run(run_dict):
    if "search_step_metrics" not in run_dict:
        run_dict["search_step_metrics"] = new_search_step_metrics()
    return run_dict["search_step_metrics"]


def record_search_step_objective_eval(metrics, objective_eval):
    metrics["objective_evaluations"] += 1
    if objective_eval.get("diagnostics_included", True):
        metrics["diagnostic_objective_evaluations"] += 1
    else:
        metrics["fast_objective_evaluations"] += 1


def hardware_status_has_curvature_violation(hardware_status):
    return not hardware_status["constraints"]["max_curvature"]["success"]


def search_step_rejection_reason(rejection_reason):
    if rejection_reason is None:
        return "surface_solve"
    return rejection_reason


def record_search_step_acceptance(metrics):
    metrics["accepted_evaluations"] += 1
    metrics["last_rejection_reason"] = None
    metrics["last_rejection_increment"] = None


def record_search_step_rejection(
    metrics,
    *,
    rejection_reason,
    stack_status,
    hardware_status,
    rejection_increment,
):
    reason = search_step_rejection_reason(rejection_reason)
    metrics["rejected_evaluations"] += 1
    metrics["last_rejection_reason"] = reason
    metrics["last_rejection_increment"] = float(rejection_increment)
    if stack_status["success"]:
        metrics["rejected_after_surface_solve"] += 1
    else:
        metrics["rejected_before_surface_solve"] += 1
    if reason == "surface_solve":
        metrics["surface_solve_rejects"] += 1
    elif reason.startswith("topology"):
        metrics["topology_rejects"] += 1
    elif reason == "hardware":
        metrics["hardware_rejects"] += 1
        if hardware_status_has_curvature_violation(hardware_status):
            metrics["curvature_rejects"] += 1
    elif reason == "curvature_precheck":
        metrics["curvature_rejects"] += 1
        metrics["curvature_precheck_rejects"] += 1
    else:
        metrics["other_rejects"] += 1


def optional_search_step_float(metrics, key):
    value = metrics[key]
    if value is None:
        return None
    return float(value)


def search_step_metrics_payload(run_dict):
    metrics = run_dict.get("search_step_metrics", new_search_step_metrics())
    return {
        "SEARCH_STEP_EVALS": int(metrics["evaluations"]),
        "SEARCH_STEP_ACCEPTED_EVALS": int(metrics["accepted_evaluations"]),
        "SEARCH_STEP_REJECTED_EVALS": int(metrics["rejected_evaluations"]),
        "SEARCH_STEP_REJECTED_AFTER_SURFACE_SOLVE": int(
            metrics["rejected_after_surface_solve"]
        ),
        "SEARCH_STEP_REJECTED_BEFORE_SURFACE_SOLVE": int(
            metrics["rejected_before_surface_solve"]
        ),
        "SEARCH_STEP_SURFACE_SOLVE_REJECTS": int(metrics["surface_solve_rejects"]),
        "SEARCH_STEP_TOPOLOGY_REJECTS": int(metrics["topology_rejects"]),
        "SEARCH_STEP_HARDWARE_REJECTS": int(metrics["hardware_rejects"]),
        "SEARCH_STEP_CURVATURE_REJECTS": int(metrics["curvature_rejects"]),
        "SEARCH_STEP_CURVATURE_PRECHECK_REJECTS": int(
            metrics["curvature_precheck_rejects"]
        ),
        "SEARCH_STEP_CURVATURE_OVERCAP_BOOZER_EVALS": int(
            metrics["curvature_overcap_boozer_evals"]
        ),
        "SEARCH_STEP_OTHER_REJECTS": int(metrics["other_rejects"]),
        "SEARCH_STEP_OBJECTIVE_EVALS": int(metrics["objective_evaluations"]),
        "SEARCH_STEP_FAST_OBJECTIVE_EVALS": int(
            metrics["fast_objective_evaluations"]
        ),
        "SEARCH_STEP_DIAGNOSTIC_OBJECTIVE_EVALS": int(
            metrics["diagnostic_objective_evaluations"]
        ),
        "SEARCH_STEP_CURVATURE_PRECHECK_SECONDS": float(
            metrics["curvature_precheck_seconds"]
        ),
        "SEARCH_STEP_SURFACE_SOLVE_SECONDS": float(
            metrics["surface_solve_seconds"]
        ),
        "SEARCH_STEP_OBJECTIVE_EVAL_SECONDS": float(
            metrics["objective_eval_seconds"]
        ),
        "SEARCH_STEP_TOPOLOGY_GATE_SECONDS": float(
            metrics["topology_gate_seconds"]
        ),
        "SEARCH_STEP_HARDWARE_SNAPSHOT_SECONDS": float(
            metrics["hardware_snapshot_seconds"]
        ),
        "SEARCH_STEP_TOTAL_SECONDS": float(metrics["total_seconds"]),
        "SEARCH_STEP_LAST_REJECTION_REASON": metrics["last_rejection_reason"],
        "SEARCH_STEP_LAST_STEP_NORM": optional_search_step_float(
            metrics,
            "last_step_norm",
        ),
        "SEARCH_STEP_LAST_REJECTION_INCREMENT": optional_search_step_float(
            metrics,
            "last_rejection_increment",
        ),
    }


def current_curvature_traversal_policy():
    return CurvatureTraversalPolicy(
        float(
            globals().get(
                "CURVATURE_TRAVERSAL_BAND",
                DEFAULT_CURVATURE_TRAVERSAL_BAND,
            )
        ),
        int(
            globals().get(
                "CURVATURE_TRAVERSAL_EVAL_BUDGET",
                DEFAULT_CURVATURE_TRAVERSAL_EVAL_BUDGET,
            )
        ),
    )


def curvature_traversal_precheck_enabled(policy):
    return (
        globals().get("CONSTRAINT_METHOD", "penalty") != "alm"
        and (policy.band_ratio > 0.0 or policy.eval_budget > 0)
    )


def evaluate_curvature_traversal_precheck(x, metrics):
    policy = current_curvature_traversal_policy()
    if not curvature_traversal_precheck_enabled(policy):
        return {"enabled": False, "allow_boozer_eval": True}

    precheck_start = time.perf_counter()
    JF.x = x
    max_curvature = float(np.max(banana_curve.kappa()))
    used_budget = int(
        run_dict.get("curvature_overcap_boozer_evals_this_iteration", 0)
    )
    decision = decide_curvature_traversal(
        max_curvature=max_curvature,
        curvature_threshold=CURVATURE_THRESHOLD,
        policy=policy,
        used_budget=used_budget,
    )
    metrics["curvature_precheck_seconds"] += time.perf_counter() - precheck_start

    status = {
        "enabled": True,
        "allow_boozer_eval": bool(decision.allow_boozer_eval),
        "over_threshold": bool(decision.over_threshold),
        "reason": decision.reason,
        "max_curvature": max_curvature,
        "curvature_threshold": float(CURVATURE_THRESHOLD),
        "far_invalid_limit": float(decision.far_invalid_limit),
        "used_budget": used_budget,
        "eval_budget": int(policy.eval_budget),
        "band_ratio": float(policy.band_ratio),
    }
    run_dict["curvature_traversal_status"] = status
    if decision.allow_boozer_eval and decision.over_threshold:
        run_dict["curvature_overcap_boozer_evals_this_iteration"] = used_budget + 1
        run_dict["curvature_overcap_boozer_evals"] = (
            int(run_dict.get("curvature_overcap_boozer_evals", 0)) + 1
        )
        metrics["curvature_overcap_boozer_evals"] += 1
    return status


def hardware_status_with_curvature_traversal(hardware_status, precheck_status):
    if (
        not precheck_status["enabled"]
        or not precheck_status["allow_boozer_eval"]
        or not precheck_status["over_threshold"]
    ):
        return hardware_status

    failed_constraint_names = {
        name
        for name, constraint in hardware_status["constraints"].items()
        if not constraint["success"]
    }
    if failed_constraint_names != {"max_curvature"}:
        return hardware_status

    adjusted_constraints = dict(hardware_status["constraints"])
    adjusted_curvature = dict(adjusted_constraints["max_curvature"])
    adjusted_curvature["success"] = True
    adjusted_curvature["violation"] = 0.0
    adjusted_curvature["curvature_traversal_allowed"] = True
    adjusted_constraints["max_curvature"] = adjusted_curvature

    adjusted_allowed_status = dict(hardware_status["allowed_traversal_status"])
    adjusted_allowed_constraints = dict(adjusted_allowed_status["constraints"])
    adjusted_allowed_constraints["max_curvature"] = adjusted_curvature
    adjusted_allowed_status["success"] = True
    adjusted_allowed_status["violations"] = []
    adjusted_allowed_status["constraints"] = adjusted_allowed_constraints

    adjusted_status = dict(hardware_status)
    adjusted_status["success"] = True
    adjusted_status["violations"] = []
    adjusted_status["constraints"] = adjusted_constraints
    adjusted_status["allowed_traversal_status"] = adjusted_allowed_status
    adjusted_ratios = dict(hardware_status["violation_ratios"])
    adjusted_ratios["max_curvature_penalty"] = 0.0
    adjusted_status["violation_ratios"] = adjusted_ratios
    adjusted_status["curvature_traversal_allowed"] = True
    adjusted_status["curvature_traversal_status"] = dict(precheck_status)
    adjusted_status["curvature_traversal_original_violations"] = list(
        hardware_status["violations"]
    )
    return adjusted_status


def reject_search_step_on_curvature_precheck(metrics, precheck_status, step_start):
    rejection_increment = hardware_rejection_increment(run_dict["J"])
    run_dict["curvature_precheck_rejects"] = (
        int(run_dict.get("curvature_precheck_rejects", 0)) + 1
    )
    run_dict["invalid_state_rejects_total"] += 1
    print("/!\\ /!\\ Curvature precheck rejected candidate /!\\ /!\\")
    print(
        "Trial max curvature "
        f"{precheck_status['max_curvature']:.6e} exceeds traversal limit "
        f"{precheck_status['far_invalid_limit']:.6e} "
        f"(threshold={precheck_status['curvature_threshold']:.6e}, "
        f"reason={precheck_status['reason']})"
    )
    record_search_step_rejection(
        metrics,
        rejection_reason="curvature_precheck",
        stack_status={"success": False},
        hardware_status=None,
        rejection_increment=rejection_increment,
    )
    JF.x = run_dict["accepted_x"]
    restore_surface_states(surface_data, run_dict["surface_state"])
    metrics["total_seconds"] += time.perf_counter() - step_start
    return {"total": run_dict["J"] + rejection_increment, "grad": run_dict["dJ"].copy()}


def evaluate_search_step(x):
    """
    Objective function for L-BFGS-B optimization.

    Evaluates the total objective function and its gradient for a given set of
    degrees of freedom (coil parameters). Attempts to solve for a valid Boozer
    surface; if unsuccessful (solver failure or self-intersection), returns an
    elevated objective value with the last accepted gradient to trigger line
    search backtracking without corrupting the L-BFGS-B Hessian approximation.

    Args:
        x: Current degrees of freedom (coil parameters)

    Returns:
        Dictionary with objective value and gradient for the current search step.
    """
    step_start = time.perf_counter()
    dx = np.linalg.norm(x - run_dict['x_prev'])
    metrics = search_step_metrics_for_run(run_dict)
    metrics["evaluations"] += 1
    metrics["last_step_norm"] = float(dx)
    outer_entry = surface_data[-1]
    run_dict['x_prev'] = x.copy()
    print(f"Step size: {dx:.2e}")

    run_dict['lscount'] += 1
    run_dict["trial_hardware_status"] = None
    run_dict.setdefault("invalid_state_rejects_total", 0)
    run_dict.setdefault("topology_gate_rejects", 0)
    run_dict.setdefault("hardware_rejects", 0)
    run_dict.setdefault("surface_solve_rejects", 0)
    run_dict.setdefault("frontier_trust_rejects", 0)
    run_dict.setdefault("curvature_precheck_rejects", 0)
    run_dict.setdefault("curvature_overcap_boozer_evals", 0)
    run_dict.setdefault("curvature_overcap_boozer_evals_this_iteration", 0)
    search_gate = build_surface_search_gate(
        len(surface_data),
        run_dict['accepted_iterations'],
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
        SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
        SS_DIST if len(surface_data) > 1 else 0.0,
    )

    curvature_precheck_status = evaluate_curvature_traversal_precheck(x, metrics)
    if not curvature_precheck_status["allow_boozer_eval"]:
        return reject_search_step_on_curvature_precheck(
            metrics,
            curvature_precheck_status,
            step_start,
        )

    surface_solve_start = time.perf_counter()
    stack_status = solve_surface_stack_at_dofs(
        x,
        JF,
        surface_data,
        run_dict['surface_state'],
        vessel_surface=VV if len(surface_data) > 1 else None,
        surface_gap_threshold=search_gate['surface_gap_threshold'],
        vessel_gap_threshold=search_gate['vessel_gap_threshold'],
        enforce_nesting=search_gate['enforce_nesting'],
    )
    metrics["surface_solve_seconds"] += time.perf_counter() - surface_solve_start
    success = stack_status['success']

    rejection_increment = None
    rejection_reason = None
    objective_eval = None
    repair_phase1_mode_active = bool(run_dict.get("phase1_repair_mode_active", False))

    if success:
        search_surface_weights = build_surface_search_weights(
            len(surface_data),
            run_dict['accepted_iterations'],
            MULTISURFACE_RAMP_ITERATIONS,
            INNER_SURFACE_INITIAL_WEIGHT,
        )
        objective_eval_start = time.perf_counter()
        objective_eval = evaluate_search_objective(
            search_surface_weights,
            include_diagnostics=frontier_mode_enabled(),
        )
        metrics["objective_eval_seconds"] += time.perf_counter() - objective_eval_start
        record_search_step_objective_eval(metrics, objective_eval)
        J = objective_eval['total']
        dJ = objective_eval['grad']
        frontier_trust_status = evaluate_frontier_trust_status(objective_eval)
        run_dict["frontier_trust_status"] = frontier_trust_status
        if frontier_trust_status["enabled"] and not frontier_trust_status["ok"]:
            trust_penalty = objective_eval.get("frontier_trust_penalty", 0.0)
            excess_ratio = objective_eval.get("frontier_boozer_trust_excess_ratio", 0.0)
            run_dict["frontier_trust_rejects"] += 1
            print("/!\\ /!\\ Frontier Boozer trust penalty active /!\\ /!\\")
            print(
                "Boozer residual "
                f"{frontier_trust_status['residual']:.6e} exceeds trust threshold "
                f"{frontier_trust_status['threshold']:.6e}"
            )
            print(
                "Frontier trust penalty = "
                f"{trust_penalty:.6e} "
                f"(excess_ratio={excess_ratio:.6e})"
            )

        hard_invalidation = _evaluate_frontier_hard_invalidation_impl(
            search_eval=objective_eval,
            surface_success=stack_status["success"],
            surface_status=stack_status,
        )
        if hard_invalidation["invalid"]:
            success = False
            rejection_reason = hard_invalidation["reason"]
            run_dict["invalid_state_rejects_total"] += 1
            print("/!\\ /!\\ Frontier search evaluation invalid /!\\ /!\\")
            print(f"Hard invalidation reason: {hard_invalidation['reason']}")
            if hard_invalidation["fields"]:
                print(
                    "Invalid frontier fields: "
                    + ", ".join(hard_invalidation["fields"])
                )

        if success:
            active_surface_mode_contract = globals().get("surface_mode_contract")
            topology_gate_start = time.perf_counter()
            topology_status = evaluate_search_topology_gate(
                len(surface_data),
                outer_entry['boozer_surface'].surface,
                bs,
                surface_mode_contract=active_surface_mode_contract,
            )
            metrics["topology_gate_seconds"] += (
                time.perf_counter() - topology_gate_start
            )
            run_dict['topology_gate_status'] = topology_status
            topology_state = _topology_gate_state(topology_status)
            if topology_state == "broken":
                success = False
                rejection_reason = "topology_broken"
                run_dict["invalid_state_rejects_total"] += 1
                print("/!\\ /!\\ Topology evaluation broken /!\\ /!\\")
                evaluation_error = topology_status.get("evaluation_error")
                if evaluation_error:
                    print(f"Topology error: {evaluation_error}")
            else:
                topology_gate_penalty_scale = float(
                    globals().get("TOPOLOGY_GATE_PENALTY_SCALE", 4.0)
                )
                topology_contract = _evaluate_frontier_topology_search_contract_impl(
                    topology_status,
                    previous_objective=run_dict['J'],
                    penalty_scale=topology_gate_penalty_scale,
                )
                if topology_contract["reject"]:
                    if frontier_mode_enabled():
                        topology_penalty = _evaluate_frontier_topology_search_penalty_impl(
                            topology_status,
                            previous_objective=run_dict['J'],
                            penalty_scale=topology_gate_penalty_scale,
                        )
                        objective_eval = _apply_frontier_search_contract_penalties_impl(
                            objective_eval,
                            topology_penalty=topology_penalty,
                        )
                        print("/!\\ /!\\ Topology gate penalty active /!\\ /!\\")
                        print(
                            "Cheap field-line survival "
                            f"{topology_status['survived_lines']}/{topology_status['nfieldlines']} "
                            f"(fraction={topology_status['survival_fraction']:.3f}, "
                            f"threshold={topology_status['survival_threshold']:.3f})"
                        )
                        print(
                            "Topology frontier penalty = "
                            f"{topology_penalty['penalty']:.6e} "
                            f"(deficit={topology_penalty['deficit']:.6e})"
                        )
                        if topology_status['first_exit_time'] is not None:
                            print(
                                "First topology exit at "
                                f"t={topology_status['first_exit_time']:.6e}, "
                                f"phi={topology_status['first_exit_angle']:.6e}, "
                                f"reason={topology_status['first_exit_reason']}"
                            )
                    else:
                        success = False
                        rejection_reason = "topology"
                        run_dict["topology_gate_rejects"] += 1
                        rejection_increment = topology_contract["rejection_increment"]
                        print("/!\\ /!\\ Topology gate rejected candidate /!\\ /!\\")
                        print(
                            "Cheap field-line survival "
                            f"{topology_status['survived_lines']}/{topology_status['nfieldlines']} "
                            f"(fraction={topology_status['survival_fraction']:.3f}, "
                            f"threshold={topology_status['survival_threshold']:.3f})"
                        )
                        print(f"Topology rejection increment = {rejection_increment:.6e}")
                        if topology_status['first_exit_time'] is not None:
                            print(
                                "First topology exit at "
                                f"t={topology_status['first_exit_time']:.6e}, "
                                f"phi={topology_status['first_exit_angle']:.6e}, "
                                f"reason={topology_status['first_exit_reason']}"
                            )

        if success:
            hardware_snapshot_start = time.perf_counter()
            hardware_snapshot = evaluate_single_stage_search_hardware_snapshot(
                objective_eval,
                CC_DIST,
                CS_DIST,
                SS_DIST,
                CURVATURE_THRESHOLD,
                **current_single_stage_hardware_snapshot_kwargs(),
            )
            metrics["hardware_snapshot_seconds"] += (
                time.perf_counter() - hardware_snapshot_start
            )
            hardware_status = hardware_status_with_curvature_traversal(
                hardware_snapshot["search_hardware_status"],
                curvature_precheck_status,
            )
            run_dict["trial_hardware_status"] = hardware_status
            if not hardware_status["success"]:
                hardware_contract = _evaluate_frontier_hardware_search_contract_impl(
                    hardware_status,
                    policy=HardwareSearchPolicy(
                        HARDWARE_SEARCH_MODE,
                        HARDWARE_SEARCH_SOFT_ITERATIONS,
                    ),
                    context=SearchContext(
                        accepted_iterations=run_dict["accepted_iterations"],
                        gate_scale=search_gate["gate_scale"],
                        previous_objective=run_dict["J"],
                    ),
                )
                if hardware_contract["reject"]:
                    if frontier_mode_enabled():
                        hardware_penalty_scale = float(
                            globals().get("HARDWARE_SEARCH_PENALTY_SCALE", 4.0)
                        )
                        hardware_penalty = _evaluate_frontier_hardware_search_penalty_impl(
                            hardware_status,
                            previous_objective=run_dict["J"],
                            penalty_scale=hardware_penalty_scale,
                        )
                        objective_eval = _apply_frontier_search_contract_penalties_impl(
                            objective_eval,
                            hardware_penalty=hardware_penalty,
                        )
                        print("/!\\ /!\\ Hardware search penalty active /!\\ /!\\")
                        print(
                            "Hardware frontier penalty = "
                            f"{hardware_penalty['penalty']:.6e} "
                            f"(max_violation_ratio="
                            f"{hardware_penalty['max_violation_ratio']:.6e})"
                        )
                    elif repair_phase1_mode_active:
                        print(
                            "/!\\ /!\\ Repair-first phase1 keeping valid hardware-bad "
                            "candidate for feasibility reduction /!\\ /!\\"
                        )
                    else:
                        success = False
                        rejection_reason = "hardware"
                        run_dict["hardware_rejects"] += 1
                        run_dict["invalid_state_rejects_total"] += 1
                        rejection_increment = hardware_contract["rejection_increment"]
                        print("/!\\ /!\\ Hardware constraints violated /!\\ /!\\")
                elif hardware_contract["warning_only"]:
                    print("/!\\ /!\\ Hardware constraints violated (warning only) /!\\ /!\\")
                for violation in hardware_status["violations"]:
                    print(violation)

        if success:
            J = objective_eval['total']
            dJ = objective_eval['grad']
            run_dict['last_successful_eval'] = objective_eval
            run_dict['last_successful_eval_weights'] = np.asarray(search_surface_weights).copy()
            print(f"Volume: {outer_entry['boozer_surface'].surface.volume()}")
            print(f"Iota: {surface_iota_terms[-1].J()}")
            if len(surface_data) > 1:
                print(f"Surface search weights: {objective_eval['surface_weights'].tolist()}")
                print(f"Surface gate scale: {search_gate['gate_scale']:.6f}")
                print(f"Adjacent surface gaps: {stack_status['adjacent_gaps']}")
                print(f"Outer vessel gap: {stack_status['outer_vessel_gap']}")

    if not success:
        if rejection_reason is None and not stack_status['success']:
            run_dict["surface_solve_rejects"] += 1
            run_dict["invalid_state_rejects_total"] += 1
        if stack_status['success']:
            print("/!\\ /!\\ Candidate rejected after surface solve /!\\ /!\\")
        else:
            run_dict['topology_gate_status'] = disabled_topology_gate_status(
                TOPOLOGY_GATE_TMAX,
                TOPOLOGY_GATE_TOL,
                TOPOLOGY_GATE_SURVIVAL_THRESHOLD,
            )
            print("/!\\ /!\\ Boozer surface rejected /!\\ /!\\")
        if not all(stack_status['solve_success']):
            print("Boozer solver failed")
        if any(stack_status['self_intersections']):
            print("Surface is self-intersecting")
        if len(surface_data) > 1:
            if not stack_status['volumes_ordered']:
                print("Surface volumes are not strictly ordered")
            if not stack_status['gap_ok']:
                print(f"Adjacent surfaces too close: {stack_status['adjacent_gaps']}")
            if not stack_status['vessel_gap_ok']:
                print(f"Outer surface too close to vessel: {stack_status['outer_vessel_gap']}")
            if search_gate['enforce_nesting'] and not stack_status['nesting_ok']:
                print(f"Surfaces are not nested on phi slices: {stack_status['bad_nesting_phis']}")
        hardware_status = run_dict.get("trial_hardware_status")
        if hardware_status is not None and not hardware_status["success"]:
            print("Hardware constraints violated")
        banana_current_diagnostics = run_dict.get("banana_current_diagnostics")
        if banana_current_diagnostics is not None:
            rejected_trial_report = build_banana_current_coordinate_report(
                JF,
                banana_current_state,
                x,
                None if objective_eval is None else objective_eval.get("grad"),
                active_optimizer_bounds=run_dict.get("active_optimizer_bounds"),
                phase="rejected_trial",
                accepted_iteration=int(run_dict.get("accepted_iterations", 0)),
                line_search_evaluations=int(run_dict.get("lscount", 0)),
                step_norm=float(dx),
                rejection_reason=(
                    rejection_reason
                    if rejection_reason is not None
                    else "surface_solve"
                ),
            )
            finalize_banana_current_diagnostics_report(
                OUT_DIR_ITER,
                banana_current_diagnostics,
                rejected_trial_report,
                rejected_trial=True,
            )

        # Elevated J violates Armijo, so the line search backtracks.
        # Returning dJ_old (not negated) avoids the old -dJ corruption path
        # and produces y_k=0 if the step is ever accepted, safely skipping
        # the BFGS Hessian update.
        if rejection_increment is None:
            rejection_increment = max(abs(run_dict['J']), 1.0)
        record_search_step_rejection(
            metrics,
            rejection_reason=rejection_reason,
            stack_status=stack_status,
            hardware_status=hardware_status,
            rejection_increment=rejection_increment,
        )
        J = run_dict['J'] + rejection_increment
        dJ = run_dict['dJ'].copy()
        JF.x = run_dict['accepted_x']
        restore_surface_states(surface_data, run_dict['surface_state'])
    else:
        record_search_step_acceptance(metrics)

    evaluation = {"total": J, "grad": dJ}
    if CONSTRAINT_METHOD == "alm":
        metric_eval = objective_eval
        if metric_eval is None or "constraint_values" not in metric_eval:
            metric_eval = run_dict.get("last_successful_eval", run_dict.get("search_eval"))
        if metric_eval is not None and "constraint_values" in metric_eval:
            evaluation.update(
                {
                    "constraint_values": np.asarray(metric_eval["constraint_values"], dtype=float),
                    "max_violation": float(metric_eval["max_violation"]),
                    "stationarity_norm": float(metric_eval["stationarity_norm"]),
                    "metric_grad": np.asarray(
                        metric_eval.get("grad", dJ),
                        dtype=float,
                    ),
                    "metric_stationarity_norm": float(metric_eval["stationarity_norm"]),
                    "constraint_names": list(metric_eval.get("constraint_names", [])),
                    "constraint_grads": [
                        np.asarray(grad, dtype=float)
                        for grad in metric_eval.get("constraint_grads", [])
                    ],
                    "constraint_activity_tolerances": np.asarray(
                        metric_eval.get("constraint_activity_tolerances", []),
                        dtype=float,
                    ),
                    "feasibility_values": np.asarray(
                        metric_eval.get("feasibility_values", metric_eval["constraint_values"]),
                        dtype=float,
                    ),
                    "dual_update_values": np.asarray(
                        metric_eval.get("dual_update_values", metric_eval["constraint_values"]),
                        dtype=float,
                    ),
                    "max_feasibility_violation": float(
                        metric_eval.get(
                            "max_feasibility_violation",
                            metric_eval["max_violation"],
                        )
                    ),
                    "base_total": float(
                        metric_eval.get(
                            "base_total",
                            metric_eval.get("total", run_dict["J"]),
                        )
                    ),
                }
            )
    metrics["total_seconds"] += time.perf_counter() - step_start
    return evaluation


def fun(x):
    evaluation = evaluate_search_step(x)
    return evaluation["total"], evaluation["grad"]


def search_eval_has_diagnostics(objective_eval):
    return bool(
        objective_eval is not None
        and objective_eval.get("diagnostics_included", True)
        and all(
            field_name in objective_eval
            for field_name in (
                "J_QS",
                "dJ_QS",
                "J_Boozer",
                "dJ_Boozer",
                "J_iota",
                "dJ_iota",
                "J_surf",
                "dJ_surf",
                "J_curvature",
                "dJ_curvature",
            )
        )
    )


def callback(x):
    """
    Callback function executed after each successful optimization iteration.

    Stores the accepted state (surface DOFs, iota, G), evaluates and prints
    detailed diagnostics for all objective function components, and logs the
    iteration summary to file. Used for monitoring optimization progress and
    recording convergence history.

    Args:
        x: Current degrees of freedom (coil parameters) from accepted step
    """
    # Update count for tracking
    run_dict['lscount'] = 0
    run_dict["curvature_overcap_boozer_evals_this_iteration"] = 0
    outer_entry = surface_data[-1]

    # Store last accepted state
    search_surface_weights = build_surface_search_weights(
        len(surface_data),
        run_dict['accepted_iterations'],
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
    )
    search_gate = build_surface_search_gate(
        len(surface_data),
        run_dict['accepted_iterations'],
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
        SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
        SS_DIST if len(surface_data) > 1 else 0.0,
    )
    if (
        'last_successful_eval' in run_dict
        and np.array_equal(
            run_dict.get('last_successful_eval_weights', None),
            search_surface_weights,
        )
        and search_eval_has_diagnostics(run_dict['last_successful_eval'])
    ):
        objective_eval = run_dict['last_successful_eval']
    else:
        objective_eval = evaluate_search_objective(
            search_surface_weights,
            include_diagnostics=True,
        )
    run_dict['surface_state'] = snapshot_surface_states(surface_data)
    run_dict['accepted_x'] = x.copy()
    run_dict['J'] = objective_eval['total']
    run_dict['dJ'] = objective_eval['grad'].copy()
    run_dict['search_eval'] = objective_eval
    topology_status = run_dict['topology_gate_status']
    search_stack_status = evaluate_surface_stack(
        surface_data,
        vessel_surface=VV if len(surface_data) > 1 else None,
        surface_gap_threshold=search_gate['surface_gap_threshold'],
        vessel_gap_threshold=search_gate['vessel_gap_threshold'],
        enforce_nesting=search_gate['enforce_nesting'],
    )
    full_stack_status = evaluate_surface_stack(
        surface_data,
        vessel_surface=VV if len(surface_data) > 1 else None,
        surface_gap_threshold=SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
        vessel_gap_threshold=SS_DIST if len(surface_data) > 1 else 0.0,
        enforce_nesting=True,
    )
    run_dict['search_surface_status'] = search_stack_status
    run_dict['surface_status'] = full_stack_status
    run_dict['topology_gate_status'] = topology_status

    # Evaluate diagnostics
    J = run_dict['J']
    grad = run_dict['dJ']
    banana_current_diagnostics = run_dict.get("banana_current_diagnostics")
    if banana_current_diagnostics is not None:
        accepted_report = build_banana_current_coordinate_report(
            JF,
            banana_current_state,
            x,
            grad,
            active_optimizer_bounds=run_dict.get("active_optimizer_bounds"),
            phase="accepted",
            accepted_iteration=int(run_dict["accepted_iterations"]) + 1,
        )
        finalize_banana_current_diagnostics_report(
            OUT_DIR_ITER,
            banana_current_diagnostics,
            accepted_report,
        )
    
    J_QS = objective_eval['J_QS']
    dJ_QS = np.linalg.norm(objective_eval['dJ_QS'])
    J_Boozer = objective_eval['J_Boozer']
    dJ_Boozer = np.linalg.norm(objective_eval['dJ_Boozer'])
    J_iota = objective_eval['J_iota']
    dJ_iota = np.linalg.norm(objective_eval['dJ_iota'])
    J_volume = objective_eval.get('J_volume', 0.0)
    dJ_volume = np.linalg.norm(objective_eval.get('dJ_volume', np.zeros_like(grad)))
    J_len = JCurveLength.J()
    dJ_len = np.linalg.norm(JCurveLength.dJ())
    J_cc = JCurveCurve.J()
    dJ_cc = np.linalg.norm(JCurveCurve.dJ())
    J_cs = JCurveSurface.J()
    dJ_cs = np.linalg.norm(JCurveSurface.dJ())
    J_surf = objective_eval['J_surf']
    dJ_surf = np.linalg.norm(objective_eval['dJ_surf'])
    J_curvature = objective_eval['J_curvature']
    dJ_curvature = np.linalg.norm(objective_eval['dJ_curvature'])

    iota_values = [term.J() for term in surface_iota_terms]
    volume_values = [entry['boozer_surface'].surface.volume() for entry in surface_data]
    iota_str = ", ".join(f"{value:.4f}" for value in iota_values)
    volume_str = ", ".join(f"{value:.4f}" for value in volume_values)

    gamma = banana_curve.gamma()
    max_r = np.max(np.sqrt(gamma[:,0]**2 + gamma[:,1]**2))
    max_z = np.max(np.abs(gamma[:,2]))
    length = curvelength.J()
    hardware_snapshot = evaluate_single_stage_hardware_snapshot(
        JCurveCurve,
        CC_DIST,
        JCurveSurface,
        CS_DIST,
        JSurfSurf,
        full_stack_status,
        SS_DIST,
        banana_curve,
        CURVATURE_THRESHOLD,
        outer_entry["boozer_surface"].surface,
        VV,
        **current_single_stage_hardware_snapshot_kwargs(coil_length=length),
    )
    curvecurve_min = hardware_snapshot["curve_curve_min_dist"]
    curvesurf_min = hardware_snapshot["curve_surface_min_dist"]
    surface_vessel_min = hardware_snapshot["surface_vessel_min_dist"]
    max_curvature = hardware_snapshot["max_curvature"]
    hardware_status = hardware_snapshot["search_hardware_status"]
    run_dict['accepted_hardware_status'] = hardware_status
    incumbent_stage = run_dict.get("accepted_boozer_stage", globals().get("stage", "initial"))
    run_dict["accepted_boozer_stage"] = incumbent_stage
    run_dict['intersecting'] = any(full_stack_status['self_intersections'])
    best_accepted_updated = maybe_update_best_accepted_incumbent(run_dict, incumbent_stage)
    best_feasible_updated = maybe_update_best_feasible_incumbent(run_dict, incumbent_stage)

    field_error, BdotN = compute_surface_field_metrics(
        outer_entry['boozer_surface'].surface,
        bs,
    )
    accepted_iteration = int(run_dict['accepted_iterations'] + 1)
    if (
        SINGLE_STAGE_GOAL_MODE == "frontier"
        and run_dict.get("frontier_conditioning_first_accepted_report") is None
    ):
        run_dict["frontier_conditioning_first_accepted_report"] = (
            build_frontier_conditioning_report(
                objective_eval,
                sample_label="first_accepted",
            )
        )
    if best_accepted_updated:
        write_preserved_timeout_artifacts(
            OUT_DIR_ITER,
            preservation_kind="best_accepted",
            results_payload=build_preserved_timeout_results_payload(
                replay_config=PRESERVED_TIMEOUT_REPLAY_CONFIG,
                preservation_kind="best_accepted",
                incumbent_stage=incumbent_stage,
                run_dict=run_dict,
                objective_eval=objective_eval,
                field_error=field_error,
                final_iota=iota_values[-1],
                final_volume=volume_values[-1],
                hardware_snapshot=hardware_snapshot,
                banana_current_state=globals().get("banana_current_state"),
                coil_length=length,
                accepted_iteration=accepted_iteration,
                alm_runtime_state=current_preserved_timeout_alm_state(),
            ),
            biotsavart=bs,
            surface_data=surface_data,
        )
    if best_feasible_updated:
        write_preserved_timeout_artifacts(
            OUT_DIR_ITER,
            preservation_kind="best_feasible",
            results_payload=build_preserved_timeout_results_payload(
                replay_config=PRESERVED_TIMEOUT_REPLAY_CONFIG,
                preservation_kind="best_feasible",
                incumbent_stage=incumbent_stage,
                run_dict=run_dict,
                objective_eval=objective_eval,
                field_error=field_error,
                final_iota=iota_values[-1],
                final_volume=volume_values[-1],
                hardware_snapshot=hardware_snapshot,
                banana_current_state=globals().get("banana_current_state"),
                coil_length=length,
                accepted_iteration=accepted_iteration,
                alm_runtime_state=current_preserved_timeout_alm_state(),
            ),
            biotsavart=bs,
            surface_data=surface_data,
        )

    width = 35
    buffer = io.StringIO()
    print("="*70, file=buffer)
    print(f"ITERATION {run_dict['it']}", file=buffer)
    print(f"{'Objective J':{width}} = {J:.6e}", file=buffer)
    if CONSTRAINT_METHOD == "alm":
        print(f"{'Base Objective J':{width}} = {objective_eval['base_total']:.6e}", file=buffer)
        print(f"{'ALM outer iter':{width}} = {run_dict.get('alm_outer_iteration')}", file=buffer)
        print(f"{'ALM penalty μ':{width}} = {ALM_PENALTY:.6e}", file=buffer)
        print(f"{'ALM multipliers':{width}} = {ALM_MULTIPLIERS.tolist()}", file=buffer)
    print(f"{'||∇J||':{width}} = {np.linalg.norm(grad):.6e}", file=buffer)
    print(f"{'nonQS ratio':{width}} = {J_QS:.6e} (dJ = {dJ_QS:.6e})", file=buffer)
    print(f"{'Boozer Residual':{width}} = {J_Boozer:.6e} (dJ = {dJ_Boozer:.6e})", file=buffer)
    iota_term_label = "ι Reward" if SINGLE_STAGE_GOAL_MODE == "frontier" else "ι Penalty"
    print(f"{iota_term_label:{width}} = {J_iota:.6e} (dJ = {dJ_iota:.6e})", file=buffer)
    if SINGLE_STAGE_GOAL_MODE == "frontier":
        print(f"{'Volume Reward':{width}} = {J_volume:.6e} (dJ = {dJ_volume:.6e})", file=buffer)
        print(
            f"{'Frontier Rank J':{width}} = "
            f"{objective_eval.get('frontier_rank_total', J):.6e}",
            file=buffer,
        )
        print(
            f"{'Frontier Trust OK':{width}} = "
            f"{objective_eval.get('frontier_trust_ok')}",
            file=buffer,
        )
        print(
            f"{'Frontier Boozer Threshold':{width}} = "
            f"{objective_eval.get('frontier_boozer_trust_threshold')}",
            file=buffer,
        )
        print(
            f"{'Frontier Trust Penalty':{width}} = "
            f"{objective_eval.get('frontier_trust_penalty')}",
            file=buffer,
        )
        print(
            f"{'Frontier Trust Excess Ratio':{width}} = "
            f"{objective_eval.get('frontier_boozer_trust_excess_ratio')}",
            file=buffer,
        )
    print(f"{'Iotas (actual)':{width}} = {iota_str}", file=buffer)
    print(f"{'Volume':{width}} = {volume_str}", file=buffer)
    print(f"{'Curve Length Penalty':{width}} = {J_len:.6e} (dJ = {dJ_len:.6e})", file=buffer)
    print(f"{'Curve-Curve Penalty':{width}} = {J_cc:.6e} (min={curvecurve_min:.3e}) (dJ = {dJ_cc:.6e})", file=buffer)
    print(f"{'Curve-Surface Penalty':{width}} = {J_cs:.6e} (min={curvesurf_min:.3e}) (dJ = {dJ_cs:.6e})", file=buffer)
    print(f"{'Surf-Vessel Penalty':{width}} = {J_surf:.6e} (dJ = {dJ_surf:.6e})", file=buffer) 
    print(f"{'Curvature Penalty':{width}} = {J_curvature:.6e} (dJ = {dJ_curvature:.6e})", file=buffer) 
    print(f"{'⟨|B·n|⟩':{width}} = {BdotN:.6e}", file=buffer)
    if len(surface_data) > 1:
        print(f"{'Surface search weights':{width}} = {objective_eval['surface_weights'].tolist()}", file=buffer)
        print(f"{'Surface gate scale':{width}} = {search_gate['gate_scale']:.6f}", file=buffer)
        print(f"{'Search gap threshold':{width}} = {search_gate['surface_gap_threshold']:.6e}", file=buffer)
        print(f"{'Search vessel gap':{width}} = {search_gate['vessel_gap_threshold']:.6e}", file=buffer)
        print(f"{'Search nesting enforced':{width}} = {search_gate['enforce_nesting']}", file=buffer)
        if topology_status['enabled']:
            print(
                f"{'Topology survival':{width}} = "
                f"{topology_status['survived_lines']}/{topology_status['nfieldlines']} "
                f"(fraction={topology_status['survival_fraction']:.6f}, "
                f"threshold={topology_status['survival_threshold']:.6f})",
                file=buffer,
            )
            print(f"{'Topology stop counts':{width}} = {topology_status['stop_reason_counts']}", file=buffer)
            print(f"{'Topology first exit time':{width}} = {topology_status['first_exit_time']}", file=buffer)
            print(f"{'Topology first exit angle':{width}} = {topology_status['first_exit_angle']}", file=buffer)
            print(f"{'Topology first exit reason':{width}} = {topology_status['first_exit_reason']}", file=buffer)
        print(f"{'Adjacent surface gaps':{width}} = {full_stack_status['adjacent_gaps']}", file=buffer)
        print(f"{'Outer vessel gap':{width}} = {full_stack_status['outer_vessel_gap']:.6e}", file=buffer)
        print(f"{'Surfaces nested':{width}} = {full_stack_status['nesting_ok']}", file=buffer)
        print(f"{'Bad nesting phis':{width}} = {full_stack_status['bad_nesting_phis']}", file=buffer)

    print(f"{'Intersecting':{width}} = {run_dict['intersecting']}", file=buffer)
    print(f"{'Max Curve R':{width}} = {max_r:.6e}", file=buffer)
    print(f"{'Max Curve Z':{width}} = {max_z:.6e}", file=buffer)
    print(f"{'Max Curvature':{width}} = {max_curvature:.6e}", file=buffer)
    print(f"{'Curve Length':{width}} = {length:.6e}", file=buffer)
    print(f"{'Surface-Vessel Min Dist':{width}} = {surface_vessel_min:.6e}", file=buffer)
    print(f"{'Hardware Constraints OK':{width}} = {hardware_status['success']}", file=buffer)
    if hardware_status["violations"]:
        print(f"{'Hardware Violations':{width}} = {hardware_status['violations']}", file=buffer)
    print("="*70, file=buffer)

    output_str = buffer.getvalue()
    buffer.close()

    print(output_str)

    filename = OUT_DIR_ITER + "/log.txt"
    with open(filename, "a") as f:
        f.write(output_str + "\n")

    # Advance iteration counter
    run_dict['accepted_iterations'] += 1
    run_dict['it'] += 1
    banana_current_replay_context = run_dict.get("banana_current_replay_context")
    if banana_current_replay_context is not None:
        record_banana_current_replay_context_snapshot(
            banana_current_replay_context,
            accepted_iteration=int(run_dict["accepted_iterations"]),
            accepted_boozer_stage=incumbent_stage,
            incumbent=snapshot_single_stage_incumbent_state(run_dict),
        )
        write_banana_current_replay_context_artifact(
            OUT_DIR_ITER,
            banana_current_replay_context,
        )

    if CHECKPOINT_EVERY > 0:
        write_single_stage_solver_checkpoint_state(
            OUT_DIR_ITER,
            run_dict,
            requested_maxiter=MAXITER,
            runtime_maxiter=RUNTIME_MAXITER,
            accepted_stage=incumbent_stage,
            goal_mode=SINGLE_STAGE_GOAL_MODE,
            constraint_method=CONSTRAINT_METHOD,
            stage2_bs_path=stage2_bs_path,
            out_dir_iter=OUT_DIR_ITER,
            alm_state=current_solver_checkpoint_alm_state(),
        )

    # Periodic checkpoint saving
    if CHECKPOINT_EVERY > 0 and run_dict['accepted_iterations'] % CHECKPOINT_EVERY == 0:
        ckpt_dir = os.path.join(OUT_DIR_ITER, f"checkpoint_iter{run_dict['accepted_iterations']:04d}")
        os.makedirs(ckpt_dir, exist_ok=True)
        bs.save(os.path.join(ckpt_dir, "biot_savart.json"))
        save_surface_artifacts(surface_data, bs, ckpt_dir, "surf", also_write_outer_legacy=False)
        print(f"  [checkpoint] Saved iteration {run_dict['accepted_iterations']} to {ckpt_dir}")

    # Periodic topology scoring (medium-fidelity confinement evaluation)
    if TOPOLOGY_SCORER_EVERY > 0 and run_dict['accepted_iterations'] % TOPOLOGY_SCORER_EVERY == 0:
        outer_surf = outer_surface_data['boozer_surface'].surface
        topo_result = safe_score_topology(
            outer_surf,
            bs,
            nfieldlines=TOPOLOGY_SCORER_NFIELDLINES,
            tmax=TOPOLOGY_SCORER_TMAX,
            **confinement_surrogate_kwargs(),
        )
        checkpoint_objective_total = (
            np.inf
            if topo_result["broken"]
            else checkpoint_confinement_objective(
                J,
                topo_result,
                CONFINEMENT_OBJECTIVE_WEIGHT,
            )
        )
        topo_entry = {
            "accepted_iteration": run_dict['accepted_iterations'],
            "J": float(J),
            "checkpoint_objective_total": checkpoint_objective_total,
            "topology_state": topo_result["evaluation_state"],
            "topology_broken": bool(topo_result["broken"]),
            "topology_error": topo_result.get("evaluation_error"),
            "topology_error_type": topo_result.get("evaluation_error_type"),
            "survival_fraction": topo_result["survival_fraction"],
            "survived_lines": topo_result["survived_lines"],
            "nfieldlines": topo_result["nfieldlines"],
            "tmax": topo_result["tmax"],
            "mean_exit_time": topo_result["mean_exit_time"],
            "confinement_score": topo_result["confinement_score"],
            "mean_line_loss": topo_result["mean_line_loss"],
            "worst_k_line_loss": topo_result["worst_k_line_loss"],
            "early_exit_fraction": topo_result["early_exit_fraction"],
            "confinement_loss": topo_result["confinement_loss"],
            "confinement_surrogate_k": topo_result["confinement_surrogate_k"],
            "confinement_early_exit_threshold": topo_result["confinement_early_exit_threshold"],
            "stop_reason_counts": topo_result["stop_reason_counts"],
            "first_exit": topo_result.get("first_exit"),
            "per_phi_hit_counts": topo_result.get("per_phi_hit_counts"),
            "line_metrics": topo_result.get("line_metrics"),
            "line_lifetimes": topo_result.get("line_lifetimes"),
            "line_losses": topo_result.get("line_losses"),
            "seed_contract": topo_result.get("seed_contract"),
            "field_model": topo_result.get("field_model"),
            "transport_diagnostics": topo_result.get("transport_diagnostics"),
        }
        # Append to archive JSONL
        archive_path = os.path.join(OUT_DIR_ITER, "topology_archive.jsonl")
        append_jsonl_artifact(archive_path, topo_entry)

        # Track best states
        if (
            not topo_result["broken"]
            and (
                'best_topology' not in run_dict
                or topo_entry['confinement_score'] > run_dict['best_topology']['confinement_score']
            )
        ):
            run_dict['best_topology'] = topo_entry
            best_dir = os.path.join(OUT_DIR_ITER, "best_topology")
            write_topology_checkpoint_artifacts(
                best_dir,
                artifact_role="best_topology_checkpoint",
                topology_entry=topo_entry,
                biotsavart=bs,
                surface_data=surface_data,
            )

        if (
            not topo_result["broken"]
            and CONFINEMENT_OBJECTIVE_WEIGHT > 0.0
            and (
                'best_confinement_objective' not in run_dict
                or topo_entry['checkpoint_objective_total'] < run_dict['best_confinement_objective']['checkpoint_objective_total']
            )
        ):
            run_dict['best_confinement_objective'] = topo_entry
            best_dir = os.path.join(OUT_DIR_ITER, "best_confinement_objective")
            write_topology_checkpoint_artifacts(
                best_dir,
                artifact_role="best_confinement_objective_checkpoint",
                topology_entry=topo_entry,
                biotsavart=bs,
                surface_data=surface_data,
            )

        if topo_result["broken"]:
            print(
                f"  [topology] iter={run_dict['accepted_iterations']}: "
                f"broken ({topo_result.get('evaluation_error_type')}: "
                f"{topo_result.get('evaluation_error')})"
            )
        else:
            print(
                f"  [topology] iter={run_dict['accepted_iterations']}: "
                f"survival={topo_result['survived_lines']}/{topo_result['nfieldlines']}, "
                f"confinement={topo_result['confinement_score']:.4f}, "
                f"loss={topo_result['confinement_loss']:.4f}, "
                f"mean_exit={topo_result['mean_exit_time']}"
            )


# Convergence tolerances for different mpol values (module-level for testability)
MULTISURFACE_RAMP_ITERATIONS = 0
INNER_SURFACE_INITIAL_WEIGHT = 1.0
TOPOLOGY_GATE_FIELDLINES = 0
TOPOLOGY_GATE_TMAX = 2.0
TOPOLOGY_GATE_TOL = 1e-7
TOPOLOGY_GATE_SURVIVAL_THRESHOLD = 0.0
CONSTRAINT_METHOD = "penalty"
SINGLE_STAGE_GOAL_MODE = "target"
ALM_FORMULATION = "weighted_sum"
ALM_MULTIPLIERS = np.zeros(0, dtype=float)
ALM_PENALTY = 1.0
JVolume = None
JnonQSRatioObjective = None
JBoozerResidualObjective = None
EFFECTIVE_RES_WEIGHT = 0.0
EFFECTIVE_IOTAS_WEIGHT = 0.0
EFFECTIVE_VOLUME_WEIGHT = 0.0
FRONTIER_GOAL_CONFIG = None
JF = None
CHECKPOINT_EVERY = 0
TOPOLOGY_SCORER_EVERY = 0
TOPOLOGY_SCORER_NFIELDLINES = 12
TOPOLOGY_SCORER_TMAX = 50.0
CONFINEMENT_OBJECTIVE_WEIGHT = 0.0
CONFINEMENT_SURROGATE_WORST_K = 3
CONFINEMENT_SURROGATE_EARLY_THRESHOLD = 0.2
CONFINEMENT_SURROGATE_MEAN_WEIGHT = 0.2
CONFINEMENT_SURROGATE_WORST_WEIGHT = 0.6
CONFINEMENT_SURROGATE_EARLY_WEIGHT = 0.2


def validate_stage2_seed_contract(stage2_results):
    _validate_stage2_seed_contract_impl(stage2_results)


if __name__ == "__main__":
    # ==============================================================================
    # CONFIGURATION PARAMETERS
    # ==============================================================================
    args = apply_default_stage2_seed_args(parse_args())
    if args.banana_current_fd_diagnostics:
        args.banana_current_diagnostics = True
    if args.banana_current_fd_relative_step_fraction <= 0.0:
        raise ValueError(
            "--banana-current-fd-relative-step-fraction must be positive."
        )
    stage2_bs_path = build_stage2_bs_path(args)
    stage2_results_path, stage2_results = load_stage2_artifact_results(stage2_bs_path)
    stage2_results = upgrade_legacy_stage2_artifact_results(
        stage2_results,
        known_num_tf_coils=args.num_tf_coils,
        known_tf_current_A=args.stage2_seed_tf_current_A,
    )
    validate_stage2_seed_contract(stage2_results)
    R0 = validate_major_radius(
        float(stage2_results["MAJOR_RADIUS"]),
        accept_offspec=args.accept_offspec_r0_seed,
    )
    s = float(stage2_results["TOROIDAL_FLUX"])
    order = int(stage2_results.get("order", args.stage2_seed_order))

    banana_surf_radius = resolve_single_stage_banana_surf_radius(
        stage2_results,
        args.banana_surf_radius,
    )
    nphi = args.nphi
    ntheta = args.ntheta
    mpol = args.mpol
    ntor = args.ntor

    # Optimization targets and weights
    vol_target = args.vol_target
    CONSTRAINT_WEIGHT = None if args.constraint_weight < 0 else args.constraint_weight
    CONSTRAINT_METHOD = args.constraint_method
    SINGLE_STAGE_GOAL_MODE = args.single_stage_goal_mode
    ALM_FORMULATION = args.alm_formulation
    ALM_MULTIPLIERS = np.zeros(0, dtype=float)
    ALM_PENALTY = args.alm_penalty_init
    requested_finite_current_mode = getattr(
        args,
        "finite_current_mode",
        DEFAULT_FINITE_CURRENT_MODE,
    )
    surface_mode_contract = resolve_surface_mode_contract(args)
    validate_surface_mode_runtime_support(surface_mode_contract)
    effective_num_surfaces = surface_mode_contract.num_surfaces
    effective_inner_surface_ratio = resolve_surface_mode_inner_surface_ratio(
        surface_mode_contract,
        fallback_inner_surface_ratio=args.inner_surface_ratio,
    )
    finite_current_mode = resolve_stage2_finite_current_mode(
        stage2_results,
        requested_finite_current_mode,
    )
    plasma_current_settings = resolve_plasma_current_settings(
        args,
        finite_current_mode=finite_current_mode,
        default_plasma_current_A=float(stage2_results.get("PROXY_PLASMA_CURRENT_A", 0.0)),
        num_surfaces=effective_num_surfaces,
    )
    boozer_I = plasma_current_settings["boozer_I"]
    plasma_current_A = plasma_current_settings["plasma_current_A"]
    plasma_current_input_source = plasma_current_settings["input_source"]
    boozer_current_convention = plasma_current_settings["boozer_current_convention"]
    finite_current_mode = plasma_current_settings["mode"]
    effective_current_mode = plasma_current_settings["effective_mode"]
    MAXITER = args.maxiter
    resume_solver_checkpoint_payload = (
        None
        if args.resume_solver_checkpoint is None
        else load_solver_checkpoint(args.resume_solver_checkpoint)
    )
    RUNTIME_MAXITER = (
        MAXITER
        if resume_solver_checkpoint_payload is None
        else int(resume_solver_checkpoint_payload["remaining_maxiter"])
    )
    CHECKPOINT_EVERY = args.checkpoint_every
    TOPOLOGY_SCORER_EVERY = args.topology_scorer_every
    TOPOLOGY_SCORER_NFIELDLINES = args.topology_scorer_nfieldlines
    TOPOLOGY_SCORER_TMAX = args.topology_scorer_tmax
    CONFINEMENT_OBJECTIVE_WEIGHT = args.confinement_objective_weight
    CONFINEMENT_SURROGATE_WORST_K = args.confinement_surrogate_worst_k
    CONFINEMENT_SURROGATE_EARLY_THRESHOLD = args.confinement_surrogate_early_threshold
    CONFINEMENT_SURROGATE_MEAN_WEIGHT = args.confinement_surrogate_mean_weight
    CONFINEMENT_SURROGATE_WORST_WEIGHT = args.confinement_surrogate_worst_weight
    CONFINEMENT_SURROGATE_EARLY_WEIGHT = args.confinement_surrogate_early_weight
    iota_target = args.iota_target
    num_tf_coils = resolve_stage2_num_tf_coils(stage2_results, args.num_tf_coils)
    if not (0.0 <= args.inner_surface_initial_weight <= 1.0):
        raise ValueError("--inner-surface-initial-weight must be between 0 and 1")
    if args.multisurface_ramp_iterations < 0:
        raise ValueError("--multisurface-ramp-iterations must be non-negative")
    if not (0.0 < args.multisurface_initial_step_scale <= 1.0):
        raise ValueError("--multisurface-initial-step-scale must be in (0, 1]")
    if args.multisurface_initial_step_maxiter < 0:
        raise ValueError("--multisurface-initial-step-maxiter must be non-negative")
    if args.topology_gate_fieldlines < 0:
        raise ValueError("--topology-gate-fieldlines must be non-negative")
    if args.topology_gate_tmax <= 0:
        raise ValueError("--topology-gate-tmax must be positive")
    if args.topology_gate_tol <= 0:
        raise ValueError("--topology-gate-tol must be positive")
    if not (0.0 <= args.topology_gate_survival_threshold <= 1.0):
        raise ValueError("--topology-gate-survival-threshold must be between 0 and 1")
    if args.topology_gate_penalty_scale < 0.0:
        raise ValueError("--topology-gate-penalty-scale must be non-negative")
    if args.hardware_search_soft_iterations < 0:
        raise ValueError("--hardware-search-soft-iterations must be non-negative")
    if args.curvature_traversal_band < 0.0:
        raise ValueError("--curvature-traversal-band must be non-negative")
    if args.curvature_traversal_eval_budget < 0:
        raise ValueError("--curvature-traversal-eval-budget must be non-negative")
    if (
        (
            args.curvature_traversal_band > 0.0
            or args.curvature_traversal_eval_budget > 0
        )
        and args.curvature_threshold <= 0.0
    ):
        raise ValueError("--curvature-threshold must be positive for curvature traversal")
    validate_alm_cli_args(args)
    validate_single_stage_alm_formulation_args(args)
    validate_confinement_surrogate_args(args)
    validate_single_stage_current_args(args)
    validate_boozer_stage_refinement_args(
        args,
        CONSTRAINT_WEIGHT,
        surface_mode_contract=surface_mode_contract,
    )
    MULTISURFACE_RAMP_ITERATIONS = args.multisurface_ramp_iterations
    INNER_SURFACE_INITIAL_WEIGHT = args.inner_surface_initial_weight
    TOPOLOGY_GATE_FIELDLINES = args.topology_gate_fieldlines
    TOPOLOGY_GATE_TMAX = args.topology_gate_tmax
    TOPOLOGY_GATE_TOL = args.topology_gate_tol
    TOPOLOGY_GATE_SURVIVAL_THRESHOLD = args.topology_gate_survival_threshold
    TOPOLOGY_GATE_PENALTY_SCALE = args.topology_gate_penalty_scale
    HARDWARE_SEARCH_MODE = args.hardware_search_mode
    HARDWARE_SEARCH_SOFT_ITERATIONS = args.hardware_search_soft_iterations
    CURVATURE_TRAVERSAL_BAND = args.curvature_traversal_band
    CURVATURE_TRAVERSAL_EVAL_BUDGET = args.curvature_traversal_eval_budget

    # Output directory setup
    OUT_DIR = args.output_root
    os.makedirs(OUT_DIR, exist_ok=True)
    stage = args.boozer_stage

    # ==============================================================================
    # LOAD EQUILIBRIUM AND COILS
    # ==============================================================================
    plasma_surf_filename = args.plasma_surf_filename
    file_loc = build_equilibrium_path(args)
    stage2_seed_surf_path = _resolved_optional_path_string(args.stage2_seed_surf_path)
    bs, coil_partitions = load_stage2_seed_biot_savart(
        stage2_bs_path,
        stage2_results=stage2_results,
        num_tf_coils=num_tf_coils,
        seed_order_upgrade=getattr(args, "seed_order_upgrade", None),
    )
    PRESERVED_TIMEOUT_REPLAY_CONFIG = PreservedTimeoutReplayConfig(
        plasma_surf_filename=plasma_surf_filename,
        plasma_surf_path=file_loc,
        stage2_bs_path=str(stage2_bs_path),
        stage2_seed_surf_path=stage2_seed_surf_path,
        stage2_results_path=str(stage2_results_path),
        mpol=mpol,
        ntor=ntor,
        nphi=nphi,
        ntheta=ntheta,
        constraint_weight=CONSTRAINT_WEIGHT,
        constraint_method=CONSTRAINT_METHOD,
        alm_formulation=ALM_FORMULATION,
        max_iterations=MAXITER,
        target_volume=vol_target,
        target_iota=iota_target,
        single_stage_goal_mode=args.single_stage_goal_mode,
        single_stage_banana_current_mode=args.single_stage_banana_current_mode,
        single_stage_banana_current_coordinate_scaling=(
            args.single_stage_banana_current_coordinate_scaling
        ),
        num_banana_current_controls=None,
        single_stage_goal_mode_impl=current_frontier_goal_mode_impl(),
        major_radius=R0,
    )

    # Initialize the boundary magnetic surface and scale it to the target major radius
    surface_configs = build_surface_configs(
        file_loc,
        nphi,
        ntheta,
        s,
        R0,
        vol_target,
        effective_num_surfaces,
        effective_inner_surface_ratio,
    )
    warm_start_surface_stem = _resolved_optional_path_string(
        args.warm_start_surface_stem
    )
    surf = surface_configs[-1]["initial_surface"]
    banana_surf_nfp = surf.nfp

    (
        VV,
        lcfs_clearance_reference,
        surf_coils,
    ) = build_hbt_reference_surfaces(banana_surf_nfp, banana_surf_radius)

    bs, coil_partitions, banana_current_state = resolve_single_stage_banana_current_state(
        bs,
        coil_partitions,
        mode=args.single_stage_banana_current_mode,
        coordinate_scaling=args.single_stage_banana_current_coordinate_scaling,
    )
    PRESERVED_TIMEOUT_REPLAY_CONFIG = replace(
        PRESERVED_TIMEOUT_REPLAY_CONFIG,
        single_stage_banana_current_coordinate_scaling=(
            banana_current_state.coordinate_scaling
        ),
        num_banana_current_controls=banana_current_state.num_control_currents(),
    )

    # Extract coil information
    coils = bs.coils
    curves = [c.curve for c in coils]
    tf_coils = list(coil_partitions.tf_coils)
    banana_coils = list(coil_partitions.banana_coils)
    banana_curves = [c.curve for c in banana_coils]
    banana_curve = banana_curves[0]
    order = int(banana_curve.order)
    # Clearance/length objectives operate on the optimizable banana curves only;
    # TF/proxy/VF curves are fixed field sources and must not enter the penalty.
    objective_curves = banana_curves
    stage2_tf_current_A = resolve_stage2_tf_current_A(stage2_results, tf_coils)
    tf_current_sum_abs_A = float(sum(abs(c.current.get_value()) for c in tf_coils))
    initial_banana_current_A = banana_current_state.compatibility_current_A()
    if CONSTRAINT_METHOD == "penalty":
        apply_single_stage_penalty_banana_current_bounds(
            banana_current_state,
            banana_current_max_A=args.banana_current_max_A,
            validate_seed=not args.init_only,
            seed_context="Loaded Stage 2 banana current",
        )
    # ALM now checks banana current as a final feasibility constraint as well,
    # but only penalty/L-BFGS-B mode keeps the hard inner box bound that forbids
    # infeasible traversal during the search itself.
    # Keep the toroidal-current seed tied to the TF bundle only. Extra Wataru
    # proxy/VF coils shape the field through the loaded Biot-Savart object and
    # should not perturb G0 a second time here.
    G0 = compute_tf_G0(tf_coils)

    # ==============================================================================
    # OPTIMIZATION SETUP
    # ==============================================================================
    print(f"\n===== Starting single stage optimization for mpol = {mpol} =====")

    # Resolve basin-hopping RNG seed early so it's available for config_hash
    if args.basin_hops > 0:
        rng_seed = args.basin_seed if args.basin_seed >= 0 else int.from_bytes(os.urandom(4), 'big')
    else:
        rng_seed = None

    run_identity_config = make_run_identity_config(
        args,
        stage2_bs_path,
        stage,
        CONSTRAINT_WEIGHT,
        args.constraint_method,
        vol_target,
        iota_target,
        boozer_I,
        plasma_current_A,
        banana_surf_radius,
        nphi,
        ntheta,
        rng_seed,
        surface_mode_contract=surface_mode_contract,
        effective_num_surfaces=effective_num_surfaces,
        effective_inner_surface_ratio=effective_inner_surface_ratio,
        num_banana_current_controls=banana_current_state.num_control_currents(),
    )
    config_str = build_run_identity_config(run_identity_config)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    OUT_DIR_ITER = OUT_DIR + f"/mpol={mpol}-ntor={ntor}-{config_hash}"
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # Initialize Boozer surfaces with target parameters
    surface_data = []
    warm_start_surface_paths = []
    stage2_seed_surface = (
        None
        if stage2_seed_surf_path is None
        else load_warm_start_boozer_seed(stage2_seed_surf_path)
    )
    for config in surface_configs:
        (
            initial_surface,
            initial_surface_guess,
            initial_iota,
            initial_G,
            warm_start_surface_path,
        ) = resolve_initial_boozer_surface_seed(
            config_name=config["name"],
            default_surface=config["initial_surface"],
            default_iota=iota_target,
            default_G=G0,
            stage2_seed_surface=stage2_seed_surface,
            warm_start_surface_stem=warm_start_surface_stem,
        )
        if warm_start_surface_path is not None:
            warm_start_surface_paths.append(warm_start_surface_path)
        boozer_surface = initialize_boozer_surface(
            initial_surface,
            mpol,
            ntor,
            bs,
            config["target_volume"],
            CONSTRAINT_WEIGHT,
            initial_iota,
            initial_G,
            boozer_I,
            initial_surface_guess=initial_surface_guess,
            nfp=banana_surf_nfp,
        )
        surface_data.append({
            "name": config["name"],
            "seed_label": config["seed_label"],
            "target_volume": config["target_volume"],
            "boozer_surface": boozer_surface,
        })
    outer_surface_data = surface_data[-1]

    # ==============================================================================
    # SAVE INITIAL STATE
    # ==============================================================================
    # Save initial coil configurations
    curves_to_vtk(curves, OUT_DIR_ITER + "/curves_init", close=True)
    bs.save(OUT_DIR_ITER + "/biot_savart_init.json")

    save_surface_artifacts(surface_data, bs, OUT_DIR_ITER, "surf_init", also_write_outer_legacy=True)
    print(f"Volume: {outer_surface_data['boozer_surface'].surface.volume()}")

    # Generate initial diagnostic plots
    initial_field_error = normPlot(outer_surface_data['boozer_surface'].surface, bs, OUT_DIR_ITER + "/NormPlotInitial")
    try:
        cross_section_plot(
            surf_coils,
            outer_surface_data['boozer_surface'].surface,
            banana_curve,
            OUT_DIR_ITER + "/CrossSectionInitial",
            lcfs_clearance_reference,
            VV,
        )
    except Exception as e:
        print(f"WARNING: CrossSectionInitial plot failed (surface may fold at high mpol): {e}")
    initial_volume = outer_surface_data['boozer_surface'].surface.volume()
    initial_iota = Iotas(outer_surface_data['boozer_surface']).J()
    initial_max_curvature = np.max(banana_curve.kappa())
    initial_surface_volumes = [entry["boozer_surface"].surface.volume() for entry in surface_data]
    initial_surface_iotas = [Iotas(entry["boozer_surface"]).J() for entry in surface_data]
    initial_qs_objective, initial_boozer_objective = measure_frontier_reference_metrics(
        stage,
        surface_data,
        coils,
    )

    # ==============================================================================
    # DEFINE OBJECTIVE FUNCTION COMPONENTS
    # ==============================================================================
    # Objective function weights and parameters (all configurable via CLI)
    # Baseline default floors enforced via max() — weights are free, thresholds are clamped.
    LENGTH_WEIGHT = args.length_weight
    RES_WEIGHT = args.res_weight
    IOTAS_WEIGHT = args.iotas_weight
    CC_WEIGHT = args.cc_weight
    CC_DIST = max(args.cc_dist, COIL_COIL_MIN_DIST_M)
    if args.cc_dist < COIL_COIL_MIN_DIST_M:
        print(
            f"WARNING: --cc-dist {args.cc_dist} below hardware floor, "
            f"clamped to {COIL_COIL_MIN_DIST_M}"
        )
    CS_WEIGHT = args.cs_weight
    CS_DIST = max(args.cs_dist, COIL_PLASMA_MIN_DIST_M)
    if args.cs_dist < COIL_PLASMA_MIN_DIST_M:
        print(
            f"WARNING: --cs-dist {args.cs_dist} below hardware floor, "
            f"clamped to {COIL_PLASMA_MIN_DIST_M}"
        )
    SURF_DIST_WEIGHT = args.surf_dist_weight
    SS_DIST = max(args.ss_dist, PLASMA_VESSEL_MIN_DIST_M)
    if args.ss_dist < PLASMA_VESSEL_MIN_DIST_M:
        print(
            f"WARNING: --ss-dist {args.ss_dist} below hardware floor, "
            f"clamped to {PLASMA_VESSEL_MIN_DIST_M}"
        )
    allow_offspec_engineering_constraints = bool(
        args.allow_offspec_engineering_constraints
    )
    CURVATURE_WEIGHT = args.curvature_weight
    CURVATURE_THRESHOLD = float(args.curvature_threshold)
    if (
        not allow_offspec_engineering_constraints
        and args.curvature_threshold > MAX_CURVATURE_INV_M
    ):
        CURVATURE_THRESHOLD = MAX_CURVATURE_INV_M
        print(
            f"WARNING: --curvature-threshold {args.curvature_threshold} above hardware ceiling, "
            f"clamped to {MAX_CURVATURE_INV_M}"
        )
    SURFACE_GAP_THRESHOLD = max(args.surface_gap_threshold, 0.0)
    if len(surface_data) > 1 and SURF_DIST_WEIGHT != 0:
        print("WARNING: SURF_DIST_WEIGHT is diagnostic-only in multi-surface mode; outer-vessel spacing is enforced as a rejection gate.")

    requested_length_target = float(args.length_target)
    length_target = requested_length_target
    if (
        not allow_offspec_engineering_constraints
        and requested_length_target > COIL_LENGTH_HARD_LIMIT_M
    ):
        length_target = COIL_LENGTH_HARD_LIMIT_M
        print(
            f"WARNING: --length-target {requested_length_target} above hardware ceiling, "
            f"clamped to {COIL_LENGTH_HARD_LIMIT_M}"
        )
    frontier_goal_config = None
    if args.single_stage_goal_mode == "frontier":
        frontier_goal_config = build_frontier_goal_config(
            initial_iota=initial_iota,
            initial_volume=initial_volume,
            initial_qs_objective=initial_qs_objective,
            initial_boozer_objective=initial_boozer_objective,
            res_weight=RES_WEIGHT,
            iotas_weight=IOTAS_WEIGHT,
            volume_weight=args.frontier_volume_weight,
            iota_reference_override=args.frontier_reference_iota,
            iota_scale_override=args.frontier_reference_iota_scale,
            volume_reference_override=args.frontier_reference_volume,
            volume_scale_override=args.frontier_reference_volume_scale,
            qs_reference_override=args.frontier_reference_qa,
            boozer_reference_override=args.frontier_reference_boozer,
            boozer_trust_threshold_override=args.frontier_boozer_trust_threshold,
            boozer_trust_penalty_scale_override=args.frontier_boozer_trust_penalty_scale,
            scalarization_type=args.frontier_scalarization_type,
            chebyshev_rho_override=args.frontier_chebyshev_rho,
            chebyshev_sharpness_override=args.frontier_chebyshev_sharpness,
            chebyshev_weight_iota_override=args.frontier_chebyshev_weight_iota,
            chebyshev_weight_volume_override=args.frontier_chebyshev_weight_volume,
            chebyshev_weight_qa_override=args.frontier_chebyshev_weight_qa,
            chebyshev_weight_boozer_override=args.frontier_chebyshev_weight_boozer,
            epsilon_constraint_qa_max_override=args.epsilon_constraint_qa_max,
            epsilon_constraint_boozer_max_override=args.epsilon_constraint_boozer_max,
            epsilon_penalty_weight_override=args.frontier_epsilon_penalty_weight,
        )
        print(frontier_goal_mode_warning_message(frontier_goal_config))

    def rebuild_stage_objective_bundle(stage_name):
        objective_bundle = build_single_stage_objective_bundle(
            stage_name,
            surface_data,
            coils,
            objective_curves,
            banana_curves,
            iota_target,
            RES_WEIGHT,
            IOTAS_WEIGHT,
            LENGTH_WEIGHT,
            CC_WEIGHT,
            CC_DIST,
            CS_WEIGHT,
            CS_DIST,
            CURVATURE_WEIGHT,
            CURVATURE_THRESHOLD,
            length_target=length_target,
            SURF_DIST_WEIGHT=SURF_DIST_WEIGHT,
            vessel_surface=VV,
            vessel_gap_threshold=SS_DIST,
            goal_mode=args.single_stage_goal_mode,
            frontier_goal_config=frontier_goal_config,
        )
        apply_single_stage_objective_bundle(objective_bundle)
        return objective_bundle

    objective_bundle = rebuild_stage_objective_bundle(stage)
    length_target = objective_bundle["length_target"]
    REQUESTED_SEED_REGIME = str(
        getattr(args, "seed_regime", _DEFAULT_SINGLE_STAGE_SEED_REGIME)
    )
    EFFECTIVE_SEED_REGIME = REQUESTED_SEED_REGIME
    PRESERVED_TIMEOUT_REPLAY_CONFIG = current_preserved_timeout_replay_config()

    # Extract degrees of freedom
    dofs = JF.x
    if CONSTRAINT_METHOD == "alm":
        ALM_MULTIPLIERS = np.zeros(
            len(
                single_stage_alm_constraint_names(
                    alm_formulation=ALM_FORMULATION,
                    include_surface_surface=JSurfSurf is not None,
                )
            ),
            dtype=float,
        )
        ALM_PENALTY = args.alm_penalty_init

    # ==============================================================================
    # INITIALIZE OPTIMIZATION STATE
    # ==============================================================================
    # Initialize run_dict after JF and boozer_surface are ready
    initial_search_surface_weights = build_surface_search_weights(
        len(surface_data),
        0,
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
    )
    initial_search_eval = evaluate_search_objective(initial_search_surface_weights)
    initial_frontier_conditioning_report = (
        None
        if args.single_stage_goal_mode != "frontier"
        else build_frontier_conditioning_report(
            initial_search_eval,
            sample_label="seed",
        )
    )
    initial_search_gate = build_surface_search_gate(
        len(surface_data),
        0,
        MULTISURFACE_RAMP_ITERATIONS,
        INNER_SURFACE_INITIAL_WEIGHT,
        SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
        SS_DIST if len(surface_data) > 1 else 0.0,
    )
    initial_surface_status = evaluate_surface_stack(
        surface_data,
        vessel_surface=VV if len(surface_data) > 1 else None,
        surface_gap_threshold=SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
        vessel_gap_threshold=SS_DIST if len(surface_data) > 1 else 0.0,
        enforce_nesting=True,
    )
    initial_search_surface_status = evaluate_surface_stack(
        surface_data,
        vessel_surface=VV if len(surface_data) > 1 else None,
        surface_gap_threshold=initial_search_gate["surface_gap_threshold"],
        vessel_gap_threshold=initial_search_gate["vessel_gap_threshold"],
        enforce_nesting=initial_search_gate["enforce_nesting"],
    )
    initial_topology_status = final_topology_gate_for_results(
        args.init_only,
        len(surface_data),
        outer_surface_data["boozer_surface"].surface,
        bs,
        surface_mode_contract=surface_mode_contract,
    )
    initial_hardware_snapshot = evaluate_single_stage_hardware_snapshot(
        JCurveCurve,
        CC_DIST,
        JCurveSurface,
        CS_DIST,
        JSurfSurf,
        initial_surface_status,
        SS_DIST,
        banana_curve,
        CURVATURE_THRESHOLD,
        outer_surface_data["boozer_surface"].surface,
        VV,
        coil_length=float(curvelength.J()),
        length_target=length_target,
        tf_current_A=stage2_tf_current_A,
        tf_current_limit_A=TF_CURRENT_HARD_LIMIT_A,
        banana_current_A=banana_current_state.control_current_A(),
        banana_current_max_A=getattr(
            args,
            "banana_current_max_A",
            BANANA_CURRENT_HARD_LIMIT_A,
        ),
    )
    run_dict = {
        'surface_state': snapshot_surface_states(surface_data),
        'J': initial_search_eval['total'],
        'dJ': initial_search_eval['grad'].copy(),
        'search_eval': initial_search_eval,
        'it': 1,
        'accepted_iterations': 0,
        'lscount': 0,
        'x_prev': dofs.copy(),
        'accepted_x': dofs.copy(),
        'intersecting': any(entry["boozer_surface"].surface.is_self_intersecting() for entry in surface_data),
        'trial_hardware_status': None,
        'accepted_hardware_status': initial_hardware_snapshot["search_hardware_status"],
        'surface_status': initial_surface_status,
        'search_surface_status': initial_search_surface_status,
        'topology_gate_status': initial_topology_status,
        'frontier_trust_status': evaluate_frontier_trust_status(initial_search_eval),
        'accepted_boozer_stage': stage,
        'alm_feasibility_tolerance': args.alm_feas_tol if CONSTRAINT_METHOD == "alm" else None,
        'alm_stationarity_tolerance': args.alm_stationarity_tol if CONSTRAINT_METHOD == "alm" else None,
        'best_accepted_incumbent': None,
        'best_accepted_metric': None,
        'best_accepted_stage': None,
        'best_feasible_incumbent': None,
        'best_feasible_metric': None,
        'best_feasible_stage': None,
        'invalid_state_rejects_total': 0,
        'topology_gate_rejects': 0,
        'hardware_rejects': 0,
        'curvature_precheck_rejects': 0,
        'curvature_overcap_boozer_evals': 0,
        'curvature_overcap_boozer_evals_this_iteration': 0,
        'surface_solve_rejects': 0,
        'frontier_trust_rejects': 0,
        'active_optimizer_bounds': current_optimizer_bounds(),
        'banana_current_diagnostics': None,
        'banana_current_replay_context': None,
        'frontier_conditioning_seed_report': initial_frontier_conditioning_report,
        'frontier_conditioning_first_accepted_report': None,
    }
    restored_from_solver_checkpoint = False
    resume_alm_state = None
    if resume_solver_checkpoint_payload is not None:
        restored_from_solver_checkpoint = True
        restored_stage = str(resume_solver_checkpoint_payload["accepted_boozer_stage"])
        objective_bundle = rebuild_stage_objective_bundle(restored_stage)
        restore_single_stage_incumbent_state(
            run_dict,
            restore_incumbent_from_solver_checkpoint(
                resume_solver_checkpoint_payload
            ),
        )
        run_counters = dict(
            resume_solver_checkpoint_payload.get("run_counters", {})
        )
        run_dict["it"] = int(run_counters.get("it", run_dict["it"]))
        run_dict["accepted_iterations"] = int(
            resume_solver_checkpoint_payload["accepted_iterations"]
        )
        run_dict["x_prev"] = np.asarray(run_dict["accepted_x"], dtype=float).copy()
        run_dict["accepted_boozer_stage"] = restored_stage
        run_dict["invalid_state_rejects_total"] = int(
            run_counters.get("invalid_state_rejects_total", 0)
        )
        run_dict["topology_gate_rejects"] = int(
            run_counters.get("topology_gate_rejects", 0)
        )
        run_dict["hardware_rejects"] = int(
            run_counters.get("hardware_rejects", 0)
        )
        run_dict["curvature_precheck_rejects"] = int(
            run_counters.get("curvature_precheck_rejects", 0)
        )
        run_dict["curvature_overcap_boozer_evals"] = int(
            run_counters.get("curvature_overcap_boozer_evals", 0)
        )
        run_dict["curvature_overcap_boozer_evals_this_iteration"] = 0
        run_dict["surface_solve_rejects"] = int(
            run_counters.get("surface_solve_rejects", 0)
        )
        run_dict["frontier_trust_rejects"] = int(
            run_counters.get("frontier_trust_rejects", 0)
        )
        restored_best_accepted_incumbent = restore_optional_incumbent(
            resume_solver_checkpoint_payload,
            "best_accepted_incumbent",
        )
        run_dict["best_accepted_stage"] = resume_solver_checkpoint_payload.get(
            "best_accepted_stage"
        )
        run_dict["best_accepted_incumbent"] = normalize_diagnostic_incumbent_for_stage(
            run_dict,
            restored_best_accepted_incumbent,
            run_dict["best_accepted_stage"],
            restored_stage,
            rebuild_stage_objective_bundle,
        )
        run_dict["best_accepted_metric"] = resume_solver_checkpoint_payload.get(
            "best_accepted_metric"
        )
        restored_best_feasible_incumbent = restore_optional_incumbent(
            resume_solver_checkpoint_payload,
            "best_feasible_incumbent",
        )
        run_dict["best_feasible_stage"] = resume_solver_checkpoint_payload.get(
            "best_feasible_stage"
        )
        run_dict["best_feasible_incumbent"] = normalize_diagnostic_incumbent_for_stage(
            run_dict,
            restored_best_feasible_incumbent,
            run_dict["best_feasible_stage"],
            restored_stage,
            rebuild_stage_objective_bundle,
        )
        run_dict["best_feasible_metric"] = resume_solver_checkpoint_payload.get(
            "best_feasible_metric"
        )
        run_dict["frontier_conditioning_seed_report"] = (
            resume_solver_checkpoint_payload.get(
                "conditioning_seed_report",
                run_dict.get("frontier_conditioning_seed_report"),
            )
        )
        run_dict["frontier_conditioning_first_accepted_report"] = (
            resume_solver_checkpoint_payload.get(
                "conditioning_first_accepted_report",
                run_dict.get("frontier_conditioning_first_accepted_report"),
            )
        )
        refresh_accepted_search_state(run_dict, restored_stage)
        dofs = np.asarray(run_dict["accepted_x"], dtype=float).copy()
        resume_alm_state = resume_solver_checkpoint_payload.get("alm_state")
        if CONSTRAINT_METHOD == "alm" and isinstance(resume_alm_state, dict):
            ALM_MULTIPLIERS = np.asarray(
                resume_alm_state.get("multipliers", ALM_MULTIPLIERS),
                dtype=float,
            )
            ALM_PENALTY = float(
                resume_alm_state.get("penalty", ALM_PENALTY)
            )
        initial_best_accepted_updated = False
        initial_best_feasible_updated = False
    else:
        initial_best_accepted_updated = maybe_update_best_accepted_incumbent(run_dict, stage)
        initial_best_feasible_updated = maybe_update_best_feasible_incumbent(run_dict, stage)
    EFFECTIVE_SEED_REGIME = resolve_single_stage_seed_regime(
        REQUESTED_SEED_REGIME,
        run_dict,
        constraint_method=CONSTRAINT_METHOD,
        num_surfaces=effective_num_surfaces,
        basin_hops=args.basin_hops,
        init_only=args.init_only,
    )
    banana_current_replay_summary = None
    if args.banana_current_diagnostics:
        coordinate_spec = resolve_banana_current_coordinate_spec(
            JF,
            banana_current_state,
        )
        finite_difference_probe_fn = None
        if args.banana_current_fd_diagnostics:
            reference_x = np.asarray(run_dict["accepted_x"], dtype=float).copy()
            reference_surface_state = snapshot_surface_states(surface_data)

            def finite_difference_probe_fn(trial_x):
                return evaluate_banana_current_fd_probe(
                    trial_x,
                    reference_x=reference_x,
                    reference_surface_state=reference_surface_state,
                    accepted_iterations=int(
                        run_dict.get("accepted_iterations", 0)
                    ),
                )

        run_dict["banana_current_diagnostics"] = build_banana_current_diagnostics_state(
            JF,
            banana_current_state,
            run_dict["accepted_x"],
            run_dict["dJ"],
            active_optimizer_bounds=run_dict.get("active_optimizer_bounds"),
            accepted_iteration=int(run_dict.get("accepted_iterations", 0)),
            baseline_total=float(run_dict["J"]),
            finite_difference_probe_fn=finite_difference_probe_fn,
            finite_difference_relative_step_fraction=(
                args.banana_current_fd_relative_step_fraction
            ),
        )
        run_dict["banana_current_replay_context"] = (
            build_banana_current_replay_context_state()
        )
        set_banana_current_replay_context_contract(
            run_dict["banana_current_replay_context"],
            mode=banana_current_state.mode,
            num_control_currents=banana_current_state.num_control_currents(),
            coordinate_dof_names=coordinate_spec.dof_names,
            current_coordinate_scale_factors_A=coordinate_spec.scale_factors_A,
            seed_currents_A=run_dict["banana_current_diagnostics"]["seed_currents_A"],
            configured_seed_currents_A=banana_current_state.seed_currents_A,
        )
        record_banana_current_replay_context_snapshot(
            run_dict["banana_current_replay_context"],
            accepted_iteration=int(run_dict.get("accepted_iterations", 0)),
            accepted_boozer_stage=run_dict.get("accepted_boozer_stage", stage),
            incumbent=snapshot_single_stage_incumbent_state(run_dict),
        )
        emit_banana_current_diagnostics_report(
            None
            if run_dict["banana_current_diagnostics"] is None
            else run_dict["banana_current_diagnostics"]["seed_report"]
        )
        write_banana_current_diagnostics_artifact(
            OUT_DIR_ITER,
            run_dict["banana_current_diagnostics"],
        )
        write_banana_current_replay_context_artifact(
            OUT_DIR_ITER,
            run_dict["banana_current_replay_context"],
        )
    if args.banana_current_replay_diagnostics_path:
        banana_current_replay_summary = (
            run_banana_current_rejected_trial_replay_study(
                args.banana_current_replay_diagnostics_path,
                replay_context_path=args.banana_current_replay_context_path,
                replay_output_path=args.banana_current_replay_output_path,
            )
        )
        if not args.banana_current_diagnostics:
            args.init_only = True
            print(
                "Skipping single-stage optimizer because "
                "--banana-current-replay-diagnostics-path was provided without "
                "--banana-current-diagnostics."
            )
    if CHECKPOINT_EVERY > 0 and not args.init_only:
        write_single_stage_solver_checkpoint_state(
            OUT_DIR_ITER,
            run_dict,
            requested_maxiter=MAXITER,
            runtime_maxiter=RUNTIME_MAXITER,
            accepted_stage=run_dict.get("accepted_boozer_stage", stage),
            goal_mode=args.single_stage_goal_mode,
            constraint_method=CONSTRAINT_METHOD,
            stage2_bs_path=stage2_bs_path,
            out_dir_iter=OUT_DIR_ITER,
            alm_state=current_solver_checkpoint_alm_state(),
        )
    PRESERVED_TIMEOUT_REPLAY_CONFIG = current_preserved_timeout_replay_config()
    initial_coil_length = curvelength.J()
    if initial_best_accepted_updated:
        write_preserved_timeout_artifacts_for_current_state(
            OUT_DIR_ITER,
            preservation_kind="best_accepted",
            incumbent_stage=stage,
            run_dict=run_dict,
            bs=bs,
            surface_data=surface_data,
            hardware_snapshot=initial_hardware_snapshot,
            field_error=initial_field_error,
            coil_length=initial_coil_length,
        )
    if initial_best_feasible_updated:
        write_preserved_timeout_artifacts_for_current_state(
            OUT_DIR_ITER,
            preservation_kind="best_feasible",
            incumbent_stage=stage,
            run_dict=run_dict,
            bs=bs,
            surface_data=surface_data,
            hardware_snapshot=initial_hardware_snapshot,
            field_error=initial_field_error,
            coil_length=initial_coil_length,
        )

    def refresh_preserved_timeout_artifacts_from_best_states():
        current_state = snapshot_single_stage_incumbent_state(run_dict)
        current_stage = run_dict.get("accepted_boozer_stage", stage)
        current_x_prev = np.asarray(run_dict.get("x_prev", run_dict["accepted_x"]), dtype=float).copy()
        current_intersecting = bool(run_dict.get("intersecting", False))
        current_frontier_trust_status = copy.deepcopy(
            run_dict.get("frontier_trust_status")
        )
        current_trial_hardware_status = copy.deepcopy(
            run_dict.get("trial_hardware_status")
        )
        try:
            for preservation_kind, incumbent_key, stage_key in (
                ("best_accepted", "best_accepted_incumbent", "best_accepted_stage"),
                ("best_feasible", "best_feasible_incumbent", "best_feasible_stage"),
            ):
                incumbent = run_dict.get(incumbent_key)
                if incumbent is None:
                    remove_preserved_timeout_artifacts(
                        OUT_DIR_ITER,
                        preservation_kind=preservation_kind,
                        surface_data=surface_data,
                    )
                    continue
                incumbent_stage = run_dict.get(stage_key, stage)
                restore_single_stage_incumbent_state(run_dict, incumbent)
                run_dict["accepted_boozer_stage"] = incumbent_stage
                run_dict["frontier_trust_status"] = evaluate_frontier_trust_status(
                    run_dict["search_eval"]
                )
                run_dict["intersecting"] = any(
                    run_dict["surface_status"]["self_intersections"]
                )
                JF.x = run_dict["accepted_x"].copy()
                restore_surface_states(surface_data, run_dict["surface_state"])
                outer_surface = surface_data[-1]["boozer_surface"].surface
                hardware_snapshot = evaluate_single_stage_hardware_snapshot(
                    JCurveCurve,
                    CC_DIST,
                    JCurveSurface,
                    CS_DIST,
                    JSurfSurf,
                    run_dict["surface_status"],
                    SS_DIST,
                    banana_curve,
                    CURVATURE_THRESHOLD,
                    outer_surface,
                    VV,
                    **current_single_stage_hardware_snapshot_kwargs(),
                )
                run_dict["accepted_hardware_status"] = hardware_snapshot["search_hardware_status"]
                field_error, _ = compute_surface_field_metrics(outer_surface, bs)
                write_preserved_timeout_artifacts_for_current_state(
                    OUT_DIR_ITER,
                    preservation_kind=preservation_kind,
                    incumbent_stage=incumbent_stage,
                    run_dict=run_dict,
                    bs=bs,
                    surface_data=surface_data,
                    hardware_snapshot=hardware_snapshot,
                    field_error=field_error,
                    coil_length=curvelength.J(),
                )
        finally:
            restore_single_stage_incumbent_state(run_dict, current_state)
            run_dict["accepted_boozer_stage"] = current_stage
            run_dict["x_prev"] = current_x_prev
            run_dict["intersecting"] = current_intersecting
            run_dict["frontier_trust_status"] = current_frontier_trust_status
            run_dict["trial_hardware_status"] = current_trial_hardware_status
            JF.x = run_dict["accepted_x"].copy()
            restore_surface_states(surface_data, run_dict["surface_state"])

    if restored_from_solver_checkpoint:
        refresh_preserved_timeout_artifacts_from_best_states()

    # ==============================================================================
    # RUN OPTIMIZATION
    # ==============================================================================
    # Get convergence tolerances for current mpol
    ftol = args.ftol
    gtol = args.gtol

    basin_hop_count = None
    basin_minimization_failures = None
    basin_accepted_hops = None
    basin_rejected_hops = None
    basin_best_objective = None
    basin_accept_test_rejections = None
    basin_accept_test_triggered = None
    basin_nonfinite_rejections = None
    basin_normalized_step_rejections = None
    basin_completed_hops = None
    basin_initial_objective = None
    basin_best_hop_objective = None
    basin_best_hop_index = None
    basin_best_result_source = None
    basin_objective_improvement = None
    termination_message = None
    optimizer_success = None
    optimizer_status = None
    phase1_iterations = None
    phase1_termination_message = None
    phase1_success = None
    phase1_outcome = None
    phase1_first_accepted_step_rms = None
    phase1_max_accepted_step_rms = None
    phase1_anchor_restore_used = False
    phase1_unsafe_accept_rollbacks = 0
    phase1_invalid_reject_attempts = 0
    phase1_recovery_used = False
    refinement_attempted = False
    refinement_success = None
    refinement_iterations = None
    refinement_chunks = None
    refinement_abort_reason = None
    refinement_termination_message = None
    startup_local_preservation_used = False
    startup_local_preservation_preserved_start = False
    startup_local_preservation_attempts = 0
    startup_local_preservation_radius = None
    startup_local_phase_used = False
    startup_local_phase_regime = None
    startup_local_recovery_achieved = False
    bridge_local_donor_ready = False
    final_source_stage = stage
    alm_result = None
    validate_surface_mode_constraint_args(
        args,
        surface_mode_contract=surface_mode_contract,
    )
    if args.init_only:
        res_nit = 0
        final_volume = initial_volume
        final_iota = initial_iota
        final_max_curvature = initial_max_curvature
        fieldError = initial_field_error
        termination_message = "init_only"
        optimizer_success = True
        print("Skipping single-stage optimizer because --init-only was provided.")
    elif CONSTRAINT_METHOD == "alm":
        if args.basin_hops > 0:
            raise ValueError("--basin-hops is not supported with --constraint-method=alm")
        alm_settings = build_single_stage_alm_settings(args)
        alm_constraint_names = single_stage_alm_constraint_names(
            alm_formulation=ALM_FORMULATION,
            include_surface_surface=JSurfSurf is not None,
        )
        alm_partial_state = {"history": []}
        resume_alm_state = (
            None
            if resume_solver_checkpoint_payload is None
            else resume_solver_checkpoint_payload.get("alm_state")
        )
        if CONSTRAINT_METHOD == "alm" and isinstance(resume_alm_state, dict):
            initial_alm_multipliers = np.asarray(
                resume_alm_state.get("multipliers", np.zeros(len(alm_constraint_names))),
                dtype=float,
            )
            initial_alm_penalty = float(
                resume_alm_state.get("penalty", args.alm_penalty_init)
            )
        else:
            initial_alm_multipliers = np.zeros(len(alm_constraint_names), dtype=float)
            initial_alm_penalty = args.alm_penalty_init

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

        def evaluate_problem(inner_x, multipliers, penalty):
            set_alm_runtime_state(multipliers, penalty)
            return evaluate_search_step(inner_x)

        def outer_state_callback(outer_iteration, multipliers, penalty):
            set_alm_runtime_state(multipliers, penalty)
            run_dict["alm_outer_iteration"] = int(outer_iteration)
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
            return snapshot_single_stage_incumbent_state(run_dict)

        def restore_incumbent_state(incumbent_state):
            restore_single_stage_incumbent_state(run_dict, incumbent_state)
            JF.x = run_dict["accepted_x"].copy()
            restore_surface_states(surface_data, run_dict["surface_state"])

        set_alm_runtime_state(
            initial_alm_multipliers,
            initial_alm_penalty,
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
                "maxiter": RUNTIME_MAXITER,
                "maxcor": args.maxcor,
                "ftol": ftol,
                "gtol": gtol,
            },
            accepted_callback=callback,
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
        res_nit = res.nit
        optimizer_status = getattr(res, "status", None)
        termination_message = normalize_optimizer_termination_message(
            res.message,
            success=bool(res.success),
            status=optimizer_status,
            invalid_state_rejects_total=run_dict["invalid_state_rejects_total"],
            surface_solve_rejects=run_dict["surface_solve_rejects"],
            hardware_rejects=run_dict["hardware_rejects"],
            topology_gate_rejects=run_dict["topology_gate_rejects"],
        )
        optimizer_success = bool(res.success)
        emit_alm_partial_state(
            res.multipliers,
            res.penalty,
            outer_iteration=getattr(res, "outer_iterations", None),
            latest_history_entry=(
                None if not alm_partial_state["history"] else alm_partial_state["history"][-1]
            ),
            termination_message=termination_message,
            optimizer_success=optimizer_success,
            termination_reason=getattr(res, "termination_reason", None),
            inner_optimizer_success=getattr(res, "optimizer_success", None),
            inner_optimizer_message=getattr(res, "optimizer_message", None),
            converged_to_tolerances=getattr(res, "converged_to_tolerances", None),
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
            final_stationarity_norm=getattr(res, "final_stationarity_norm", None),
        )
        print(termination_message)
    elif args.basin_hops > 0:
        # Basin-hopping: perturb DOFs and re-run L-BFGS-B multiple times, keep best
        run_dict["active_optimizer_bounds"] = current_optimizer_bounds()
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'jac': True,
            'bounds': current_optimizer_bounds(),
            'callback': callback,
            'options': {'maxiter': RUNTIME_MAXITER, 'maxcor': args.maxcor, 'ftol': ftol, 'gtol': gtol},
        }
        basin_niter_success = args.basin_niter_success if args.basin_niter_success > 0 else None
        print(
            f"Basin-hopping with {args.basin_hops} hops, "
            f"stepsize={args.basin_stepsize}, "
            f"T={args.basin_temperature}, "
            f"niter_success={basin_niter_success}, "
            f"seed={rng_seed}"
        )
        res, basin_telemetry = run_basin_hopping(
            fun,
            dofs,
            basin_hops=args.basin_hops,
            basin_stepsize=args.basin_stepsize,
            basin_temperature=args.basin_temperature,
            basin_niter_success=basin_niter_success,
            rng_seed=rng_seed,
            minimizer_kwargs=minimizer_kwargs,
            disp=True,
        )
        (
            basin_accepted_hops,
            basin_rejected_hops,
            basin_best_objective,
            basin_accept_test_rejections,
            basin_accept_test_triggered,
        ) = basin_telemetry_values(basin_telemetry)
        basin_nonfinite_rejections = basin_telemetry.get("basin_nonfinite_rejections")
        basin_normalized_step_rejections = basin_telemetry.get(
            "basin_normalized_step_rejections"
        )
        basin_completed_hops = basin_telemetry.get("basin_completed_hops")
        basin_initial_objective = basin_telemetry.get("basin_initial_objective")
        basin_best_hop_objective = basin_telemetry.get("basin_best_hop_objective")
        basin_best_hop_index = basin_telemetry.get("basin_best_hop_index")
        basin_best_result_source = basin_telemetry.get("basin_best_result_source")
        basin_objective_improvement = basin_telemetry.get("basin_objective_improvement")
        basin_hop_count = res.nit if hasattr(res, 'nit') else None
        basin_minimization_failures = res.minimization_failures if hasattr(res, 'minimization_failures') else None
        if hasattr(res, 'lowest_optimization_result') and hasattr(res.lowest_optimization_result, 'nit'):
            res_nit = res.lowest_optimization_result.nit
        else:
            res_nit = basin_hop_count if basin_hop_count is not None else 0
        if hasattr(res, 'lowest_optimization_result'):
            lowest_result = res.lowest_optimization_result
            optimizer_status = getattr(lowest_result, "status", None)
            termination_message = normalize_optimizer_termination_message(
                getattr(lowest_result, "message", "basinhopping_complete"),
                success=bool(getattr(lowest_result, "success", True)),
                status=optimizer_status,
                invalid_state_rejects_total=run_dict["invalid_state_rejects_total"],
                surface_solve_rejects=run_dict["surface_solve_rejects"],
                hardware_rejects=run_dict["hardware_rejects"],
                topology_gate_rejects=run_dict["topology_gate_rejects"],
            )
            optimizer_success = bool(getattr(lowest_result, 'success', True))
        else:
            termination_message = str(getattr(res, 'message', 'basinhopping_complete'))
            optimizer_success = True
            optimizer_status = getattr(res, "status", None)
        print(f"Basin-hopping complete. Best fun={res.fun:.6e}, hops={args.basin_hops}, seed={rng_seed}")
    else:
        enable_local_preservation = penalty_feasible_start_local_preservation_enabled(
            run_dict,
            constraint_method=CONSTRAINT_METHOD,
            num_surfaces=effective_num_surfaces,
            basin_hops=args.basin_hops,
            init_only=args.init_only,
        )
        phase1_config = build_phase1_config(
            cc_weight=CC_WEIGHT,
            cs_weight=CS_WEIGHT,
            curvature_weight=CURVATURE_WEIGHT,
            surf_dist_weight=SURF_DIST_WEIGHT,
        )

        def restore_accepted_state():
            JF.x = run_dict["accepted_x"].copy()
            restore_surface_states(surface_data, run_dict["surface_state"])
            run_dict["x_prev"] = run_dict["accepted_x"].copy()

        phase1_result = run_penalty_phase1(
            dofs,
            total_maxiter=RUNTIME_MAXITER,
            maxcor=args.maxcor,
            ftol=ftol,
            gtol=gtol,
            initial_step_scale=args.multisurface_initial_step_scale,
            initial_step_maxiter=args.multisurface_initial_step_maxiter,
            enable_local_preservation=enable_local_preservation,
            seed_regime=EFFECTIVE_SEED_REGIME,
            is_frontier_mode=args.single_stage_goal_mode == "frontier",
            lower_bounds=JF.lower_bounds,
            upper_bounds=JF.upper_bounds,
            run_dict=run_dict,
            objective_fn=fun,
            callback_fn=callback,
            refinement_eligible_fn=refinement_eligible_incumbent,
            repair_progress_state_fn=repair_progress_state,
            phase1_config=phase1_config,
            objective_eval_fn=evaluate_search_step,
            normalize_message_fn=normalize_optimizer_termination_message,
            restore_accepted_state_fn=restore_accepted_state,
            refresh_preserved_timeout_artifacts_fn=(
                refresh_preserved_timeout_artifacts_from_best_states
            ),
        )
        if phase1_result["used_phase1"]:
            phase1_iterations = phase1_result["phase1_iterations"]
            phase1_termination_message = phase1_result["phase1_termination_message"]
            phase1_success = phase1_result["phase1_success"]
            phase1_outcome = phase1_result["phase1_outcome"]
            phase1_first_accepted_step_rms = phase1_result[
                "phase1_first_accepted_step_rms"
            ]
            phase1_max_accepted_step_rms = phase1_result[
                "phase1_max_accepted_step_rms"
            ]
            phase1_anchor_restore_used = phase1_result["phase1_anchor_restore_used"]
            phase1_unsafe_accept_rollbacks = phase1_result[
                "phase1_unsafe_accept_rollbacks"
            ]
            phase1_invalid_reject_attempts = phase1_result[
                "phase1_invalid_reject_attempts"
            ]
            phase1_recovery_used = phase1_result["phase1_recovery_used"]
            startup_local_preservation_used = phase1_result["local_preservation_used"]
            startup_local_preservation_preserved_start = phase1_result[
                "local_preservation_preserved_start"
            ]
            startup_local_preservation_attempts = phase1_result[
                "local_preservation_attempts"
            ]
            startup_local_preservation_radius = phase1_result[
                "local_preservation_radius"
            ]
            startup_local_phase_used = phase1_result["local_preservation_used"]
            startup_local_phase_regime = phase1_result["startup_local_phase_regime"]
            startup_local_recovery_achieved = phase1_result[
                "startup_local_recovery_achieved"
            ]
            bridge_local_donor_ready = phase1_result["bridge_local_donor_ready"]
            if startup_local_preservation_used:
                print(
                    "Running startup local phase with "
                    f"regime={startup_local_phase_regime} "
                    f"radius={startup_local_preservation_radius:.3e} and "
                    f"attempts={startup_local_preservation_attempts}"
                )
            elif args.multisurface_initial_step_scale < 1.0:
                print(
                    "Running scaled initial continuation phase with "
                    f"step_scale={args.multisurface_initial_step_scale} and "
                    f"maxiter={phase1_iterations}"
                )
            print(phase1_termination_message)
            dofs = phase1_result["next_dofs"]
            run_dict["x_prev"] = dofs.copy()

        remaining_maxiter = max(RUNTIME_MAXITER - (phase1_iterations or 0), 0)
        if remaining_maxiter > 0 and phase1_result["continue_search"]:
            phase2_bounds = build_penalty_phase2_bounds(
                dofs,
                lower_bounds=JF.lower_bounds,
                upper_bounds=JF.upper_bounds,
                phase1_result=phase1_result,
            )
            if startup_local_preservation_used:
                print(
                    "Continuing penalty search with donor-local bounds at "
                    f"radius={startup_local_preservation_radius:.3e}"
                )
            with temporary_run_dict_value(
                run_dict,
                "active_optimizer_bounds",
                phase2_bounds,
            ):
                res = minimize(
                    fun,
                    dofs,
                    jac=True,
                    method='L-BFGS-B',
                    bounds=phase2_bounds,
                    callback=callback,
                    options={'maxiter': remaining_maxiter, 'maxcor': args.maxcor, 'ftol': ftol, 'gtol': gtol},
                )
            res_nit = (phase1_iterations or 0) + res.nit
            optimizer_status = getattr(res, "status", None)
            termination_message = normalize_optimizer_termination_message(
                res.message,
                success=bool(res.success),
                status=optimizer_status,
                invalid_state_rejects_total=run_dict["invalid_state_rejects_total"],
                surface_solve_rejects=run_dict["surface_solve_rejects"],
                hardware_rejects=run_dict["hardware_rejects"],
                topology_gate_rejects=run_dict["topology_gate_rejects"],
            )
            optimizer_success = bool(res.success)
            if phase1_termination_message is not None:
                termination_message = f"phase1={phase1_termination_message}; phase2={termination_message}"
            print(termination_message)
        else:
            res = SimpleNamespace(
                x=dofs.copy(),
                nit=phase1_iterations or 0,
                message=phase1_termination_message
                or (
                    "feasible_start_preserved"
                    if startup_local_preservation_preserved_start
                    else "phase1_only"
                ),
                success=bool(phase1_success) and not startup_local_preservation_preserved_start,
            )
            res_nit = res.nit
            termination_message = str(res.message)
            optimizer_success = bool(res.success)
            optimizer_status = getattr(res, "status", None)
            print(termination_message)

    if (
        not args.init_only
        and args.boozer_stage_refinement
        and CONSTRAINT_METHOD == "penalty"
        and args.basin_hops == 0
    ):
        phase1_incumbent = run_dict.get("best_feasible_incumbent")
        phase1_metric = run_dict.get("best_feasible_metric")
        phase1_stage = run_dict.get("best_feasible_stage", stage)
        if phase1_incumbent is not None and phase1_metric is not None:
            refinement_attempted = True
            stage = args.refinement_boozer_stage
            refinement_result = run_chunked_refinement(
                run_dict,
                phase1_incumbent,
                phase1_metric,
                phase1_stage,
                stage,
                rebuild_stage_objective_bundle,
                args.refinement_maxiter,
                args.refinement_chunk_maxiter,
                args.refinement_max_stalled_chunks,
                args.maxcor,
                ftol,
                gtol,
            )
            refinement_iterations = refinement_result["iterations"]
            refinement_chunks = refinement_result["chunks"]
            refinement_abort_reason = refinement_result["abort_reason"]
            refinement_termination_message = refinement_result["termination_message"]
            if refinement_termination_message is not None:
                print(refinement_termination_message)

            refinement_incumbent = refinement_result["best_incumbent"]
            refinement_metric = refinement_result["best_metric"]
            refinement_total_iterations = (res_nit or 0) + refinement_iterations
            (
                refinement_status_message,
                refinement_status_success,
                refinement_status_result,
            ) = summarize_refinement_result(
                refinement_result,
                refinement_total_iterations,
                run_dict["accepted_x"],
            )
            if refinement_incumbent is not None:
                refinement_success = True
                stage = args.refinement_boozer_stage
                objective_bundle = restore_incumbent_for_stage(
                    run_dict,
                    refinement_incumbent,
                    stage,
                    rebuild_stage_objective_bundle,
                )
                length_target = objective_bundle["length_target"]
                run_dict["best_feasible_incumbent"] = refinement_incumbent
                run_dict["best_feasible_metric"] = refinement_metric
                run_dict["best_feasible_stage"] = stage
                final_source_stage = stage
                res_nit = refinement_total_iterations
                termination_message = refinement_status_message
                optimizer_success = refinement_status_success
                res = refinement_status_result
            else:
                refinement_success = False
                stage = phase1_stage
                objective_bundle = restore_incumbent_for_stage(
                    run_dict,
                    phase1_incumbent,
                    stage,
                    rebuild_stage_objective_bundle,
                )
                length_target = objective_bundle["length_target"]
                run_dict["best_feasible_incumbent"] = phase1_incumbent
                run_dict["best_feasible_metric"] = phase1_metric
                run_dict["best_feasible_stage"] = phase1_stage
                final_source_stage = phase1_stage
                res_nit = refinement_total_iterations
                termination_message = refinement_status_message
                optimizer_success = refinement_status_success
                res = refinement_status_result

    if alm_result is not None:
        set_alm_runtime_state(alm_result.multipliers, alm_result.penalty)

    # ==============================================================================
    # SAVE OPTIMIZED STATE
    # ==============================================================================
    final_hardware_snapshot = None
    if not args.init_only:
        # Re-solve the surface stack at the reported optimizer endpoint so saved
        # artifacts reflect the actual final coil DOFs rather than the last
        # callback-accepted surface state.
        finalize_surface_stack(
            res.x,
            JF,
            surface_data,
            run_dict,
            vessel_surface=VV if len(surface_data) > 1 else None,
            surface_gap_threshold=SURFACE_GAP_THRESHOLD if len(surface_data) > 1 else 0.0,
            vessel_gap_threshold=SS_DIST if len(surface_data) > 1 else 0.0,
        )
        final_source_stage = run_dict.get("accepted_boozer_stage", final_source_stage)

        full_objective_eval = evaluate_search_objective(np.ones(len(surface_data)))
        run_dict['J'] = full_objective_eval['total']
        run_dict['dJ'] = full_objective_eval['grad'].copy()
        run_dict['full_eval'] = full_objective_eval
        run_dict['base_eval'] = evaluate_base_objective(
            np.ones(len(surface_data)),
            nonQSs,
            brs,
            RES_WEIGHT,
            Jiota,
            IOTAS_WEIGHT,
            JCurveLength,
            LENGTH_WEIGHT,
        )

        # Save optimized coil configurations
        curves_to_vtk(curves, OUT_DIR_ITER + "/curves_opt", close=True)
        bs.save(OUT_DIR_ITER + "/biot_savart_opt.json")

        save_surface_artifacts(surface_data, bs, OUT_DIR_ITER, "surf_opt", also_write_outer_legacy=True)

        final_volume = outer_surface_data['boozer_surface'].surface.volume()
        final_iota = Iotas(outer_surface_data['boozer_surface']).J()
        final_hardware_snapshot = evaluate_single_stage_hardware_snapshot(
            JCurveCurve,
            CC_DIST,
            JCurveSurface,
            CS_DIST,
            JSurfSurf,
            run_dict["surface_status"],
            SS_DIST,
            banana_curve,
            CURVATURE_THRESHOLD,
            outer_surface_data["boozer_surface"].surface,
            VV,
            coil_length=float(curvelength.J()),
            length_target=length_target,
            tf_current_A=stage2_tf_current_A,
            tf_current_limit_A=TF_CURRENT_HARD_LIMIT_A,
            banana_current_A=banana_current_state.control_current_A(),
            banana_current_max_A=args.banana_current_max_A,
        )
        final_max_curvature = final_hardware_snapshot["max_curvature"]
        final_surface_volumes = [entry["boozer_surface"].surface.volume() for entry in surface_data]
        final_surface_iotas = [Iotas(entry["boozer_surface"]).J() for entry in surface_data]
        print(f"Volume: {final_volume}")
        print(f"Iota: {final_iota}")
        print(f"Max Curvature: {final_max_curvature}")

        # Generate final diagnostic plots
        fieldError = normPlot(outer_surface_data['boozer_surface'].surface, bs, OUT_DIR_ITER + "/NormPlotOptimized")
        try:
            cross_section_plot(
                surf_coils,
                outer_surface_data['boozer_surface'].surface,
                banana_curve,
                OUT_DIR_ITER + "/CrossSectionOptimized",
                lcfs_clearance_reference,
                VV,
            )
        except Exception as e:
            print(f"WARNING: CrossSectionOptimized plot failed (surface may fold at high mpol): {e}")
    else:
        final_surface_volumes = initial_surface_volumes
        final_surface_iotas = initial_surface_iotas
        run_dict['full_eval'] = None
        run_dict['base_eval'] = None
        run_dict['J'] = None
        run_dict['dJ'] = None

    final_topology_status = final_topology_gate_for_results(
        args.init_only,
        len(surface_data),
        outer_surface_data['boozer_surface'].surface,
        bs,
        surface_mode_contract=surface_mode_contract,
    )
    surface_mode_metadata = _build_surface_mode_metadata_impl(surface_mode_contract)
    objective_j = float(run_dict['J']) if run_dict['J'] is not None else None
    base_objective_j = None if run_dict['base_eval'] is None else float(run_dict['base_eval']['total'])
    search_objective_j = float(run_dict['search_eval']['total'])
    final_frontier_trust_status = evaluate_frontier_trust_status(run_dict["search_eval"])
    frontier_rank_objective_j = run_dict["search_eval"].get("frontier_rank_total")
    final_search_surface_weights = run_dict['search_eval']['surface_weights'].tolist()
    if final_hardware_snapshot is None:
        final_hardware_snapshot = evaluate_single_stage_hardware_snapshot(
            JCurveCurve,
            CC_DIST,
            JCurveSurface,
            CS_DIST,
            JSurfSurf,
            run_dict["surface_status"],
            SS_DIST,
            banana_curve,
            CURVATURE_THRESHOLD,
            outer_surface_data["boozer_surface"].surface,
            VV,
            coil_length=float(curvelength.J()),
            length_target=length_target,
            tf_current_A=stage2_tf_current_A,
            tf_current_limit_A=TF_CURRENT_HARD_LIMIT_A,
            banana_current_A=banana_current_state.control_current_A(),
            banana_current_max_A=args.banana_current_max_A,
        )
    nonqs_ratio = None if args.init_only else float(JnonQSRatio.J())
    boozer_residual = None if args.init_only else float(JBoozerResidual.J())
    final_hardware_status = final_hardware_snapshot["artifact_hardware_status"]
    final_feasibility_ok = refinement_eligible_for_hardware_status(
        run_dict,
        final_hardware_status,
    )
    best_feasible_results = build_best_feasible_results_summary(
        run_dict,
        JCurveCurve,
        JCurveSurface,
        JSurfSurf,
        banana_curve,
        curvelength,
        CC_DIST,
        CS_DIST,
        SS_DIST,
        CURVATURE_THRESHOLD,
        length_target,
        stage2_tf_current_A,
        banana_current_state,
        args.banana_current_max_A,
        outer_surface_data["boozer_surface"].surface,
        VV,
    )
    if not final_hardware_status["success"]:
        optimizer_success = False
        if termination_message:
            termination_message = f"{termination_message}; hardware_constraints_failed"
        else:
            termination_message = "hardware_constraints_failed"

    # Save the results of optimization to a separate file
    frontier_conditioning_gate = build_frontier_conditioning_gate(
        seed_report=run_dict.get("frontier_conditioning_seed_report"),
        first_accepted_report=run_dict.get(
            "frontier_conditioning_first_accepted_report"
        ),
    )
    constraint_cli_layer = {
        "tf_current_A": float(stage2_tf_current_A),
        "banana_current_max_A": float(args.banana_current_max_A),
        "length_target": float(length_target),
        "cc_threshold": float(CC_DIST),
        "coil_plasma_min_dist_m": float(CS_DIST),
        "plasma_vessel_min_dist_m": float(SS_DIST),
        "curvature_threshold": float(CURVATURE_THRESHOLD),
        "banana_surf_radius": float(banana_surf_radius),
    }
    single_stage_constraint_contract, single_stage_constraint_trace = (
        _resolve_constraint_contract_from_wire_names_impl(
            cli_overrides=constraint_cli_layer,
            allow_offspec_engineering=allow_offspec_engineering_constraints,
        )
    )
    constraint_override_reason = apply_offspec_engineering_override_reason(
        args.constraint_override_reason,
        layer=constraint_cli_layer,
        allow_offspec_engineering=allow_offspec_engineering_constraints,
    )
    constraint_profile_label = (
        "single_stage_solver"
        if args.constraint_profile_label in {None, ""}
        else str(args.constraint_profile_label)
    )
    constraint_metadata = build_constraint_metadata(
        single_stage_constraint_contract,
        profile_name=constraint_profile_label,
        override_reason=constraint_override_reason,
        trace=single_stage_constraint_trace,
    )
    results = {
        "PLASMA_SURF_FILENAME": plasma_surf_filename,
        "PLASMA_SURF_PATH": file_loc,
        "STAGE2_SOURCE": args.stage2_source,
        "STAGE2_BS_PATH": stage2_bs_path,
        "STAGE2_SEED_SURF_PATH": stage2_seed_surf_path,
        "STAGE2_RESULTS_PATH": str(stage2_results_path),
        "WARM_START_SURFACE_STEM": warm_start_surface_stem,
        "WARM_START_SURFACE_PATHS": (
            None if not warm_start_surface_paths else warm_start_surface_paths
        ),
        STAGE2_SEED_CONTRACT_HASH_KEY: stage2_results.get("CONTRACT_HASH"),
        "STAGE2_SEED_MAJOR_RADIUS": R0,
        "STAGE2_SEED_TOROIDAL_FLUX": s,
        "STAGE2_SEED_BANANA_SURF_RADIUS": float(stage2_results["banana_surf_radius"]),
        "STAGE2_SEED_TF_CURRENT_A": stage2_tf_current_A,
        "STAGE2_SEED_ORDER": order,
        "STAGE2_FINITE_CURRENT_MODE": stage2_results["FINITE_CURRENT_MODE"],
        "STAGE2_BOOZER_CURRENT_CONVENTION": stage2_results["BOOZER_CURRENT_CONVENTION"],
        "STAGE2_NUM_BANANA_COILS": coil_partitions.num_banana_coils,
        "STAGE2_NUM_PROXY_COILS": coil_partitions.num_proxy_coils,
        "STAGE2_NUM_VF_COILS": coil_partitions.num_vf_coils,
        "STAGE2_PROXY_PLASMA_CURRENT_A": float(
            stage2_results.get("PROXY_PLASMA_CURRENT_A", 0.0)
        ),
        "STAGE2_VF_CURRENT_A": float(stage2_results.get("VF_CURRENT_A", 0.0)),
        "STAGE2_VF_TEMPLATE_PATH": stage2_results.get("VF_TEMPLATE_PATH"),
        "STAGE2_TF_CURRENT_A": stage2_tf_current_A,
        "STAGE2_TF_CURRENT_SUM_ABS_A": tf_current_sum_abs_A,
        "BANANA_INIT_CURRENT_A": initial_banana_current_A,
        "mpol": mpol,
        "ntor": ntor,
        "nphi": nphi,
        "ntheta": ntheta,
        "NUM_SURFACES": effective_num_surfaces,
        "INNER_SURFACE_RATIO": effective_inner_surface_ratio,
        **surface_mode_metadata,
        "SURFACE_GAP_THRESHOLD": SURFACE_GAP_THRESHOLD,
        "MULTISURFACE_RAMP_ITERATIONS": MULTISURFACE_RAMP_ITERATIONS,
        "INNER_SURFACE_INITIAL_WEIGHT": INNER_SURFACE_INITIAL_WEIGHT,
        "MULTISURFACE_INITIAL_STEP_SCALE": args.multisurface_initial_step_scale,
        "MULTISURFACE_INITIAL_STEP_MAXITER": args.multisurface_initial_step_maxiter,
        "TOPOLOGY_GATE_FIELDLINES": TOPOLOGY_GATE_FIELDLINES,
        "TOPOLOGY_GATE_TMAX": TOPOLOGY_GATE_TMAX,
        "TOPOLOGY_GATE_TOL": TOPOLOGY_GATE_TOL,
        "TOPOLOGY_GATE_SURVIVAL_THRESHOLD": TOPOLOGY_GATE_SURVIVAL_THRESHOLD,
        "TOPOLOGY_GATE_PENALTY_SCALE": TOPOLOGY_GATE_PENALTY_SCALE,
        "HARDWARE_SEARCH_MODE": HARDWARE_SEARCH_MODE,
        "HARDWARE_SEARCH_SOFT_ITERATIONS": HARDWARE_SEARCH_SOFT_ITERATIONS,
        "CURVATURE_TRAVERSAL_BAND": CURVATURE_TRAVERSAL_BAND,
        "CURVATURE_TRAVERSAL_EVAL_BUDGET": CURVATURE_TRAVERSAL_EVAL_BUDGET,
        "boozer_stage": reported_boozer_stage(args.boozer_stage, final_source_stage),
        "REQUESTED_BOOZER_STAGE": args.boozer_stage,
        "BOOZER_STAGE_REFINEMENT": bool(args.boozer_stage_refinement),
        "REFINEMENT_BOOZER_STAGE": args.refinement_boozer_stage if args.boozer_stage_refinement else None,
        "REFINEMENT_MAXITER": args.refinement_maxiter if args.boozer_stage_refinement else None,
        "REFINEMENT_CHUNK_MAXITER": args.refinement_chunk_maxiter if args.boozer_stage_refinement else None,
        "REFINEMENT_MAX_STALLED_CHUNKS": args.refinement_max_stalled_chunks if args.boozer_stage_refinement else None,
        "REFINEMENT_ATTEMPTED": refinement_attempted,
        "REFINEMENT_SUCCESS": refinement_success,
        "REFINEMENT_ITERATIONS": refinement_iterations,
        "REFINEMENT_CHUNKS": refinement_chunks,
        "REFINEMENT_TERMINATION_MESSAGE": refinement_termination_message,
        "REFINEMENT_ABORT_REASON": refinement_abort_reason,
        "FINAL_SOURCE_STAGE": final_source_stage,
        "CONSTRAINT_WEIGHT": CONSTRAINT_WEIGHT,
        "CONSTRAINT_METHOD": CONSTRAINT_METHOD,
        "ALM_FORMULATION": ALM_FORMULATION if CONSTRAINT_METHOD == "alm" else None,
        "REQUESTED_SEED_REGIME": REQUESTED_SEED_REGIME,
        "EFFECTIVE_SEED_REGIME": EFFECTIVE_SEED_REGIME,
        "CC_DIST": CC_DIST,
        "CC_WEIGHT": CC_WEIGHT,
        "CS_DIST": CS_DIST,
        "CS_WEIGHT": CS_WEIGHT,
        "SS_DIST": SS_DIST,
        "SURF_DIST_WEIGHT": SURF_DIST_WEIGHT,
        "CURVATURE_WEIGHT": CURVATURE_WEIGHT,
        "CURVATURE_THRESHOLD": CURVATURE_THRESHOLD,
        "LENGTH_WEIGHT": LENGTH_WEIGHT,
        "RES_WEIGHT": RES_WEIGHT,
        "IOTAS_WEIGHT": IOTAS_WEIGHT,
        "MAJOR_RADIUS": R0,
        "R0_OFF_SPEC": is_major_radius_offspec(R0),
        "TOROIDAL_FLUX": s,
        "banana_surf_radius": banana_surf_radius,
        "order": order,
        "init_only": args.init_only,
        "max_iterations": MAXITER,
        "iterations": res_nit,
        "FTOL": ftol,
        "GTOL": gtol,
        "ALM_MAX_OUTER_ITERS": args.alm_max_outer_iters if CONSTRAINT_METHOD == "alm" else None,
        "ALM_OUTER_ITERATIONS": getattr(alm_result, "outer_iterations", None),
        "ALM_PENALTY_INIT": args.alm_penalty_init if CONSTRAINT_METHOD == "alm" else None,
        "ALM_PENALTY_SCALE": args.alm_penalty_scale if CONSTRAINT_METHOD == "alm" else None,
        "ALM_PENALTY_MAX": args.alm_penalty_max if CONSTRAINT_METHOD == "alm" else None,
        "ALM_FEAS_TOL": args.alm_feas_tol if CONSTRAINT_METHOD == "alm" else None,
        "ALM_STATIONARITY_TOL": args.alm_stationarity_tol if CONSTRAINT_METHOD == "alm" else None,
        "ALM_TRUST_RADIUS_INIT": args.alm_trust_radius_init if CONSTRAINT_METHOD == "alm" else None,
        "ALM_TRUST_RADIUS_MIN": args.alm_trust_radius_min if CONSTRAINT_METHOD == "alm" else None,
        "ALM_TRUST_RADIUS_SHRINK": args.alm_trust_radius_shrink if CONSTRAINT_METHOD == "alm" else None,
        "ALM_TRUST_RADIUS_GROW": args.alm_trust_radius_grow if CONSTRAINT_METHOD == "alm" else None,
        "ALM_MAX_INNER_ATTEMPTS": args.alm_max_inner_attempts if CONSTRAINT_METHOD == "alm" else None,
        "ALM_MAX_SUBPROBLEM_CONTINUATIONS": args.alm_max_subproblem_continuations if CONSTRAINT_METHOD == "alm" else None,
        "ALM_DISTANCE_SMOOTHING": args.alm_distance_smoothing if CONSTRAINT_METHOD == "alm" else None,
        "ALM_CURVATURE_SMOOTHING": args.alm_curvature_smoothing if CONSTRAINT_METHOD == "alm" else None,
        "ALM_QS_THRESHOLD": args.alm_qs_threshold if CONSTRAINT_METHOD == "alm" else None,
        "ALM_BOOZER_THRESHOLD": args.alm_boozer_threshold if CONSTRAINT_METHOD == "alm" else None,
        "ALM_IOTA_PENALTY_THRESHOLD": (
            args.alm_iota_penalty_threshold if CONSTRAINT_METHOD == "alm" else None
        ),
        "ALM_LENGTH_PENALTY_THRESHOLD": (
            args.alm_length_penalty_threshold if CONSTRAINT_METHOD == "alm" else None
        ),
        "ALM_PARTIAL_STATE_FILENAME": "alm_state.partial.json" if CONSTRAINT_METHOD == "alm" else None,
        "ALM_TERMINATION_REASON": getattr(alm_result, "termination_reason", None),
        "ALM_CONVERGED": getattr(alm_result, "converged_to_tolerances", None),
        "ALM_RESTORED_BEST_FEASIBLE": getattr(alm_result, "restored_best_feasible", None),
        "ALM_RESTORED_BEST_FEASIBLE_REASON": getattr(
            alm_result,
            "restored_best_feasible_reason",
            None,
        ),
        "ALM_INNER_OPTIMIZER_SUCCESS": getattr(alm_result, "optimizer_success", None),
        "ALM_INNER_OPTIMIZER_MESSAGE": getattr(alm_result, "optimizer_message", None),
        "ALM_FINAL_MAX_FEASIBILITY_VIOLATION": getattr(
            alm_result,
            "final_max_feasibility_violation",
            None,
        ),
        "ALM_FINAL_STATIONARITY_NORM": getattr(alm_result, "final_stationarity_norm", None),
        "ALM_FINAL_RAW_STATIONARITY_NORM": getattr(
            alm_result,
            "final_raw_stationarity_norm",
            None,
        ),
        "ALM_FINAL_KKT_STATIONARITY_NORM": getattr(
            alm_result,
            "final_kkt_stationarity_norm",
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
        "ALM_FINAL_PENALTY": getattr(alm_result, "penalty", None),
        "ALM_FINAL_MULTIPLIERS": getattr(alm_result, "multipliers", None),
        "ALM_CONSTRAINT_NAMES": getattr(alm_result, "constraint_names", None),
        "ALM_FINAL_CONSTRAINT_VALUES": getattr(alm_result, "constraint_values", None),
        "ALM_FINAL_SOLVER_CONSTRAINT_VALUES": getattr(
            alm_result,
            "solver_constraint_values",
            None,
        ),
        "ALM_FINAL_TRUST_RADIUS": getattr(alm_result, "trust_radius", None),
        **alm_result_diagnostics_fields(alm_result),
        "ALM_HISTORY": getattr(alm_result, "history", None),
        "TERMINATION_MESSAGE": termination_message,
        "OPTIMIZER_SUCCESS": optimizer_success,
        "OPTIMIZER_STATUS": optimizer_status,
        "CHECKPOINT_EVERY": CHECKPOINT_EVERY,
        "TOPOLOGY_SCORER_EVERY": TOPOLOGY_SCORER_EVERY,
        "TOPOLOGY_SCORER_NFIELDLINES": TOPOLOGY_SCORER_NFIELDLINES,
        "TOPOLOGY_SCORER_TMAX": TOPOLOGY_SCORER_TMAX,
        "CONFINEMENT_OBJECTIVE_WEIGHT": CONFINEMENT_OBJECTIVE_WEIGHT,
        "CONFINEMENT_SURROGATE_WORST_K": CONFINEMENT_SURROGATE_WORST_K,
        "CONFINEMENT_SURROGATE_EARLY_THRESHOLD": CONFINEMENT_SURROGATE_EARLY_THRESHOLD,
        "CONFINEMENT_SURROGATE_MEAN_WEIGHT": CONFINEMENT_SURROGATE_MEAN_WEIGHT,
        "CONFINEMENT_SURROGATE_WORST_WEIGHT": CONFINEMENT_SURROGATE_WORST_WEIGHT,
        "CONFINEMENT_SURROGATE_EARLY_WEIGHT": CONFINEMENT_SURROGATE_EARLY_WEIGHT,
        "basin_hops": args.basin_hops,
        "basin_stepsize": args.basin_stepsize if args.basin_hops > 0 else None,
        "basin_temperature": args.basin_temperature if args.basin_hops > 0 else None,
        "basin_niter_success": (
            args.basin_niter_success
            if args.basin_hops > 0 and args.basin_niter_success > 0
            else None
        ),
        "basin_seed": rng_seed if args.basin_hops > 0 else None,
        "basin_iterations": basin_hop_count,
        "basin_minimization_failures": basin_minimization_failures,
        "basin_accepted_hops": basin_accepted_hops,
        "basin_rejected_hops": basin_rejected_hops,
        "basin_best_objective": basin_best_objective,
        "basin_accept_test_rejections": basin_accept_test_rejections,
        "basin_accept_test_triggered": basin_accept_test_triggered,
        "basin_nonfinite_rejections": basin_nonfinite_rejections,
        "basin_normalized_step_rejections": basin_normalized_step_rejections,
        "basin_completed_hops": basin_completed_hops,
        "basin_initial_objective": basin_initial_objective,
        "basin_best_hop_objective": basin_best_hop_objective,
        "basin_best_hop_index": basin_best_hop_index,
        "basin_best_result_source": basin_best_result_source,
        "basin_objective_improvement": basin_objective_improvement,
        "PHASE1_ITERATIONS": phase1_iterations,
        "PHASE1_TERMINATION_MESSAGE": phase1_termination_message,
        "PHASE1_SUCCESS": phase1_success,
        "PHASE1_OUTCOME": phase1_outcome,
        "PHASE1_FIRST_ACCEPTED_STEP_RMS": phase1_first_accepted_step_rms,
        "PHASE1_MAX_ACCEPTED_STEP_RMS": phase1_max_accepted_step_rms,
        "PHASE1_ANCHOR_RESTORE_USED": phase1_anchor_restore_used,
        "PHASE1_UNSAFE_ACCEPT_ROLLBACKS": phase1_unsafe_accept_rollbacks,
        "PHASE1_INVALID_REJECT_ATTEMPTS": phase1_invalid_reject_attempts,
        "PHASE1_RECOVERY_USED": phase1_recovery_used,
        "SEED_REGIME_OUTCOME": phase1_outcome,
        "STARTUP_LOCAL_PHASE_USED": startup_local_phase_used,
        "STARTUP_LOCAL_PHASE_REGIME": startup_local_phase_regime,
        "STARTUP_LOCAL_RECOVERY_ACHIEVED": startup_local_recovery_achieved,
        "BRIDGE_LOCAL_DONOR_READY": bridge_local_donor_ready,
        "STARTUP_LOCAL_PRESERVATION_USED": startup_local_preservation_used,
        "STARTUP_LOCAL_PRESERVED_START": startup_local_preservation_preserved_start,
        "STARTUP_LOCAL_PRESERVATION_ATTEMPTS": startup_local_preservation_attempts,
        "STARTUP_LOCAL_PRESERVATION_RADIUS": startup_local_preservation_radius,
        "NFP": int(banana_surf_nfp),
        "FINAL_TOPOLOGY_GATE_EVALUATED": final_topology_status["evaluated"],
        "FINAL_TOPOLOGY_GATE_SUCCESS": final_topology_status["success"],
        "FINAL_TOPOLOGY_GATE_STATE": final_topology_status.get("state"),
        "FINAL_TOPOLOGY_GATE_ERROR": final_topology_status.get("evaluation_error"),
        "FINAL_TOPOLOGY_GATE_DIAGNOSTICS": build_topology_gate_diagnostics(
            final_topology_status,
            artifact_role="final_topology_gate",
        ),
        "FINAL_TOPOLOGY_TRANSPORT_DIAGNOSTICS": final_topology_status.get(
            "transport_diagnostics"
        ),
        "FINAL_TOPOLOGY_SURVIVED_LINES": final_topology_status["survived_lines"],
        "FINAL_TOPOLOGY_SURVIVAL_FRACTION": final_topology_status["survival_fraction"],
        "FINAL_TOPOLOGY_FIRST_EXIT_TIME": final_topology_status["first_exit_time"],
        "FINAL_TOPOLOGY_FIRST_EXIT_ANGLE": final_topology_status["first_exit_angle"],
        "FINAL_TOPOLOGY_FIRST_EXIT_REASON": final_topology_status["first_exit_reason"],
        "FINAL_TOPOLOGY_STOP_REASON_COUNTS": final_topology_status["stop_reason_counts"],
        "TARGET_VOLUME": None if args.single_stage_goal_mode == "frontier" else float(vol_target),
        "TARGET_IOTA": None if args.single_stage_goal_mode == "frontier" else float(iota_target),
        "BOOZER_SURFACE_TARGET_VOLUMES": [float(entry["target_volume"]) for entry in surface_data],
        "SINGLE_STAGE_GOAL_MODE": args.single_stage_goal_mode,
        "SINGLE_STAGE_GOAL_MODE_IMPL": current_frontier_goal_mode_impl(),
        "FRONTIER_REFERENCE_IOTA": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.iota_reference
        ),
        "FRONTIER_REFERENCE_IOTA_SCALE": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.iota_scale
        ),
        "FRONTIER_REFERENCE_VOLUME": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.volume_reference
        ),
        "FRONTIER_REFERENCE_VOLUME_SCALE": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.volume_scale
        ),
        "FRONTIER_REFERENCE_QA": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.qs_reference
        ),
        "FRONTIER_REFERENCE_BOOZER": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.boozer_reference
        ),
        "FRONTIER_SCALARIZATION_TYPE": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.scalarization_type
        ),
        "FRONTIER_CHEBYSHEV_RHO": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.chebyshev_rho
        ),
        "FRONTIER_CHEBYSHEV_SHARPNESS": (
            None
            if FRONTIER_GOAL_CONFIG is None
            else FRONTIER_GOAL_CONFIG.chebyshev_sharpness
        ),
        "FRONTIER_CHEBYSHEV_WEIGHT_IOTA": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.chebyshev_weight_iota
        ),
        "FRONTIER_CHEBYSHEV_WEIGHT_VOLUME": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.chebyshev_weight_volume
        ),
        "FRONTIER_CHEBYSHEV_WEIGHT_QA": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.chebyshev_weight_qa
        ),
        "FRONTIER_CHEBYSHEV_WEIGHT_BOOZER": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.chebyshev_weight_boozer
        ),
        "EPSILON_CONSTRAINT_QA_MAX": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.epsilon_constraint_qa_max
        ),
        "EPSILON_CONSTRAINT_BOOZER_MAX": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.epsilon_constraint_boozer_max
        ),
        "FRONTIER_EPSILON_PENALTY_WEIGHT": (
            None
            if FRONTIER_GOAL_CONFIG is None
            else FRONTIER_GOAL_CONFIG.epsilon_penalty_weight
        ),
        "FRONTIER_BOOZER_TRUST_PENALTY_SCALE": (
            None
            if FRONTIER_GOAL_CONFIG is None
            else FRONTIER_GOAL_CONFIG.boozer_trust_penalty_scale
        ),
        "FRONTIER_EFFECTIVE_QA_WEIGHT": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.effective_qs_weight
        ),
        "FRONTIER_EFFECTIVE_BOOZER_WEIGHT": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.effective_boozer_weight
        ),
        "FRONTIER_EFFECTIVE_IOTA_WEIGHT": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.effective_iota_weight
        ),
        "FRONTIER_EFFECTIVE_VOLUME_WEIGHT": (
            None if FRONTIER_GOAL_CONFIG is None else FRONTIER_GOAL_CONFIG.effective_volume_weight
        ),
        "FRONTIER_VOLUME_WEIGHT_INPUT": (
            args.frontier_volume_weight if args.single_stage_goal_mode == "frontier" else None
        ),
        "FRONTIER_BOOZER_TRUST_THRESHOLD": final_frontier_trust_status["threshold"],
        "PLASMA_CURRENT_A": float(plasma_current_A),
        "PLASMA_CURRENT_INPUT_SOURCE": plasma_current_input_source,
        "PLASMA_CURRENT_SURROGATE_SCOPE": (
            "shared_all_surfaces"
            if effective_num_surfaces > 1
            else "single_surface"
        ),
        "FINITE_CURRENT_MODE": finite_current_mode,
        "BOOZER_CURRENT_CONVENTION": boozer_current_convention,
        "EFFECTIVE_CURRENT_MODE": effective_current_mode,
        "BOOZER_I": float(boozer_I),
        "FINAL_VOLUME": float(final_volume),
        "FINAL_IOTA": float(final_iota),
        "FIELD_ERROR": float(fieldError),
        "OBJECTIVE_J": objective_j,
        "BASE_OBJECTIVE_J": base_objective_j,
        "SEARCH_OBJECTIVE_J": search_objective_j,
        **constraint_metadata,
        "FRONTIER_RANK_OBJECTIVE_J": (
            None if frontier_rank_objective_j is None else float(frontier_rank_objective_j)
        ),
        "FINAL_SEARCH_SURFACE_WEIGHTS": final_search_surface_weights,
        "FINAL_FEASIBILITY_OK": final_feasibility_ok,
        "SELF_INTERSECTING": run_dict['intersecting'],
        **build_single_stage_banana_current_payload_fields(banana_current_state),
        "BANANA_CURRENT_DIAGNOSTICS_ENABLED": (
            run_dict.get("banana_current_diagnostics") is not None
        ),
        "BANANA_CURRENT_DIAGNOSTICS_FILENAME": (
            BANANA_CURRENT_DIAGNOSTICS_FILENAME
            if run_dict.get("banana_current_diagnostics") is not None
            else None
        ),
        "BANANA_CURRENT_REPLAY_CONTEXT_FILENAME": (
            BANANA_CURRENT_REPLAY_CONTEXT_FILENAME
            if run_dict.get("banana_current_replay_context") is not None
            else None
        ),
        "BANANA_CURRENT_FD_DIAGNOSTICS_ENABLED": bool(
            args.banana_current_fd_diagnostics
        ),
        "BANANA_CURRENT_FD_RELATIVE_STEP_FRACTION": (
            float(args.banana_current_fd_relative_step_fraction)
            if args.banana_current_fd_diagnostics
            else None
        ),
        "BANANA_CURRENT_REPLAY_STUDY_ENABLED": bool(
            args.banana_current_replay_diagnostics_path
        ),
        "BANANA_CURRENT_REPLAY_STUDY_ARTIFACT_PATH": (
            None
            if banana_current_replay_summary is None
            else str(banana_current_replay_summary["artifact_path"])
        ),
        "BANANA_CURRENT_REPLAY_STUDY_FILENAME": (
            None
            if banana_current_replay_summary is None
            else BANANA_CURRENT_REJECTED_TRIAL_REPLAY_FILENAME
        ),
        **build_hardware_constraint_artifact_payload_fields(final_hardware_snapshot),
        "INVALID_STATE_REJECTS_TOTAL": run_dict["invalid_state_rejects_total"],
        "TOPOLOGY_GATE_REJECTS": run_dict["topology_gate_rejects"],
        "HARDWARE_REJECTS": run_dict["hardware_rejects"],
        "CURVATURE_PRECHECK_REJECTS": run_dict["curvature_precheck_rejects"],
        "CURVATURE_OVERCAP_BOOZER_EVALS": run_dict[
            "curvature_overcap_boozer_evals"
        ],
        "SURFACE_SOLVE_REJECTS": run_dict["surface_solve_rejects"],
        "FRONTIER_TRUST_REJECTS": run_dict["frontier_trust_rejects"],
        **search_step_metrics_payload(run_dict),
        "NONQS_RATIO": nonqs_ratio,
        "BOOZER_RESIDUAL": boozer_residual,
        "FRONTIER_TRUST_OK": final_frontier_trust_status["ok"],
        "FRONTIER_BOOZER_TRUST_EXCESS": final_frontier_trust_status["excess"],
        "FRONTIER_BOOZER_TRUST_EXCESS_RATIO": run_dict["search_eval"].get(
            "frontier_boozer_trust_excess_ratio"
        ),
        "FRONTIER_TRUST_PENALTY": run_dict["search_eval"].get("frontier_trust_penalty"),
        "FRONTIER_CONTRACT_PENALTY": run_dict["search_eval"].get("frontier_contract_penalty"),
        "FRONTIER_EPSILON_PENALTY": run_dict["search_eval"].get("frontier_epsilon_penalty"),
        "FRONTIER_HARDWARE_PENALTY": run_dict["search_eval"].get("frontier_hardware_penalty"),
        "FRONTIER_HARDWARE_MAX_VIOLATION_RATIO": run_dict["search_eval"].get(
            "frontier_hardware_max_violation_ratio"
        ),
        "FRONTIER_TOPOLOGY_PENALTY": run_dict["search_eval"].get("frontier_topology_penalty"),
        "FRONTIER_TOPOLOGY_DEFICIT": run_dict["search_eval"].get(
            "frontier_topology_deficit"
        ),
        "FRONTIER_CONDITIONING_SCHEMA_VERSION": FRONTIER_CONDITIONING_SCHEMA_VERSION,
        "FRONTIER_CONDITIONING_SEED_REPORT": run_dict.get(
            "frontier_conditioning_seed_report"
        ),
        "FRONTIER_CONDITIONING_FIRST_ACCEPTED_REPORT": run_dict.get(
            "frontier_conditioning_first_accepted_report"
        ),
        "FRONTIER_CONDITIONING_GATE": frontier_conditioning_gate,
        "FRONTIER_CONDITIONING_GATE_OK": frontier_conditioning_gate.get(
            "usable_scale_ok"
        ),
        "FRONTIER_VOLUME_OBJECTIVE": (
            None if args.init_only else float(run_dict["search_eval"].get("J_volume", 0.0))
        ),
        "INITIAL_VOLUME": float(initial_volume),
        "INITIAL_IOTA": float(initial_iota),
        "INITIAL_FIELD_ERROR": float(initial_field_error),
        "INITIAL_MAX_CURVATURE": float(initial_max_curvature),
        "BEST_TOPOLOGY_ACCEPTED_ITERATION": run_dict.get("best_topology", {}).get("accepted_iteration"),
        "BEST_TOPOLOGY_CONFINEMENT_SCORE": run_dict.get("best_topology", {}).get("confinement_score"),
        "BEST_TOPOLOGY_CONFINEMENT_LOSS": run_dict.get("best_topology", {}).get("confinement_loss"),
        "BEST_TOPOLOGY_TRANSPORT_DIAGNOSTICS": run_dict.get(
            "best_topology",
            {},
        ).get("transport_diagnostics"),
        "BEST_TOPOLOGY_DIAGNOSTICS": optional_topology_score_diagnostics(
            run_dict.get("best_topology"),
            artifact_role="best_topology",
        ),
        "BEST_CONFINEMENT_OBJECTIVE_ACCEPTED_ITERATION": run_dict.get("best_confinement_objective", {}).get("accepted_iteration"),
        "BEST_CONFINEMENT_OBJECTIVE_TOTAL": run_dict.get("best_confinement_objective", {}).get("checkpoint_objective_total"),
        "BEST_CONFINEMENT_OBJECTIVE_PROXY_J": run_dict.get("best_confinement_objective", {}).get("J"),
        "BEST_CONFINEMENT_OBJECTIVE_LOSS": run_dict.get("best_confinement_objective", {}).get("confinement_loss"),
        "BEST_CONFINEMENT_OBJECTIVE_TRANSPORT_DIAGNOSTICS": run_dict.get(
            "best_confinement_objective",
            {},
        ).get("transport_diagnostics"),
        "BEST_CONFINEMENT_OBJECTIVE_DIAGNOSTICS": optional_topology_score_diagnostics(
            run_dict.get("best_confinement_objective"),
            artifact_role="best_confinement_objective",
        ),
    }
    results.update(best_feasible_results)
    results.update(
        collect_surface_run_metadata(
            surface_data,
            run_dict['surface_status'],
            initial_surface_volumes,
            initial_surface_iotas,
            final_surface_volumes,
            final_surface_iotas,
        )
    )
    write_json_artifact(os.path.join(OUT_DIR_ITER, "results.json"), results)
