#!/usr/bin/env python3
"""Preflight and run entrypoint for DESC joint banana optimization."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

DescRuntimeDeviceName = Literal["cpu", "gpu"]
_DESC_JOINT_RUNTIME_DEVICE_ENV = "DESC_JOINT_DESC_DEVICE"
_DESC_RUNTIME_DEVICE_NAMES: tuple[DescRuntimeDeviceName, ...] = ("cpu", "gpu")


def _bootstrap_desc_runtime_device(argv: Sequence[str]) -> DescRuntimeDeviceName | None:
    """Select DESC's runtime device before any DESC backend module is imported."""

    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument("--desc-source-root")
    bootstrap_parser.add_argument(
        "--desc-runtime-device",
        choices=_DESC_RUNTIME_DEVICE_NAMES,
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args(argv)
    raw_device = bootstrap_args.desc_runtime_device
    if raw_device is None:
        raw_device = os.environ.get(_DESC_JOINT_RUNTIME_DEVICE_ENV)
    if raw_device is None or raw_device == "":
        return None
    normalized_device = raw_device.lower()
    if normalized_device == "cpu":
        selected_device: DescRuntimeDeviceName = "cpu"
    elif normalized_device == "gpu":
        selected_device = "gpu"
    else:
        raise SystemExit(
            f"{_DESC_JOINT_RUNTIME_DEVICE_ENV} must be one of "
            f"{', '.join(_DESC_RUNTIME_DEVICE_NAMES)}; got {raw_device!r}."
        )
    desc_source_root = bootstrap_args.desc_source_root
    if desc_source_root is not None:
        resolved_root = Path(desc_source_root).expanduser().resolve()
        if not resolved_root.is_dir():
            raise SystemExit(
                "--desc-source-root must point to an existing directory before "
                f"DESC runtime device bootstrap; got {desc_source_root!r}."
            )
        sys.path.insert(0, os.fspath(resolved_root))
    import desc

    desc.set_device(kind=selected_device)
    return selected_device


_BOOTSTRAPPED_DESC_RUNTIME_DEVICE = _bootstrap_desc_runtime_device(sys.argv[1:])

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
if os.fspath(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(EXAMPLES_ROOT))

from banana_opt.desc_bridge.equilibrium_seed import (  # noqa: E402
    DescEquilibriumRuntimeLoadError,
    DescEquilibriumSeedSpec,
    load_desc_equilibrium_seed_runtime,
    load_desc_equilibrium_seed_spec,
)
from banana_opt.desc_bridge.conversion_artifacts import (  # noqa: E402
    DescConversionOnlyArtifacts,
    materialize_conversion_only_artifacts,
)
from banana_opt.desc_bridge.objective_factory import (  # noqa: E402
    BOUNDARY_FIDELITY_OFF,
    DEFAULT_BOUNDARY_FIDELITY_FREE_MODE_SUM,
    DEFAULT_DESC_OBJECTIVE_DERIV_MODE,
    DEFAULT_DESC_OBJECTIVE_USE_JIT,
    DESC_BOUNDARY_FIDELITY_POLICIES,
    DESC_JOINT_CONSTRAINT_POLICIES,
    DESC_OBJECTIVE_ABLATION_POLICIES,
    DESC_OBJECTIVE_DERIV_MODES,
    FULL_DESC_OBJECTIVE_ABLATION_POLICY,
    HARD_VOLUME_AND_FORCE_BALANCE_POLICY,
    DescBoundaryFidelityPolicy,
    DescJointConstraintPolicy,
    DescObjectiveAblationPolicy,
    DescObjectiveDerivMode,
    DescObjectiveRuntimeAssemblyError,
    DescObjectiveRuntimeEvaluationError,
    assemble_desc_objective_stack_runtime,
    build_desc_objective_stack_plan,
    evaluate_desc_objective_stack_runtime,
)
from banana_opt.desc_bridge.runtime_coilset import (  # noqa: E402
    DEFAULT_OPTIMIZED_COIL_GROUPS,
    DescRuntimeCoilsetBuildError,
    build_desc_runtime_coilset_from_simsopt_field,
    load_desc_runtime_coilset_checkpoint,
    scope_desc_coilset_optimization_to_groups,
)
from banana_opt.desc_bridge.runtime_export import (  # noqa: E402
    DescOptimizedSimsoptExportArtifacts,
    DescOptimizedSimsoptExportError,
    DescOptimizedSurfaceExportArtifacts,
    DescOptimizedSurfaceExportError,
    materialize_optimized_desc_coil_artifact_simsopt_export,
    materialize_optimized_desc_equilibrium_surface_simsopt_export,
)
from banana_opt.desc_bridge.runtime_solve import (  # noqa: E402
    DescFixedPolishRuntimeSolveError,
    DescFixedPolishRuntimeSolveReport,
    DescJointRuntimeSolveError,
    DescJointRuntimeSolveReport,
    DescOptimizerControls,
    build_desc_optimizer_controls,
    desc_fixed_equilibrium_polish_setup_failure_report,
    desc_high_memory_optimizer_blocked_reason,
    desc_joint_optimization_setup_failure_report,
    run_desc_fixed_equilibrium_polish_runtime,
    run_desc_joint_optimization_runtime,
)
from banana_opt.desc_joint_hardware_spec import (  # noqa: E402
    load_desc_joint_hardware_spec,
)
from banana_opt.desc_joint_field_inventory import (  # noqa: E402
    load_desc_joint_field_inventory,
)
from banana_opt.desc_joint_result_schema import (  # noqa: E402
    DescJointRunMode,
    DescJointStatus,
    build_preflight_result_payload,
    validate_desc_joint_mode,
    validate_desc_joint_result_payload,
)
from banana_opt.desc_joint_validation import (  # noqa: E402
    build_desc_joint_validation_manifest,
    render_desc_joint_validation_report,
)
from banana_opt.desc_joint_seed_manifest import (  # noqa: E402
    DescJointSeedCandidate,
    load_desc_joint_seed_manifest,
)
from banana_opt.hardware_constraint_schema import (  # noqa: E402
    build_hardware_constraint_artifact_payload_fields,
)
from simsopt import load as load_simsopt  # noqa: E402

ResolutionPresetName = Literal["smoke", "production"]


@dataclass(frozen=True, slots=True)
class _DescJointResolutionPreset:
    name: ResolutionPresetName
    desc_fourier_order: int
    conversion_sample_count: int
    simsopt_fourier_order: int
    desc_equilibrium_lcfs_mpol: int
    desc_equilibrium_lcfs_ntor: int
    desc_grid_n: int
    desc_bs_chunk_size: int
    desc_dist_chunk_size: int
    desc_jac_chunk_size: int


_DESC_JOINT_RESOLUTION_PRESETS: dict[
    ResolutionPresetName,
    _DescJointResolutionPreset,
] = {
    "smoke": _DescJointResolutionPreset(
        name="smoke",
        desc_fourier_order=3,
        conversion_sample_count=64,
        simsopt_fourier_order=3,
        desc_equilibrium_lcfs_mpol=4,
        desc_equilibrium_lcfs_ntor=4,
        desc_grid_n=50,
        desc_bs_chunk_size=10,
        desc_dist_chunk_size=2,
        desc_jac_chunk_size=5,
    ),
    "production": _DescJointResolutionPreset(
        name="production",
        desc_fourier_order=10,
        conversion_sample_count=256,
        simsopt_fourier_order=10,
        desc_equilibrium_lcfs_mpol=10,
        desc_equilibrium_lcfs_ntor=10,
        desc_grid_n=96,
        desc_bs_chunk_size=25,
        desc_dist_chunk_size=8,
        desc_jac_chunk_size=10,
    ),
}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    preflight_start = time.perf_counter()
    _resolve_resolution_args(args)
    validate_desc_joint_mode(args.mode)
    mode: DescJointRunMode = args.mode
    hardware_spec = load_desc_joint_hardware_spec(args.hardware_spec)
    seed_manifest = load_desc_joint_seed_manifest(args.seed_manifest)
    selected_seed = seed_manifest.candidate_by_label(args.seed_label)
    selected_field_inventory = load_desc_joint_field_inventory(selected_seed.field_path)
    equilibrium_seed = load_desc_equilibrium_seed_spec(args.equilibrium_seed)
    _validate_seed_source_matches_equilibrium_seed(selected_seed, equilibrium_seed)
    equilibrium_seed = _attach_selected_seed_target_lcfs_G(
        selected_seed,
        equilibrium_seed,
    )
    equilibrium_seed = equilibrium_seed.with_lcfs_resolution(
        lcfs_mpol=args.desc_equilibrium_lcfs_mpol,
        lcfs_ntor=args.desc_equilibrium_lcfs_ntor,
    )
    desc_source_root = (
        None
        if args.desc_source_root is None
        else _require_existing_directory(
            args.desc_source_root,
            field_name="--desc-source-root",
        )
    )
    desc_coilset_checkpoint = (
        None
        if args.desc_coilset_checkpoint is None
        else _require_existing_file(
            args.desc_coilset_checkpoint,
            field_name="--desc-coilset-checkpoint",
        )
    )
    fixed_polish_predecessor_manifest = _optional_existing_file(
        args.fixed_polish_predecessor_manifest,
        field_name="--fixed-polish-predecessor-manifest",
    )
    lane_b_predecessor_manifest = _optional_existing_file(
        args.lane_b_predecessor_manifest,
        field_name="--lane-b-predecessor-manifest",
    )
    _validate_predecessor_manifest_args(
        mode=mode,
        fixed_polish_predecessor_manifest=fixed_polish_predecessor_manifest,
        lane_b_predecessor_manifest=lane_b_predecessor_manifest,
    )
    optimized_coil_groups = _parse_optimized_coil_groups(
        args.desc_optimized_coil_groups,
    )
    optimizer_controls = _desc_optimizer_controls_from_args(args)
    objective_stack = build_desc_objective_stack_plan(
        mode,
        include_hardware_keepout=hardware_spec.hardware_sdf_manifest_path is not None,
        joint_constraint_policy=args.desc_joint_constraint_policy,
        boundary_fidelity_policy=args.desc_boundary_fidelity_policy,
        objective_ablation_policy=args.desc_objective_ablation_policy,
    )
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    input_contract = {
        "hardware": hardware_spec.to_input_contract(),
        "seed_manifest": seed_manifest.to_input_contract(),
        "selected_seed": selected_seed.to_json_dict(),
        "selected_seed_field_inventory": selected_field_inventory.to_json_dict(),
        "selected_seed_coil_group_counts": {
            coil_group.name: coil_group.count
            for coil_group in selected_seed.coil_groups
        },
        "selected_seed_coil_group_source": selected_seed.coil_group_source,
        "equilibrium_seed": equilibrium_seed.to_input_contract(),
        "desc_source_root": (
            None if desc_source_root is None else os.fspath(desc_source_root)
        ),
        "desc_coilset_checkpoint": (
            None
            if desc_coilset_checkpoint is None
            else os.fspath(desc_coilset_checkpoint)
        ),
    }
    preflight_payload = build_preflight_result_payload(
        mode=mode,
        input_contract=input_contract,
        objective_stack=[entry.name for entry in objective_stack],
    )
    _attach_predecessor_statuses(
        preflight_payload,
        fixed_polish_predecessor_manifest=fixed_polish_predecessor_manifest,
        lane_b_predecessor_manifest=lane_b_predecessor_manifest,
    )
    preflight_payload["run_configuration"] = _run_configuration_payload(
        args,
        desc_source_root=desc_source_root,
        optimized_coil_groups=optimized_coil_groups,
        optimizer_controls=optimizer_controls,
        fixed_polish_predecessor_manifest=fixed_polish_predecessor_manifest,
        lane_b_predecessor_manifest=lane_b_predecessor_manifest,
    )
    preflight_payload["objective_stack_plan"] = [
        entry.to_json_dict()
        for entry in objective_stack
    ]
    preflight_payload["run_timing_seconds"] = _empty_run_timing_seconds()
    preflight_payload["run_timing_seconds"]["preflight"] = _elapsed(preflight_start)
    preflight_path = output_root / "desc_joint_preflight.json"
    preflight_path.write_text(
        json.dumps(preflight_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.preflight_only:
        print(os.fspath(preflight_path))
        return 0
    if args.equilibrium_load_only:
        load_report_path, exit_code = _run_equilibrium_load_only_lane(
            mode=mode,
            output_root=output_root,
            equilibrium_seed=equilibrium_seed,
            desc_source_root=desc_source_root,
        )
        print(os.fspath(load_report_path))
        return exit_code
    if args.objective_assembly_only:
        objective_report_path, exit_code = _run_objective_assembly_only_lane(
            mode=mode,
            output_root=output_root,
            hardware_keepout_enabled=hardware_spec.hardware_sdf_manifest_path
            is not None,
            hardware_sdf_manifest_path=hardware_spec.hardware_sdf_manifest_path,
            hardware_glb_path=hardware_spec.glb_path,
            selected_seed=selected_seed,
            equilibrium_seed=equilibrium_seed,
            desc_source_root=desc_source_root,
            desc_coilset_checkpoint=desc_coilset_checkpoint,
            desc_fourier_order=args.desc_fourier_order,
            sample_count=args.conversion_sample_count,
            grid_n=args.desc_grid_n,
            bs_chunk_size=args.desc_bs_chunk_size,
            dist_chunk_size=args.desc_dist_chunk_size,
            jac_chunk_size=args.desc_jac_chunk_size,
            objective_use_jit=args.desc_objective_use_jit,
            objective_deriv_mode=args.desc_objective_deriv_mode,
            joint_constraint_policy=args.desc_joint_constraint_policy,
            boundary_fidelity_policy=args.desc_boundary_fidelity_policy,
            boundary_fidelity_free_mode_sum=args.desc_boundary_fidelity_free_mode_sum,
            objective_ablation_policy=args.desc_objective_ablation_policy,
            optimized_coil_groups=optimized_coil_groups,
            evaluate_objective=False,
            evaluate_jacobian=False,
            evaluate_gradient=False,
        )
        print(os.fspath(objective_report_path))
        return exit_code
    if args.objective_eval_only:
        objective_report_path, exit_code = _run_objective_assembly_only_lane(
            mode=mode,
            output_root=output_root,
            hardware_keepout_enabled=hardware_spec.hardware_sdf_manifest_path
            is not None,
            hardware_sdf_manifest_path=hardware_spec.hardware_sdf_manifest_path,
            hardware_glb_path=hardware_spec.glb_path,
            selected_seed=selected_seed,
            equilibrium_seed=equilibrium_seed,
            desc_source_root=desc_source_root,
            desc_coilset_checkpoint=desc_coilset_checkpoint,
            desc_fourier_order=args.desc_fourier_order,
            sample_count=args.conversion_sample_count,
            grid_n=args.desc_grid_n,
            bs_chunk_size=args.desc_bs_chunk_size,
            dist_chunk_size=args.desc_dist_chunk_size,
            jac_chunk_size=args.desc_jac_chunk_size,
            objective_use_jit=args.desc_objective_use_jit,
            objective_deriv_mode=args.desc_objective_deriv_mode,
            joint_constraint_policy=args.desc_joint_constraint_policy,
            boundary_fidelity_policy=args.desc_boundary_fidelity_policy,
            boundary_fidelity_free_mode_sum=args.desc_boundary_fidelity_free_mode_sum,
            objective_ablation_policy=args.desc_objective_ablation_policy,
            optimized_coil_groups=optimized_coil_groups,
            evaluate_objective=True,
            evaluate_jacobian=args.objective_eval_jacobian,
            evaluate_gradient=args.objective_eval_gradient,
        )
        print(os.fspath(objective_report_path))
        return exit_code
    if args.fixed_polish_only:
        result_path, exit_code = _run_fixed_polish_only_lane(
            mode=mode,
            output_root=output_root,
            hardware_keepout_enabled=hardware_spec.hardware_sdf_manifest_path
            is not None,
            hardware_sdf_manifest_path=hardware_spec.hardware_sdf_manifest_path,
            hardware_glb_path=hardware_spec.glb_path,
            selected_seed=selected_seed,
            equilibrium_seed=equilibrium_seed,
            desc_source_root=desc_source_root,
            desc_coilset_checkpoint=desc_coilset_checkpoint,
            preflight_payload=preflight_payload,
            desc_fourier_order=args.desc_fourier_order,
            sample_count=args.conversion_sample_count,
            grid_n=args.desc_grid_n,
            bs_chunk_size=args.desc_bs_chunk_size,
            dist_chunk_size=args.desc_dist_chunk_size,
            jac_chunk_size=args.desc_jac_chunk_size,
            objective_use_jit=args.desc_objective_use_jit,
            objective_deriv_mode=args.desc_objective_deriv_mode,
            boundary_fidelity_policy=args.desc_boundary_fidelity_policy,
            boundary_fidelity_free_mode_sum=args.desc_boundary_fidelity_free_mode_sum,
            optimizer_method=args.desc_optimizer_method,
            maxiter=args.desc_maxiter,
            optimizer_verbose=args.desc_optimizer_verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=args.allow_high_memory_desc_optimizer,
            simsopt_fourier_order=args.simsopt_fourier_order,
            optimized_coil_groups=optimized_coil_groups,
        )
        print(os.fspath(result_path))
        return exit_code
    if args.joint_run_only:
        result_path, exit_code = _run_joint_run_only_lane(
            mode=mode,
            output_root=output_root,
            hardware_keepout_enabled=hardware_spec.hardware_sdf_manifest_path
            is not None,
            hardware_sdf_manifest_path=hardware_spec.hardware_sdf_manifest_path,
            hardware_glb_path=hardware_spec.glb_path,
            selected_seed=selected_seed,
            equilibrium_seed=equilibrium_seed,
            desc_source_root=desc_source_root,
            desc_coilset_checkpoint=desc_coilset_checkpoint,
            preflight_payload=preflight_payload,
            desc_fourier_order=args.desc_fourier_order,
            sample_count=args.conversion_sample_count,
            grid_n=args.desc_grid_n,
            bs_chunk_size=args.desc_bs_chunk_size,
            dist_chunk_size=args.desc_dist_chunk_size,
            jac_chunk_size=args.desc_jac_chunk_size,
            objective_use_jit=args.desc_objective_use_jit,
            objective_deriv_mode=args.desc_objective_deriv_mode,
            joint_constraint_policy=args.desc_joint_constraint_policy,
            boundary_fidelity_policy=args.desc_boundary_fidelity_policy,
            boundary_fidelity_free_mode_sum=args.desc_boundary_fidelity_free_mode_sum,
            objective_ablation_policy=args.desc_objective_ablation_policy,
            optimizer_method=args.desc_optimizer_method,
            maxiter=args.desc_maxiter,
            optimizer_verbose=args.desc_optimizer_verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=args.allow_high_memory_desc_optimizer,
            simsopt_fourier_order=args.simsopt_fourier_order,
            optimized_coil_groups=optimized_coil_groups,
        )
        print(os.fspath(result_path))
        return exit_code
    if args.conversion_only:
        conversion_result_path = _run_conversion_only_lane(
            mode=mode,
            output_root=output_root,
            selected_seed=selected_seed,
            preflight_payload=preflight_payload,
            desc_fourier_order=args.desc_fourier_order,
            sample_count=args.conversion_sample_count,
            simsopt_fourier_order=args.simsopt_fourier_order,
        )
        print(os.fspath(conversion_result_path))
        return 0
    raise NotImplementedError(
        "No DESC execution lane was selected. Use --preflight-only, "
        "--equilibrium-load-only, --objective-assembly-only, --objective-eval-only, "
        "--conversion-only, --fixed-polish-only, or --joint-run-only."
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Preflight DESC hardware-first banana joint optimization.",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("fixed_equilibrium_polish", "vacuum_joint", "finite_beta_joint"),
    )
    parser.add_argument("--hardware-spec", required=True)
    parser.add_argument("--seed-manifest", required=True)
    parser.add_argument("--seed-label", required=True)
    parser.add_argument("--equilibrium-seed", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--desc-source-root")
    parser.add_argument(
        "--desc-coilset-checkpoint",
        help=(
            "Load a saved DESC CoilSet checkpoint instead of rebuilding the "
            "runtime CoilSet from the SIMSOPT seed field. Intended for explicit "
            "optimizer continuation; it does not make failed checkpoints "
            "promotion artifacts."
        ),
    )
    parser.add_argument(
        "--fixed-polish-predecessor-manifest",
        help=(
            "Passed Lane A fixed-equilibrium polish validation manifest required "
            "to promote vacuum_joint or finite_beta_joint results."
        ),
    )
    parser.add_argument(
        "--lane-b-predecessor-manifest",
        help=(
            "Passed Lane B vacuum_joint validation manifest required to promote "
            "finite_beta_joint results."
        ),
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--equilibrium-load-only",
        action="store_true",
        help=(
            "Load the declared DESC equilibrium seed with DESC runtime APIs and "
            "write desc_equilibrium_load_report.json without optimizing."
        ),
    )
    parser.add_argument(
        "--objective-assembly-only",
        action="store_true",
        help=(
            "Load the DESC equilibrium seed, build a DESC CoilSet from the "
            "SIMSOPT seed field, assemble the DESC objective stack, and write "
            "runtime reports without optimizing."
        ),
    )
    parser.add_argument(
        "--objective-eval-only",
        action="store_true",
        help=(
            "Run the objective assembly lane and perform one DESC objective "
            "value smoke evaluation without optimizing."
        ),
    )
    parser.add_argument(
        "--objective-eval-jacobian",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether --objective-eval-only also computes jac_scaled_error.",
    )
    parser.add_argument(
        "--objective-eval-gradient",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Whether --objective-eval-only also computes scalar gradients "
            "sequentially per objective term."
        ),
    )
    parser.add_argument(
        "--conversion-only",
        action="store_true",
        help=(
            "Execute the SIMSOPT -> sampled-DESC-coil -> SIMSOPT bridge and "
            "write artifacts without running DESC optimization."
        ),
    )
    parser.add_argument(
        "--fixed-polish-only",
        action="store_true",
        help=(
            "Run Lane A fixed-equilibrium DESC coil polish. Without "
            "--allow-high-memory-desc-optimizer, fail closed before DESC "
            "optimizer execution and write desc_result.json without optimized "
            "artifacts."
        ),
    )
    parser.add_argument(
        "--joint-run-only",
        action="store_true",
        help=(
            "Run DESC joint equilibrium+coil optimization for vacuum_joint or "
            "finite_beta_joint. Without --allow-high-memory-desc-optimizer, "
            "fail closed before DESC optimizer execution and write "
            "desc_result.json without optimized artifacts."
        ),
    )
    parser.add_argument(
        "--resolution-preset",
        choices=tuple(_DESC_JOINT_RESOLUTION_PRESETS),
        default="smoke",
        help=(
            "Resolution profile for operational defaults. Individual resolution "
            "flags override the preset."
        ),
    )
    parser.add_argument("--desc-fourier-order", type=int)
    parser.add_argument("--conversion-sample-count", type=int)
    parser.add_argument("--simsopt-fourier-order", type=int)
    parser.add_argument("--desc-equilibrium-lcfs-mpol", type=int)
    parser.add_argument("--desc-equilibrium-lcfs-ntor", type=int)
    parser.add_argument("--desc-grid-n", type=int)
    parser.add_argument("--desc-bs-chunk-size", type=int)
    parser.add_argument("--desc-dist-chunk-size", type=int)
    parser.add_argument("--desc-jac-chunk-size", type=int)
    parser.add_argument(
        "--desc-objective-use-jit",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_DESC_OBJECTIVE_USE_JIT,
        help=(
            "Whether the top-level DESC ObjectiveFunction uses JIT-compiled "
            "combined compute/jacobian methods. Default: no-JIT for memory-safe "
            "banana debug lanes."
        ),
    )
    parser.add_argument(
        "--desc-objective-deriv-mode",
        choices=DESC_OBJECTIVE_DERIV_MODES,
        default=DEFAULT_DESC_OBJECTIVE_DERIV_MODE,
        help=(
            "Top-level DESC ObjectiveFunction derivative mode. Default: blocked "
            "to keep objective-term derivative work separated."
        ),
    )
    parser.add_argument(
        "--desc-joint-constraint-policy",
        choices=DESC_JOINT_CONSTRAINT_POLICIES,
        default=HARD_VOLUME_AND_FORCE_BALANCE_POLICY,
        help=(
            "Joint-mode constraint formulation. Default keeps Volume and "
            "ForceBalance as hard constraints. hard-hardware-and-force-balance "
            "also routes bounded hardware terms into DESC constraints. "
            "hard-linking-current-and-force-balance keeps coil/plasma linking "
            "current and ForceBalance hard while staging Volume as an objective. "
            "proximal-force-balance keeps ForceBalance hard and stages Volume "
            "as an objective so DESC proximal optimizers can project force "
            "balance."
        ),
    )
    parser.add_argument(
        "--desc-boundary-fidelity-policy",
        choices=DESC_BOUNDARY_FIDELITY_POLICIES,
        default=BOUNDARY_FIDELITY_OFF,
        help=(
            "Joint-mode LCFS topology guard. Default off. fix-high-modes fixes "
            "boundary R/Z modes with |m| + |n| above "
            "--desc-boundary-fidelity-free-mode-sum while leaving low-order "
            "boundary modes free."
        ),
    )
    parser.add_argument(
        "--desc-boundary-fidelity-free-mode-sum",
        type=int,
        default=DEFAULT_BOUNDARY_FIDELITY_FREE_MODE_SUM,
        help=(
            "For --desc-boundary-fidelity-policy fix-high-modes, keep boundary "
            "modes with |m| + |n| <= this value free and fix higher modes."
        ),
    )
    parser.add_argument(
        "--desc-objective-ablation-policy",
        choices=DESC_OBJECTIVE_ABLATION_POLICIES,
        default=FULL_DESC_OBJECTIVE_ABLATION_POLICY,
        help=(
            "Joint-mode diagnostic objective ablation. Default keeps the full "
            "DESC objective stack; non-default policies remove objective "
            "families before runtime assembly instead of zero-weighting them."
        ),
    )
    parser.add_argument(
        "--desc-optimized-coil-groups",
        default=",".join(DEFAULT_OPTIMIZED_COIL_GROUPS),
        help=(
            "Comma-separated seed coil groups exposed to DESC optimizer "
            "variables. Default: banana."
        ),
    )
    parser.add_argument(
        "--desc-runtime-device",
        choices=_DESC_RUNTIME_DEVICE_NAMES,
        default=os.environ.get(_DESC_JOINT_RUNTIME_DEVICE_ENV),
        help=(
            "Select DESC's runtime device before importing desc.backend. "
            f"Defaults to ${_DESC_JOINT_RUNTIME_DEVICE_ENV} when set."
        ),
    )
    parser.add_argument("--desc-optimizer-method", default="lsq-exact")
    parser.add_argument("--desc-maxiter", type=int)
    parser.add_argument("--desc-optimizer-verbose", type=int, default=1)
    parser.add_argument("--desc-optimizer-ftol", type=float)
    parser.add_argument("--desc-optimizer-xtol", type=float)
    parser.add_argument("--desc-optimizer-gtol", type=float)
    parser.add_argument("--desc-optimizer-ctol", type=float)
    parser.add_argument("--desc-optimizer-max-nfev", type=int)
    parser.add_argument("--desc-optimizer-max-dx", type=float)
    parser.add_argument("--desc-optimizer-initial-trust-radius", type=float)
    parser.add_argument("--desc-optimizer-max-trust-radius", type=float)
    parser.add_argument("--desc-optimizer-min-trust-radius", type=float)
    parser.add_argument("--desc-proximal-perturb-order", type=int)
    parser.add_argument("--desc-proximal-solve-maxiter", type=int)
    parser.add_argument(
        "--desc-proximal-solve-during-build",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pass solve_options['solve_during_proximal_build'] to DESC "
            "ProximalProjection for prox-/proximal- optimizer methods."
        ),
    )
    parser.add_argument(
        "--allow-high-memory-desc-optimizer",
        action="store_true",
        help=(
            "Allow fixed-polish/joint DESC optimizer execution. By default "
            "these lanes fail closed because DESC's combined ObjectiveFunction "
            "optimizer path is not memory-bounded for real banana seeds."
        ),
    )
    return parser


def _resolve_resolution_args(args: argparse.Namespace) -> None:
    preset = _DESC_JOINT_RESOLUTION_PRESETS[args.resolution_preset]
    _resolve_resolution_arg(args, "desc_fourier_order", preset.desc_fourier_order)
    _resolve_resolution_arg(
        args,
        "conversion_sample_count",
        preset.conversion_sample_count,
    )
    _resolve_resolution_arg(args, "simsopt_fourier_order", preset.simsopt_fourier_order)
    _resolve_resolution_arg(
        args,
        "desc_equilibrium_lcfs_mpol",
        preset.desc_equilibrium_lcfs_mpol,
    )
    _resolve_resolution_arg(
        args,
        "desc_equilibrium_lcfs_ntor",
        preset.desc_equilibrium_lcfs_ntor,
    )
    _resolve_resolution_arg(args, "desc_grid_n", preset.desc_grid_n)
    _resolve_resolution_arg(args, "desc_bs_chunk_size", preset.desc_bs_chunk_size)
    _resolve_resolution_arg(args, "desc_dist_chunk_size", preset.desc_dist_chunk_size)
    _resolve_resolution_arg(args, "desc_jac_chunk_size", preset.desc_jac_chunk_size)


def _parse_optimized_coil_groups(raw_value: str) -> tuple[str, ...]:
    groups = tuple(group.strip() for group in raw_value.split(","))
    if not groups or any(group == "" for group in groups):
        raise ValueError("--desc-optimized-coil-groups must name at least one group.")
    duplicates = sorted(group for group in set(groups) if groups.count(group) > 1)
    if duplicates:
        raise ValueError(
            "--desc-optimized-coil-groups contains duplicate groups: "
            f"{', '.join(duplicates)}."
        )
    return groups


def _resolve_resolution_arg(
    args: argparse.Namespace,
    field_name: str,
    preset_value: int,
) -> None:
    raw_value = getattr(args, field_name)
    value = preset_value if raw_value is None else raw_value
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    setattr(args, field_name, value)


def _require_existing_directory(path: str, *, field_name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"{field_name} does not exist: {resolved}.")
    if not resolved.is_dir():
        raise ValueError(f"{field_name} must be a directory: {resolved}.")
    return resolved


def _require_existing_file(path: str, *, field_name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise ValueError(f"{field_name} does not exist: {resolved}.")
    if not resolved.is_file():
        raise ValueError(f"{field_name} must be a file: {resolved}.")
    return resolved


def _optional_existing_file(path: str | None, *, field_name: str) -> Path | None:
    if path is None:
        return None
    return _require_existing_file(path, field_name=field_name)


def _validate_predecessor_manifest_args(
    *,
    mode: DescJointRunMode,
    fixed_polish_predecessor_manifest: Path | None,
    lane_b_predecessor_manifest: Path | None,
) -> None:
    if mode == "fixed_equilibrium_polish" and (
        fixed_polish_predecessor_manifest is not None
        or lane_b_predecessor_manifest is not None
    ):
        raise ValueError(
            "fixed_equilibrium_polish is the Lane A predecessor; predecessor "
            "manifests are only valid for joint modes."
        )
    if mode == "vacuum_joint" and lane_b_predecessor_manifest is not None:
        raise ValueError(
            "--lane-b-predecessor-manifest is only valid for finite_beta_joint."
        )


def _attach_predecessor_statuses(
    payload: dict[str, object],
    *,
    fixed_polish_predecessor_manifest: Path | None,
    lane_b_predecessor_manifest: Path | None,
) -> None:
    if fixed_polish_predecessor_manifest is not None:
        payload["fixed_polish_predecessor_status"] = _predecessor_status(
            fixed_polish_predecessor_manifest,
            reason="fixed-polish predecessor validation manifest supplied",
        )
    if lane_b_predecessor_manifest is not None:
        payload["lane_b_predecessor_status"] = _predecessor_status(
            lane_b_predecessor_manifest,
            reason="Lane B predecessor validation manifest supplied",
        )


def _predecessor_status(path: Path, *, reason: str) -> dict[str, object]:
    return {
        "state": "passed",
        "reason": reason,
        "artifact_paths": [os.fspath(path)],
    }


def _run_conversion_only_lane(
    *,
    mode: DescJointRunMode,
    output_root: Path,
    selected_seed: DescJointSeedCandidate,
    preflight_payload: dict[str, object],
    desc_fourier_order: int,
    sample_count: int,
    simsopt_fourier_order: int,
) -> Path:
    run_timing_seconds = _copy_run_timing_seconds(preflight_payload)
    conversion_start = time.perf_counter()
    if mode != "fixed_equilibrium_polish":
        raise ValueError(
            "--conversion-only is a Lane A fixed_equilibrium_polish smoke path; "
            f"got mode {mode!r}."
        )
    conversion_artifacts = materialize_conversion_only_artifacts(
        source_field_path=selected_seed.field_path,
        source_artifacts=_selected_seed_source_artifacts(selected_seed),
        coil_group_counts={
            coil_group.name: coil_group.count
            for coil_group in selected_seed.coil_groups
        },
        output_root=output_root,
        desc_fourier_order=desc_fourier_order,
        sample_count=sample_count,
        simsopt_fourier_order=simsopt_fourier_order,
        source_nfp=selected_seed.source_nfp,
        source_stellarator_symmetry=selected_seed.source_stellarator_symmetry,
    )
    run_timing_seconds["conversion"] = _elapsed(conversion_start)
    result_payload = _conversion_only_result_payload(
        preflight_payload=preflight_payload,
        conversion_artifacts=conversion_artifacts,
        run_timing_seconds=run_timing_seconds,
    )
    result_path = output_root / "desc_result.json"
    _write_result_validation_outputs(
        result_path=result_path,
        result_payload=result_payload,
        exported_artifact_paths=(
            os.fspath(conversion_artifacts.exported_biot_savart_path),
        ),
        run_timing_seconds=run_timing_seconds,
    )
    return result_path


def _run_equilibrium_load_only_lane(
    *,
    mode: DescJointRunMode,
    output_root: Path,
    equilibrium_seed: DescEquilibriumSeedSpec,
    desc_source_root: Path | None,
) -> tuple[Path, int]:
    try:
        loaded_equilibrium = load_desc_equilibrium_seed_runtime(
            equilibrium_seed,
            desc_source_root=desc_source_root,
        )
        report = loaded_equilibrium.report.to_json_dict()
        report["mode_profile_adjustment"] = (
            _prepare_loaded_equilibrium_profiles_for_mode(
                mode,
                loaded_equilibrium.equilibrium,
            )
        )
        exit_code = 0
    except DescEquilibriumRuntimeLoadError as exc:
        report = exc.report.to_json_dict()
        exit_code = 1
    report_path = output_root / "desc_equilibrium_load_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report_path, exit_code


@dataclass(frozen=True, slots=True)
class _DescRuntimeObjectiveContext:
    equilibrium: object
    coilset: object
    objective_function: object
    constraints: tuple[object, ...]
    desc_export_coil_group_counts: dict[str, int]


def _prepare_desc_runtime_objective_context(
    *,
    mode: DescJointRunMode,
    output_root: Path,
    hardware_keepout_enabled: bool,
    hardware_sdf_manifest_path: Path | None,
    hardware_glb_path: Path | None,
    selected_seed: DescJointSeedCandidate,
    equilibrium_seed: DescEquilibriumSeedSpec,
    desc_source_root: Path | None,
    desc_coilset_checkpoint: Path | None,
    desc_fourier_order: int,
    sample_count: int,
    grid_n: int,
    bs_chunk_size: int,
    dist_chunk_size: int,
    jac_chunk_size: int,
    objective_use_jit: bool,
    objective_deriv_mode: DescObjectiveDerivMode,
    joint_constraint_policy: DescJointConstraintPolicy,
    boundary_fidelity_policy: DescBoundaryFidelityPolicy,
    boundary_fidelity_free_mode_sum: int,
    objective_ablation_policy: DescObjectiveAblationPolicy,
    optimized_coil_groups: tuple[str, ...],
    run_timing_seconds: dict[str, float | None],
) -> tuple[_DescRuntimeObjectiveContext | None, Path, int]:
    try:
        equilibrium_load_start = time.perf_counter()
        loaded_equilibrium = load_desc_equilibrium_seed_runtime(
            equilibrium_seed,
            desc_source_root=desc_source_root,
        )
        run_timing_seconds["equilibrium_load"] = _elapsed(equilibrium_load_start)
    except DescEquilibriumRuntimeLoadError as exc:
        run_timing_seconds["equilibrium_load"] = _elapsed(equilibrium_load_start)
        load_report_path = output_root / "desc_equilibrium_load_report.json"
        _write_json(load_report_path, exc.report.to_json_dict())
        return None, load_report_path, 1
    mode_profile_adjustment = _prepare_loaded_equilibrium_profiles_for_mode(
        mode,
        loaded_equilibrium.equilibrium,
    )
    load_report = loaded_equilibrium.report.to_json_dict()
    load_report["mode_profile_adjustment"] = mode_profile_adjustment
    load_report_path = output_root / "desc_equilibrium_load_report.json"
    _write_json(load_report_path, load_report)

    try:
        coilset_build_start = time.perf_counter()
        coil_group_counts = _selected_seed_coil_group_counts(selected_seed)
        if desc_coilset_checkpoint is None:
            runtime_coilset = build_desc_runtime_coilset_from_simsopt_field(
                source_field_path=selected_seed.field_path,
                source_artifacts=_selected_seed_source_artifacts(selected_seed),
                coil_group_counts=coil_group_counts,
                desc_fourier_order=desc_fourier_order,
                sample_count=sample_count,
                source_nfp=selected_seed.source_nfp,
                source_stellarator_symmetry=selected_seed.source_stellarator_symmetry,
                desc_source_root=desc_source_root,
                field_sample_chunk_size=bs_chunk_size,
            )
        else:
            runtime_coilset = load_desc_runtime_coilset_checkpoint(
                checkpoint_path=desc_coilset_checkpoint,
                coil_group_counts=coil_group_counts,
                source_nfp=selected_seed.source_nfp,
                source_stellarator_symmetry=selected_seed.source_stellarator_symmetry,
                desc_source_root=desc_source_root,
                desc_fourier_order=desc_fourier_order,
                sample_count=sample_count,
                field_sample_chunk_size=bs_chunk_size,
            )
        run_timing_seconds["coilset_build"] = _elapsed(coilset_build_start)
    except DescRuntimeCoilsetBuildError as exc:
        run_timing_seconds["coilset_build"] = _elapsed(coilset_build_start)
        coilset_report_path = output_root / "desc_runtime_coilset_build_report.json"
        _write_json(coilset_report_path, exc.report.to_json_dict())
        return None, coilset_report_path, 1
    coilset_report_path = output_root / "desc_runtime_coilset_build_report.json"
    _write_json(coilset_report_path, runtime_coilset.report.to_json_dict())
    desc_export_coil_group_counts = _runtime_desc_export_coil_group_counts(
        runtime_coilset.report,
        fallback_group_counts=coil_group_counts,
    )

    try:
        scoped_coilset = scope_desc_coilset_optimization_to_groups(
            coilset=runtime_coilset.coilset,
            coil_group_counts=desc_export_coil_group_counts,
            optimized_group_names=optimized_coil_groups,
            desc_source_root=desc_source_root,
        )
    except Exception as exc:
        scope_report = {
            "schema_version": "desc_coilset_optimization_scope_report_v1",
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "optimized_group_names": list(optimized_coil_groups),
        }
        scope_report_path = output_root / "desc_runtime_optimizer_scope_report.json"
        _write_json(scope_report_path, scope_report)
        return None, scope_report_path, 1
    scope_report_path = output_root / "desc_runtime_optimizer_scope_report.json"
    _write_json(
        scope_report_path,
        {
            "schema_version": "desc_coilset_optimization_scope_report_v1",
            "status": "passed",
            "reason": (
                "DESC runtime CoilSet scoped to configured optimizer coil groups."
            ),
            "scope": scoped_coilset.scope.to_json_dict(),
        },
    )

    try:
        objective_assembly_start = time.perf_counter()
        volume_target_m3 = _joint_seed_volume_target_m3(
            mode=mode,
            selected_seed=selected_seed,
            equilibrium=loaded_equilibrium.equilibrium,
        )
        objective_assembly = assemble_desc_objective_stack_runtime(
            mode=mode,
            equilibrium=loaded_equilibrium.equilibrium,
            coilset=scoped_coilset.coilset,
            include_hardware_keepout=hardware_keepout_enabled,
            hardware_sdf_manifest_path=hardware_sdf_manifest_path,
            hardware_glb_path=hardware_glb_path,
            desc_source_root=desc_source_root,
            grid_n=grid_n,
            bs_chunk_size=bs_chunk_size,
            dist_chunk_size=dist_chunk_size,
            jac_chunk_size=jac_chunk_size,
            objective_use_jit=objective_use_jit,
            objective_deriv_mode=objective_deriv_mode,
            joint_constraint_policy=joint_constraint_policy,
            boundary_fidelity_policy=boundary_fidelity_policy,
            boundary_fidelity_free_mode_sum=boundary_fidelity_free_mode_sum,
            objective_ablation_policy=objective_ablation_policy,
            volume_target_m3=volume_target_m3,
        )
        run_timing_seconds["objective_assembly"] = _elapsed(
            objective_assembly_start
        )
    except DescObjectiveRuntimeAssemblyError as exc:
        run_timing_seconds["objective_assembly"] = _elapsed(
            objective_assembly_start
        )
        objective_report_path = output_root / "desc_objective_assembly_report.json"
        _write_json(objective_report_path, exc.report.to_json_dict())
        return None, objective_report_path, 1
    objective_report_path = output_root / "desc_objective_assembly_report.json"
    _write_json(objective_report_path, objective_assembly.report.to_json_dict())
    return (
        _DescRuntimeObjectiveContext(
            equilibrium=loaded_equilibrium.equilibrium,
            coilset=scoped_coilset.coilset,
            objective_function=objective_assembly.objective_function,
            constraints=objective_assembly.constraints,
            desc_export_coil_group_counts=desc_export_coil_group_counts,
        ),
        objective_report_path,
        0,
    )


def _prepare_loaded_equilibrium_profiles_for_mode(
    mode: DescJointRunMode,
    equilibrium: object,
) -> dict[str, object]:
    before = _equilibrium_profile_max_abs(equilibrium)
    if mode != "vacuum_joint":
        after = _equilibrium_profile_max_abs(equilibrium)
        return {
            "mode": mode,
            "action": "preserved",
            "reason": "mode does not use DESC VacuumBoundaryError.",
            "before_max_abs": before,
            "after_max_abs": after,
        }

    equilibrium.pressure = 0.0
    equilibrium.current = 0.0
    after = _equilibrium_profile_max_abs(equilibrium)
    return {
        "mode": mode,
        "action": "zeroed_pressure_current",
        "reason": (
            "vacuum_joint uses DESC VacuumBoundaryError, so loaded DESC "
            "equilibrium pressure and toroidal-current profiles are set to zero "
            "before objective assembly."
        ),
        "before_max_abs": before,
        "after_max_abs": after,
    }


def _equilibrium_profile_max_abs(equilibrium: object) -> dict[str, float]:
    return {
        "pressure": _profile_max_abs(getattr(equilibrium, "pressure", None)),
        "current": _profile_max_abs(getattr(equilibrium, "current", None)),
    }


def _profile_max_abs(profile: object | None) -> float:
    if profile is None:
        return 0.0
    if callable(profile):
        values = profile(np.linspace(0.0, 1.0, 9))
    else:
        values = profile
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def _run_objective_assembly_only_lane(
    *,
    mode: DescJointRunMode,
    output_root: Path,
    hardware_keepout_enabled: bool,
    hardware_sdf_manifest_path: Path | None,
    hardware_glb_path: Path | None,
    selected_seed: DescJointSeedCandidate,
    equilibrium_seed: DescEquilibriumSeedSpec,
    desc_source_root: Path | None,
    desc_coilset_checkpoint: Path | None,
    desc_fourier_order: int,
    sample_count: int,
    grid_n: int,
    bs_chunk_size: int,
    dist_chunk_size: int,
    jac_chunk_size: int,
    objective_use_jit: bool,
    objective_deriv_mode: DescObjectiveDerivMode,
    joint_constraint_policy: DescJointConstraintPolicy,
    boundary_fidelity_policy: DescBoundaryFidelityPolicy,
    boundary_fidelity_free_mode_sum: int,
    objective_ablation_policy: DescObjectiveAblationPolicy,
    optimized_coil_groups: tuple[str, ...],
    evaluate_objective: bool,
    evaluate_jacobian: bool,
    evaluate_gradient: bool,
) -> tuple[Path, int]:
    context, report_path, exit_code = _prepare_desc_runtime_objective_context(
        mode=mode,
        output_root=output_root,
        hardware_keepout_enabled=hardware_keepout_enabled,
        hardware_sdf_manifest_path=hardware_sdf_manifest_path,
        hardware_glb_path=hardware_glb_path,
        selected_seed=selected_seed,
        equilibrium_seed=equilibrium_seed,
        desc_source_root=desc_source_root,
        desc_coilset_checkpoint=desc_coilset_checkpoint,
        desc_fourier_order=desc_fourier_order,
        sample_count=sample_count,
        grid_n=grid_n,
        bs_chunk_size=bs_chunk_size,
        dist_chunk_size=dist_chunk_size,
        jac_chunk_size=jac_chunk_size,
        objective_use_jit=objective_use_jit,
        objective_deriv_mode=objective_deriv_mode,
        joint_constraint_policy=joint_constraint_policy,
        boundary_fidelity_policy=boundary_fidelity_policy,
        boundary_fidelity_free_mode_sum=boundary_fidelity_free_mode_sum,
        objective_ablation_policy=objective_ablation_policy,
        optimized_coil_groups=optimized_coil_groups,
        run_timing_seconds=_empty_run_timing_seconds(),
    )
    if context is None:
        return report_path, exit_code
    if not evaluate_objective:
        return report_path, 0
    try:
        evaluation_report = evaluate_desc_objective_stack_runtime(
            context.objective_function,
            use_jit=False,
            compute_jacobian=evaluate_jacobian,
            compute_gradient=evaluate_gradient,
            gradient_progress_path=(
                output_root / "desc_objective_gradient_progress.jsonl"
                if evaluate_gradient
                else None
            ),
        )
    except DescObjectiveRuntimeEvaluationError as exc:
        evaluation_report_path = output_root / "desc_objective_evaluation_report.json"
        _write_json(evaluation_report_path, exc.report.to_json_dict())
        return evaluation_report_path, 1
    evaluation_report_path = output_root / "desc_objective_evaluation_report.json"
    _write_json(evaluation_report_path, evaluation_report.to_json_dict())
    return evaluation_report_path, 0


def _run_fixed_polish_only_lane(
    *,
    mode: DescJointRunMode,
    output_root: Path,
    hardware_keepout_enabled: bool,
    hardware_sdf_manifest_path: Path | None,
    hardware_glb_path: Path | None,
    selected_seed: DescJointSeedCandidate,
    equilibrium_seed: DescEquilibriumSeedSpec,
    desc_source_root: Path | None,
    desc_coilset_checkpoint: Path | None,
    preflight_payload: dict[str, object],
    desc_fourier_order: int,
    sample_count: int,
    grid_n: int,
    bs_chunk_size: int,
    dist_chunk_size: int,
    jac_chunk_size: int,
    objective_use_jit: bool,
    objective_deriv_mode: DescObjectiveDerivMode,
    boundary_fidelity_policy: DescBoundaryFidelityPolicy,
    boundary_fidelity_free_mode_sum: int,
    optimizer_method: str,
    maxiter: int | None,
    optimizer_verbose: int,
    optimizer_controls: DescOptimizerControls,
    allow_high_memory_optimizer: bool,
    simsopt_fourier_order: int,
    optimized_coil_groups: tuple[str, ...],
) -> tuple[Path, int]:
    if mode != "fixed_equilibrium_polish":
        raise ValueError(
            "--fixed-polish-only is a Lane A fixed_equilibrium_polish path; "
            f"got mode {mode!r}."
        )
    run_timing_seconds = _copy_run_timing_seconds(preflight_payload)
    if not allow_high_memory_optimizer:
        solve_report = desc_fixed_equilibrium_polish_setup_failure_report(
            reason=desc_high_memory_optimizer_blocked_reason(
                "fixed-equilibrium polish"
            ),
            desc_source_root=desc_source_root,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=optimizer_verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
        )
        result_path = _materialize_fixed_polish_result_contract(
            output_root=output_root,
            preflight_payload=preflight_payload,
            solve_report=solve_report,
            setup_failure_report_path=None,
            optimized_export_artifacts=None,
            optimized_export_failure_report_path=None,
            run_timing_seconds=run_timing_seconds,
        )
        return result_path, 1
    context, report_path, exit_code = _prepare_desc_runtime_objective_context(
        mode=mode,
        output_root=output_root,
        hardware_keepout_enabled=hardware_keepout_enabled,
        hardware_sdf_manifest_path=hardware_sdf_manifest_path,
        hardware_glb_path=hardware_glb_path,
        selected_seed=selected_seed,
        equilibrium_seed=equilibrium_seed,
        desc_source_root=desc_source_root,
        desc_coilset_checkpoint=desc_coilset_checkpoint,
        desc_fourier_order=desc_fourier_order,
        sample_count=sample_count,
        grid_n=grid_n,
        bs_chunk_size=bs_chunk_size,
        dist_chunk_size=dist_chunk_size,
        jac_chunk_size=jac_chunk_size,
        objective_use_jit=objective_use_jit,
        objective_deriv_mode=objective_deriv_mode,
        joint_constraint_policy=HARD_VOLUME_AND_FORCE_BALANCE_POLICY,
        boundary_fidelity_policy=boundary_fidelity_policy,
        boundary_fidelity_free_mode_sum=boundary_fidelity_free_mode_sum,
        objective_ablation_policy=FULL_DESC_OBJECTIVE_ABLATION_POLICY,
        optimized_coil_groups=optimized_coil_groups,
        run_timing_seconds=run_timing_seconds,
    )
    if context is None:
        setup_failure_report = desc_fixed_equilibrium_polish_setup_failure_report(
            reason=(
                "DESC fixed-equilibrium polish setup failed before optimizer "
                f"execution; see {os.fspath(report_path)}."
            ),
            desc_source_root=desc_source_root,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=optimizer_verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
        )
        result_path = _materialize_fixed_polish_result_contract(
            output_root=output_root,
            preflight_payload=preflight_payload,
            solve_report=setup_failure_report,
            setup_failure_report_path=report_path,
            optimized_export_artifacts=None,
            optimized_export_failure_report_path=None,
            run_timing_seconds=run_timing_seconds,
        )
        return result_path, exit_code
    optimized_export_artifacts: DescOptimizedSimsoptExportArtifacts | None = None
    optimized_export_failure_report_path: Path | None = None
    pre_optimizer_report = desc_fixed_equilibrium_polish_setup_failure_report(
        reason=(
            "DESC fixed-equilibrium polish optimizer execution has started but "
            "has not returned yet; this fail-closed result is written before "
            "optimizer execution and will be overwritten if the optimizer "
            "returns."
        ),
        desc_source_root=desc_source_root,
        optimizer_method=optimizer_method,
        maxiter=maxiter,
        verbose=optimizer_verbose,
        optimizer_controls=optimizer_controls,
        allow_high_memory_optimizer=allow_high_memory_optimizer,
    )
    _materialize_fixed_polish_result_contract(
        output_root=output_root,
        preflight_payload=preflight_payload,
        solve_report=pre_optimizer_report,
        setup_failure_report_path=None,
        optimized_export_artifacts=None,
        optimized_export_failure_report_path=None,
        run_timing_seconds=run_timing_seconds,
    )
    try:
        optimizer_start = time.perf_counter()
        solve_result = run_desc_fixed_equilibrium_polish_runtime(
            coilset=context.coilset,
            objective_function=context.objective_function,
            constraints=context.constraints,
            output_root=output_root,
            desc_source_root=desc_source_root,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=optimizer_verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
        )
        run_timing_seconds["optimizer"] = _elapsed(optimizer_start)
        solve_report = solve_result.report
        exit_code = 0
        try:
            export_start = time.perf_counter()
            optimized_export_artifacts = (
                materialize_optimized_desc_coil_artifact_simsopt_export(
                    optimized_coilset_path=solve_report.optimized_coilset_path,
                    source_artifacts=_selected_seed_source_artifacts(selected_seed),
                    coil_group_counts=context.desc_export_coil_group_counts,
                    output_root=output_root,
                    sample_count=sample_count,
                    simsopt_fourier_order=simsopt_fourier_order,
                    desc_source_root=desc_source_root,
                )
            )
            run_timing_seconds["optimized_simsopt_export"] = _elapsed(export_start)
        except DescOptimizedSimsoptExportError:
            run_timing_seconds["optimized_simsopt_export"] = _elapsed(export_start)
            optimized_export_failure_report_path = (
                output_root / "desc_optimized_simsopt_export_report.json"
            )
            exit_code = 1
    except DescFixedPolishRuntimeSolveError as exc:
        run_timing_seconds["optimizer"] = _elapsed(optimizer_start)
        solve_report = exc.report
        exit_code = 1
    result_path = _materialize_fixed_polish_result_contract(
        output_root=output_root,
        preflight_payload=preflight_payload,
        solve_report=solve_report,
        setup_failure_report_path=None,
        optimized_export_artifacts=optimized_export_artifacts,
        optimized_export_failure_report_path=optimized_export_failure_report_path,
        run_timing_seconds=run_timing_seconds,
    )
    return result_path, exit_code


def _run_joint_run_only_lane(
    *,
    mode: DescJointRunMode,
    output_root: Path,
    hardware_keepout_enabled: bool,
    hardware_sdf_manifest_path: Path | None,
    hardware_glb_path: Path | None,
    selected_seed: DescJointSeedCandidate,
    equilibrium_seed: DescEquilibriumSeedSpec,
    desc_source_root: Path | None,
    desc_coilset_checkpoint: Path | None,
    preflight_payload: dict[str, object],
    desc_fourier_order: int,
    sample_count: int,
    grid_n: int,
    bs_chunk_size: int,
    dist_chunk_size: int,
    jac_chunk_size: int,
    objective_use_jit: bool,
    objective_deriv_mode: DescObjectiveDerivMode,
    joint_constraint_policy: DescJointConstraintPolicy,
    boundary_fidelity_policy: DescBoundaryFidelityPolicy,
    boundary_fidelity_free_mode_sum: int,
    objective_ablation_policy: DescObjectiveAblationPolicy,
    optimizer_method: str,
    maxiter: int | None,
    optimizer_verbose: int,
    optimizer_controls: DescOptimizerControls,
    allow_high_memory_optimizer: bool,
    simsopt_fourier_order: int,
    optimized_coil_groups: tuple[str, ...],
) -> tuple[Path, int]:
    if mode == "fixed_equilibrium_polish":
        raise ValueError(
            "--joint-run-only requires vacuum_joint or finite_beta_joint; "
            f"got mode {mode!r}."
        )
    run_timing_seconds = _copy_run_timing_seconds(preflight_payload)
    if not allow_high_memory_optimizer:
        solve_report = desc_joint_optimization_setup_failure_report(
            reason=desc_high_memory_optimizer_blocked_reason(
                "joint equilibrium+coil"
            ),
            desc_source_root=desc_source_root,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=optimizer_verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
        )
        result_path = _materialize_joint_result_contract(
            output_root=output_root,
            preflight_payload=preflight_payload,
            solve_report=solve_report,
            setup_failure_report_path=None,
            optimized_export_artifacts=None,
            optimized_export_failure_report_path=None,
            optimized_surface_export_artifacts=None,
            optimized_surface_export_failure_report_path=None,
            run_timing_seconds=run_timing_seconds,
        )
        return result_path, 1
    context, report_path, exit_code = _prepare_desc_runtime_objective_context(
        mode=mode,
        output_root=output_root,
        hardware_keepout_enabled=hardware_keepout_enabled,
        hardware_sdf_manifest_path=hardware_sdf_manifest_path,
        hardware_glb_path=hardware_glb_path,
        selected_seed=selected_seed,
        equilibrium_seed=equilibrium_seed,
        desc_source_root=desc_source_root,
        desc_coilset_checkpoint=desc_coilset_checkpoint,
        desc_fourier_order=desc_fourier_order,
        sample_count=sample_count,
        grid_n=grid_n,
        bs_chunk_size=bs_chunk_size,
        dist_chunk_size=dist_chunk_size,
        jac_chunk_size=jac_chunk_size,
        objective_use_jit=objective_use_jit,
        objective_deriv_mode=objective_deriv_mode,
        joint_constraint_policy=joint_constraint_policy,
        boundary_fidelity_policy=boundary_fidelity_policy,
        boundary_fidelity_free_mode_sum=boundary_fidelity_free_mode_sum,
        objective_ablation_policy=objective_ablation_policy,
        optimized_coil_groups=optimized_coil_groups,
        run_timing_seconds=run_timing_seconds,
    )
    if context is None:
        setup_failure_report = desc_joint_optimization_setup_failure_report(
            reason=(
                "DESC joint optimization setup failed before optimizer execution; "
                f"see {os.fspath(report_path)}."
            ),
            desc_source_root=desc_source_root,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=optimizer_verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
        )
        result_path = _materialize_joint_result_contract(
            output_root=output_root,
            preflight_payload=preflight_payload,
            solve_report=setup_failure_report,
            setup_failure_report_path=report_path,
            optimized_export_artifacts=None,
            optimized_export_failure_report_path=None,
            optimized_surface_export_artifacts=None,
            optimized_surface_export_failure_report_path=None,
            run_timing_seconds=run_timing_seconds,
        )
        return result_path, exit_code
    optimized_export_artifacts: DescOptimizedSimsoptExportArtifacts | None = None
    optimized_export_failure_report_path: Path | None = None
    optimized_surface_export_artifacts: DescOptimizedSurfaceExportArtifacts | None = None
    optimized_surface_export_failure_report_path: Path | None = None
    pre_optimizer_report = desc_joint_optimization_setup_failure_report(
        reason=(
            "DESC joint optimizer execution has started but has not returned "
            "yet; this fail-closed result is written before optimizer execution "
            "and will be overwritten if the optimizer returns."
        ),
        desc_source_root=desc_source_root,
        optimizer_method=optimizer_method,
        maxiter=maxiter,
        verbose=optimizer_verbose,
        optimizer_controls=optimizer_controls,
        allow_high_memory_optimizer=allow_high_memory_optimizer,
    )
    _materialize_joint_result_contract(
        output_root=output_root,
        preflight_payload=preflight_payload,
        solve_report=pre_optimizer_report,
        setup_failure_report_path=None,
        optimized_export_artifacts=None,
        optimized_export_failure_report_path=None,
        optimized_surface_export_artifacts=None,
        optimized_surface_export_failure_report_path=None,
        run_timing_seconds=run_timing_seconds,
    )
    try:
        optimizer_start = time.perf_counter()
        solve_result = run_desc_joint_optimization_runtime(
            equilibrium=context.equilibrium,
            coilset=context.coilset,
            objective_function=context.objective_function,
            constraints=context.constraints,
            output_root=output_root,
            desc_source_root=desc_source_root,
            optimizer_method=optimizer_method,
            maxiter=maxiter,
            verbose=optimizer_verbose,
            optimizer_controls=optimizer_controls,
            allow_high_memory_optimizer=allow_high_memory_optimizer,
        )
        run_timing_seconds["optimizer"] = _elapsed(optimizer_start)
        solve_report = solve_result.report
        exit_code = 0
        try:
            export_start = time.perf_counter()
            optimized_export_artifacts = (
                materialize_optimized_desc_coil_artifact_simsopt_export(
                    optimized_coilset_path=solve_report.optimized_coilset_path,
                    source_artifacts=_selected_seed_source_artifacts(selected_seed),
                    coil_group_counts=context.desc_export_coil_group_counts,
                    output_root=output_root,
                    sample_count=sample_count,
                    simsopt_fourier_order=simsopt_fourier_order,
                    desc_source_root=desc_source_root,
                )
            )
            run_timing_seconds["optimized_simsopt_export"] = _elapsed(export_start)
        except DescOptimizedSimsoptExportError:
            run_timing_seconds["optimized_simsopt_export"] = _elapsed(export_start)
            optimized_export_failure_report_path = (
                output_root / "desc_optimized_simsopt_export_report.json"
            )
            exit_code = 1
        try:
            surface_export_start = time.perf_counter()
            optimized_surface_export_artifacts = (
                materialize_optimized_desc_equilibrium_surface_simsopt_export(
                    optimized_equilibrium_path=solve_report.optimized_equilibrium_path,
                    output_root=output_root,
                    desc_source_root=desc_source_root,
                    nfp=equilibrium_seed.nfp,
                    stellarator_symmetry=equilibrium_seed.stellarator_symmetry,
                    mpol=equilibrium_seed.lcfs_mpol,
                    ntor=equilibrium_seed.lcfs_ntor,
                )
            )
            run_timing_seconds["optimized_simsopt_surface_export"] = _elapsed(
                surface_export_start
            )
        except DescOptimizedSurfaceExportError:
            run_timing_seconds["optimized_simsopt_surface_export"] = _elapsed(
                surface_export_start
            )
            optimized_surface_export_failure_report_path = (
                output_root / "desc_optimized_surface_export_report.json"
            )
            exit_code = 1
    except DescJointRuntimeSolveError as exc:
        run_timing_seconds["optimizer"] = _elapsed(optimizer_start)
        solve_report = exc.report
        exit_code = 1
    result_path = _materialize_joint_result_contract(
        output_root=output_root,
        preflight_payload=preflight_payload,
        solve_report=solve_report,
        setup_failure_report_path=None,
        optimized_export_artifacts=optimized_export_artifacts,
        optimized_export_failure_report_path=optimized_export_failure_report_path,
        optimized_surface_export_artifacts=optimized_surface_export_artifacts,
        optimized_surface_export_failure_report_path=(
            optimized_surface_export_failure_report_path
        ),
        run_timing_seconds=run_timing_seconds,
    )
    return result_path, exit_code


def _materialize_fixed_polish_result_contract(
    *,
    output_root: Path,
    preflight_payload: dict[str, object],
    solve_report: DescFixedPolishRuntimeSolveReport,
    setup_failure_report_path: Path | None,
    optimized_export_artifacts: DescOptimizedSimsoptExportArtifacts | None,
    optimized_export_failure_report_path: Path | None,
    run_timing_seconds: dict[str, float | None],
) -> Path:
    solve_report_path = output_root / "desc_fixed_polish_solve_report.json"
    _write_json(solve_report_path, solve_report.to_json_dict())
    result_payload = _fixed_polish_result_payload(
        preflight_payload=preflight_payload,
        solve_report=solve_report,
        solve_report_path=solve_report_path,
        setup_failure_report_path=setup_failure_report_path,
        optimized_export_artifacts=optimized_export_artifacts,
        optimized_export_failure_report_path=optimized_export_failure_report_path,
        run_timing_seconds=run_timing_seconds,
    )
    result_path = output_root / "desc_result.json"
    exported_artifact_paths = _fixed_polish_exported_artifact_paths(
        optimized_export_artifacts,
    )
    _write_result_validation_outputs(
        result_path=result_path,
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        run_timing_seconds=run_timing_seconds,
    )
    return result_path


def _materialize_joint_result_contract(
    *,
    output_root: Path,
    preflight_payload: dict[str, object],
    solve_report: DescJointRuntimeSolveReport,
    setup_failure_report_path: Path | None,
    optimized_export_artifacts: DescOptimizedSimsoptExportArtifacts | None,
    optimized_export_failure_report_path: Path | None,
    optimized_surface_export_artifacts: DescOptimizedSurfaceExportArtifacts | None,
    optimized_surface_export_failure_report_path: Path | None,
    run_timing_seconds: dict[str, float | None],
) -> Path:
    solve_report_path = output_root / "desc_joint_runtime_solve_report.json"
    _write_json(solve_report_path, solve_report.to_json_dict())
    result_payload = _joint_result_payload(
        preflight_payload=preflight_payload,
        solve_report=solve_report,
        solve_report_path=solve_report_path,
        setup_failure_report_path=setup_failure_report_path,
        optimized_export_artifacts=optimized_export_artifacts,
        optimized_export_failure_report_path=optimized_export_failure_report_path,
        optimized_surface_export_artifacts=optimized_surface_export_artifacts,
        optimized_surface_export_failure_report_path=(
            optimized_surface_export_failure_report_path
        ),
        run_timing_seconds=run_timing_seconds,
    )
    result_path = output_root / "desc_result.json"
    exported_artifact_paths = _optimized_exported_artifact_paths(
        optimized_export_artifacts,
    )
    _write_result_validation_outputs(
        result_path=result_path,
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        run_timing_seconds=run_timing_seconds,
    )
    return result_path


def _conversion_only_result_payload(
    *,
    preflight_payload: dict[str, object],
    conversion_artifacts: DescConversionOnlyArtifacts,
    run_timing_seconds: dict[str, float | None],
) -> dict[str, object]:
    payload = dict(preflight_payload)
    payload["desc_solve_status"] = DescJointStatus(
        state="blocked",
        reason=(
            "conversion-only bridge artifacts were written; DESC optimizer has "
            "not run"
        ),
        artifact_paths=(os.fspath(conversion_artifacts.desc_coils_path),),
    ).to_json_dict()
    payload["artifact_hardware_status"] = DescJointStatus(
        state="blocked",
        reason="exported SIMSOPT artifact exists, but hardware oracle has not run",
        artifact_paths=(os.fspath(conversion_artifacts.exported_biot_savart_path),),
    ).to_json_dict()
    payload["physics_validation_status"] = DescJointStatus(
        state="not_run",
        reason="SIMSOPT Boozer/Poincare validation has not run on this export",
    ).to_json_dict()
    payload["promotion_status"] = DescJointStatus(
        state="blocked",
        reason=(
            "promotion requires DESC optimization, SIMSOPT physics validation, "
            "artifact hardware validation, and direct hardware oracle evidence"
        ),
    ).to_json_dict()
    payload.update(_legacy_fail_closed_hardware_result_fields(preflight_payload))
    payload["conversion_artifacts"] = {
        "desc_coils": os.fspath(conversion_artifacts.desc_coils_path),
        "export_report": os.fspath(conversion_artifacts.export_report_path),
        "exported_biot_savart": os.fspath(
            conversion_artifacts.exported_biot_savart_path
        ),
        "import_report": os.fspath(conversion_artifacts.import_report_path),
    }
    payload["run_timing_seconds"] = dict(run_timing_seconds)
    validate_desc_joint_result_payload(payload)
    return payload


def _joint_result_payload(
    *,
    preflight_payload: dict[str, object],
    solve_report: DescJointRuntimeSolveReport,
    solve_report_path: Path,
    setup_failure_report_path: Path | None,
    optimized_export_artifacts: DescOptimizedSimsoptExportArtifacts | None,
    optimized_export_failure_report_path: Path | None,
    optimized_surface_export_artifacts: DescOptimizedSurfaceExportArtifacts | None,
    optimized_surface_export_failure_report_path: Path | None,
    run_timing_seconds: dict[str, float | None],
) -> dict[str, object]:
    payload = dict(preflight_payload)
    optimized_artifacts = _joint_optimized_desc_artifact_paths(solve_report)
    exported_artifact_paths = _optimized_exported_artifact_paths(
        optimized_export_artifacts,
    )
    exported_surface_paths = _optimized_surface_exported_artifact_paths(
        optimized_surface_export_artifacts,
    )
    payload["desc_solve_status"] = DescJointStatus(
        state="passed" if solve_report.status == "passed" else "failed",
        reason=solve_report.reason,
        artifact_paths=optimized_artifacts,
    ).to_json_dict()
    payload["artifact_hardware_status"] = DescJointStatus(
        state="blocked",
        reason=(
            "exported SIMSOPT artifact exists, but hardware oracle has not run"
            if exported_artifact_paths
            else (
                "no exported SIMSOPT artifact from the DESC joint solve exists "
                "for hardware oracle validation"
            )
        ),
        artifact_paths=exported_artifact_paths,
    ).to_json_dict()
    physics_ready = bool(exported_artifact_paths and exported_surface_paths)
    physics_state = "not_run" if physics_ready else "blocked"
    payload["physics_validation_status"] = DescJointStatus(
        state=physics_state,
        reason=(
            "SIMSOPT Boozer/Poincare validation has not run on this DESC joint export"
            if physics_ready
            else (
                "SIMSOPT field and moved-boundary surface exports from the DESC "
                "joint solve are required before physics validation can run"
            )
        ),
        artifact_paths=exported_artifact_paths + exported_surface_paths,
    ).to_json_dict()
    payload["promotion_status"] = DescJointStatus(
        state="blocked",
        reason=(
            "promotion requires SIMSOPT export, physics validation, artifact "
            "hardware validation, and direct hardware oracle evidence"
        ),
    ).to_json_dict()
    payload.update(_legacy_fail_closed_hardware_result_fields(preflight_payload))
    payload["desc_runtime_artifacts"] = {
        "joint_solve_report": os.fspath(solve_report_path),
        "constraint_feasibility_report": (
            None
            if solve_report.constraint_feasibility_report_path is None
            else os.fspath(solve_report.constraint_feasibility_report_path)
        ),
        "setup_failure_report": (
            None
            if setup_failure_report_path is None
            else os.fspath(setup_failure_report_path)
        ),
        "optimized_simsopt_export_report": _optimized_export_report_path(
            optimized_export_artifacts,
            optimized_export_failure_report_path,
        ),
        "optimized_surface_export_report": _optimized_surface_export_report_path(
            optimized_surface_export_artifacts,
            optimized_surface_export_failure_report_path,
        ),
        "desc_equilibrium": (
            None
            if solve_report.optimized_equilibrium_path is None
            else os.fspath(solve_report.optimized_equilibrium_path)
        ),
        "desc_coils": (
            None
            if solve_report.optimized_coilset_path is None
            else os.fspath(solve_report.optimized_coilset_path)
        ),
        "failed_optimizer_checkpoint_desc_equilibrium": (
            None
            if solve_report.failed_optimizer_equilibrium_checkpoint_path is None
            else os.fspath(solve_report.failed_optimizer_equilibrium_checkpoint_path)
        ),
        "failed_optimizer_checkpoint_desc_coils": (
            None
            if solve_report.failed_optimizer_coilset_checkpoint_path is None
            else os.fspath(solve_report.failed_optimizer_coilset_checkpoint_path)
        ),
        "exported_biot_savart": (
            None
            if optimized_export_artifacts is None
            else os.fspath(optimized_export_artifacts.exported_biot_savart_path)
        ),
        "exported_surface": (
            None
            if optimized_surface_export_artifacts is None
            else os.fspath(optimized_surface_export_artifacts.exported_surface_path)
        ),
        "desc_coil_import_report": (
            None
            if optimized_export_artifacts is None
            else os.fspath(optimized_export_artifacts.import_report_path)
        ),
    }
    payload["desc_optimizer_result"] = {
        "success": solve_report.optimizer_success,
        "status": solve_report.optimizer_status,
        "message": solve_report.optimizer_message,
        "nit": solve_report.optimizer_nit,
        "nfev": solve_report.optimizer_nfev,
        "method": solve_report.optimizer_method,
        "controls": solve_report.optimizer_controls.to_json_dict(),
    }
    payload["run_timing_seconds"] = dict(run_timing_seconds)
    validate_desc_joint_result_payload(payload)
    return payload


def _fixed_polish_result_payload(
    *,
    preflight_payload: dict[str, object],
    solve_report: DescFixedPolishRuntimeSolveReport,
    solve_report_path: Path,
    setup_failure_report_path: Path | None,
    optimized_export_artifacts: DescOptimizedSimsoptExportArtifacts | None,
    optimized_export_failure_report_path: Path | None,
    run_timing_seconds: dict[str, float | None],
) -> dict[str, object]:
    payload = dict(preflight_payload)
    optimized_coilset_path = solve_report.optimized_coilset_path
    optimized_artifacts = (
        ()
        if optimized_coilset_path is None
        else (os.fspath(optimized_coilset_path),)
    )
    exported_artifact_paths = _fixed_polish_exported_artifact_paths(
        optimized_export_artifacts,
    )
    payload["desc_solve_status"] = DescJointStatus(
        state="passed" if solve_report.status == "passed" else "failed",
        reason=solve_report.reason,
        artifact_paths=optimized_artifacts,
    ).to_json_dict()
    payload["artifact_hardware_status"] = DescJointStatus(
        state="blocked",
        reason=(
            "exported SIMSOPT artifact exists, but hardware oracle has not run"
            if exported_artifact_paths
            else (
                "no exported SIMSOPT artifact from the DESC solve exists for "
                "hardware oracle validation"
            )
        ),
        artifact_paths=exported_artifact_paths,
    ).to_json_dict()
    physics_state = "not_run" if exported_artifact_paths else "blocked"
    payload["physics_validation_status"] = DescJointStatus(
        state=physics_state,
        reason=(
            "SIMSOPT Boozer/Poincare validation has not run on this DESC export"
            if exported_artifact_paths
            else (
                "SIMSOPT export from the DESC solve is unavailable; physics "
                "validation cannot run"
            )
        ),
    ).to_json_dict()
    payload["promotion_status"] = DescJointStatus(
        state="blocked",
        reason=(
            "promotion requires SIMSOPT export, physics validation, artifact "
            "hardware validation, and direct hardware oracle evidence"
        ),
    ).to_json_dict()
    payload.update(_legacy_fail_closed_hardware_result_fields(preflight_payload))
    payload["desc_runtime_artifacts"] = {
        "fixed_polish_solve_report": os.fspath(solve_report_path),
        "setup_failure_report": (
            None
            if setup_failure_report_path is None
            else os.fspath(setup_failure_report_path)
        ),
        "optimized_simsopt_export_report": _fixed_polish_export_report_path(
            optimized_export_artifacts,
            optimized_export_failure_report_path,
        ),
        "desc_coils": (
            None if optimized_coilset_path is None else os.fspath(optimized_coilset_path)
        ),
        "failed_optimizer_checkpoint_desc_coils": (
            None
            if solve_report.failed_optimizer_coilset_checkpoint_path is None
            else os.fspath(solve_report.failed_optimizer_coilset_checkpoint_path)
        ),
        "exported_biot_savart": (
            None
            if optimized_export_artifacts is None
            else os.fspath(optimized_export_artifacts.exported_biot_savart_path)
        ),
        "desc_coil_import_report": (
            None
            if optimized_export_artifacts is None
            else os.fspath(optimized_export_artifacts.import_report_path)
        ),
    }
    payload["desc_optimizer_result"] = {
        "success": solve_report.optimizer_success,
        "status": solve_report.optimizer_status,
        "message": solve_report.optimizer_message,
        "nit": solve_report.optimizer_nit,
        "nfev": solve_report.optimizer_nfev,
        "method": solve_report.optimizer_method,
        "controls": solve_report.optimizer_controls.to_json_dict(),
    }
    payload["run_timing_seconds"] = dict(run_timing_seconds)
    validate_desc_joint_result_payload(payload)
    return payload


def _joint_optimized_desc_artifact_paths(
    solve_report: DescJointRuntimeSolveReport,
) -> tuple[str, ...]:
    artifacts: list[str] = []
    if solve_report.optimized_equilibrium_path is not None:
        artifacts.append(os.fspath(solve_report.optimized_equilibrium_path))
    if solve_report.optimized_coilset_path is not None:
        artifacts.append(os.fspath(solve_report.optimized_coilset_path))
    return tuple(artifacts)


def _optimized_exported_artifact_paths(
    optimized_export_artifacts: DescOptimizedSimsoptExportArtifacts | None,
) -> tuple[str, ...]:
    if optimized_export_artifacts is None:
        return ()
    return (os.fspath(optimized_export_artifacts.exported_biot_savart_path),)


def _optimized_export_report_path(
    optimized_export_artifacts: DescOptimizedSimsoptExportArtifacts | None,
    optimized_export_failure_report_path: Path | None,
) -> str | None:
    if optimized_export_artifacts is not None:
        return os.fspath(optimized_export_artifacts.export_report_path)
    if optimized_export_failure_report_path is not None:
        return os.fspath(optimized_export_failure_report_path)
    return None


def _optimized_surface_exported_artifact_paths(
    optimized_surface_export_artifacts: DescOptimizedSurfaceExportArtifacts | None,
) -> tuple[str, ...]:
    if optimized_surface_export_artifacts is None:
        return ()
    return (os.fspath(optimized_surface_export_artifacts.exported_surface_path),)


def _optimized_surface_export_report_path(
    optimized_surface_export_artifacts: DescOptimizedSurfaceExportArtifacts | None,
    optimized_surface_export_failure_report_path: Path | None,
) -> str | None:
    if optimized_surface_export_artifacts is not None:
        return os.fspath(optimized_surface_export_artifacts.export_report_path)
    if optimized_surface_export_failure_report_path is not None:
        return os.fspath(optimized_surface_export_failure_report_path)
    return None


def _fixed_polish_exported_artifact_paths(
    optimized_export_artifacts: DescOptimizedSimsoptExportArtifacts | None,
) -> tuple[str, ...]:
    return _optimized_exported_artifact_paths(optimized_export_artifacts)


def _fixed_polish_export_report_path(
    optimized_export_artifacts: DescOptimizedSimsoptExportArtifacts | None,
    optimized_export_failure_report_path: Path | None,
) -> str | None:
    return _optimized_export_report_path(
        optimized_export_artifacts,
        optimized_export_failure_report_path,
    )


def _legacy_fail_closed_hardware_result_fields(
    preflight_payload: Mapping[str, object],
) -> dict[str, object]:
    constraint_names = _hardware_constraint_names(preflight_payload)
    fields = build_hardware_constraint_artifact_payload_fields(
        None,
        names=constraint_names,
    )
    fields.update(
        build_hardware_constraint_artifact_payload_fields(
            None,
            prefix="BEST_FEASIBLE_",
            names=constraint_names,
        )
    )
    fields["BEST_FEASIBLE_AVAILABLE"] = False
    fields["FINAL_FEASIBILITY_OK"] = False
    return fields


def _hardware_constraint_names(
    preflight_payload: Mapping[str, object],
) -> tuple[str, ...] | None:
    input_contract = preflight_payload.get("input_contract")
    if not isinstance(input_contract, Mapping):
        return None
    hardware_contract = input_contract.get("hardware")
    if not isinstance(hardware_contract, Mapping):
        return None
    raw_constraint_names = hardware_contract.get("constraint_names")
    if isinstance(raw_constraint_names, str) or not isinstance(
        raw_constraint_names,
        Sequence,
    ):
        return None
    return tuple(
        constraint_name
        for constraint_name in raw_constraint_names
        if isinstance(constraint_name, str)
    )


def _run_configuration_payload(
    args: argparse.Namespace,
    *,
    desc_source_root: Path | None,
    optimized_coil_groups: tuple[str, ...],
    optimizer_controls: DescOptimizerControls,
    fixed_polish_predecessor_manifest: Path | None,
    lane_b_predecessor_manifest: Path | None,
) -> dict[str, object]:
    return {
        "schema_version": "desc_joint_run_configuration_v1",
        "resolution_preset": str(args.resolution_preset),
        "lanes": {
            "preflight_only": bool(args.preflight_only),
            "equilibrium_load_only": bool(args.equilibrium_load_only),
            "objective_assembly_only": bool(args.objective_assembly_only),
            "objective_eval_only": bool(args.objective_eval_only),
            "conversion_only": bool(args.conversion_only),
            "fixed_polish_only": bool(args.fixed_polish_only),
            "joint_run_only": bool(args.joint_run_only),
        },
        "desc_runtime": {
            "desc_source_root": (
                None if desc_source_root is None else os.fspath(desc_source_root)
            ),
            "desc_runtime_device": args.desc_runtime_device,
            "bootstrapped_desc_runtime_device": _BOOTSTRAPPED_DESC_RUNTIME_DEVICE,
            "desc_coilset_checkpoint": (
                None
                if args.desc_coilset_checkpoint is None
                else os.fspath(Path(args.desc_coilset_checkpoint).expanduser().resolve())
            ),
            "desc_grid_n": int(args.desc_grid_n),
            "desc_equilibrium_lcfs_mpol": int(args.desc_equilibrium_lcfs_mpol),
            "desc_equilibrium_lcfs_ntor": int(args.desc_equilibrium_lcfs_ntor),
            "desc_bs_chunk_size": int(args.desc_bs_chunk_size),
            "desc_dist_chunk_size": int(args.desc_dist_chunk_size),
            "desc_jac_chunk_size": int(args.desc_jac_chunk_size),
            "desc_objective_use_jit": bool(args.desc_objective_use_jit),
            "desc_objective_deriv_mode": str(args.desc_objective_deriv_mode),
            "desc_joint_constraint_policy": str(args.desc_joint_constraint_policy),
            "desc_boundary_fidelity_policy": str(
                args.desc_boundary_fidelity_policy
            ),
            "desc_boundary_fidelity_free_mode_sum": int(
                args.desc_boundary_fidelity_free_mode_sum
            ),
            "desc_objective_ablation_policy": str(
                args.desc_objective_ablation_policy
            ),
        },
        "conversion": {
            "desc_fourier_order": int(args.desc_fourier_order),
            "conversion_sample_count": int(args.conversion_sample_count),
            "simsopt_fourier_order": int(args.simsopt_fourier_order),
        },
        "optimizer": {
            "method": str(args.desc_optimizer_method),
            "maxiter": args.desc_maxiter,
            "verbose": int(args.desc_optimizer_verbose),
            "controls": optimizer_controls.to_json_dict(),
            "allow_high_memory_optimizer": bool(
                args.allow_high_memory_desc_optimizer
            ),
            "optimized_coil_groups": list(optimized_coil_groups),
        },
        "predecessors": {
            "fixed_polish_predecessor_manifest": (
                None
                if fixed_polish_predecessor_manifest is None
                else os.fspath(fixed_polish_predecessor_manifest)
            ),
            "lane_b_predecessor_manifest": (
                None
                if lane_b_predecessor_manifest is None
                else os.fspath(lane_b_predecessor_manifest)
            ),
        },
        "objective_eval": {
            "jacobian": bool(args.objective_eval_jacobian),
            "gradient": bool(args.objective_eval_gradient),
        },
    }


def _desc_optimizer_controls_from_args(
    args: argparse.Namespace,
) -> DescOptimizerControls:
    return build_desc_optimizer_controls(
        ftol=args.desc_optimizer_ftol,
        xtol=args.desc_optimizer_xtol,
        gtol=args.desc_optimizer_gtol,
        ctol=args.desc_optimizer_ctol,
        max_nfev=args.desc_optimizer_max_nfev,
        max_dx=args.desc_optimizer_max_dx,
        initial_trust_radius=args.desc_optimizer_initial_trust_radius,
        max_trust_radius=args.desc_optimizer_max_trust_radius,
        min_trust_radius=args.desc_optimizer_min_trust_radius,
        proximal_perturb_order=args.desc_proximal_perturb_order,
        proximal_solve_maxiter=args.desc_proximal_solve_maxiter,
        proximal_solve_during_build=args.desc_proximal_solve_during_build,
    )


def _empty_run_timing_seconds() -> dict[str, float | None]:
    return {
        "preflight": None,
        "equilibrium_load": None,
        "coilset_build": None,
        "objective_assembly": None,
        "objective_evaluation_build": None,
        "objective_evaluation_value": None,
        "objective_evaluation_jacobian": None,
        "objective_evaluation_gradient": None,
        "optimizer": None,
        "conversion": None,
        "optimized_simsopt_export": None,
        "optimized_simsopt_surface_export": None,
        "result_materialization": None,
        "validation_manifest": None,
    }


def _copy_run_timing_seconds(
    payload: Mapping[str, object],
) -> dict[str, float | None]:
    timings = _empty_run_timing_seconds()
    raw_timings = payload.get("run_timing_seconds")
    if not isinstance(raw_timings, Mapping):
        return timings
    for name in timings:
        value = raw_timings.get(name)
        if value is None:
            timings[name] = None
        elif isinstance(value, (int, float)):
            timings[name] = float(value)
    return timings


def _elapsed(start: float) -> float:
    return max(time.perf_counter() - start, 0.0)


def _write_result_validation_outputs(
    *,
    result_path: Path,
    result_payload: dict[str, object],
    exported_artifact_paths: tuple[str, ...],
    run_timing_seconds: dict[str, float | None],
) -> None:
    inventory_path = result_path.parent / "desc_joint_run_inventory.json"
    result_payload["run_inventory_path"] = os.fspath(inventory_path)
    result_payload["run_timing_seconds"] = dict(run_timing_seconds)
    result_write_start = time.perf_counter()
    _write_json(result_path, result_payload)
    run_timing_seconds["result_materialization"] = _elapsed(result_write_start)

    validation_start = time.perf_counter()
    validation_manifest = build_desc_joint_validation_manifest(
        result_payload=result_payload,
        exported_artifact_paths=exported_artifact_paths,
        physics_validation_passed=None,
        artifact_hardware_passed=None,
        search_hardware_passed=None,
        final_oracle_passed=False,
        final_oracle_evidence_path=None,
    )
    validation_path = result_path.parent / "desc_joint_validation_manifest.json"
    _write_json(validation_path, validation_manifest)
    report_markdown_path = result_path.parent / "desc_joint_validation_report.md"
    report_markdown_path.write_text(
        render_desc_joint_validation_report(validation_manifest),
        encoding="utf-8",
    )
    run_timing_seconds["validation_manifest"] = _elapsed(validation_start)
    result_payload["run_timing_seconds"] = dict(run_timing_seconds)
    validate_desc_joint_result_payload(result_payload)
    _write_json(result_path, result_payload)
    _write_run_inventory(
        inventory_path=inventory_path,
        result_path=result_path,
        result_payload=result_payload,
        validation_manifest_path=validation_path,
        validation_report_path=report_markdown_path,
    )


def _write_run_inventory(
    *,
    inventory_path: Path,
    result_path: Path,
    result_payload: Mapping[str, object],
    validation_manifest_path: Path,
    validation_report_path: Path,
) -> None:
    inventory = {
        "schema_version": "desc_joint_run_inventory_v1",
        "result_path": os.fspath(result_path),
        "run_mode": result_payload.get("run_mode"),
        "run_configuration": result_payload.get("run_configuration"),
        "run_timing_seconds": result_payload.get("run_timing_seconds"),
        "objective_stack": result_payload.get("objective_stack"),
        "objective_stack_plan": result_payload.get("objective_stack_plan"),
        "status_sections": {
            section: result_payload.get(section)
            for section in (
                "desc_solve_status",
                "search_hardware_status",
                "artifact_hardware_status",
                "physics_validation_status",
                "promotion_status",
            )
        },
        "input_artifacts": _inventory_input_artifacts(result_payload),
        "input_artifact_checksums": _inventory_input_artifact_checksums(
            result_payload
        ),
        "output_artifacts": _inventory_output_artifacts(result_payload),
        "validation_artifacts": {
            "manifest": os.fspath(validation_manifest_path),
            "report": os.fspath(validation_report_path),
        },
    }
    _write_json(inventory_path, inventory)


def _inventory_input_artifacts(
    result_payload: Mapping[str, object],
) -> dict[str, object]:
    input_contract = result_payload.get("input_contract")
    if not isinstance(input_contract, Mapping):
        return {}
    selected_seed = input_contract.get("selected_seed")
    equilibrium_seed = input_contract.get("equilibrium_seed")
    artifacts: dict[str, object] = {}
    if isinstance(selected_seed, Mapping):
        for field_name in (
            "surface",
            "field",
            "source_results",
            "state",
            "poincare_metrics",
            "poincare_png",
        ):
            value = selected_seed.get(field_name)
            if isinstance(value, str) and value != "":
                artifacts[f"selected_seed_{field_name}"] = value
    if isinstance(equilibrium_seed, Mapping):
        source_path = equilibrium_seed.get("source_path")
        if isinstance(source_path, str) and source_path != "":
            artifacts["equilibrium_seed_source"] = source_path
    return artifacts


def _inventory_input_artifact_checksums(
    result_payload: Mapping[str, object],
) -> dict[str, object]:
    input_contract = result_payload.get("input_contract")
    if not isinstance(input_contract, Mapping):
        return {}
    selected_seed = input_contract.get("selected_seed")
    if not isinstance(selected_seed, Mapping):
        return {}
    source_checksums = selected_seed.get("source_checksums")
    if not isinstance(source_checksums, Mapping):
        return {}
    return {
        key: value
        for key, value in source_checksums.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _inventory_output_artifacts(
    result_payload: Mapping[str, object],
) -> dict[str, object]:
    artifacts: dict[str, object] = {}
    _collect_artifact_mapping(
        artifacts,
        result_payload.get("conversion_artifacts"),
        prefix="conversion",
    )
    _collect_artifact_mapping(
        artifacts,
        result_payload.get("desc_runtime_artifacts"),
        prefix="desc_runtime",
    )
    for section in (
        "desc_solve_status",
        "artifact_hardware_status",
        "physics_validation_status",
        "promotion_status",
    ):
        section_payload = result_payload.get(section)
        if isinstance(section_payload, Mapping):
            raw_paths = section_payload.get("artifact_paths")
            if not isinstance(raw_paths, str) and isinstance(raw_paths, Sequence):
                artifacts[f"{section}_artifact_paths"] = [
                    path for path in raw_paths if isinstance(path, str)
                ]
    return artifacts


def _collect_artifact_mapping(
    artifacts: dict[str, object],
    raw_mapping: object,
    *,
    prefix: str,
) -> None:
    if not isinstance(raw_mapping, Mapping):
        return
    for key, value in raw_mapping.items():
        if isinstance(key, str) and isinstance(value, str) and value != "":
            artifacts[f"{prefix}_{key}"] = value


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _attach_selected_seed_target_lcfs_G(
    selected_seed: DescJointSeedCandidate,
    equilibrium_seed: DescEquilibriumSeedSpec,
) -> DescEquilibriumSeedSpec:
    if equilibrium_seed.source_kind != "simsopt_surface":
        return equilibrium_seed
    if selected_seed.state_path is None:
        return equilibrium_seed
    state_G = _selected_seed_state_G(selected_seed.state_path)
    # SIMSOPT sidecars store G = mu0 * I; DESC LCC computes I = 2*pi*G/mu0.
    target_lcfs_G = state_G / (2.0 * math.pi)
    if equilibrium_seed.target_lcfs_G is not None:
        if not math.isclose(
            equilibrium_seed.target_lcfs_G,
            target_lcfs_G,
            rel_tol=1.0e-10,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "Selected seed state G does not match equilibrium seed "
                "target_lcfs_G: "
                f"{target_lcfs_G!r} vs {equilibrium_seed.target_lcfs_G!r}."
            )
        return equilibrium_seed
    return equilibrium_seed.with_target_lcfs_G(target_lcfs_G)


def _selected_seed_state_G(state_path: Path) -> float:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"Selected seed state must be a JSON object: {state_path}.")
    raw_G = payload.get("G")
    if isinstance(raw_G, bool) or not isinstance(raw_G, (int, float)):
        raise ValueError(
            f"Selected seed state must contain finite numeric G: {state_path}."
        )
    state_G = float(raw_G)
    if not math.isfinite(state_G):
        raise ValueError(
            f"Selected seed state must contain finite numeric G: {state_path}."
        )
    return state_G


def _selected_seed_source_artifacts(
    selected_seed: DescJointSeedCandidate,
) -> dict[str, Path]:
    artifacts = {
        "seed_surface": selected_seed.surface_path,
        "seed_field": selected_seed.field_path,
    }
    if selected_seed.source_results_path is not None:
        artifacts["seed_source_results"] = selected_seed.source_results_path
    if selected_seed.state_path is not None:
        artifacts["seed_state"] = selected_seed.state_path
    if selected_seed.poincare_metrics_path is not None:
        artifacts["seed_poincare_metrics"] = selected_seed.poincare_metrics_path
    if selected_seed.poincare_png_path is not None:
        artifacts["seed_poincare_png"] = selected_seed.poincare_png_path
    return artifacts


def _selected_seed_coil_group_counts(
    selected_seed: DescJointSeedCandidate,
) -> dict[str, int]:
    return {
        coil_group.name: coil_group.count
        for coil_group in selected_seed.coil_groups
    }


def _runtime_desc_export_coil_group_counts(
    coilset_report: object,
    *,
    fallback_group_counts: Mapping[str, int],
) -> dict[str, int]:
    export_report = getattr(coilset_report, "export_report", None)
    if export_report is None:
        return dict(fallback_group_counts)
    return {
        group: int(export_report.group_counts[group])
        for group in export_report.group_order
    }


def _joint_seed_volume_target_m3(
    *,
    mode: DescJointRunMode,
    selected_seed: DescJointSeedCandidate,
    equilibrium: object,
) -> float | None:
    if mode == "fixed_equilibrium_polish":
        return None
    loaded_artifact = load_simsopt(os.fspath(selected_seed.surface_path))
    surface = getattr(loaded_artifact, "surface", loaded_artifact)
    volume = getattr(surface, "volume", None)
    if not callable(volume):
        raise ValueError(
            "DESC joint Volume objective requires a SIMSOPT seed surface with "
            f"a volume() method: {selected_seed.surface_path}."
        )
    volume_m3 = abs(float(volume()))
    if not math.isfinite(volume_m3) or volume_m3 <= 0.0:
        raise ValueError(
            "DESC joint seed surface volume must be finite and positive; "
            f"got {volume_m3!r} from {selected_seed.surface_path}."
        )
    return math.copysign(volume_m3, _desc_equilibrium_volume_sign(equilibrium))


def _desc_equilibrium_volume_sign(equilibrium: object) -> float:
    compute = getattr(equilibrium, "compute", None)
    if not callable(compute):
        raise ValueError(
            "DESC joint Volume objective requires an equilibrium with compute('V')."
        )
    volume_payload = compute("V")
    if not isinstance(volume_payload, Mapping):
        raise ValueError("equilibrium.compute('V') must return a mapping.")
    if "V" not in volume_payload:
        raise ValueError("equilibrium.compute('V') must include key 'V'.")
    signed_volume = float(volume_payload["V"])
    if not math.isfinite(signed_volume) or signed_volume == 0.0:
        raise ValueError(
            "equilibrium.compute('V')['V'] must be a finite nonzero number."
        )
    return math.copysign(1.0, signed_volume)


def _validate_seed_source_matches_equilibrium_seed(
    selected_seed: DescJointSeedCandidate,
    equilibrium_seed: DescEquilibriumSeedSpec,
) -> None:
    if (
        selected_seed.source_nfp is not None
        and selected_seed.source_nfp != equilibrium_seed.nfp
    ):
        raise ValueError(
            "DESC joint seed source NFP mismatch: selected seed records "
            f"{selected_seed.source_nfp}, equilibrium seed records "
            f"{equilibrium_seed.nfp}."
        )
    if (
        selected_seed.source_stellarator_symmetry is not None
        and selected_seed.source_stellarator_symmetry
        != equilibrium_seed.stellarator_symmetry
    ):
        raise ValueError(
            "DESC joint seed source stellarator_symmetry mismatch: selected seed "
            f"records {selected_seed.source_stellarator_symmetry}, equilibrium "
            f"seed records {equilibrium_seed.stellarator_symmetry}."
        )


if __name__ == "__main__":
    raise SystemExit(main())
