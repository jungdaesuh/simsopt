import argparse
from dataclasses import dataclass
import hashlib
import logging
import os
import io
import json
import time
import jax
import numpy as np

# SIMSOPT imports
from simsopt._core.optimizable import Optimizable
from simsopt.geo import (
    SurfaceRZFourier,
    SurfaceXYZTensorFourier,
    BoozerSurface,
    curves_to_vtk,
    CurveLength,
    LpCurveCurvature,
)
import simsopt.geo.surface as surface_module
from simsopt.geo.surfaceobjectives import (
    Volume,
    BoozerResidual,
    Iotas,
    NonQuasiSymmetricRatio,
    SurfaceSurfaceDistance,
    boozer_surface_residual,
    boozer_surface_residual_dB,
)
from simsopt.geo.curveobjectives import CurveCurveDistance, CurveSurfaceDistance
from simsopt.field import BiotSavart
from simsopt.objectives import QuadraticPenalty
from simsopt.objectives.utilities import forward_backward
from simsopt._core.optimizable import load
from simsopt._core.derivative import derivative_dec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

import sys

sys.path.insert(0, EXAMPLE_ROOT)
from hardware_constraints import (
    apply_hardware_constraint_verdict,
    sanitize_json_payload,
)
from jax_host_boundary import host_array, host_bool, host_float
from plotting_utils import cross_section_plot, norm_field_plot, norm_field_summary

SIMSOPT_ROOT = os.path.abspath(os.path.join(EXAMPLE_ROOT, "..", ".."))
REPO_ROOT = os.path.abspath(os.path.join(SIMSOPT_ROOT, ".."))
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
_TIMED_STAGE_LABELS = frozenset(
    {
        "after_boozer_surface_fit",
        "after_boozer_setup",
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
class SingleStageOuterOptimizerState:
    coil_dofs: object


jax.tree_util.register_dataclass(
    SingleStageOuterOptimizerState,
    data_fields=["coil_dofs"],
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


def _record_prefixed_stage_timings(timings, stage_marks, *, prefix, solve_start_s):
    if "after_boozer_lbfgs" in stage_marks:
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


def should_write_single_stage_artifacts(benchmark_mode: bool) -> bool:
    """Return whether the run should emit heavy plotting/VTK artifacts."""
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
        default=int(os.environ.get("MAXCOR", "300")),
        help="L-BFGS-B memory (number of corrections, default 300).",
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
        choices=["scipy", "hybrid", "ondevice"],
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
        choices=["scipy", "hybrid", "ondevice"],
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
            "Defaults to the historical quasi-Newton path when omitted."
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
        "--profile-target-lane",
        action="store_true",
        help=(
            "Record first-vs-warm timing breakdowns for the traceable target-lane "
            "objective closure suite used to study the outer optimization path."
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
    args.optimizer_backend = resolve_single_stage_default_optimizer_backend(
        args.backend,
        args.optimizer_backend,
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
    optimizer_backend="scipy",
    boozer_least_squares_algorithm=None,
    boozer_limited_memory=False,
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
    optimizer_backend: JAX inner optimizer selector recorded in metadata
    boozer_least_squares_algorithm: optional JAX Boozer LS algorithm override
    boozer_limited_memory: force the JAX Boozer LS solve through ondevice
        limited-memory routing without changing the default contract elsewhere
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
        if backend == "jax" and on_stage is not None:
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
    surf.least_squares_fit(surf_prev.gamma())
    fit_end_s = _perf_counter_s()
    _record_timing(timings, "boozer_surface_fit_s", fit_start_s, fit_end_s)
    if surface_dofs_override is not None:
        surf.set_dofs(np.asarray(surface_dofs_override, dtype=float))
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
            options["optimizer_backend"] = optimizer_backend
            if boozer_least_squares_algorithm is not None:
                options["least_squares_algorithm"] = boozer_least_squares_algorithm
            if optimizer_backend == "ondevice" and boozer_limited_memory:
                options["force_ondevice_limited_memory"] = True
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
    from simsopt.jax_core.curve_geometry import (
        curve_gamma_and_dash_from_spec,
        curve_geometry_from_spec,
    )
    from simsopt.jax_core.surface_rzfourier import (
        surface_rz_fourier_gamma_from_spec,
        surface_rz_fourier_spec_from_dofs,
    )

    try:
        banana_curve_index = next(
            coil_index
            for coil_index, coil in enumerate(bs.coils)
            if coil.curve is banana_curve
        )
    except StopIteration as exc:
        raise RuntimeError(
            "single-stage target-lane hardware filter could not locate the banana "
            "curve in bs.coils."
        ) from exc

    optimize_G = boozer_surface.res.get("G") is not None
    surface_spec = boozer_surface.surface.surface_spec()
    vessel_gamma = surface_rz_fourier_gamma_from_spec(
        vessel_surface.surface_spec()
    ).reshape((-1, 3))
    cc_dist_jax = jax.device_put(np.asarray(cc_dist, dtype=np.float64))
    cs_dist_jax = jax.device_put(np.asarray(cs_dist, dtype=np.float64))
    ss_dist_jax = jax.device_put(np.asarray(ss_dist, dtype=np.float64))
    curvature_threshold_jax = jax.device_put(
        np.asarray(curvature_threshold, dtype=np.float64)
    )
    inf = jax.device_put(np.asarray(np.inf, dtype=np.float64))

    def _pairwise_min_distance(points_a, points_b):
        deltas = points_a[:, None, :] - points_b[None, :, :]
        return jnp.min(jnp.sqrt(jnp.sum(jnp.square(deltas), axis=2)))

    def _coil_gamma_points(coil_spec):
        gamma, _ = curve_gamma_and_dash_from_spec(coil_spec.curve)
        if coil_spec.symmetry.has_rotation:
            gamma = gamma @ coil_spec.symmetry.rotmat
        return gamma.reshape((-1, 3))

    def _curve_curve_min_distance(coil_gammas):
        minimum = inf
        for i, gamma_i in enumerate(coil_gammas):
            for gamma_j in coil_gammas[:i]:
                minimum = jnp.minimum(
                    minimum,
                    _pairwise_min_distance(gamma_i, gamma_j),
                )
        return minimum

    def _curve_surface_min_distance(coil_gammas, surface_gamma):
        minimum = inf
        for gamma in coil_gammas:
            minimum = jnp.minimum(
                minimum,
                _pairwise_min_distance(gamma, surface_gamma),
            )
        return minimum

    def _surface_spec_from_sdofs(sdofs):
        return surface_rz_fourier_spec_from_dofs(
            sdofs,
            quadpoints_phi=surface_spec.quadpoints_phi,
            quadpoints_theta=surface_spec.quadpoints_theta,
            mpol=surface_spec.mpol,
            ntor=surface_spec.ntor,
            nfp=surface_spec.nfp,
            stellsym=surface_spec.stellsym,
        )

    def success_filter(coil_dofs, solved_x):
        coil_set_spec = bs.coil_set_spec_from_dofs(coil_dofs)
        coil_specs = bs.coil_specs_from_dofs(coil_dofs)
        sdofs, _iota, _G = boozer_surface._unpack_decision_vector_jax(
            solved_x,
            optimize_G,
            coil_set_spec=coil_set_spec,
        )
        surface_gamma = surface_rz_fourier_gamma_from_spec(
            _surface_spec_from_sdofs(sdofs)
        ).reshape((-1, 3))
        coil_gammas = tuple(_coil_gamma_points(coil_spec) for coil_spec in coil_specs)
        curve_curve_min_dist = _curve_curve_min_distance(coil_gammas)
        curve_surface_min_dist = _curve_surface_min_distance(
            coil_gammas,
            surface_gamma,
        )
        surface_vessel_min_dist = _pairwise_min_distance(surface_gamma, vessel_gamma)
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
            & (curve_curve_min_dist >= cc_dist_jax)
            & (curve_surface_min_dist >= cs_dist_jax)
            & (surface_vessel_min_dist >= ss_dist_jax)
            & (max_curvature <= curvature_threshold_jax)
        )

    return success_filter


def build_target_lane_outer_objectives(
    boozer_surface,
    bs,
    iota_target,
    *,
    use_value_and_grad: bool,
    profile_target_lane: bool,
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
            success_filter=success_filter,
        )

    if use_value_and_grad:
        target_value_and_grad_objective = runtime_bundle["value_and_grad"]
    else:
        target_scalar_objective = runtime_bundle["objective"]

    if profile_target_lane:
        target_lane_profile = profile_traceable_target_lane_objective(
            runtime_bundle["profile_suite"],
            np.asarray(bs.x.copy(), dtype=float),
        )

    return (
        target_scalar_objective,
        target_value_and_grad_objective,
        target_lane_profile,
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
    """Resolve the inner Boozer LS backend without changing CPU behavior."""
    if field_backend != "jax":
        return None
    if boozer_optimizer_backend is None:
        return optimizer_backend
    return boozer_optimizer_backend


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


_SINGLE_STAGE_COMPONENT_LABEL = "the single-stage outer loop"


def resolve_single_stage_optimizer_contract(field_backend, optimizer_backend):
    """Resolve the optimizer contract for the single-stage outer loop."""
    from simsopt.geo.optimizer_jax import resolve_outer_loop_optimizer_contract

    return resolve_outer_loop_optimizer_contract(
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
    return np.asarray(x, dtype=float)


def _single_stage_outer_optimizer_state(x):
    if isinstance(x, SingleStageOuterOptimizerState):
        return x
    return SingleStageOuterOptimizerState(coil_dofs=x)


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


def resolve_single_stage_outer_optimizer_initial_dofs(
    JF,
    bs,
    *,
    use_target_lane,
):
    """Return the optimizer-space DOFs for the selected outer-loop contract."""
    if use_target_lane:
        return _single_stage_outer_optimizer_state(bs.x.copy())
    return _single_stage_optimizer_dofs_array(JF.x.copy())


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
    callback,
    scalar_fun=None,
):
    """Run the single-stage outer optimization through the shared adapter."""
    from simsopt.geo.optimizer_jax import jax_minimize

    optimizer_dofs = dofs
    if fun is not None:
        optimizer_fun = fun
        value_and_grad = True
    elif contract.use_scalar_objective:
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
    if contract.use_scalar_objective:
        optimizer_dofs = _single_stage_outer_optimizer_state(dofs)
        if value_and_grad:
            optimizer_fun = _wrap_single_stage_outer_value_and_grad_objective(
                optimizer_fun
            )
        else:
            optimizer_fun = _wrap_single_stage_outer_scalar_objective(optimizer_fun)
    return jax_minimize(
        optimizer_fun,
        optimizer_dofs,
        method=contract.method,
        tol=gtol,
        maxiter=maxiter,
        options={
            "maxcor": int(maxcor),
            "ftol": float(ftol),
        },
        value_and_grad=value_and_grad,
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


def _restore_cpu_boozer_state(boozer_surface, run_dict):
    """Restore CPU BoozerSurface warm-start state from run_dict snapshot."""
    boozer_surface.surface.x = run_dict["sdofs"]
    boozer_surface.res["iota"] = run_dict["iota"]
    boozer_surface.res["G"] = run_dict["G"]


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
    is_cpu = isinstance(boozer_surface, BoozerSurface)
    if is_cpu:
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

        if is_cpu:
            _restore_cpu_boozer_state(boozer_surface, run_dict)

    return J, dJ


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


def snapshot_accepted_step_state(run_dict, boozer_surface, JF):
    """Persist the accepted-step solver/objective state into run_dict."""
    run_dict["lscount"] = 0
    run_dict["sdofs"] = boozer_surface.surface.x.copy()
    run_dict["iota"] = host_float(boozer_surface.res["iota"])
    run_dict["G"] = host_float(boozer_surface.res["G"])
    run_dict["J"] = host_float(JF.J())
    run_dict["dJ"] = host_array(JF.dJ())
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
        np.mean(
            np.abs(np.sum(host_array(bs.B()).reshape(unitn.shape) * unitn, axis=2))
        )
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
    print(f"{'Surface-Vessel Min Dist':{width}} = {surface_vessel_min:.6e}", file=buffer)
    print(f"{'Hardware Constraints OK':{width}} = {hardware_status['success']}", file=buffer)
    if hardware_status["violations"]:
        print(f"{'Hardware Violations':{width}} = {hardware_status['violations']}", file=buffer)
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
        _evaluate_candidate_impl(
            x_array,
            self.run_dict,
            self.boozer_surface,
            self.JF,
            self.objectives,
            self.diagnostics,
        )
        self.run_dict["x_prev"] = x_array.copy()

    def sync_accepted_step(self, x):
        """Refresh mutable state if needed, then snapshot one accepted step."""
        if self.reevaluate_before_accept:
            self._reevaluate_accepted_step(x)
        if self.benchmark_mode:
            snapshot_accepted_step_state(
                self.run_dict,
                self.boozer_surface,
                self.JF,
            )
            return
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
        before delegating.  This is the only mutation site for the outer
        loop — ``evaluate_candidate`` itself is mutation-free.
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

    run_dict = {
        "sdofs": boozer_surface.surface.x.copy(),
        "iota": host_float(boozer_surface.res["iota"]),
        "G": host_float(boozer_surface.res["G"]),
        "J": host_float(JF.J()),
        "dJ": host_array(JF.dJ()),
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
    args.curvature_threshold = resolve_curvature_threshold(args.curvature_threshold)
    stage2_bs_path = build_stage2_bs_path(args)
    stage2_results_path, stage2_results = load_stage2_results(stage2_bs_path)
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

    # Output directory setup
    OUT_DIR = args.output_root
    os.makedirs(OUT_DIR, exist_ok=True)
    boozer_type = {"initial": "least_squares", "final": "exact"}  # example
    stage = args.boozer_stage

    # ==============================================================================
    # SURFACE GEOMETRY DEFINITIONS
    # ==============================================================================
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

    # ==============================================================================
    # LOAD EQUILIBRIUM AND COILS
    # ==============================================================================
    plasma_surf_filename = args.plasma_surf_filename
    file_loc = build_equilibrium_path(args)
    bs = load(stage2_bs_path)

    use_jax = args.backend == "jax"
    write_artifacts = should_write_single_stage_artifacts(args.benchmark_mode)

    # JAX backend: wrap the loaded BiotSavart coils in BiotSavartJAX.
    # Keep a CPU reference for diagnostics and artifact output.
    bs_cpu_diag = bs if use_jax else None
    if use_jax:
        from simsopt.field.biotsavart_jax_backend import BiotSavartJAX

        _, IotasJAX, NonQuasiSymmetricRatioJAX = get_jax_surface_objective_classes()
        bs = BiotSavartJAX(bs.coils)
        iota_cls = IotasJAX
    else:
        iota_cls = Iotas

    bs_diag = diagnostic_field(bs, bs_cpu_diag)

    # Initialize the boundary magnetic surface and scale it to the target major radius
    surf = SurfaceRZFourier.from_wout(
        file_loc, range="half period", nphi=nphi, ntheta=ntheta, s=s
    )
    # scale the surface down to the target appropriate major radius
    surf.set_dofs(surf.get_dofs() * R0 / surf.major_radius())

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
        str(banana_surf_radius),
        str(nphi),
        str(ntheta),
        str(args.init_only),
        str(args.benchmark_mode),
        str(args.profile_target_lane),
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
    if boozer_least_squares_algorithm_record is not None:
        config_parts.append(str(boozer_least_squares_algorithm_record))
    config_str = "|".join(config_parts)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]
    OUT_DIR_ITER = OUT_DIR + f"/mpol={mpol}-ntor={ntor}-{config_hash}"
    os.makedirs(OUT_DIR_ITER, exist_ok=True)

    # Initialize Boozer surface with target parameters
    timings = {}
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
        boozer_least_squares_algorithm=boozer_least_squares_algorithm_record,
        timings_out=timings,
    )

    # ==============================================================================
    # SAVE INITIAL STATE
    # ==============================================================================
    initial_artifacts_start_s = _perf_counter_s()
    if write_artifacts:
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
    if write_artifacts:
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
    initial_iota = host_float(build_iota_objective(boozer_surface, iota_cls).J())
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
    use_target_lane = outer_contract.use_scalar_objective
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
        target_scalar_objective = None
        target_value_and_grad_objective = None
        target_lane_success_filter = None
        if use_target_lane:
            target_lane_success_filter = (
                build_single_stage_target_lane_hardware_success_filter(
                    boozer_surface,
                    bs,
                    banana_curve,
                    VV,
                    cc_dist=CC_DIST,
                    cs_dist=CS_DIST,
                    ss_dist=SS_DIST,
                    curvature_threshold=CURVATURE_THRESHOLD,
                )
            )
            (
                target_scalar_objective,
                target_value_and_grad_objective,
                target_lane_profile,
            ) = build_target_lane_outer_objectives(
                boozer_surface,
                bs,
                iota_target,
                use_value_and_grad=use_target_lane_vg,
                profile_target_lane=args.profile_target_lane,
                success_filter=target_lane_success_filter,
            )
        accepted_step_callback = resolve_target_lane_accepted_step_callback(
            adapter,
            use_target_lane=use_target_lane,
            sync_policy=effective_target_lane_sync_policy,
        )
        res = run_single_stage_optimizer(
            target_value_and_grad_objective if use_target_lane else adapter,
            dofs,
            callback=accepted_step_callback,
            contract=outer_contract,
            maxiter=MAXITER,
            ftol=ftol,
            gtol=gtol,
            maxcor=args.maxcor,
            scalar_fun=target_scalar_objective,
        )
        outer_optimizer_end_s = _perf_counter_s()
        termination_message = str(res.message)
        optimizer_success = bool(res.success)
        _record_timing(
            timings,
            "outer_optimizer_s",
            outer_optimizer_start_s,
            outer_optimizer_end_s,
        )
        res_nit = res.nit
        print(res.message)
        if use_target_lane and accepted_step_callback is None and res.nit > 0:
            target_lane_sync_start_s = _perf_counter_s()
            adapter.sync_accepted_step(res.x)
            _record_timing(
                timings,
                "target_lane_final_sync_s",
                target_lane_sync_start_s,
                _perf_counter_s(),
            )

        # Restore final accepted state to the Optimizable graph so
        # post-optimization diagnostics and artifact writers see
        # consistent values even if the last evaluate_candidate was a
        # rejected trial.
        restore_from_pytree(
            JF,
            boozer_surface,
            run_dict,
            coil_dofs=res.x,
            apply_coil_dofs=dof_setter,
        )

        # ==============================================================================
        # SAVE OPTIMIZED STATE
        # ==============================================================================
        final_artifacts_start_s = _perf_counter_s()
        if write_artifacts:
            # Save optimized coil configurations
            curves_to_vtk(curves, OUT_DIR_ITER + "/curves_opt", close=True)
            bs_diag.save(OUT_DIR_ITER + "/biot_savart_opt.json")

            # Save optimized surface with magnetic field normal component data
            bs_diag.set_points(boozer_surface.surface.gamma().reshape((-1, 3)))
            unitn = boozer_surface.surface.unitnormal()
            pointData = {
                "B_N/B": np.sum(bs_diag.B().reshape(unitn.shape) * unitn, axis=2)[
                    :, :, None
                ]
                / np.sqrt(np.sum(bs_diag.B().reshape(unitn.shape) ** 2, axis=2))[
                    :, :, None
                ]
            }

            # Print final results
            boozer_surface.surface.to_vtk(
                OUT_DIR_ITER + "/surf_opt",
                extra_data=pointData,
            )
            boozer_surface.surface.save(OUT_DIR_ITER + "/surf_opt.json")

        final_volume = host_float(boozer_surface.surface.volume())
        final_iota = host_float(build_iota_objective(boozer_surface, iota_cls).J())
        final_max_curvature = np.max(banana_curve.kappa())
        print(f"Volume: {final_volume}")
        print(f"Iota: {final_iota}")
        print(f"Max Curvature: {final_max_curvature}")

        # Generate final diagnostic plots
        if write_artifacts:
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

    final_self_intersecting = update_self_intersection_status(
        run_dict, boozer_surface.surface
    )
    final_self_intersection_check_available = run_dict[
        "self_intersection_check_available"
    ]

    # Save the results of optimization to a separate file
    final_hardware_status = evaluate_single_stage_hardware_constraints(
        float(JCurveCurve.shortest_distance()),
        CC_DIST,
        float(JCurveSurface.shortest_distance()),
        CS_DIST,
        float(JSurfSurf.shortest_distance()),
        SS_DIST,
        float(final_max_curvature),
        CURVATURE_THRESHOLD,
    )
    optimizer_success, termination_message = apply_hardware_constraint_verdict(
        optimizer_success,
        termination_message,
        final_hardware_status,
        init_only=args.init_only,
    )
    results = {
        "PLASMA_SURF_FILENAME": plasma_surf_filename,
        "PLASMA_SURF_PATH": file_loc,
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
        "profile_target_lane": bool(args.profile_target_lane),
        "init_only": args.init_only,
        "max_iterations": MAXITER,
        "maxcor": args.maxcor,
        "iterations": res_nit,
        "TERMINATION_MESSAGE": termination_message,
        "OPTIMIZER_SUCCESS": optimizer_success,
        "TARGET_VOLUME": float(vol_target),
        "TARGET_IOTA": float(iota_target),
        "FINAL_VOLUME": float(final_volume),
        "FINAL_IOTA": float(final_iota),
        "FIELD_ERROR": float(fieldError),
        "SELF_INTERSECTING": final_self_intersecting,
        "SELF_INTERSECTION_CHECK_AVAILABLE": final_self_intersection_check_available,
        "MAX_CURVATURE": float(final_max_curvature),
        "COIL_LENGTH": float(curvelength.J()),
        "CURVE_CURVE_MIN_DIST": float(JCurveCurve.shortest_distance()),
        "CURVE_SURFACE_MIN_DIST": float(JCurveSurface.shortest_distance()),
        "SURFACE_VESSEL_MIN_DIST": float(JSurfSurf.shortest_distance()),
        "HARDWARE_CONSTRAINTS_OK": final_hardware_status["success"],
        "HARDWARE_CONSTRAINT_VIOLATIONS": final_hardware_status["violations"],
        "INITIAL_VOLUME": float(initial_volume),
        "INITIAL_IOTA": float(initial_iota),
        "INITIAL_FIELD_ERROR": float(initial_field_error),
        "INITIAL_MAX_CURVATURE": float(initial_max_curvature),
        "num_tf_coils": static_config["num_tf_coils"],
        "tf_currents": static_config["tf_currents"],
    }
    timings["script_total_s"] = _elapsed_s(run_wall_start_s, _perf_counter_s())
    results["TIMINGS"] = timings
    if target_lane_profile is not None:
        results["TARGET_LANE_PROFILE"] = target_lane_profile
    with open(os.path.join(OUT_DIR_ITER, "results.json"), "w") as outfile:
        json.dump(sanitize_json_payload(results), outfile, indent=2, allow_nan=False)
