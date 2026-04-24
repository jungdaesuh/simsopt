from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from workflow_runner_common import (  # noqa: E402
    SINGLE_STAGE_SCRIPT_PATH,
    add_seed_order_upgrade_argument,
    append_allow_offspec_engineering_flag,
    add_stage2_warm_start_seed_arguments,
    append_bool_flag,
    append_optional_flag,
    append_single_stage_handoff_flags,
    discover_single_results_path,
    load_json,
    load_validated_stage2_seed_results,
    maybe_load_validated_stage2_seed_results,
    resolved_optional_path,
    resolved_path,
    run_command,
    snapshot_single_results_paths,
    timeout_or_none,
)
from banana_opt.single_stage_banana_current_mode import (  # noqa: E402
    BANANA_CURRENT_MODE_INDEPENDENT,
    BANANA_CURRENT_MODE_SHARED,
)
from banana_opt.lbfgsb_defaults import DEFAULT_LBFGSB_MAXCOR  # noqa: E402
from banana_opt.surface_mode_contracts import (  # noqa: E402
    DEFAULT_INNER_SURFACE_RATIO,
    SURFACE_MODE_CHOICES,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_single_stage_goal_mode_comparison"
DEFAULT_SUMMARY_JSON = "single_stage_goal_mode_comparison_summary.json"
GOAL_MODES = ("target", "frontier")
PRESERVED_RESULT_SOURCES = (
    ("best_feasible_partial", "results_best_feasible.partial.json"),
    ("best_accepted_partial", "results_best_accepted.partial.json"),
)
_PRESERVED_BIOTSAVART_FILENAMES = {
    "best_feasible_partial": "biot_savart_best_feasible.json",
    "best_accepted_partial": "biot_savart_best_accepted.json",
}
_PRESERVED_SURFACE_STEMS = {
    "best_feasible_partial": "surf_best_feasible",
    "best_accepted_partial": "surf_best_accepted",
}
_RESULT_BIOTSAVART_FILENAMES = {
    "final": "biot_savart_opt.json",
    **_PRESERVED_BIOTSAVART_FILENAMES,
}
_RESULT_SURFACE_STEMS = {
    "final": "surf_opt",
    **_PRESERVED_SURFACE_STEMS,
}


def _single_stage_preserved_result_matches(
    output_root: str | Path,
    filename: str,
) -> list[Path]:
    return sorted(Path(output_root).glob(f"mpol=*-ntor=*/{filename}"))


def snapshot_single_stage_preserved_results_paths(output_root: str | Path) -> dict[Path, int]:
    snapshot: dict[Path, int] = {}
    for _, filename in PRESERVED_RESULT_SOURCES:
        for path in _single_stage_preserved_result_matches(output_root, filename):
            snapshot[path] = path.stat().st_mtime_ns
    return snapshot


def _expect_single_preserved_result_match(
    matches: list[Path],
    *,
    source_label: str,
    filename: str,
    output_root: str | Path,
    match_kind: str | None = None,
) -> Path | None:
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        return None

    match_descriptor = "" if match_kind is None else f"{match_kind} "
    raise FileNotFoundError(
        f"Expected exactly one {match_descriptor}{filename} for {source_label} under "
        f"{output_root}, found {len(matches)}"
    )


def discover_single_stage_salvage_results_path(
    output_root: str | Path,
    *,
    previous_snapshot: dict[Path, int] | None = None,
) -> tuple[str, Path]:
    for source_label, filename in PRESERVED_RESULT_SOURCES:
        matches = _single_stage_preserved_result_matches(output_root, filename)
        if not matches:
            continue
        if previous_snapshot is not None:
            new_match = _expect_single_preserved_result_match(
                [path for path in matches if path not in previous_snapshot],
                source_label=source_label,
                filename=filename,
                output_root=output_root,
                match_kind="new",
            )
            if new_match is not None:
                return source_label, new_match
            updated_matches = [
                path
                for path in matches
                if previous_snapshot.get(path) != path.stat().st_mtime_ns
            ]
            updated_match = _expect_single_preserved_result_match(
                updated_matches,
                source_label=source_label,
                filename=filename,
                output_root=output_root,
                match_kind="updated",
            )
            if updated_match is not None:
                return source_label, updated_match
            continue
        match = _expect_single_preserved_result_match(
            matches,
            source_label=source_label,
            filename=filename,
            output_root=output_root,
        )
        if match is not None:
            return source_label, match
    raise FileNotFoundError(
        "Expected one preserved partial single-stage result after the run, but found "
        f"neither {PRESERVED_RESULT_SOURCES[0][1]!r} nor {PRESERVED_RESULT_SOURCES[1][1]!r} "
        f"under {output_root}"
    )


def single_stage_artifact_bundle_from_results(
    result_source: str,
    results_path: str | Path,
) -> dict[str, Path]:
    results_path = Path(results_path)
    if result_source in _RESULT_BIOTSAVART_FILENAMES:
        surface_stem = results_path.with_name(_RESULT_SURFACE_STEMS[result_source])
        outer_boozer_surface_name = (
            "surf_opt_boozer_surface.json"
            if result_source == "final"
            else f"{surface_stem.name}_outer_boozer_surface.json"
        )
        return {
            "results_path": results_path,
            "bs_path": results_path.with_name(
                _RESULT_BIOTSAVART_FILENAMES[result_source]
            ),
            "surface_stem": surface_stem,
            "outer_boozer_surface_path": results_path.with_name(
                outer_boozer_surface_name
            ),
        }
    raise ValueError(f"Unsupported single-stage result source {result_source!r}.")


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run matched single-stage target-vs-frontier comparisons from one explicit "
            "Stage 2 seed artifact."
        ),
        add_help=add_help,
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--plasma-surf-filename",
        required=True,
        help="VMEC wout filename used as the single-stage target surface.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        required=True,
        help="Path to the Stage 2 biot_savart_opt.json seed artifact shared by both goal modes.",
    )
    parser.add_argument(
        "--allow-init-only-stage2-seed",
        action="store_true",
        help=(
            "Allow reusing a Stage 2 artifact whose sibling results.json reports "
            "init_only=true. Disabled by default because init-only smoke seeds can "
            "land single-stage in the wrong transform basin."
        ),
    )
    parser.add_argument("--equilibria-dir", default=None)
    parser.add_argument(
        "--equilibrium-path",
        default=None,
        help="Optional explicit equilibrium path forwarded into the single-stage run.",
    )
    add_seed_order_upgrade_argument(parser)
    add_stage2_warm_start_seed_arguments(parser)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--summary-json",
        default=None,
        help=f"Optional summary path. Defaults to <output-root>/{DEFAULT_SUMMARY_JSON}.",
    )
    parser.add_argument("--single-stage-timeout-seconds", type=float, default=0.0)
    parser.add_argument("--nphi", type=int, default=91)
    parser.add_argument("--ntheta", type=int, default=32)
    parser.add_argument("--mpol", type=int, default=8)
    parser.add_argument("--ntor", type=int, default=6)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--maxcor", type=int, default=DEFAULT_LBFGSB_MAXCOR)
    parser.add_argument("--ftol", type=float, default=1e-15)
    parser.add_argument("--gtol", type=float, default=1e-15)
    parser.add_argument("--constraint-method", choices=["penalty", "alm"], default="penalty")
    parser.add_argument("--alm-max-outer-iters", type=int, default=10)
    parser.add_argument("--alm-penalty-init", type=float, default=1.0)
    parser.add_argument("--alm-penalty-scale", type=float, default=10.0)
    parser.add_argument("--alm-penalty-max", type=float, default=1.0e8)
    parser.add_argument("--alm-feas-tol", type=float, default=1e-6)
    parser.add_argument("--alm-stationarity-tol", type=float, default=1e-6)
    parser.add_argument("--alm-trust-radius-init", type=float, default=0.05)
    parser.add_argument("--alm-trust-radius-min", type=float, default=1e-4)
    parser.add_argument("--alm-trust-radius-shrink", type=float, default=0.5)
    parser.add_argument("--alm-trust-radius-grow", type=float, default=1.5)
    parser.add_argument("--alm-max-inner-attempts", type=int, default=4)
    parser.add_argument("--alm-max-subproblem-continuations", type=int, default=20)
    parser.add_argument("--alm-distance-smoothing", type=float, default=0.005)
    parser.add_argument("--alm-curvature-smoothing", type=float, default=0.05)
    parser.add_argument("--alm-formulation", choices=["weighted_sum", "thresholded_physics"], default=os.environ.get("ALM_FORMULATION", "weighted_sum"))
    parser.add_argument("--alm-qs-threshold", type=float, default=float(os.environ["ALM_QS_THRESHOLD"]) if "ALM_QS_THRESHOLD" in os.environ else None)
    parser.add_argument("--alm-boozer-threshold", type=float, default=float(os.environ["ALM_BOOZER_THRESHOLD"]) if "ALM_BOOZER_THRESHOLD" in os.environ else None)
    parser.add_argument(
        "--alm-iota-penalty-threshold",
        type=float,
        default=float(os.environ["ALM_IOTA_PENALTY_THRESHOLD"]) if "ALM_IOTA_PENALTY_THRESHOLD" in os.environ else None,
    )
    parser.add_argument(
        "--alm-length-penalty-threshold",
        type=float,
        default=float(os.environ["ALM_LENGTH_PENALTY_THRESHOLD"]) if "ALM_LENGTH_PENALTY_THRESHOLD" in os.environ else None,
    )
    parser.add_argument("--iota-target", type=float, default=0.15)
    parser.add_argument("--vol-target", type=float, default=0.10)
    parser.add_argument(
        "--constraint-weight",
        type=float,
        default=float(os.environ.get("CONSTRAINT_WEIGHT", "1.0")),
        help=(
            "Boozer constraint weight forwarded into the single-stage run. "
            "Use a negative value to select the exact Boozer Newton solver."
        ),
    )
    parser.add_argument("--boozer-I", type=float, default=float(os.environ["BOOZER_I"]) if "BOOZER_I" in os.environ else None)
    parser.add_argument(
        "--plasma-current-A",
        type=float,
        default=float(os.environ["PLASMA_CURRENT_A"]) if "PLASMA_CURRENT_A" in os.environ else None,
    )
    parser.add_argument(
        "--single-stage-banana-current-mode",
        choices=[BANANA_CURRENT_MODE_SHARED, BANANA_CURRENT_MODE_INDEPENDENT],
        default=os.environ.get(
            "SINGLE_STAGE_BANANA_CURRENT_MODE",
            BANANA_CURRENT_MODE_SHARED,
        ),
        help=(
            "Banana-current control mode forwarded into the single-stage run. "
            "'shared' preserves the legacy one-current contract, while "
            "'independent' gives each loaded banana coil its own current DOF."
        ),
    )
    parser.add_argument(
        "--num-tf-coils",
        type=int,
        default=int(os.environ.get("NUM_TF_COILS", "20")),
        help="Expected number of TF coils in the loaded Stage 2 artifact.",
    )
    parser.add_argument("--banana-surf-radius", type=float, default=float(os.environ["BANANA_SURF_RADIUS"]) if "BANANA_SURF_RADIUS" in os.environ else None)
    parser.add_argument(
        "--stage2-seed-tf-current-A",
        type=float,
        default=float(os.environ["STAGE2_SEED_TF_CURRENT_A"]) if "STAGE2_SEED_TF_CURRENT_A" in os.environ else None,
        help=(
            "Optional legacy backfill for TF_CURRENT_A when the loaded Stage 2 "
            "artifact predates that metadata field."
        ),
    )
    parser.add_argument("--num-surfaces", type=int, choices=[1, 2], default=int(os.environ.get("NUM_SURFACES", "1")))
    parser.add_argument(
        "--inner-surface-ratio",
        type=float,
        default=float(
            os.environ.get(
                "INNER_SURFACE_RATIO",
                str(DEFAULT_INNER_SURFACE_RATIO),
            )
        ),
    )
    parser.add_argument(
        "--surface-mode",
        choices=SURFACE_MODE_CHOICES,
        default=os.environ.get("SURFACE_MODE"),
    )
    parser.add_argument("--surface-gap-threshold", type=float, default=float(os.environ.get("SURFACE_GAP_THRESHOLD", "0.0")))
    parser.add_argument("--multisurface-ramp-iterations", type=int, default=int(os.environ.get("MULTISURFACE_RAMP_ITERATIONS", "5")))
    parser.add_argument("--inner-surface-initial-weight", type=float, default=float(os.environ.get("INNER_SURFACE_INITIAL_WEIGHT", "0.0")))
    parser.add_argument("--multisurface-initial-step-scale", type=float, default=float(os.environ.get("MULTISURFACE_INITIAL_STEP_SCALE", "1.0")))
    parser.add_argument("--multisurface-initial-step-maxiter", type=int, default=int(os.environ.get("MULTISURFACE_INITIAL_STEP_MAXITER", "0")))
    parser.add_argument("--boozer-stage", choices=["initial", "final"], default=os.environ.get("BOOZER_STAGE", "initial"))
    parser.add_argument("--boozer-stage-refinement", action="store_true")
    parser.add_argument("--refinement-boozer-stage", choices=["initial", "final"], default=os.environ.get("REFINEMENT_BOOZER_STAGE", "final"))
    parser.add_argument("--refinement-maxiter", type=int, default=int(os.environ.get("REFINEMENT_MAXITER", "100")))
    parser.add_argument("--refinement-chunk-maxiter", type=int, default=int(os.environ.get("REFINEMENT_CHUNK_MAXITER", "20")))
    parser.add_argument("--refinement-max-stalled-chunks", type=int, default=int(os.environ.get("REFINEMENT_MAX_STALLED_CHUNKS", "2")))
    parser.add_argument("--res-weight", type=float, default=1000.0)
    parser.add_argument("--iotas-weight", type=float, default=100.0)
    parser.add_argument(
        "--frontier-volume-weight",
        type=float,
        default=float(os.environ["FRONTIER_VOLUME_WEIGHT"]) if "FRONTIER_VOLUME_WEIGHT" in os.environ else None,
        help=(
            "Independent volume-reward weight for frontier mode. When omitted, forwarded "
            "as-is (None) so the single-stage script falls back to --iotas-weight."
        ),
    )
    parser.add_argument("--frontier-scalarization-type", default=None)
    parser.add_argument("--frontier-reference-iota", type=float, default=None)
    parser.add_argument("--frontier-reference-iota-scale", type=float, default=None)
    parser.add_argument("--frontier-reference-volume", type=float, default=None)
    parser.add_argument("--frontier-reference-volume-scale", type=float, default=None)
    parser.add_argument("--frontier-reference-qa", type=float, default=None)
    parser.add_argument("--frontier-reference-boozer", type=float, default=None)
    parser.add_argument("--frontier-boozer-trust-threshold", type=float, default=None)
    parser.add_argument("--frontier-boozer-trust-penalty-scale", type=float, default=None)
    parser.add_argument("--frontier-chebyshev-rho", type=float, default=None)
    parser.add_argument("--frontier-chebyshev-sharpness", type=float, default=None)
    parser.add_argument("--frontier-chebyshev-weight-iota", type=float, default=None)
    parser.add_argument("--frontier-chebyshev-weight-volume", type=float, default=None)
    parser.add_argument("--frontier-chebyshev-weight-qa", type=float, default=None)
    parser.add_argument("--frontier-chebyshev-weight-boozer", type=float, default=None)
    parser.add_argument("--epsilon-constraint-qa-max", type=float, default=None)
    parser.add_argument("--epsilon-constraint-boozer-max", type=float, default=None)
    parser.add_argument("--frontier-epsilon-penalty-weight", type=float, default=None)
    parser.add_argument("--cc-weight", type=float, default=100.0)
    parser.add_argument("--curvature-weight", type=float, default=0.1)
    parser.add_argument("--length-weight", type=float, default=1.0)
    parser.add_argument("--length-target", type=float, default=float(os.environ["SS_LENGTH_TARGET"]) if "SS_LENGTH_TARGET" in os.environ else None)
    parser.add_argument("--cs-weight", type=float, default=1.0)
    parser.add_argument("--surf-dist-weight", type=float, default=1000.0)
    parser.add_argument("--cc-dist", type=float, default=0.05)
    parser.add_argument("--cs-dist", type=float, default=0.015)
    parser.add_argument("--ss-dist", type=float, default=0.04)
    parser.add_argument("--curvature-threshold", type=float, default=100.0)
    parser.add_argument("--checkpoint-every", type=int, default=int(os.environ.get("CHECKPOINT_EVERY", "0")))
    parser.add_argument(
        "--resume-solver-checkpoint",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--topology-gate-fieldlines", type=int, default=int(os.environ.get("TOPOLOGY_GATE_FIELDLINES", "4")))
    parser.add_argument("--topology-gate-tmax", type=float, default=float(os.environ.get("TOPOLOGY_GATE_TMAX", "2.0")))
    parser.add_argument("--topology-gate-tol", type=float, default=float(os.environ.get("TOPOLOGY_GATE_TOL", "1e-7")))
    parser.add_argument("--topology-gate-survival-threshold", type=float, default=float(os.environ.get("TOPOLOGY_GATE_SURVIVAL_THRESHOLD", "0.25")))
    parser.add_argument("--topology-gate-penalty-scale", type=float, default=float(os.environ.get("TOPOLOGY_GATE_PENALTY_SCALE", "4.0")))
    parser.add_argument("--topology-scorer-every", type=int, default=int(os.environ.get("TOPOLOGY_SCORER_EVERY", "0")))
    parser.add_argument("--topology-scorer-nfieldlines", type=int, default=int(os.environ.get("TOPOLOGY_SCORER_NFIELDLINES", "12")))
    parser.add_argument("--topology-scorer-tmax", type=float, default=float(os.environ.get("TOPOLOGY_SCORER_TMAX", "50.0")))
    parser.add_argument("--confinement-objective-weight", type=float, default=float(os.environ.get("CONFINEMENT_OBJECTIVE_WEIGHT", "0.0")))
    parser.add_argument("--confinement-surrogate-worst-k", type=int, default=int(os.environ.get("CONFINEMENT_SURROGATE_WORST_K", "3")))
    parser.add_argument("--confinement-surrogate-early-threshold", type=float, default=float(os.environ.get("CONFINEMENT_SURROGATE_EARLY_THRESHOLD", "0.2")))
    parser.add_argument("--confinement-surrogate-mean-weight", type=float, default=float(os.environ.get("CONFINEMENT_SURROGATE_MEAN_WEIGHT", "0.2")))
    parser.add_argument("--confinement-surrogate-worst-weight", type=float, default=float(os.environ.get("CONFINEMENT_SURROGATE_WORST_WEIGHT", "0.6")))
    parser.add_argument("--confinement-surrogate-early-weight", type=float, default=float(os.environ.get("CONFINEMENT_SURROGATE_EARLY_WEIGHT", "0.2")))
    parser.add_argument(
        "--hardware-search-mode",
        choices=["hard", "warn", "adaptive"],
        default="hard",
    )
    parser.add_argument("--hardware-search-soft-iterations", type=int, default=0)
    parser.add_argument("--basin-hops", type=int, default=0)
    parser.add_argument("--basin-stepsize", type=float, default=0.01)
    parser.add_argument("--basin-temperature", type=float, default=1.0)
    parser.add_argument("--basin-niter-success", type=int, default=0)
    parser.add_argument("--basin-seed", type=int, default=-1)
    parser.add_argument("--init-only", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def load_validated_stage2_seed_metadata(
    args: argparse.Namespace,
    *,
    stage2_bs_path: Path | None = None,
) -> tuple[Path, Path, dict]:
    return load_validated_stage2_seed_results(
        args,
        owner_label="run_single_stage_goal_mode_comparison.py",
        stage2_bs_path=stage2_bs_path,
    )


def maybe_load_validated_stage2_seed_metadata(
    args: argparse.Namespace,
) -> tuple[Path, Path | None, dict | None]:
    return maybe_load_validated_stage2_seed_results(
        args,
        owner_label="run_single_stage_goal_mode_comparison.py",
    )


def build_single_stage_goal_mode_command(
    args: argparse.Namespace,
    *,
    goal_mode: str,
    stage2_bs_path: Path,
    case_output_root: Path,
) -> list[str]:
    equilibria_dir = resolved_optional_path(args.equilibria_dir)
    command = [
        args.python_executable,
        str(SINGLE_STAGE_SCRIPT_PATH),
        "--plasma-surf-filename",
        Path(args.plasma_surf_filename).name,
        "--stage2-bs-path",
        str(stage2_bs_path),
        "--output-root",
        str(case_output_root),
        "--single-stage-goal-mode",
        goal_mode,
        "--nphi",
        str(args.nphi),
        "--ntheta",
        str(args.ntheta),
        "--mpol",
        str(args.mpol),
        "--ntor",
        str(args.ntor),
        "--maxiter",
        str(args.maxiter),
        "--maxcor",
        str(args.maxcor),
        "--ftol",
        str(args.ftol),
        "--gtol",
        str(args.gtol),
        "--constraint-method",
        args.constraint_method,
        "--single-stage-banana-current-mode",
        args.single_stage_banana_current_mode,
        "--alm-formulation",
        args.alm_formulation,
        "--iota-target",
        str(args.iota_target),
        "--vol-target",
        str(args.vol_target),
        "--res-weight",
        str(args.res_weight),
        "--iotas-weight",
        str(args.iotas_weight),
        "--cc-weight",
        str(args.cc_weight),
        "--curvature-weight",
        str(args.curvature_weight),
        "--length-weight",
        str(args.length_weight),
        "--cs-weight",
        str(args.cs_weight),
        "--surf-dist-weight",
        str(args.surf_dist_weight),
        "--cc-dist",
        str(args.cc_dist),
        "--cs-dist",
        str(args.cs_dist),
        "--ss-dist",
        str(args.ss_dist),
        "--curvature-threshold",
        str(args.curvature_threshold),
        "--num-surfaces",
        str(args.num_surfaces),
        "--inner-surface-ratio",
        str(args.inner_surface_ratio),
    ]
    if getattr(args, "surface_mode", None):
        command.extend(
            [
                "--surface-mode",
                str(args.surface_mode),
            ]
        )
    command.extend(
        [
            "--surface-gap-threshold",
            str(args.surface_gap_threshold),
            "--multisurface-ramp-iterations",
            str(args.multisurface_ramp_iterations),
            "--inner-surface-initial-weight",
            str(args.inner_surface_initial_weight),
            "--multisurface-initial-step-scale",
            str(args.multisurface_initial_step_scale),
            "--multisurface-initial-step-maxiter",
            str(args.multisurface_initial_step_maxiter),
            "--boozer-stage",
            args.boozer_stage,
            "--refinement-boozer-stage",
            args.refinement_boozer_stage,
            "--refinement-maxiter",
            str(args.refinement_maxiter),
            "--refinement-chunk-maxiter",
            str(args.refinement_chunk_maxiter),
            "--refinement-max-stalled-chunks",
            str(args.refinement_max_stalled_chunks),
            "--checkpoint-every",
            str(args.checkpoint_every),
            "--topology-gate-fieldlines",
            str(args.topology_gate_fieldlines),
            "--topology-gate-tmax",
            str(args.topology_gate_tmax),
            "--topology-gate-tol",
            str(args.topology_gate_tol),
            "--topology-gate-survival-threshold",
            str(args.topology_gate_survival_threshold),
            "--topology-gate-penalty-scale",
            str(args.topology_gate_penalty_scale),
            "--topology-scorer-every",
            str(args.topology_scorer_every),
            "--topology-scorer-nfieldlines",
            str(args.topology_scorer_nfieldlines),
            "--topology-scorer-tmax",
            str(args.topology_scorer_tmax),
            "--confinement-objective-weight",
            str(args.confinement_objective_weight),
            "--confinement-surrogate-worst-k",
            str(args.confinement_surrogate_worst_k),
            "--confinement-surrogate-early-threshold",
            str(args.confinement_surrogate_early_threshold),
            "--confinement-surrogate-mean-weight",
            str(args.confinement_surrogate_mean_weight),
            "--confinement-surrogate-worst-weight",
            str(args.confinement_surrogate_worst_weight),
            "--confinement-surrogate-early-weight",
            str(args.confinement_surrogate_early_weight),
            "--hardware-search-mode",
            args.hardware_search_mode,
            "--hardware-search-soft-iterations",
            str(args.hardware_search_soft_iterations),
        ]
    )
    if equilibria_dir is not None:
        command.extend(["--equilibria-dir", str(equilibria_dir)])
    append_single_stage_handoff_flags(command, args)
    append_optional_flag(command, "--length-target", args.length_target)
    append_allow_offspec_engineering_flag(
        command,
        length_target=args.length_target,
        curvature_threshold=args.curvature_threshold,
    )
    append_optional_flag(command, "--frontier-volume-weight", args.frontier_volume_weight)
    append_optional_flag(
        command,
        "--resume-solver-checkpoint",
        getattr(args, "resume_solver_checkpoint", None),
    )
    append_optional_flag(
        command,
        "--frontier-scalarization-type",
        getattr(args, "frontier_scalarization_type", None),
    )
    append_optional_flag(
        command,
        "--frontier-reference-iota",
        getattr(args, "frontier_reference_iota", None),
    )
    append_optional_flag(
        command,
        "--frontier-reference-iota-scale",
        getattr(args, "frontier_reference_iota_scale", None),
    )
    append_optional_flag(
        command,
        "--frontier-reference-volume",
        getattr(args, "frontier_reference_volume", None),
    )
    append_optional_flag(
        command,
        "--frontier-reference-volume-scale",
        getattr(args, "frontier_reference_volume_scale", None),
    )
    append_optional_flag(
        command,
        "--frontier-reference-qa",
        getattr(args, "frontier_reference_qa", None),
    )
    append_optional_flag(
        command,
        "--frontier-reference-boozer",
        getattr(args, "frontier_reference_boozer", None),
    )
    append_optional_flag(
        command,
        "--frontier-boozer-trust-threshold",
        getattr(args, "frontier_boozer_trust_threshold", None),
    )
    append_optional_flag(
        command,
        "--frontier-boozer-trust-penalty-scale",
        getattr(args, "frontier_boozer_trust_penalty_scale", None),
    )
    append_optional_flag(
        command,
        "--frontier-chebyshev-rho",
        getattr(args, "frontier_chebyshev_rho", None),
    )
    append_optional_flag(
        command,
        "--frontier-chebyshev-sharpness",
        getattr(args, "frontier_chebyshev_sharpness", None),
    )
    append_optional_flag(
        command,
        "--frontier-chebyshev-weight-iota",
        getattr(args, "frontier_chebyshev_weight_iota", None),
    )
    append_optional_flag(
        command,
        "--frontier-chebyshev-weight-volume",
        getattr(args, "frontier_chebyshev_weight_volume", None),
    )
    append_optional_flag(
        command,
        "--frontier-chebyshev-weight-qa",
        getattr(args, "frontier_chebyshev_weight_qa", None),
    )
    append_optional_flag(
        command,
        "--frontier-chebyshev-weight-boozer",
        getattr(args, "frontier_chebyshev_weight_boozer", None),
    )
    append_optional_flag(
        command,
        "--epsilon-constraint-qa-max",
        getattr(args, "epsilon_constraint_qa_max", None),
    )
    append_optional_flag(
        command,
        "--epsilon-constraint-boozer-max",
        getattr(args, "epsilon_constraint_boozer_max", None),
    )
    append_optional_flag(
        command,
        "--frontier-epsilon-penalty-weight",
        getattr(args, "frontier_epsilon_penalty_weight", None),
    )
    append_optional_flag(command, "--alm-qs-threshold", args.alm_qs_threshold)
    append_optional_flag(command, "--alm-boozer-threshold", args.alm_boozer_threshold)
    append_optional_flag(command, "--alm-iota-penalty-threshold", args.alm_iota_penalty_threshold)
    append_optional_flag(command, "--alm-length-penalty-threshold", args.alm_length_penalty_threshold)
    append_bool_flag(command, "--boozer-stage-refinement", args.boozer_stage_refinement)
    if args.constraint_method == "alm":
        command.extend(
            [
                "--alm-max-outer-iters",
                str(args.alm_max_outer_iters),
                "--alm-penalty-init",
                str(args.alm_penalty_init),
                "--alm-penalty-scale",
                str(args.alm_penalty_scale),
                "--alm-penalty-max",
                str(args.alm_penalty_max),
                "--alm-feas-tol",
                str(args.alm_feas_tol),
                "--alm-stationarity-tol",
                str(args.alm_stationarity_tol),
                "--alm-trust-radius-init",
                str(args.alm_trust_radius_init),
                "--alm-trust-radius-min",
                str(args.alm_trust_radius_min),
                "--alm-trust-radius-shrink",
                str(args.alm_trust_radius_shrink),
                "--alm-trust-radius-grow",
                str(args.alm_trust_radius_grow),
                "--alm-max-inner-attempts",
                str(args.alm_max_inner_attempts),
                "--alm-max-subproblem-continuations",
                str(args.alm_max_subproblem_continuations),
                "--alm-distance-smoothing",
                str(args.alm_distance_smoothing),
                "--alm-curvature-smoothing",
                str(args.alm_curvature_smoothing),
            ]
        )
    if args.basin_hops > 0:
        command.extend(
            [
                "--basin-hops",
                str(args.basin_hops),
                "--basin-stepsize",
                str(args.basin_stepsize),
                "--basin-temperature",
                str(args.basin_temperature),
            ]
        )
        if args.basin_niter_success > 0:
            command.extend(["--basin-niter-success", str(args.basin_niter_success)])
        if args.basin_seed >= 0:
            command.extend(["--basin-seed", str(args.basin_seed)])
    if args.init_only:
        command.append("--init-only")
    return command


def result_metric_subset(results: dict) -> dict:
    return {
        "goal_mode": results.get("SINGLE_STAGE_GOAL_MODE"),
        "goal_mode_impl": results.get("SINGLE_STAGE_GOAL_MODE_IMPL"),
        "target_iota": results.get("TARGET_IOTA"),
        "target_volume": results.get("TARGET_VOLUME"),
        "boozer_surface_target_volumes": results.get("BOOZER_SURFACE_TARGET_VOLUMES"),
        "termination_message": results.get("TERMINATION_MESSAGE"),
        "optimizer_success": results.get("OPTIMIZER_SUCCESS"),
        "final_feasibility_ok": results.get("FINAL_FEASIBILITY_OK"),
        "hardware_constraints_ok": results.get("HARDWARE_CONSTRAINTS_OK"),
        "final_topology_gate_success": results.get("FINAL_TOPOLOGY_GATE_SUCCESS"),
        "final_iota": results.get("FINAL_IOTA"),
        "final_volume": results.get("FINAL_VOLUME"),
        "nonqs_ratio": results.get("NONQS_RATIO"),
        "boozer_residual": results.get("BOOZER_RESIDUAL"),
        "coil_length": results.get("COIL_LENGTH"),
        "max_curvature": results.get("MAX_CURVATURE"),
        "curve_curve_min_dist": results.get("CURVE_CURVE_MIN_DIST"),
        "curve_surface_min_dist": results.get("CURVE_SURFACE_MIN_DIST"),
        "surface_vessel_min_dist": results.get("SURFACE_VESSEL_MIN_DIST"),
        "invalid_state_rejects_total": results.get("INVALID_STATE_REJECTS_TOTAL"),
        "topology_gate_rejects": results.get("TOPOLOGY_GATE_REJECTS"),
        "hardware_rejects": results.get("HARDWARE_REJECTS"),
        "surface_solve_rejects": results.get("SURFACE_SOLVE_REJECTS"),
        "best_feasible_available": results.get("BEST_FEASIBLE_AVAILABLE"),
        "best_feasible_stage": results.get("BEST_FEASIBLE_STAGE"),
        "best_feasible_frontier_rank_objective_j": results.get("BEST_FEASIBLE_FRONTIER_RANK_OBJECTIVE_J"),
        "best_feasible_frontier_trust_ok": results.get("BEST_FEASIBLE_FRONTIER_TRUST_OK"),
        "best_feasible_final_iota": results.get("BEST_FEASIBLE_FINAL_IOTA"),
        "best_feasible_final_volume": results.get("BEST_FEASIBLE_FINAL_VOLUME"),
        "best_feasible_qa_objective": results.get("BEST_FEASIBLE_QA_OBJECTIVE"),
        "best_feasible_boozer_objective": results.get("BEST_FEASIBLE_BOOZER_OBJECTIVE"),
        "best_feasible_search_objective_j": results.get("BEST_FEASIBLE_SEARCH_OBJECTIVE_J"),
        "best_feasible_base_objective_j": results.get("BEST_FEASIBLE_BASE_OBJECTIVE_J"),
        "best_feasible_curve_curve_min_dist": results.get("BEST_FEASIBLE_CURVE_CURVE_MIN_DIST"),
        "best_feasible_curve_surface_min_dist": results.get("BEST_FEASIBLE_CURVE_SURFACE_MIN_DIST"),
        "best_feasible_surface_vessel_min_dist": results.get("BEST_FEASIBLE_SURFACE_VESSEL_MIN_DIST"),
        "best_feasible_max_curvature": results.get("BEST_FEASIBLE_MAX_CURVATURE"),
        "best_feasible_hardware_constraints_ok": results.get("BEST_FEASIBLE_HARDWARE_CONSTRAINTS_OK"),
        "best_feasible_final_topology_gate_success": results.get("BEST_FEASIBLE_FINAL_TOPOLOGY_GATE_SUCCESS"),
        "search_objective_j": results.get("SEARCH_OBJECTIVE_J"),
        "objective_j": results.get("OBJECTIVE_J"),
        "base_objective_j": results.get("BASE_OBJECTIVE_J"),
        "frontier_rank_objective_j": results.get("FRONTIER_RANK_OBJECTIVE_J"),
        "frontier_trust_ok": results.get("FRONTIER_TRUST_OK"),
        "frontier_boozer_trust_threshold": results.get("FRONTIER_BOOZER_TRUST_THRESHOLD"),
        "frontier_boozer_trust_excess": results.get("FRONTIER_BOOZER_TRUST_EXCESS"),
        "frontier_boozer_trust_excess_ratio": results.get("FRONTIER_BOOZER_TRUST_EXCESS_RATIO"),
        "frontier_boozer_trust_penalty_scale": results.get("FRONTIER_BOOZER_TRUST_PENALTY_SCALE"),
        "frontier_trust_penalty": results.get("FRONTIER_TRUST_PENALTY"),
        "frontier_trust_rejects": results.get("FRONTIER_TRUST_REJECTS"),
        "frontier_reference_iota": results.get("FRONTIER_REFERENCE_IOTA"),
        "frontier_reference_volume": results.get("FRONTIER_REFERENCE_VOLUME"),
        "frontier_reference_qa": results.get("FRONTIER_REFERENCE_QA"),
        "frontier_reference_boozer": results.get("FRONTIER_REFERENCE_BOOZER"),
        "frontier_effective_iota_weight": results.get("FRONTIER_EFFECTIVE_IOTA_WEIGHT"),
        "frontier_effective_volume_weight": results.get("FRONTIER_EFFECTIVE_VOLUME_WEIGHT"),
        "frontier_effective_boozer_weight": results.get("FRONTIER_EFFECTIVE_BOOZER_WEIGHT"),
        "frontier_volume_objective": results.get("FRONTIER_VOLUME_OBJECTIVE"),
        "banana_current_a": results.get("BANANA_CURRENT_A"),
        "banana_current_mode": results.get("BANANA_CURRENT_MODE"),
        "banana_currents_a": results.get("BANANA_CURRENTS_A"),
        "banana_current_max_abs_a": results.get("BANANA_CURRENT_MAX_ABS_A"),
        "banana_current_control_metric": results.get("BANANA_CURRENT_CONTROL_METRIC"),
        "best_feasible_banana_current_a": results.get("BEST_FEASIBLE_BANANA_CURRENT_A"),
        "best_feasible_banana_current_mode": results.get("BEST_FEASIBLE_BANANA_CURRENT_MODE"),
        "best_feasible_banana_currents_a": results.get("BEST_FEASIBLE_BANANA_CURRENTS_A"),
        "best_feasible_banana_current_max_abs_a": results.get(
            "BEST_FEASIBLE_BANANA_CURRENT_MAX_ABS_A"
        ),
        "best_feasible_banana_current_control_metric": results.get(
            "BEST_FEASIBLE_BANANA_CURRENT_CONTROL_METRIC"
        ),
    }


def delta(frontier_value, target_value):
    if frontier_value is None or target_value is None:
        return None
    return float(frontier_value) - float(target_value)


def build_summary(
    args: argparse.Namespace,
    commands_by_mode: dict[str, list[str]],
    *,
    stage2_bs_path: Path,
    stage2_results_path: Path | None = None,
    stage2_results: dict | None = None,
    mode_payloads: dict[str, dict] | None = None,
) -> dict:
    output_root = resolved_path(args.output_root)
    summary = {
        "plasma_surf_filename": Path(args.plasma_surf_filename).name,
        "stage2_bs_path": str(stage2_bs_path),
        "output_root": str(output_root),
        "goal_modes": list(GOAL_MODES),
        "dry_run": bool(args.dry_run),
        "search_objective_values_comparable": False,
        "mode_runs": {
            goal_mode: {
                "output_root": str(output_root / goal_mode),
                "command": commands_by_mode[goal_mode],
            }
            for goal_mode in GOAL_MODES
        },
    }
    if stage2_results_path is not None:
        summary["stage2_results_path"] = str(stage2_results_path)
    if stage2_results is not None:
        summary["stage2_artifact_plasma_surf_filename"] = stage2_results.get(
            "PLASMA_SURF_FILENAME"
        )
        summary["stage2_artifact_init_only"] = stage2_results.get("init_only")
        summary["stage2_banana_current_a"] = stage2_results.get("BANANA_CURRENT_A")
        summary["stage2_banana_current_max_a"] = stage2_results.get("BANANA_CURRENT_MAX_A")
    if mode_payloads is None:
        return summary

    for goal_mode, payload in mode_payloads.items():
        mode_entry = summary["mode_runs"][goal_mode]
        mode_entry["results_path"] = str(payload["results_path"])
        mode_entry["result_source"] = payload["result_source"]
        mode_entry["results"] = result_metric_subset(payload["results"])

    target_results = mode_payloads["target"]["results"]
    frontier_results = mode_payloads["frontier"]["results"]
    summary["comparison"] = {
        "frontier_minus_target_final_iota": delta(
            frontier_results.get("FINAL_IOTA"),
            target_results.get("FINAL_IOTA"),
        ),
        "frontier_minus_target_final_volume": delta(
            frontier_results.get("FINAL_VOLUME"),
            target_results.get("FINAL_VOLUME"),
        ),
        "frontier_minus_target_nonqs_ratio": delta(
            frontier_results.get("NONQS_RATIO"),
            target_results.get("NONQS_RATIO"),
        ),
        "frontier_minus_target_boozer_residual": delta(
            frontier_results.get("BOOZER_RESIDUAL"),
            target_results.get("BOOZER_RESIDUAL"),
        ),
        "both_final_feasibility_ok": bool(
            target_results.get("FINAL_FEASIBILITY_OK")
            and frontier_results.get("FINAL_FEASIBILITY_OK")
        ),
        "both_hardware_feasible": bool(
            target_results.get("HARDWARE_CONSTRAINTS_OK")
            and frontier_results.get("HARDWARE_CONSTRAINTS_OK")
        ),
        "both_optimizer_success": bool(
            target_results.get("OPTIMIZER_SUCCESS")
            and frontier_results.get("OPTIMIZER_SUCCESS")
        ),
    }
    return summary


def load_single_stage_results_with_salvage(
    case_output_root: Path,
    *,
    previous_results_snapshot: dict[Path, int],
    previous_preserved_snapshot: dict[Path, int],
) -> tuple[str, Path, dict]:
    result_source = "final"
    try:
        results_path = discover_single_results_path(
            case_output_root,
            previous_snapshot=previous_results_snapshot,
        )
        results = load_json(results_path)
    except (FileNotFoundError, json.JSONDecodeError):
        result_source, results_path = discover_single_stage_salvage_results_path(
            case_output_root,
            previous_snapshot=previous_preserved_snapshot,
        )
        results = load_json(results_path)
    return result_source, results_path, results


def run_goal_mode_case(
    args: argparse.Namespace,
    *,
    goal_mode: str,
    stage2_bs_path: Path,
    output_root: Path,
) -> dict:
    case_output_root = output_root / goal_mode
    case_output_root.mkdir(parents=True, exist_ok=True)
    previous_results_snapshot = snapshot_single_results_paths(case_output_root)
    previous_preserved_snapshot = snapshot_single_stage_preserved_results_paths(
        case_output_root
    )
    salvage_kwargs = {
        "case_output_root": case_output_root,
        "previous_results_snapshot": previous_results_snapshot,
        "previous_preserved_snapshot": previous_preserved_snapshot,
    }
    command = build_single_stage_goal_mode_command(
        args,
        goal_mode=goal_mode,
        stage2_bs_path=stage2_bs_path,
        case_output_root=case_output_root,
    )
    if args.dry_run:
        return {"command": command}
    timeout_seconds = timeout_or_none(args.single_stage_timeout_seconds)
    try:
        run_command(
            command,
            timeout_seconds=timeout_seconds,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as error:
        try:
            result_source, results_path, results = load_single_stage_results_with_salvage(
                **salvage_kwargs
            )
        except (FileNotFoundError, json.JSONDecodeError):
            raise error
    else:
        result_source, results_path, results = load_single_stage_results_with_salvage(
            **salvage_kwargs
        )
    return {
        "command": command,
        "results_path": results_path,
        "result_source": result_source,
        "results": results,
    }


def main() -> int:
    args = parse_args()
    output_root = resolved_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = resolved_optional_path(args.summary_json)
    if summary_path is None:
        summary_path = output_root / DEFAULT_SUMMARY_JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        stage2_bs_path, stage2_results_path, stage2_results = (
            maybe_load_validated_stage2_seed_metadata(args)
        )
    else:
        stage2_bs_path, stage2_results_path, stage2_results = (
            load_validated_stage2_seed_metadata(args)
        )

    mode_runs = {
        goal_mode: run_goal_mode_case(
            args,
            goal_mode=goal_mode,
            stage2_bs_path=stage2_bs_path,
            output_root=output_root,
        )
        for goal_mode in GOAL_MODES
    }
    commands_by_mode = {
        goal_mode: mode_runs[goal_mode]["command"] for goal_mode in GOAL_MODES
    }
    summary = build_summary(
        args,
        commands_by_mode,
        stage2_bs_path=stage2_bs_path,
        stage2_results_path=stage2_results_path,
        stage2_results=stage2_results,
        mode_payloads=None if args.dry_run else mode_runs,
    )

    with summary_path.open("w", encoding="utf-8") as outfile:
        json.dump(summary, outfile, indent=2)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
