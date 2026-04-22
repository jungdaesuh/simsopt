from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from simsopt._core.optimizable import load

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from banana_opt.stage2_single_stage_handoff import partition_loaded_stage2_coils  # noqa: E402
from banana_opt.artifact_contracts import upgrade_legacy_stage2_artifact_results  # noqa: E402
from run_banana_current_scan import materialize_stage2_seed_variant_from_currents  # noqa: E402
from workflow_runner_common import (  # noqa: E402
    load_stage2_artifact_results,
    parse_csv,
    resolved_optional_path,
    resolved_path,
    write_json,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_perturbed_banana_seed"
DEFAULT_SUMMARY_JSON = "perturbed_banana_seed_summary.json"


def build_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize a Stage 2 donor variant with independent per-coil banana "
            "current seeds while preserving the sibling results.json handoff "
            "contract."
        ),
        add_help=add_help,
    )
    parser.add_argument(
        "--stage2-bs-path",
        required=True,
        help="Path to the donor Stage 2 biot_savart_opt.json artifact.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory that will receive the perturbed Stage 2 variant bundle.",
    )
    parser.add_argument(
        "--summary-json",
        default=None,
        help=f"Optional summary path. Defaults to <output-root>/{DEFAULT_SUMMARY_JSON}.",
    )
    parser.add_argument(
        "--banana-currents-A",
        dest="banana_currents_a",
        default=None,
        help=(
            "Comma-separated explicit banana-current vector in physical amperes. "
            "Length must match the donor banana-coil count."
        ),
    )
    parser.add_argument(
        "--relative-perturbation-max",
        type=float,
        default=None,
        help=(
            "Uniform relative perturbation bound applied independently to each "
            "donor banana current. For example 0.05 samples each coil in "
            "[-5%%, +5%%] around its donor current."
        ),
    )
    parser.add_argument(
        "--perturbation-seed",
        type=int,
        default=42,
        help="Random seed used with --relative-perturbation-max (default 42).",
    )
    parser.add_argument(
        "--num-tf-coils",
        type=int,
        default=None,
        help=(
            "Optional TF-coil count override for legacy Stage 2 artifacts that do "
            "not record NUM_TF_COILS."
        ),
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def _validated_requested_num_tf_coils(args: argparse.Namespace, stage2_results: dict) -> int:
    if args.num_tf_coils is not None:
        return int(args.num_tf_coils)
    recorded_num_tf_coils = stage2_results.get("NUM_TF_COILS")
    if recorded_num_tf_coils is None:
        raise ValueError(
            "Perturbed banana seed generation requires NUM_TF_COILS in the donor "
            "results.json or an explicit --num-tf-coils override."
        )
    return int(recorded_num_tf_coils)


def _load_validated_stage2_seed(
    args: argparse.Namespace,
) -> tuple[Path, Path, dict, int]:
    stage2_bs_path = resolved_path(args.stage2_bs_path)
    stage2_results_path, stage2_results = load_stage2_artifact_results(stage2_bs_path)
    stage2_results = upgrade_legacy_stage2_artifact_results(
        stage2_results,
        known_num_tf_coils=args.num_tf_coils,
    )
    requested_num_tf_coils = _validated_requested_num_tf_coils(args, stage2_results)
    return stage2_bs_path, stage2_results_path, stage2_results, requested_num_tf_coils


def _load_donor_banana_currents_a(
    *,
    stage2_bs_path: Path,
    stage2_results: dict,
    requested_num_tf_coils: int,
) -> list[float]:
    bs = load(str(stage2_bs_path))
    coil_partitions = partition_loaded_stage2_coils(
        bs.coils,
        stage2_results=stage2_results,
        requested_num_tf_coils=requested_num_tf_coils,
    )
    return [
        float(coil.current.get_value())
        for coil in coil_partitions.banana_coils
    ]


def _effective_perturbation_seed(
    *,
    perturbation_mode: str,
    perturbation_seed: int,
) -> int | None:
    if perturbation_mode == "explicit_vector":
        return None
    return int(perturbation_seed)


def _resolved_perturbation_mode(args: argparse.Namespace) -> str:
    has_explicit_vector = args.banana_currents_a not in {None, ""}
    has_random_relative = args.relative_perturbation_max is not None
    if has_explicit_vector == has_random_relative:
        raise ValueError(
            "Specify exactly one of --banana-currents-A or "
            "--relative-perturbation-max."
        )
    if has_explicit_vector:
        return "explicit_vector"
    return "uniform_relative_random"


def _resolved_banana_currents_a(
    args: argparse.Namespace,
    *,
    donor_banana_currents_a: list[float],
) -> tuple[str, list[float]]:
    perturbation_mode = _resolved_perturbation_mode(args)
    if perturbation_mode == "explicit_vector":
        return perturbation_mode, parse_csv(str(args.banana_currents_a), float)
    relative_perturbation_max = float(args.relative_perturbation_max)
    if relative_perturbation_max < 0.0:
        raise ValueError("--relative-perturbation-max must be non-negative.")
    rng = np.random.default_rng(args.perturbation_seed)
    perturbations = rng.uniform(
        -relative_perturbation_max,
        relative_perturbation_max,
        size=len(donor_banana_currents_a),
    )
    return (
        perturbation_mode,
        [
            float(donor_current_A) * (1.0 + float(perturbation))
            for donor_current_A, perturbation in zip(
                donor_banana_currents_a,
                perturbations,
                strict=True,
            )
        ],
    )


def _recommended_single_stage_flags(variant_bs_path: Path) -> list[str]:
    return [
        "--stage2-bs-path",
        str(variant_bs_path),
        "--single-stage-banana-current-mode",
        "independent",
    ]


def _variant_results_updates(
    *,
    stage2_bs_path: Path,
    stage2_results_path: Path,
    perturbation_mode: str,
    relative_perturbation_max: float | None,
    perturbation_seed: int,
) -> dict[str, object]:
    return {
        "DONOR_STAGE2_BS_PATH": str(stage2_bs_path),
        "DONOR_STAGE2_RESULTS_PATH": str(stage2_results_path),
        "PERTURBATION_MODE": perturbation_mode,
        "RELATIVE_PERTURBATION_MAX": relative_perturbation_max,
        "PERTURBATION_SEED": _effective_perturbation_seed(
            perturbation_mode=perturbation_mode,
            perturbation_seed=perturbation_seed,
        ),
    }


def _resolved_summary_path(
    *,
    output_root: Path,
    summary_json: str | None,
) -> Path:
    summary_path = resolved_optional_path(summary_json)
    if summary_path is None:
        summary_path = output_root / DEFAULT_SUMMARY_JSON
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    return summary_path


def build_summary(
    *,
    stage2_bs_path: Path,
    stage2_results_path: Path,
    donor_init_only: bool,
    requested_num_tf_coils: int,
    output_root: Path,
    variant_bs_path: Path,
    variant_results_path: Path,
    donor_banana_currents_a: list[float],
    banana_currents_a: list[float],
    perturbation_mode: str,
    relative_perturbation_max: float | None,
    perturbation_seed: int | None,
) -> dict[str, object]:
    compatibility_current_A = max(abs(current_A) for current_A in banana_currents_a)
    return {
        "experiment_family": "perturbed_banana_seed",
        "output_root": str(output_root),
        "stage2_bs_path": str(stage2_bs_path),
        "stage2_results_path": str(stage2_results_path),
        "donor_init_only": bool(donor_init_only),
        "requested_num_tf_coils": int(requested_num_tf_coils),
        "variant_bs_path": str(variant_bs_path),
        "variant_results_path": str(variant_results_path),
        "banana_current_mode": "independent",
        "donor_banana_currents_a": [float(value) for value in donor_banana_currents_a],
        "perturbed_banana_currents_a": [float(value) for value in banana_currents_a],
        "compatibility_banana_current_a": float(compatibility_current_A),
        "perturbation_mode": perturbation_mode,
        "relative_perturbation_max": (
            None
            if relative_perturbation_max is None
            else float(relative_perturbation_max)
        ),
        "perturbation_seed": _effective_perturbation_seed(
            perturbation_mode=perturbation_mode,
            perturbation_seed=int(perturbation_seed),
        ),
        "recommended_single_stage_flags": _recommended_single_stage_flags(
            variant_bs_path
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = resolved_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = _resolved_summary_path(
        output_root=output_root,
        summary_json=args.summary_json,
    )

    (
        stage2_bs_path,
        stage2_results_path,
        stage2_results,
        requested_num_tf_coils,
    ) = _load_validated_stage2_seed(args)
    donor_banana_currents_a = _load_donor_banana_currents_a(
        stage2_bs_path=stage2_bs_path,
        stage2_results=stage2_results,
        requested_num_tf_coils=requested_num_tf_coils,
    )
    perturbation_mode, banana_currents_a = _resolved_banana_currents_a(
        args,
        donor_banana_currents_a=donor_banana_currents_a,
    )
    variant_bs_path, variant_results_path = (
        materialize_stage2_seed_variant_from_currents(
            stage2_bs_path=stage2_bs_path,
            stage2_results=stage2_results,
            variant_root=output_root,
            banana_currents_a=banana_currents_a,
            requested_num_tf_coils=requested_num_tf_coils,
            extra_results_updates=_variant_results_updates(
                stage2_bs_path=stage2_bs_path,
                stage2_results_path=stage2_results_path,
                perturbation_mode=perturbation_mode,
                relative_perturbation_max=args.relative_perturbation_max,
                perturbation_seed=args.perturbation_seed,
            ),
        )
    )
    summary = build_summary(
        stage2_bs_path=stage2_bs_path,
        stage2_results_path=stage2_results_path,
        donor_init_only=bool(stage2_results.get("init_only", False)),
        requested_num_tf_coils=requested_num_tf_coils,
        output_root=output_root,
        variant_bs_path=variant_bs_path,
        variant_results_path=variant_results_path,
        donor_banana_currents_a=donor_banana_currents_a,
        banana_currents_a=[float(value) for value in banana_currents_a],
        perturbation_mode=perturbation_mode,
        relative_perturbation_max=args.relative_perturbation_max,
        perturbation_seed=args.perturbation_seed,
    )
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
