from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from import_provenance import configure_local_simsopt_imports

EXAMPLE_ROOT, SIMSOPT_ROOT, SRC_ROOT = configure_local_simsopt_imports(__file__)

from simsopt._core.optimizable import load

from banana_opt.topology_fidelity_ladder import (
    DEFAULT_TOPOLOGY_TIER_SPECS,
    build_topology_fidelity_report,
    topology_tier_passed,
)
from topology_scorer import safe_score_topology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate cheap / medium / strict topology fidelity settings on one "
            "or more single-stage output directories and write a JSON report. "
            "All three settings use the shared midplane radial sweep seed contract."
        )
    )
    parser.add_argument(
        "output_dirs",
        nargs="+",
        help=(
            "Single-stage output directories containing matching init or opt "
            "biot_savart / surface JSON artifacts."
        ),
    )
    parser.add_argument(
        "--report-path",
        default="topology_fidelity_ladder_report.json",
        help="Path to the JSON report to write.",
    )
    return parser.parse_args()


def resolve_field_and_surface(output_dir: str | Path) -> tuple[object, object, str]:
    output_path = Path(output_dir).resolve()
    opt_bs_path = output_path / "biot_savart_opt.json"
    opt_surf_path = output_path / "surf_opt.json"
    init_bs_path = output_path / "biot_savart_init.json"
    init_surf_path = output_path / "surf_init.json"

    if opt_bs_path.exists() and opt_surf_path.exists():
        return load(opt_bs_path), load(opt_surf_path), "opt"
    if init_bs_path.exists() and init_surf_path.exists():
        return load(init_bs_path), load(init_surf_path), "init"
    raise FileNotFoundError(
        "Could not find matching opt or init topology artifacts in "
        f"{output_path}"
    )


def build_tier_case_record(
    result: dict[str, object],
    *,
    survival_threshold: float,
) -> dict[str, object]:
    return {
        "passed": topology_tier_passed(
            result,
            survival_threshold=survival_threshold,
        ),
        "survival_fraction": float(result["survival_fraction"]),
        "confinement_score": float(result["confinement_score"]),
        "broken": bool(result["broken"]),
        "evaluation_state": result["evaluation_state"],
        "evaluation_error": result.get("evaluation_error"),
        "evaluation_error_type": result.get("evaluation_error_type"),
        "seed_contract": result.get("seed_contract"),
        "field_model": result.get("field_model"),
    }


def evaluate_case(output_dir: str | Path) -> dict[str, object]:
    bfield, surface, field_label = resolve_field_and_surface(output_dir)
    case_record: dict[str, object] = {
        "label": Path(output_dir).resolve().name,
        "output_dir": str(Path(output_dir).resolve()),
        "field_label": field_label,
    }
    for tier_name, tier_spec in DEFAULT_TOPOLOGY_TIER_SPECS.items():
        result = safe_score_topology(
            surface,
            bfield,
            nfieldlines=tier_spec.nfieldlines,
            tmax=tier_spec.tmax,
            nphis=tier_spec.nphis,
            inset_fraction=tier_spec.inset_fraction,
            field_policy=tier_spec.field_policy,
        )
        case_record[tier_name] = build_tier_case_record(
            result,
            survival_threshold=tier_spec.survival_threshold,
        )
    return case_record


def main() -> None:
    args = parse_args()
    case_records = [evaluate_case(output_dir) for output_dir in args.output_dirs]
    report = build_topology_fidelity_report(case_records)
    report_path = Path(args.report_path).resolve()
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote topology fidelity ladder report to {report_path}")


if __name__ == "__main__":
    main()
