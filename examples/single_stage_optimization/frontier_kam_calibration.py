from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from import_provenance import configure_local_simsopt_imports

configure_local_simsopt_imports(__file__)

from banana_opt.json_compat import load_boozer_finite_i as load
from frontier_pareto_trajectory import (
    bool_or_none,
    finite_float_or_none,
    root_topology_artifact_paths,
    topology_surface_for_scoring,
)
from topology_scorer import finalize_topology_score_result, score_topology


SCHEMA_VERSION = "frontier_kam_calibration_v1"
DEFAULT_OUTPUT_STEM = "frontier_kam_calibration"
DEFAULT_FRONTIER_KAM_MIN = 0.0
DEFAULT_NFIELDLINES = 12
DEFAULT_TMAX = 50.0
DEFAULT_NPHIS = 4
DEFAULT_KAM_WIDTH_RATIO = 0.25
DEFAULT_INSET_FRACTION = 0.05


@dataclass(frozen=True)
class CalibrationRow:
    donor_label: str
    run_dir: str
    biot_savart_path: str
    surface_path: str
    results_path: str
    hardware_constraints_ok: bool | None
    alm_hard_constraints_feasible: bool | None
    objective_j: float | None
    nonqs_ratio: float | None
    boozer_residual: float | None
    final_iota: float | None
    final_volume: float | None
    topology_broken: bool
    survival_fraction: float
    survived_lines: int
    nfieldlines: int
    tmax: float
    nphis: int
    kam_fraction: float
    kam_median_width: float
    kam_width_ratio: float
    cross_section_span: float
    seed_contract_json: str
    stop_reason_counts_json: str
    field_model_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate frontier KAM certification thresholds on known-good "
            "donor run directories. Each donor is scored from final root "
            "biot_savart/surface artifacts with the same Poincare scorer "
            "settings used for certification."
        )
    )
    parser.add_argument("donor_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    parser.add_argument("--nfieldlines", type=int, default=DEFAULT_NFIELDLINES)
    parser.add_argument("--tmax", type=float, default=DEFAULT_TMAX)
    parser.add_argument("--nphis", type=int, default=DEFAULT_NPHIS)
    parser.add_argument("--kam-width-ratio", type=float, default=DEFAULT_KAM_WIDTH_RATIO)
    parser.add_argument("--inset-fraction", type=float, default=DEFAULT_INSET_FRACTION)
    parser.add_argument(
        "--frontier-kam-min",
        type=float,
        default=DEFAULT_FRONTIER_KAM_MIN,
        help="Currently configured threshold to evaluate against donors.",
    )
    parser.add_argument(
        "--selection-margin",
        type=float,
        default=None,
        help=(
            "Margin below the minimum HW-clean donor KAM fraction. Defaults to "
            "one seed-line fraction, 1 / nfieldlines."
        ),
    )
    return parser.parse_args()


def read_results_payload(run_dir: Path) -> Mapping[str, object]:
    results_path = run_dir / "results.json"
    if not results_path.is_file():
        raise FileNotFoundError(f"donor results.json not found: {results_path}")
    return json.loads(results_path.read_text(encoding="utf-8"))


def donor_label(run_dir: Path) -> str:
    parent = run_dir.parent.name
    if parent in {"runs", "analysis"}:
        return run_dir.name
    return f"{parent}/{run_dir.name}"


def score_donor_run(
    run_dir: Path,
    *,
    nfieldlines: int,
    tmax: float,
    nphis: int,
    kam_width_ratio: float,
    inset_fraction: float,
) -> CalibrationRow:
    paths = root_topology_artifact_paths(run_dir, "final_results")
    if paths is None:
        raise FileNotFoundError(
            f"donor final topology artifacts not found under {run_dir}"
        )
    bs_path, surface_path = paths
    payload = read_results_payload(run_dir)
    topology_result = finalize_topology_score_result(
        score_topology(
            topology_surface_for_scoring(surface_path),
            load(bs_path),
            nfieldlines=nfieldlines,
            tmax=tmax,
            nphis=nphis,
            kam_width_ratio=kam_width_ratio,
            inset_fraction=inset_fraction,
            compute_transport_diagnostics=False,
        )
    )
    return CalibrationRow(
        donor_label=donor_label(run_dir),
        run_dir=str(run_dir),
        biot_savart_path=str(bs_path),
        surface_path=str(surface_path),
        results_path=str(run_dir / "results.json"),
        hardware_constraints_ok=bool_or_none(payload.get("HARDWARE_CONSTRAINTS_OK")),
        alm_hard_constraints_feasible=bool_or_none(
            payload.get("ALM_HARD_CONSTRAINTS_FEASIBLE")
        ),
        objective_j=finite_float_or_none(
            payload.get("SEARCH_OBJECTIVE_J", payload.get("OBJECTIVE_J"))
        ),
        nonqs_ratio=finite_float_or_none(payload.get("NONQS_RATIO")),
        boozer_residual=finite_float_or_none(payload.get("BOOZER_RESIDUAL")),
        final_iota=finite_float_or_none(payload.get("FINAL_IOTA")),
        final_volume=finite_float_or_none(payload.get("FINAL_VOLUME")),
        topology_broken=bool(topology_result["broken"]),
        survival_fraction=float(topology_result["survival_fraction"]),
        survived_lines=int(topology_result["survived_lines"]),
        nfieldlines=int(topology_result["nfieldlines"]),
        tmax=float(topology_result["tmax"]),
        nphis=int(nphis),
        kam_fraction=float(topology_result["kam_fraction"]),
        kam_median_width=float(topology_result["kam_median_width"]),
        kam_width_ratio=float(topology_result["kam_width_ratio"]),
        cross_section_span=float(topology_result["cross_section_span"]),
        seed_contract_json=json.dumps(
            topology_result["seed_contract"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        stop_reason_counts_json=json.dumps(
            topology_result["stop_reason_counts"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        field_model_json=json.dumps(
            topology_result["field_model"],
            sort_keys=True,
            separators=(",", ":"),
        ),
    )


def hw_clean_kam_values(rows: Iterable[CalibrationRow]) -> list[float]:
    return [
        row.kam_fraction
        for row in rows
        if row.hardware_constraints_ok is True and not row.topology_broken
    ]


def calibration_summary(
    rows: list[CalibrationRow],
    *,
    frontier_kam_min: float,
    selection_margin: float,
) -> dict[str, object]:
    kam_values = hw_clean_kam_values(rows)
    rejecting_labels = [
        row.donor_label
        for row in rows
        if (
            row.hardware_constraints_ok is True
            and not row.topology_broken
            and row.kam_fraction < frontier_kam_min
        )
    ]
    recommended = None
    if kam_values:
        recommended = max(0.0, min(kam_values) - float(selection_margin))
    return {
        "configured_frontier_kam_min": float(frontier_kam_min),
        "selection_margin": float(selection_margin),
        "hw_clean_donor_count": len(kam_values),
        "kam_fraction_min": None if not kam_values else min(kam_values),
        "kam_fraction_median": None if not kam_values else statistics.median(kam_values),
        "kam_fraction_max": None if not kam_values else max(kam_values),
        "configured_threshold_rejecting_hw_clean_donors": rejecting_labels,
        "configured_threshold_accepts_all_hw_clean_donors": not rejecting_labels,
        "recommended_frontier_kam_min": recommended,
    }


def calibration_payload(
    *,
    rows: list[CalibrationRow],
    run_dirs: list[Path],
    nfieldlines: int,
    tmax: float,
    nphis: int,
    kam_width_ratio: float,
    inset_fraction: float,
    frontier_kam_min: float,
    selection_margin: float,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "donor_dirs": [str(path) for path in run_dirs],
        "scorer_settings": {
            "nfieldlines": int(nfieldlines),
            "tmax": float(tmax),
            "nphis": int(nphis),
            "kam_width_ratio": float(kam_width_ratio),
            "inset_fraction": float(inset_fraction),
            "compute_transport_diagnostics": False,
        },
        "summary": calibration_summary(
            rows,
            frontier_kam_min=frontier_kam_min,
            selection_margin=selection_margin,
        ),
        "rows": [asdict(row) for row in rows],
    }


def write_csv(path: Path, rows: list[CalibrationRow]) -> None:
    fieldnames = list(CalibrationRow.__dataclass_fields__)
    with path.open("w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def main() -> None:
    args = parse_args()
    if args.nfieldlines <= 0:
        raise ValueError("--nfieldlines must be positive")
    if args.nphis <= 0:
        raise ValueError("--nphis must be positive")
    if not math.isfinite(args.tmax) or args.tmax <= 0.0:
        raise ValueError("--tmax must be finite and positive")
    if not math.isfinite(args.kam_width_ratio) or args.kam_width_ratio <= 0.0:
        raise ValueError("--kam-width-ratio must be finite and positive")
    if not math.isfinite(args.inset_fraction) or args.inset_fraction < 0.0:
        raise ValueError("--inset-fraction must be finite and non-negative")
    if not math.isfinite(args.frontier_kam_min) or not (
        0.0 <= args.frontier_kam_min <= 1.0
    ):
        raise ValueError("--frontier-kam-min must be finite and in [0, 1]")

    selection_margin = (
        1.0 / int(args.nfieldlines)
        if args.selection_margin is None
        else float(args.selection_margin)
    )
    if not math.isfinite(selection_margin) or selection_margin < 0.0:
        raise ValueError("--selection-margin must be finite and non-negative")

    run_dirs = [path.resolve() for path in args.donor_dirs]
    rows = [
        score_donor_run(
            run_dir,
            nfieldlines=int(args.nfieldlines),
            tmax=float(args.tmax),
            nphis=int(args.nphis),
            kam_width_ratio=float(args.kam_width_ratio),
            inset_fraction=float(args.inset_fraction),
        )
        for run_dir in run_dirs
    ]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.output_stem}.json"
    csv_path = output_dir / f"{args.output_stem}.csv"
    json_path.write_text(
        json.dumps(
            calibration_payload(
                rows=rows,
                run_dirs=run_dirs,
                nfieldlines=int(args.nfieldlines),
                tmax=float(args.tmax),
                nphis=int(args.nphis),
                kam_width_ratio=float(args.kam_width_ratio),
                inset_fraction=float(args.inset_fraction),
                frontier_kam_min=float(args.frontier_kam_min),
                selection_margin=selection_margin,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    write_csv(csv_path, rows)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "rows": len(rows),
                "json": str(json_path),
                "csv": str(csv_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
