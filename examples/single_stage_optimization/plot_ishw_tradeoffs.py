from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from workflow_runner_common import (  # noqa: E402
    load_json,
    parse_csv,
    resolved_optional_path,
    resolved_path,
    run_poincare_artifact,
    timeout_or_none,
    write_json,
)

DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs_ishw_tradeoffs"
DEFAULT_MANIFEST_JSON = "ishw_plot_manifest.json"
DEFAULT_PLOT_FORMATS = "png"
STATUS_ORDER = {
    "boozer_failed": 0,
    "poincare_only_fallback": 1,
    "success": 2,
}
ERROR_METRIC_LABELS = {
    "nonqs_ratio": "Non-QS Ratio [-]",
    "field_error": "Field Error [-]",
}


def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return None
    return numeric_value


def _resolved_error_metric_key(rows: list[dict]) -> str:
    if any(_finite_float_or_none(row.get("nonqs_ratio")) is not None for row in rows):
        return "nonqs_ratio"
    return "field_error"


def _resolved_error_metric_spec(rows: list[dict]) -> tuple[str, str]:
    key = _resolved_error_metric_key(rows)
    return key, ERROR_METRIC_LABELS[key]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate slide-ready ISHW tradeoff plots from the iota sweep and "
            "banana-current scan summaries."
        )
    )
    parser.add_argument("--iota-sweep-summary", default=None)
    parser.add_argument("--banana-current-scan-summary", default=None)
    parser.add_argument(
        "--field-error-coil-length-path",
        default=None,
        help=(
            "Optional CSV or JSON file with external field-error vs coil-length rows. "
            "Falls back to the iota sweep summary when omitted."
        ),
    )
    parser.add_argument(
        "--reference-poincare-dir",
        default=None,
        help=(
            "Optional single-stage or fallback Poincare directory to rerun and copy "
            "into the slide output root."
        ),
    )
    parser.add_argument(
        "--poincare-timeout-seconds",
        type=float,
        default=0.0,
        help="Optional timeout for the reference Poincare subprocess.",
    )
    parser.add_argument(
        "--plot-formats",
        default=DEFAULT_PLOT_FORMATS,
        help="Comma-separated output formats, for example png or png,pdf.",
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--manifest-json",
        default=None,
        help=f"Optional manifest path. Defaults to <output-root>/{DEFAULT_MANIFEST_JSON}.",
    )
    return parser.parse_args(argv)


def _save_figure(
    figure: plt.Figure,
    *,
    output_root: Path,
    stem: str,
    formats: list[str],
) -> list[str]:
    written_paths: list[str] = []
    for plot_format in formats:
        path = output_root / f"{stem}.{plot_format}"
        figure.savefig(path, dpi=300, bbox_inches="tight")
        written_paths.append(str(path))
    plt.close(figure)
    return written_paths


def _completed_iota_rows(summary: dict | None) -> list[dict]:
    if summary is None:
        return []
    rows = []
    for case in summary.get("cases", []):
        if case.get("status") != "completed":
            continue
        results_summary = case.get("results_summary")
        if results_summary is None:
            continue
        rows.append(
            {
                "label": case["case_id"],
                "iota_target": float(case["iota_target"]),
                **results_summary,
            }
        )
    return rows


def _banana_scan_rows(summary: dict | None) -> list[dict]:
    if summary is None:
        return []
    rows = []
    for case in summary.get("cases", []):
        row = {
            "label": case["case_id"],
            "banana_current_scale": case["banana_current_scale"],
            "banana_current_a": case["banana_current_a"],
            "classification": case["classification"],
            "poincare_status": case.get("poincare_status"),
        }
        if case.get("results_summary") is not None:
            row.update(case["results_summary"])
        rows.append(row)
    return rows


def _plot_xy(
    rows: list[dict],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_root: Path,
    stem: str,
    formats: list[str],
    annotate_label: bool = False,
) -> list[str]:
    points: list[tuple[float, float, object]] = []
    for row in rows:
        x_value = _finite_float_or_none(row.get(x_key))
        y_value = _finite_float_or_none(row.get(y_key))
        if x_value is None or y_value is None:
            continue
        points.append((x_value, y_value, row.get("label")))
    if not points:
        return []
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.plot(
        [point[0] for point in points],
        [point[1] for point in points],
        marker="o",
        linewidth=1.5,
    )
    if annotate_label:
        for x_value, y_value, label in points:
            axis.annotate(str(label), (x_value, y_value), textcoords="offset points", xytext=(5, 5), fontsize=8)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, linewidth=0.4)
    return _save_figure(
        figure,
        output_root=output_root,
        stem=stem,
        formats=formats,
    )


def _plot_banana_success_status(
    rows: list[dict],
    *,
    output_root: Path,
    formats: list[str],
) -> list[str]:
    points = [
        (
            float(row["banana_current_scale"]),
            STATUS_ORDER[row["classification"]],
        )
        for row in rows
        if row.get("banana_current_scale") is not None
        and row.get("classification") in STATUS_ORDER
    ]
    if not points:
        return []
    figure, axis = plt.subplots(figsize=(6.5, 4.5))
    axis.scatter(
        [point[0] for point in points],
        [point[1] for point in points],
        s=48,
    )
    axis.set_xlabel("Banana Current Scale [-]")
    axis.set_ylabel("Startup Outcome")
    axis.set_title("Banana Current Scale vs Boozer Startup Outcome")
    axis.set_yticks([STATUS_ORDER[key] for key in ("boozer_failed", "poincare_only_fallback", "success")])
    axis.set_yticklabels(["boozer_failed", "poincare_only_fallback", "success"])
    axis.grid(True, linewidth=0.4)
    return _save_figure(
        figure,
        output_root=output_root,
        stem="banana_current_scale_vs_startup_outcome",
        formats=formats,
    )


def _load_field_error_coil_length_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as infile:
            loaded = json.load(infile)
        if isinstance(loaded, dict):
            rows = loaded.get("rows", [])
        else:
            rows = loaded
        return [dict(row) for row in rows]
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            rows.append(dict(row))
    return rows


def _external_or_fallback_field_error_rows(
    *,
    field_error_coil_length_path: Path | None,
    iota_rows: list[dict],
) -> list[dict]:
    if field_error_coil_length_path is not None:
        return _load_field_error_coil_length_rows(field_error_coil_length_path)
    return [
        {
            "label": row.get("label"),
            "coil_length": row.get("coil_length"),
            "field_error": row.get("field_error"),
        }
        for row in iota_rows
    ]


def _export_reference_poincare(
    *,
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, object] | None:
    if args.reference_poincare_dir is None:
        return None
    source_dir = resolved_path(args.reference_poincare_dir)
    timeout_seconds = timeout_or_none(args.poincare_timeout_seconds)
    command = run_poincare_artifact(
        output_dir=source_dir,
        python_executable=args.python_executable,
        timeout_seconds=timeout_seconds,
    )
    export_dir = output_root / "reference_poincare"
    export_dir.mkdir(parents=True, exist_ok=True)
    copied_paths: list[str] = []
    for pattern in ("PoincarePlot_*.png", "PoincareMetrics_*.json"):
        for path in sorted(source_dir.glob(pattern)):
            target = export_dir / path.name
            shutil.copy2(path, target)
            copied_paths.append(str(target))
    return {
        "source_dir": str(source_dir),
        "command": command,
        "copied_paths": copied_paths,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = resolved_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = resolved_optional_path(args.manifest_json)
    if manifest_path is None:
        manifest_path = output_root / DEFAULT_MANIFEST_JSON

    iota_summary = (
        None
        if args.iota_sweep_summary is None
        else load_json(resolved_path(args.iota_sweep_summary))
    )
    banana_summary = (
        None
        if args.banana_current_scan_summary is None
        else load_json(resolved_path(args.banana_current_scan_summary))
    )
    if iota_summary is None and banana_summary is None:
        raise ValueError(
            "Provide at least one of --iota-sweep-summary or --banana-current-scan-summary."
        )

    plot_formats = parse_csv(args.plot_formats, str)
    iota_rows = _completed_iota_rows(iota_summary)
    banana_rows = _banana_scan_rows(banana_summary)
    external_field_error_path = resolved_optional_path(args.field_error_coil_length_path)
    generated_plots: dict[str, list[str]] = {}

    iota_error_key, iota_error_label = _resolved_error_metric_spec(iota_rows)
    generated_plots["iota_target_vs_coil_length"] = _plot_xy(
        iota_rows,
        x_key="iota_target",
        y_key="coil_length",
        xlabel="Target Iota [-]",
        ylabel="Coil Length [m]",
        title="Target Iota vs Coil Length",
        output_root=output_root,
        stem="iota_target_vs_coil_length",
        formats=plot_formats,
    )
    generated_plots["iota_target_vs_max_curvature"] = _plot_xy(
        iota_rows,
        x_key="iota_target",
        y_key="max_curvature",
        xlabel="Target Iota [-]",
        ylabel="Max Curvature [1/m]",
        title="Target Iota vs Max Curvature",
        output_root=output_root,
        stem="iota_target_vs_max_curvature",
        formats=plot_formats,
    )
    generated_plots["iota_target_vs_qs_proxy"] = _plot_xy(
        iota_rows,
        x_key="iota_target",
        y_key=iota_error_key,
        xlabel="Target Iota [-]",
        ylabel=iota_error_label,
        title="Target Iota vs QS/Error Metric",
        output_root=output_root,
        stem="iota_target_vs_qs_proxy",
        formats=plot_formats,
    )

    banana_error_key, banana_error_label = _resolved_error_metric_spec(banana_rows)
    generated_plots["banana_current_scale_vs_qs_proxy"] = _plot_xy(
        banana_rows,
        x_key="banana_current_scale",
        y_key=banana_error_key,
        xlabel="Banana Current Scale [-]",
        ylabel=banana_error_label,
        title="Banana Current Scale vs QS/Error Metric",
        output_root=output_root,
        stem="banana_current_scale_vs_qs_proxy",
        formats=plot_formats,
    )
    generated_plots["banana_current_scale_vs_iota"] = _plot_xy(
        banana_rows,
        x_key="banana_current_scale",
        y_key="final_iota",
        xlabel="Banana Current Scale [-]",
        ylabel="Final Iota [-]",
        title="Banana Current Scale vs Final Iota",
        output_root=output_root,
        stem="banana_current_scale_vs_iota",
        formats=plot_formats,
    )
    generated_plots["banana_current_scale_vs_startup_outcome"] = (
        _plot_banana_success_status(
            banana_rows,
            output_root=output_root,
            formats=plot_formats,
        )
    )

    field_error_rows = _external_or_fallback_field_error_rows(
        field_error_coil_length_path=external_field_error_path,
        iota_rows=iota_rows,
    )
    generated_plots["field_error_vs_coil_length"] = _plot_xy(
        field_error_rows,
        x_key="coil_length",
        y_key="field_error",
        xlabel="Coil Length [m]",
        ylabel="Field Error [-]",
        title="Field Error vs Coil Length",
        output_root=output_root,
        stem="field_error_vs_coil_length",
        formats=plot_formats,
        annotate_label=True,
    )

    reference_poincare = _export_reference_poincare(args=args, output_root=output_root)
    manifest = {
        "output_root": str(output_root),
        "plot_formats": plot_formats,
        "iota_sweep_summary": (
            None if args.iota_sweep_summary is None else str(resolved_path(args.iota_sweep_summary))
        ),
        "banana_current_scan_summary": (
            None
            if args.banana_current_scan_summary is None
            else str(resolved_path(args.banana_current_scan_summary))
        ),
        "field_error_coil_length_path": (
            None
            if external_field_error_path is None
            else str(external_field_error_path)
        ),
        "generated_plots": generated_plots,
        "reference_poincare": reference_poincare,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
