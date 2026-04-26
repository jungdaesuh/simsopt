from __future__ import annotations

import argparse
import csv
import json
import platform
import sqlite3
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SCHEMA_VERSION = "alm_autoresearch_benchmark_v1"
PRE_NORMALIZATION_CONTRACT_VERSION = "pre_normalization"
BENCHMARK_DIR_NAME = "alm_normalization_benchmarks"
DEFAULT_AUTORESEARCH_ROOT = Path("/Users/suhjungdae/code/columbia/autoresearch")

_ALM_PARAM_KEYS = (
    "alm",
    "alm_enabled",
    "alm_outer_iters",
    "alm_formulation",
    "alm_penalty_init",
    "alm_penalty_scale",
    "alm_feas_tol",
    "alm_stationarity_tol",
    "alm_qs_threshold",
    "alm_boozer_threshold",
    "alm_iota_penalty_threshold",
    "alm_length_penalty_threshold",
    "alm_penalty_max",
)

_METRIC_CANDIDATES: Mapping[str, tuple[str, ...]] = {
    "hard_feasible_success": (
        "hard_feasible_success",
        "HARDWARE_PASS",
        "hardware_pass",
        "promotion_ready",
    ),
    "normalized_alm_feasible_success": (
        "normalized_alm_feasible_success",
        "ALM_FINAL_NORMALIZED_FEASIBLE",
    ),
    "restored_best_feasible": (
        "restored_best_feasible",
        "ALM_RESTORED_BEST_FEASIBLE",
        "restored_best_feasible_incumbent",
    ),
    "max_raw_hard_violation": (
        "max_raw_hard_violation",
        "ALM_FINAL_MAX_RAW_HARD_VIOLATION",
        "max_hardware_violation",
    ),
    "max_normalized_violation": (
        "max_normalized_violation",
        "ALM_FINAL_MAX_NORMALIZED_VIOLATION",
        "max_feasibility_violation",
    ),
    "best_feasible_base_objective": (
        "best_feasible_base_objective",
        "ALM_BEST_FEASIBLE_OBJECTIVE",
        "best_feasible_objective",
    ),
    "final_base_objective": (
        "final_base_objective",
        "objective_J",
        "field_error",
        "J",
    ),
    "outer_iterations": (
        "outer_iterations",
        "ALM_OUTER_ITERATIONS",
        "iterations",
    ),
    "objective_eval_count": (
        "objective_eval_count",
        "ALM_OBJECTIVE_EVALS",
        "nfev",
    ),
    "wall_time_s": (
        "wall_time_s",
        "elapsed",
        "runtime_s",
    ),
    "penalty_cap_hit_count": (
        "penalty_cap_hit_count",
        "ALM_PENALTY_CAP_HIT_COUNT",
    ),
    "multiplier_cap_hit_count": (
        "multiplier_cap_hit_count",
        "ALM_MULTIPLIER_CAP_HIT_COUNT",
    ),
    "blocking_constraint_name": (
        "blocking_constraint_name",
        "ALM_BLOCKING_CONSTRAINT_NAME",
    ),
    "blocking_constraint_block": (
        "blocking_constraint_block",
        "ALM_BLOCKING_CONSTRAINT_BLOCK",
    ),
}

_SUMMARY_COLUMNS = (
    "case",
    "before_success",
    "after_success",
    "before_best_feasible_objective",
    "after_best_feasible_objective",
    "before_max_raw_hard_violation",
    "after_max_raw_hard_violation",
    "before_max_normalized_violation",
    "after_max_normalized_violation",
    "before_outer_iters",
    "after_outer_iters",
    "before_evals",
    "after_evals",
    "before_wall_s",
    "after_wall_s",
    "before_penalty_cap_hits",
    "after_penalty_cap_hits",
    "before_multiplier_cap_hits",
    "after_multiplier_cap_hits",
    "blocking_constraint_before",
    "blocking_constraint_after",
)


@dataclass(frozen=True)
class AutoresearchArtifactRoots:
    registry_dir: Path
    runs_dir: Path
    ledger_path: Path
    artifact_exports_dir: Path
    harvested_seeds_dir: Path

    @classmethod
    def from_autoresearch_root(cls, autoresearch_root: Path) -> "AutoresearchArtifactRoots":
        return cls(
            registry_dir=autoresearch_root / "registry",
            runs_dir=autoresearch_root / "runs",
            ledger_path=autoresearch_root / "results_surrogate_legacy_vmec.jsonl",
            artifact_exports_dir=autoresearch_root / "artifact_exports",
            harvested_seeds_dir=autoresearch_root / "harvested_seeds",
        )


def benchmark_output_dir(roots: AutoresearchArtifactRoots) -> Path:
    return roots.artifact_exports_dir / BENCHMARK_DIR_NAME


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _read_json(path: Path) -> Mapping[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_mapping_or_skip(path: Path) -> tuple[Mapping[str, object] | None, dict[str, object] | None]:
    try:
        payload = _read_json(path)
    except json.JSONDecodeError as error:
        return None, {
            "source_path": str(path),
            "skip_reason": "invalid_json",
            "error": str(error),
        }
    if not isinstance(payload, Mapping):
        return None, {
            "source_path": str(path),
            "skip_reason": "json_root_not_object",
            "error": type(payload).__name__,
        }
    return payload, None


def _iter_jsonl(path: Path) -> Iterable[tuple[int, Mapping[str, object]]]:
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            stripped = line.strip()
            if stripped:
                yield row_index, json.loads(stripped)


def _flatten_mapping(
    payload: Mapping[str, object],
    *,
    prefix: str = "",
) -> dict[str, object]:
    flattened: dict[str, object] = {}
    for key, value in payload.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        flattened[full_key] = value
        if isinstance(value, Mapping):
            flattened.update(_flatten_mapping(value, prefix=full_key))
    return flattened


def _first_present(
    payload: Mapping[str, object],
    keys: Sequence[str],
) -> object:
    flattened = _flatten_mapping(payload)
    for key in keys:
        if key in flattened:
            return flattened[key]
    for key in keys:
        suffix = f".{key}"
        for flat_key, value in flattened.items():
            if flat_key.endswith(suffix):
                return value
    return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "alm"}
    return value is not None


def alm_relevance_reason(payload: Mapping[str, object]) -> str | None:
    flattened = _flatten_mapping(payload)
    for key in ("constraint_method", "params.constraint_method"):
        if str(flattened.get(key, "")).strip().lower() == "alm":
            return key
    for key in _ALM_PARAM_KEYS:
        value = flattened.get(key)
        if _truthy(value):
            return key
        value = flattened.get(f"params.{key}")
        if _truthy(value):
            return f"params.{key}"
    for key in flattened:
        key_lower = key.lower()
        if key_lower.startswith("alm_") or ".alm_" in key_lower:
            return key
    return None


def _normalization_fields_available(payload: Mapping[str, object]) -> bool:
    flattened = _flatten_mapping(payload)
    normalized_markers = {
        "ALM_FINAL_NORMALIZED_CONSTRAINT_VALUES",
        "ALM_FINAL_MAX_NORMALIZED_VIOLATION",
        "normalized_signed_constraint_values",
        "normalized_feasibility_values",
    }
    return bool(normalized_markers.intersection(flattened))


def _extract_metrics(payload: Mapping[str, object]) -> dict[str, object]:
    return {
        metric_name: _first_present(payload, candidates)
        for metric_name, candidates in _METRIC_CANDIDATES.items()
    }


def baseline_row(
    *,
    source_kind: str,
    source_path: Path,
    payload: Mapping[str, object],
    relevance_reason: str,
    run_id: object = None,
    ledger_file: Path | None = None,
    ledger_row_index: int | None = None,
    registry_database: Path | None = None,
    registry_table: str | None = None,
    artifact_export_path: Path | None = None,
    seed_artifact_path: Path | None = None,
) -> dict[str, object]:
    normalized_fields_available = _normalization_fields_available(payload)
    metrics = _extract_metrics(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "normalization_contract_version": (
            _first_present(payload, ("ALM_SCHEMA_VERSION", "alm_schema_version"))
            if normalized_fields_available
            else PRE_NORMALIZATION_CONTRACT_VERSION
        ),
        "normalized_fields_available": normalized_fields_available,
        "source_kind": source_kind,
        "source_path": str(source_path),
        "relevance_reason": relevance_reason,
        "run_id": run_id if run_id is not None else _first_present(payload, ("run_id",)),
        "ledger_file": str(ledger_file) if ledger_file is not None else None,
        "ledger_row_index": ledger_row_index,
        "registry_database": str(registry_database) if registry_database is not None else None,
        "registry_table": registry_table,
        "artifact_export_path": (
            str(artifact_export_path) if artifact_export_path is not None else None
        ),
        "seed_artifact_path": str(seed_artifact_path) if seed_artifact_path is not None else None,
        "solver_checkout": _first_present(payload, ("solver_checkout", "solver_root")),
        "solver_commit": _first_present(payload, ("solver_commit",)),
        "created_at": _first_present(payload, ("created_at", "timestamp")),
        **metrics,
    }


def _registry_rows(registry_db: Path) -> list[dict[str, object]]:
    if not registry_db.exists():
        return []
    connection = sqlite3.connect(registry_db)
    connection.row_factory = sqlite3.Row
    query = """
        SELECT
            runs.run_id,
            runs.created_at,
            runs.runtime_s,
            runs.solver,
            runs.method,
            runs.iterations,
            runs.objective_J,
            runs.optimizer_converged,
            runs.validation_status,
            runs.solver_commit,
            runs.artifact_dir,
            runs.single_stage_artifact_dir,
            runs.results_json_path,
            params.constraint_method,
            params.alm_enabled,
            params.alm_outer_iters,
            params.alm_formulation,
            params.alm_penalty_init,
            params.alm_penalty_scale,
            params.alm_feas_tol,
            params.alm_stationarity_tol,
            params.alm_qs_threshold,
            params.alm_boozer_threshold,
            params.alm_iota_penalty_threshold,
            params.alm_length_penalty_threshold,
            params.alm_penalty_max,
            metrics.field_error,
            metrics.qs_error,
            metrics.boozer_residual,
            metrics.iota_actual,
            seeds.seed_path
        FROM runs
        LEFT JOIN params ON params.run_id = runs.run_id
        LEFT JOIN metrics ON metrics.run_id = runs.run_id
        LEFT JOIN seeds ON seeds.run_id = runs.run_id
        WHERE
            params.constraint_method = 'alm'
            OR params.alm_enabled = 1
            OR params.alm_outer_iters IS NOT NULL
            OR params.alm_formulation IS NOT NULL
        ORDER BY runs.created_at, runs.run_id
    """
    rows = [dict(row) for row in connection.execute(query)]
    connection.close()
    return rows


def collect_registry_baseline(roots: AutoresearchArtifactRoots) -> list[dict[str, object]]:
    registry_db = roots.registry_dir / "registry.db"
    rows = []
    for payload in _registry_rows(registry_db):
        reason = alm_relevance_reason(payload)
        if reason is not None:
            rows.append(
                baseline_row(
                    source_kind="registry",
                    source_path=registry_db,
                    payload=payload,
                    relevance_reason=reason,
                    run_id=payload.get("run_id"),
                    registry_database=registry_db,
                    registry_table="runs/params/metrics/seeds",
                    artifact_export_path=(
                        Path(str(payload["artifact_dir"]))
                        if payload.get("artifact_dir") not in {None, ""}
                        else None
                    ),
                    seed_artifact_path=(
                        Path(str(payload["seed_path"]))
                        if payload.get("seed_path") not in {None, ""}
                        else None
                    ),
                )
            )
    return rows


def collect_ledger_baseline(roots: AutoresearchArtifactRoots) -> list[dict[str, object]]:
    if not roots.ledger_path.exists():
        return []
    rows = []
    for row_index, payload in _iter_jsonl(roots.ledger_path):
        reason = alm_relevance_reason(payload)
        if reason is not None:
            rows.append(
                baseline_row(
                    source_kind="ledger",
                    source_path=roots.ledger_path,
                    payload=payload,
                    relevance_reason=reason,
                    ledger_file=roots.ledger_path,
                    ledger_row_index=row_index,
                )
            )
    return rows


def _candidate_artifact_json_paths(root: Path) -> Iterable[Path]:
    if not root.exists():
        return ()
    return sorted(
        path
        for path in root.rglob("*.json")
        if path.name in {"results.json", "manifest.json"}
        or "results" in path.name
        or "alm" in path.name.lower()
    )


def collect_run_artifact_baseline(
    roots: AutoresearchArtifactRoots,
    skipped_artifacts: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    rows = []
    for path in _candidate_artifact_json_paths(roots.runs_dir):
        payload, skipped = _read_json_mapping_or_skip(path)
        if skipped is not None:
            if skipped_artifacts is not None:
                skipped_artifacts.append(skipped)
            continue
        reason = alm_relevance_reason(payload)
        if reason is not None:
            rows.append(
                baseline_row(
                    source_kind="run_artifact",
                    source_path=path,
                    payload=payload,
                    relevance_reason=reason,
                    artifact_export_path=path.parent,
                )
            )
    return rows


def collect_harvested_seed_manifest(
    roots: AutoresearchArtifactRoots,
    skipped_artifacts: list[dict[str, object]] | None = None,
) -> list[dict[str, object]]:
    manifest = []
    for path in _candidate_artifact_json_paths(roots.harvested_seeds_dir):
        payload, skipped = _read_json_mapping_or_skip(path)
        if skipped is not None:
            if skipped_artifacts is not None:
                skipped_artifacts.append(skipped)
            continue
        reason = alm_relevance_reason(payload)
        if reason is not None:
            manifest.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "source_kind": "harvested_seed",
                    "source_path": str(path),
                    "seed_artifact_path": str(path.parent),
                    "run_id": _first_present(payload, ("run_id",)),
                    "relevance_reason": reason,
                    "solver_commit": _first_present(payload, ("solver_commit",)),
                    "created_at": _first_present(payload, ("created_at", "timestamp")),
                    **_extract_metrics(payload),
                }
            )
    return manifest


def build_baseline_summary(
    roots: AutoresearchArtifactRoots,
    *,
    created_at_utc: str | None = None,
) -> dict[str, object]:
    created_at = created_at_utc or datetime.now(timezone.utc).isoformat()
    skipped_artifacts: list[dict[str, object]] = []
    registry_rows = collect_registry_baseline(roots)
    ledger_rows = collect_ledger_baseline(roots)
    artifact_rows = collect_run_artifact_baseline(roots, skipped_artifacts)
    fixture_manifest = collect_harvested_seed_manifest(roots, skipped_artifacts)
    rows = [*registry_rows, *ledger_rows, *artifact_rows]
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": created_at,
        "python_version": sys.version,
        "platform": platform.platform(),
        "artifact_roots": {
            "registry_dir": str(roots.registry_dir),
            "runs_dir": str(roots.runs_dir),
            "ledger_path": str(roots.ledger_path),
            "artifact_exports_dir": str(roots.artifact_exports_dir),
            "harvested_seeds_dir": str(roots.harvested_seeds_dir),
        },
        "counts": {
            "registry_rows": len(registry_rows),
            "ledger_rows": len(ledger_rows),
            "run_artifact_rows": len(artifact_rows),
            "harvested_seed_fixtures": len(fixture_manifest),
            "skipped_artifacts": len(skipped_artifacts),
            "baseline_rows": len(rows),
        },
        "baseline_rows": rows,
        "fixture_manifest": fixture_manifest,
        "skipped_artifacts": skipped_artifacts,
    }


def empty_comparison_rows(baseline_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for row in baseline_rows:
        case = row.get("run_id") or row.get("source_path")
        rows.append(
            {
                "case": case,
                "before_success": row.get("hard_feasible_success"),
                "after_success": None,
                "before_best_feasible_objective": row.get("best_feasible_base_objective"),
                "after_best_feasible_objective": None,
                "before_max_raw_hard_violation": row.get("max_raw_hard_violation"),
                "after_max_raw_hard_violation": None,
                "before_max_normalized_violation": row.get("max_normalized_violation"),
                "after_max_normalized_violation": None,
                "before_outer_iters": row.get("outer_iterations"),
                "after_outer_iters": None,
                "before_evals": row.get("objective_eval_count"),
                "after_evals": None,
                "before_wall_s": row.get("wall_time_s"),
                "after_wall_s": None,
                "before_penalty_cap_hits": row.get("penalty_cap_hit_count"),
                "after_penalty_cap_hits": None,
                "before_multiplier_cap_hits": row.get("multiplier_cap_hit_count"),
                "after_multiplier_cap_hits": None,
                "blocking_constraint_before": row.get("blocking_constraint_name"),
                "blocking_constraint_after": None,
            }
        )
    return rows


def render_markdown_summary(summary: Mapping[str, object]) -> str:
    counts = summary["counts"]
    lines = [
        "# ALM Normalization Baseline",
        "",
        f"- Schema: `{summary['schema_version']}`",
        f"- Created UTC: `{summary['created_at_utc']}`",
        f"- Registry rows: `{counts['registry_rows']}`",
        f"- Ledger rows: `{counts['ledger_rows']}`",
        f"- Run artifact rows: `{counts['run_artifact_rows']}`",
        f"- Harvested seed fixtures: `{counts['harvested_seed_fixtures']}`",
        f"- Skipped artifacts: `{counts['skipped_artifacts']}`",
        f"- Baseline rows: `{counts['baseline_rows']}`",
        "",
        "| Source | Run ID | Reason | Normalized fields | Solver commit |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in summary["baseline_rows"][:50]:
        lines.append(
            "| "
            f"{row['source_kind']} | "
            f"{row.get('run_id')} | "
            f"{row['relevance_reason']} | "
            f"{row['normalized_fields_available']} | "
            f"{row.get('solver_commit')} |"
        )
    return "\n".join(lines) + "\n"


def write_baseline_outputs(
    summary: Mapping[str, object],
    output_dir: Path,
    *,
    stamp: str,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = output_dir / f"baseline_{stamp}.jsonl"
    fixture_manifest_path = output_dir / f"fixture_manifest_{stamp}.json"
    skipped_artifacts_path = output_dir / f"skipped_artifacts_{stamp}.json"
    comparison_csv_path = output_dir / f"comparison_{stamp}.csv"
    comparison_md_path = output_dir / f"comparison_{stamp}.md"

    with baseline_path.open("w", encoding="utf-8") as handle:
        for row in summary["baseline_rows"]:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    fixture_manifest_path.write_text(
        json.dumps(summary["fixture_manifest"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    skipped_artifacts_path.write_text(
        json.dumps(summary["skipped_artifacts"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    comparison_rows = empty_comparison_rows(summary["baseline_rows"])
    with comparison_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(comparison_rows)

    comparison_md_path.write_text(render_markdown_summary(summary), encoding="utf-8")

    return {
        "baseline": baseline_path,
        "fixture_manifest": fixture_manifest_path,
        "skipped_artifacts": skipped_artifacts_path,
        "comparison_csv": comparison_csv_path,
        "comparison_markdown": comparison_md_path,
    }


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture read-only ALM baseline rows from autoresearch artifacts."
    )
    parser.add_argument(
        "--autoresearch-root",
        type=Path,
        default=DEFAULT_AUTORESEARCH_ROOT,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stamp", default=utc_stamp())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    roots = AutoresearchArtifactRoots.from_autoresearch_root(args.autoresearch_root)
    output_dir = args.output_dir or benchmark_output_dir(roots)
    summary = build_baseline_summary(roots)
    paths = write_baseline_outputs(summary, output_dir, stamp=args.stamp)
    print(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "counts": summary["counts"],
                "paths": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0
