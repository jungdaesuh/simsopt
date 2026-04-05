"""Render a standardized markdown benchmark report from a manifest and JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a standardized markdown benchmark report."
    )
    parser.add_argument("--manifest-json", required=True)
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-md", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as infile:
        return json.load(infile)


def _format_float(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}"
    return "n/a"


def _format_speedup(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.2f}x"
    return "n/a"


def _build_summary_rows(summary: list[dict]) -> str:
    rows: list[str] = []
    for item in summary:
        rows.append(
            "| {name} | {passed} | {outer_elapsed_s} | {cpu_elapsed_s} | "
            "{lane_elapsed_s} | {speedup_vs_cpu} |".format(
                name=item["name"],
                passed=str(bool(item["passed"])).lower(),
                outer_elapsed_s=_format_float(item.get("outer_elapsed_s")),
                cpu_elapsed_s=_format_float(item.get("cpu_elapsed_s")),
                lane_elapsed_s=_format_float(item.get("lane_elapsed_s")),
                speedup_vs_cpu=_format_speedup(item.get("speedup_vs_cpu")),
            )
        )
    return "\n".join(rows)


def render_report(manifest: dict, payload: dict) -> str:
    provenance = payload["provenance"]
    hardware = manifest["hardware_contract"]
    report = manifest["report"]
    aggregate = payload["aggregate"]
    summary = payload["summary"]
    notes = "\n".join(f"- {note}" for note in hardware["notes"])
    performance_budget = manifest.get("performance_budget", {})
    memory_budget = manifest.get("memory_budget", {})
    performance_failures = aggregate.get("performance_failures", [])
    failure_lines = (
        "\n".join(f"- {failure}" for failure in performance_failures)
        if performance_failures
        else "- none"
    )

    return f"""# {manifest['title']}

## Run Identity

- benchmark id: `{manifest['benchmark_id']}`
- title: `{provenance['title']}`
- repo sha: `{provenance['repo_sha']}`
- workflow: `{manifest['schedule']['workflow']}`
- output markdown: `{report['output_markdown']}`

## Hardware Contract

- runner labels: `{", ".join(hardware['runner_labels'])}`
- platform: `{hardware['platform']}`
- stable-hardware expectation: {hardware['expectation']}
{notes}

## Runtime Contract

- default rollout lane: `{manifest['default_rollout_lane']}`
- benchmark lane: `{provenance.get('lane', 'n/a')}`
- backend: `{provenance['backend']}`
- devices: `{provenance['devices']}`
- JAX / jaxlib: `{provenance['jax']}` / `{provenance['jaxlib']}`
- x64 enabled: `{provenance['x64_enabled']}`
- compile behavior: `{provenance.get('compile_behavior', 'n/a')}`
- compilation cache policy: `{provenance['compilation_cache_policy']}`

## Fixture Summary

- fixture: `{provenance.get('fixture', 'n/a')}`
- stage 2 grid: `{provenance.get('stage2_nphi', 'n/a')} x {provenance.get('stage2_ntheta', 'n/a')}`
- single-stage grid: `{provenance.get('single_stage_nphi', 'n/a')} x {provenance.get('single_stage_ntheta', 'n/a')}`
- optimizer backend: `{provenance.get('optimizer_backend', 'n/a')}`

## Timing Summary

| rung | passed | outer elapsed s | cpu elapsed s | lane elapsed s | speedup vs cpu |
| --- | --- | --- | --- | --- | --- |
{_build_summary_rows(summary)}

## Memory Summary

- peak RSS MB: `{_format_float(provenance.get('peak_rss_mb'))}`
- GPU memory MB: `{_format_float(provenance.get('gpu_memory_mb'))}`
- grouped-adjoint budget fixture: `{memory_budget.get('fixture', 'n/a')}`
- grouped-adjoint RSS ceiling MB: `{_format_float(memory_budget.get('max_peak_rss_mb'))}`
- grouped-adjoint GPU ceiling MB: `{_format_float(memory_budget.get('max_peak_gpu_memory_mb'))}`

## Aggregate

- aggregate passed: `{str(bool(aggregate['passed'])).lower()}`
- aggregate lane label: `{aggregate['lane_label']}`
- total outer elapsed s: `{_format_float(aggregate['total_outer_elapsed_s'])}`
- performance budget profile: `{performance_budget.get('profile', 'n/a')}`
- Stage 2 cold speed floor: `{_format_speedup(performance_budget.get('tier2_stage2_e2e', {}).get('min_outer_speedup_vs_cpu'))}`
- Stage 2 warm speed floor: `{_format_speedup(performance_budget.get('tier2_stage2_e2e', {}).get('min_warm_speedup_vs_cpu'))}`
- Stage 2 compile ceiling s: `{_format_float(performance_budget.get('tier2_stage2_e2e', {}).get('max_compile_overhead_s'))}`

## Regression Gates

{failure_lines}

## Honest Interpretation

- cold compile notes: first-run timing can still be dominated by compilation and cache state.
- warm timing notes: treat repeated steady-state runs as the throughput signal, not the first call.
- parity-vs-fast note: this report is for parity-aligned benchmark reporting, not `jax_gpu_fast` throughput claims.
- memory caveats: compare memory only on the same fixture and the same stable hardware contract.
"""


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest_json)
    payload = load_json(args.input_json)
    report_text = render_report(manifest, payload)
    output_path = Path(args.output_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


if __name__ == "__main__":
    main()
