"""Tier 5 trusted-fixture performance characterization for CPU vs GPU lanes."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_IOTA_TARGET,
    DEFAULT_OPTIMIZER_BACKEND,
    DEFAULT_PLASMA_SURF_FILENAME,
    DEFAULT_SMOKE_MPOL,
    DEFAULT_SMOKE_NPHI,
    DEFAULT_SMOKE_NTHETA,
    DEFAULT_SMOKE_NTOR,
    DEFAULT_STAGE2_BS_PATH,
    DEFAULT_VOL_TARGET,
)
from benchmarks.validation_ladder_common import (
    apply_compilation_cache_policy,
    apply_requested_platform,
    build_provenance,
    describe_compile_behavior,
    evaluate_tier5_performance_budget,
    load_json,
    maybe_initialize_distributed_runtime,
    preparse_platform,
    print_provenance,
    require_x64_runtime,
    resolve_probe_lane,
    repo_pythonpath_env,
    run_python_script,
    tier5_performance_budget,
    write_json,
)

TIER1_PARITY_RUNG = "tier1b_real_stage2"
TIER2_PERFORMANCE_RUNG = "tier2_stage2_e2e"
TIER3_INIT_RUNG = "tier3_single_stage_init"
TIER4_ADJOINT_RUNG = "tier4_adjoint_fd"
_TIER5_PERFORMANCE_BUDGET_PROFILE = "stable_hardware_weekly"
_TIER5_PHASE_CHOICES = ("full", "gpu", "cpu", "aggregate")

_RUNTIME_JAX = None
_RUNTIME_JAXLIB = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Tier 5 performance characterization on the trusted public-lane fixtures."
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="JAX platform to request before import/use.",
    )
    parser.add_argument(
        "--output-json",
        required=True,
        help="Path to write structured Tier 5 performance results.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=DEFAULT_PLASMA_SURF_FILENAME,
        help="VMEC equilibrium filename for the trusted public fixture.",
    )
    parser.add_argument(
        "--equilibria-dir",
        default=str(DEFAULT_EQUILIBRIA_DIR),
        help="Directory that contains VMEC equilibrium files.",
    )
    parser.add_argument(
        "--equilibrium-path",
        default=None,
        help="Explicit equilibrium path override.",
    )
    parser.add_argument(
        "--stage2-bs-path",
        default=str(DEFAULT_STAGE2_BS_PATH),
        help="Path to the fixed Stage 2 seed biot_savart_opt.json fixture.",
    )
    parser.add_argument(
        "--stage2-nphi",
        type=int,
        default=255,
        help="Surface toroidal grid points for the real Stage 2 fixture.",
    )
    parser.add_argument(
        "--stage2-ntheta",
        type=int,
        default=64,
        help="Surface poloidal grid points for the real Stage 2 fixture.",
    )
    parser.add_argument(
        "--single-stage-nphi",
        type=int,
        default=DEFAULT_SMOKE_NPHI,
        help="Surface toroidal grid points for the reduced-grid trusted fixture.",
    )
    parser.add_argument(
        "--single-stage-ntheta",
        type=int,
        default=DEFAULT_SMOKE_NTHETA,
        help="Surface poloidal grid points for the reduced-grid trusted fixture.",
    )
    parser.add_argument(
        "--mpol",
        type=int,
        default=DEFAULT_SMOKE_MPOL,
        help="Surface poloidal mode count for the trusted single-stage fixture.",
    )
    parser.add_argument(
        "--ntor",
        type=int,
        default=DEFAULT_SMOKE_NTOR,
        help="Surface toroidal mode count for the trusted single-stage fixture.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=20,
        help="Short Stage 2 optimization budget used by Tier 2.",
    )
    parser.add_argument(
        "--vol-target",
        type=float,
        default=DEFAULT_VOL_TARGET,
        help="Single-stage target volume.",
    )
    parser.add_argument(
        "--iota-target",
        type=float,
        default=DEFAULT_IOTA_TARGET,
        help="Single-stage target iota.",
    )
    parser.add_argument(
        "--optimizer-backend",
        choices=("scipy", "hybrid", "ondevice"),
        default=DEFAULT_OPTIMIZER_BACKEND,
        help="JAX Boozer optimizer backend for the single-stage trusted fixture.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Finite-difference sample count for Tier 4 timing.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-4,
        help="Finite-difference perturbation magnitude for Tier 4 timing.",
    )
    parser.add_argument(
        "--benchmark-mode",
        action="store_true",
        help=(
            "Enable benchmark-mode on the Tier 3 single-stage rung so timing "
            "skips heavy target-lane artifacts."
        ),
    )
    parser.add_argument(
        "--phase",
        choices=_TIER5_PHASE_CHOICES,
        default="full",
        help=(
            "Execution mode: full runs all rungs, gpu runs the CUDA-relevant rungs "
            "plus the Tier 4 lane probe, cpu runs only the Tier 4 CPU baseline, and "
            "aggregate merges prior gpu/cpu phase artifacts."
        ),
    )
    parser.add_argument(
        "--gpu-input-json",
        default=None,
        help="GPU-phase Tier 5 artifact to merge when --phase=aggregate.",
    )
    parser.add_argument(
        "--cpu-input-json",
        default=None,
        help="CPU-phase Tier 5 artifact to merge when --phase=aggregate.",
    )
    return parser.parse_args()


def _runtime_modules() -> tuple[Any, Any]:
    global _RUNTIME_JAX, _RUNTIME_JAXLIB
    if _RUNTIME_JAX is None or _RUNTIME_JAXLIB is None:
        requested_platform = preparse_platform(sys.argv[1:])
        apply_requested_platform(requested_platform)
        apply_compilation_cache_policy()
        import jax as runtime_jax
        import jaxlib as runtime_jaxlib

        maybe_initialize_distributed_runtime()
        runtime_jax.config.update("jax_enable_x64", True)
        require_x64_runtime(runtime_jax, context="Tier 5 performance characterization")
        _RUNTIME_JAX = runtime_jax
        _RUNTIME_JAXLIB = runtime_jaxlib
    return _RUNTIME_JAX, _RUNTIME_JAXLIB


def _stage2_value_gradient_script() -> Path:
    return REPO_ROOT / "benchmarks" / "stage2_value_gradient_parity.py"


def _stage2_e2e_script() -> Path:
    return REPO_ROOT / "benchmarks" / "stage2_e2e_comparison.py"


def _single_stage_init_script() -> Path:
    return REPO_ROOT / "benchmarks" / "single_stage_init_parity.py"


def _adjoint_fd_script() -> Path:
    return REPO_ROOT / "benchmarks" / "adjoint_fd_validation.py"


def _common_equilibrium_args(args: argparse.Namespace) -> list[str]:
    if args.equilibrium_path:
        return ["--equilibrium-path", args.equilibrium_path]
    return [
        "--plasma-surf-filename",
        args.plasma_surf_filename,
        "--equilibria-dir",
        args.equilibria_dir,
    ]


def _trusted_single_stage_args(args: argparse.Namespace) -> list[str]:
    return [
        "--stage2-bs-path",
        args.stage2_bs_path,
        "--nphi",
        str(args.single_stage_nphi),
        "--ntheta",
        str(args.single_stage_ntheta),
        "--mpol",
        str(args.mpol),
        "--ntor",
        str(args.ntor),
        "--vol-target",
        str(args.vol_target),
        "--iota-target",
        str(args.iota_target),
        "--optimizer-backend",
        args.optimizer_backend,
    ]


def _single_stage_init_probe_args(args: argparse.Namespace) -> list[str]:
    command = [
        "--platform",
        args.platform,
        *_common_equilibrium_args(args),
        *_trusted_single_stage_args(args),
    ]
    if bool(args.benchmark_mode):
        command.append("--benchmark-mode")
    return command


def _stage2_e2e_probe_args(args: argparse.Namespace) -> list[str]:
    return [
        "--platform",
        args.platform,
        "--nphi",
        str(args.stage2_nphi),
        "--ntheta",
        str(args.stage2_ntheta),
        "--maxiter",
        str(args.maxiter),
        "--optimizer-backend",
        args.optimizer_backend,
        *_common_equilibrium_args(args),
    ]


def safe_speedup(reference_s: float | None, candidate_s: float | None) -> float | None:
    if reference_s is None or candidate_s is None or candidate_s <= 0.0:
        return None
    return float(reference_s / candidate_s)


def _float_or_none(value: Any) -> float | None:
    return float(value) if value is not None else None


def _format_elapsed_s(value: Any) -> str:
    elapsed_s = _float_or_none(value)
    return f"{(elapsed_s if elapsed_s is not None else float('nan')):.2f}s"


def _format_speedup(value: Any) -> str:
    return f"{value:.2f}x" if isinstance(value, float) else "n/a"


def _timed_probe(
    script_path: Path,
    command_args: list[str],
    *,
    platform: str,
) -> tuple[dict[str, Any], float]:
    with tempfile.TemporaryDirectory(prefix=f"{script_path.stem}-") as temp_dir:
        output_json = str(Path(temp_dir) / f"{script_path.stem}.json")
        start = time.perf_counter()
        run_python_script(
            script_path,
            [*command_args, "--output-json", output_json],
            env=repo_pythonpath_env(platform=platform),
            cwd=REPO_ROOT,
            bootstrap_repo=True,
            stream_output=True,
        )
        elapsed_s = time.perf_counter() - start
        return load_json(output_json), float(elapsed_s)


def summarize_pair_probe(
    *,
    name: str,
    payload: dict[str, Any],
    outer_elapsed_s: float,
    cpu_elapsed_s: float,
    lane_elapsed_s: float,
    lane_label: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(payload.get("passed", False)),
        "outer_elapsed_s": float(outer_elapsed_s),
        "cpu_elapsed_s": float(cpu_elapsed_s),
        "lane_elapsed_s": float(lane_elapsed_s),
        "lane_label": lane_label,
        "speedup_vs_cpu": safe_speedup(cpu_elapsed_s, lane_elapsed_s),
    }


def _with_performance_contract(
    summary: dict[str, Any],
    *,
    timing_semantics: str,
    recommended_question: str,
    supports_performance_headline: bool,
    headline_metric: str | None = None,
    headline_speedup_vs_cpu: float | None = None,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = dict(summary)
    enriched["timing_semantics"] = timing_semantics
    enriched["recommended_question"] = recommended_question
    enriched["supports_performance_headline"] = bool(supports_performance_headline)
    if headline_metric is not None:
        enriched["headline_metric"] = headline_metric
    if headline_speedup_vs_cpu is not None:
        enriched["headline_speedup_vs_cpu"] = float(headline_speedup_vs_cpu)
    if extra_fields:
        enriched.update(extra_fields)
    return enriched


def summarize_informational_pair_probe(
    *,
    name: str,
    payload: dict[str, Any],
    outer_elapsed_s: float,
    cpu_elapsed_s: float,
    lane_elapsed_s: float,
    lane_label: str,
    timing_semantics: str,
    recommended_question: str,
) -> dict[str, Any]:
    return _with_performance_contract(
        summarize_pair_probe(
            name=name,
            payload=payload,
            outer_elapsed_s=outer_elapsed_s,
            cpu_elapsed_s=cpu_elapsed_s,
            lane_elapsed_s=lane_elapsed_s,
            lane_label=lane_label,
        ),
        timing_semantics=timing_semantics,
        recommended_question=recommended_question,
        supports_performance_headline=False,
    )


def summarize_stage2_e2e_performance_probe(
    *,
    payload: dict[str, Any],
    outer_elapsed_s: float,
    lane_label: str,
) -> dict[str, Any]:
    comparison = payload["comparison"]
    timings = payload["timings"]
    cpu_elapsed_s = float(comparison["cpu_elapsed_s"])
    cold_elapsed_s = float(comparison["jax_elapsed_s"])
    outer_lane_elapsed_s = float(timings.get("jax_outer_elapsed_s", cold_elapsed_s))
    warm_elapsed_s = _float_or_none(timings.get("jax_optimizer_warm_run_s"))
    warm_speedup_vs_cpu = (
        safe_speedup(cpu_elapsed_s, warm_elapsed_s) if warm_elapsed_s is not None else None
    )
    outer_speedup_vs_cpu = safe_speedup(cpu_elapsed_s, outer_lane_elapsed_s)
    return _with_performance_contract(
        summarize_pair_probe(
            name=TIER2_PERFORMANCE_RUNG,
            payload=payload,
            outer_elapsed_s=outer_elapsed_s,
            cpu_elapsed_s=cpu_elapsed_s,
            lane_elapsed_s=cold_elapsed_s,
            lane_label=lane_label,
        ),
        timing_semantics="separate_cold_end_to_end_and_warm_steady_state",
        recommended_question="cold_and_warm_performance",
        supports_performance_headline=True,
        headline_metric="outer_speedup_vs_cpu",
        headline_speedup_vs_cpu=outer_speedup_vs_cpu,
        extra_fields={
            "cpu_outer_elapsed_s": float(timings.get("cpu_outer_elapsed_s", cpu_elapsed_s)),
            "lane_outer_elapsed_s": outer_lane_elapsed_s,
            "outer_speedup_vs_cpu": outer_speedup_vs_cpu,
            "lane_warm_elapsed_s": warm_elapsed_s,
            "warm_speedup_vs_cpu": warm_speedup_vs_cpu,
            "lane_compile_overhead_s": _float_or_none(
                timings.get("jax_optimizer_compile_overhead_s")
            ),
        },
    )


def summarize_single_lane_probe(
    *,
    name: str,
    payload: dict[str, Any],
    outer_elapsed_s: float,
    lane_label: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(payload.get("passed", False)),
        "outer_elapsed_s": float(outer_elapsed_s),
        "lane_label": lane_label,
    }


def _build_summary_by_name(summary: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["name"]: item for item in summary}


def build_tier5_performance_contract(summary: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = _build_summary_by_name(summary)
    tier1 = by_name[TIER1_PARITY_RUNG]
    tier2 = by_name[TIER2_PERFORMANCE_RUNG]
    tier3 = by_name[TIER3_INIT_RUNG]
    headline_metric = str(tier2.get("headline_metric", "speedup_vs_cpu"))
    headline_speedup = tier2.get("headline_speedup_vs_cpu")

    return {
        "parity_source": {
            "rung": tier1["name"],
            "metric_path": f"rungs.{TIER1_PARITY_RUNG}.results.comparisons",
            "timing_semantics": tier1["timing_semantics"],
        },
        "cold_end_to_end_source": {
            "rung": tier2["name"],
            "metric_path": (
                f"summary_by_name.{TIER2_PERFORMANCE_RUNG}.outer_speedup_vs_cpu"
            ),
            "speedup_vs_cpu": tier2.get("outer_speedup_vs_cpu"),
        },
        "warm_steady_state_source": (
            {
                "rung": tier2["name"],
                "metric_path": (
                    f"summary_by_name.{TIER2_PERFORMANCE_RUNG}.warm_speedup_vs_cpu"
                ),
                "speedup_vs_cpu": tier2.get("warm_speedup_vs_cpu"),
            }
            if tier2.get("warm_speedup_vs_cpu") is not None
            else None
        ),
        "headline_performance_source": {
            "rung": tier2["name"],
            "metric_path": (
                f"summary_by_name.{TIER2_PERFORMANCE_RUNG}.{headline_metric}"
            ),
            "speedup_vs_cpu": headline_speedup,
        },
        "sharding_source": {
            "rung": tier2["name"],
            "active_path": f"rungs.{TIER2_PERFORMANCE_RUNG}.provenance.sharding_active",
            "strategy_path": (
                f"rungs.{TIER2_PERFORMANCE_RUNG}.provenance.sharding_strategy"
            ),
            "device_count_path": (
                f"rungs.{TIER2_PERFORMANCE_RUNG}.provenance.sharding_device_count"
            ),
        },
        "do_not_use_for_performance_headline": [
            rung["name"]
            for rung in (tier1, tier3, by_name.get(TIER4_ADJOINT_RUNG, {}))
            if not bool(rung.get("supports_performance_headline", False))
            and "name" in rung
        ],
    }


def evaluate_tier5_sharding_contract(provenance: dict[str, Any]) -> list[str]:
    """Require active sharding when a multi-device sharded lane is configured."""
    device_count = int(provenance.get("sharding_device_count") or 0)
    strategy = str(provenance.get("sharding_strategy") or "none")
    if device_count <= 1 or strategy == "none":
        return []
    if bool(provenance.get("sharding_active")):
        return []
    return [
        "Tier 5 sharded lane reported multiple visible devices and sharding "
        f"strategy {strategy!r}, but provenance.sharding_active was false."
    ]


def _render_tier5_summary_line(item: dict[str, Any]) -> str:
    if item.get("name") == TIER2_PERFORMANCE_RUNG:
        return (
            f"{item['name']}: passed={item['passed']}  "
            f"outer={item['outer_elapsed_s']:.2f}s  "
            f"cpu={_format_elapsed_s(item.get('cpu_elapsed_s'))}  "
            f"{item['lane_label']}_cold={_format_elapsed_s(item.get('lane_elapsed_s'))}  "
            f"{item['lane_label']}_outer={_format_elapsed_s(item.get('lane_outer_elapsed_s'))}  "
            f"{item['lane_label']}_warm={_format_elapsed_s(item.get('lane_warm_elapsed_s'))}  "
            f"cold_speedup_vs_cpu={_format_speedup(item.get('speedup_vs_cpu'))}  "
            f"outer_speedup_vs_cpu={_format_speedup(item.get('outer_speedup_vs_cpu'))}  "
            f"warm_speedup_vs_cpu={_format_speedup(item.get('warm_speedup_vs_cpu'))}"
        )
    if item.get("supports_performance_headline") is False:
        return (
            f"{item['name']}: passed={item['passed']}  "
            f"outer={item['outer_elapsed_s']:.2f}s  "
            f"cpu={_format_elapsed_s(item.get('cpu_elapsed_s'))}  "
            f"{item['lane_label']}={_format_elapsed_s(item.get('lane_elapsed_s'))}  "
            "timings=informational-only"
        )
    return (
        f"{item['name']}: passed={item['passed']}  "
        f"outer={item['outer_elapsed_s']:.2f}s  "
        f"cpu={_format_elapsed_s(item.get('cpu_elapsed_s'))}  "
        f"{item['lane_label']}={_format_elapsed_s(item.get('lane_elapsed_s'))}  "
        f"speedup_vs_cpu={_format_speedup(item.get('speedup_vs_cpu'))}"
    )


def _tier4_probe_args(args: argparse.Namespace) -> list[str]:
    return [
        *_common_equilibrium_args(args),
        *_trusted_single_stage_args(args),
        "--samples",
        str(args.samples),
        "--eps",
        str(args.eps),
    ]


def _run_tier4_pair(args: argparse.Namespace) -> dict[str, Any]:
    base_args = _tier4_probe_args(args)
    cpu_payload, cpu_outer_elapsed_s = _timed_probe(
        _adjoint_fd_script(),
        ["--platform", "cpu", *base_args],
        platform="cpu",
    )

    if args.platform == "cpu":
        lane_payload = cpu_payload
        lane_outer_elapsed_s = cpu_outer_elapsed_s
        lane_elapsed_s = cpu_outer_elapsed_s
    else:
        lane_payload, lane_outer_elapsed_s = _timed_probe(
            _adjoint_fd_script(),
            ["--platform", args.platform, *base_args],
            platform=args.platform,
        )
        lane_elapsed_s = lane_outer_elapsed_s

    return {
        "cpu_payload": cpu_payload,
        "lane_payload": lane_payload,
        "summary": summarize_pair_probe(
            name=TIER4_ADJOINT_RUNG,
            payload=lane_payload,
            outer_elapsed_s=cpu_outer_elapsed_s + lane_outer_elapsed_s,
            cpu_elapsed_s=cpu_outer_elapsed_s,
            lane_elapsed_s=lane_elapsed_s,
            lane_label="jax-cpu" if args.platform == "cpu" else f"jax-{args.platform}",
        ),
        "probe_timings": {
            "tier4_adjoint_fd_cpu": float(cpu_outer_elapsed_s),
            "tier4_adjoint_fd_lane": float(lane_outer_elapsed_s),
        },
    }


def _run_tier4_cpu_probe(args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    return _timed_probe(
        _adjoint_fd_script(),
        ["--platform", "cpu", *_tier4_probe_args(args)],
        platform="cpu",
    )


def _run_tier4_lane_probe(args: argparse.Namespace) -> tuple[dict[str, Any], float]:
    return _timed_probe(
        _adjoint_fd_script(),
        ["--platform", args.platform, *_tier4_probe_args(args)],
        platform=args.platform,
    )


def _summarize_tier4_pair(
    *,
    cpu_payload: dict[str, Any],
    cpu_outer_elapsed_s: float,
    lane_payload: dict[str, Any],
    lane_outer_elapsed_s: float,
    lane_label: str,
) -> dict[str, Any]:
    return summarize_pair_probe(
        name=TIER4_ADJOINT_RUNG,
        payload=lane_payload,
        outer_elapsed_s=float(cpu_outer_elapsed_s + lane_outer_elapsed_s),
        cpu_elapsed_s=float(cpu_outer_elapsed_s),
        lane_elapsed_s=float(lane_outer_elapsed_s),
        lane_label=lane_label,
    )


def _combine_phase_payload(
    *,
    provenance: dict[str, Any],
    lane_label: str,
    phase: str,
    rungs: dict[str, Any],
    summary: list[dict[str, Any]],
    probe_timings: dict[str, float],
    phase_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    summary_by_name = _build_summary_by_name(summary)
    expected_rungs = {
        TIER1_PARITY_RUNG,
        TIER2_PERFORMANCE_RUNG,
        TIER3_INIT_RUNG,
        TIER4_ADJOINT_RUNG,
    }
    missing_rungs = sorted(expected_rungs - set(summary_by_name))
    total_outer_elapsed_s = float(sum(item["outer_elapsed_s"] for item in summary))
    if TIER4_ADJOINT_RUNG not in summary_by_name:
        total_outer_elapsed_s += float(probe_timings.get("tier4_adjoint_fd_cpu", 0.0))
        total_outer_elapsed_s += float(probe_timings.get("tier4_adjoint_fd_lane", 0.0))
    performance_budget = tier5_performance_budget(
        profile=_TIER5_PERFORMANCE_BUDGET_PROFILE
    )
    performance_contract = (
        build_tier5_performance_contract(summary)
        if {
            TIER1_PARITY_RUNG,
            TIER2_PERFORMANCE_RUNG,
            TIER3_INIT_RUNG,
        }.issubset(summary_by_name)
        else None
    )
    performance_failures = (
        evaluate_tier5_performance_budget(summary_by_name, performance_budget)
        if TIER2_PERFORMANCE_RUNG in summary_by_name
        else []
    )
    sharding_failures = (
        evaluate_tier5_sharding_contract(dict(rungs[TIER2_PERFORMANCE_RUNG].get("provenance", provenance)))
        if TIER2_PERFORMANCE_RUNG in rungs
        else []
    )
    phase_passed = all(bool(item["passed"]) for item in summary) and not (
        performance_failures or sharding_failures
    )
    aggregate_complete = not missing_rungs
    payload: dict[str, Any] = {
        "phase": phase,
        "provenance": provenance,
        "rungs": rungs,
        "summary": summary,
        "summary_by_name": summary_by_name,
        "probe_timings": probe_timings,
        "aggregate": {
            "lane_label": lane_label,
            "total_outer_elapsed_s": total_outer_elapsed_s,
            "complete": aggregate_complete,
            "phase_passed": phase_passed,
            "passed": phase_passed if aggregate_complete else False,
            "pending_rungs": missing_rungs,
            "performance_contract": performance_contract,
            "performance_budget_profile": _TIER5_PERFORMANCE_BUDGET_PROFILE,
            "performance_budget": performance_budget,
            "performance_failures": performance_failures,
            "sharding_failures": sharding_failures,
        },
    }
    if phase_inputs is not None:
        payload["phase_inputs"] = phase_inputs
    return payload


def _validate_phase_artifact(
    payload: dict[str, Any],
    *,
    expected_phase: str,
    required_rungs: set[str],
) -> None:
    observed_phase = str(payload.get("phase"))
    if observed_phase != expected_phase:
        raise ValueError(
            f"Expected {expected_phase!r} phase artifact, got {observed_phase!r}."
        )
    observed_rungs = set(payload.get("rungs", {}))
    missing_rungs = sorted(required_rungs - observed_rungs)
    if missing_rungs:
        raise ValueError(
            f"{expected_phase!r} phase artifact missing required rungs: {missing_rungs}."
        )


def _build_aggregate_payload(
    *,
    gpu_payload: dict[str, Any],
    cpu_payload: dict[str, Any],
) -> dict[str, Any]:
    _validate_phase_artifact(
        gpu_payload,
        expected_phase="gpu",
        required_rungs={
            TIER1_PARITY_RUNG,
            TIER2_PERFORMANCE_RUNG,
            TIER3_INIT_RUNG,
            "tier4_adjoint_fd_lane",
        },
    )


def _run_stage123_probes(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], float, dict[str, Any], float, dict[str, Any], float]:
    tier1_payload, tier1_outer = _timed_probe(
        _stage2_value_gradient_script(),
        [
            "--platform",
            args.platform,
            "--fixture",
            "real",
            "--nphi",
            str(args.stage2_nphi),
            "--ntheta",
            str(args.stage2_ntheta),
            *_common_equilibrium_args(args),
        ],
        platform=args.platform,
    )
    tier2_payload, tier2_outer = _timed_probe(
        _stage2_e2e_script(),
        _stage2_e2e_probe_args(args),
        platform=args.platform,
    )
    tier3_payload, tier3_outer = _timed_probe(
        _single_stage_init_script(),
        _single_stage_init_probe_args(args),
        platform=args.platform,
    )
    return (
        tier1_payload,
        tier1_outer,
        tier2_payload,
        tier2_outer,
        tier3_payload,
        tier3_outer,
    )


def _summarize_stage123_probes(
    *,
    tier1_payload: dict[str, Any],
    tier1_outer: float,
    tier2_payload: dict[str, Any],
    tier2_outer: float,
    tier3_payload: dict[str, Any],
    tier3_outer: float,
    lane_label: str,
) -> list[dict[str, Any]]:
    return [
        summarize_informational_pair_probe(
            name=TIER1_PARITY_RUNG,
            payload=tier1_payload,
            outer_elapsed_s=tier1_outer,
            cpu_elapsed_s=float(tier1_payload["results"]["cpu"]["elapsed_s"]),
            lane_elapsed_s=float(tier1_payload["results"]["jax"]["elapsed_s"]),
            lane_label=lane_label,
            timing_semantics="correctness_probe_only",
            recommended_question="parity",
        ),
        summarize_stage2_e2e_performance_probe(
            payload=tier2_payload,
            outer_elapsed_s=tier2_outer,
            lane_label=lane_label,
        ),
        summarize_informational_pair_probe(
            name=TIER3_INIT_RUNG,
            payload=tier3_payload,
            outer_elapsed_s=tier3_outer,
            cpu_elapsed_s=float(tier3_payload["timings"]["cpu_elapsed_s"]),
            lane_elapsed_s=float(tier3_payload["timings"]["jax_elapsed_s"]),
            lane_label=lane_label,
            timing_semantics="initialization_probe_only",
            recommended_question="initialization_diagnostics",
        ),
    ]


def _build_aggregate_payload(
    *,
    gpu_payload: dict[str, Any],
    cpu_payload: dict[str, Any],
) -> dict[str, Any]:
    _validate_phase_artifact(
        gpu_payload,
        expected_phase="gpu",
        required_rungs={
            TIER1_PARITY_RUNG,
            TIER2_PERFORMANCE_RUNG,
            TIER3_INIT_RUNG,
            "tier4_adjoint_fd_lane",
        },
    )
    _validate_phase_artifact(
        cpu_payload,
        expected_phase="cpu",
        required_rungs={"tier4_adjoint_fd_cpu"},
    )
    lane_label = str(gpu_payload["aggregate"]["lane_label"])
    cpu_outer_elapsed_s = float(cpu_payload["probe_timings"]["tier4_adjoint_fd_cpu"])
    lane_outer_elapsed_s = float(gpu_payload["probe_timings"]["tier4_adjoint_fd_lane"])
    tier4_summary = _summarize_tier4_pair(
        cpu_payload=cpu_payload["rungs"]["tier4_adjoint_fd_cpu"],
        cpu_outer_elapsed_s=cpu_outer_elapsed_s,
        lane_payload=gpu_payload["rungs"]["tier4_adjoint_fd_lane"],
        lane_outer_elapsed_s=lane_outer_elapsed_s,
        lane_label=lane_label,
    )
    summary = [
        *gpu_payload["summary"],
        tier4_summary,
    ]
    return _combine_phase_payload(
        provenance=dict(gpu_payload["provenance"]),
        lane_label=lane_label,
        phase="aggregate",
        rungs={
            TIER1_PARITY_RUNG: gpu_payload["rungs"][TIER1_PARITY_RUNG],
            TIER2_PERFORMANCE_RUNG: gpu_payload["rungs"][TIER2_PERFORMANCE_RUNG],
            TIER3_INIT_RUNG: gpu_payload["rungs"][TIER3_INIT_RUNG],
            "tier4_adjoint_fd_cpu": cpu_payload["rungs"]["tier4_adjoint_fd_cpu"],
            "tier4_adjoint_fd_lane": gpu_payload["rungs"]["tier4_adjoint_fd_lane"],
        },
        summary=summary,
        probe_timings={
            **{
                str(key): float(value)
                for key, value in gpu_payload.get("probe_timings", {}).items()
            },
            **{
                str(key): float(value)
                for key, value in cpu_payload.get("probe_timings", {}).items()
            },
        },
        phase_inputs={
            "gpu": gpu_payload.get("provenance"),
            "cpu": cpu_payload.get("provenance"),
        },
    )


def _print_tier5_payload(payload: dict[str, Any]) -> None:
    print("\nTier 5 summary")
    print("--------------")
    for item in payload["summary"]:
        print(_render_tier5_summary_line(item))
    print(f"total outer elapsed: {payload['aggregate']['total_outer_elapsed_s']:.2f}s")
    performance_contract = payload["aggregate"].get("performance_contract")
    warm_source = None
    if performance_contract is not None:
        print(
            "performance contract: "
            f"use {TIER1_PARITY_RUNG} for parity, "
            f"{TIER2_PERFORMANCE_RUNG}.outer_speedup_vs_cpu for the main cold first-run wall-clock headline"
        )
        warm_source = performance_contract.get("warm_steady_state_source")
    if warm_source is not None:
        print(
            "secondary steady-state metric: "
            f"{TIER2_PERFORMANCE_RUNG}.{warm_source['metric_path'].split('.')[-1]}"
        )
    aggregate_failures = [
        *payload["aggregate"]["performance_failures"],
        *payload["aggregate"]["sharding_failures"],
    ]
    if aggregate_failures:
        print("Tier 5 performance gate failed")
        for failure in aggregate_failures:
            print(f"  - {failure}")
        raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.phase == "aggregate":
        if args.gpu_input_json is None or args.cpu_input_json is None:
            raise SystemExit(
                "--gpu-input-json and --cpu-input-json are required when --phase=aggregate."
            )
        aggregate_payload = _build_aggregate_payload(
            gpu_payload=load_json(args.gpu_input_json),
            cpu_payload=load_json(args.cpu_input_json),
        )
        write_json(args.output_json, aggregate_payload)
        _print_tier5_payload(aggregate_payload)
        return

    runtime_jax, runtime_jaxlib = _runtime_modules()
    benchmark_mode = bool(args.benchmark_mode)
    provenance = build_provenance(
        runtime_jax,
        runtime_jaxlib,
        title="Tier 5 trusted-fixture performance characterization",
        extra={
            "lane": resolve_probe_lane(optimizer_backend=args.optimizer_backend),
            "fixture": "trusted-public-lane",
            "platform_request": args.platform,
            "plasma_surf_filename": args.plasma_surf_filename,
            "stage2_seed_path": args.stage2_bs_path,
            "stage2_nphi": int(args.stage2_nphi),
            "stage2_ntheta": int(args.stage2_ntheta),
            "single_stage_nphi": int(args.single_stage_nphi),
            "single_stage_ntheta": int(args.single_stage_ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "stage2_maxiter": int(args.maxiter),
            "optimizer_backend": args.optimizer_backend,
            "benchmark_mode": benchmark_mode,
            "fd_samples": int(args.samples),
            "fd_eps": float(args.eps),
            "phase": args.phase,
            "compile_behavior": describe_compile_behavior(uses_subprocesses=True),
        },
    )
    print_provenance(provenance)

    lane_label = "jax-cpu" if args.platform == "cpu" else f"jax-{args.platform}"

    if args.phase == "cpu":
        tier4_cpu_payload, tier4_cpu_outer = _run_tier4_cpu_probe(args)
        payload = _combine_phase_payload(
            provenance=provenance,
            lane_label=lane_label,
            phase="cpu",
            rungs={"tier4_adjoint_fd_cpu": tier4_cpu_payload},
            summary=[],
            probe_timings={"tier4_adjoint_fd_cpu": tier4_cpu_outer},
        )
    else:
        (
            tier1b_payload,
            tier1b_outer,
            tier2_payload,
            tier2_outer,
            tier3_payload,
            tier3_outer,
        ) = _run_stage123_probes(args)
        tier123_summary = _summarize_stage123_probes(
            tier1_payload=tier1b_payload,
            tier1_outer=tier1b_outer,
            tier2_payload=tier2_payload,
            tier2_outer=tier2_outer,
            tier3_payload=tier3_payload,
            tier3_outer=tier3_outer,
            lane_label=lane_label,
        )
        phase_rungs = {
            TIER1_PARITY_RUNG: tier1b_payload,
            TIER2_PERFORMANCE_RUNG: tier2_payload,
            TIER3_INIT_RUNG: tier3_payload,
        }
        phase_probe_timings = {
            TIER1_PARITY_RUNG: tier1b_outer,
            TIER2_PERFORMANCE_RUNG: tier2_outer,
            TIER3_INIT_RUNG: tier3_outer,
        }
        if args.phase == "gpu":
            tier4_lane_payload, tier4_lane_outer = _run_tier4_lane_probe(args)
            phase_rungs["tier4_adjoint_fd_lane"] = tier4_lane_payload
            phase_probe_timings["tier4_adjoint_fd_lane"] = tier4_lane_outer
        else:
            tier4_pair = _run_tier4_pair(args)
            phase_rungs["tier4_adjoint_fd_cpu"] = tier4_pair["cpu_payload"]
            phase_rungs["tier4_adjoint_fd_lane"] = tier4_pair["lane_payload"]
            phase_probe_timings.update(tier4_pair["probe_timings"])
        payload = _combine_phase_payload(
            provenance=provenance,
            lane_label=lane_label,
            phase=args.phase,
            rungs=phase_rungs,
            summary=(
                tier123_summary
                if args.phase == "gpu"
                else [*tier123_summary, tier4_pair["summary"]]
            ),
            probe_timings=phase_probe_timings,
        )

    write_json(args.output_json, payload)
    _print_tier5_payload(payload)


if __name__ == "__main__":
    main()
