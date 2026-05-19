"""Dedicated grouped-adjoint memory probe on the real reduced single-stage fixture."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import resource
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from benchmarks.adjoint_probe_common import (
    accumulate_grouped_adjoint_dofs_gradient,
    compute_adjoint_state,
    compute_gradient_l2_metrics,
    iter_grouped_adjoint_cotangents,
)
from benchmarks.single_stage_backend_routing import (
    resolve_boozer_limited_memory,
    resolve_boozer_optimizer_backend,
)
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
    build_real_single_stage_init_fixture,
)
from benchmarks.validation_ladder_common import (
    apply_compilation_cache_policy,
    apply_requested_platform,
    bootstrap_local_simsopt,
    build_provenance,
    describe_compile_behavior,
    evaluate_grouped_adjoint_memory_budget,
    grouped_adjoint_memory_budget,
    maybe_initialize_distributed_runtime,
    preparse_platform,
    print_provenance,
    query_gpu_memory_mb,
    require_x64_runtime,
    resolve_probe_lane,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)
apply_compilation_cache_policy()
bootstrap_local_simsopt()

import jax
import jaxlib

maybe_initialize_distributed_runtime()
jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="Grouped adjoint memory probe")

_GROUPED_ADJOINT_FIXTURE = "real-single-stage-init"
_GROUPED_VJP_MIN_HOT_PATH_FRACTION = 0.10
_GROUPED_VJP_IMMEDIATE_KILL_FRACTION = 0.05
_GROUPED_VJP_MIN_STEADY_STATE_SPEEDUP_FRACTION = 0.25
_GROUPED_VJP_MIN_PEAK_DEVICE_MEMORY_REDUCTION_FRACTION = 0.40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure grouped-adjoint memory behavior on the real reduced single-stage fixture."
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
        help="Path to write structured probe results.",
    )
    parser.add_argument(
        "--plasma-surf-filename",
        default=DEFAULT_PLASMA_SURF_FILENAME,
        help="VMEC equilibrium filename for the real single-stage fixture.",
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
        "--nphi",
        type=int,
        default=DEFAULT_SMOKE_NPHI,
        help="Surface toroidal grid points.",
    )
    parser.add_argument(
        "--ntheta",
        type=int,
        default=DEFAULT_SMOKE_NTHETA,
        help="Surface poloidal grid points.",
    )
    parser.add_argument(
        "--mpol",
        type=int,
        default=DEFAULT_SMOKE_MPOL,
        help="Surface poloidal mode count.",
    )
    parser.add_argument(
        "--ntor",
        type=int,
        default=DEFAULT_SMOKE_NTOR,
        help="Surface toroidal mode count.",
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
        choices=(DEFAULT_OPTIMIZER_BACKEND,),
        default=DEFAULT_OPTIMIZER_BACKEND,
        help="JAX target-lane optimizer backend for the grouped-adjoint probe.",
    )
    parser.add_argument(
        "--boozer-optimizer-backend",
        choices=(DEFAULT_OPTIMIZER_BACKEND,),
        default=None,
        help=(
            "Optional override for the inner JAX Boozer LS backend. "
            "When provided it must stay ondevice."
        ),
    )
    parser.add_argument(
        "--boozer-limited-memory",
        action="store_true",
        help=(
            "Request the limited-memory ondevice Boozer LS route. "
            "This only takes effect when --boozer-optimizer-backend=ondevice."
        ),
    )
    parser.add_argument(
        "--device-memory-profile-out",
        default=None,
        help=(
            "Optional JAX device-memory profile artifact to write after the "
            "grouped-adjoint probe."
        ),
    )
    parser.add_argument(
        "--grouped-vjp-timing-repeats",
        type=int,
        default=3,
        help=(
            "Total streamed grouped-VJP timing passes to record. The first "
            "pass is the production DOF-gradient projection; later passes are "
            "warm streaming-only cache-stability probes."
        ),
    )
    parser.add_argument(
        "--record-jax-compile-diagnostics",
        action="store_true",
        help=(
            "Enable jax_log_compiles and jax_explain_cache_misses while "
            "recording grouped-VJP timing passes."
        ),
    )
    parser.add_argument(
        "--baseline-json",
        default=None,
        help=(
            "Optional prior-HEAD grouped_adjoint_memory_probe JSON. When supplied, "
            "the probe enforces the ship gate: >=25% steady-state grouped-VJP "
            "speedup or >=40% peak device-memory reduction."
        ),
    )
    return parser.parse_args()


def _rss_high_water_mark_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def record_memory_snapshot(
    label: str,
    started_at: float,
    **extra: float | str | None,
) -> dict[str, float | str | None]:
    snapshot = {
        "label": label,
        "elapsed_s": float(time.perf_counter() - started_at),
        "rss_mb": _rss_high_water_mark_mb(),
        "gpu_memory_mb": query_gpu_memory_mb(),
    }
    snapshot.update(extra)
    return snapshot


def print_memory_snapshot(snapshot: dict[str, float | str | None]) -> None:
    gpu_memory = snapshot["gpu_memory_mb"]
    group_count = snapshot.get("group_count")
    iteration = snapshot.get("iteration")
    objective = snapshot.get("objective")
    grad_inf = snapshot.get("grad_inf")
    method = snapshot.get("method")
    group_suffix = (
        f", groups={int(group_count)}" if isinstance(group_count, (int, float)) else ""
    )
    progress_suffix = (
        f", iter={int(iteration)}" if isinstance(iteration, (int, float)) else ""
    )
    objective_suffix = (
        f", fun={float(objective):.6e}" if isinstance(objective, (int, float)) else ""
    )
    grad_suffix = (
        f", ||grad||_inf={float(grad_inf):.3e}"
        if isinstance(grad_inf, (int, float))
        else ""
    )
    method_suffix = f", method={method}" if isinstance(method, str) else ""
    gpu_suffix = f"{float(gpu_memory):.2f} MB" if gpu_memory is not None else "n/a"
    print(
        "[snapshot] "
        f"{snapshot['label']}: "
        f"elapsed={float(snapshot['elapsed_s']):.2f}s, "
        f"rss={float(snapshot['rss_mb']):.2f} MB, "
        f"gpu={gpu_suffix}"
        f"{group_suffix}",
        end="",
        flush=False,
    )
    print(
        f"{progress_suffix}{objective_suffix}{grad_suffix}{method_suffix}",
        flush=True,
    )


def _peak_snapshot_value(
    snapshots: list[dict[str, float | str | None]],
    key: str,
) -> float | None:
    values = [snapshot[key] for snapshot in snapshots if snapshot[key] is not None]
    if not values:
        return None
    return max(float(value) for value in values)


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _block_until_ready(value: object) -> None:
    for leaf in jax.tree.leaves(value):
        block_until_ready = getattr(leaf, "block_until_ready", None)
        if callable(block_until_ready):
            block_until_ready()


class _GroupedVJPTimingRecorder:
    """Record streamed grouped-VJP timing without retaining group cotangents."""

    def __init__(self, requested_stream_pass_count: int):
        self.requested_stream_pass_count = int(requested_stream_pass_count)
        self.passes: list[dict[str, Any]] = []

    def timed_cotangents(self, jr_jax, adjoint: np.ndarray, *, label: str):
        grouped_iter = iter(iter_grouped_adjoint_cotangents(jr_jax, adjoint))
        group_times_s: list[float] = []
        pass_started_at = time.perf_counter()
        while True:
            group_started_at = time.perf_counter()
            try:
                d_coil_array, coil_group_indices = next(grouped_iter)
            except StopIteration:
                break
            _block_until_ready(d_coil_array)
            group_times_s.append(float(time.perf_counter() - group_started_at))
            yield d_coil_array, coil_group_indices

        total_s = float(time.perf_counter() - pass_started_at)
        self.passes.append(
            {
                "label": label,
                "group_count": int(len(group_times_s)),
                "total_s": total_s,
                "group_times_s": group_times_s,
                "first_group_s": (float(group_times_s[0]) if group_times_s else None),
                "steady_state_group_median_s": _median_or_none(group_times_s[1:]),
            }
        )

    def summary(self, *, total_representative_run_wall_s: float) -> dict[str, Any]:
        first_pass = self.passes[0] if self.passes else {}
        warm_passes = self.passes[1:]
        warm_stream_times_s = [
            float(passed["total_s"])
            for passed in warm_passes
            if passed.get("total_s") is not None
        ]
        warm_group_times_s = [
            float(group_time)
            for passed in warm_passes
            for group_time in passed.get("group_times_s", [])
        ]
        first_pass_group_times = list(first_pass.get("group_times_s", []))
        total_wall_s = float(total_representative_run_wall_s)
        steady_state_grouped_vjp_time_s = _median_or_none(warm_stream_times_s)
        wall_fraction = (
            None
            if steady_state_grouped_vjp_time_s is None or total_wall_s <= 0.0
            else float(steady_state_grouped_vjp_time_s) / total_wall_s
        )
        return {
            "requested_stream_pass_count": int(self.requested_stream_pass_count),
            "stream_pass_count": int(len(self.passes)),
            "group_count": first_pass.get("group_count"),
            "first_stream_s": first_pass.get("total_s"),
            "first_compile_time_s": first_pass.get("first_group_s"),
            "first_compile_time_note": (
                "First grouped-VJP group call measured with block_until_ready; "
                "includes compilation plus first group execution."
            ),
            "first_stream_steady_state_group_median_s": _median_or_none(
                [float(value) for value in first_pass_group_times[1:]]
            ),
            "warm_stream_times_s": warm_stream_times_s,
            "steady_state_grouped_vjp_time_s": steady_state_grouped_vjp_time_s,
            "steady_state_grouped_vjp_per_group_s": _median_or_none(warm_group_times_s),
            "total_representative_run_wall_s": total_wall_s,
            "steady_state_grouped_vjp_wall_fraction": wall_fraction,
            "passes": self.passes,
        }


def _consume_grouped_vjp_stream(grouped_cotangents) -> int:
    group_count = 0
    for _d_coil_array, _coil_group_indices in grouped_cotangents:
        group_count += 1
    return group_count


class _JaxCompileDiagnosticsRecorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.compile_messages: list[str] = []
        self.cache_miss_messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        lower_message = message.lower()
        if "compil" in lower_message:
            self.compile_messages.append(message)
        if "cache miss" in lower_message:
            self.cache_miss_messages.append(message)

    def summary(self) -> dict[str, Any]:
        return {
            "compile_event_count": int(len(self.compile_messages)),
            "cache_miss_count": int(len(self.cache_miss_messages)),
            "compile_messages": list(self.compile_messages),
            "cache_miss_messages": list(self.cache_miss_messages),
        }


@contextmanager
def _maybe_record_jax_compile_diagnostics(enabled: bool):
    if not enabled:
        yield None
        return

    logger = logging.getLogger("jax")
    recorder = _JaxCompileDiagnosticsRecorder()
    previous_level = logger.level
    previous_propagate = logger.propagate
    override_level = (
        previous_level == logging.NOTSET or previous_level > logging.WARNING
    )
    if override_level:
        logger.setLevel(logging.WARNING)
    logger.propagate = False
    previous_explain_cache_misses = bool(jax.config.jax_explain_cache_misses)
    logger.addHandler(recorder)
    try:
        jax.config.update("jax_explain_cache_misses", True)
        with jax.log_compiles(True):
            yield recorder
    finally:
        logger.removeHandler(recorder)
        jax.config.update(
            "jax_explain_cache_misses",
            previous_explain_cache_misses,
        )
        logger.propagate = previous_propagate
        if override_level:
            logger.setLevel(previous_level)


def _recorder_summary(
    recorder: _JaxCompileDiagnosticsRecorder | None,
) -> dict[str, Any] | None:
    return recorder.summary() if recorder is not None else None


def _build_grouped_vjp_cache_stability(
    *,
    diagnostics_requested: bool,
    production_recorder: _JaxCompileDiagnosticsRecorder | None,
    warm_recorder: _JaxCompileDiagnosticsRecorder | None,
    warm_pass_count: int,
) -> dict[str, Any]:
    diagnostics_enabled = bool(diagnostics_requested)
    warm_summary = _recorder_summary(warm_recorder)
    warm_compile_event_count = (
        None if warm_summary is None else int(warm_summary["compile_event_count"])
    )
    warm_cache_miss_count = (
        None if warm_summary is None else int(warm_summary["cache_miss_count"])
    )
    unexpected_steady_state_recompile = (
        None
        if warm_compile_event_count is None or warm_pass_count == 0
        else warm_compile_event_count > 0
    )
    return {
        "diagnostics_requested": bool(diagnostics_requested),
        "diagnostics_enabled": diagnostics_enabled,
        "jax_log_compiles": diagnostics_enabled,
        "jax_explain_cache_misses": diagnostics_enabled,
        "warm_pass_count": int(warm_pass_count),
        "warm_compile_event_count": warm_compile_event_count,
        "warm_cache_miss_count": warm_cache_miss_count,
        "unexpected_steady_state_recompile": unexpected_steady_state_recompile,
        "production_pass": _recorder_summary(production_recorder),
        "warm_passes": warm_summary,
    }


def _build_grouped_adjoint_metrics(
    adjoint: np.ndarray,
    adjoint_residual_rel: float,
    implicit_gradient_norm: float,
    implicit_gradient_finite: bool,
    snapshots: list[dict[str, float | str | None]],
    grouped_vjp_timings: dict[str, Any],
    cache_stability: dict[str, Any],
) -> dict[str, Any]:
    adjoint_norm = float(np.linalg.norm(adjoint))
    return {
        "adjoint_residual_rel": float(adjoint_residual_rel),
        "adjoint_finite": bool(np.all(np.isfinite(adjoint))),
        "adjoint_norm": adjoint_norm,
        "implicit_gradient_finite": bool(implicit_gradient_finite),
        "implicit_gradient_norm": float(implicit_gradient_norm),
        "snapshots": snapshots,
        "grouped_vjp_timings": grouped_vjp_timings,
        "cache_stability": cache_stability,
    }


def _load_baseline_payload(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _positive_float(value: object, *, field: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field} must be positive and finite.")
    return number


def _payload_positive_float(
    payload: dict[str, Any],
    section: str,
    key: str,
) -> float:
    return _positive_float(payload[section][key], field=f"{section}.{key}")


def _peak_device_memory_mb_from_metrics(metrics: dict[str, Any]) -> float | None:
    return _peak_snapshot_value(list(metrics["snapshots"]), "gpu_memory_mb")


def _build_grouped_adjoint_baseline_comparison(
    *,
    current_metrics: dict[str, Any],
    baseline_payload: dict[str, Any],
    baseline_json: str,
) -> dict[str, Any]:
    current_time_s = _positive_float(
        current_metrics["grouped_vjp_timings"]["steady_state_grouped_vjp_time_s"],
        field="current.timings.steady_state_grouped_vjp_time_s",
    )
    baseline_time_s = _payload_positive_float(
        baseline_payload,
        "timings",
        "steady_state_grouped_vjp_time_s",
    )
    time_ratio = current_time_s / baseline_time_s
    speedup_fraction = 1.0 - time_ratio
    speedup_gate_passed = (
        speedup_fraction >= _GROUPED_VJP_MIN_STEADY_STATE_SPEEDUP_FRACTION
    )

    current_peak_device_memory_mb = _peak_device_memory_mb_from_metrics(current_metrics)
    baseline_peak_device_memory_mb = baseline_payload["memory"].get(
        "peak_gpu_memory_mb"
    )
    memory_reduction_fraction = None
    memory_gate_passed = False
    if (
        current_peak_device_memory_mb is not None
        and baseline_peak_device_memory_mb is not None
    ):
        baseline_peak_device_memory_mb = _positive_float(
            baseline_peak_device_memory_mb,
            field="baseline.memory.peak_gpu_memory_mb",
        )
        current_peak_device_memory_mb = _positive_float(
            current_peak_device_memory_mb,
            field="current.memory.peak_gpu_memory_mb",
        )
        memory_ratio = current_peak_device_memory_mb / baseline_peak_device_memory_mb
        memory_reduction_fraction = 1.0 - memory_ratio
        memory_gate_passed = (
            memory_reduction_fraction
            >= _GROUPED_VJP_MIN_PEAK_DEVICE_MEMORY_REDUCTION_FRACTION
        )

    return {
        "baseline_json": str(baseline_json),
        "min_steady_state_speedup_fraction": (
            _GROUPED_VJP_MIN_STEADY_STATE_SPEEDUP_FRACTION
        ),
        "min_peak_device_memory_reduction_fraction": (
            _GROUPED_VJP_MIN_PEAK_DEVICE_MEMORY_REDUCTION_FRACTION
        ),
        "baseline_steady_state_grouped_vjp_time_s": baseline_time_s,
        "current_steady_state_grouped_vjp_time_s": current_time_s,
        "current_to_baseline_time_ratio": time_ratio,
        "steady_state_speedup_fraction": speedup_fraction,
        "speedup_gate_passed": bool(speedup_gate_passed),
        "baseline_peak_device_memory_mb": baseline_peak_device_memory_mb,
        "current_peak_device_memory_mb": current_peak_device_memory_mb,
        "peak_device_memory_reduction_fraction": memory_reduction_fraction,
        "memory_gate_passed": bool(memory_gate_passed),
        "passed": bool(speedup_gate_passed or memory_gate_passed),
    }


def _build_grouped_adjoint_payload(
    *,
    provenance: dict[str, Any],
    fixture: dict[str, object],
    base_result: dict[str, Any],
    metrics: dict[str, Any],
    memory_budget: dict[str, float | None],
    failures: list[str],
    snapshots: list[dict[str, float | str | None]],
    boozer_limited_memory: bool,
    boozer_limited_memory_requested: bool,
    device_memory_profile_path: str | None,
) -> dict[str, Any]:
    return {
        "provenance": provenance,
        "baseline": {
            "solve_success": bool(base_result.get("success", False)),
            "equilibrium_path": str(fixture["equilibrium_path"]),
            "stage2_bs_path": str(fixture["stage2_bs_path"]),
            "boozer_optimizer_backend": fixture["boozer_optimizer_backend"],
            "optimizer_method": base_result.get("optimizer_method"),
            "boozer_limited_memory": bool(boozer_limited_memory),
            "boozer_limited_memory_requested": bool(boozer_limited_memory_requested),
        },
        "grouped_adjoint": {
            "adjoint_residual_rel": metrics["adjoint_residual_rel"],
            "adjoint_norm": metrics["adjoint_norm"],
            "adjoint_finite": metrics["adjoint_finite"],
            "implicit_gradient_norm": metrics["implicit_gradient_norm"],
            "implicit_gradient_finite": metrics["implicit_gradient_finite"],
        },
        "timings": metrics["grouped_vjp_timings"],
        "cache_stability": metrics["cache_stability"],
        "baseline_comparison": metrics.get("baseline_comparison"),
        "memory": {
            "snapshots": snapshots,
            "peak_rss_mb": _peak_snapshot_value(snapshots, "rss_mb"),
            "peak_gpu_memory_mb": _peak_snapshot_value(snapshots, "gpu_memory_mb"),
            "budget": memory_budget,
            "device_memory_profile_path": device_memory_profile_path,
        },
        "failures": failures,
        "passed": not failures,
    }


def _build_probe_provenance(args: argparse.Namespace) -> dict[str, Any]:
    resolved_boozer_optimizer_backend = resolve_boozer_optimizer_backend(
        args.optimizer_backend,
        args.boozer_optimizer_backend,
    )
    boozer_limited_memory_requested = bool(args.boozer_limited_memory)
    boozer_limited_memory = resolve_boozer_limited_memory(
        resolved_boozer_optimizer_backend,
        boozer_limited_memory_requested,
    )
    return build_provenance(
        jax,
        jaxlib,
        title="Grouped adjoint memory probe",
        extra={
            "lane": resolve_probe_lane(optimizer_backend=args.optimizer_backend),
            "fixture": "real-single-stage-init",
            "platform_request": args.platform,
            "plasma_surf_filename": args.plasma_surf_filename,
            "stage2_seed_path": str(Path(args.stage2_bs_path)),
            "optimizer_backend": args.optimizer_backend,
            "boozer_optimizer_backend": resolved_boozer_optimizer_backend,
            "boozer_optimizer_backend_requested": args.boozer_optimizer_backend,
            "boozer_limited_memory": boozer_limited_memory,
            "boozer_limited_memory_requested": boozer_limited_memory_requested,
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "compile_behavior": describe_compile_behavior(uses_subprocesses=False),
        },
    )


def _memory_budget_is_blocking(
    snapshots: list[dict[str, float | str | None]],
    budget: dict[str, float | None] | None,
) -> bool:
    if budget is None:
        return False
    peak_rss_mb = _peak_snapshot_value(snapshots, "rss_mb")
    peak_gpu_memory_mb = _peak_snapshot_value(snapshots, "gpu_memory_mb")
    max_peak_rss_mb = budget.get("max_peak_rss_mb")
    max_peak_gpu_memory_mb = budget.get("max_peak_gpu_memory_mb")
    return (
        peak_rss_mb is not None
        and max_peak_rss_mb is not None
        and peak_rss_mb > max_peak_rss_mb
    ) or (
        peak_gpu_memory_mb is not None
        and max_peak_gpu_memory_mb is not None
        and peak_gpu_memory_mb > max_peak_gpu_memory_mb
    )


def _representative_run_wall_s(
    snapshots: list[dict[str, float | str | None]],
) -> float:
    return float(
        next(
            snapshot["elapsed_s"]
            for snapshot in reversed(snapshots)
            if snapshot["label"] == "after_dofs_gradient_projection"
        )
    )


def evaluate_grouped_adjoint_memory_probe(
    metrics: dict[str, Any],
    *,
    budget: dict[str, float | None] | None = None,
) -> list[str]:
    failures: list[str] = []
    snapshots = list(metrics.get("snapshots", []))
    required_labels = {
        "start",
        "after_stage2_results_load",
        "after_biotsavart_load",
        "after_surface_seed_setup",
        "after_boozer_surface_fit",
        "after_boozer_setup",
        "after_boozer_lbfgs",
        "before_boozer_newton",
        "after_boozer_newton",
        "after_boozer_solve",
        "after_boozer_postprocess",
        "after_fixture",
        "after_objective",
        "after_adjoint_solve",
        "before_grouped_adjoint_vjp",
        "after_grouped_adjoint_vjp_first_group",
        "after_grouped_adjoint_vjp_end",
        "after_dofs_gradient_projection",
        "after_norm_metrics",
    }
    labels = {str(snapshot.get("label")) for snapshot in snapshots}
    missing_labels = sorted(required_labels - labels)
    if missing_labels:
        failures.append(
            "Grouped-adjoint memory probe did not record all required snapshots: "
            + ", ".join(missing_labels)
        )
    if not bool(metrics.get("adjoint_finite", False)):
        failures.append(
            "Grouped-adjoint memory probe produced a non-finite adjoint state."
        )
    if not bool(metrics.get("implicit_gradient_finite", False)):
        failures.append(
            "Grouped-adjoint memory probe produced a non-finite implicit gradient."
        )
    if float(metrics.get("implicit_gradient_norm", 0.0)) <= 0.0:
        failures.append(
            "Grouped-adjoint memory probe produced a zero implicit gradient."
        )
    if not np.isfinite(float(metrics.get("adjoint_residual_rel", np.inf))):
        failures.append("Grouped-adjoint memory probe adjoint residual is not finite.")
    timings = metrics.get("grouped_vjp_timings")
    if not isinstance(timings, dict):
        failures.append(
            "Grouped-adjoint memory probe did not record grouped-VJP timings."
        )
    else:
        required_timing_keys = {
            "first_compile_time_s",
            "steady_state_grouped_vjp_time_s",
            "total_representative_run_wall_s",
        }
        missing_timing_keys = sorted(required_timing_keys - set(timings))
        if missing_timing_keys:
            failures.append(
                "Grouped-adjoint memory probe did not record required timing fields: "
                + ", ".join(missing_timing_keys)
            )
        for key in required_timing_keys:
            value = timings.get(key)
            if not isinstance(value, (int, float)) or not np.isfinite(float(value)):
                failures.append(
                    f"Grouped-adjoint memory probe timing {key} is not finite."
                )
        steady_state_s = timings.get("steady_state_grouped_vjp_time_s")
        total_wall_s = timings.get("total_representative_run_wall_s")
        if isinstance(steady_state_s, (int, float)) and isinstance(
            total_wall_s,
            (int, float),
        ):
            total_wall_s = float(total_wall_s)
            if total_wall_s > 0.0:
                wall_fraction = float(steady_state_s) / total_wall_s
                memory_blocking = _memory_budget_is_blocking(snapshots, budget)
                if (
                    wall_fraction < _GROUPED_VJP_IMMEDIATE_KILL_FRACTION
                    and not memory_blocking
                ):
                    failures.append(
                        "Grouped-adjoint steady-state VJP is below 5% of "
                        "representative wall time and is not a hot path "
                        f"({wall_fraction:.3%})."
                    )
                elif (
                    wall_fraction < _GROUPED_VJP_MIN_HOT_PATH_FRACTION
                    and not memory_blocking
                ):
                    failures.append(
                        "Grouped-adjoint steady-state VJP is below the 10% "
                        "representativeness gate without a peak-memory blocker "
                        f"({wall_fraction:.3%})."
                    )
    cache_stability = metrics.get("cache_stability")
    if not isinstance(cache_stability, dict):
        failures.append(
            "Grouped-adjoint memory probe did not record cache-stability metadata."
        )
    elif cache_stability.get("unexpected_steady_state_recompile") is True:
        failures.append(
            "Grouped-adjoint memory probe recorded steady-state grouped-VJP recompilation."
        )
    baseline_comparison = metrics.get("baseline_comparison")
    if baseline_comparison is not None and not bool(
        baseline_comparison.get("passed", False)
    ):
        failures.append(
            "Grouped-adjoint ship gate missed both thresholds: "
            "requires >=25% steady-state grouped-VJP speedup or >=40% peak "
            "device-memory reduction versus baseline."
        )
    if budget is not None:
        metric_with_peaks = dict(metrics)
        metric_with_peaks["peak_rss_mb"] = _peak_snapshot_value(snapshots, "rss_mb")
        metric_with_peaks["peak_gpu_memory_mb"] = _peak_snapshot_value(
            snapshots, "gpu_memory_mb"
        )
        failures.extend(
            evaluate_grouped_adjoint_memory_budget(metric_with_peaks, budget)
        )
    return failures


def _save_device_memory_profile(path: str) -> str:
    profile_path = Path(path)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    jax.device_put(np.asarray(0.0, dtype=np.float64)).block_until_ready()
    jax.profiler.save_device_memory_profile(str(profile_path))
    return str(profile_path)


def main() -> None:
    args = parse_args()
    resolved_boozer_optimizer_backend = resolve_boozer_optimizer_backend(
        args.optimizer_backend,
        args.boozer_optimizer_backend,
    )
    boozer_limited_memory_requested = bool(args.boozer_limited_memory)
    boozer_limited_memory = resolve_boozer_limited_memory(
        resolved_boozer_optimizer_backend,
        boozer_limited_memory_requested,
    )
    provenance = _build_probe_provenance(args)
    print_provenance(provenance)

    started_at = time.perf_counter()
    snapshots: list[dict[str, float | str | None]] = []

    def capture_snapshot(label: str, **extra: float | str | None) -> None:
        snapshot = record_memory_snapshot(label, started_at, **extra)
        snapshots.append(snapshot)
        print_memory_snapshot(snapshot)

    capture_snapshot("start")

    fixture = build_real_single_stage_init_fixture(
        backend="jax",
        plasma_surf_filename=args.plasma_surf_filename,
        equilibria_dir=args.equilibria_dir,
        equilibrium_path=args.equilibrium_path,
        stage2_bs_path=args.stage2_bs_path,
        nphi=args.nphi,
        ntheta=args.ntheta,
        mpol=args.mpol,
        ntor=args.ntor,
        vol_target=args.vol_target,
        iota_target=args.iota_target,
        optimizer_backend=args.optimizer_backend,
        boozer_optimizer_backend=args.boozer_optimizer_backend,
        boozer_limited_memory=boozer_limited_memory,
        on_stage=capture_snapshot,
    )
    capture_snapshot("after_fixture")

    base_result = fixture["boozer_surface"].res
    if base_result is None or not base_result.get("success", False):
        raise RuntimeError(
            "Baseline Boozer solve failed; cannot probe grouped-adjoint memory."
        )

    from simsopt.geo.surfaceobjectives_jax import BoozerResidualJAX

    jr_jax = BoozerResidualJAX(fixture["boozer_surface"], fixture["bs"])
    capture_snapshot("after_objective")

    adjoint, adjoint_residual_rel = compute_adjoint_state(jr_jax)
    capture_snapshot("after_adjoint_solve")

    grouped_vjp_timing_recorder = _GroupedVJPTimingRecorder(
        args.grouped_vjp_timing_repeats
    )
    with _maybe_record_jax_compile_diagnostics(
        bool(args.record_jax_compile_diagnostics)
    ) as production_compile_recorder:
        grouped_cotangents = grouped_vjp_timing_recorder.timed_cotangents(
            jr_jax,
            adjoint,
            label="production_dofs_gradient_projection",
        )
        implicit_gradient = accumulate_grouped_adjoint_dofs_gradient(
            fixture["bs"],
            grouped_cotangents,
            on_stage=capture_snapshot,
        )

    warm_pass_count = max(0, int(args.grouped_vjp_timing_repeats) - 1)
    with _maybe_record_jax_compile_diagnostics(
        bool(args.record_jax_compile_diagnostics) and warm_pass_count > 0
    ) as warm_compile_recorder:
        for warm_pass_index in range(warm_pass_count):
            _consume_grouped_vjp_stream(
                grouped_vjp_timing_recorder.timed_cotangents(
                    jr_jax,
                    adjoint,
                    label=f"warm_stream_{warm_pass_index + 1}",
                )
            )

    implicit_gradient_norm, implicit_gradient_finite = compute_gradient_l2_metrics(
        implicit_gradient
    )
    capture_snapshot("after_norm_metrics")
    total_representative_run_wall_s = _representative_run_wall_s(snapshots)
    grouped_vjp_timings = grouped_vjp_timing_recorder.summary(
        total_representative_run_wall_s=float(total_representative_run_wall_s)
    )
    cache_stability = _build_grouped_vjp_cache_stability(
        diagnostics_requested=bool(args.record_jax_compile_diagnostics),
        production_recorder=production_compile_recorder,
        warm_recorder=warm_compile_recorder,
        warm_pass_count=warm_pass_count,
    )

    metrics = _build_grouped_adjoint_metrics(
        adjoint,
        adjoint_residual_rel,
        implicit_gradient_norm,
        implicit_gradient_finite,
        snapshots,
        grouped_vjp_timings,
        cache_stability,
    )
    baseline_payload = _load_baseline_payload(args.baseline_json)
    if baseline_payload is not None:
        metrics["baseline_comparison"] = _build_grouped_adjoint_baseline_comparison(
            current_metrics=metrics,
            baseline_payload=baseline_payload,
            baseline_json=args.baseline_json,
        )
    budget_platform = (
        args.platform if args.platform != "auto" else str(jax.default_backend())
    )
    memory_budget = grouped_adjoint_memory_budget(
        fixture=_GROUPED_ADJOINT_FIXTURE,
        platform=budget_platform,
    )
    failures = evaluate_grouped_adjoint_memory_probe(metrics, budget=memory_budget)
    device_memory_profile_path = None
    if args.device_memory_profile_out is not None:
        device_memory_profile_path = _save_device_memory_profile(
            args.device_memory_profile_out
        )

    payload = _build_grouped_adjoint_payload(
        provenance=provenance,
        fixture=fixture,
        base_result=base_result,
        metrics=metrics,
        memory_budget=memory_budget,
        failures=failures,
        snapshots=snapshots,
        boozer_limited_memory=boozer_limited_memory,
        boozer_limited_memory_requested=boozer_limited_memory_requested,
        device_memory_profile_path=device_memory_profile_path,
    )
    write_json(args.output_json, payload)
    if failures:
        print("GROUPED ADJOINT MEMORY PROBE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("GROUPED ADJOINT MEMORY PROBE PASSED")


if __name__ == "__main__":
    main()
