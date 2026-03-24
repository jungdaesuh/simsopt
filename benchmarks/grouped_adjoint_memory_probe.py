"""Dedicated grouped-adjoint memory probe on the real reduced single-stage fixture."""

from __future__ import annotations

import argparse
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
    accumulate_grouped_adjoint_derivative,
    compute_adjoint_state,
    compute_derivative_l2_metrics,
    iter_grouped_adjoint_cotangents,
)
from benchmarks.single_stage_smoke_fixture import (
    DEFAULT_EQUILIBRIA_DIR,
    DEFAULT_IOTA_TARGET,
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

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)
require_x64_runtime(jax, context="Grouped adjoint memory probe")


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
        choices=("scipy", "hybrid", "ondevice"),
        default="ondevice",
        help="JAX Boozer optimizer backend for the grouped-adjoint probe.",
    )
    return parser.parse_args()


def _rss_high_water_mark_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return float(rss) / (1024.0 * 1024.0)
    return float(rss) / 1024.0


def record_memory_snapshot(label: str, started_at: float) -> dict[str, float | str | None]:
    return {
        "label": label,
        "elapsed_s": float(time.perf_counter() - started_at),
        "rss_mb": _rss_high_water_mark_mb(),
        "gpu_memory_mb": query_gpu_memory_mb(),
    }


def _peak_snapshot_value(
    snapshots: list[dict[str, float | str | None]],
    key: str,
) -> float | None:
    values = [snapshot[key] for snapshot in snapshots if snapshot[key] is not None]
    if not values:
        return None
    return max(float(value) for value in values)


def _build_grouped_adjoint_metrics(
    adjoint: np.ndarray,
    adjoint_residual_rel: float,
    implicit_gradient_norm: float,
    implicit_gradient_finite: bool,
    snapshots: list[dict[str, float | str | None]],
) -> dict[str, Any]:
    adjoint_norm = float(np.linalg.norm(adjoint))
    return {
        "adjoint_residual_rel": float(adjoint_residual_rel),
        "adjoint_finite": bool(np.all(np.isfinite(adjoint))),
        "adjoint_norm": adjoint_norm,
        "implicit_gradient_finite": bool(implicit_gradient_finite),
        "implicit_gradient_norm": float(implicit_gradient_norm),
        "snapshots": snapshots,
    }


def evaluate_grouped_adjoint_memory_probe(metrics: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    snapshots = list(metrics.get("snapshots", []))
    required_labels = {
        "start",
        "after_fixture",
        "after_objective",
        "after_adjoint_solve",
        "after_grouped_adjoint_vjp",
    }
    labels = {str(snapshot.get("label")) for snapshot in snapshots}
    missing_labels = sorted(required_labels - labels)
    if missing_labels:
        failures.append(
            "Grouped-adjoint memory probe did not record all required snapshots: "
            + ", ".join(missing_labels)
        )
    if not bool(metrics.get("adjoint_finite", False)):
        failures.append("Grouped-adjoint memory probe produced a non-finite adjoint state.")
    if not bool(metrics.get("implicit_gradient_finite", False)):
        failures.append("Grouped-adjoint memory probe produced a non-finite implicit gradient.")
    if float(metrics.get("implicit_gradient_norm", 0.0)) <= 0.0:
        failures.append("Grouped-adjoint memory probe produced a zero implicit gradient.")
    if not np.isfinite(float(metrics.get("adjoint_residual_rel", np.inf))):
        failures.append("Grouped-adjoint memory probe adjoint residual is not finite.")
    return failures


def main() -> None:
    args = parse_args()
    bootstrap_local_simsopt()
    provenance = build_provenance(
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
            "nphi": int(args.nphi),
            "ntheta": int(args.ntheta),
            "mpol": int(args.mpol),
            "ntor": int(args.ntor),
            "compile_behavior": describe_compile_behavior(uses_subprocesses=False),
        },
    )
    print_provenance(provenance)

    started_at = time.perf_counter()
    snapshots = [record_memory_snapshot("start", started_at)]

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
    )
    snapshots.append(record_memory_snapshot("after_fixture", started_at))

    base_result = fixture["boozer_surface"].res
    if base_result is None or not base_result.get("success", False):
        raise RuntimeError("Baseline Boozer solve failed; cannot probe grouped-adjoint memory.")

    from simsopt.geo.surfaceobjectives_jax import BoozerResidualJAX

    jr_jax = BoozerResidualJAX(fixture["boozer_surface"], fixture["bs"])
    snapshots.append(record_memory_snapshot("after_objective", started_at))

    adjoint, adjoint_residual_rel = compute_adjoint_state(jr_jax)
    snapshots.append(record_memory_snapshot("after_adjoint_solve", started_at))

    grouped_cotangents = iter_grouped_adjoint_cotangents(jr_jax, adjoint)
    implicit_derivative = accumulate_grouped_adjoint_derivative(
        fixture["bs"], grouped_cotangents
    )

    implicit_gradient_norm, implicit_gradient_finite = compute_derivative_l2_metrics(
        implicit_derivative, fixture["bs"]
    )
    snapshots.append(record_memory_snapshot("after_grouped_adjoint_vjp", started_at))

    metrics = _build_grouped_adjoint_metrics(
        adjoint,
        adjoint_residual_rel,
        implicit_gradient_norm,
        implicit_gradient_finite,
        snapshots,
    )
    failures = evaluate_grouped_adjoint_memory_probe(metrics)

    payload = {
        "provenance": provenance,
        "baseline": {
            "solve_success": bool(base_result.get("success", False)),
            "equilibrium_path": str(fixture["equilibrium_path"]),
            "stage2_bs_path": str(fixture["stage2_bs_path"]),
        },
        "grouped_adjoint": {
            "adjoint_residual_rel": metrics["adjoint_residual_rel"],
            "adjoint_norm": metrics["adjoint_norm"],
            "adjoint_finite": metrics["adjoint_finite"],
            "implicit_gradient_norm": metrics["implicit_gradient_norm"],
            "implicit_gradient_finite": metrics["implicit_gradient_finite"],
        },
        "memory": {
            "snapshots": snapshots,
            "peak_rss_mb": _peak_snapshot_value(snapshots, "rss_mb"),
            "peak_gpu_memory_mb": _peak_snapshot_value(snapshots, "gpu_memory_mb"),
        },
        "failures": failures,
        "passed": not failures,
    }
    write_json(args.output_json, payload)
    if failures:
        print("GROUPED ADJOINT MEMORY PROBE FAILED")
        for failure in failures:
            print(f"  - {failure}")
        raise SystemExit(1)
    print("GROUPED ADJOINT MEMORY PROBE PASSED")


if __name__ == "__main__":
    main()
