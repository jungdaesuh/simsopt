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
    apply_requested_platform,
    build_provenance,
    load_json,
    preparse_platform,
    print_provenance,
    repo_pythonpath_env,
    run_python_script,
    write_json,
)


REQUESTED_PLATFORM = preparse_platform(sys.argv[1:])
apply_requested_platform(REQUESTED_PLATFORM)

import jax
import jaxlib

jax.config.update("jax_enable_x64", True)


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
    return parser.parse_args()


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


def safe_speedup(reference_s: float | None, candidate_s: float | None) -> float | None:
    if reference_s is None or candidate_s is None or candidate_s <= 0.0:
        return None
    return float(reference_s / candidate_s)


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


def _run_tier4_pair(args: argparse.Namespace) -> dict[str, Any]:
    base_args = [
        *_common_equilibrium_args(args),
        *_trusted_single_stage_args(args),
        "--samples",
        str(args.samples),
        "--eps",
        str(args.eps),
    ]
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
            name="tier4_adjoint_fd",
            payload=lane_payload,
            outer_elapsed_s=cpu_outer_elapsed_s + lane_outer_elapsed_s,
            cpu_elapsed_s=cpu_outer_elapsed_s,
            lane_elapsed_s=lane_elapsed_s,
            lane_label="jax-cpu" if args.platform == "cpu" else f"jax-{args.platform}",
        ),
    }


def main() -> None:
    args = parse_args()
    provenance = build_provenance(
        jax,
        jaxlib,
        title="Tier 5 trusted-fixture performance characterization",
        extra={
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
            "fd_samples": int(args.samples),
            "fd_eps": float(args.eps),
        },
    )
    print_provenance(provenance)

    lane_label = "jax-cpu" if args.platform == "cpu" else f"jax-{args.platform}"

    tier1b_payload, tier1b_outer = _timed_probe(
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
        [
            "--platform",
            args.platform,
            "--nphi",
            str(args.stage2_nphi),
            "--ntheta",
            str(args.stage2_ntheta),
            "--maxiter",
            str(args.maxiter),
            *_common_equilibrium_args(args),
        ],
        platform=args.platform,
    )
    tier3_payload, tier3_outer = _timed_probe(
        _single_stage_init_script(),
        [
            "--platform",
            args.platform,
            *_common_equilibrium_args(args),
            *_trusted_single_stage_args(args),
        ],
        platform=args.platform,
    )
    tier4_pair = _run_tier4_pair(args)

    tier1b_summary = summarize_pair_probe(
        name="tier1b_real_stage2",
        payload=tier1b_payload,
        outer_elapsed_s=tier1b_outer,
        cpu_elapsed_s=float(tier1b_payload["results"]["cpu"]["elapsed_s"]),
        lane_elapsed_s=float(tier1b_payload["results"]["jax"]["elapsed_s"]),
        lane_label=lane_label,
    )
    tier2_summary = summarize_pair_probe(
        name="tier2_stage2_e2e",
        payload=tier2_payload,
        outer_elapsed_s=tier2_outer,
        cpu_elapsed_s=float(tier2_payload["comparison"]["cpu_elapsed_s"]),
        lane_elapsed_s=float(tier2_payload["comparison"]["jax_elapsed_s"]),
        lane_label=lane_label,
    )
    tier3_summary = summarize_pair_probe(
        name="tier3_single_stage_init",
        payload=tier3_payload,
        outer_elapsed_s=tier3_outer,
        cpu_elapsed_s=float(tier3_payload["timings"]["cpu_elapsed_s"]),
        lane_elapsed_s=float(tier3_payload["timings"]["jax_elapsed_s"]),
        lane_label=lane_label,
    )

    summary = [tier1b_summary, tier2_summary, tier3_summary, tier4_pair["summary"]]
    total_outer_elapsed_s = float(sum(item["outer_elapsed_s"] for item in summary))

    payload = {
        "provenance": provenance,
        "rungs": {
            "tier1b_real_stage2": tier1b_payload,
            "tier2_stage2_e2e": tier2_payload,
            "tier3_single_stage_init": tier3_payload,
            "tier4_adjoint_fd_cpu": tier4_pair["cpu_payload"],
            "tier4_adjoint_fd_lane": tier4_pair["lane_payload"],
        },
        "summary": summary,
        "aggregate": {
            "lane_label": lane_label,
            "total_outer_elapsed_s": total_outer_elapsed_s,
            "passed": all(bool(item["passed"]) for item in summary),
        },
    }
    write_json(args.output_json, payload)

    print("\nTier 5 summary")
    print("--------------")
    for item in summary:
        speedup = item.get("speedup_vs_cpu")
        speedup_str = (
            f"{speedup:.2f}x"
            if isinstance(speedup, float)
            else "n/a"
        )
        print(
            f"{item['name']}: passed={item['passed']}  "
            f"outer={item['outer_elapsed_s']:.2f}s  "
            f"cpu={item.get('cpu_elapsed_s', float('nan')):.2f}s  "
            f"{item['lane_label']}={item.get('lane_elapsed_s', float('nan')):.2f}s  "
            f"speedup_vs_cpu={speedup_str}"
        )
    print(f"total outer elapsed: {total_outer_elapsed_s:.2f}s")


if __name__ == "__main__":
    main()
