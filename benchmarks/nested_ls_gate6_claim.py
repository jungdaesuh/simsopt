"""Gate-6 process-wall vs process-wall claim run.

Interleaves complete native banana (best-of-contract OMP) and JAX
Shamanskii+cache processes. Both sides use parent subprocess wait.
JAX claim wall subtracts native reconstruct and rejudge from that
parent wait. Not F3 7.70x. nested_speed_claim is set only when
physics holds and min JAX claim wall beats min native parent wait.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from simsopt_jax_adapters.geo.nested_ls_contract import (
    NESTED_LS_GATE6_AGGREGATION,
    NESTED_LS_GATE6_IOTA_G_TOL,
    NESTED_LS_GATE6_NATIVE_OMP_THREADS,
    NESTED_LS_NEWTON_TOL,
)

from benchmarks.nested_ls_shamanskii_attribution import (
    CACHE_SHAMANSKII,
    PYTHON,
    REPEATS,
    git_implementation_dirty,
    jax_claim_wall_seconds,
    launch_nested_ls_gpu_child,
    write_strict_json,
)

EVIDENCE = REPO / "docs" / "receipts" / "evidence"
BANANA_CHILD = REPO / "benchmarks" / "nested_ls_banana_omp_child.py"
IOTA_G_TOL = NESTED_LS_GATE6_IOTA_G_TOL
GRAD_TOL = NESTED_LS_NEWTON_TOL


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate-6 process-wall claim run.")
    parser.add_argument(
        "--omp",
        type=int,
        default=NESTED_LS_GATE6_NATIVE_OMP_THREADS,
        help="Native banana OMP_NUM_THREADS (best-of-contract on this host).",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Receipt suffix, e.g. a100 → nested_ls_reduced_gpu_gate6_20260822.a100.json",
    )
    return parser.parse_args(argv)


def _require_clean_tree() -> str:
    dirty = git_implementation_dirty().strip()
    if dirty:
        raise SystemExit(
            f"Gate-6 requires a clean tree (implementation, not evidence):\n{dirty}"
        )
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(REPO), text=True
    ).strip()


def _launch_native_banana(*, omp_num_threads: int) -> dict[str, object]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(omp_num_threads)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env.pop("SIMSOPT_BACKEND_MODE", None)
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix="nested_ls_gate6_native_", delete=False
    ) as handle:
        child_out = Path(handle.name)
    started = time.perf_counter()
    completed = subprocess.run(
        [PYTHON, str(BANANA_CHILD), str(child_out)],
        cwd=str(REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    process_wall_seconds = float(time.perf_counter() - started)
    if completed.returncode != 0 or not child_out.is_file():
        raise RuntimeError(
            f"gate6 native banana failed rc={completed.returncode} "
            f"stderr={completed.stderr[-2000:]}"
        )
    payload = json.loads(child_out.read_text(encoding="utf-8"))
    child_out.unlink(missing_ok=True)
    threading = payload["threading"]
    observed_omp = int(payload["omp_num_threads"])
    row = {
        "side": "native",
        "omp_num_threads": int(omp_num_threads),
        "observed_omp_num_threads": observed_omp,
        "omp_pinned": bool(payload["omp_pinned"]),
        "omp_proc_bind": threading["OMP_PROC_BIND"],
        "omp_places": threading["OMP_PLACES"],
        "success": bool(payload["success"]),
        "inner_solver_seconds": float(payload["seconds"]),
        "process_wall_seconds": process_wall_seconds,
        "claim_wall_seconds": process_wall_seconds,
        "iota": float(payload["iota"]),
        "G": float(payload["G"]),
        "coil_delta_inf": float(payload["coil_delta_inf"]),
        "bfgs_iter": int(payload["bfgs_iter"]),
        "newton_iter": int(payload["newton_iter"]),
    }
    print(
        "gate6 native"
        f" success={row['success']!r} inner={row['inner_solver_seconds']!r}"
        f" wall={process_wall_seconds!r} omp={omp_num_threads}",
        flush=True,
    )
    return row


def _physics_ok(native: dict[str, object], jax_row: dict[str, object]) -> str | None:
    if not bool(native["success"]):
        return "native_failed"
    if not bool(native["omp_pinned"]):
        return "native_omp_unpinned"
    if int(native["observed_omp_num_threads"]) != int(native["omp_num_threads"]):
        return "native_omp_not_contract"
    if float(native["coil_delta_inf"]) != 0.0:
        return "native_coil_moved"
    if not bool(jax_row["success"]):
        return "jax_failed"
    if float(jax_row["coil_delta_inf"]) != 0.0:
        return "jax_coil_moved"
    if int(jax_row["native_rejudge_iter"]) != 0:
        return "jax_rejudge_not_noop"
    if float(jax_row["grad_l2"]) > GRAD_TOL:
        return "jax_grad_tol"
    if abs(float(jax_row["iota"]) - float(native["iota"])) > IOTA_G_TOL:
        return "iota_mismatch"
    if abs(float(jax_row["G"]) - float(native["G"])) > IOTA_G_TOL:
        return "g_mismatch"
    return None


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    omp_num_threads = int(args.omp)
    tag = str(args.tag).strip()
    suffix = f".{tag}" if tag else ""
    out_json = EVIDENCE / f"nested_ls_reduced_gpu_gate6_20260822{suffix}.json"
    out_log = EVIDENCE / f"nested_ls_reduced_gpu_gate6_20260822{suffix}.log"
    publication = (
        "Gate-6 process-wall vs process-wall claim run. Native banana at "
        f"best-of-contract OMP={omp_num_threads}, JAX Shamanskii with "
        "persistent compile cache. Not F3 7.70x."
    )
    sha = _require_clean_tree()
    print("gate6 prime shamanskii cache", "omp", omp_num_threads, flush=True)
    prime = launch_nested_ls_gpu_child(
        linear_solver="shamanskii",
        cache_dir=CACHE_SHAMANSKII,
        disable_cache=False,
    )
    prime["side"] = "jax"
    prime["lane"] = "both"
    prime["role"] = "prime"
    prime["repeat"] = -1
    pairs: list[dict[str, object]] = []
    native_walls: list[float] = []
    jax_walls: list[float] = []
    fail_reason: str | None = None
    for repeat in range(REPEATS):
        native = _launch_native_banana(omp_num_threads=omp_num_threads)
        native["role"] = "measure"
        native["repeat"] = int(repeat)
        jax_row = launch_nested_ls_gpu_child(
            linear_solver="shamanskii",
            cache_dir=CACHE_SHAMANSKII,
            disable_cache=False,
        )
        jax_row["side"] = "jax"
        jax_row["lane"] = "both"
        jax_row["role"] = "measure"
        jax_row["repeat"] = int(repeat)
        jax_claim = jax_claim_wall_seconds(jax_row)
        jax_row["claim_wall_seconds"] = jax_claim
        native_claim = float(native["claim_wall_seconds"])
        native_walls.append(native_claim)
        jax_walls.append(jax_claim)
        reason = _physics_ok(native, jax_row)
        pair = {
            "repeat": int(repeat),
            "native": native,
            "jax": jax_row,
            "physics_ok": reason is None,
            "fail_closed_reason": reason,
        }
        pairs.append(pair)
        print(
            "gate6 pair"
            f" repeat={repeat} physics={reason is None}"
            f" native_wall={native_claim!r}"
            f" jax_wall={jax_claim!r}"
            f" reason={reason!r}",
            flush=True,
        )
        if fail_reason is None and reason is not None:
            fail_reason = reason
    native_min = min(native_walls)
    jax_min = min(jax_walls)
    physics_ok = fail_reason is None
    nested_speed_claim = bool(physics_ok and jax_min < native_min)
    payload: dict[str, object] = {
        "claim_boundary": {
            "aggregation": NESTED_LS_GATE6_AGGREGATION,
            "jax_claim_clock": "parent_wait_minus_reconstruct_rejudge",
            "native_claim_clock": "parent_wait",
            "cap_2048_attempted": False,
            "comparable_operators": False,
            "explicit_inverse_m_production": False,
            "f3_sealed": True,
            "inherits_f3_7_70x": False,
            "inner_and_process_wall": True,
            "interleaved_repeats": True,
            "jax_linear_solver": "shamanskii",
            "jax_persistent_cache": True,
            "native_omp_num_threads": omp_num_threads,
            "nested_speed_claim": nested_speed_claim,
            "one_lane_per_process": True,
            "repeats": int(REPEATS),
            "tag": tag or None,
        },
        "command": (
            "SIMSOPT_BACKEND_MODE=jax_gpu_fast JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 "
            "python benchmarks/nested_ls_gate6_claim.py"
            + (f" --omp {omp_num_threads}" if omp_num_threads else "")
            + (f" --tag {tag}" if tag else "")
        ),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "driver": "benchmarks.nested_ls_gate6_claim",
        "execution_log": str(out_log.relative_to(REPO)),
        "fail_closed_reason": fail_reason,
        "git_head": sha,
        "native_min_process_wall_seconds": native_min,
        "native_median_process_wall_seconds": float(statistics.median(native_walls)),
        "jax_min_process_wall_seconds": jax_min,
        "jax_median_process_wall_seconds": float(statistics.median(jax_walls)),
        "pairs": pairs,
        "prime": prime,
        "publication": publication,
        "schema": "nested-ls-reduced-gpu-gate6.v1",
        "written_by_pytest": False,
    }
    write_strict_json(out_json, payload)
    print("wrote", out_json, flush=True)
    print(
        "ok",
        physics_ok,
        "nested_speed_claim",
        nested_speed_claim,
        "jax_min",
        jax_min,
        "native_min",
        native_min,
        flush=True,
    )
    if not physics_ok:
        raise SystemExit(f"Gate-6 physics failed: {fail_reason}")


if __name__ == "__main__":
    main()
