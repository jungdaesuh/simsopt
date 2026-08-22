"""Cache-only / lag-only / both attribution for nested-LS Shamanskii.

One lane per process. Cache lanes prime then measure. ``--lane all``
spawns each lane as its own process, then merges. Not a nested speed
claim.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Final


def write_strict_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from simsopt_jax_adapters.geo.nested_ls_contract import NESTED_LS_GATE6_CLAIM_REPEATS

EVIDENCE = REPO / "docs" / "receipts" / "evidence"
CHILD = REPO / "benchmarks" / "nested_ls_shamanskii_child.py"
PUBLICATION = (
    "Shamanskii and compile-cache attribution: cache-only, lag-only, and both. "
    "Not a nested speed claim and not F3 7.70x."
)
OUT_JSON = EVIDENCE / "nested_ls_reduced_gpu_shamanskii_attr_20260822.json"
OUT_LOG = EVIDENCE / "nested_ls_reduced_gpu_shamanskii_attr_20260822.log"
FLOOR_JSON = EVIDENCE / "nested_ls_reduced_gpu_jax_floor_20260822.json"
CACHE_DENSE = REPO / ".artifacts" / "nested-ls-shamanskii-xla-dense"
CACHE_SHAMANSKII = REPO / ".artifacts" / "nested-ls-shamanskii-xla-shamanskii"
REPEATS = NESTED_LS_GATE6_CLAIM_REPEATS
PYTHON = sys.executable
LANES: Final[tuple[str, ...]] = ("cache_only", "lag_only", "both")
LANE_SPECS: Final[dict[str, dict[str, object]]] = {
    "cache_only": {
        "linear_solver": "dense_lu",
        "cache_dir": CACHE_DENSE,
        "disable_cache": False,
        "prime": True,
    },
    "lag_only": {
        "linear_solver": "shamanskii",
        "cache_dir": None,
        "disable_cache": True,
        "prime": False,
    },
    "both": {
        "linear_solver": "shamanskii",
        "cache_dir": CACHE_SHAMANSKII,
        "disable_cache": False,
        "prime": True,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Shamanskii/cache attribution. One lane per process; "
            "--lane all spawns cache_only, lag_only, both, then merges."
        )
    )
    parser.add_argument(
        "--lane",
        choices=("cache_only", "lag_only", "both", "floor", "all"),
        default="all",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge existing per-lane JSON into the canonical receipt.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Mint the canonical JSON even when git status is dirty.",
    )
    return parser.parse_args(argv)


def jax_claim_wall_seconds(row: dict[str, object]) -> float:
    """Parent wait minus native reconstruct and rejudge."""

    return (
        float(row["process_wall_seconds"])
        - float(row["reconstruct_seconds"])
        - float(row["native_rejudge_seconds"])
    )


def lane_json_path(lane: str) -> Path:
    return EVIDENCE / f"nested_ls_reduced_gpu_shamanskii_attr_20260822.{lane}.json"


def git_status_short() -> str:
    return subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=str(REPO), text=True
    )


def launch_nested_ls_gpu_child(
    *,
    linear_solver: str,
    cache_dir: Path | None,
    disable_cache: bool,
) -> dict[str, object]:
    env = dict(os.environ)
    env["SIMSOPT_BACKEND_MODE"] = "jax_gpu_fast"
    env["JAX_PLATFORMS"] = "cuda,cpu"
    env["JAX_ENABLE_X64"] = "1"
    if disable_cache:
        env["SIMSOPT_DISABLE_JAX_COMPILATION_CACHE"] = "1"
        env.pop("JAX_COMPILATION_CACHE_DIR", None)
        env.pop("SIMSOPT_JAX_COMPILATION_CACHE_DIR", None)
    else:
        env.pop("SIMSOPT_DISABLE_JAX_COMPILATION_CACHE", None)
        if cache_dir is None:
            raise ValueError("cache lane requires cache_dir")
        cache_dir.mkdir(parents=True, exist_ok=True)
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
        env["SIMSOPT_JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    with tempfile.NamedTemporaryFile(
        suffix=".json", prefix="nested_ls_shamanskii_", delete=False
    ) as handle:
        child_out = Path(handle.name)
    started = time.perf_counter()
    completed = subprocess.run(
        [PYTHON, str(CHILD), str(child_out), linear_solver],
        cwd=str(REPO),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    process_wall_seconds = float(time.perf_counter() - started)
    if completed.returncode != 0 or not child_out.is_file():
        raise RuntimeError(
            f"shamanskii child failed solver={linear_solver} "
            f"rc={completed.returncode} stderr={completed.stderr[-2000:]}"
        )
    payload = json.loads(child_out.read_text(encoding="utf-8"))
    child_out.unlink(missing_ok=True)
    cache_dir_rel: str | None
    if cache_dir is None:
        cache_dir_rel = None
    else:
        cache_dir_rel = str(cache_dir.relative_to(REPO))
    clocks = {
        "process_wall_seconds": process_wall_seconds,
        "process_elapsed_seconds": float(payload["process_elapsed_seconds"]),
        "reconstruct_seconds": float(payload["reconstruct_seconds"]),
        "walk_seconds": float(payload["walk_seconds"]),
        "native_rejudge_seconds": float(payload["native_rejudge_seconds"]),
        "jax_process_wall_seconds": float(payload["jax_process_wall_seconds"]),
        "jax_floor_seconds": float(payload["jax_floor_seconds"]),
    }
    if linear_solver == "floor":
        row: dict[str, object] = {
            "linear_solver": linear_solver,
            "disable_cache": disable_cache,
            "cache_dir": cache_dir_rel,
            "success": True,
            "iteration_count": 0,
            "grad_l2": None,
            "iota": None,
            "G": None,
            "coil_delta_inf": None,
            "native_rejudge_iter": None,
            "rejudge_vs_jax_surface_inf": None,
            "assembled_steps": [],
            "shamanskii_reused_steps": [],
            "shamanskii_reassembled_steps": [],
            "shamanskii_refine_passes": [],
            **clocks,
        }
        print(
            "attr"
            f" solver={linear_solver} cache_disabled={disable_cache}"
            f" success=True"
            f" floor={row['jax_floor_seconds']!r} wall={process_wall_seconds!r}",
            flush=True,
        )
        return row
    probe = payload["probe"]
    steps = probe["steps"]
    row = {
        "linear_solver": linear_solver,
        "disable_cache": disable_cache,
        "cache_dir": cache_dir_rel,
        "success": bool(probe["success"]),
        "iteration_count": int(probe["iteration_count"]),
        "grad_l2": float(probe["grad_l2"]),
        "iota": float(probe["iota"]),
        "G": float(probe["G"]),
        "coil_delta_inf": float(probe["coil_delta_inf"]),
        "native_rejudge_iter": int(probe["native_rejudge_iter"]),
        "rejudge_vs_jax_surface_inf": float(probe["rejudge_vs_jax_surface_inf"]),
        "assembled_steps": [
            int(step["iteration"]) for step in steps if step.get("assembled")
        ],
        "shamanskii_reused_steps": [
            int(step["iteration"]) for step in steps if step.get("shamanskii_reused")
        ],
        "shamanskii_reassembled_steps": [
            int(step["iteration"])
            for step in steps
            if step.get("shamanskii_reassembled")
        ],
        "shamanskii_refine_passes": [
            int(step.get("shamanskii_refine_passes", 0)) for step in steps
        ],
        **clocks,
    }
    print(
        "attr"
        f" solver={linear_solver} cache_disabled={disable_cache}"
        f" success={row['success']!r}"
        f" walk={row['walk_seconds']!r} wall={process_wall_seconds!r}"
        f" jax_wall={row['jax_process_wall_seconds']!r}"
        f" floor={row['jax_floor_seconds']!r}"
        f" reused={row['shamanskii_reused_steps']}"
        f" reassembled={row['shamanskii_reassembled_steps']}"
        f" refine={row['shamanskii_refine_passes']}",
        flush=True,
    )
    return row


def _measured_lane(
    *,
    name: str,
    linear_solver: str,
    cache_dir: Path | None,
    disable_cache: bool,
    prime: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if prime:
        prime_row = launch_nested_ls_gpu_child(
            linear_solver=linear_solver,
            cache_dir=cache_dir,
            disable_cache=False,
        )
        prime_row["lane"] = name
        prime_row["role"] = "prime"
        prime_row["repeat"] = -1
        rows.append(prime_row)
    for repeat in range(REPEATS):
        row = launch_nested_ls_gpu_child(
            linear_solver=linear_solver,
            cache_dir=cache_dir,
            disable_cache=disable_cache,
        )
        row["lane"] = name
        row["role"] = "measure"
        row["repeat"] = int(repeat)
        rows.append(row)
    return rows


def _floor_rows() -> list[dict[str, object]]:
    warm = launch_nested_ls_gpu_child(
        linear_solver="floor",
        cache_dir=CACHE_SHAMANSKII,
        disable_cache=False,
    )
    warm["lane"] = "floor_cache_on"
    warm["role"] = "floor"
    warm["repeat"] = 0
    cold = launch_nested_ls_gpu_child(
        linear_solver="floor",
        cache_dir=None,
        disable_cache=True,
    )
    cold["lane"] = "floor_cache_off"
    cold["role"] = "floor"
    cold["repeat"] = 0
    return [warm, cold]


def claim_boundary() -> dict[str, object]:
    return {
        "cap_2048_attempted": False,
        "comparable_operators": False,
        "explicit_inverse_m_production": False,
        "f3_sealed": True,
        "inherits_f3_7_70x": False,
        "nested_speed_claim": False,
        "shamanskii_attribution": True,
        "inner_and_process_wall": True,
        "one_lane_per_process": True,
        "repeats": int(REPEATS),
        "jax_claim_clock": "parent_wait_minus_reconstruct_rejudge",
        "jax_floor_on_walk_rows": "process_elapsed_minus_reconstruct_walk_rejudge",
        "jax_floor_on_floor_rows": "import_init_cache_load",
    }


def payload_for_rows(
    rows: list[dict[str, object]], *, driver: str
) -> dict[str, object]:
    return {
        "claim_boundary": claim_boundary(),
        "command": (
            "SIMSOPT_BACKEND_MODE=jax_gpu_fast JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 "
            ".venv-qn-gpu/bin/python benchmarks/nested_ls_shamanskii_attribution.py"
            + ((" " + " ".join(sys.argv[1:])) if sys.argv[1:] else " --lane all")
        ),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "driver": driver,
        "execution_log": str(OUT_LOG.relative_to(REPO)),
        "publication": PUBLICATION,
        "repeats": int(REPEATS),
        "rows": rows,
        "schema": "nested-ls-reduced-gpu-shamanskii-attr.v1",
        "written_by_pytest": False,
    }


def write_lane_json(lane: str, rows: list[dict[str, object]]) -> Path:
    path = lane_json_path(lane) if lane != "floor" else FLOOR_JSON
    write_strict_json(
        path,
        payload_for_rows(
            rows, driver=f"benchmarks.nested_ls_shamanskii_attribution.{lane}"
        ),
    )
    print("wrote", path, flush=True)
    return path


def merge_lane_files(*, allow_dirty: bool) -> dict[str, object]:
    if not allow_dirty:
        dirty = git_status_short().strip()
        if dirty:
            raise SystemExit(
                "refusing to mint claim-grade attribution JSON on a dirty tree:\n"
                f"{dirty}"
            )
    rows: list[dict[str, object]] = []
    for lane in LANES:
        path = lane_json_path(lane)
        if not path.is_file():
            raise FileNotFoundError(f"missing per-lane JSON: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.extend(payload["rows"])
    if not FLOOR_JSON.is_file():
        raise FileNotFoundError(f"missing floor JSON: {FLOOR_JSON}")
    rows.extend(json.loads(FLOOR_JSON.read_text(encoding="utf-8"))["rows"])
    measures = [row for row in rows if row["role"] == "measure"]
    if not all(bool(row["success"]) for row in measures):
        raise SystemExit("refusing to mint attribution JSON with a failed measure row")
    measure_counts = Counter(str(row["lane"]) for row in measures)
    for lane in LANES:
        if int(measure_counts[lane]) < int(REPEATS):
            raise SystemExit(
                f"lane {lane} has {measure_counts[lane]} measure rows, need {REPEATS}"
            )
    if not any(row["role"] == "floor" for row in rows):
        raise SystemExit("refusing to mint attribution JSON without floor rows")
    merged = payload_for_rows(
        rows, driver="benchmarks.nested_ls_shamanskii_attribution"
    )
    write_strict_json(OUT_JSON, merged)
    print("wrote", OUT_JSON, flush=True)
    print("ok", True, flush=True)
    return merged


def _spawn_lane(lane: str) -> None:
    completed = subprocess.run(
        [PYTHON, str(Path(__file__).resolve()), "--lane", lane],
        cwd=str(REPO),
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"lane {lane} failed rc={completed.returncode}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.merge:
        merge_lane_files(allow_dirty=bool(args.allow_dirty))
        return
    if args.lane == "all":
        for lane in LANES:
            print("lane", lane, flush=True)
            _spawn_lane(lane)
        print("lane floor", flush=True)
        _spawn_lane("floor")
        merge_lane_files(allow_dirty=bool(args.allow_dirty))
        return
    if args.lane == "floor":
        print("lane floor", flush=True)
        write_lane_json("floor", _floor_rows())
        return
    spec = LANE_SPECS[args.lane]
    cache_dir = spec["cache_dir"]
    print("lane", args.lane, flush=True)
    rows = _measured_lane(
        name=args.lane,
        linear_solver=str(spec["linear_solver"]),
        cache_dir=cache_dir if isinstance(cache_dir, Path) else None,
        disable_cache=bool(spec["disable_cache"]),
        prime=bool(spec["prime"]),
    )
    write_lane_json(args.lane, rows)


if __name__ == "__main__":
    main()
