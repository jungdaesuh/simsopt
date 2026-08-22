"""Fresh-process nested-LS walk child for Shamanskii/cache attribution.

Parent sets GPU env and cache policy before launch. Writes OUT.json.
Process clock ``_T0`` starts before JAX import. ``floor`` skips the
walk and records import+init+cache load. Not a nested speed claim.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_T0 = time.perf_counter()

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.validation_ladder_common import apply_compilation_cache_policy

apply_compilation_cache_policy(os.environ.get("JAX_COMPILATION_CACHE_DIR"))

import jax


def _clocks(
    *,
    process_elapsed: float,
    reconstruct_seconds: float,
    walk_seconds: float,
    rejudge_seconds: float,
) -> dict[str, float]:
    jax_process_wall = (
        float(process_elapsed) - float(reconstruct_seconds) - float(rejudge_seconds)
    )
    jax_floor = jax_process_wall - float(walk_seconds)
    return {
        "process_elapsed_seconds": float(process_elapsed),
        "reconstruct_seconds": float(reconstruct_seconds),
        "walk_seconds": float(walk_seconds),
        "native_rejudge_seconds": float(rejudge_seconds),
        "jax_process_wall_seconds": float(jax_process_wall),
        "jax_floor_seconds": float(jax_floor),
    }


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: nested_ls_shamanskii_child.py OUT.json LINEAR_SOLVER")
    out_path = Path(sys.argv[1])
    linear_solver = str(sys.argv[2])
    if jax.default_backend() != "gpu":
        raise SystemExit(f"expected gpu, got {jax.default_backend()!r}")
    cache_meta = {
        "linear_solver": linear_solver,
        "jax_default_backend": jax.default_backend(),
        "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        "cache_disabled": os.environ.get("SIMSOPT_DISABLE_JAX_COMPILATION_CACHE"),
    }
    if linear_solver == "floor":
        elapsed = float(time.perf_counter() - _T0)
        payload = {
            **cache_meta,
            **_clocks(
                process_elapsed=elapsed,
                reconstruct_seconds=0.0,
                walk_seconds=0.0,
                rejudge_seconds=0.0,
            ),
            "probe": None,
        }
        out_path.write_text(
            json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8"
        )
        return 0
    from simsopt_jax_adapters.geo.nested_ls_contract import NESTED_LS_NEWTON_MAXITER
    from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
        DEFAULT_F3_B37_GPU_LANE,
        evaluate_f3_b37_schur_newton_walk,
        load_archived_nested_ls_pair,
        load_flat675_lane_blocks,
    )

    coils, surface, _meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    del _target
    walk = evaluate_f3_b37_schur_newton_walk(
        native,
        jax_boozer,
        maxiter=int(NESTED_LS_NEWTON_MAXITER),
        linear_solver=linear_solver,
    )
    elapsed = float(time.perf_counter() - _T0)
    payload = {
        **cache_meta,
        **_clocks(
            process_elapsed=elapsed,
            reconstruct_seconds=float(walk.reconstruct_seconds),
            walk_seconds=float(walk.walk_seconds),
            rejudge_seconds=float(walk.native_rejudge_seconds),
        ),
        "probe": walk.as_payload(),
    }
    out_path.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
