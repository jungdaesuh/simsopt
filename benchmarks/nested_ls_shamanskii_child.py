"""Fresh-process nested-LS walk child for Shamanskii/cache attribution.

Parent sets GPU env and cache policy before launch. Writes OUT.json.
Not a nested speed claim.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("SIMSOPT_BACKEND_MODE", "jax_gpu_fast")
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from benchmarks.validation_ladder_common import apply_compilation_cache_policy

apply_compilation_cache_policy(os.environ.get("JAX_COMPILATION_CACHE_DIR"))

import jax
from simsopt_jax_adapters.geo.nested_ls_contract import NESTED_LS_NEWTON_MAXITER
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    DEFAULT_F3_B37_GPU_LANE,
    evaluate_f3_b37_schur_newton_walk,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: nested_ls_shamanskii_child.py OUT.json LINEAR_SOLVER")
    out_path = Path(sys.argv[1])
    linear_solver = str(sys.argv[2])
    if jax.default_backend() != "gpu":
        raise SystemExit(f"expected gpu, got {jax.default_backend()!r}")
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
    payload = {
        "linear_solver": linear_solver,
        "jax_default_backend": jax.default_backend(),
        "jax_compilation_cache_dir": os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        "cache_disabled": os.environ.get("SIMSOPT_DISABLE_JAX_COMPILATION_CACHE"),
        "probe": walk.as_payload(),
    }
    out_path.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
