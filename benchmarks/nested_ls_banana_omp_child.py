"""Fresh-process native banana ``run_code`` with OMP pinned in the environment.

Parent sets ``OMP_NUM_THREADS`` before launching this file. Prints one JSON
object on stdout. Not a speed claim.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
from simsopt_jax_adapters.geo.nested_ls_reduced import (
    nested_ls_reduced_closures,
    require_full_y_rank,
    solve_projected_y,
)
from simsopt_jax_adapters.geo.nested_ls_reduced_scale import (
    DEFAULT_F3_B37_GPU_LANE,
    _run_native_banana_bfgs_then_newton,
    load_archived_nested_ls_pair,
    load_flat675_lane_blocks,
    nested_ls_omp_threads_pinned,
    nested_ls_threading_env,
)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: nested_ls_banana_omp_child.py OUT.json")
    out_path = Path(sys.argv[1])
    threading = nested_ls_threading_env()
    coils, surface, _meta = load_flat675_lane_blocks(DEFAULT_F3_B37_GPU_LANE)
    native, jax_boozer, _target = load_archived_nested_ls_pair(
        coil_coordinates=coils,
        surface_coordinates=surface,
    )
    del _target
    residual_fn, _objective_fn, _phi_hat = nested_ls_reduced_closures(jax_boozer)
    del _phi_hat
    surface0 = np.asarray(jax_boozer.surface.get_dofs(), dtype=np.float64)
    solution = solve_projected_y(residual_fn, surface0, np.zeros(2, dtype=np.float64))
    require_full_y_rank(solution)
    y_star = np.asarray(solution.solution, dtype=np.float64)
    native.surface.set_dofs(np.asarray(native.surface.get_dofs(), dtype=np.float64))
    result = _run_native_banana_bfgs_then_newton(
        native,
        iota=float(y_star[0]),
        G=float(y_star[1]),
    )
    payload = {
        **result,
        "threading": threading,
        "omp_pinned": nested_ls_omp_threads_pinned(threading),
        "omp_num_threads": threading["OMP_NUM_THREADS"],
    }
    out_path.write_text(json.dumps(payload, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
