"""Cache-only / lag-only / both attribution for nested-LS Shamanskii.

Each row is a fresh GPU process. Cache lanes prime then measure.
Not a nested speed claim.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path


def write_strict_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, allow_nan=False, indent=2) + "\n", encoding="utf-8"
    )


REPO = Path(__file__).resolve().parents[1]
EVIDENCE = REPO / "docs" / "receipts" / "evidence"
CHILD = REPO / "benchmarks" / "nested_ls_shamanskii_child.py"
PUBLICATION = (
    "Shamanskii and compile-cache attribution: cache-only, lag-only, and both. "
    "Not a nested speed claim and not F3 7.70x."
)
OUT_JSON = EVIDENCE / "nested_ls_reduced_gpu_shamanskii_attr_20260822.json"
OUT_LOG = EVIDENCE / "nested_ls_reduced_gpu_shamanskii_attr_20260822.log"
CACHE_DENSE = REPO / ".artifacts" / "nested-ls-shamanskii-xla-dense"
CACHE_SHAMANSKII = REPO / ".artifacts" / "nested-ls-shamanskii-xla-shamanskii"
REPEATS = 2
PYTHON = sys.executable


def _launch(
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
    probe = payload["probe"]
    steps = probe["steps"]
    row = {
        "linear_solver": linear_solver,
        "disable_cache": disable_cache,
        "cache_dir": None if cache_dir is None else str(cache_dir.relative_to(REPO)),
        "success": bool(probe["success"]),
        "walk_seconds": float(probe["walk_seconds"]),
        "process_wall_seconds": process_wall_seconds,
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
    }
    print(
        "attr"
        f" solver={linear_solver} cache_disabled={disable_cache}"
        f" walk={row['walk_seconds']!r} wall={process_wall_seconds!r}"
        f" reused={row['shamanskii_reused_steps']}"
        f" reassembled={row['shamanskii_reassembled_steps']}",
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
        prime_row = _launch(
            linear_solver=linear_solver,
            cache_dir=cache_dir,
            disable_cache=False,
        )
        prime_row["lane"] = name
        prime_row["role"] = "prime"
        prime_row["repeat"] = -1
        rows.append(prime_row)
    for repeat in range(REPEATS):
        row = _launch(
            linear_solver=linear_solver,
            cache_dir=cache_dir,
            disable_cache=disable_cache,
        )
        row["lane"] = name
        row["role"] = "measure"
        row["repeat"] = int(repeat)
        rows.append(row)
    return rows


def main() -> None:
    rows: list[dict[str, object]] = []
    print("lane cache_only", flush=True)
    rows.extend(
        _measured_lane(
            name="cache_only",
            linear_solver="dense_lu",
            cache_dir=CACHE_DENSE,
            disable_cache=False,
            prime=True,
        )
    )
    print("lane lag_only", flush=True)
    rows.extend(
        _measured_lane(
            name="lag_only",
            linear_solver="shamanskii",
            cache_dir=None,
            disable_cache=True,
            prime=False,
        )
    )
    print("lane both", flush=True)
    rows.extend(
        _measured_lane(
            name="both",
            linear_solver="shamanskii",
            cache_dir=CACHE_SHAMANSKII,
            disable_cache=False,
            prime=True,
        )
    )
    payload: dict[str, object] = {
        "claim_boundary": {
            "cap_2048_attempted": False,
            "comparable_operators": False,
            "explicit_inverse_m_production": False,
            "f3_sealed": True,
            "inherits_f3_7_70x": False,
            "nested_speed_claim": False,
            "shamanskii_attribution": True,
            "inner_and_process_wall": True,
        },
        "command": (
            "SIMSOPT_BACKEND_MODE=jax_gpu_fast JAX_PLATFORMS=cuda,cpu JAX_ENABLE_X64=1 "
            ".venv-qn-gpu/bin/python benchmarks/nested_ls_shamanskii_attribution.py"
        ),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "driver": "benchmarks.nested_ls_shamanskii_attribution",
        "execution_log": str(OUT_LOG.relative_to(REPO)),
        "publication": PUBLICATION,
        "rows": rows,
        "schema": "nested-ls-reduced-gpu-shamanskii-attr.v1",
        "written_by_pytest": False,
    }
    write_strict_json(OUT_JSON, payload)
    print("wrote", OUT_JSON, flush=True)
    print(
        "ok",
        all(bool(row["success"]) for row in rows if row["role"] == "measure"),
        flush=True,
    )


if __name__ == "__main__":
    main()
